const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

pub const App = @This();

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const workgroup_size = 64;
const buffer_size = 1000;

// const Config = struct {
//     dim: u32, // transformer dimension
//     hidden_dim: u32, // for ffn layers
//     n_layers: u32, // number of layers
//     n_heads: u32, // number of query heads
//     n_kv_heads: u32, // number of key/value heads (can be < query heads because of multiquery)
//     vocab_size: u32, // vocabulary size, usually 256 (byte-level)
//     seq_len: u32, // max sequence length
// };

const ConfigReader = extern struct {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
};

const Tokenizer = struct {
    pub const Token = struct {
        logit: f32,
        idx: u32,
    };

    tokens: std.ArrayList([]const u8),
    back_map: std.StringHashMap(Token),
};

pub fn tokenize(allocator_: std.mem.Allocator, str: []const u8, tokenizer: *Tokenizer) ![]u32 {
    var arena = std.heap.ArenaAllocator.init(allocator_);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokens = std.ArrayList(u32).init(allocator);

    // encode bytes
    for (str) |byte| {
        const token_str = try std.fmt.allocPrint(allocator, "{c}", .{byte});
        defer allocator.free(token_str);

        if (tokenizer.back_map.get(token_str)) |match| {
            try tokens.append(match.idx);
        } else {
            return error.TokenizerFail;
        }
    }

    // compress
    while (true) {
        var i: usize = 0;
        var best_idx: usize = 0;
        var best_token: ?Tokenizer.Token = null;

        while (i + 1 < tokens.items.len) : (i += 1) {
            const token_a = tokens.items[i];
            const token_b = tokens.items[i + 1];
            const token_str = try std.fmt.allocPrint(allocator, "{s}{s}", .{
                tokenizer.tokens.items[token_a],
                tokenizer.tokens.items[token_b],
            });

            defer allocator.free(token_str); //even though we have arena, we still free. This clears mem but arena will leave space allocated for next token
            if (tokenizer.back_map.get(token_str)) |match| {
                if (best_token) |cur_best| {
                    if (match.logit > cur_best.logit) {
                        best_idx = i;
                        best_token = match;
                    }
                } else {
                    best_idx = i;
                    best_token = match;
                }
            }
        }

        if (best_token) |token| {
            //replace the consecutive tokens with the new one
            _ = tokens.orderedRemove(best_idx);
            _ = tokens.orderedRemove(best_idx);
            try tokens.insert(best_idx, token.idx);
        } else {
            break; //no match, we are done
        }
    }

    return try allocator_.dupe(u32, tokens.items);
}

pub fn read_tokenizer(base_allocator: std.mem.Allocator) !*Tokenizer {
    var arena = std.heap.ArenaAllocator.init(base_allocator);
    const arena_allocator = arena.allocator();

    const checkpoint_path = "/mnt/data/LLaMA/model44m.bin";
    var file = try std.fs.cwd().openFile(checkpoint_path, .{});
    defer file.close();

    var model_file_buffered = std.io.bufferedReader(file.reader());
    var model_reader = model_file_buffered.reader();

    // Read weights file
    var config = try model_reader.readStruct(ConfigReader);

    std.log.info("{}", .{config});

    const n_layers = @as(usize, @intCast(config.n_layers));
    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));
    const vocab_size = @as(usize, @intCast(config.vocab_size));
    const seq_len = @as(usize, @intCast(config.seq_len));

    //Global
    // - token_embedding = vocab_size * dim

    //Layer:
    // - rms attention weight = layer * dim
    // - query weight = layer * dim * dim
    // - key weight = layer * dim * dim
    // - value weight = layer * dim * dim
    // - output weight = layer * dim * dim
    // - rms ffn weight = layer * dim * dim
    // - w1 weight = layer * dim * hidden_dim
    // - w2 weights = layer * hidden_dim * dim
    // - w3 weights = layer * dim * hidden_dim

    // final
    // - rms final weight = dim
    // - freq real = seq_len * head_size / 2
    // - freq img = seq_len * head_size / 2
    //  - final class weights (if not shared with embedding) vocab * dim

    var weight_buffer = std.ArrayList(f32).init(arena_allocator);
    try weight_buffer.resize(@as(usize, @intCast(vocab_size * dim)));
    _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

    var token_embedding = try Tensor.init_from_data(
        base_allocator,
        &[_]usize{ vocab_size, dim },
        .Storage,
        weight_buffer.items,
    );

    const LayerWeights = struct {
        rms_attention: *Tensor,
        query_weight: *Tensor,
        key_weight: *Tensor,
        value_weight: *Tensor,
        output_weight: *Tensor,
        rms_ffn: *Tensor,
        w1: *Tensor,
        w2: *Tensor,
        w3: *Tensor,
    };

    var layer_weights = std.ArrayList(LayerWeights).init(base_allocator);
    try layer_weights.resize(n_layers);

    // rms_attention
    try weight_buffer.resize(dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.rms_attention = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{dim},
            .Storage,
            weight_buffer.items,
        );
    }

    // query_weight
    try weight_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.query_weight = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // key_weight
    try weight_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.key_weight = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // value_weight
    try weight_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.value_weight = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    //output_weight
    try weight_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.output_weight = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // rms_ffn
    try weight_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.rms_ffn = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // w1
    try weight_buffer.resize(dim * hidden_dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.w1 = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, hidden_dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // w2
    try weight_buffer.resize(hidden_dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.w2 = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ hidden_dim, dim },
            .Storage,
            weight_buffer.items,
        );
    }

    // w3
    try weight_buffer.resize(dim * hidden_dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.asBytes(&weight_buffer.items));

        layer.w3 = try Tensor.init_from_data(
            base_allocator,
            &[_]usize{ dim, hidden_dim },
            .Storage,
            weight_buffer.items,
        );
    }

    var final_rms_weight = try Tensor.init_from_data(
        base_allocator,
        &[_]usize{dim},
        .Storage,
        weight_buffer.items,
    );

    _ = token_embedding;
    _ = seq_len;
    _ = final_rms_weight;

    // Read tokenizer

    var token_file = try std.fs.cwd().openFile("/mnt/data/LLaMA/tokenizer.bin", .{});

    // var token_file = try std.fs.cwd().openFile("/mnt/xfs/LLaMA/tokenizer.bin", .{});
    defer token_file.close();

    var token_reader = token_file.reader();

    var max_token_length = try token_reader.readInt(u32, std.builtin.Endian.Little);
    std.log.info("Max token len: {}", .{max_token_length});

    // if ((try token_file.read(std.mem.asBytes(&max_token_length))) != 4)
    //     return error.InvalidTokenizerFile;

    var vocab = std.ArrayList([]const u8).init(base_allocator);

    // var scores = std.ArrayList(f32).init(allocator);

    var tokenizer = try base_allocator.create(Tokenizer);
    tokenizer.* = .{
        .tokens = std.ArrayList([]const u8).init(base_allocator),
        .back_map = std.StringHashMap(Tokenizer.Token).init(base_allocator),
    };

    for (0..@as(usize, @intCast(config.vocab_size))) |idx| {
        var score: f32 = 0;
        _ = try token_reader.readAll(std.mem.asBytes(&score));
        // const score: f32 = @bitCast(try token_reader.readInt(u32, std.builtin.Endian.Little)); //strange
        const token_len = try token_reader.readInt(u32, std.builtin.Endian.Little);

        var tokens = try base_allocator.alloc(u8, token_len);
        const read_amt = try token_reader.readAll(tokens);
        if (read_amt != token_len) {
            return error.UnexpectedEof;
        }

        try vocab.append(tokens);
        // try scores.append(score);
        try tokenizer.tokens.append(try base_allocator.dupe(u8, tokens));

        try tokenizer.back_map.put(try base_allocator.dupe(u8, tokens), .{
            .logit = score,
            .idx = @as(u32, @intCast(idx)),
        });
    }

    return tokenizer;
}

const Tensor = struct {
    const Type = enum {
        Storage,
        Target,
    };

    shape: []const usize,
    N: usize,
    buffer: *gpu.Buffer,

    pub fn init(allocator: std.mem.Allocator, shape: []usize, tensor_type: Type) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }
        tensor.* = .{
            .N = N,
            .shape = shape,
            .buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = N * @sizeOf(f32),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(f32, 0, N);
        @memset(buffer_mapped.?, 0.0);
        tensor.buffer.unmap();

        return tensor;
    }

    pub fn init_from_data(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []f32) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }
        tensor.* = .{
            .N = N,
            .shape = shape,
            .buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = N * @sizeOf(f32),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(f32, 0, N);
        std.mem.copy(f32, buffer_mapped.?, data);
        tensor.buffer.unmap();

        return tensor;
    }

    fn read_data(self: *Tensor) void {
        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();

        const output_buffer = self.create_matching_output_buffer();
        defer output_buffer.release();

        command_encoder.copyBufferToBuffer(self.buffer, 0, output_buffer, 0, self.N * @sizeOf(f32));

        // Setup response callback
        var response: gpu.Buffer.MapAsyncStatus = undefined;
        const callback = (struct {
            pub inline fn callback(ctx: *gpu.Buffer.MapAsyncStatus, status: gpu.Buffer.MapAsyncStatus) void {
                ctx.* = status;
            }
        }).callback;

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});

        // Copy result
        output_buffer.mapAsync(.{ .read = true }, 0, self.N * @sizeOf(f32), &response, callback);
        while (true) {
            if (response == gpu.Buffer.MapAsyncStatus.success) {
                break;
            } else {
                core.device.tick();
            }
        }

        const output_mapped = output_buffer.getConstMappedRange(f32, 0, self.N);
        defer output_buffer.unmap();
        for (output_mapped.?) |v| {
            std.debug.print("{d} ", .{v});
        }
        std.debug.print("\n", .{});
    }

    fn create_matching_output_buffer(self: *Tensor) *gpu.Buffer {
        const output_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
            .label = "output_buffer",
            .usage = .{ .copy_dst = true, .map_read = true },
            .size = self.N * @sizeOf(f32),
        });
        return output_buffer;
    }
};

const AttentionOperator = struct {
    shader_module_slate: *gpu.ShaderModule,
    shader_module_softmax_value: *gpu.ShaderModule,
    pipeline_slate: *gpu.ComputePipeline,
    pipeline_softmax_value: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
        n_heads: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*AttentionOperator {
        var operator = try allocator.create(AttentionOperator);
        const shader_module_slate = core.device.createShaderModuleWGSL(
            "attention_slate.wgsl",
            @embedFile("attention_slate.wgsl"),
        );
        const pipeline_slate = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{
            .compute = gpu.ProgrammableStageDescriptor{
                .module = shader_module_slate,
                .entry_point = "main",
            },
        });

        const shader_module_softmax_value = core.device.createShaderModuleWGSL(
            "attention_softmax_value.wgsl",
            @embedFile("attention_softmax_value.wgsl"),
        );
        const pipeline_softmax_value = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{
            .compute = gpu.ProgrammableStageDescriptor{
                .module = shader_module_softmax_value,
                .entry_point = "main",
            },
        });

        operator.* = .{
            .shader_module_slate = shader_module_slate,
            .pipeline_slate = pipeline_slate,

            .shader_module_softmax_value = shader_module_softmax_value,
            .pipeline_softmax_value = pipeline_softmax_value,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *AttentionOperator,
        Q: *Tensor,
        K: *Tensor,
        V: *Tensor,
        slate: *Tensor,
        output: *Tensor,
        n_heads: u32,
    ) void {
        std.debug.assert(Q.shape.len == 2);
        std.debug.assert(K.shape.len == 2);
        std.debug.assert(V.shape.len == 2);

        const params: Params = .{
            .L = @as(u32, @intCast(Q.shape[0])),
            .dim = @as(u32, @intCast(Q.shape[1])),
            .n_heads = @as(u32, @intCast(n_heads)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const slate_bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline_slate.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, K.buffer, 0, K.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, Q.buffer, 0, Q.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, slate.buffer, 0, slate.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(3, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer slate_bindings.release();

        const softmax_bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline_softmax_value.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, output.buffer, 0, output.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, V.buffer, 0, V.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, slate.buffer, 0, slate.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(3, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer softmax_bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();

        std.log.info("here", .{});
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_slate);
            pass_encoder.setBindGroup(0, slate_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        std.log.info("here", .{});
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_softmax_value);
            pass_encoder.setBindGroup(0, softmax_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
        // _ = output;
        std.log.info("here", .{});
        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

const RMSNormOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*RMSNormOperator {
        var operator = try allocator.create(RMSNormOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "rmsnorm_inplace.wgsl",
            @embedFile("rmsnorm_inplace.wgsl"),
        );
        const pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
            .module = shader_module,
            .entry_point = "main",
        } });

        operator.* = .{
            .shader_module = shader_module,
            .pipeline = pipeline,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *RMSNormOperator,
        x: *Tensor,
    ) void {
        std.debug.assert(x.shape.len == 2);

        const params: Params = .{
            .L = @as(u32, @intCast(x.shape[0])),
            .dim = @as(u32, @intCast(x.shape[1])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, x.buffer, 0, x.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

const AddOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*AddOperator {
        var operator = try allocator.create(AddOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "add_inplace.wgsl",
            @embedFile("add_inplace.wgsl"),
        );
        const pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
            .module = shader_module,
            .entry_point = "main",
        } });

        operator.* = .{
            .shader_module = shader_module,
            .pipeline = pipeline,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *AddOperator,
        left: *Tensor,
        right: *Tensor,
    ) void {
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);
        std.debug.assert(std.mem.eql(usize, left.shape, right.shape));

        const params: Params = .{
            .L = @as(u32, @intCast(left.shape[0])),
            .dim = @as(u32, @intCast(left.shape[1])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, left.buffer, 0, left.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, right.buffer, 0, right.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

const SILUOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*SILUOperator {
        var operator = try allocator.create(SILUOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "silu_inplace.wgsl",
            @embedFile("silu_inplace.wgsl"),
        );
        const pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
            .module = shader_module,
            .entry_point = "main",
        } });

        operator.* = .{
            .shader_module = shader_module,
            .pipeline = pipeline,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *SILUOperator,
        x: *Tensor,
    ) void {
        std.debug.assert(x.shape.len == 2);

        const params: Params = .{
            .L = @as(u32, @intCast(x.shape[0])),
            .dim = @as(u32, @intCast(x.shape[1])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, x.buffer, 0, x.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

const RopeOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
        n_heads: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*RopeOperator {
        var operator = try allocator.create(RopeOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "rope.wgsl",
            @embedFile("rope.wgsl"),
        );
        const pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
            .module = shader_module,
            .entry_point = "main",
        } });

        operator.* = .{
            .shader_module = shader_module,
            .pipeline = pipeline,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *RopeOperator,
        k: *Tensor,
        v: *Tensor,
        n_heads: u32,
    ) void {
        std.debug.assert(k.shape.len == 2);
        std.debug.assert(v.shape.len == 2);
        std.debug.assert(std.mem.eql(usize, k.shape, v.shape));

        const params: Params = .{
            .L = @as(u32, @intCast(k.shape[0])),
            .dim = @as(u32, @intCast(k.shape[1])),
            .n_heads = @as(u32, @intCast(n_heads)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, k.buffer, 0, k.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, v.buffer, 0, v.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

const MatOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        M: u32,
        K: u32,
        N: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*MatOperator {
        var operator = try allocator.create(MatOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "matmul.wgsl",
            @embedFile("matmul.wgsl"),
        );
        const pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
            .module = shader_module,
            .entry_point = "main",
        } });

        operator.* = .{
            .shader_module = shader_module,
            .pipeline = pipeline,

            .param_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "param_buffer",
                .usage = .{ .uniform = true, .copy_dst = true },
                .size = @sizeOf(Params),
            }),
        };
        return operator;
    }

    pub fn execute(
        self: *MatOperator,
        left: *Tensor,
        right: *Tensor,
        output: *Tensor,
    ) void {
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);
        std.debug.assert(output.shape.len == 2);

        const params: Params = .{
            .N = @as(u32, @intCast(left.shape[0])),
            .M = @as(u32, @intCast(right.shape[1])),
            .K = @as(u32, @intCast(left.shape[1])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, output.buffer, 0, output.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, left.buffer, 0, left.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, right.buffer, 0, right.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(3, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const DispatchGroups = struct {
            X: u32,
            Y: u32,
            Z: u32,
        };
        const dispatch_groups = DispatchGroups{
            .X = params.M,
            .Y = params.N,
            .Z = 1,
        };

        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();
        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        // Submit commands
        var command = command_encoder.finish(null);
        defer command.release();

        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};

pub fn init(app: *App) !void {
    const allocator = gpa.allocator();

    try core.init(.{});
    app.* = .{};

    const tokenizer = try read_tokenizer(allocator);

    const str = "Hello this is a test the monkey sat on a banana-pie, and he squished it. What a mess? for (int i = 0; i < 1204; i += 1) {dosomethign(); } tekening";
    const tokens = try tokenize(allocator, str, tokenizer);
    std.log.info("Tokenized:", .{});
    for (tokens) |token| {
        std.log.info("token: {s}", .{tokenizer.tokens.items[token]});
    }

    const mat_operator = try MatOperator.init(allocator);

    const rope_operator = try RopeOperator.init(allocator);

    const add_operator = try AddOperator.init(allocator);

    const attention_operator = try AttentionOperator.init(allocator);

    const rmsnorm_operator = try RMSNormOperator.init(allocator);

    const silu_operator = try SILUOperator.init(allocator);

    const n_heads = 4;

    const L = 2048;
    const dim = 512;
    var x_shape = [_]usize{ L, dim };
    var x = try Tensor.init(allocator, &x_shape, .Storage);

    var k_shape = [_]usize{ L, dim };
    var k = try Tensor.init(allocator, &k_shape, .Storage);

    var v_shape = [_]usize{ L, dim };
    var v = try Tensor.init(allocator, &v_shape, .Storage);

    var l_shape = [_]usize{ dim, dim };
    var l = try Tensor.init(allocator, &l_shape, .Storage);

    var o_shape = [_]usize{ L, dim };
    var o = try Tensor.init(allocator, &o_shape, .Storage);

    var slate_shape = [_]usize{ n_heads, L, L };
    var slate = try Tensor.init(allocator, &slate_shape, .Storage);

    std.log.info("mat", .{});
    mat_operator.execute(l, x, o);

    std.log.info("add", .{});
    add_operator.execute(x, o);

    std.log.info("rms", .{});
    rmsnorm_operator.execute(x);

    std.log.info("rope", .{});
    rope_operator.execute(x, k, n_heads);

    silu_operator.execute(x);
    // _ = v;
    // _ = slate;
    // _ = attention_operator;
    std.log.info("attention", .{});
    attention_operator.execute(x, k, v, slate, o, n_heads);

    o.read_data();
}

pub fn deinit(app: *App) void {
    _ = app;
    // defer _ = gpa.deinit();
    core.deinit();
}

pub fn update(_: *App) !bool {
    return true;
}
