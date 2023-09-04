const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

const llm = @import("index.zig");
const Tensor = llm.Tensor;
const Tokenizer = llm.Tokenizer;

const io = llm.io;

pub const App = @This();

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const workgroup_size = 64;
const buffer_size = 1000;

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
            @embedFile("shaders/attention_slate.wgsl"),
        );
        const pipeline_slate = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{
            .compute = gpu.ProgrammableStageDescriptor{
                .module = shader_module_slate,
                .entry_point = "main",
            },
        });

        const shader_module_softmax_value = core.device.createShaderModuleWGSL(
            "attention_softmax_value.wgsl",
            @embedFile("shaders/attention_softmax_value.wgsl"),
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
        n_heads: usize,
    ) void {
        std.debug.assert(Q.shape.len == 2);
        std.debug.assert(K.shape.len == 2);
        std.debug.assert(V.shape.len == 2);

        const params: Params = .{
            .L = @as(u32, @intCast(Q.shape[1])),
            .dim = @as(u32, @intCast(Q.shape[0])),
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
            @embedFile("shaders/rmsnorm_inplace.wgsl"),
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
            .L = @as(u32, @intCast(x.shape[1])),
            .dim = @as(u32, @intCast(x.shape[0])),
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
            @embedFile("shaders/add_inplace.wgsl"),
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
            .L = @as(u32, @intCast(left.shape[1])),
            .dim = @as(u32, @intCast(left.shape[0])),
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

const ArgmaxOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*ArgmaxOperator {
        var operator = try allocator.create(ArgmaxOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "argmax.wgsl",
            @embedFile("shaders/argmax.wgsl"),
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
        self: *ArgmaxOperator,
        values: *Tensor,
        max_index: *Tensor,
    ) void {
        const params: Params = .{
            .L = @as(u32, @intCast(values.shape[1])),
            .dim = @as(u32, @intCast(values.shape[0])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, values.buffer, 0, values.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, max_index.buffer, 0, max_index.N * @sizeOf(u32)),
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

const EmbedOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
        n_tokens: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*EmbedOperator {
        var operator = try allocator.create(EmbedOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "embed_token.wgsl",
            @embedFile("shaders/embed_token.wgsl"),
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
        self: *EmbedOperator,
        embeddings: *Tensor,
        tokens: *Tensor,
        seq_len: usize,
        x: *Tensor,
    ) void {
        const params: Params = .{
            .L = @as(u32, @intCast(seq_len)),
            .dim = @as(u32, @intCast(embeddings.shape[0])),
            .n_tokens = @as(u32, @intCast(tokens.N)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, embeddings.buffer, 0, embeddings.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, tokens.buffer, 0, tokens.N * @sizeOf(u32)),
                gpu.BindGroup.Entry.buffer(2, x.buffer, 0, x.N * @sizeOf(f32)),
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

const ElMulOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*ElMulOperator {
        var operator = try allocator.create(ElMulOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "elmul_inplace.wgsl",
            @embedFile("shaders/elmul_inplace.wgsl"),
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
        self: *ElMulOperator,
        left: *Tensor,
        right: *Tensor,
    ) void {
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);
        std.debug.assert(std.mem.eql(usize, left.shape, right.shape));

        const params: Params = .{
            .L = @as(u32, @intCast(left.shape[1])),
            .dim = @as(u32, @intCast(left.shape[0])),
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
            @embedFile("shaders/silu_inplace.wgsl"),
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
            .L = @as(u32, @intCast(x.shape[1])),
            .dim = @as(u32, @intCast(x.shape[0])),
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
            @embedFile("shaders/rope.wgsl"),
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
        q: *Tensor,
        n_heads: usize,
    ) void {
        std.debug.assert(k.shape.len == 2);
        std.debug.assert(q.shape.len == 2);
        std.debug.assert(std.mem.eql(usize, k.shape, q.shape));

        const params: Params = .{
            .L = @as(u32, @intCast(k.shape[1])),
            .dim = @as(u32, @intCast(k.shape[0])),
            .n_heads = @as(u32, @intCast(n_heads)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, k.buffer, 0, k.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, q.buffer, 0, q.N * @sizeOf(f32)),
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
            @embedFile("shaders/matmul.wgsl"),
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

        std.debug.assert(left.shape[1] == right.shape[0]);
        std.debug.assert(left.shape[0] == output.shape[0]);
        std.debug.assert(right.shape[1] == output.shape[1]);

        const params: Params = .{
            .M = @as(u32, @intCast(left.shape[0])),
            .N = @as(u32, @intCast(right.shape[1])),
            .K = @as(u32, @intCast(right.shape[0])),
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

    const seed: u64 = 123;
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    const tokenizer_path = "/home/marijnfs/Downloads/tokenizer.bin";
    const model_path = "/home/marijnfs/Downloads/stories15M.bin";

    const model_weights = try io.read_model_weights(allocator, model_path);
    const config = model_weights.config;

    const vocab_size = @as(usize, @intCast(config.vocab_size));

    const tokenizer = try io.read_tokenizer(allocator, vocab_size, tokenizer_path);

    const str = "Hello this is a test the monkey sat on a banana-pie, and he squished it. What a mess? for (int i = 0; i < 1204; i += 1) {dosomethign(); } tekening";
    const tokens = try llm.tokenize(allocator, str, tokenizer);
    std.log.info("Tokenized:", .{});
    for (tokens) |token| {
        std.log.info("token: {s}", .{tokenizer.tokens.items[token]});
    }

    const mat_operator = try MatOperator.init(allocator);

    const rope_operator = try RopeOperator.init(allocator);

    const elmul_operator = try ElMulOperator.init(allocator);

    const add_operator = try AddOperator.init(allocator);

    const attention_operator = try AttentionOperator.init(allocator);

    const rmsnorm_operator = try RMSNormOperator.init(allocator);

    const silu_operator = try SILUOperator.init(allocator);

    const embed_operator = try EmbedOperator.init(allocator);

    const argmax_operator = try ArgmaxOperator.init(allocator);

    const n_heads = @as(usize, @intCast(config.n_heads));

    // Steps:
    // -> RMS norm x, with weights
    // -> matmul x to q, k, v
    // -> rope q and k (some versions only rope one)
    // -> attention
    // -> matmul attention output with out weights
    // -> add x (before rms norm)
    // -> rms norm, with weights again -> output
    // -> matmul output with w1, silu output
    // -> matmul output with w3 (bad naming)
    // -> elmul both outputs
    // -> matmul with w2
    // -> add x again

    // -> final steps
    // -> rmsnorm with weights again
    // -> matmul with class weights toward vocab size

    const L = 512;
    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));

    var random_values = try allocator.alloc(f32, L * dim);
    defer allocator.free(random_values);

    for (random_values) |*v| {
        v.* = (random.float(f32) * 2 - 1) * 0.05;
    }

    var x = try Tensor.init_from_data(allocator, &[_]usize{ L, dim }, .Storage, random_values);
    var x_copy = try Tensor.init(allocator, &[_]usize{ L, dim }, .Storage);

    var k = try Tensor.init(allocator, &[_]usize{ L, dim }, .Storage);
    var q = try Tensor.init(allocator, &[_]usize{ L, dim }, .Storage);
    var v = try Tensor.init(allocator, &[_]usize{ L, dim }, .Storage);

    var out = try Tensor.init(allocator, &[_]usize{ L, dim }, .Storage);

    var w1_slate = try Tensor.init(allocator, &[_]usize{ L, hidden_dim }, .Storage);
    var w3_slate = try Tensor.init(allocator, &[_]usize{ L, hidden_dim }, .Storage);

    var logits = try Tensor.init(allocator, &[_]usize{ L, vocab_size }, .Storage);
    var slate = try Tensor.init(allocator, &[_]usize{ n_heads, L, L }, .Storage);

    var max_index = try Tensor.init_u32(allocator, &[_]usize{L}, .Storage);

    // var tokens_tensor = try Tensor.init_from_tokens(allocator, tokens);
    // embed_operator.execute(model_weights.token_embedding, tokens_tensor, L, x);
    _ = embed_operator;

    for (model_weights.layers.items) |*layer| {
        x.copy_to(x_copy);
        rmsnorm_operator.execute(x);
        mat_operator.execute(layer.query_weight, x, q);
        mat_operator.execute(layer.key_weight, x, k);
        mat_operator.execute(layer.value_weight, x, v);

        rope_operator.execute(k, q, n_heads);

        attention_operator.execute(q, k, v, slate, out, n_heads);
        add_operator.execute(out, x_copy);
        rmsnorm_operator.execute(out);

        mat_operator.execute(layer.w1, out, w1_slate);
        mat_operator.execute(layer.w3, out, w3_slate);
        silu_operator.execute(w1_slate);

        elmul_operator.execute(w1_slate, w3_slate);
        mat_operator.execute(layer.w2, w1_slate, x);
        add_operator.execute(x, x_copy);
    }

    rmsnorm_operator.execute(x);
    const final_weights = model_weights.final_class_weights orelse model_weights.token_embedding;
    mat_operator.execute(final_weights, x, logits);

    argmax_operator.execute(logits, max_index);
    max_index.read_data_u32(tokenizer);
}

pub fn deinit(app: *App) void {
    _ = app;
    // defer _ = gpa.deinit();
    core.deinit();
}

pub fn update(_: *App) !bool {
    return true;
}
