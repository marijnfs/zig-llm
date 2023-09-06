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
        std.log.debug("Q:{any} K:{any} V:{any}", .{ Q.shape, K.shape, V.shape });
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
                gpu.BindGroup.Entry.buffer(0, slate.buffer, 0, slate.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, K.buffer, 0, K.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, Q.buffer, 0, Q.N * @sizeOf(f32)),
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

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_slate);
            pass_encoder.setBindGroup(0, slate_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_softmax_value);
            pass_encoder.setBindGroup(0, softmax_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

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

const TransposeOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim0: u32,
        dim1: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*TransposeOperator {
        var operator = try allocator.create(TransposeOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "transpose.wgsl",
            @embedFile("shaders/transpose.wgsl"),
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
        self: *TransposeOperator,
        left: *Tensor,
        right: *Tensor,
    ) void {
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);

        std.debug.assert(left.shape[0] == right.shape[1]);
        std.debug.assert(left.shape[1] == right.shape[0]);

        const params: Params = .{
            .dim0 = @as(u32, @intCast(right.shape[0])),
            .dim1 = @as(u32, @intCast(right.shape[1])),
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
            .X = params.dim0,
            .Y = params.dim1,
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
        max_index: *Tensor,
        values: *Tensor,
    ) void {
        const params: Params = .{
            .L = @as(u32, @intCast(values.shape[1])),
            .dim = @as(u32, @intCast(values.shape[0])),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, max_index.buffer, 0, max_index.N * @sizeOf(u32)),
                gpu.BindGroup.Entry.buffer(1, values.buffer, 0, values.N * @sizeOf(f32)),
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
        x: *Tensor,
        tokens: *Tensor,
        embeddings: *Tensor,
        seq_len: usize,
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
                gpu.BindGroup.Entry.buffer(0, x.buffer, 0, x.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, tokens.buffer, 0, tokens.N * @sizeOf(u32)),
                gpu.BindGroup.Entry.buffer(2, embeddings.buffer, 0, embeddings.N * @sizeOf(f32)),
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
            .X = params.dim,
            .Y = params.L,
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

const ScaleOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*ScaleOperator {
        var operator = try allocator.create(ScaleOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "scale.wgsl",
            @embedFile("shaders/scale.wgsl"),
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
        self: *ScaleOperator,
        left: *Tensor,
        right: *Tensor,
    ) void {
        std.log.info("{any} {any}", .{ left.shape, right.shape });
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 1);
        std.debug.assert(left.shape[0] == right.shape[0]);

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
        std.log.info("mat: {any} {any} {any}", .{ left.shape, right.shape, output.shape });

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

// Mat operator where left matrix is transposed (makes more sense memory wise)
const TransposeMatOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        M: u32,
        K: u32,
        N: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*TransposeMatOperator {
        var operator = try allocator.create(TransposeMatOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "transposematmul.wgsl",
            @embedFile("shaders/transposematmul.wgsl"),
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
        self: *TransposeMatOperator,
        left: *Tensor,
        right: *Tensor,
        output: *Tensor,
    ) void {
        std.log.info("mat: {any} {any} {any}", .{ left.shape, right.shape, output.shape });

        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);
        std.debug.assert(output.shape.len == 2);

        std.debug.assert(left.shape[0] == right.shape[0]);
        std.debug.assert(left.shape[1] == output.shape[0]);
        std.debug.assert(right.shape[1] == output.shape[1]);

        const params: Params = .{
            .M = @as(u32, @intCast(left.shape[1])),
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

    const str = "Once upon a";
    const tokens = try llm.tokenize(allocator, str, tokenizer);
    // _ = tokens;

    std.log.info("Tokenized:", .{});
    for (tokens) |token| {
        std.log.info("token: '{s}' {}", .{ tokenizer.tokens.items[token], token });
    }

    // const mat_operator = try MatOperator.init(allocator);

    const tmat_operator = try TransposeMatOperator.init(allocator);

    const rope_operator = try RopeOperator.init(allocator);

    const elmul_operator = try ElMulOperator.init(allocator);

    const scale_operator = try ScaleOperator.init(allocator);

    const add_operator = try AddOperator.init(allocator);

    const attention_operator = try AttentionOperator.init(allocator);

    const rmsnorm_operator = try RMSNormOperator.init(allocator);

    const silu_operator = try SILUOperator.init(allocator);

    const embed_operator = try EmbedOperator.init(allocator);

    const argmax_operator = try ArgmaxOperator.init(allocator);

    const transpose_operator = try TransposeOperator.init(allocator);

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

    const L = tokens.len;

    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));

    var random_values = try allocator.alloc(f32, L * dim);
    defer allocator.free(random_values);

    for (random_values) |*v| {
        v.* = (random.float(f32) * 2 - 1) * 0.25;
    }

    var embedding_transposed = try Tensor.init(allocator, &[_]usize{ vocab_size, dim }, .Storage);
    transpose_operator.execute(embedding_transposed, model_weights.token_embedding);

    var x = try Tensor.init_from_data(allocator, &[_]usize{ dim, L }, .Storage, random_values);
    var x_copy = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);

    var k = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);
    var q = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);
    var v = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);

    var attention_out = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);
    var out = try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage);

    var w1_slate = try Tensor.init(allocator, &[_]usize{ hidden_dim, L }, .Storage);
    var w3_slate = try Tensor.init(allocator, &[_]usize{ hidden_dim, L }, .Storage);

    var logits = try Tensor.init(allocator, &[_]usize{ vocab_size, L }, .Storage);
    var slate = try Tensor.init(allocator, &[_]usize{ L, L, n_heads }, .Storage);

    var max_index = try Tensor.init_u32(allocator, &[_]usize{L}, .Storage);

    // {
    //     mat_operator.execute(embedding_transposed, x, logits);

    //     argmax_operator.execute(max_index, logits);
    //     max_index.read_data_tokens(tokenizer);
    //     if (true)
    //         return;
    // }
    // std.log.info("init", .{});

    var tokens_tensor = try Tensor.init_from_tokens(allocator, tokens);
    embed_operator.execute(x, tokens_tensor, model_weights.token_embedding, L);
    // _ = embed_operator;

    for (model_weights.layers.items) |*layer| {
        x.copy_to(x_copy);

        rmsnorm_operator.execute(x);
        scale_operator.execute(x, layer.rms_attention);

        tmat_operator.execute(layer.query_weight, x, q);
        tmat_operator.execute(layer.key_weight, x, k);
        tmat_operator.execute(layer.value_weight, x, v);

        rope_operator.execute(k, q, n_heads);

        attention_operator.execute(q, k, v, slate, attention_out, n_heads);
        // out.read_data();
        // if (true) return;
        tmat_operator.execute(layer.output_weight, attention_out, out);

        add_operator.execute(out, x_copy);
        out.copy_to(x_copy);

        rmsnorm_operator.execute(out);
        scale_operator.execute(out, layer.rms_ffn);

        tmat_operator.execute(layer.w1, out, w1_slate);
        tmat_operator.execute(layer.w3, out, w3_slate);
        silu_operator.execute(w1_slate);

        elmul_operator.execute(w1_slate, w3_slate);
        tmat_operator.execute(layer.w2, w1_slate, x);
        add_operator.execute(x, x_copy);
    }

    rmsnorm_operator.execute(x);
    scale_operator.execute(x, model_weights.final_rms_weight);

    // _ = logits;
    // _ = max_index;
    // _ = argmax_operator;
    const final_weights = model_weights.final_class_weights orelse model_weights.token_embedding;
    tmat_operator.execute(final_weights, x, logits);

    argmax_operator.execute(max_index, logits);
    max_index.read_data_tokens(tokenizer);
}

pub fn deinit(app: *App) void {
    _ = app;
    // defer _ = gpa.deinit();
    core.deinit();
}

pub fn update(_: *App) !bool {
    return true;
}
