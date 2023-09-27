const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

const llm = @import("index.zig");
const Tensor = llm.Tensor;

const DispatchGroups = struct {
    X: u32,
    Y: u32,
    Z: u32,
};

fn div_ceil(a: u32, divider: u32) u32 {
    return (a + divider - 1) / divider;
}

pub const AttentionOperator = struct {
    shader_module_slate: *gpu.ShaderModule,
    shader_module_softmax_value: *gpu.ShaderModule,
    pipeline_slate: *gpu.ComputePipeline,
    pipeline_softmax_value: *gpu.ComputePipeline,
    pipeline_aggregate_value: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L_k: u32,
        L_q: u32,
        n_heads: u32,
        n_kv_heads: u32,
        K_max: u32,
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

        const shader_module_aggregate_value = core.device.createShaderModuleWGSL(
            "attention_aggregate_value.wgsl",
            @embedFile("shaders/attention_aggregate_value.wgsl"),
        );
        const pipeline_aggregate_value = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{
            .compute = gpu.ProgrammableStageDescriptor{
                .module = shader_module_aggregate_value,
                .entry_point = "main",
            },
        });

        operator.* = .{
            .shader_module_slate = shader_module_slate,
            .pipeline_slate = pipeline_slate,

            .shader_module_softmax_value = shader_module_softmax_value,
            .pipeline_softmax_value = pipeline_softmax_value,
            .pipeline_aggregate_value = pipeline_aggregate_value,

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
        n_kv_heads: usize,
        K_max: usize,
        command_encoder: anytype,
    ) void {
        std.log.debug("Q:{any} K:{any} V:{any}", .{ Q.shape, K.shape, V.shape });
        std.debug.assert(Q.shape.len == 2);
        std.debug.assert(K.shape.len == 2);
        std.debug.assert(V.shape.len == 2);

        const params: Params = .{
            .L_q = @as(u32, @intCast(Q.shape[1])),
            .L_k = @as(u32, @intCast(K.shape[1])),
            .dim = @as(u32, @intCast(Q.shape[0])),
            .n_heads = @as(u32, @intCast(n_heads)),
            .n_kv_heads = @intCast(n_kv_heads),
            .K_max = @as(u32, @intCast(K_max)),
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
                gpu.BindGroup.Entry.buffer(0, slate.buffer, 0, slate.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer softmax_bindings.release();

        const aggregate_bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline_aggregate_value.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, output.buffer, 0, output.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, V.buffer, 0, V.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(2, slate.buffer, 0, slate.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(3, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer aggregate_bindings.release();

        {
            const dispatch_groups = DispatchGroups{
                .X = div_ceil(params.L_q, 1),
                .Y = div_ceil(params.K_max, 16),
                .Z = div_ceil(params.n_heads, 16),
            };
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_slate);
            pass_encoder.setBindGroup(0, slate_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        {
            const dispatch_groups = DispatchGroups{
                .X = params.L_q,
                .Y = 1,
                .Z = 1,
            };
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_softmax_value);
            pass_encoder.setBindGroup(0, softmax_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }

        {
            const dispatch_groups = DispatchGroups{
                .X = params.L_q,
                .Y = div_ceil(params.dim, 32),
                .Z = 1,
            };

            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline_aggregate_value);
            pass_encoder.setBindGroup(0, aggregate_bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const RMSNormOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const AddOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const LookupOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        N: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*LookupOperator {
        var operator = try allocator.create(LookupOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "lookup.wgsl",
            @embedFile("shaders/lookup.wgsl"),
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
        self: *LookupOperator,
        output: *Tensor,
        source: *Tensor,
        command_encoder: anytype,
    ) void {
        std.debug.assert(output.N == source.N);

        const params: Params = .{
            .N = @as(u32, @intCast(source.N)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, output.buffer, 0, output.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, source.buffer, 0, source.N * @sizeOf(u8)),
                gpu.BindGroup.Entry.buffer(2, source.lookup_buffer.?, 0, 256 * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(3, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const dispatch_groups = DispatchGroups{
            .X = params.N / 1024,
            .Y = 1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const TransposeOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.dim0,
            .Y = params.dim1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const ArgmaxOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const EmbedOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.dim,
            .Y = params.L,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const ElMulOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const CopyOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        X: u32,
    };

    pub fn init(allocator: std.mem.Allocator) !*CopyOperator {
        var operator = try allocator.create(CopyOperator);
        const shader_module = core.device.createShaderModuleWGSL(
            "elmul_inplace.wgsl",
            @embedFile("shaders/copy.wgsl"),
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
        self: *CopyOperator,
        left: *Tensor,
        right: *Tensor,
        command_encoder: anytype,
    ) void {
        std.debug.assert(left.N == right.N);

        const params: Params = .{
            .X = @as(u32, @intCast(left.N)),
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

        const dispatch_groups = DispatchGroups{
            .X = div_ceil(params.X, 64),
            .Y = 1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const ScaleOperator = struct {
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
        command_encoder: anytype,
    ) void {
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const SILUOperator = struct {
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
        command_encoder: anytype,
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

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = params.dim,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const RopeOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        dim: u32,
        L: u32,
        n_heads: u32,
        base_freq: f32,
        l_offset: u32,
        write_l_offset: u32,
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
        n_heads: usize,
        l_offset: ?usize,
        write_l_offset: ?usize,
        command_encoder: anytype,
    ) void {
        std.debug.assert(k.shape.len == 2);

        const params: Params = .{
            .L = @as(u32, @intCast(k.shape[1])),
            .dim = @as(u32, @intCast(k.shape[0])),
            .n_heads = @as(u32, @intCast(n_heads)),
            .base_freq = 10000.0,
            .l_offset = @as(u32, @intCast(l_offset orelse 0)),
            .write_l_offset = @as(u32, @intCast(write_l_offset orelse 0)),
        };

        core.queue.writeBuffer(self.param_buffer, 0, std.mem.asBytes(&params));

        const bindings = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
            .layout = self.pipeline.getBindGroupLayout(0),
            .entries = &.{
                gpu.BindGroup.Entry.buffer(0, k.buffer, 0, k.N * @sizeOf(f32)),
                gpu.BindGroup.Entry.buffer(1, self.param_buffer, 0, @sizeOf(Params)),
            },
        }));
        defer bindings.release();

        const dispatch_groups = DispatchGroups{
            .X = params.L,
            .Y = 1,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

pub const MatOperator = struct {
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
        command_encoder: anytype,
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

        const G = 1;
        const dispatch_groups = DispatchGroups{
            .X = (params.M + G - 1) / G,
            .Y = (params.N + G - 1) / G,
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};

// Mat operator where left matrix is transposed (makes more sense memory wise)
pub const TransposeMatOperator = struct {
    shader_module: *gpu.ShaderModule,
    pipeline: *gpu.ComputePipeline,
    param_buffer: *gpu.Buffer,

    const Params = struct {
        M: u32,
        K: u32,
        N: u32,
        output_offset: u32,
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
        target_idx: ?usize,
        command_encoder: anytype,
    ) void {
        std.debug.assert(left.shape.len == 2);
        std.debug.assert(right.shape.len == 2);
        std.debug.assert(output.shape.len == 2);

        std.debug.assert(left.shape[0] == right.shape[0]);
        std.debug.assert(left.shape[1] == output.shape[0]);
        // std.debug.assert(right.shape[1] == output.shape[1]);

        const dim = left.shape[0];

        const output_offset = if (target_idx) |idx| idx * dim else 0;

        const params: Params = .{
            .M = @as(u32, @intCast(left.shape[1])),
            .N = @as(u32, @intCast(right.shape[1])),
            .K = @as(u32, @intCast(right.shape[0])),
            .output_offset = @as(u32, @intCast(output_offset)),
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

        const dispatch_groups = DispatchGroups{
            .X = div_ceil(params.M, 8),
            .Y = div_ceil(params.N, 8),
            .Z = 1,
        };

        {
            const pass_encoder = command_encoder.beginComputePass(null);
            pass_encoder.setPipeline(self.pipeline);
            pass_encoder.setBindGroup(0, bindings, null);
            pass_encoder.dispatchWorkgroups(dispatch_groups.X, dispatch_groups.Y, dispatch_groups.Z);
            pass_encoder.end();
        }
    }
};
