const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

const llm = @import("index.zig");

pub const Tensor = struct {
    const Type = enum {
        Storage,
        Target,
    };

    shape: []const usize,
    N: usize,
    buffer: *gpu.Buffer,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }
        tensor.* = .{
            .N = N,
            .shape = try allocator.dupe(usize, shape),
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

    pub fn init_u32(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }
        tensor.* = .{
            .N = N,
            .shape = try allocator.dupe(usize, shape),
            .buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = N * @sizeOf(u32),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(u32, 0, N);
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
            .shape = try allocator.dupe(usize, shape),
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

    pub fn init_from_tokens(allocator: std.mem.Allocator, data: []u32) !*Tensor {
        var tensor = try allocator.create(Tensor);

        var N: usize = data.len;
        tensor.* = .{
            .N = N,
            .shape = try allocator.dupe(usize, &[_]usize{N}),
            .buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = N * @sizeOf(u32),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(u32, 0, N);
        std.mem.copy(u32, buffer_mapped.?, data);
        tensor.buffer.unmap();

        return tensor;
    }

    pub fn read_data(self: *Tensor) void {
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

    fn read_data_u32(self: *Tensor, tokenizer: *llm.Tokenizer) void {
        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();

        const output_buffer = self.create_matching_output_buffer();
        defer output_buffer.release();

        command_encoder.copyBufferToBuffer(self.buffer, 0, output_buffer, 0, self.N * @sizeOf(u32));

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
        output_buffer.mapAsync(.{ .read = true }, 0, self.N * @sizeOf(u32), &response, callback);
        while (true) {
            if (response == gpu.Buffer.MapAsyncStatus.success) {
                break;
            } else {
                core.device.tick();
            }
        }

        const output_mapped = output_buffer.getConstMappedRange(u32, 0, self.N);
        defer output_buffer.unmap();
        for (output_mapped.?) |v| {
            if (v == 0)
                continue;
            std.debug.print("{d} {s}\n", .{ v, tokenizer.tokens.items[v] });
        }
        std.debug.print("\n", .{});
    }

    pub fn create_matching_output_buffer(self: *Tensor) *gpu.Buffer {
        const output_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
            .label = "output_buffer",
            .usage = .{ .copy_dst = true, .map_read = true },
            .size = self.N * @sizeOf(f32),
        });
        return output_buffer;
    }

    pub fn copy_to(self: *Tensor, to: *Tensor) void {
        std.debug.assert(self.N == to.N);
        const command_encoder = core.device.createCommandEncoder(null);
        defer command_encoder.release();

        command_encoder.copyBufferToBuffer(self.buffer, 0, to.buffer, 0, self.N * @sizeOf(f32));

        var command = command_encoder.finish(null);
        defer command.release();
        core.queue.submit(&[_]*gpu.CommandBuffer{command});
    }
};
