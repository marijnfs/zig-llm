const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

const llm = @import("index.zig");

// Tensor
//   - convention: shape size order starts with 'first' dimension at index 0

pub const Tensor = struct {
    const Type = enum {
        Storage,
        Target,
    };
    shape: []const usize,
    N: usize,
    buffer: *gpu.Buffer,
    lookup_buffer: ?*gpu.Buffer = null, // Buffer used when we have lookup table compression

    pub fn init_f32(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type) !*Tensor {
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

    pub fn init_f16(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type) !*Tensor {
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
                .size = N * @sizeOf(f16),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(f16, 0, N);
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

    pub fn init_from_data_f16(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []const f16) !*Tensor {
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
                .size = N * @sizeOf(f16),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(f16, 0, N);
        std.mem.copy(f16, buffer_mapped.?, data);
        tensor.buffer.unmap();

        return tensor;
    }

    pub fn init_from_data_f16_to_f32(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []const f16) !*Tensor {
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
        for (buffer_mapped.?, data) |*target, source| {
            target.* = source;
        }
        tensor.buffer.unmap();

        return tensor;
    }

    pub fn init_from_data_f32(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []const f32) !*Tensor {
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

    pub fn init_from_tokens(allocator: std.mem.Allocator, data: []const u32) !*Tensor {
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

    pub fn init_from_data_q8_lookup(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []const u8, lookup_table: []const f16) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }

        const TableSize = 256;

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
                .size = N * @sizeOf(u8),
                .mapped_at_creation = .true,
            }),
            .lookup_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = TableSize * @sizeOf(f16),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(u8, 0, N);
        std.mem.copy(u8, buffer_mapped.?, data);
        tensor.buffer.unmap();

        var lookup_buffer_mapped = tensor.lookup_buffer.?.getMappedRange(f16, 0, TableSize);
        std.mem.copy(f16, lookup_buffer_mapped.?, lookup_table);
        tensor.lookup_buffer.?.unmap();

        return tensor;
    }

    pub fn init_from_data_q8_lookup_to_f32(allocator: std.mem.Allocator, shape: []const usize, tensor_type: Type, data: []const u8, lookup_table: []const f16) !*Tensor {
        var tensor = try allocator.create(Tensor);
        _ = tensor_type;

        var N: usize = 1;
        for (shape) |dim| {
            N *= dim;
        }

        const TableSize = 256;

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
                .size = N * @sizeOf(u8),
                .mapped_at_creation = .true,
            }),
            .lookup_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
                .label = "buffer",
                .usage = .{
                    .storage = true,
                    .copy_dst = true,
                    .copy_src = true,
                },
                .size = TableSize * @sizeOf(f32),
                .mapped_at_creation = .true,
            }),
        };

        var buffer_mapped = tensor.buffer.getMappedRange(u8, 0, N);
        std.mem.copy(u8, buffer_mapped.?, data);
        tensor.buffer.unmap();

        var lookup_buffer_mapped = tensor.lookup_buffer.?.getMappedRange(f32, 0, TableSize);

        // we use a loop copy, to convert f16 to f32
        for (lookup_buffer_mapped.?, lookup_table) |*target, source| {
            target.* = source;
        }
        // std.mem.copy(f16, lookup_buffer_mapped.?, lookup_table);
        tensor.lookup_buffer.?.unmap();

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
        for (output_mapped.?, 0..) |v, idx| {
            if (idx % 4 == 0)
                std.debug.print("[{}] {d:.8}\n", .{ idx, v });
            //std.debug.print("{s}{d:.8}\n", .{ if (idx % self.shape[0] == 0) "\n" else "", v });
        }
        std.debug.print("\n", .{});
    }

    pub fn read_data_to_file(self: *Tensor, path: []const u8) !void {
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

        // Create file
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Setup buffered writer
        var model_file_buffered = std.io.bufferedWriter(file.writer());
        var model_writer = model_file_buffered.writer();

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
            try model_writer.print("{d:.8} ", .{v});
        }
        try model_file_buffered.flush();

        std.debug.print("\n", .{});
    }

    pub fn read_data_tokens(self: *Tensor, allocator: std.mem.Allocator) ![]const u32 {
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

        const token_copy = try allocator.dupe(u32, output_mapped.?);
        return token_copy;
    }

    pub fn create_matching_output_buffer(self: *Tensor) *gpu.Buffer {
        const output_buffer = core.device.createBuffer(&gpu.Buffer.Descriptor{
            .label = "output_buffer",
            .usage = .{ .copy_dst = true, .map_read = true },
            .size = self.N * @sizeOf(f32),
        });
        return output_buffer;
    }

    pub fn copy_to(self: *Tensor, to: *Tensor, command_encoder: anytype) void {
        std.debug.assert(self.N == to.N);
        command_encoder.copyBufferToBuffer(self.buffer, 0, to.buffer, 0, self.N * @sizeOf(f32));
    }

    pub fn size_from(self: *Tensor, idx: usize) usize {
        var size: usize = 1;
        for (self.shape[idx..]) |dim| {
            size *= dim;
        }
        return size;
    }

    pub fn size_until(self: *Tensor, idx: usize) usize {
        var size: usize = 1;
        for (self.shape[0..idx]) |dim| {
            size *= dim;
        }
        return size;
    }
};
