const std = @import("std");
const llm = @import("index.zig");
const model = llm.model;

const ModelWeights = llm.model.ModelWeights;
const Tensor = llm.Tensor;
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

pub fn read_model_weights(base_allocator: std.mem.Allocator, path: []const u8) !*ModelWeights {

    // Open file
    const checkpoint_path = path;
    var file = try std.fs.cwd().openFile(checkpoint_path, .{});
    defer file.close();

    const magic = b: {
        var model_reader = file.reader();
        const magic = try model_reader.readInt(u32, .Little);
        try file.seekTo(0); //set back file after reading
        break :b magic;
    };

    const karpathy_magic_byte = 0x616b3432;
    const our_magic_byte = 0x7a657865;

    if (magic == karpathy_magic_byte) {
        return error.ModelFormatNotSupported;
    } else if (magic == our_magic_byte) {
        return try read_model_weights_ours(base_allocator, file.reader());
    } else {
        // Assume legacy format
        return try read_model_weights_karpathy_legacy(base_allocator, file.reader());
    }

    return error.ModelFormatNotSupported;
}

pub fn read_model_weights_ours(base_allocator: std.mem.Allocator, base_reader: anytype) !*ModelWeights {
    // Setup arena
    var arena = std.heap.ArenaAllocator.init(base_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    // Setup buffered reader
    var model_file_buffered = std.io.bufferedReader(base_reader);
    var model_reader = model_file_buffered.reader();

    // Reading our version header
    const our_magic_byte = 0x7a657865;

    const Header = extern struct {
        magic: u32,
        major: u32,
        minor: u32,
    };

    var header = try model_reader.readStruct(Header);
    std.log.info("Header: {}", .{header});

    std.debug.assert(header.magic == our_magic_byte);

    if (header.major > 0) {
        return error.VersionTooNew;
    }

    // Setup data

    var model_weights = try base_allocator.create(ModelWeights);

    // Read buffer
    var weight_read_buffer = std.ArrayList(f16).init(arena_allocator);
    var q8_read_buffer = std.ArrayList(u8).init(arena_allocator);

    // Read config file
    var config = try model_reader.readStruct(model.ModelConfig);

    std.log.info("Config: {}", .{config});

    const n_layers = @as(usize, @intCast(config.n_layers));
    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));
    const vocab_size = @as(usize, @intCast(config.vocab_size));
    const n_heads = @as(usize, @intCast(config.n_heads));

    const head_size = dim / n_heads;
    const n_kv_heads = @as(usize, @intCast(config.n_kv_heads));
    const kv_dim = n_kv_heads * head_size;

    // Read token embedding
    const read_f16 = struct {
        fn f(allocator: std.mem.Allocator, shape: []const usize, reader_: anytype, weight_buffer: anytype) !*Tensor {
            var N: usize = 1;
            for (shape) |val| {
                N *= val;
            }

            try weight_buffer.resize(N);
            _ = try reader_.readAll(std.mem.sliceAsBytes(weight_buffer.items));

            return try Tensor.init_from_data_f16_to_f32(allocator, shape, .Storage, weight_buffer.items);
        }
    }.f;

    const read_q8 = struct {
        fn f(allocator: std.mem.Allocator, shape: []const usize, reader_: anytype, lookups_buffer: anytype, table_buffer: anytype) !*Tensor {
            var N: usize = 1;
            for (shape) |val| {
                N *= val;
            }

            try table_buffer.resize(256);
            try lookups_buffer.resize(N);
            _ = try reader_.readAll(std.mem.sliceAsBytes(table_buffer.items));
            _ = try reader_.readAll(std.mem.sliceAsBytes(lookups_buffer.items));

            return try Tensor.init_from_data_q8_lookup_to_f32(allocator, shape, .Storage, lookups_buffer.items, table_buffer.items);
        }
    }.f;

    var token_embedding = try read_f16(base_allocator, &[_]usize{ dim, vocab_size }, model_reader, &weight_read_buffer);

    var output_embedding = try read_f16(base_allocator, &[_]usize{ dim, vocab_size }, model_reader, &weight_read_buffer);

    var final_rms_weight = try read_f16(base_allocator, &[_]usize{dim}, model_reader, &weight_read_buffer);

    const n_freqs = head_size / 2;
    var freqs = try read_f16(base_allocator, &[_]usize{n_freqs}, model_reader, &weight_read_buffer);

    // Start reading weights
    var layer_weights = std.ArrayList(model.LayerWeights).init(base_allocator);
    try layer_weights.resize(n_layers);

    // query_weight
    for (layer_weights.items) |*layer| {
        layer.query_weight = try read_q8(base_allocator, &[_]usize{ dim, dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // key_weight
    for (layer_weights.items) |*layer| {
        layer.key_weight = try read_q8(base_allocator, &[_]usize{ kv_dim, dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // value_weight
    for (layer_weights.items) |*layer| {
        layer.value_weight = try read_q8(base_allocator, &[_]usize{ kv_dim, dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // output_weight
    for (layer_weights.items) |*layer| {
        layer.output_weight = try read_q8(base_allocator, &[_]usize{ dim, dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // w1
    for (layer_weights.items) |*layer| {
        layer.w1 = try read_q8(base_allocator, &[_]usize{ dim, hidden_dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // w2
    for (layer_weights.items) |*layer| {
        layer.w2 = try read_q8(base_allocator, &[_]usize{ hidden_dim, dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // w3
    for (layer_weights.items) |*layer| {
        layer.w3 = try read_q8(base_allocator, &[_]usize{ dim, hidden_dim }, model_reader, &q8_read_buffer, &weight_read_buffer);
    }

    // rms_attention
    for (layer_weights.items) |*layer| {
        layer.rms_attention = try read_f16(base_allocator, &[_]usize{dim}, model_reader, &weight_read_buffer);
    }

    // rms_ffn
    for (layer_weights.items) |*layer| {
        layer.rms_ffn = try read_f16(base_allocator, &[_]usize{dim}, model_reader, &weight_read_buffer);
    }

    model_weights.* = .{
        .config = config,
        .layers = layer_weights,
        .token_embedding = token_embedding,
        .output_embedding = output_embedding,
        .final_rms_weight = final_rms_weight,
        .freqs = freqs,
    };

    return model_weights;
}

pub fn read_model_weights_karpathy_legacy(base_allocator: std.mem.Allocator, reader: anytype) !*ModelWeights {

    // Setup arena
    var arena = std.heap.ArenaAllocator.init(base_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var model_weights = try base_allocator.create(ModelWeights);

    // Read buffer
    var weight_read_buffer = std.ArrayList(f32).init(arena_allocator);

    var model_file_buffered = std.io.bufferedReader(reader);
    var model_reader = model_file_buffered.reader();

    // Read config file
    var config = try model_reader.readStruct(model.ModelConfig);

    std.log.info("Config: {}", .{config});

    const n_layers = @as(usize, @intCast(config.n_layers));
    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));
    const vocab_size = @as(usize, @intCast(config.vocab_size));
    const n_heads = @as(usize, @intCast(config.n_heads));

    const head_size = dim / n_heads;
    const n_kv_heads = @as(usize, @intCast(config.n_kv_heads));
    const kv_dim = n_kv_heads * head_size;

    // Read token embedding
    try weight_read_buffer.resize(vocab_size * dim);
    const read = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));
    std.log.debug("read: {}", .{read});

    var token_embedding = try Tensor.init_from_data_f32(
        base_allocator,
        &[_]usize{ dim, vocab_size },
        .Storage,
        weight_read_buffer.items,
    );

    // Start reading weights
    var layer_weights = std.ArrayList(model.LayerWeights).init(base_allocator);
    try layer_weights.resize(n_layers);

    // rms_attention
    try weight_read_buffer.resize(dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.rms_attention = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{dim},
            .Storage,
            weight_read_buffer.items,
        );
    }

    // query_weight
    try weight_read_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.query_weight = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    // key_weight
    try weight_read_buffer.resize(kv_dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.key_weight = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ kv_dim, dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    // value_weight
    try weight_read_buffer.resize(kv_dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.value_weight = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ kv_dim, dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    //output_weight
    try weight_read_buffer.resize(dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.output_weight = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ dim, dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    // rms_ffn
    try weight_read_buffer.resize(dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.rms_ffn = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{dim},
            .Storage,
            weight_read_buffer.items,
        );
    }

    // w1
    try weight_read_buffer.resize(hidden_dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.w1 = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ dim, hidden_dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    // w2
    try weight_read_buffer.resize(dim * hidden_dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.w2 = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ hidden_dim, dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    // w3
    try weight_read_buffer.resize(hidden_dim * dim);
    for (layer_weights.items) |*layer| {
        _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

        layer.w3 = try Tensor.init_from_data_f32(
            base_allocator,
            &[_]usize{ dim, hidden_dim },
            .Storage,
            weight_read_buffer.items,
        );
    }

    try weight_read_buffer.resize(dim);
    _ = try model_reader.readAll(std.mem.sliceAsBytes(weight_read_buffer.items));

    var final_rms_weight = try Tensor.init_from_data_f32(
        base_allocator,
        &[_]usize{dim},
        .Storage,
        weight_read_buffer.items,
    );

    model_weights.* = .{
        .config = config,
        .layers = layer_weights,
        .token_embedding = token_embedding,
        .output_embedding = token_embedding,
        .final_rms_weight = final_rms_weight,
        .freqs = undefined,
    };

    return model_weights;
}

pub fn read_tokenizer(base_allocator: std.mem.Allocator, vocab_size: usize, path: []const u8) !*llm.Tokenizer {
    // Read tokenizer
    var token_file = try std.fs.cwd().openFile(path, .{});
    defer token_file.close();

    var token_reader = token_file.reader();

    var max_token_length = try token_reader.readInt(u32, std.builtin.Endian.Little);
    std.log.debug("Max token len: {}", .{max_token_length});

    var tokenizer = try base_allocator.create(llm.Tokenizer);
    tokenizer.* = .{
        .tokens = std.ArrayList([]const u8).init(base_allocator),
        .back_map = std.StringHashMap(llm.Tokenizer.Token).init(base_allocator),
    };

    for (0..vocab_size) |idx| {
        var score: f32 = 0;
        _ = try token_reader.readAll(std.mem.asBytes(&score));
        // const score: f32 = @bitCast(try token_reader.readInt(u32, std.builtin.Endian.Little)); //strange
        const token_len = try token_reader.readInt(u32, std.builtin.Endian.Little);

        var tokens = try base_allocator.alloc(u8, token_len);
        defer base_allocator.free(tokens);
        const read_amt = try token_reader.readAll(tokens);
        if (read_amt != token_len) {
            return error.UnexpectedEof;
        }

        // std.log.info("{} {s}, len: {}", .{ idx, tokens, token_len });

        try tokenizer.tokens.append(try base_allocator.dupe(u8, tokens));

        try tokenizer.back_map.put(try base_allocator.dupe(u8, tokens), .{
            .logit = score,
            .idx = @as(u32, @intCast(idx)),
        });
    }

    return tokenizer;
}
