const std = @import("std");
const core = @import("core");
const gpu = core.gpu;

const llm = @import("index.zig");
const operators = llm.operators;
const Tensor = llm.Tensor;
const Tokenizer = llm.Tokenizer;

const io = llm.io;

pub const App = @This();

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const workgroup_size = 64;
const buffer_size = 1000;

pub fn init(app: *App) !void {
    const allocator = gpa.allocator();

    try core.init(.{});
    app.* = .{};

    const Mode = enum {
        Cached,
        Uncached,
    };

    var mode: Mode = .Cached; //.Uncached;

    const seed: u64 = 123;
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    const tokenizer_path = "/home/marijnfs/Downloads/tokenizer.bin";
    const model_path = "/home/marijnfs/Downloads/stories15M.bin";

    const model_weights = try io.read_model_weights(allocator, model_path);
    const config = model_weights.config;

    const vocab_size = @as(usize, @intCast(config.vocab_size));

    const tokenizer = try io.read_tokenizer(allocator, vocab_size, tokenizer_path);

    const str = "Never gonna give you";
    const tokens = try llm.tokenize(allocator, str, tokenizer);
    // _ = tokens;

    std.log.info("Tokenized:", .{});
    for (tokens) |token| {
        std.log.info("token: '{s}' {}", .{ tokenizer.tokens.items[token], token });
    }

    // const mat_operator = try MatOperator.init(allocator);

    const tmat_operator = try operators.TransposeMatOperator.init(allocator);

    const rope_operator = try operators.RopeOperator.init(allocator);

    const elmul_operator = try operators.ElMulOperator.init(allocator);

    const scale_operator = try operators.ScaleOperator.init(allocator);

    const add_operator = try operators.AddOperator.init(allocator);

    const attention_operator = try operators.AttentionOperator.init(allocator);

    const rmsnorm_operator = try operators.RMSNormOperator.init(allocator);

    const silu_operator = try operators.SILUOperator.init(allocator);

    const embed_operator = try operators.EmbedOperator.init(allocator);

    const argmax_operator = try operators.ArgmaxOperator.init(allocator);

    const transpose_operator = try operators.TransposeOperator.init(allocator);

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

    var L: usize = @as(usize, @intCast(config.seq_len));
    L = 16;

    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));

    _ = random;
    // var random_values = try allocator.alloc(f32, L * dim);
    // defer allocator.free(random_values);

    // for (random_values) |*v| {
    //     v.* = (random.float(f32) * 2 - 1) * 0.25;
    // }

    var embedding_transposed = try Tensor.init(allocator, &[_]usize{ vocab_size, dim }, .Storage);
    transpose_operator.execute(embedding_transposed, model_weights.token_embedding);

    // L cache is the size of the caches during computation
    const L_cache = if (mode == .Uncached) L else 1;

    var x = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var x_copy = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var q = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var k = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var v = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var k_caches = std.ArrayList(*Tensor).init(allocator);
    var v_caches = std.ArrayList(*Tensor).init(allocator);

    if (mode == .Cached) {
        for (model_weights.layers.items) |_| {
            try k_caches.append(try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage));
            try v_caches.append(try Tensor.init(allocator, &[_]usize{ dim, L }, .Storage));
        }
    }

    var attention_out = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var out = try Tensor.init(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var w1_slate = try Tensor.init(allocator, &[_]usize{ hidden_dim, L_cache }, .Storage);
    var w3_slate = try Tensor.init(allocator, &[_]usize{ hidden_dim, L_cache }, .Storage);

    var logits = try Tensor.init(allocator, &[_]usize{ vocab_size, L_cache }, .Storage);
    var slate = try Tensor.init(allocator, &[_]usize{ L, L_cache, n_heads }, .Storage);

    var max_index = try Tensor.init_u32(allocator, &[_]usize{L_cache}, .Storage);

    var all_predicted = std.ArrayList(u32).init(allocator);

    // {
    //     mat_operator.execute(embedding_transposed, x, logits);

    //     argmax_operator.execute(max_index, logits);
    //     max_index.read_data_tokens(tokenizer);
    //     if (true)
    //         return;
    // }

    // _ = embed_operator;

    var last_predicted_token: u32 = 1;

    for (0..L) |token_idx| {

        // uncached, take all tokens; uncached, take tokens one by one and then the predicted ones.
        const embed_tokens = b: {
            if (mode == .Uncached) {
                break :b tokens;
            } else {
                if (token_idx < tokens.len) {
                    break :b tokens[token_idx..][0..1];
                } else {
                    break :b &[_]u32{last_predicted_token};
                }
            }
        };

        var tokens_tensor = try Tensor.init_from_tokens(allocator, embed_tokens);
        embed_operator.execute(x, tokens_tensor, model_weights.token_embedding, L);

        const cur_idx = if (mode == .Cached) token_idx else null;

        for (model_weights.layers.items, 0..) |*layer, layer_idx| {
            const k_cache = if (mode == .Cached) k_caches.items[layer_idx] else k;
            const v_cache = if (mode == .Cached) v_caches.items[layer_idx] else v;

            x.copy_to(x_copy);

            rmsnorm_operator.execute(x);
            scale_operator.execute(x, layer.rms_attention);

            tmat_operator.execute(layer.query_weight, x, q, null);

            tmat_operator.execute(layer.key_weight, x, k_cache, cur_idx);
            tmat_operator.execute(layer.value_weight, x, v_cache, cur_idx);

            rope_operator.execute(k_cache, n_heads, cur_idx);
            rope_operator.execute(q, n_heads, cur_idx);

            const L_k = token_idx + 1;
            attention_operator.execute(q, k_cache, v_cache, slate, attention_out, n_heads, L_k);

            tmat_operator.execute(layer.output_weight, attention_out, out, null);

            add_operator.execute(out, x_copy);
            out.copy_to(x_copy);

            rmsnorm_operator.execute(out);
            scale_operator.execute(out, layer.rms_ffn);

            tmat_operator.execute(layer.w1, out, w1_slate, null);
            tmat_operator.execute(layer.w3, out, w3_slate, null);
            silu_operator.execute(w1_slate);

            elmul_operator.execute(w1_slate, w3_slate);
            tmat_operator.execute(layer.w2, w1_slate, x, null);
            add_operator.execute(x, x_copy);
        }
        rmsnorm_operator.execute(x);
        scale_operator.execute(x, model_weights.final_rms_weight);

        // _ = logits;
        // _ = max_index;
        // _ = argmax_operator;
        const final_weights = model_weights.final_class_weights orelse model_weights.token_embedding;
        tmat_operator.execute(final_weights, x, logits, null);

        argmax_operator.execute(max_index, logits);

        //TODO: set predicted token
        const predicted_tokens = try max_index.read_data_tokens(tokenizer, allocator);
        std.log.info("predicted_tokens: {any}", .{predicted_tokens});
        last_predicted_token = predicted_tokens[predicted_tokens.len - 1];
        try all_predicted.append(last_predicted_token);
    }

    for (all_predicted.items) |token| {
        std.log.info("token: '{s}' {}", .{ tokenizer.tokens.items[token], token });
    }
}

pub fn deinit(app: *App) void {
    _ = app;
    // defer _ = gpa.deinit();
    core.deinit();
}

pub fn update(_: *App) !bool {
    return true;
}
