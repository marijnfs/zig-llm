const std = @import("std");
const core = @import("core");
const gpu = core.gpu;
const clap = @import("clap");

const llm = @import("index.zig");
const operators = llm.operators;
const Tensor = llm.Tensor;
const Tokenizer = llm.Tokenizer;

const io = llm.io;

pub const App = @This();

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const workgroup_size = 64;
const buffer_size = 1000;

pub const log_level: std.log.Level = .info;

pub fn init(app: *App) !void {
    const allocator = gpa.allocator();

    try core.init(.{
        .required_limits = gpu.Limits{
            .max_buffer_size = 1 << 30,
            .max_storage_buffer_binding_size = 1 << 30,
            .max_compute_workgroup_size_x = 1 << 10,
            .max_compute_workgroups_per_dimension = 1 << 10,
            .max_compute_invocations_per_workgroup = 1 << 10,
        },
        // .required_features = &[_]gpu.FeatureName{.shader_f16},
    });
    app.* = .{};

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                    Display this help and exit.
        \\--prompt <str>                Prompt string as input to the model.
        \\--model <str>                 Path to model file (.bin format).
        \\--tokenizer <str>             Path to tokenizer (.bin format).
        \\--length <usize>              Sequence length to predict.
        \\
    );

    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
    }) catch |err| {
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };

    const args = res.args;

    if (args.help != 0)
        return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});

    const Mode = enum {
        Cached,
        Uncached,
    };

    var mode: Mode = .Cached;
    // var mode: Mode = .Uncached;

    const seed: u64 = 123;
    var prng = std.rand.DefaultPrng.init(seed);
    const random = prng.random();

    if (args.model == null or args.tokenizer == null) {
        std.log.warn("provide model / tokenizer paths", .{});
        return;
    }

    const model_path = args.model.?;
    const tokenizer_path = args.tokenizer.?;

    const model_weights = try io.read_model_weights(allocator, model_path);
    const config = model_weights.config;

    const vocab_size = @as(usize, @intCast(config.vocab_size));

    const tokenizer = try io.read_tokenizer(allocator, vocab_size, tokenizer_path);

    const str = args.prompt orelse "";
    const tokens = try llm.tokenize(allocator, str, tokenizer);

    std.log.info("Tokenized:", .{});
    const writer = std.io.getStdOut().writer();

    for (tokens) |token| {
        try writer.print("{s}-", .{tokenizer.tokens.items[token]});
    }
    _ = try writer.writeAll("\n");

    // Because of different operations needing different parameters, I lazily copied the operations here to allow for different parameters to be used in one command buffer.
    // This is wasteful, since only a little parameter buffer is needed, not a whole pipeline.
    // TODO: improve this
    const tmat_operator = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_1 = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_2 = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_3 = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_4 = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_5 = try operators.TransposeMatOperator.init(allocator);
    const tmat_operator_6 = try operators.TransposeMatOperator.init(allocator);

    const rope_operator = try operators.RopeOperator.init(allocator);
    const rope_operator_1 = try operators.RopeOperator.init(allocator);

    const elmul_operator = try operators.ElMulOperator.init(allocator);

    const scale_operator = try operators.ScaleOperator.init(allocator);
    const scale_operator_1 = try operators.ScaleOperator.init(allocator);

    const add_operator = try operators.AddOperator.init(allocator);
    const add_operator_1 = try operators.AddOperator.init(allocator);

    const attention_operator = try operators.AttentionOperator.init(allocator);

    const rmsnorm_operator = try operators.RMSNormOperator.init(allocator);
    const rmsnorm_operator_1 = try operators.RMSNormOperator.init(allocator);

    const silu_operator = try operators.SILUOperator.init(allocator);

    const embed_operator = try operators.EmbedOperator.init(allocator);

    const argmax_operator = try operators.ArgmaxOperator.init(allocator);

    const lookup_operator = try operators.LookupOperator.init(allocator);
    const lookup_operator_1 = try operators.LookupOperator.init(allocator);
    const lookup_operator_2 = try operators.LookupOperator.init(allocator);
    const lookup_operator_3 = try operators.LookupOperator.init(allocator);
    const lookup_operator_4 = try operators.LookupOperator.init(allocator);
    const lookup_operator_5 = try operators.LookupOperator.init(allocator);
    const lookup_operator_6 = try operators.LookupOperator.init(allocator);

    // const copy_operator = try operators.CopyOperator.init(allocator);
    // const transpose_operator = try operators.TransposeOperator.init(allocator);

    const n_heads = @as(usize, @intCast(config.n_heads));
    const n_kv_heads = @as(usize, @intCast(config.n_kv_heads));
    const dim = @as(usize, @intCast(config.dim));
    const hidden_dim = @as(usize, @intCast(config.hidden_dim));
    const dim_per_head = dim / n_heads;
    const kv_dim = n_kv_heads * dim_per_head;

    // Steps in a layer:
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

    const L: usize = args.length orelse @as(usize, @intCast(config.seq_len));
    const L_cache = if (mode == .Uncached) L else 1;

    _ = random;

    // var random_values = try allocator.alloc(f32, dim * L_cache);
    // defer allocator.free(random_values);

    // for (random_values) |*v| {
    //     v.* = (random.float(f32) * 2 - 1) * 0.25;
    // }

    // var x = try Tensor.init_from_data_f32(allocator, &[_]usize{ dim, L_cache }, .Storage, random_values);

    // L cache is the size of the caches during computation
    //

    var x = try Tensor.init_f32(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var xb = try Tensor.init_f32(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var q = try Tensor.init_f32(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var k = try Tensor.init_f32(allocator, &[_]usize{ kv_dim, L_cache }, .Storage);
    var v = try Tensor.init_f32(allocator, &[_]usize{ kv_dim, L_cache }, .Storage);

    var k_caches = std.ArrayList(*Tensor).init(allocator);
    var v_caches = std.ArrayList(*Tensor).init(allocator);

    if (mode == .Cached) {
        for (model_weights.layers.items) |_| {
            try k_caches.append(try Tensor.init_f32(allocator, &[_]usize{ kv_dim, L }, .Storage));
            try v_caches.append(try Tensor.init_f32(allocator, &[_]usize{ kv_dim, L }, .Storage));
        }
    }

    var attention_out = try Tensor.init_f32(allocator, &[_]usize{ dim, L_cache }, .Storage);
    var out = try Tensor.init_f32(allocator, &[_]usize{ dim, L_cache }, .Storage);

    var w1_slate = try Tensor.init_f32(allocator, &[_]usize{ hidden_dim, L_cache }, .Storage);
    var w3_slate = try Tensor.init_f32(allocator, &[_]usize{ hidden_dim, L_cache }, .Storage);

    var logits = try Tensor.init_f32(allocator, &[_]usize{ vocab_size, L_cache }, .Storage);
    var slate = try Tensor.init_f32(allocator, &[_]usize{ L, L_cache, n_heads }, .Storage);

    var max_index = try Tensor.init_u32(allocator, &[_]usize{L_cache}, .Storage);

    var all_predicted = std.ArrayList(u32).init(allocator);

    var last_predicted_token: u32 = 1;

    // Weight holders to uncompress into
    var query_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, dim }, .Storage);
    var key_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, kv_dim }, .Storage);
    var value_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, kv_dim }, .Storage);

    var output_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, dim }, .Storage);

    var w1_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, hidden_dim }, .Storage);
    var w2_weight = try Tensor.init_f32(allocator, &[_]usize{ hidden_dim, dim }, .Storage);
    var w3_weight = try Tensor.init_f32(allocator, &[_]usize{ dim, hidden_dim }, .Storage);

    _ = try writer.writeAll("Prediction:\n");
    for (0..L) |token_idx| {
        // uncached, take all tokens; cached, take tokens one by one and then the predicted ones.
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

        if (mode == .Cached) {
            if (token_idx < tokens.len) {
                try writer.print("{s}", .{tokenizer.tokens.items[tokens[token_idx]]});
            } else {
                try writer.print("{s}", .{tokenizer.tokens.items[last_predicted_token]});
            }
        }

        var tokens_tensor = try Tensor.init_from_tokens(allocator, embed_tokens);

        const cur_idx = if (mode == .Cached) token_idx else null;

        const command_encoder = core.device.createCommandEncoder(null);

        embed_operator.execute(x, tokens_tensor, model_weights.token_embedding, L, command_encoder);

        for (model_weights.layers.items, 0..) |*layer, layer_idx| {
            const k_cache = if (mode == .Cached) k_caches.items[layer_idx] else k;
            const v_cache = if (mode == .Cached) v_caches.items[layer_idx] else v;

            lookup_operator.execute(query_weight, layer.query_weight, command_encoder);
            lookup_operator_1.execute(key_weight, layer.key_weight, command_encoder);
            lookup_operator_2.execute(value_weight, layer.value_weight, command_encoder);
            lookup_operator_3.execute(output_weight, layer.output_weight, command_encoder);
            lookup_operator_4.execute(w1_weight, layer.w1, command_encoder);
            lookup_operator_5.execute(w2_weight, layer.w2, command_encoder);
            lookup_operator_6.execute(w3_weight, layer.w3, command_encoder);

            x.copy_to(xb, command_encoder);

            rmsnorm_operator.execute(xb, command_encoder);
            scale_operator.execute(xb, layer.rms_attention, command_encoder);

            tmat_operator.execute(q, query_weight, xb, null, command_encoder);
            tmat_operator_2.execute(v_cache, value_weight, xb, cur_idx, command_encoder);
            tmat_operator_1.execute(k_cache, key_weight, xb, cur_idx, command_encoder);

            rope_operator.execute(k_cache, model_weights.freqs, n_kv_heads, dim_per_head, cur_idx, cur_idx, command_encoder);
            rope_operator_1.execute(q, model_weights.freqs, n_heads, dim_per_head, cur_idx, 0, command_encoder);

            const L_k = token_idx + 1;
            attention_operator.execute(attention_out, q, k_cache, v_cache, slate, n_heads, n_kv_heads, L_k, command_encoder);

            tmat_operator_3.execute(out, output_weight, attention_out, null, command_encoder);

            // Add activation into x
            add_operator.execute(x, out, command_encoder);

            x.copy_to(xb, command_encoder);

            rmsnorm_operator_1.execute(xb, command_encoder);
            scale_operator_1.execute(xb, layer.rms_ffn, command_encoder);

            tmat_operator_4.execute(w1_slate, w1_weight, xb, null, command_encoder);
            tmat_operator_5.execute(w3_slate, w3_weight, xb, null, command_encoder);

            silu_operator.execute(w1_slate, command_encoder);

            elmul_operator.execute(w1_slate, w3_slate, command_encoder);

            tmat_operator_6.execute(out, w2_weight, w1_slate, null, command_encoder);

            add_operator_1.execute(x, out, command_encoder);
        }

        {
            rmsnorm_operator.execute(x, command_encoder);
            scale_operator.execute(x, model_weights.final_rms_weight, command_encoder);

            tmat_operator.execute(logits, model_weights.output_embedding, x, null, command_encoder);

            argmax_operator.execute(max_index, logits, command_encoder);

            { //submit commands
                var command = command_encoder.finish(null);
                defer command.release();

                core.queue.submit(&[_]*gpu.CommandBuffer{command});
            }
        }

        const predicted_tokens = try max_index.read_data_tokens(allocator);
        //std.log.info("predicted_tokens: {any}", .{predicted_tokens});
        last_predicted_token = predicted_tokens[predicted_tokens.len - 1];
        try all_predicted.append(last_predicted_token);
    }

    for (all_predicted.items) |token| {
        std.log.debug("token: '{s}' {}", .{ tokenizer.tokens.items[token], token });
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
