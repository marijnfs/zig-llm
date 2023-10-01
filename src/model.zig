const std = @import("std");
const llm = @import("index.zig");
const Tensor = llm.Tensor;

pub const ModelConfig = extern struct {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
};

pub const ExtraConfig = extern struct {
    sliding_window: i32,
    base_freq: f32,
};

pub const LayerWeights = struct {
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

pub const ModelWeights = struct {
    config: ModelConfig,
    extra_config: ExtraConfig,

    layers: std.ArrayList(LayerWeights),

    token_embedding: *Tensor,
    output_embedding: *Tensor,
    final_rms_weight: *Tensor,
    freqs: ?*Tensor,

    // final_class_weights: ?*Tensor,

    // - rms final weight = dim
    // - freq real = seq_len * head_size / 2
    // - freq img = seq_len * head_size / 2
    //  - final class weights (if not shared with embedding) vocab * dim
};
