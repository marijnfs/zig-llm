const std = @import("std");

pub const Tokenizer = struct {
    pub const Token = struct {
        logit: f32,
        idx: u32,
    };

    tokens: std.ArrayList([]const u8),
    back_map: std.StringHashMap(Token),
};
