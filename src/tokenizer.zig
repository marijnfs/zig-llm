const std = @import("std");

pub const Tokenizer = struct {
    pub const Token = struct {
        logit: f32,
        idx: u32,
    };

    tokens: std.ArrayList([]const u8),
    back_map: std.StringHashMap(Token),
};

pub fn tokenize(allocator_: std.mem.Allocator, str: []const u8, tokenizer: *Tokenizer) ![]u32 {
    var arena = std.heap.ArenaAllocator.init(allocator_);
    defer arena.deinit();
    const allocator = arena.allocator();

    var tokens = std.ArrayList(u32).init(allocator);

    // encode bytes
    for (str) |byte| {
        const token_str = try std.fmt.allocPrint(allocator, "{c}", .{byte});
        defer allocator.free(token_str);

        if (tokenizer.back_map.get(token_str)) |match| {
            try tokens.append(match.idx);
        } else {
            return error.TokenizerFail;
        }
    }

    // compress
    while (true) {
        var i: usize = 0;
        var best_idx: usize = 0;
        var best_token: ?Tokenizer.Token = null;

        while (i + 1 < tokens.items.len) : (i += 1) {
            const token_a = tokens.items[i];
            const token_b = tokens.items[i + 1];
            const token_str = try std.fmt.allocPrint(allocator, "{s}{s}", .{
                tokenizer.tokens.items[token_a],
                tokenizer.tokens.items[token_b],
            });

            defer allocator.free(token_str); //even though we have arena, we still free. This clears mem but arena will leave space allocated for next token
            if (tokenizer.back_map.get(token_str)) |match| {
                if (best_token) |cur_best| {
                    if (match.logit > cur_best.logit) {
                        best_idx = i;
                        best_token = match;
                    }
                } else {
                    best_idx = i;
                    best_token = match;
                }
            }
        }

        if (best_token) |token| {
            //replace the consecutive tokens with the new one
            _ = tokens.orderedRemove(best_idx);
            _ = tokens.orderedRemove(best_idx);
            try tokens.insert(best_idx, token.idx);
        } else {
            break; //no match, we are done
        }
    }

    return try allocator_.dupe(u32, tokens.items);
}
