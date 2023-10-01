import sys
import torch
import ujson
import struct
import numpy as np
 
from sklearn.cluster import KMeans

model_path = sys.argv[1]
config_path = sys.argv[2]

output_path = sys.argv[3]

# model_params['dim']
# model_params['n_heads']
# model_params['n_layers']
# model_params['vocab_size']

# tok_embeddings.weight
# norm.weight
# output.weight
# rope.freqs

# layers.31.attention.wq.weight
# layers.31.attention.wk.weight
# layers.31.attention.wv.weight
# layers.31.attention.wo.weight
# layers.31.feed_forward.w1.weight
# layers.31.feed_forward.w2.weight
# layers.31.feed_forward.w3.weight
# layers.31.attention_norm.weight
# layers.31.ffn_norm.weight

def serialize_fp32(file, tensor):
    flat = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(flat)}f', *flat)
    file.write(b)

def serialize_fp16(file, tensor):
    flat = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    b = struct.pack(f'{len(flat)}e', *flat)
    file.write(b)


# Serialize with 8bit lookup table
def serialize_lookup_q8(file, tensor):
    flat = tensor.detach().cpu().view(-1).to(torch.float16).numpy().astype(np.float16)
    flat_2d = flat.reshape(-1, 1)

    print("first val:", flat[0])

    N = len(flat)

    # lookup_table = np.zeros(256, dtype=np.float16) 
    # lookup_values = np.zeros(N, dtype=np.uint8)

    kmean_iterations = 16 #make lower for faster (but worse) results
    min_val = np.min(flat)
    max_val = np.max(flat)

    # init_range = np.linspace(flat.min(), flat.max(), 256, dtype=np.float16).reshape(-1, 1)
    init_range = np.concatenate([np.linspace(min_val, 0, 128, dtype=np.float16), np.linspace(0, max_val, 129, dtype=np.float16)[1:]]).reshape(-1, 1)
    print("fitting")
    kmeans = KMeans(n_clusters=256, max_iter=kmean_iterations, n_init=1, init=init_range).fit(flat_2d)
    print("done")


    lookup_table = k_means.cluster_centers_.astype(np.float16)
    lookup_values = k_means.predict(flat_2d).astype(np.uint8)

    # for i in range(256):
    #     start = i * N // 256
    #     end = (i + 1)  * N // 256

    #     total = 0.0;
    #     for index in range(start, end):
    #         total += flat[sorted_indices[index]]
    #     mean_value = total / (end - start)
    #     lookup_table[i] = mean_value

    # print(lookup_table)
    # for i in range(256):
    #     idx = i * N // 255
    #     if idx >= N:
    #         idx = N - 1
    #     lookup_table[i] = flat[sorted_indices[idx]]

    # min_val = np.min(flat)
    # max_val = np.max(flat)

    # # Create lookup table, with 0 at idx 127
    # # Second linspace offset makes sure we don't have 0 twice
    # lookup_table = np.concatenate([np.linspace(min_val, 0, 128, dtype=np.float16), np.linspace(0, max_val, 129, dtype=np.float16)[1:]])

    # # Now create 256-1 compare table, such that we can simply binary search
    # # These will have the mean values between the values
    # compare_table = (lookup_table[1:] + lookup_table[:-1]) / 2.0
    # lookup_values = np.searchsorted(compare_table, flat).astype(np.uint8)

    # for i in range(256):
    #     idx = i * N // 255
    #     if idx >= N:
    #         idx = N - 1
    #     lookup_table[i] = min_val + (max_val - min_val) * (i / 255)

    # for _ in range(1):
    #     lookups = np.searchsorted(lookup_table, flat)

    #     for (i, lookup) in zip(range(N), lookups):
    #         idx = lookup
    #         if idx == len(lookup_table):
    #             idx = idx - 1

    #         val = flat[i]
    #         if idx > 0:
    #             if abs(lookup_table[idx - 1] - val) < abs(lookup_table[idx] - val):
    #                 idx = idx - 1

    #         lookup_values[i] = idx;

        # lookup_table_sum = np.zeros(256, dtype=np.float32) 
        # lookup_table_count = np.zeros(256, dtype=np.uint32)

        # for i in range(N):
        #     lookup = lookup_values[i]
        #     lookup_table_sum[lookup] += flat[i]
        #     lookup_table_count[lookup] += 1

        # for n in range(len(lookup_table_sum)):
        #     if lookup_table_count[n] == 0: continue
        #     lookup_table[n] = lookup_table_sum[n] / lookup_table_count[n]

    # for i in range(256):
    #     start = i * N // 256
    #     end = (i + 1)  * N // 256

        # total = 0.0;
        # for index in range(start, end):
        #     total += flat[sorted_indices[index]]
        # mean_value = total / (end - start)
        # lookup_table[i] = mean_value

        # for index in range(start, end):
        #     target_index = sorted_indices[index]
        #     lookup_values[target_index] = i

    # Verify differences
    # max_diff = 0
    # for i in range(N):
    #     real_val = flat[i]
    #     lookup_val = lookup_table[lookup_values[i]]
    #     absdiff = abs(real_val - lookup_val)
    #     if absdiff > max_diff:
    #         print(i, real_val, lookup_val, real_val - lookup_val)
    #         max_diff = absdiff

    packed_lookup_table = struct.pack(f'{len(lookup_table)}e', *lookup_table)
    packed_lookup_values = struct.pack(f'{len(lookup_values)}B', *lookup_values)

    file.write(packed_lookup_table)
    file.write(packed_lookup_values)


def save_layers_f16(file, n_layers, weights, weight_string):
    for layer in range(n_layers):
        key = weight_string.format(layer=layer)
        w = weights[key]
        print(F"saving {key}")
        serialize_fp16(file, w)

def save_layers_lookup_q8(file, n_layers, weights, weight_string):
    for layer in range(n_layers):
        key = weight_string.format(layer=layer)
        w = weights[key]
        print(F"saving {key}")
        serialize_lookup_q8(file, w)

def save_header(out_file, model, params, has_rope_freq):
    major_version = 0
    minor_version = 1

    out_file.write(struct.pack('I', 0x7a657865))
    out_file.write(struct.pack('I', major_version))
    out_file.write(struct.pack('I', minor_version))

    hidden_dim = model['layers.0.feed_forward.w1.weight'].shape[0]
    
    n_kv_heads = params['n_heads']
    vocab_size = model['tok_embeddings.weight'].shape[0]


    sliding_window = 0
    if 'sliding_window' in params:
        sliding_window = params['sliding_window']


    max_seq_len = 1024 #made up
    basefreq = 10000.0 #default
    if has_rope_freq:
        basefreq = 0 #freqs come from array


    n_kv_heads = params['n_kv_heads'] if 'n_kv_heads' in params else params['n_heads']
    header = struct.pack('iiiiiiiif', params['dim'], hidden_dim, params['n_layers'], params['n_heads'],
                                    n_kv_heads, vocab_size, max_seq_len, sliding_window, base_freq)
    out_file.write(header)


# Open files

config_file = open(config_path)
model_params = ujson.load(config_file)
print(F"model params: {model_params}")

model_weights = torch.load(model_path)

dim = model_params['dim']
n_heads = model_params['n_heads']
n_layers = model_params['n_layers']

# Now save the weights

output_file = open(output_path, 'bw+')


has_rope_freq = 'rope.freqs' in model_weights
save_header(output_file, model_weights, model_params, has_rope_freq=has_rope_freq)

general_weights = ['tok_embeddings.weight', 'output.weight', 'norm.weight']
if has_rope_freq:
    general_weights.append('rope.freqs')

for general_weight in general_weights:
    weights = model_weights[general_weight]
    serialize_fp16(output_file, weights)

save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wq.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wk.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wv.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wo.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w1.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w2.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w3.weight")
save_layers_f16(output_file, n_layers, model_weights, "layers.{layer}.attention_norm.weight")
save_layers_f16(output_file, n_layers, model_weights, "layers.{layer}.ffn_norm.weight")
