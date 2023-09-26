import sys
import torch
import ujson
import struct
import numpy as np

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

    sorted_indices = np.argsort(flat)
    N = len(flat)

    # We sorted the values, now we'll mentally divide the sequence in 256 sequences and take the median value (seems the most fair).
    # this means taking indices (x * 2 + 1) * len(values) / 512
    mid_indices = [((x * 2 + 1) * N) // 512 for x in range(256)]
    lookup_table = np.array([flat[i] for i in mid_indices], dtype=np.float16)

    mid_indices.insert(0,0)

    lookup_values = np.zeros(N, dtype=np.uint8)
    for i, start, end in zip(range(256), mid_indices[0:256], mid_indices[1:256+1]):
        for index in range(start, end):
            lookup_values[index] = i

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

def save_header(out_file, model, params):
    major_version = 0
    minor_version = 0

    out_file.write(struct.pack('I', 0x7a657865))
    out_file.write(struct.pack('I', major_version))
    out_file.write(struct.pack('I', minor_version))

    hidden_dim = model['layers.0.feed_forward.w1.weight'].shape[1]
    
    print(params)
    n_kv_heads = params['n_heads']
    vocab_size = model['tok_embeddings.weight'].shape[0]

    print("hidden dim: ", hidden_dim)
    max_seq_len = 1024 #made up

    n_kv_heads = params['n_kv_heads'] if 'n_kv_heads' in params else params['n_heads']
    header = struct.pack('iiiiiii', params['dim'], hidden_dim, params['n_layers'], params['n_heads'],
                                    n_kv_heads, vocab_size, max_seq_len)


# Open files

config_file = open(config_path)
model_params = ujson.load(config_file)

model_weights = torch.load(model_path)

dim = model_params['dim']
n_heads = model_params['n_heads']
n_layers = model_params['n_layers']

# Now save the weights

output_file = open(output_path, 'bw+')

save_header(output_file, model_weights, model_params)

for general_weights in ['tok_embeddings.weight', 'output.weight', 'norm.weight', 'rope.freqs']:
    weights = model_weights[general_weights]
    serialize_fp16(output_file, weights)

save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wk.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wv.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.attention.wo.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w1.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w2.weight")
save_layers_lookup_q8(output_file, n_layers, model_weights, "layers.{layer}.feed_forward.w3.weight")
save_layers_f16(output_file, n_layers, model_weights, "layers.{layer}.attention_norm.weight")
save_layers_f16(output_file, n_layers, model_weights, "layers.{layer}.ffn_norm.weight")
