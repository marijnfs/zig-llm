import sys
import torch
import ujson
import numpy as np

model_path = sys.argv[1]
config_path = sys.argv[2]

output_path = sys.argv[3]

# model_config['dim']
# model_config['n_heads']
# model_config['n_layers']
# model_config['vocab_size']

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
    """ writes one fp32 tensor to file that is open in wb mode """
    flat = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(flat)}f', *flat)
    file.write(b)

def serialize_fp16(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    flat = tensor.detach().cpu().view(-1).to(torch.float16).numpy()
    b = struct.pack(f'{len(flat)}e', *flat)
    file.write(b)

def serialize_int8(file, tensor):
    flat = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(flat)}b', *flat)
    file.write(b)

# Serialize with 8bit lookup table
def serialize_lookup_q8(file, tensor):

    flat = tensor.detach().cpu().view(-1).to(torch.float32).numpy().astype(np.float16)

    N = len(flat)
    # Enumerate the array to create pairs of (index, value)
    indexed = list(enumerate(flat))

    # Sort the indexed array based on values

    sorted_values = sorted(indexed, key=lambda x: x[1])

    # We sorted the values, now we'll mentally divide the sequence in 256 sequences and take the median value (seems the most fair).
    # this means taking indices (x * 2 + 1) * len(values) / 512
    mid_indices = [((x * 2 + 1) * N) // 512 for x in range(256)]
    lookup_table = [sorted_values[mid][1] for mid in mid_indices]
    mid_indices.insert(0,0)

    lookup_values = np.zeros(N, dtype=np.uint8)
    for i in range(256):
        for sorted_value in sorted_values[mid_indices[i]:mid_indices[i + 1]]:
            index = sorted_value[0]
            lookup_values[index] = i


    packed_lookup_table = struct.pack(f'{len(lookup_table)}e', *lookup_table)
    packed_lookup_values = struct.pack(f'{len(lookup_values)}b', *lookup_values)

    file.write(packed_lookup_table)
    file.write(packed_lookup_values)


def save_layers_f16(n_layers, weights, weight_string, file):
    for layer in range(n_layers):
        key = weight_string.format(layer=layer)
        w = weights[key]
        serialize_fp16(file, w)

def save_layers_lookup_q8(n_layers, weights, weight_string, file):
    for layer in range(n_layers):
        key = weight_string.format(layer=layer)
        w = weights[key]
        serialize_lookup_q8(file, w)

# Open files

config_file = open(config_path)
model_config = ujson.load(config_file)

model_weights = torch.load(model_path)

dim = model_config['dim']
n_heads = model_config['n_heads']
n_layers = model_config['n_layers']

# Now save the weights

output_file = open(output_path, 'bw+')

for general_weights in ['tok_embeddings.weight', 'output.weight', 'norm.weight', 'rope.freqs']:
    weights = model_weights[general_weights]
    serialize_fp16(output_file, weights)


save_layers_lookup_q8(n_layers, model, "layers.{layer}.attention.wk.weight", output_file)
save_layers_lookup_q8(n_layers, model, "layers.{layer}.attention.wv.weight", output_file)
save_layers_lookup_q8(n_layers, model, "layers.{layer}.attention.wo.weight", output_file)
save_layers_lookup_q8(n_layers, model, "layers.{layer}.feed_forward.w1.weight", output_file)
save_layers_lookup_q8(n_layers, model, "layers.{layer}.feed_forward.w2.weight", output_file)
save_layers_lookup_q8(n_layers, model, "layers.{layer}.feed_forward.w3.weight", output_file)
save_layers_f16(n_layers, model, "layers.{layer}.attention_norm.weight", output_file)
save_layers_f16(n_layers, model, "layers.{layer}.ffn_norm.weight", output_file)
