
struct Params {
  dim: u32,
  L: u32,
  n_tokens: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f32>;
@binding(1) @group(0) var<storage, read> tokens : array<u32>;
@binding(2) @group(0) var<storage, read> embeddings : array<f32>;
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let d : u32 = GlobalInvocationID.x;
  let l : u32 = GlobalInvocationID.y;
  let dim = params.dim;
  let L = params.L;

  if (l >= L || d >= dim)
  {
    return;
  }

  if (l < params.n_tokens)
  {
    output[l * dim + d] = embeddings[tokens[l] * dim + d];
  }
  else
  {
    output[l * dim + d] = 1.0;
  }
}
