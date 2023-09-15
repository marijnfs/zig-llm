struct Params {
  dim : u32,
  L_k : u32,
  L_q : u32,
  n_heads: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f32>; //L * dim
@binding(1) @group(0) var<storage, read> V : array<f32>; //L * dim
@binding(2) @group(0) var<storage, read_write> slate : array<f32>; //L * L
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  let k : u32 = GlobalInvocationID.y; //sequence number

  if (l >= params.L_q || k >= params.dim)
    break;

  // apply attention to value
  // first reset output
  output[l * params.dim + k] = 0.0f;

  let head_dim = params.dim / params.n_heads;
  let k_head = k % head_dim;
  output[l * params.dim + k] = 0;
  for (var l_ : u32 = 0u; l_ < L; l_ = l_ + 1u) {
    let value = V[l_ * params.dim + k];
    output[l * params.dim + k] += slate[h * L2 + l * L + l_] * value;
  }
}
