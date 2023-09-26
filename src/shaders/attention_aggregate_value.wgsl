struct Params {
  dim : u32,
  L_k : u32,
  L_q : u32,
  n_heads: u32,
  n_kv_heads: u32,
  K_max: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f16>; //L * dim
@binding(1) @group(0) var<storage, read> V : array<f16>; //L * dim
@binding(2) @group(0) var<storage, read_write> slate : array<f16>; //L * L
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1, 32)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  let k : u32 = GlobalInvocationID.y; //sequence number

  if (l >= params.L_q || k >= params.dim) {
    return;
  }

  // apply attention to value
  // first reset output
  output[l * params.dim + k] = 0.0f;

  let head_dim = params.dim / params.n_heads;
  let q_per_v = params.n_heads / params.n_kv_heads;

  let h = k / head_dim;
  let h_v = h / q_per_v;

  let k_head = h % head_dim; //index in the head, needed to compute proper value id taking into account n_kv_heads

  let k_v = h_v * head_dim + k_head;

  let L2 = params.L_k * params.L_q;

  output[l * params.dim + k] = 0;
  for (var l_ : u32 = 0u; l_ < params.K_max; l_ = l_ + 1u) {
    let value = V[l_ * params.dim + k];
    output[l * params.dim + k] += slate[h * L2 + l * params.L_k + l_] * value;
  }
}
