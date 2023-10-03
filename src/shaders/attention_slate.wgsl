// We will calculate the attention in a transformer
// dim is the dimension
// L is the sequence length
// n_heads is the number of heads
// We assume the K/V/Q matrices are already calculated in a previous matmul step
// These matrices are dim * L and the dim is divided over the number of heads.
// The slate needs to be a L * L * n_heads slate (could be big!) that represents the attention calculation
//
// The dispatch is divided over the sequence length L, and search invocation will perform L comparisons (attention is O(L^2))

struct Params {
  dim : u32,
  L_k : u32,
  L_q : u32,
  n_heads: u32,
  n_kv_heads : u32,
  key_window: u32,
};

@binding(0) @group(0) var<storage, read_write> slate : array<f32>; //L * L * n_heads
@binding(1) @group(0) var<storage, read> K : array<f32>; //L * dim
@binding(2) @group(0) var<storage, read> Q : array<f32>; //L * dim
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let h : u32 = GlobalInvocationID.x; //Head index
  let l_k : u32 = GlobalInvocationID.y; //key sequence number
  let l_q : u32 = GlobalInvocationID.z; //query sequence number, always 0 in case of cached

  if (l_q >= params.L_q || l_k >= params.key_window || h > params.n_heads)
  {
    return;
  }

  let dim_per_head = params.dim / params.n_heads;

  let k_dim = dim_per_head * params.n_kv_heads;
  let head_per_q = params.n_heads / params.n_kv_heads;


  var q_offset : u32 = h * dim_per_head;
  var k_offset : u32 = (h / head_per_q) * dim_per_head;

  var dot = 0.0f;
  for (var k : u32 = 0u; k < dim_per_head; k = k + 1u) {
    dot += Q[l_q * params.dim + q_offset + k] * K[l_k * k_dim + k_offset + k];
  }
  slate[h * (params.L_q * params.L_k) + l_q * (params.L_k) + l_k] = dot / sqrt(f32(dim_per_head));
}
