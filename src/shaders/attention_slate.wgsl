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
  L : u32,
  n_heads: u32,
};

@binding(0) @group(0) var<storage, read_write> slate : array<f32>; //L * L * n_heads
@binding(1) @group(0) var<storage, read> K : array<f32>; //L * dim
@binding(2) @group(0) var<storage, read> Q : array<f32>; //L * dim
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l_q : u32 = GlobalInvocationID.x; //sequence number
  
  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l_q >= params.L)
  {
    return;
  }

  let dim_per_head = params.dim / params.n_heads;

  for (var l_k : u32 = 0u; l_k < params.L; l_k = l_k + 1u) {
    var k : u32 = 0u;
    for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
      var dot = 0.0f;
      for (var k_head : u32 = 0u; k_head < dim_per_head; k_head = k_head + 1u) {
        dot += Q[l_q * params.dim + k] * K[l_k * params.dim + k];
        k = k + 1u;
      }
      slate[h * (params.L * params.L) + l_q * (params.L) + l_k] = dot;
    }
  }
}
