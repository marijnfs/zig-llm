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
};

@binding(0) @group(0) var<storage, read_write> slate : array<f32>; //L * L
@binding(1) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  
  let L = params.L_k;
  let L2 = params.L_k * params.L_q;

  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l >= params.L_q)
  {
    return;
  }

  let dim_per_head = params.dim / params.n_heads;

  for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
    //find max
    var max_value = slate[h * L2 + l * L];
    for (var l_ : u32 = 0u; l_ < params.L_k; l_ = l_ + 1u) {
      let value = slate[h * L2 + l * L + l_];
      if (value > max_value)
      {
        max_value = value;
      }
    }

    // calculate exponential
    var sum = 0.0f;
    for (var l_ : u32 = 0u; l_ < L; l_ = l_ + 1u) {
      let value = slate[h * L2 + l * L + l_];
      let exp_value = exp(value - max_value);
      sum += exp_value;
      slate[h * L2 + l * L + l_] = exp_value;
    }

    // normalize
    for (var l_ : u32 = 0u; l_ < L; l_ = l_ + 1u) {
      let value = slate[h * L2 + l * L + l_];
      slate[h * L2 + l * L + l_] = value / sum;
    }
  }
  
}
