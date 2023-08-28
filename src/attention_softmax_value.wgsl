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

@binding(0) @group(0) var<storage, read_write> output : array<f32>; //L * dim
@binding(1) @group(0) var<storage, read> V : array<f32>; //L * dim
@binding(2) @group(0) var<storage, read_write> slate : array<f32>; //L * L
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  
  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l >= params.L)
  {
    return;
  }

  let dim_per_head = params.dim / params.n_heads;

  for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
    //find max
    var max = 0.0f;
    for (var l_ : u32 = 0u; l_ < params.L; l_ = l_ + 1u) {
      let value = slate[h * params.L * params.L + l * params.L + l_];
      if (value > max)
      {
        max = value;
      }
    }

    // calculate exponential
    var sum = 0.0f;
    for (var l_ : u32 = 0u; l_ < params.L; l_ = l_ + 1u) {
      let value = slate[h * params.L * params.L + l * params.L + l_];
      let exp_value = exp(value - max);
      sum += exp_value;
      slate[h * params.L * params.L + l * params.L + l_] = exp_value;
    }

    // normalize
    for (var l_ : u32 = 0u; l_ < params.L; l_ = l_ + 1u) {
      let value = slate[h * params.L * params.L + l * params.L + l_];
      slate[h * params.L * params.L + l * params.L + l_] = value / sum;
    }

    // apply attention to value
    // first reset output
    for (var k: u32 = 0u; k < params.dim; k = k + 1u) {
      output[l * params.dim + k] = 0.0f;
    }

    // add the weighted values
    for (var l_ : u32 = 0u; l_ < params.L; l_ = l_ + 1u) {
      var k : u32 = 0u;
      for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
        var dot = 0.0f;
        for (var k_ : u32 = 0u; k < dim_per_head; k_ = k_ + 1u) {
          output[l * params.dim + k] += slate[h * params.L * params.L + l * params.L + l_] * V[l_ * params.dim + k];
          k = k + 1u;
        }
      }
    }
  }
}
