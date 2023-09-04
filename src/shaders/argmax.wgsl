// We will calculate a matrix multiplication
// Output = Left * Right
// O = M * N, Left = M * K, Right = K * N
// We assume column major storage

struct Params {
  dim: u32,
  L: u32,
};

@binding(0) @group(0) var<storage, read> values : array<f32>;
@binding(1) @group(0) var<storage, read_write> max_index : array<u32>;
@binding(2) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x;
  let dim = params.dim;
  let L = params.L;

  if (l >= L)
  {
    return;
  }

  var idx: u32 = 0;
  var value: f32 = values[l * dim];
  for (var d : u32 = 0u; d < dim; d = d + 1u) {
    let cmp_value = values[l * dim + d];
    if (cmp_value > value) {
      idx = d;
      value = cmp_value;
    }
  }

  max_index[l] = idx;
}
