// We will calculate a matrix multiplication
// Output = Left * Right
// O = M * N, Left = M * K, Right = K * N
// We assume column major storage

struct Params {
  dim: u32,
  L: u32,
};

@binding(0) @group(0) var<storage, read_write> x : array<f32>;
@binding(2) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x;
  let d : u32 = GlobalInvocationID.y;
  let dim = params.dim;
  let L = params.L;

  if (l >= L || d >= dim)
  {
    return;
  }

  let value = x[l * dim + d];
  x[l * dim + d] = value * (1.0f / (1.0f / exp(-value)));
}
