// We will calculate a matrix multiplication
// Output = Left * Right
// O = M * N, Left = M * K, Right = K * N
// We assume column major storage

struct Params {
  M : u32,
  K : u32,
  N : u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f32>;
@binding(1) @group(0) var<storage, read> left : array<f32>;
@binding(2) @group(0) var<storage, read> right : array<f32>;
@binding(3) @group(0) var<uniform> params : Params;

const WORKGROUP_SIZE_X: u32 = 1;
const WORKGROUP_SIZE_Y: u32 = 1;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let m : u32 = GlobalInvocationID.x;
  let n : u32 = GlobalInvocationID.y;
  let M = params.M;
  let K = params.K;
  let N = params.N;

  if (m >= M || n >= N)
  {
    return;
  }

  var accum = 0.0f;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    accum += left[k * M + m] * right[n * K + k];
  }
  output[n * M + m] = accum;
}
