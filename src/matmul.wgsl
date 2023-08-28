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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  var m : u32 = GlobalInvocationID.x;
  var n : u32 = GlobalInvocationID.y;
  var M = params.M;
  var K = params.K;
  var N = params.N;

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
