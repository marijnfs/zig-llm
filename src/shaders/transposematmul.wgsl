// We will calculate a matrix multiplication
// Output = Left * Right
// O = M * N, Left = M * K, Right = K * N
// We assume column major storage



struct Params {
  M : u32,
  K : u32,
  N : u32,
  output_offset: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f32>;
@binding(1) @group(0) var<storage, read> left : array<f32>;
@binding(2) @group(0) var<storage, read> right : array<f32>;
@binding(3) @group(0) var<uniform> params : Params;

const WORKGROUP_SIZE_X = 16;
const WORKGROUP_SIZE_Y = 16;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let m : u32 = GlobalInvocationID.x;
  let n : u32 = GlobalInvocationID.y;

  let M = params.M;
  let N = params.N;
  let K: u32 = params.K;

  //var slate : array<f32, 4096>;

  if (m >= M || n >= N)
  {
    return;
  }

  let M_start = m * K;
  let N_start = n * K;

  //var accum: f32 = 0.0;
  
  /*
  for (var k : u32 = 0u; k < K; k += 1) {
    slate[k] = left[M_start + k] * right[N_start + k];
  }

  var stride: u32 = 1;
  for (; stride * 2 < K; stride *= 2) {
    for (var k : u32 = 0u; k + stride < K; k += stride * 2) {
      slate[k] = slate[k] + slate[k + stride];
    }
  }

  let accum = slate[0];
  */

  /*
  var accum: f32 = 0;
  var c: f32 = 0;
  for (var k : u32 = 0u; k < K; k += 1) {
    let mult: f32 = left[M_start + k] * right[N_start + k];
    let y: f32 = mult - c;
    let tmp: f32 = accum + y;
    c = (tmp - accum) - y;
    accum = tmp;
    //accum += left[M_start + k];// * right[N_start + k];
    //let bla = right[0];
  }
 */
  var accum: f32 = 0;
  var c: f32 = 0;
  for (var k : u32 = 0u; k < K; k += 1) {
    accum += left[M_start + k] * right[N_start + k];
  }
  
  output[params.output_offset + n * M + m] = accum;
  //output[params.output_offset + n * M + m] = f32(K);
}
