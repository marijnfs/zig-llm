// We will normalize the vector in place

struct Params {
  dim : u32,
  L : u32,
};

@binding(0) @group(0) var<storage, read_write> M : array<f32>; //L * dim
@binding(1) @group(0) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  
  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l >= params.L)
  {
    return;
  }

  var sum = 0.0f;
  var c = 0.0f;
  let m_offset = l * params.dim;
  for (var k : u32 = 0u; k < params.dim; k = k + 1u) {
    let value =  M[m_offset + k];

    /*
    let mult: f32 = value * value;
    let y: f32 = mult - c;
    let tmp: f32 = sum + y;
    c = (tmp - sum) - y;
    sum = tmp;
    */
    sum += value * value;
  }

  sum /= f32(params.dim);
  sum += 1.0e-5; //for stability
  let factor = 1.0f / sqrt(sum);
  for (var k : u32 = 0u; k < params.dim; k = k + 1u) {
    M[m_offset + k] *= factor;
  }
}
