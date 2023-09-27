
struct Params {
  dim0: u32,
  dim1: u32,
};

@binding(0) @group(0) var<storage, read_write> left : array<f32>;
@binding(1) @group(0) var<storage, read> right : array<f32>;
@binding(2) @group(0) var<uniform> params : Params;

const WORKGROUP_SIZE_X: u32 = 1;
const WORKGROUP_SIZE_Y: u32 = 1;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let idx0 : u32 = GlobalInvocationID.x;
  let idx1 : u32 = GlobalInvocationID.y;


  if (idx0 >= params.dim0 || idx1 >= params.dim1)
  {
    return;
  }

  left[idx0 * params.dim1 + idx1] += right[idx1 * params.dim0 + idx0];
}
