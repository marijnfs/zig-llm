
struct Params {
  dim: u32,
  L: u32,
};

@binding(0) @group(0) var<storage, read_write> left : array<f32>;
@binding(1) @group(0) var<storage, read> right : array<f32>;
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

  left[l * dim + d] *= right[l * dim + d];
}
