
struct Params {
  X: u32,
};

@binding(0) @group(0) var<storage, read_write> left : array<f32>;
@binding(1) @group(0) var<storage, read> right : array<f32>;
@binding(2) @group(0) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let x : u32 = GlobalInvocationID.x;

  if (x >= params.X)
  {
    return;
  }

  left[x] = right[x];
}
