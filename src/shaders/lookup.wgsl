
struct Params {
  N: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f16>;
@binding(1) @group(0) var<storage, read_write> values : array<u8>;
@binding(2) @group(0) var<storage, read> lookup_table : array<f16>;
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x;

  if (l >= params.N)
  {
    return;
  }

  output[l] = lookup_table[values[l]];
}
