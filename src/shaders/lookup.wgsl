
struct Params {
  N: u32,
};

@binding(0) @group(0) var<storage, read_write> output : array<f32>;
@binding(1) @group(0) var<storage, read_write> values : array<u32>; //Underlying array is u8, but WGSL doesn't have that
@binding(2) @group(0) var<storage, read> lookup_table : array<f32>;
@binding(3) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1024)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x;

  if (l >= params.N)
  {
    return;
  }

  let int_index = l / 4;
  let byte_index = l % 4;
  let value = values[int_index];



  var byte_value: u32 = 0;
  switch (value) {
    case 0: {
      byte_value = (value & 0xFFu);
    }
    case 1: {
      byte_value = ((value >> 8u) & 0xFFu);
    }
    case 2: {
      byte_value = ((value >> 16u) & 0xFFu);
    }
    case 3: {
      byte_value = ((value >> 24u) & 0xFFu);
    }
    default: {}
  }

  output[l] = lookup_table[byte_value];
}
