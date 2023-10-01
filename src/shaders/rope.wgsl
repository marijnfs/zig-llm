
struct Params {
  dim : u32,
  L : u32,
  n_heads: u32,
  base_freq: f32,
  l_offset : u32,
  write_l_offset: u32,
};

@binding(0) @group(0) var<storage, read_write> K : array<f32>; //L * dim
//@binding(1) @group(0) var<storage, read_write> freqs : array<f32>; 
@binding(1) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l_base : u32 = GlobalInvocationID.x; //sequence number
  
  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l_base >= params.L)
  {
    return;
  }

  let l = l_base + params.l_offset;
  let write_l = l_base + params.write_l_offset;

  let dim_per_head = params.dim / params.n_heads;

  var k : u32 = 0u;
  for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
    for (var head_k : u32 = 0u; head_k < dim_per_head; head_k = head_k + 2u) {
      //let freq = freqs[head_k / 2];
      let freq = 1.0 / pow(params.base_freq, f32(head_k) / f32(dim_per_head));
      let val = f32(l) * freq;
      let real = cos(val);
      let img = sin(val);

      let kr = K[write_l * params.dim + k];
      let ki = K[write_l * params.dim + k + 1];

      K[write_l * params.dim + k] = real * kr - img * ki;
      K[write_l * params.dim + k + 1] = real * ki + img * kr;

      k = k + 2u;
    }
  }
}
