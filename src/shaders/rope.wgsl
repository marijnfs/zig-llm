
struct Params {
  dim : u32,
  L : u32,
  n_heads: u32,
};

@binding(0) @group(0) var<storage, read_write> K : array<f32>; //L * dim
@binding(1) @group(0) var<storage, read_write> Q : array<f32>; //L * dim
@binding(2) @group(0) var<uniform> params : Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
  let l : u32 = GlobalInvocationID.x; //sequence number
  
  // invocations often need to be diadic or power of some number, so we need to explicitly check this of lengths that are off
  if (l >= params.L)
  {
    return;
  }

  let dim_per_head = params.dim / params.n_heads;

  var k : u32 = 0u;
  for (var h: u32 = 0u; h < params.n_heads; h = h + 1u) {
    for (var head_k : u32 = 0u; head_k < dim_per_head; head_k = head_k + 2u) {
      let freq = 1.0 / pow(10000.0f, f32(head_k) / f32(dim_per_head));
      let val = f32(l) * freq;
      let real = cos(val);
      let img = sin(val);

      let kr = K[l * params.dim + k];
      let ki = K[l * params.dim + k + 1];

      let qr = Q[l * params.dim + k];
      let qi = Q[l * params.dim + k + 1];

      K[l * params.dim + k] = real * kr - img * ki;
      K[l * params.dim + k + 1] = real * ki + img * kr;

      Q[l * params.dim + k] = real * qr - img * qi;
      Q[l * params.dim + k + 1] = real * qi + img * qr;

      k = k + 2u;
    }
  }
}
