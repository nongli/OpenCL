__kernel void 
SimpleKernel(const __global float* input, __global float* output) {
  size_t global_id = get_global_id(0);
  output[global_id] = sin(fabs(input[global_id]));
}

__kernel void 
SimpleKernel4(const __global float4* input, __global float4* output) {
  size_t global_id = get_global_id(0);
  output[global_id] = sin(fabs(input[global_id]));
}

