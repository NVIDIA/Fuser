namespace nvf {

namespace bq {

__device__ __inline__ void quadMaxReduction(
    const float* input,
    float& local_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // --- Reduction Step 1 ---
  // Exchange and compare with thread 2 lanes away within the quad.
  // e.g., thread 0 exchanges with 2; thread 1 with 3.
  // The XOR pattern naturally keeps the operation within each quad.
  local_max = nv_fmax(local_max, __shfl_xor_sync(mask, local_max, 2));

  // --- Reduction Step 2 ---
  // Exchange and compare with thread 1 lane away.
  // e.g., thread 0 exchanges with 1; thread 2 with 3.
  local_max = nv_fmax(local_max, __shfl_xor_sync(mask, local_max, 1));

  // At this point, all threads in a quad hold the maximum value for that quad.
}

/**
 * Templated CUDA kernel for float to FP4/FP8 conversion with vectorized loads
 *
 * Template Parameters:
 * - BLOCK_DIM_X: Block dimension in X direction (must be 4 for vectorized
 * loads)
 * - BLOCK_DIM_Y: Block dimension in Y direction
 *
 * Parameters:
 * - input: Pointer to 1D array representing 2D data where inner dimension is
 * fixed to 16
 * - output_e2m1: Output pointer for __e2m1 (FP4 E2M1) format
 * - output_e4m3: Output pointer for __e4m3 (FP8 E4M3) format
 * - num_rows: Number of rows in the input (outer dimension)
 */
__global__ void float_to_nvfp4_conversion_kernel(
    const float* __restrict__ input,
    __e2m1* __restrict__ output_e2m1,
    __e4m3* __restrict__ output_e4m3,
    int num_rows,
    int total_elements) {
  // Ensure BLOCK_DIM_X is multiple of 4 for vectorized loads
  assert(blockDim.x % 4 == 0);

  const int tiles_16_processed_by_dimx =
      (blockDim.x * 4) / 16; // 4 elements per thread

  const int grid_offset = gridDim.x * blockIdx.y + blockIdx.x;
  const int tiles_processed_by_ctas =
      grid_offset * blockDim.y * tiles_16_processed_by_dimx;

  const int current_offset_of_cta =
      threadIdx.y * tiles_16_processed_by_dimx * 16 + threadIdx.x * 4;
  const int total_offset = tiles_processed_by_ctas * 16 + current_offset_of_cta;

  if (total_offset >= total_elements) {
    // No work for this thread
    return;
  }

  // Create an array of 4 floats for vectorized load using nvfuser-style
  Array<float, 4, 4> vec4;
  vec4.set(0.0f); // Initialize to zero like nvfuser does

  if (total_offset + 3 < total_elements) {
    // Use loadGlobalToLocal for vectorized load similar to nvfuser pattern
    loadGlobalToLocal<
        float,
        /*vec_size=*/4,
        /*is_volatile=*/false,
        CacheOp::Streaming>(
        &vec4.array[0], const_cast<float*>(&input[total_offset]));
  } else {
    // Assert that total_elements is even for proper FP4 handling
    assert(total_elements % 2 == 0);

    // Handle boundary case with element-wise loads
    loadGlobalToLocal<
        float,
        /*vec_size=*/2,
        /*is_volatile=*/false,
        CacheOp::Streaming>(
        &vec4.array[0], const_cast<float*>(&input[total_offset]));
  }

  // Calculate the max of the values in vec4
  float local_max = NEG_INFINITY;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    local_max = nv_fmax(local_max, fabsf(vec4[i]));
  }

  // Perform block-wide maximum reduction across threads
  float block_max = NEG_INFINITY;
  quadMaxReduction(input, local_max);
  block_max = local_max;

  float scaled_max = block_max / 6.000000000e+00f;
  float clamped_max = clamp(
      scaled_max, 1.562500000e-02f, 4.480000000e+02f); // Clamp between 0 and 1

  __e4m3 clamped_max_fp8 = __float2e4m3(clamped_max);

  // Convert back from FP8 to float using __e4m32float
  float clamped_max_converted = __e4m32float(clamped_max_fp8);

  // Broadcast clamped_max_converted from thread 0 in X dimension to all threads
  float broadcasted_clamped_max = NEG_INFINITY;
  broadcasted_clamped_max = clamped_max_converted;

  // Process vec4 array: divide each element by broadcasted_clamped_max and
  // clamp
  Array<float, 4, 4> clamped_vals;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float scaled_val = vec4[i] / broadcasted_clamped_max;
    clamped_vals[i] = clamp(scaled_val, -6.000000000e+00f, 6.000000000e+00f);
  }

  // Convert clamped_vals to FP4 E2M1 format using nvfuser-style vectorized
  // operations T10 corresponds to clamped_vals, T11 corresponds to fp4_vals
  Array<__e2m1, 4, 1> fp4_vals;
  *reinterpret_cast<Array<__e2m1, 4, 4>*>(&fp4_vals[0]) =
      __float2e2m1(*reinterpret_cast<Array<float, 4, 4>*>(&clamped_vals[0]));

  Array<__e2m1, 4, 4> fp4_vals_aligned;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fp4_vals_aligned[i] = fp4_vals[i];
  }

  // Write back the clamped max value if this is thread 0 in X dimension
  if (threadIdx.x % 4 == 0) {
    output_e4m3[total_offset / 16] = clamped_max_fp8;
  }

  if (total_offset + 3 < total_elements) {
    // Store fp4_vals_aligned using nvfuser-style vectorized store
    loadLocalToGlobal<__e2m1, /*vec_size=*/4, /*is_volatile=*/false>(
        &output_e2m1[total_offset / 2], &fp4_vals_aligned.array[0]);
  } else {
    // Store fp4_vals_aligned using nvfuser-style vectorized store
    loadLocalToGlobal<__e2m1, /*vec_size=*/2, /*is_volatile=*/false>(
        &output_e2m1[total_offset / 2], &fp4_vals_aligned.array[0]);
  }
}
} // namespace bq
} // namespace nvf
