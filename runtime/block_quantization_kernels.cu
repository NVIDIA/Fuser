// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

__device__ __inline__ void quadMaxReduction(float& local_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // --- Reduction Step 1 ---
  // Exchange and compare with thread 2 lanes away within the quad.
  // e.g., thread 0 exchanges with 2; thread 1 with 3.
  // The XOR pattern naturally keeps the operation within each quad.
  local_max = fmax(local_max, __shfl_xor_sync(mask, local_max, 2));

  // --- Reduction Step 2 ---
  // Exchange and compare with thread 1 lane away.
  // e.g., thread 0 exchanges with 1; thread 2 with 3.
  local_max = fmax(local_max, __shfl_xor_sync(mask, local_max, 1));

  // At this point, all threads in a quad hold the maximum value for that quad.
}

// TODO: Add a template parameter fnor input type.
// For now we just work on float.
// This also assumes a block of 16. That should be a
// template parameter.

// This assumes that ITEMS_PER_THREAD is 4.
// This assumes for block quantization, the block size is 16.
// This works for float but will extended to work with bfloat.
template <int ITEMS_PER_THREAD>
__device__ void block_quantize_to_nvfp4(
    Array<float, ITEMS_PER_THREAD, 1>& input,
    __e2m1& output,
    __e4m3& fp8_output) {
  assert(blockDim.x % 4 == 0);
  assert(blockDim.z == 1 && gridDim.z == 1);
  static_assert(
      ITEMS_PER_THREAD % 4 == 0, "ITEMS_PER_THREAD must be multiple of 4");

  Array<float, 4, 4> vec4;
  vec4.set(0.0f); // Initialize to zero like nvfuser does

  for (auto i = 0; i < ITEMS_PER_THREAD; i++) {
    vec4[i] = input[i];
  }

  float local_max = NEG_INFINITY;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    local_max = fmax(local_max, fabsf(vec4[i]));
  }

  // Perform block(16 elements)-wide reduction (max)
  // across 4- threads
  float block_max = NEG_INFINITY;
  quadMaxReduction(local_max);
  block_max = local_max;

  // This division should be replaced with a multiplication
  // by a reciprocal for better performance.
  float scaled_max = block_max / 6.000000000e+00f;
  float clamped_max = clamp(
      scaled_max, 1.562500000e-02f, 4.480000000e+02f); // Clamp between 0 and 1

  __e4m3 clamped_max_fp8 = __float2e4m3(clamped_max);

  float clamped_max_converted = __e4m32float(clamped_max_fp8);

  // Convert back from FP8 to float using __e4m32float
  if (threadIdx.x % 4 == 0) // Only one thread per quad writes
  {
    fp8_output = clamped_max_fp8; // Broadcast to all threads
  }

  Array<float, 4, 4> clamped_vals;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float scaled_val = vec4[i] / clamped_max_converted;
    clamped_vals[i] = clamp(scaled_val, -6.000000000e+00f, 6.000000000e+00f);
  }

  Array<__e2m1, 4, 1> fp4_vals;
  *reinterpret_cast<Array<__e2m1, 4, 4>*>(&fp4_vals[0]) =
      __float2e2m1(*reinterpret_cast<Array<float, 4, 4>*>(&clamped_vals[0]));

  Array<__e2m1, 4, 4> fp4_vals_aligned;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fp4_vals_aligned[i] = fp4_vals[i];
  }

  loadLocalToGlobal<__e2m1, /*vec_size=*/4, /*is_volatile=*/false>(
      &output, &fp4_vals_aligned.array[0]);
}

} // namespace bq
} // namespace nvf
