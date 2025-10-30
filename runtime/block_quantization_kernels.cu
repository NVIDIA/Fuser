// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

template <int ITEMS_PER_THREAD>
__device__ __inline__ void localMaxReduction(float& local_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // --- Reduction Step 1 ---
  // Exchange and compare with thread 2 lanes away within the quad.
  // e.g., thread 0 exchanges with 2; thread 1 with 3.
  // The XOR pattern naturally keeps the operation within each quad.

  if constexpr (ITEMS_PER_THREAD == 2) {
    local_max = fmax(local_max, __shfl_xor_sync(mask, local_max, 4));
  }

  if constexpr (ITEMS_PER_THREAD == 2 || ITEMS_PER_THREAD == 4) {
    local_max = fmax(local_max, __shfl_xor_sync(mask, local_max, 2));
  }

  // --- Reduction Step 2 ---
  // Exchange and compare with thread 1 lane away.
  // e.g., thread 0 exchanges with 1; thread 2 with 3.
  local_max = fmax(local_max, __shfl_xor_sync(mask, local_max, 1));

  // At this point, all threads in a quad hold the maximum value for that quad.
}

// TODO: Add a template parameter for input type.
// For now we just work on float.
// This also assumes a block of 16. That should be a
// template parameter.

// This assumes that ITEMS_PER_THREAD is 4.
// This assumes for block quantization, the block size is 16.
// This works for float but will extended to work with bfloat.
template <
    int ITEMS_PER_THREAD,
    typename T,
    int ALIGNMENT_1,
    int ALIGNMENT_2,
    int BLOCK_SCALE_DIM,
    int BLOCK_SCALE_ALLOC>
__device__ void block_quantize_to_nvfp4(
    Array<T, ITEMS_PER_THREAD, ALIGNMENT_1>& input,
    Array<__e2m1, ITEMS_PER_THREAD, ALIGNMENT_2>& output,
    Tensor<__e4m3, BLOCK_SCALE_DIM, BLOCK_SCALE_ALLOC>& fp8_output,
    nvfuser_index_t logical_index,
    int input_logical_inner_dim_size,
    int64_t fp8_output_inner_dim = -1,
    int64_t alloc_dim0 = -1,
    int64_t alloc_dim1 = -1,
    int64_t alloc_dim2 = -1,
    int64_t alloc_dim3 = -1,
    int64_t alloc_dim4 = -1) {
  constexpr bool is_half_or_bfloat =
      std::is_same<T, __bfloat>::value || std::is_same<T, __half>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(
      is_float || is_half_or_bfloat,
      "Input type must be float, __half or __bfloat");

  if constexpr (is_float) {
    static_assert(
        ITEMS_PER_THREAD == 4 || ITEMS_PER_THREAD == 2,
        "ITEMS_PER_THREAD must be 4 or 2 for float type");
  } else if constexpr (is_half_or_bfloat) {
    static_assert(
        ITEMS_PER_THREAD == 8 || ITEMS_PER_THREAD == 4 || ITEMS_PER_THREAD == 2,
        "ITEMS_PER_THREAD must be 8, 4 or 2 for __bfloat or __half type");
  }

  assert(input_logical_inner_dim_size % 16 == 0);

  int THREADS_PER_SCALING_FACTOR = 16 / ITEMS_PER_THREAD;

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> vec_in;
  vec_in.set(0.0f); // Initialize to zero like nvfuser does

  for (auto i = 0; i < ITEMS_PER_THREAD; i++) {
    if constexpr (std::is_same<T, float>::value) {
      vec_in[i] = input[i];
    } else if constexpr (std::is_same<T, __bfloat>::value) {
      vec_in[i] = __bfloat2float(input[i]);
    } else if constexpr (std::is_same<T, __half>::value) {
      vec_in[i] = __half2float(input[i]);
    }
  }

  float local_max = NEG_INFINITY;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    local_max = fmax(local_max, fabsf(vec_in[i]));
  }

  // Perform block(16 elements)-wide reduction (max)
  // across 4- threads
  float block_max = NEG_INFINITY;
  localMaxReduction<ITEMS_PER_THREAD>(local_max);
  block_max = local_max;

  // This division should be replaced with a multiplication
  // by a reciprocal for better performance.
  float scaled_max = block_max / 6.000000000e+00f;
  float clamped_max = clamp(
      scaled_max, 1.562500000e-02f, 4.480000000e+02f); // Clamp between 0 and 1

  __e4m3 clamped_max_fp8 = __float2e4m3(clamped_max);

  float clamped_max_converted = __e4m32float(clamped_max_fp8);

  int offset_y_blocks = blockIdx.y * blockDim.y * blockDim.x * gridDim.x;
  int offset_dim_y = threadIdx.y * blockDim.x * gridDim.x;
  int offset_into_block = blockIdx.x * blockDim.x + threadIdx.x;

  int offset = logical_index / 16;

  if (fp8_output_inner_dim > 0) {
    auto stride_4 = 1;
    auto stride_3 = stride_4 * alloc_dim4;
    auto stride_2 = stride_3 * alloc_dim3;
    auto stride_1 = stride_2 * alloc_dim2;
    auto stride_0 = stride_1 * alloc_dim1;

    auto logical_inner = offset % fp8_output_inner_dim;
    auto logical_outer = offset / fp8_output_inner_dim;

    // The allocation domain swizzle logic is:
    // m, k -> m, k/4, 4
    // m, k/4, 4 -> m/128, 128, k/4, 4 ->
    // m/128, 4(m), 32, k/4, 4(k) ->
    // m/128, k/4, 32, 4(m), 4(k)

    auto pos_4 = logical_inner % 4;
    auto pos_1 = logical_inner / 4;
    auto pos_t = logical_outer % 128;
    auto pos_0 = logical_outer / 128;
    auto pos_3 = pos_t / 32;
    auto pos_2 = pos_t % 32;

    offset = pos_4 * stride_4 + pos_3 * stride_3 + pos_2 * stride_2 +
        pos_1 * stride_1 + pos_0 * stride_0;
  }

  // Convert back from FP8 to float using __e4m32float
  if (threadIdx.x % THREADS_PER_SCALING_FACTOR == 0) {
    fp8_output[offset] = clamped_max_fp8; // Broadcast to all threads
  }

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> clamped_vals;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    float scaled_val = vec_in[i] / clamped_max_converted;
    clamped_vals[i] = clamp(scaled_val, -6.000000000e+00f, 6.000000000e+00f);
  }

  Array<__e2m1, ITEMS_PER_THREAD, 1> fp4_vals;
  *reinterpret_cast<Array<__e2m1, ITEMS_PER_THREAD, ITEMS_PER_THREAD>*>(
      &fp4_vals[0]) =
      __float2e2m1(
          *reinterpret_cast<Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD>*>(
              &clamped_vals[0]));

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    output[i] = fp4_vals[i];
  }
}

} // namespace bq
} // namespace nvf
