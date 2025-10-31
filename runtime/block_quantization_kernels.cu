// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

template <typename T>
__device__ __inline__ void localMaxReduction(float& local_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // --- Reduction Step 1 ---
  // Exchange and compare with thread 2 lanes away within the quad.
  // e.g., thread 0 exchanges with 2; thread 1 with 3.
  // The XOR pattern naturally keeps the operation within each quad.
  if (std::is_same<T, float>::value) {
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
    const Array<T, ITEMS_PER_THREAD, ALIGNMENT_1>& input,
    Array<__e2m1, ITEMS_PER_THREAD, ALIGNMENT_2>& output,
    Tensor<__e4m3, BLOCK_SCALE_DIM, BLOCK_SCALE_ALLOC>& block_scales,
    nvfuser_index_t logical_index,
    int input_logical_inner_dim_size) {
  constexpr bool is_half_or_bfloat =
      std::is_same<T, __bfloat>::value || std::is_same<T, __half>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(
      is_float || is_half_or_bfloat,
      "Input type must be float, __half or __bfloat");

  if constexpr (is_float) {
    assert(blockDim.x % 4 == 0);
  } else if constexpr (is_half_or_bfloat) {
    assert(blockDim.x % 2 == 0);
  }

  static_assert(
      (is_float && ITEMS_PER_THREAD == 4) ||
          (is_half_or_bfloat && ITEMS_PER_THREAD == 8),
      "ITEMS_PER_THREAD must be 4 for float type or 8 for __bfloat or __half "
      "type");

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
  localMaxReduction<T>(local_max);
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
