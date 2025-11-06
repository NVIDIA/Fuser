// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

// This helper function is templatized of over types float, __half, and
// __bfloat. This assumes that for float, each thread was working on 4 elements.
// Thus 4 threads were working to find the max of 16 elements, and hence we need
// two steps to find the maximum. If the type is __bfloat or __half, then we
// only need a single step to find the maximum of 16 elements as each thread was
// working on 8 elements and 2 threads are required to compute the max of 16
// elements.
// This function assumes for float each thread has already computed the max of 4
// elements (8 elements for the other 2 data types) and the block size is 16, so
// we have 4 threads (2 for bf16/fp16) participating in the reduction.
// TODO: For FP32 support the cases where each thread works on 2 or 4 elements.
// TODO: For bf16/fp16 support the cases where each thread works on 2,4, or 8
// elements.
template <typename T>
__device__ __inline__ void reduceAcrossThreads(float& per_thread_computed_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // --- Reduction Step 1 ---
  // Exchange and compare with thread 2 lanes away within the quad.
  // e.g., thread 0 exchanges with 2; thread 1 with 3.
  // The XOR pattern naturally keeps the operation within each quad.
  if (std::is_same<T, float>::value) {
    per_thread_computed_max = fmax(
        per_thread_computed_max,
        __shfl_xor_sync(mask, per_thread_computed_max, 2));
  }

  // --- Reduction Step 2 ---
  // Exchange and compare with thread 1 lane away.
  // e.g., thread 0 exchanges with 1; thread 2 with 3.
  per_thread_computed_max = fmax(
      per_thread_computed_max,
      __shfl_xor_sync(mask, per_thread_computed_max, 1));

  // At this point, all threads in a quad hold the maximum value for that
  // quad(pair of 2 threads).
}

// A runtime function to compute quantized nvfp4 output (output) and fp8 block
// scaling (block_scales) factors from fp32, fp16, bf16 inputs (input).
// The function is templatized over input type T (float, __half, __bfloat).
// This function assumes that for float, each thread is working on 4 elements.
// Thus 4 threads are working to quantize 16 elements. If the type is __bfloat
// or
// __half, then 2 threads are working to quantize 16 elements as each thread
// is working on 8 elements.
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
    nvfuser_index_t logical_index) {
  constexpr bool is_half_or_bfloat =
      std::is_same<T, __bfloat>::value || std::is_same<T, __half>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  static_assert(
      is_float || is_half_or_bfloat,
      "Input type must be float, __half or __bfloat");

  static_assert(
      (is_float && ITEMS_PER_THREAD == 4) ||
          (is_half_or_bfloat && ITEMS_PER_THREAD == 8),
      "ITEMS_PER_THREAD must be 4 for float type or 8 for __bfloat or __half "
      "type");

  // Number of threads involved in computing one block scaling factor
  constexpr int THREADS_PER_SCALING_FACTOR = 16 / ITEMS_PER_THREAD;

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> vec_in;
  vec_in.set(0.0f);

#pragma unroll
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

  // Compute the max accross 4 threads (float) or 2 threads (bf16/fp16)
  // This assumes each thread has already computed is local max of 4 (fp32) or
  // 8 (bf16/fp16) elements.
  reduceAcrossThreads<T>(local_max);
  float block_max = local_max;

  // This division should be replaced with a multiplication
  // by a reciprocal for better performance.
  float scaled_max = block_max / 6.000000000e+00f;
  float clamped_max = clamp(
      scaled_max, 1.562500000e-02f, 4.480000000e+02f); // Clamp between 0 and 1

  __e4m3 clamped_max_fp8 = __float2e4m3(clamped_max);

  // Convert back from FP8 to float using __e4m32float
  float clamped_max_converted = __e4m32float(clamped_max_fp8);

  // Write out the block scaling factor to global memory.
  // This assumes 16 elements in the input were contiguous.
  // Only one block scaling factor is written out per 16(assumed block size)
  // elements.
  int offset = logical_index / 16;
  if (threadIdx.x % THREADS_PER_SCALING_FACTOR == 0) {
    block_scales[offset] = clamped_max_fp8;
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
