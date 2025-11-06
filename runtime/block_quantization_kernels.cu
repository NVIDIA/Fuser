// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

// This helper function finds the max of NUM_ELEMENTS (2, 4, or 8) values
// using the same number of threads.
template <int NUM_ELEMENTS>
__device__ __inline__ void reduceAcrossThreads(float& per_thread_computed_max) {
  // The mask 0xffffffff indicates all 32 threads in the warp are participating.
  unsigned int mask = 0xffffffff;

  // Perform reduction across threads in log2(NUM_ELEMENTS) stages
  // The reduction happens by progressively halving the distance between
  // threads that exchange values using XOR shuffle.
  // For NUM_ELEMENTS=8 (e.g., ITEMS_PER_THREAD=2): 3 stages (XOR with 4, 2, 1)
  // For NUM_ELEMENTS=4 (e.g., ITEMS_PER_THREAD=4): 2 stages (XOR with 2, 1)
  // For NUM_ELEMENTS=2 (e.g., ITEMS_PER_THREAD=8): 1 stage (XOR with 1)
#pragma unroll
  for (int offset = NUM_ELEMENTS / 2; offset > 0; offset /= 2) {
    per_thread_computed_max = fmax(
        per_thread_computed_max,
        __shfl_xor_sync(mask, per_thread_computed_max, offset));
  }

  // At this point, all threads involved hold the maximum value for the
  // (quantization) block.
}

// A runtime function to compute quantized nvfp4 output (output) and fp8 block
// scaling (block_scales) factors from fp32, fp16, bf16 inputs (input).
// The function is templatized over input type T (float, __half, __bfloat).
// This function assumes that for float, each thread is working on 2, 4 or 8
// elements (ITEMS_PER_THREAD). Thus n threads are working to quantize 16
// elements, where n = 16 / ITEMS_PER_THREAD.
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
      (is_float && (ITEMS_PER_THREAD == 4 || ITEMS_PER_THREAD == 2)) ||
          (is_half_or_bfloat &&
           (ITEMS_PER_THREAD == 8 || ITEMS_PER_THREAD == 4 ||
            ITEMS_PER_THREAD == 2)),
      "ITEMS_PER_THREAD must be 2, 4 for float type or 2, 4, or 8 for __bfloat "
      "or __half "
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

  // Compute the max accross  16/ITEMS_PER_THREAD threads
  // This assumes each thread has already computed is local max of 2, 4 (fp32)
  // or 2,4, 8 (bf16/fp16) elements.
  constexpr int NUM_ELEMENTS = 16 / ITEMS_PER_THREAD;
  reduceAcrossThreads<NUM_ELEMENTS>(local_max);
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
