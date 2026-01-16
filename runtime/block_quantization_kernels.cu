// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace nvf {
namespace bq {

namespace {

// TODO: support vectorized store
template <int BLOCK_ROW_OUTER, int BLOCK_ROW_INNER, int BLOCK_COL>
__device__ nvfuser_index_t outputOffsetAfterSwizzlePadding(
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const nvfuser_index_t padded_col_size) {
  constexpr nvfuser_index_t BLOCK_ROW_SIZE = BLOCK_ROW_OUTER * BLOCK_ROW_INNER;

  /* logical dimension of matrix [ row_size, col_size]
   *
   * while logical domain after padding can be viewed as
   *   [ (row_tile*BLOCK_ROW_INNER*BLOCK_ROW_OUTER), (col_tile*BLOCK_COL) ]
   * where
   *   row_tile = ceilDiv(row_size / BLOCK_ROW_OUTER * BLOCK_ROW_INNER)
   *   col_tile = ceilDiv(col_size / BLOCK_COL)
   */

  // we first convert `row_idx` and `col_idx` to the logical index on the 5d
  // tensor.
  nvfuser_index_t row_tile_idx = row_idx / BLOCK_ROW_SIZE;
  nvfuser_index_t row_block_idx = row_idx % BLOCK_ROW_SIZE;
  nvfuser_index_t row_block_inner_idx = row_block_idx / BLOCK_ROW_OUTER;
  nvfuser_index_t row_block_outer_idx = row_block_idx % BLOCK_ROW_OUTER;
  nvfuser_index_t col_tile_idx = col_idx / BLOCK_COL;
  nvfuser_index_t col_block_idx = col_idx % BLOCK_COL;

  /* layout for matrix [ row_size, col_size]
   * it is viewed
   *   [row_tile, BLOCK_ROW_INNER, BLOCK_ROW_OUTER, col_tile, BLOCK_COL]
   * then transposed with axis (1, 3)
   *   [row_tile, col_tile, BLOCK_ROW_OUTER, BLOCK_ROW_INNER, BLOCK_COL]
   * and then made contiguous
   * So we can compute the corresponding stride for each dimension
   */
  constexpr nvfuser_index_t COL_TILE_STRIDE = BLOCK_ROW_SIZE * BLOCK_COL;
  constexpr nvfuser_index_t BLOCK_ROW_OUTER_STRIDE =
      BLOCK_ROW_INNER * BLOCK_COL;
  constexpr nvfuser_index_t BLOCK_ROW_INNER_STRIDE = BLOCK_COL;

  return row_tile_idx * padded_col_size * BLOCK_ROW_SIZE +
      col_tile_idx * COL_TILE_STRIDE +
      row_block_outer_idx * BLOCK_ROW_OUTER_STRIDE +
      row_block_inner_idx * BLOCK_ROW_INNER_STRIDE + col_block_idx;
}

} // namespace

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

// Helper function to convert input array to float and compute local maximum
template <int ITEMS_PER_THREAD, typename T, int ALIGNMENT>
__device__ __inline__ void convertToFloatAndComputeLocalMax(
    const Array<T, ITEMS_PER_THREAD, ALIGNMENT>& input,
    Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD>& vec_in,
    float& local_max) {
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

  local_max = NEG_INFINITY;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    local_max = fmax(local_max, fabsf(vec_in[i]));
  }
}

// Fast reciprocal of 2^biased_exp using bit manipulation
// Returns 1.0 for biased_exp==0, otherwise returns 2^(-biased_exp)
constexpr uint32_t FP32_MANTISSA_BITS = 23;
__device__ __forceinline__ float exp2f_rcp(uint8_t biased_exp) {
  return (biased_exp == 0)
      ? 1
      : __int_as_float(
            (254 - biased_exp)
            << FP32_MANTISSA_BITS); // 127 - (biased_exp - 127)
}

template <
    int ITEMS_PER_THREAD,
    typename T,
    int ALIGNMENT_1,
    int ALIGNMENT_2,
    int BLOCK_SCALE_DIM,
    int BLOCK_SCALE_ALLOC>
__device__ void block_quantize_to_mxfp8(
    const Array<T, ITEMS_PER_THREAD, ALIGNMENT_1>& input,
    Array<__e4m3, ITEMS_PER_THREAD, ALIGNMENT_2>& output,
    Tensor<__e8m0, BLOCK_SCALE_DIM, BLOCK_SCALE_ALLOC>& block_scales,
    nvfuser_index_t logical_index) {
  // Number of threads involved in computing one block scaling factor
  constexpr int THREADS_PER_SCALING_FACTOR = 32 / ITEMS_PER_THREAD;

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> vec_in;
  float local_max;
  convertToFloatAndComputeLocalMax<ITEMS_PER_THREAD, T, ALIGNMENT_1>(
      input, vec_in, local_max);

  // Compute the max accross  32/ITEMS_PER_THREAD threads
  // This assumes each thread has already computed is local max of 2, 4 (fp32)
  // or 2,4, 8 (bf16/fp16) elements.
  reduceAcrossThreads<THREADS_PER_SCALING_FACTOR>(local_max);
  float block_max = local_max;

  static constexpr float max_norm_rcp = 1.0f / 448;
  __e8m0 exponent = __float2e8m0(block_max * max_norm_rcp);

  // Write out the block scaling factor to global memory.
  // This assumes block_size (32) elements in the input were contiguous.
  // Only one block scaling factor is written out per 32(assumed block size)
  // elements.
  int offset = logical_index / 32;
  if (threadIdx.x % THREADS_PER_SCALING_FACTOR == 0) {
    block_scales[offset] = exponent;
  }

  const float block_scale_inverse = exp2f_rcp(exponent.raw());

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    output[i] = __float2e4m3(vec_in[i] * block_scale_inverse);
  }
}

// A runtime function to compute quantized nvfp4 output (output) and fp8 block
// scaling (block_scales) factors from fp32, fp16, bf16 inputs (input).
// The function is templatized over input type T (float, __half, __bfloat).
// This function assumes that for float, each thread is working on 2 or 4
// elements (ITEMS_PER_THREAD). Thus n threads are working to quantize 16
// elements, where n = 16 / ITEMS_PER_THREAD.
template <
    bool USE_GLOBAL_SCALE,
    int ITEMS_PER_THREAD,
    typename T,
    int ALIGNMENT_1,
    int ALIGNMENT_2,
    int BLOCK_SCALE_DIM,
    int BLOCK_SCALE_ALLOC>
__device__ void block_quantize_to_nvfp4_util(
    const Array<T, ITEMS_PER_THREAD, ALIGNMENT_1>& input,
    Array<__e2m1, ITEMS_PER_THREAD, ALIGNMENT_2>& output,
    Tensor<__e4m3, BLOCK_SCALE_DIM, BLOCK_SCALE_ALLOC>& block_scales,
    Tensor<float, 0, 0>& global_scale,
    int64_t offset) {
  // Number of threads involved in computing one block scaling factor
  constexpr int THREADS_PER_SCALING_FACTOR = 16 / ITEMS_PER_THREAD;

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> vec_in;
  float local_max;
  convertToFloatAndComputeLocalMax<ITEMS_PER_THREAD, T, ALIGNMENT_1>(
      input, vec_in, local_max);

  // Compute the max accross  16/ITEMS_PER_THREAD threads
  // This assumes each thread has already computed is local max of 2, 4 (fp32)
  // or 2,4, 8 (bf16/fp16) elements.
  reduceAcrossThreads<THREADS_PER_SCALING_FACTOR>(local_max);
  float block_max = local_max;

  float scaled_max = 0;
  constexpr float float_max = 3.40282346638528859811704183484516925440e+38f;
  scaled_max = block_max / 6.0f;
  if constexpr (USE_GLOBAL_SCALE) {
    scaled_max = scaled_max * global_scale[0];
  }
  scaled_max = fminf(scaled_max, float_max);

  __e4m3 clamped_max_fp8 = __float2e4m3(scaled_max);

  // Convert back from FP8 to float using __e4m32float
  scaled_max = __e4m32float(clamped_max_fp8);

  if constexpr (USE_GLOBAL_SCALE) {
    auto decode_scale = 1.0f / global_scale[0];
    scaled_max = fminf(1.0f / (scaled_max * decode_scale), float_max);
  } else {
    scaled_max = fminf(1.0f / scaled_max, float_max);
  }

  if (threadIdx.x % THREADS_PER_SCALING_FACTOR == 0) {
    block_scales[offset] = clamped_max_fp8;
  }

  Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD> scaled_vals;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    scaled_vals[i] = vec_in[i] * scaled_max;
  }

  Array<__e2m1, ITEMS_PER_THREAD, 1> fp4_vals;
  *reinterpret_cast<Array<__e2m1, ITEMS_PER_THREAD, ITEMS_PER_THREAD>*>(
      &fp4_vals[0]) =
      __float2e2m1(
          *reinterpret_cast<Array<float, ITEMS_PER_THREAD, ITEMS_PER_THREAD>*>(
              &scaled_vals[0]));

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    output[i] = fp4_vals[i];
  }
}

template <
    bool USE_GLOBAL_SCALE,
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
    Tensor<float, 0, 0> global_scale,
    int64_t fp8_scaling_factors_inner_dim = -1,
    int64_t alloc_dim0 = -1,
    int64_t alloc_dim1 = -1,
    int64_t alloc_dim2 = -1,
    int64_t alloc_dim3 = -1,
    int64_t alloc_dim4 = -1) {
  // Write out the block scaling factor to global memory.
  // This assumes 16 elements in the input were contiguous.
  // Only one block scaling factor is written out per 16(assumed block size)
  // elements.
  int offset = logical_index / 16;

  if (fp8_scaling_factors_inner_dim > 0) {
    auto stride_4 = 1;
    auto stride_3 = stride_4 * alloc_dim4;
    auto stride_2 = stride_3 * alloc_dim3;
    auto stride_1 = stride_2 * alloc_dim2;
    auto stride_0 = stride_1 * alloc_dim1;

    auto logical_inner = offset % fp8_scaling_factors_inner_dim;
    auto logical_outer = offset / fp8_scaling_factors_inner_dim;

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
  block_quantize_to_nvfp4_util<USE_GLOBAL_SCALE>(
      input, output, block_scales, global_scale, offset);
}

template <
    typename T,
    typename Index_T,
    int BLOCK_ROW_OUTER,
    int BLOCK_ROW_INNER,
    int BLOCK_COL,
    int UNROLL_FACTOR>
__device__ void preprocessGroupedMatmulInputSf(
    T* output,
    const T* input,
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const Index_T* input_offsets,
    const Index_T* output_offsets,
    const nvfuser_index_t col_size,
    const nvfuser_index_t group_size) {
  // find corresponding expert_id
  int expert_id = group_size - 1;
  for (int i = 1; i < group_size; ++i) {
    if (row_idx < input_offsets[i]) {
      expert_id = i - 1;
      break;
    }
  }

  // row idx for current group
  nvfuser_index_t c_row_idx = row_idx - input_offsets[expert_id];
  // compute output group offset for current group
  nvfuser_index_t padded_col_size =
      (col_size + BLOCK_COL - 1) / BLOCK_COL * BLOCK_COL;
  T* out_group_offset = output + output_offsets[expert_id] * padded_col_size;

  // TODO: vectorized load/store instead of for loop
  for (int i = 0; i < UNROLL_FACTOR && col_idx + i < col_size; ++i) {
    nvfuser_index_t index = outputOffsetAfterSwizzlePadding<
        BLOCK_ROW_OUTER,
        BLOCK_ROW_INNER,
        BLOCK_COL>(c_row_idx, col_idx + i, padded_col_size);
    out_group_offset[index] = input[i];
  }
}

template <
    bool USE_GLOBAL_SCALE,
    int BLOCK_ROW_OUTER,
    int BLOCK_ROW_INNER,
    int BLOCK_COL,
    int ITEMS_PER_THREAD,
    typename T,
    typename Index_T,
    int ALIGNMENT_1,
    int ALIGNMENT_2,
    int BLOCK_SCALE_DIM,
    int BLOCK_SCALE_ALLOC>
__device__ void grouped_block_quantize_to_nvfp4(
    const Array<T, ITEMS_PER_THREAD, ALIGNMENT_1>& input,
    Array<__e2m1, ITEMS_PER_THREAD, ALIGNMENT_2>& output,
    Tensor<__e4m3, BLOCK_SCALE_DIM, BLOCK_SCALE_ALLOC>& block_scales,
    const nvfuser_index_t row_idx,
    const nvfuser_index_t col_idx,
    const Index_T* input_offsets,
    const Index_T* output_offsets,
    const nvfuser_index_t col_size,
    const nvfuser_index_t group_size,
    Tensor<float, 0, 0> global_scale) {
  // find corresponding expert_id
  int expert_id = group_size - 1;
  for (int i = 1; i < group_size; ++i) {
    if (row_idx < input_offsets[i]) {
      expert_id = i - 1;
      break;
    }
  }
  // NOTE: col_size and col_idx needs to be divided by block size for scaling
  // factor
  constexpr nvfuser_index_t BLOCK_SIZE = 16;
  // row idx for current group
  nvfuser_index_t c_row_idx = row_idx - input_offsets[expert_id];
  // compute output group offset for current group
  nvfuser_index_t padded_col_size =
      (col_size / BLOCK_SIZE + BLOCK_COL - 1) / BLOCK_COL * BLOCK_COL;
  nvfuser_index_t out_group_offset =
      output_offsets[expert_id] * padded_col_size;
  // compute the offset
  nvfuser_index_t index = outputOffsetAfterSwizzlePadding<
      BLOCK_ROW_OUTER,
      BLOCK_ROW_INNER,
      BLOCK_COL>(c_row_idx, col_idx / BLOCK_SIZE, padded_col_size);
  nvfuser_index_t offset = out_group_offset + index;

  block_quantize_to_nvfp4_util<USE_GLOBAL_SCALE>(
      input, output, block_scales, global_scale, offset);
}

} // namespace bq
} // namespace nvf
