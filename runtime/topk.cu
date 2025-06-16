// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cub/block/block_radix_sort.cuh>

namespace nvf {
namespace topk {

// Block state constants following nvFuser conventions from fused_reduction.cu
// Sort Domain - TEMPLATE STATE 0
//   - Participating in the topk selection, has values coming in, selected
//   values coming out
// Iteration Domain - TEMPLATE STATE 1
//   - Not participating in the topk selection, has values across the dimension
//   after selection
constexpr __device__ bool isSort(int STATE) {
  return STATE == 0;
}

constexpr __device__ bool isIter(int STATE) {
  return STATE == 1;
}

// TODO: We have exactly the same code in argsort.cu, we should refactor it.
// Type utils for interoperability between our own half types and the
// CUDA standard types (identical to argsort implementation)
template <typename T>
struct CudaType {
  using type = T;

  __device__ inline static T get(const T& t) {
    return t;
  }
};

#ifdef __NVFUSER_HAS_HALF__
template <>
struct CudaType<__half> {
  using type = __nv_half;

  __device__ inline static typename CudaType<__half>::type get(
      const __half& t) {
    return __ushort_as_half(__NVFUSER_HALF_TO_CUS(t));
  }
};
#endif // __NVFUSER_HAS_HALF__

#ifdef __NVFUSER_HAS_BFLOAT__
template <>
struct CudaType<__bfloat> {
  using type = __nv_bfloat16;

  __device__ inline static typename CudaType<__bfloat>::type get(
      const __bfloat& t) {
    return __ushort_as_bfloat16(__NVFUSER_BFLOAT_TO_CUS(t));
  }
};
#endif // __NVFUSER_HAS_BFLOAT__

// Block-parallel topk using CUB BlockRadixSort
// Following nvFuser dimensional template parameter pattern like
// fused_reduction.cu and argsort.cu
//
// For simplicity, we assume that:
// - top_values: Output array containing top-K values. Each thread holds
//   ITEMS_PER_THREAD values.
// - top_indices: Output array containing original indices of top-K values
// - Each thread holds exactly ITEMS_PER_THREAD elements in both output arrays
// - In the actual nvFuser-generated kernel, only the first k elements
//   that correspond to the logical domain should be used (via predication)
// - This implementation sorts ALL elements and returns them in the output
//   arrays, but the consuming kernel should only read the first k elements
template <
    int BLOCK_DIM_X,
    int BLOCK_DIM_Y,
    int BLOCK_DIM_Z,
    int BLOCK_STATE_X,
    int BLOCK_STATE_Y,
    int BLOCK_STATE_Z,
    typename DataT,
    int ITEMS_PER_THREAD,
    typename BlockDimT>
__device__ void blockTopK(
    DataT (&top_values)[ITEMS_PER_THREAD],
    int64_t (&top_indices)[ITEMS_PER_THREAD],
    const DataT (&input_data)[ITEMS_PER_THREAD],
    int k,
    bool largest,
    bool sorted,
    BlockDimT block_dim) {
  // For now, only support all dimensions participating in sort (state=0)
  // This follows the same constraint as argsort implementation
  static_assert(
      (isSort(BLOCK_STATE_X) || BLOCK_DIM_X == 1) &&
          (isSort(BLOCK_STATE_Y) || BLOCK_DIM_Y == 1) &&
          (isSort(BLOCK_STATE_Z) || BLOCK_DIM_Z == 1),
      "For now, active TID dimensions must participate in sorting");

  // Initialize output arrays directly (no temporary buffers needed)
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    top_values[i] = CudaType<DataT>::get(input_data[i]);
  }

  // CUB BlockRadixSort setup - with proper multi-dimensional block support
  // CUB supports multi-dimensional blocks when BLOCK_DIM_Y and BLOCK_DIM_Z are
  // specified (identical to argsort configuration)
  using BlockRadixSort = cub::BlockRadixSort<
      typename CudaType<DataT>::type, // Key type
      BLOCK_DIM_X, // X dimension
      ITEMS_PER_THREAD, // Items per thread
      int64_t, // Value type (for key-value sorting)
      4, // RADIX_BITS (default)
      true, // MEMOIZE_OUTER_SCAN (default for modern architectures)
      cub::BLOCK_SCAN_WARP_SCANS, // INNER_SCAN_ALGORITHM (default)
      cudaSharedMemBankSizeFourByte, // SMEM_CONFIG (default)
      BLOCK_DIM_Y, // Y dimension
      BLOCK_DIM_Z // Z dimension
      >;

  // Allocate shared memory for CUB operations
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // Thread ID using all threads in the block (CUB requires full block
  // participation) CUB doesn't support sorting of just blockDim.x, so all
  // threads participate (identical to argsort pattern)
  unsigned int thread_id =
      index_utils::maskedOffset<true, true, true>(threadIdx, block_dim);

  // Initialize indices array: 0, 1, 2, ..., n-1 per thread
  // This creates the mapping between sorted values and their original positions
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    top_indices[i] = thread_id * ITEMS_PER_THREAD + i;
  }

  // Perform key-value sorting using CUB directly on output arrays
  // Keys = top_values, Values = top_indices
  // After sorting, top_indices array contains mapping to original positions
  if (largest) {
    // For k largest elements, sort in descending order
    BlockRadixSort(temp_storage).SortDescending(top_values, top_indices);
  } else {
    // For k smallest elements, sort in ascending order
    BlockRadixSort(temp_storage).Sort(top_values, top_indices);
  }

  // Output arrays are already sorted in-place by CUB - no copying needed!

  // IMPLEMENTATION NOTE: This test implementation populates ALL
  // ITEMS_PER_THREAD elements in the output arrays with sorted results.
  // However, in the actual nvFuser-generated kernel, only the first k elements
  // should be consumed from each thread's output arrays (top_values[0..k-1] and
  // top_indices[0..k-1]). The remaining elements beyond index k-1 should be
  // ignored via predication.

  // Note: 'sorted' parameter is currently ignored as CUB BlockRadixSort
  // always produces sorted output. Future optimization could skip sorting
  // when sorted=false by using a selection algorithm instead.
}

} // namespace topk
} // namespace nvf
