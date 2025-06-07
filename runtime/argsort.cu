#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>

#include "index_utils.cu"

namespace nvfuser_runtime {

// Block state constants following nvFuser conventions from fused_reduction.cu
// Sort Domain - TEMPLATE STATE 0
//   - Participating in the sort, has values coming in, sorted values coming out
// Iteration Domain - TEMPLATE STATE 1
//   - Not participating in the sort, has values across the dimension after sorting
constexpr __device__ bool isSort(int STATE) {
  return STATE == 0;
}

constexpr __device__ bool isIter(int STATE) {
  return STATE == 1;
}

// Block-parallel argsort using CUB BlockRadixSort
// Following nvFuser dimensional template parameter pattern like fused_reduction.cu
template <
    int BLOCK_DIM_X,
    int BLOCK_DIM_Y,
    int BLOCK_DIM_Z,
    int BLOCK_STATE_X,
    int BLOCK_STATE_Y,
    int BLOCK_STATE_Z,
    bool Aligned,
    typename DataT,
    int ITEMS_PER_THREAD,
    typename BlockDimT>
__device__ void blockArgsort(
    DataT (&input_data)[ITEMS_PER_THREAD],
    nvfuser_index_t (&indices)[ITEMS_PER_THREAD],
    bool descending,
    DataT* shared_mem,
    BlockDimT block_dim) {

    // For now, only support all dimensions participating in sort (state=0)
    static_assert(
        isSort(BLOCK_STATE_X) && isSort(BLOCK_STATE_Y) && isSort(BLOCK_STATE_Z),
        "For now, only all sort states are supported");

    // CUB BlockRadixSort setup - with proper multi-dimensional block support
    // CUB supports multi-dimensional blocks when BLOCK_DIM_Y and BLOCK_DIM_Z are specified
    using BlockRadixSort = cub::BlockRadixSort<
        DataT,                    // Key type
        BLOCK_DIM_X,             // X dimension
        ITEMS_PER_THREAD,        // Items per thread
        nvfuser_index_t,         // Value type (for key-value sorting)
        4,                       // RADIX_BITS (default)
        true,                    // MEMOIZE_OUTER_SCAN (default for modern architectures)
        cub::BLOCK_SCAN_WARP_SCANS,  // INNER_SCAN_ALGORITHM (default)
        cudaSharedMemBankSizeFourByte,  // SMEM_CONFIG (default)
        BLOCK_DIM_Y,             // Y dimension
        BLOCK_DIM_Z              // Z dimension
    >;

    // Allocate shared memory for CUB operations
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Thread ID using all threads in the block (CUB requires full block participation)
    // CUB doesn't support sorting of just blockDim.x, so all threads participate
    unsigned int thread_id =
        index_utils::maskedOffset<true, true, true>(threadIdx, block_dim);

    // Initialize indices array: 0, 1, 2, ..., n-1 per thread
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        indices[i] = thread_id * ITEMS_PER_THREAD + i;
    }

    // Perform key-value sorting using CUB
    // Keys = input_data, Values = indices
    // After sorting, indices array contains argsort result
    if (descending) {
        BlockRadixSort(temp_storage).SortDescending(input_data, indices);
    } else {
        BlockRadixSort(temp_storage).Sort(input_data, indices);
    }
}

} // namespace nvfuser_runtime