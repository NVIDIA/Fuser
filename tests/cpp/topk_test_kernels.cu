// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Define nvfuser_index_t at global scope for runtime files
using nvfuser_index_t = int64_t;

#include <tests/cpp/topk_test_helper.h>

// Need to be included before argsort because of the dependency
// from argsort
namespace nvf {
#include <runtime/index_utils.cu>
} // namespace nvf

#include <runtime/topk.cu>

// Standard C++ headers
#include <cstdint>

// CUDA headers
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Test framework headers
#include <gtest/gtest.h>

namespace nvfuser {

//============================================================================
// Fixed template parameter kernels (avoiding dynamic template instantiation)
//============================================================================

// Basic topk test kernel with configurable block size
template <int BLOCK_SIZE, typename DataT, int ITEMS_PER_THREAD>
__global__ void basicTopkTestKernel(
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  DataT input_data[ITEMS_PER_THREAD];
  DataT top_values[ITEMS_PER_THREAD];
  int64_t top_indices[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load input data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    input_data[i] = input[global_offset + i];
  }

  // Call blockTopk with configurable block size template parameter
  nvf::topk::blockTopK<BLOCK_SIZE, 1, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

// Multi-dimensional 2D test kernel (4x2 block)
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void multiDim2dTopkTestKernel(
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  DataT input_data[ITEMS_PER_THREAD];
  DataT top_values[ITEMS_PER_THREAD];
  int64_t top_indices[ITEMS_PER_THREAD];

  // 2D thread indexing
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load input data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    input_data[i] = input[global_offset + i];
  }

  // Call blockTopk with 2D configuration
  nvf::topk::blockTopK<4, 2, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

// Multi-dimensional 3D test kernel (2x2x2 block)
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void multiDim3dTopkTestKernel(
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  DataT input_data[ITEMS_PER_THREAD];
  DataT top_values[ITEMS_PER_THREAD];
  int64_t top_indices[ITEMS_PER_THREAD];

  // 3D thread indexing
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load input data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    input_data[i] = input[global_offset + i];
  }

  // Call blockTopk with 3D configuration
  nvf::topk::blockTopK<2, 2, 2, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

//============================================================================
// Launch function implementations
//============================================================================

template <typename DataT, int ITEMS_PER_THREAD>
void launchBasicTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int block_size,
    int k,
    bool largest) {
  // Validate that k does not exceed total number of elements
  const int total_elements = block_size * ITEMS_PER_THREAD;
  ASSERT_LE(k, total_elements)
      << "k=" << k << " must be <= total_elements=" << total_elements;

  // Use consolidated kernel with appropriate block size template parameter
  if (block_size == 4) {
    basicTopkTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else if (block_size == 32) {
    basicTopkTestKernel<32, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else if (block_size == 64) {
    basicTopkTestKernel<64, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else {
    // Default to 4-thread configuration to match test
    basicTopkTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, 4, 0, stream>>>(input, output_values, output_indices, k, largest);
  }
}

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim2dTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  dim3 block_dim(4, 2, 1); // 2D block: 4x2 = 8 threads

  // Validate that k does not exceed total number of elements
  const int total_elements =
      block_dim.x * block_dim.y * block_dim.z * ITEMS_PER_THREAD;
  ASSERT_LE(k, total_elements)
      << "k=" << k << " must be <= total_elements=" << total_elements;
  multiDim2dTopkTestKernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(
          input, output_values, output_indices, k, largest);
}

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim3dTopkTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  dim3 block_dim(2, 2, 2); // 3D block: 2x2x2 = 8 threads

  // Validate that k does not exceed total number of elements
  const int total_elements =
      block_dim.x * block_dim.y * block_dim.z * ITEMS_PER_THREAD;
  ASSERT_LE(k, total_elements)
      << "k=" << k << " must be <= total_elements=" << total_elements;
  multiDim3dTopkTestKernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(
          input, output_values, output_indices, k, largest);
}

//============================================================================
// Explicit template instantiations for common types
//============================================================================

// Macros to simplify template instantiations
#define INSTANTIATE_BASIC_TOPK_LAUNCHER(DataT)                 \
  template void launchBasicTopkTestKernel<DataT, 2>(           \
      cudaStream_t, DataT*, DataT*, int64_t*, int, int, bool); \
  template void launchBasicTopkTestKernel<DataT, 4>(           \
      cudaStream_t, DataT*, DataT*, int64_t*, int, int, bool);

#define INSTANTIATE_BASIC_TOPK_LAUNCHER_SINGLE(DataT, ItemsPerThread) \
  template void launchBasicTopkTestKernel<DataT, ItemsPerThread>(     \
      cudaStream_t, DataT*, DataT*, int64_t*, int, int, bool);

#define INSTANTIATE_MULTIDIM_TOPK_LAUNCHER(DataT)         \
  template void launchMultiDim2dTopkTestKernel<DataT, 2>( \
      cudaStream_t, DataT*, DataT*, int64_t*, int, bool); \
  template void launchMultiDim3dTopkTestKernel<DataT, 2>( \
      cudaStream_t, DataT*, DataT*, int64_t*, int, bool);

// Basic topk test kernel instantiations
INSTANTIATE_BASIC_TOPK_LAUNCHER(float)
INSTANTIATE_BASIC_TOPK_LAUNCHER(double)
INSTANTIATE_BASIC_TOPK_LAUNCHER(int)
INSTANTIATE_BASIC_TOPK_LAUNCHER(int64_t)

// BFloat16 only supports 2 items per thread
INSTANTIATE_BASIC_TOPK_LAUNCHER_SINGLE(__nv_bfloat16, 2)

// Multi-dimensional test kernel instantiations (only support 2 items per
// thread)
INSTANTIATE_MULTIDIM_TOPK_LAUNCHER(float)

// Clean up macros to avoid polluting global namespace
#undef INSTANTIATE_BASIC_TOPK_LAUNCHER
#undef INSTANTIATE_BASIC_TOPK_LAUNCHER_SINGLE
#undef INSTANTIATE_MULTIDIM_TOPK_LAUNCHER

} // namespace nvfuser
