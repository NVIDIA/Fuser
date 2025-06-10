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

namespace nvfuser {

//============================================================================
// Fixed template parameter kernels (avoiding dynamic template instantiation)
//============================================================================

// Basic topk test kernel with fixed block size 4 (test configuration)
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void basic_topk_test_kernel_4(
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

  // Call blockTopk with fixed template parameters (4x1x1 block)
  nvf::topk::blockTopK<4, 1, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

// Basic topk test kernel with fixed block size 32
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void basic_topk_test_kernel_32(
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

  // Call blockTopk with fixed template parameters
  nvf::topk::blockTopK<32, 1, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

// Basic topk test kernel with fixed block size 64
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void basic_topk_test_kernel_64(
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

  // Call blockTopk with fixed template parameters
  nvf::topk::blockTopK<64, 1, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      top_values, top_indices, input_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = top_values[i];
    output_indices[global_offset + i] = top_indices[i];
  }
}

// Multi-dimensional 2D test kernel (4x2 block)
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void multi_dim_2d_topk_test_kernel(
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
__global__ void multi_dim_3d_topk_test_kernel(
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

// BFloat16 specialized kernel (fixed 4x1 block, 2 items per thread)
template <int ITEMS_PER_THREAD>
__global__ void bfloat16_topk_test_kernel(
    __nv_bfloat16* input,
    __nv_bfloat16* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  __nv_bfloat16 thread_data[ITEMS_PER_THREAD];
  __nv_bfloat16 thread_values[ITEMS_PER_THREAD];
  int64_t thread_indices[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_data[i] = input[global_offset + i];
  }

  nvf::topk::blockTopK<4, 1, 1, 0, 0, 0, __nv_bfloat16, ITEMS_PER_THREAD>(
      thread_values, thread_indices, thread_data, k, largest, true, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_values[global_offset + i] = thread_values[i];
    output_indices[global_offset + i] = thread_indices[i];
  }
}

//============================================================================
// Launch function implementations
//============================================================================

template <typename DataT, int ITEMS_PER_THREAD>
void launch_basic_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int block_size,
    int k,
    bool largest) {
  // Use specific kernel based on block size and items per thread
  if (block_size == 4) {
    basic_topk_test_kernel_4<DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else if (block_size == 32) {
    basic_topk_test_kernel_32<DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else if (block_size == 64) {
    basic_topk_test_kernel_64<DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output_values, output_indices, k, largest);
  } else {
    // Default to 4-thread configuration to match test
    basic_topk_test_kernel_4<DataT, ITEMS_PER_THREAD>
        <<<1, 4, 0, stream>>>(input, output_values, output_indices, k, largest);
  }
}

template <typename DataT, int ITEMS_PER_THREAD>
void launch_multi_dim_2d_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  dim3 block_dim(4, 2, 1); // 2D block: 4x2 = 8 threads
  multi_dim_2d_topk_test_kernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(
          input, output_values, output_indices, k, largest);
}

template <typename DataT, int ITEMS_PER_THREAD>
void launch_multi_dim_3d_topk_test_kernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  dim3 block_dim(2, 2, 2); // 3D block: 2x2x2 = 8 threads
  multi_dim_3d_topk_test_kernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(
          input, output_values, output_indices, k, largest);
}

template <int ITEMS_PER_THREAD>
void launch_bfloat16_topk_test_kernel(
    cudaStream_t stream,
    __nv_bfloat16* input,
    __nv_bfloat16* output_values,
    int64_t* output_indices,
    int k,
    bool largest) {
  bfloat16_topk_test_kernel<ITEMS_PER_THREAD>
      <<<1, 4, 0, stream>>>(input, output_values, output_indices, k, largest);
}

//============================================================================
// Explicit template instantiations for common types
//============================================================================

// Basic topk test kernel instantiations
template void launch_basic_topk_test_kernel<float, 2>(
    cudaStream_t,
    float*,
    float*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<float, 4>(
    cudaStream_t,
    float*,
    float*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<double, 2>(
    cudaStream_t,
    double*,
    double*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<double, 4>(
    cudaStream_t,
    double*,
    double*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<int, 2>(
    cudaStream_t,
    int*,
    int*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<int, 4>(
    cudaStream_t,
    int*,
    int*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<int64_t, 2>(
    cudaStream_t,
    int64_t*,
    int64_t*,
    int64_t*,
    int,
    int,
    bool);

template void launch_basic_topk_test_kernel<int64_t, 4>(
    cudaStream_t,
    int64_t*,
    int64_t*,
    int64_t*,
    int,
    int,
    bool);

// Multi-dimensional test kernel instantiations (only support 2 items per
// thread)
template void launch_multi_dim_2d_topk_test_kernel<float, 2>(
    cudaStream_t,
    float*,
    float*,
    int64_t*,
    int,
    bool);

template void launch_multi_dim_3d_topk_test_kernel<float, 2>(
    cudaStream_t,
    float*,
    float*,
    int64_t*,
    int,
    bool);

// BFloat16 test kernel instantiation (only support 2 items per thread)
template void launch_bfloat16_topk_test_kernel<2>(
    cudaStream_t,
    __nv_bfloat16*,
    __nv_bfloat16*,
    int64_t*,
    int,
    bool);

} // namespace nvfuser
