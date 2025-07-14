// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Define nvfuser_index_t at global scope for runtime files
using nvfuser_index_t = int64_t;

// nvFuser headers
#include <tests/cpp/argsort_test_helper.h>

// Need to be included before argsort because of the dependency
// from argsort
namespace nvf {
#include <runtime/index_utils.cu>
} // namespace nvf

#include <runtime/cub_utils.cu>

#include <runtime/argsort.cu>

// Standard C++ headers
#include <cstdint>

// CUDA headers
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace nvfuser {

// Test kernels for different configurations
template <typename DataT, int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void basicArgsortTestKernel(
    DataT* input,
    int64_t* output_indices,
    bool descending) {
  DataT thread_data[ITEMS_PER_THREAD];
  int64_t thread_indices[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_data[i] = input[global_offset + i];
  }

  // Perform block-parallel argsort using the actual runtime implementation
  nvf::argsort::
      blockArgsort<BLOCK_SIZE, 1, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
          thread_indices, thread_data, descending, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_indices[global_offset + i] = thread_indices[i];
  }
}

// Multi-dimensional test kernels
template <typename DataT, int ITEMS_PER_THREAD>
__global__ void multiDim2dArgsortTestKernel(
    DataT* input,
    int64_t* output_indices,
    bool descending) {
  DataT thread_data[ITEMS_PER_THREAD];
  int64_t thread_indices[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_data[i] = input[global_offset + i];
  }

  // Test 2D block: 4x2x1 (8 threads total) using actual runtime implementation
  nvf::argsort::blockArgsort<4, 2, 1, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      thread_indices, thread_data, descending, blockDim);

  // Store results back
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_indices[global_offset + i] = thread_indices[i];
  }
}

template <typename DataT, int ITEMS_PER_THREAD>
__global__ void multiDim3dArgsortTestKernel(
    DataT* input,
    int64_t* output_indices,
    bool descending) {
  DataT thread_data[ITEMS_PER_THREAD];
  int64_t thread_indices[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    thread_data[i] = input[global_offset + i];
  }

  // Test 3D block: 2x2x2 (8 threads total) using actual runtime implementation
  nvf::argsort::blockArgsort<2, 2, 2, 0, 0, 0, DataT, ITEMS_PER_THREAD>(
      thread_indices, thread_data, descending, blockDim);

  // Store results back
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output_indices[global_offset + i] = thread_indices[i];
  }
}

// Template launcher functions
template <typename DataT>
void launchBasicArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending) {
// Use a macro to reduce repetition for all combinations
#define LAUNCH_KERNEL(BS, IPT)                                     \
  if (block_size == BS && items_per_thread == IPT) {               \
    basicArgsortTestKernel<DataT, BS, IPT>                         \
        <<<1, BS, 0, stream>>>(input, output_indices, descending); \
    return;                                                        \
  }

  // Block size 4
  LAUNCH_KERNEL(4, 1)
  LAUNCH_KERNEL(4, 2)
  LAUNCH_KERNEL(4, 3)
  LAUNCH_KERNEL(4, 4)
  LAUNCH_KERNEL(4, 5)
  LAUNCH_KERNEL(4, 8)

  // Original configurations for other tests
  LAUNCH_KERNEL(16, 4)

  // Block size 32
  LAUNCH_KERNEL(32, 1)
  LAUNCH_KERNEL(32, 2)
  LAUNCH_KERNEL(32, 3)
  LAUNCH_KERNEL(32, 4)
  LAUNCH_KERNEL(32, 5)

  // Block size 64
  LAUNCH_KERNEL(64, 1)
  LAUNCH_KERNEL(64, 2)
  LAUNCH_KERNEL(64, 3)
  LAUNCH_KERNEL(64, 4)
  LAUNCH_KERNEL(64, 5)
  LAUNCH_KERNEL(64, 8) // Keep existing configuration

  // Block size 128
  LAUNCH_KERNEL(128, 1)
  LAUNCH_KERNEL(128, 2)
  LAUNCH_KERNEL(128, 3)
  LAUNCH_KERNEL(128, 4)
  LAUNCH_KERNEL(128, 5)
  LAUNCH_KERNEL(128, 8) // Keep existing configuration

  // Block size 256
  LAUNCH_KERNEL(256, 1)
  LAUNCH_KERNEL(256, 2)
  LAUNCH_KERNEL(256, 3)
  LAUNCH_KERNEL(256, 4)
  LAUNCH_KERNEL(256, 5)
  LAUNCH_KERNEL(256, 8) // Keep existing configuration

  // Block size 512 (keep existing)
  LAUNCH_KERNEL(512, 8)

#undef LAUNCH_KERNEL

  // If we get here, the combination is not supported
  FAIL() << "Unsupported block_size/items_per_thread combination";
}

template <typename DataT>
void launchMultiDim2dArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending) {
  dim3 block_2d(4, 2, 1);
  if (items_per_thread == 2) {
    multiDim2dArgsortTestKernel<DataT, 2>
        <<<1, block_2d, 0, stream>>>(input, output_indices, descending);
  } else {
    FAIL() << "Unsupported items_per_thread for 2D block";
  }
}

template <typename DataT>
void launchMultiDim3dArgsortTestKernel(
    cudaStream_t stream,
    DataT* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending) {
  dim3 block_3d(2, 2, 2);
  if (items_per_thread == 2) {
    multiDim3dArgsortTestKernel<DataT, 2>
        <<<1, block_3d, 0, stream>>>(input, output_indices, descending);
  } else {
    FAIL() << "Unsupported items_per_thread for 3D block";
  }
}

// Explicit template instantiations
template void launchBasicArgsortTestKernel<float>(
    cudaStream_t stream,
    float* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template void launchBasicArgsortTestKernel<double>(
    cudaStream_t stream,
    double* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template void launchBasicArgsortTestKernel<int>(
    cudaStream_t stream,
    int* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template void launchBasicArgsortTestKernel<int64_t>(
    cudaStream_t stream,
    int64_t* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template void launchBasicArgsortTestKernel<__nv_bfloat16>(
    cudaStream_t stream,
    __nv_bfloat16* input,
    int64_t* output_indices,
    int block_size,
    int items_per_thread,
    bool descending);

template void launchMultiDim2dArgsortTestKernel<float>(
    cudaStream_t stream,
    float* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending);

template void launchMultiDim3dArgsortTestKernel<float>(
    cudaStream_t stream,
    float* input,
    int64_t* output_indices,
    int items_per_thread,
    bool descending);

} // namespace nvfuser
