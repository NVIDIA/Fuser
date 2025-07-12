// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Define nvfuser_index_t at global scope for runtime files
using nvfuser_index_t = int64_t;

#include <tests/cpp/scan_test_helper.h>

// Need to be included before scan because of dependencies
namespace nvf {
#include <runtime/index_utils.cu>
} // namespace nvf

#include <runtime/scan.cu>

// Include BinaryOpType enum definition - placed after runtime includes
// to avoid dependency issues while ensuring enum values are available
#include <csrc/type.h>

// Standard C++ headers
#include <cstdint>

// CUDA headers
#include <cuda_runtime.h>

// Test framework headers
#include <gtest/gtest.h>

namespace nvfuser {

// Scan operation function objects to avoid lambda return type deduction issues
template <typename DataT>
struct AddOp {
  __device__ DataT operator()(DataT a, DataT b) const {
    return a + b;
  }
};

template <typename DataT>
struct MaxOp {
  __device__ DataT operator()(DataT a, DataT b) const {
    return (a > b) ? a : b;
  }
};

template <typename DataT>
struct MinOp {
  __device__ DataT operator()(DataT a, DataT b) const {
    return (a < b) ? a : b;
  }
};

template <typename DataT>
struct MulOp {
  __device__ DataT operator()(DataT a, DataT b) const {
    return a * b;
  }
};

// Generic binary operation dispatcher to eliminate switch statement duplication
template <typename DataT, typename KernelFunc>
__device__ void dispatchBinaryOp(
    DataT* input,
    DataT* output,
    DataT init_value,
    BinaryOpType binary_op_type,
    KernelFunc kernel_func) {
  switch (binary_op_type) {
    case BinaryOpType::Add:
      kernel_func(input, output, init_value, AddOp<DataT>{});
      break;
    case BinaryOpType::Max:
      kernel_func(input, output, init_value, MaxOp<DataT>{});
      break;
    case BinaryOpType::Min:
      kernel_func(input, output, init_value, MinOp<DataT>{});
      break;
    case BinaryOpType::Mul:
      kernel_func(input, output, init_value, MulOp<DataT>{});
      break;
    default: // Default to Add
      kernel_func(input, output, init_value, AddOp<DataT>{});
      break;
  }
}

// Template-based scan test kernel with specific operation type
template <
    int BLOCK_SIZE,
    typename DataT,
    int ITEMS_PER_THREAD,
    typename ScanOpT>
__device__ void basicScanTestKernelTyped(
    DataT* input,
    DataT* output,
    DataT init_value,
    ScanOpT scan_op) {
  DataT input_data[ITEMS_PER_THREAD];
  DataT scan_output[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load input data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    input_data[i] = input[global_offset + i];
  }

  // Call blockScan with configurable block size template parameter and
  // operation
  nvf::scan::blockScan<BLOCK_SIZE, 1, 1, 0, 1, 1, DataT, ITEMS_PER_THREAD>(
      scan_output, input_data, init_value, scan_op, blockDim);

  // Store results back to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output[global_offset + i] = scan_output[i];
  }
}

// Wrapper kernel that dispatches to the appropriate typed kernel
template <int BLOCK_SIZE, typename DataT, int ITEMS_PER_THREAD>
__global__ void basicScanTestKernel(
    DataT* input,
    DataT* output,
    DataT init_value,
    BinaryOpType binary_op_type) {
  // Use lambda to capture template parameters and create callable for
  // dispatcher
  auto kernel_func = [=] __device__(
                         DataT * in, DataT * out, DataT init, auto scan_op) {
    basicScanTestKernelTyped<BLOCK_SIZE, DataT, ITEMS_PER_THREAD>(
        in, out, init, scan_op);
  };

  dispatchBinaryOp<DataT>(
      input, output, init_value, binary_op_type, kernel_func);
}

//============================================================================
// Launch function implementations
//============================================================================

template <typename DataT, int ITEMS_PER_THREAD>
void launchBasicScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output,
    DataT init_value,
    int block_size,
    BinaryOpType binary_op_type) {
  // Dispatch based on block size
  if (block_size == 4) {
    basicScanTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output, init_value, binary_op_type);
  } else if (block_size == 8) {
    basicScanTestKernel<8, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output, init_value, binary_op_type);
  } else if (block_size == 16) {
    basicScanTestKernel<16, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output, init_value, binary_op_type);
  } else if (block_size == 32) {
    basicScanTestKernel<32, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, output, init_value, binary_op_type);
  } else {
    // Default to block size 4 for unsupported sizes
    basicScanTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, 4, 0, stream>>>(input, output, init_value, binary_op_type);
  }
}

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim2dScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output,
    DataT init_value,
    BinaryOpType binary_op_type) {
  dim3 block_dim(8, 1, 1);
  multiDim2dScanTestKernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(input, output, init_value, binary_op_type);
}

template <typename DataT, int ITEMS_PER_THREAD>
void launchMultiDim3dScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* output,
    DataT init_value,
    BinaryOpType binary_op_type) {
  dim3 block_dim(8, 1, 1);
  multiDim3dScanTestKernel<DataT, ITEMS_PER_THREAD>
      <<<1, block_dim, 0, stream>>>(input, output, init_value, binary_op_type);
}

//============================================================================
// Explicit template instantiations for common types
//============================================================================

// float instantiations
template void launchBasicScanTestKernel<float, 1>(
    cudaStream_t stream,
    float* input,
    float* output,
    float init_value,
    int block_size,
    BinaryOpType binary_op_type);
template void launchBasicScanTestKernel<float, 2>(
    cudaStream_t stream,
    float* input,
    float* output,
    float init_value,
    int block_size,
    BinaryOpType binary_op_type);
template void launchBasicScanTestKernel<float, 4>(
    cudaStream_t stream,
    float* input,
    float* output,
    float init_value,
    int block_size,
    BinaryOpType binary_op_type);

// double instantiations
template void launchBasicScanTestKernel<double, 1>(
    cudaStream_t stream,
    double* input,
    double* output,
    double init_value,
    int block_size,
    BinaryOpType binary_op_type);
template void launchBasicScanTestKernel<double, 2>(
    cudaStream_t stream,
    double* input,
    double* output,
    double init_value,
    int block_size,
    BinaryOpType binary_op_type);

// int instantiations
template void launchBasicScanTestKernel<int, 1>(
    cudaStream_t stream,
    int* input,
    int* output,
    int init_value,
    int block_size,
    BinaryOpType binary_op_type);
template void launchBasicScanTestKernel<int, 2>(
    cudaStream_t stream,
    int* input,
    int* output,
    int init_value,
    int block_size,
    BinaryOpType binary_op_type);

// int64_t instantiations
template void launchBasicScanTestKernel<int64_t, 1>(
    cudaStream_t stream,
    int64_t* input,
    int64_t* output,
    int64_t init_value,
    int block_size,
    BinaryOpType binary_op_type);
template void launchBasicScanTestKernel<int64_t, 2>(
    cudaStream_t stream,
    int64_t* input,
    int64_t* output,
    int64_t init_value,
    int block_size,
    BinaryOpType binary_op_type);

} // namespace nvfuser
