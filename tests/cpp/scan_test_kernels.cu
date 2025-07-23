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

#include <runtime/cub_utils.cu>

#include <runtime/scan.cu>

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



// Custom data type for discount scan following CUB pattern
template <typename DataT>
struct DiscountScanTuple {
  DataT value;                // The partial result of the discounted sum
  DataT discount_accumulator; // The accumulated discount factor

  __device__ __forceinline__ DiscountScanTuple() = default;
  __device__ __forceinline__ DiscountScanTuple(DataT v, DataT d) : value(v), discount_accumulator(d) {}
};

// Custom associative operator for discount scan following CUB pattern
template <typename DataT>
struct DiscountScanOp {
  __device__ __forceinline__ DiscountScanTuple<DataT> operator()(
      const DiscountScanTuple<DataT>& a, 
      const DiscountScanTuple<DataT>& b) const {
    DiscountScanTuple<DataT> result;
    // Combine segments: new_value = a.value * b.discount + b.value
    result.value = a.value * b.discount_accumulator + b.value;
    // Accumulate discount factors
    result.discount_accumulator = a.discount_accumulator * b.discount_accumulator;
    return result;
  }
};

// Specialized discount scan kernel using CUB BlockScan with custom operator
template <
    int BLOCK_SIZE,
    typename DataT,
    int ITEMS_PER_THREAD>
__device__ void discountScanTestKernelTyped(
    DataT* input,
    DataT* discount,
    DataT* output,
    DataT init_value) {
  
  using TupleT = DiscountScanTuple<DataT>;
  
  // Input data arrays
  DataT input_data[ITEMS_PER_THREAD];
  DataT discount_data[ITEMS_PER_THREAD];
  TupleT tuple_data[ITEMS_PER_THREAD];
  TupleT scan_output[ITEMS_PER_THREAD];

  int thread_id = threadIdx.x;
  int global_offset = thread_id * ITEMS_PER_THREAD;

  // Load input and discount data for this thread
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    input_data[i] = input[global_offset + i];
    discount_data[i] = discount[global_offset + i];
    // Create tuple: (value, discount_factor)
    tuple_data[i] = TupleT(input_data[i], discount_data[i]);
  }

  // CUB BlockScan setup for custom tuple type
  using BlockScan = cub::BlockScan<TupleT, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Custom discount scan operator
  DiscountScanOp<DataT> scan_op;

  // Perform inclusive scan with custom operator
  BlockScan(temp_storage).InclusiveScan(tuple_data, scan_output, scan_op);

  // Extract values from tuples and store to global memory
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    output[global_offset + i] = scan_output[i].value;
  }
}

// Generic binary operation dispatcher to eliminate switch statement duplication
template <typename DataT, typename KernelFunc>
__device__ void dispatchBinaryOp(
    DataT* input,
    DataT* output,
    DataT init_value,
    const ScanBinaryOpType binary_op_type,
    KernelFunc kernel_func) {
  switch (binary_op_type) {
    case ScanBinaryOpType::Add:
      kernel_func(input, output, init_value, AddOp<DataT>{});
      break;
    case ScanBinaryOpType::Max:
      kernel_func(input, output, init_value, MaxOp<DataT>{});
      break;
    case ScanBinaryOpType::Min:
      kernel_func(input, output, init_value, MinOp<DataT>{});
      break;
    case ScanBinaryOpType::Mul:
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
    ScanBinaryOpType binary_op_type) {
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

// Discount scan kernel wrapper
template <int BLOCK_SIZE, typename DataT, int ITEMS_PER_THREAD>
__global__ void discountScanTestKernel(
    DataT* input,
    DataT* discount,
    DataT* output,
    DataT init_value) {
  discountScanTestKernelTyped<BLOCK_SIZE, DataT, ITEMS_PER_THREAD>(
      input, discount, output, init_value);
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
    ScanBinaryOpType binary_op_type) {
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
void launchDiscountScanTestKernel(
    cudaStream_t stream,
    DataT* input,
    DataT* discount,
    DataT* output,
    DataT init_value,
    int block_size) {
  // Dispatch based on block size
  if (block_size == 4) {
    discountScanTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, discount, output, init_value);
  } else if (block_size == 8) {
    discountScanTestKernel<8, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, discount, output, init_value);
  } else if (block_size == 16) {
    discountScanTestKernel<16, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, discount, output, init_value);
  } else if (block_size == 32) {
    discountScanTestKernel<32, DataT, ITEMS_PER_THREAD>
        <<<1, block_size, 0, stream>>>(
            input, discount, output, init_value);
  } else {
    // Default to block size 4 for unsupported sizes
    discountScanTestKernel<4, DataT, ITEMS_PER_THREAD>
        <<<1, 4, 0, stream>>>(input, discount, output, init_value);
  }
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
    ScanBinaryOpType binary_op_type);
template void launchBasicScanTestKernel<float, 2>(
    cudaStream_t stream,
    float* input,
    float* output,
    float init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);
template void launchBasicScanTestKernel<float, 4>(
    cudaStream_t stream,
    float* input,
    float* output,
    float init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);

// double instantiations
template void launchBasicScanTestKernel<double, 1>(
    cudaStream_t stream,
    double* input,
    double* output,
    double init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);
template void launchBasicScanTestKernel<double, 2>(
    cudaStream_t stream,
    double* input,
    double* output,
    double init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);

// int instantiations
template void launchBasicScanTestKernel<int, 1>(
    cudaStream_t stream,
    int* input,
    int* output,
    int init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);
template void launchBasicScanTestKernel<int, 2>(
    cudaStream_t stream,
    int* input,
    int* output,
    int init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);

// int64_t instantiations
template void launchBasicScanTestKernel<int64_t, 1>(
    cudaStream_t stream,
    int64_t* input,
    int64_t* output,
    int64_t init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);
template void launchBasicScanTestKernel<int64_t, 2>(
    cudaStream_t stream,
    int64_t* input,
    int64_t* output,
    int64_t init_value,
    int block_size,
    ScanBinaryOpType binary_op_type);

//============================================================================
// Discount scan template instantiations
//============================================================================

// float instantiations
template void launchDiscountScanTestKernel<float, 1>(
    cudaStream_t stream,
    float* input,
    float* discount,
    float* output,
    float init_value,
    int block_size);
template void launchDiscountScanTestKernel<float, 2>(
    cudaStream_t stream,
    float* input,
    float* discount,
    float* output,
    float init_value,
    int block_size);
template void launchDiscountScanTestKernel<float, 4>(
    cudaStream_t stream,
    float* input,
    float* discount,
    float* output,
    float init_value,
    int block_size);

// double instantiations
template void launchDiscountScanTestKernel<double, 1>(
    cudaStream_t stream,
    double* input,
    double* discount,
    double* output,
    double init_value,
    int block_size);
template void launchDiscountScanTestKernel<double, 2>(
    cudaStream_t stream,
    double* input,
    double* discount,
    double* output,
    double init_value,
    int block_size);

} // namespace nvfuser
