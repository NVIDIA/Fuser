#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// NEG_INFINITY constant
#ifndef NEG_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#endif

// nvfuser-style block dimension structure
struct DefaultBlockDim {
  const uint32_t x, y, z;
  __device__ DefaultBlockDim() : x(blockDim.x), y(blockDim.y), z(blockDim.z) {}
  __device__ operator dim3() const {
    return blockDim;
  }
};

// Index utilities for blockReduce
namespace index_utils {
template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE>
__device__ bool maskedIsZero(const dim3& idx) {
  return (!X_REDUCE || idx.x == 0) && (!Y_REDUCE || idx.y == 0) &&
      (!Z_REDUCE || idx.z == 0);
}

template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename BlockDimT>
__device__ unsigned int maskedSize(const BlockDimT& block_dim) {
  return (X_REDUCE ? block_dim.x : 1) * (Y_REDUCE ? block_dim.y : 1) *
      (Z_REDUCE ? block_dim.z : 1);
}

template <bool X_REDUCE, bool Y_REDUCE, bool Z_REDUCE, typename BlockDimT>
__device__ unsigned int maskedOffset(
    const dim3& idx,
    const BlockDimT& block_dim) {
  unsigned int offset = 0;
  if (X_REDUCE)
    offset += idx.x;
  if (Y_REDUCE)
    offset += idx.y * (X_REDUCE ? block_dim.x : 1);
  if (Z_REDUCE)
    offset +=
        idx.z * (X_REDUCE ? block_dim.x : 1) * (Y_REDUCE ? block_dim.y : 1);
  return offset;
}
} // namespace index_utils

// Simplified blockReduce implementation
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func,
    typename BlockDimT>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val,
    BlockDimT block_dim) {
  // Initialize shared memory offset
  unsigned int smem_offset = threadIdx.x + threadIdx.y * block_dim.x +
      threadIdx.z * block_dim.x * block_dim.y;

  // Store input value to shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  __syncthreads();

  // Perform reduction in shared memory
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(block_dim);

  for (unsigned int stride = reduction_size / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + stride]);
    }
    __syncthreads();
  }

  // Write result
  if (index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx) &&
      write_pred) {
    out = shared_mem[0];
  }
}

// Forward declarations for nvfuser types
struct __e2m1 {
  uint8_t data;
  __host__ __device__ __e2m1() = default;
};

struct __e4m3 {
  uint8_t __x;
  __host__ __device__ __e4m3() = default;
  __host__ __device__ __e4m3(uint8_t x) : __x(x) {}
  __host__ __device__ uint8_t raw() const {
    return __x;
  }
};

// Conversion function declarations (would be implemented in nvfuser runtime)
__device__ __e2m1 __float2e2m1(float f);
__device__ __e4m3 __float2e4m3(float f);

// Simple implementation of __e4m32float for testing
__device__ float __e4m32float(__e4m3 x) {
  // Simple approximation: reverse the conversion done in __float2e4m3
  // This is a placeholder - the actual implementation would be in nvfuser
  // runtime
  return static_cast<float>(x.raw()) / 255.0f;
}

// Clamp function from nvfuser helpers.cu - avoiding conflicts with CUDA std lib
__device__ float clamp(float x, float minv, float maxv) {
  return fminf(fmaxf(x, minv), maxv);
}

__device__ double clamp(double x, double minv, double maxv) {
  return fmin(fmax(x, minv), maxv);
}

/**
 * Templated CUDA kernel for float to FP4/FP8 conversion with vectorized loads
 *
 * Template Parameters:
 * - BLOCK_DIM_X: Block dimension in X direction (must be 4 for vectorized
 * loads)
 * - BLOCK_DIM_Y: Block dimension in Y direction
 *
 * Parameters:
 * - input: Pointer to 2D array where inner dimension is fixed to 16
 * - output_e2m1: Output pointer for __e2m1 (FP4 E2M1) format
 * - output_e4m3: Output pointer for __e4m3 (FP8 E4M3) format
 * - num_rows: Number of rows in the input (outer dimension)
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void float_to_fp_conversion_kernel(
    const float (*__restrict__ input)[16],
    __e2m1* __restrict__ output_e2m1,
    __e4m3* __restrict__ output_e4m3,
    int num_rows) {
  // Ensure BLOCK_DIM_X is 4 for vectorized loads
  static_assert(
      BLOCK_DIM_X == 4,
      "BLOCK_DIM_X must be 4 for vectorized loads of 16 elements");

  // Calculate thread indices
  const int tid_x = threadIdx.x; // 0, 1, 2, or 3
  const int tid_y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

  const int tile_id_to_process = blockIdx.x * blockDim.y + threadIdx.y;
  const int memory_offset_of_tile = tile_id_to_process * 16;

  // Create an array of 4 floats for vectorized load
  float vec4[4];
  for (int i = 0; i < 4; ++i) {
    vec4[i] = input[tile_id_to_process][4 * tid_x + i];
  }
  // Calculate the max of the values in vec4
  float local_max = vec4[0];
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    local_max = fmaxf(local_max, vec4[i]);
  }

  // Block-level reduction using nvfuser-style blockReduce
  // Allocate shared memory for the reduction
  extern __shared__ float shared_mem[];

  // Perform block-wide maximum reduction across threads
  float block_max = local_max;
  blockReduce<true, false, false, true>(
      block_max,
      local_max,
      [](float& a, float b) { a = fmaxf(a, b); },
      shared_mem,
      true,
      true,
      static_cast<float>(NEG_INFINITY),
      DefaultBlockDim());

  // Apply scaling and clamping using nvfuser-style clamp function
  float scaled_max = block_max / 6.000000000e+00f;
  float clamped_max = clamp(
      scaled_max, 1.562500000e-02f, 4.480000000e+02f); // Clamp between 0 and 1
  __e4m3 clamped_max_fp8 = __float2e4m3(clamped_max);

  // Convert back from FP8 to float using __e4m32float
  float clamped_max_converted = __e4m32float(clamped_max_fp8);


  // Write back the clamped max value if this is thread 0 in X dimension
  if (tid_x == 0) {
    output_e4m3[tile_id_to_process] = clamped_max_fp8;
  }


  // Calculate the row index this thread will process
  const int row_idx = tid_y * gridDim.x + blockIdx.x;

  // Early exit if thread is out of bounds
  if (row_idx >= num_rows) {
    return;
  }

  // Each thread in X dimension loads 4 consecutive elements
  // Thread 0: elements 0-3, Thread 1: elements 4-7, Thread 2: elements 8-11,
  // Thread 3: elements 12-15
  const int start_col = tid_x * 4;
  const int base_offset = row_idx * 16;

// Vectorized load: each thread loads 4 elements
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int col_idx = start_col + i;
    const int global_idx = base_offset + col_idx;
    const float input_val = input[row_idx][col_idx];

    // Convert to FP4 E2M1 format
    output_e2m1[global_idx] = __float2e2m1(input_val);

    // Convert to FP8 E4M3 format
    output_e4m3[global_idx] = __float2e4m3(input_val);
  }
}

/**
 * Host function to launch the templated kernel
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
cudaError_t launch_float_to_fp_kernel(
    const float (*d_input)[16],
    __e2m1* d_output_e2m1,
    __e4m3* d_output_e4m3,
    int num_rows,
    cudaStream_t stream = 0) {
  // Validate template parameters
  static_assert(BLOCK_DIM_X == 4, "BLOCK_DIM_X must be 4 for vectorized loads");
  static_assert(BLOCK_DIM_Y > 0, "BLOCK_DIM_Y must be positive");
  static_assert(
      BLOCK_DIM_X * BLOCK_DIM_Y <= 1024,
      "Total threads per block cannot exceed 1024");

  // Calculate grid dimensions
  // Each block in X processes one row (with 4 threads handling 16 elements)
  // Each block in Y processes BLOCK_DIM_Y rows
  const int grid_x =
      1; // Only need 1 block in X since 4 threads handle all 16 elements
  const int grid_y = (num_rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

  dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid_dim(grid_x, grid_y);

  // Calculate shared memory size for blockReduce
  const int shared_mem_size = BLOCK_DIM_X * BLOCK_DIM_Y * sizeof(float);

  // Launch the kernel with shared memory
  float_to_fp_conversion_kernel<BLOCK_DIM_X, BLOCK_DIM_Y>
      <<<grid_dim, block_dim, shared_mem_size, stream>>>(
          d_input, d_output_e2m1, d_output_e4m3, num_rows);

  return cudaGetLastError();
}

// Convenient wrapper function for common use cases
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
class FpConversionKernel {
 public:
  static cudaError_t execute(
      const float (*h_input)[16],
      int num_rows,
      __e2m1** h_output_e2m1 = nullptr,
      __e4m3** h_output_e4m3 = nullptr,
      cudaStream_t stream = 0) {
    const int INNER_DIM = 16;
    const int total_elements = num_rows * INNER_DIM;

    // Allocate device memory
    float (*d_input)[16];
    __e2m1* d_output_e2m1;
    __e4m3* d_output_e4m3;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(&d_input, total_elements * sizeof(float));
    if (err != cudaSuccess)
      return err;

    err = cudaMalloc(&d_output_e2m1, total_elements * sizeof(__e2m1));
    if (err != cudaSuccess) {
      cudaFree(d_input);
      return err;
    }

    err = cudaMalloc(&d_output_e4m3, total_elements * sizeof(__e4m3));
    if (err != cudaSuccess) {
      cudaFree(d_input);
      cudaFree(d_output_e2m1);
      return err;
    }

    // Copy input to device
    err = cudaMemcpy(
        d_input,
        h_input,
        total_elements * sizeof(float),
        cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_input);
      cudaFree(d_output_e2m1);
      cudaFree(d_output_e4m3);
      return err;
    }

    // Launch kernel
    err = launch_float_to_fp_kernel<BLOCK_DIM_X, BLOCK_DIM_Y>(
        d_input, d_output_e2m1, d_output_e4m3, num_rows, stream);

    if (err != cudaSuccess) {
      cudaFree(d_input);
      cudaFree(d_output_e2m1);
      cudaFree(d_output_e4m3);
      return err;
    }

    // Copy results back if requested
    if (h_output_e2m1 != nullptr) {
      *h_output_e2m1 = new __e2m1[total_elements];
      err = cudaMemcpy(
          *h_output_e2m1,
          d_output_e2m1,
          total_elements * sizeof(__e2m1),
          cudaMemcpyDeviceToHost);
    }

    if (h_output_e4m3 != nullptr) {
      *h_output_e4m3 = new __e4m3[total_elements];
      err = cudaMemcpy(
          *h_output_e4m3,
          d_output_e4m3,
          total_elements * sizeof(__e4m3),
          cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output_e2m1);
    cudaFree(d_output_e4m3);

    return err;
  }
};

// Example usage macros for different configurations (BLOCK_DIM_X must be 4)
#define LAUNCH_KERNEL_4x1(input, rows, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 1>::execute(input, rows, e2m1_out, e4m3_out, stream)

#define LAUNCH_KERNEL_4x4(input, rows, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 4>::execute(input, rows, e2m1_out, e4m3_out, stream)

#define LAUNCH_KERNEL_4x8(input, rows, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 8>::execute(input, rows, e2m1_out, e4m3_out, stream)
