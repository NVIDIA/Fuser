#pragma once

#include <cuda_runtime.h>

namespace nvfp4_types {
// Forward declarations for nvfuser types
struct alignas(1) __e2m1 {
  uint8_t data;
  __host__ __device__ __e2m1() = default;
  __host__ __device__ __e2m1(const __e2m1& other) = default;
  __host__ __device__ __e2m1& operator=(const __e2m1& other) = default;
  __host__ __device__ volatile __e2m1& operator=(const __e2m1& other) volatile {
    data = other.data;
    return *this;
  }
};

struct __align__(1) __e4m3 {
  uint8_t __x;
  __host__ __device__ __e4m3() = default;
  __host__ __device__ __e4m3(uint8_t x) : __x(x) {}
  __host__ __device__ uint8_t raw() const {
    return __x;
  }
};

} // namespace nvfp4_types

// Function declaration for FP conversion kernel execution
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
cudaError_t fp_conversion_kernel_execute(
    const float* d_input,
    const int INNER_DIM,
    int total_elements,
    nvfp4_types::__e2m1** h_output_e2m1 = nullptr,
    nvfp4_types::__e4m3** h_output_e4m3 = nullptr);
