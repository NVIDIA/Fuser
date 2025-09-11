
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
// Include the header for the class declaration
#include "nvfp4_quantization_test_helpers.h"

// Use the types from the nvfp4_types namespace
using nvfp4_types::__e2m1;
using nvfp4_types::__e4m3;

namespace {
// NEG_INFINITY constant
#ifndef NEG_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#endif

template <typename T>
__device__ T abs(T a) {
  return a > 0 ? a : -a;
}

__device__ __half abs(__half a) {
  return __float2half(fabs(__half2float(a)));
}

__device__ float nv_fmax(float a, float b) {
  // check and propagate NaN
  if (a != a) {
    return a;
  } else { // If b is nan, it will be returned in the next line
    return a > b ? a : b;
  }
}

// MaybeVolatile helper for nvfuser-style memory operations
template <typename Type, bool is_volatile>
struct MaybeVolatile;

template <typename Type>
struct MaybeVolatile<Type, true> {
  using type = volatile Type;
};

template <typename Type>
struct MaybeVolatile<Type, false> {
  using type = Type;
};

// Helper function to compute vector size in bits
template <typename scalar_t>
__host__ __device__ constexpr int64_t vecSizeBit(int vec_size) {
  return vec_size * sizeof(scalar_t) * 8;
}

template <>
__host__ __device__ constexpr int64_t vecSizeBit<__e2m1>(int vec_size) {
  return vec_size * 4;
}

// Cache operation types for nvfuser-style loads
enum class CacheOp {
  AllLevels,
  Streaming,
  Global,
};

// Helper function for cached loads
template <typename T, CacheOp cache_op>
__device__ void loadGlobalToLocalCached(void* to, void* from) {
  T* typed_to = reinterpret_cast<T*>(to);
  T* typed_from = reinterpret_cast<T*>(from);
  switch (cache_op) {
    case CacheOp::AllLevels:
      *typed_to = *typed_from;
      break;
    case CacheOp::Streaming:
      *typed_to = __ldcs(typed_from);
      break;
    case CacheOp::Global:
      *typed_to = __ldcg(typed_from);
      break;
  }
}

// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size = 1>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }

  __device__ const scalar_t& operator[](const unsigned int i) const {
    return array[i];
  }

  __device__ Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = a[i];
    }
    return *this;
  }
};

template <int size, int align_size>
struct alignas(align_size / 2) Array<__e2m1, size, align_size> {
  static_assert(size % 2 == 0, "There must be an even number of fp4 elements");
  __e2m1 array[size / 2];

  __device__ __e2m1& operator[](const unsigned int i) {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(i % 2 == 0);
    return array[i / 2];
  }

  __device__ const __e2m1& operator[](const unsigned int i) const {
    // For performance reason, we do not check the index is even, but we assume
    // it. assert(i % 2 == 0);
    return array[i / 2];
  }

  __device__ Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size / 2; ++i) {
      array[i] = a.array[i];
    }
    return *this;
  }
};

// nvfuser-style loadLocalToGlobal function for vectorized stores
template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadLocalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile>::type* to,
    scalar_t* from) {
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");

  switch (vec_size_bit) {
    case 8: {
      *reinterpret_cast<typename MaybeVolatile<uint8_t, is_volatile>::type*>(
          to) = *reinterpret_cast<uint8_t*>(from);
      break;
    }
    case 16: {
      *reinterpret_cast<typename MaybeVolatile<uint16_t, is_volatile>::type*>(
          to) = *reinterpret_cast<uint16_t*>(from);
      break;
    }
    case 32: {
      *reinterpret_cast<typename MaybeVolatile<uint32_t, is_volatile>::type*>(
          to) = *reinterpret_cast<uint32_t*>(from);
      break;
    }
    case 64: {
      uint2 const& data = *reinterpret_cast<uint2*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      } else {
        asm volatile(
            "st.global.cs.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      }
      break;
    }
    case 128: {
      uint4 const& data = *reinterpret_cast<uint4*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      } else {
        asm volatile(
            "st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      }
      break;
    }
    default: {
// Fallback to element-wise copy for other sizes
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        if (is_volatile) {
          static_cast<volatile scalar_t*>(to)[i] = from[i];
        } else {
          to[i] = from[i];
        }
      }
      break;
    }
  }
}

// nvfuser-style loadGlobalToLocal function for vectorized loads
template <typename scalar_t, int vec_size, bool is_volatile, CacheOp cache_op>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  constexpr int64_t vec_size_bit = vecSizeBit<scalar_t>(vec_size);
  static_assert(vec_size_bit % 8 == 0, "vec_size_bit must be a multiple of 8");

  switch (vec_size_bit) {
    case 8:
    case 16:
    case 32: {
// Use simple copy for smaller sizes
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        if (is_volatile) {
          to[i] = static_cast<volatile scalar_t*>(from)[i];
        } else {
          to[i] = from[i];
        }
      }
      break;
    }
    case 64: {
      if (is_volatile) {
        uint2& data = *reinterpret_cast<uint2*>(to);
        asm volatile("ld.volatile.global.v2.s32 {%0,%1}, [%2];"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"((uint2*)from));
      } else {
        loadGlobalToLocalCached<uint2, cache_op>(
            to, const_cast<scalar_t*>(from));
      }
      break;
    }
    case 128: {
      if (is_volatile) {
        uint4& data = *reinterpret_cast<uint4*>(to);
        asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"((uint4*)from));
      } else {
        loadGlobalToLocalCached<uint4, cache_op>(
            to, const_cast<scalar_t*>(from));
      }
      break;
    }
    default: {
// Fallback to element-wise copy for other sizes
#pragma unroll
      for (int i = 0; i < vec_size; ++i) {
        if (is_volatile) {
          to[i] = static_cast<volatile scalar_t*>(from)[i];
        } else {
          to[i] = from[i];
        }
      }
      break;
    }
  }
}

// nvfuser-style __float2e4m3 implementation using PTX inline assembly
template <int align>
__device__ __inline__ Array<__e4m3, 2, align> __float2e4m3(
    const Array<float, 2, align>& input) {
  Array<__e4m3, 2, align> result;
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;}"
      : "=h"(result_scalar)
      : "f"(input[1]), "f"(input[0]));
  return result;
}

__device__ __inline__ __e4m3 __float2e4m3(const float f) {
  Array<float, 2, 1> input = {f, f};
  return __float2e4m3(input)[0];
}

// Vectorized conversion functions for Array types
template <int align>
__device__ __inline__ Array<__e2m1, 4, align> __float2e2m1(
    const Array<float, 4, align>& input) {
  // Use PTX assembly for SM_100A where e2m1x2 instruction is available
  // Note: Inline PTX can not pass 8-bit register as parameter
  // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  Array<__e2m1, 4, align> result;
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "mov.b16 %0, {byte0, byte1};\n"
      "}"
      : "=h"(result_scalar)
      : "f"(input[0]), "f"(input[1]), "f"(input[2]), "f"(input[3]));
  return result;
}

// nvfuser-style conversion functions

// __e4m32half implementation using PTX inline assembly
template <int align>
__device__ __inline__ Array<__half, 2, align> __e4m32half(
    const Array<__e4m3, 2, align>& input) {
  Array<__half, 2, align> result;
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}"
      : "=r"(result_scalar)
      : "h"(input_scalar));
  return result;
}

__device__ __inline__ __half __e4m32half(const __e4m3 b) {
  Array<__e4m3, 2, 1> input = {b, b};
  return __e4m32half(input)[0];
}

// nvfuser-style __e4m32float implementation
__device__ __inline__ float __e4m32float(const __e4m3 b) {
  return __half2float(__e4m32half(b));
}

// Clamp function from nvfuser helpers.cu - avoiding conflicts with CUDA std lib
__device__ float clamp(float x, float minv, float maxv) {
  return fminf(fmaxf(x, minv), maxv);
}

} // namespace

#include "runtime/block_quantization.cu"

/**
 * Host function to launch the templated kernel
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
cudaError_t launch_float_to_fp_kernel(
    const float* d_input,
    __e2m1* d_output_e2m1,
    __e4m3* d_output_e4m3,
    int num_rows,
    int total_elements) {
  // Validate template parameters
  static_assert(
      BLOCK_DIM_X * BLOCK_DIM_Y <= 1024,
      "Total threads per block cannot exceed 1024");

  const int elements_per_cta =
      BLOCK_DIM_X * BLOCK_DIM_Y * 4; // 4 elements per thread
  const int ctas_needed =
      (total_elements + elements_per_cta - 1) / elements_per_cta;

  dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid_dim(ctas_needed, 1 /*grid_y*/);

  // Launch the kernel with shared memory
  nvf::bq::float_to_nvfp4_conversion_kernel<<<grid_dim, block_dim>>>(
      d_input, d_output_e2m1, d_output_e4m3, num_rows, total_elements);

  return cudaGetLastError();
}

// Template function implementation for fp_conversion_kernel_execute
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
cudaError_t fp_conversion_kernel_execute(
    const float* d_input,
    const int INNER_DIM,
    int total_elements,
    nvfp4_types::__e2m1** h_output_e2m1,
    nvfp4_types::__e4m3** h_output_e4m3) {
  const int num_rows = (total_elements + INNER_DIM - 1) / INNER_DIM;

  // Allocate device memory for outputs only
  nvfp4_types::__e2m1* d_output_e2m1;
  nvfp4_types::__e4m3* d_output_e4m3;

  cudaError_t err = cudaSuccess;

  // Use aligned memory allocation for better performance
  size_t e2m1_size = total_elements * sizeof(__e2m1);
  size_t e4m3_size = num_rows * sizeof(__e4m3);

  // Align to 16-byte boundaries for optimal vectorized access
  const size_t alignment = 16;
  // e2m1_size = ((e2m1_size + alignment - 1) / alignment) * alignment;
  e4m3_size = ((e4m3_size + alignment - 1) / alignment) * alignment;

  err = cudaMalloc(&d_output_e2m1, e2m1_size);
  if (err != cudaSuccess)
    return err;

  err = cudaMalloc(&d_output_e4m3, e4m3_size);
  if (err != cudaSuccess) {
    cudaFree(d_output_e2m1);
    return err;
  }

  // Launch kernel (d_input is already on device)
  err = launch_float_to_fp_kernel<BLOCK_DIM_X, BLOCK_DIM_Y>(
      d_input, d_output_e2m1, d_output_e4m3, num_rows, total_elements);

  if (err != cudaSuccess) {
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
    *h_output_e4m3 = new __e4m3[num_rows];
    err = cudaMemcpy(
        *h_output_e4m3,
        d_output_e4m3,
        num_rows * sizeof(__e4m3),
        cudaMemcpyDeviceToHost);
  }

  // Cleanup (don't free d_input since it's managed externally)
  cudaFree(d_output_e2m1);
  cudaFree(d_output_e4m3);

  return err;
}

// Macro to generate explicit template instantiations
#define INSTANTIATE_FP_CONVERSION_KERNEL(BLOCK_DIM_X, BLOCK_DIM_Y)             \
  template cudaError_t fp_conversion_kernel_execute<BLOCK_DIM_X, BLOCK_DIM_Y>( \
      const float* d_input,                                                    \
      const int INNER_DIM,                                                     \
      int total_elements,                                                      \
      nvfp4_types::__e2m1** h_output_e2m1,                                     \
      nvfp4_types::__e4m3** h_output_e4m3);

// Explicit template instantiations for common configurations (BLOCK_DIM_X is
// always 4)
INSTANTIATE_FP_CONVERSION_KERNEL(4, 1)
INSTANTIATE_FP_CONVERSION_KERNEL(4, 4)
INSTANTIATE_FP_CONVERSION_KERNEL(4, 8)
INSTANTIATE_FP_CONVERSION_KERNEL(4, 32)
INSTANTIATE_FP_CONVERSION_KERNEL(8, 16)
