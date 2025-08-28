#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
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

// Block broadcast functionality
namespace broadcast {
// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// Aligned: Called from aligned threads if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename BlockDimT>
__device__ void blockBroadcast(
    T& out,
    const T& inp_val,
    T* shared_mem,
    bool read_write_pred,
    BlockDimT block_dim) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, block_dim);

  if (has_valid_data && read_write_pred) {
    shared_mem[shared_offset] = inp_val;
  }

  __syncthreads();

  if (read_write_pred) {
    out = shared_mem[shared_offset];
  }

  __syncthreads();
}
} // namespace broadcast

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

struct __e4m3 {
  uint8_t __x;
  __host__ __device__ __e4m3() = default;
  __host__ __device__ __e4m3(uint8_t x) : __x(x) {}
  __host__ __device__ uint8_t raw() const {
    return __x;
  }
};

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
 * - input: Pointer to 1D array representing 2D data where inner dimension is
 * fixed to 16
 * - output_e2m1: Output pointer for __e2m1 (FP4 E2M1) format
 * - output_e4m3: Output pointer for __e4m3 (FP8 E4M3) format
 * - num_rows: Number of rows in the input (outer dimension)
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void float_to_fp_conversion_kernel(
    const float* __restrict__ input,
    __e2m1* __restrict__ output_e2m1,
    __e4m3* __restrict__ output_e4m3,
    int num_rows,
    int total_elements) {
  // Ensure BLOCK_DIM_X is 4 for vectorized loads
  static_assert(
      BLOCK_DIM_X == 4, "we need to reduce 16 elements to find the max");

  // Calculate thread indices
  const int tid_x = threadIdx.x; // 0, 1, 2, or 3

  const int tile_id_to_process = blockIdx.x * blockDim.y + threadIdx.y;

  auto elements_seen = tile_id_to_process * 16 + tid_x * 4;
  if (elements_seen >= total_elements) {
    // No work for this thread
    return;
  }
  // Create an array of 4 floats for vectorized load using nvfuser-style
  Array<float, 4, 4> vec4;
  vec4.set(0.0f); // Initialize to zero like nvfuser does

  if (elements_seen + 3 < total_elements) {
    // Use loadGlobalToLocal for vectorized load similar to nvfuser pattern
    loadGlobalToLocal<
        float,
        /*vec_size=*/4,
        /*is_volatile=*/false,
        CacheOp::Streaming>(
        &vec4.array[0],
        const_cast<float*>(&input[tile_id_to_process * 16 + 4 * tid_x]));
  } else {
    // Assert that total_elements is even for proper FP4 handling
    assert(total_elements % 2 == 0);

    // Handle boundary case with element-wise loads
    loadGlobalToLocal<
        float,
        /*vec_size=*/2,
        /*is_volatile=*/false,
        CacheOp::Streaming>(
        &vec4.array[0],
        const_cast<float*>(&input[tile_id_to_process * 16 + 4 * tid_x]));
  }

  // Calculate the max of the values in vec4
  float local_max = fabsf(vec4[0]);
#pragma unroll
  for (int i = 1; i < 4; ++i) {
    local_max = fmaxf(local_max, fabsf(vec4[i]));
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

  // Broadcast clamped_max_converted from thread 0 in X dimension to all threads
  float broadcasted_clamped_max;
  broadcast::blockBroadcast<true, false, false, true>(
      broadcasted_clamped_max,
      clamped_max_converted,
      shared_mem,
      true,
      DefaultBlockDim());

  // Process vec4 array: divide each element by broadcasted_clamped_max and
  // clamp
  Array<float, 4, 4> clamped_vals;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float scaled_val = vec4[i] / broadcasted_clamped_max;
    clamped_vals[i] = clamp(scaled_val, -6.000000000e+00f, 6.000000000e+00f);
  }

  // Convert clamped_vals to FP4 E2M1 format using nvfuser-style vectorized
  // operations T10 corresponds to clamped_vals, T11 corresponds to fp4_vals
  Array<__e2m1, 4, 1> fp4_vals;
  *reinterpret_cast<Array<__e2m1, 4, 4>*>(&fp4_vals[0]) =
      __float2e2m1(*reinterpret_cast<Array<float, 4, 4>*>(&clamped_vals[0]));

  Array<__e2m1, 4, 4> fp4_vals_aligned;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fp4_vals_aligned[i] = fp4_vals[i];
  }

  // Write back the clamped max value if this is thread 0 in X dimension
  if (tid_x == 0) {
    output_e4m3[tile_id_to_process] = clamped_max_fp8;
  }

  // Calculate the global offset for this thread's starting position
  const int start_col = tid_x * 4;
  const int base_offset = tile_id_to_process * 16 + start_col;

  if (elements_seen + 3 < total_elements) {
    // Store fp4_vals_aligned using nvfuser-style vectorized store

    // Check alignment of the address
    void* addr = &output_e2m1[(base_offset) / 2];
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
    bool is_2byte_aligned = (addr_val % 2) == 0;
    bool is_4byte_aligned = (addr_val % 4) == 0;

    printf(
        "Address: %p, base_offset: %d, offset/2: %d, 2-byte aligned: %s, "
        "4-byte aligned: %s\n",
        addr,
        base_offset,
        base_offset / 2,
        is_2byte_aligned ? "yes" : "no",
        is_4byte_aligned ? "yes" : "no");

    loadLocalToGlobal<__e2m1, /*vec_size=*/4, /*is_volatile=*/false>(
        &output_e2m1[(base_offset) / 2], &fp4_vals_aligned.array[0]);
  } else {
    // Store fp4_vals_aligned using nvfuser-style vectorized store
    loadLocalToGlobal<__e2m1, /*vec_size=*/2, /*is_volatile=*/false>(
        &output_e2m1[base_offset], &fp4_vals_aligned.array[0]);
  }
}

/**
 * Host function to launch the templated kernel
 */
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
cudaError_t launch_float_to_fp_kernel(
    const float* d_input,
    __e2m1* d_output_e2m1,
    __e4m3* d_output_e4m3,
    int num_rows,
    int total_elements,
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
  const int grid_x = (num_rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
  const int grid_y = 1;

  dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid_dim(grid_x, grid_y);

  // Calculate shared memory size for blockReduce
  const int shared_mem_size = BLOCK_DIM_X * BLOCK_DIM_Y * sizeof(float);

  // Launch the kernel with shared memory
  float_to_fp_conversion_kernel<BLOCK_DIM_X, BLOCK_DIM_Y>
      <<<grid_dim, block_dim, shared_mem_size, stream>>>(
          d_input, d_output_e2m1, d_output_e4m3, num_rows, total_elements);

  return cudaGetLastError();
}

// Convenient wrapper function for common use cases
template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
class FpConversionKernel {
 public:
  static cudaError_t execute(
      const float* h_input,
      const int INNER_DIM,
      int total_elements,
      __e2m1** h_output_e2m1 = nullptr,
      __e4m3** h_output_e4m3 = nullptr,
      cudaStream_t stream = 0) {
    const int num_rows = (total_elements + INNER_DIM - 1) / INNER_DIM;

    // Allocate device memory
    float* d_input;
    __e2m1* d_output_e2m1;
    __e4m3* d_output_e4m3;

    cudaError_t err = cudaSuccess;

    // Use aligned memory allocation for better performance
    size_t input_size = total_elements * sizeof(float);
    size_t e2m1_size = total_elements * sizeof(__e2m1);
    size_t e4m3_size = num_rows * sizeof(__e4m3);

    // Align to 16-byte boundaries for optimal vectorized access
    const size_t alignment = 16;
    input_size = ((input_size + alignment - 1) / alignment) * alignment;
    // e2m1_size = ((e2m1_size + alignment - 1) / alignment) * alignment;
    e4m3_size = ((e4m3_size + alignment - 1) / alignment) * alignment;

    err = cudaMalloc(&d_input, input_size);
    if (err != cudaSuccess)
      return err;

    err = cudaMalloc(&d_output_e2m1, e2m1_size);
    if (err != cudaSuccess) {
      cudaFree(d_input);
      return err;
    }

    err = cudaMalloc(&d_output_e4m3, e4m3_size);
    if (err != cudaSuccess) {
      cudaFree(d_input);
      cudaFree(d_output_e2m1);
      return err;
    }

    // Copy input to device using aligned memory copy
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
        d_input,
        d_output_e2m1,
        d_output_e4m3,
        num_rows,
        total_elements,
        stream);

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
      *h_output_e4m3 = new __e4m3[num_rows];
      err = cudaMemcpy(
          *h_output_e4m3,
          d_output_e4m3,
          num_rows * sizeof(__e4m3),
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
#define LAUNCH_KERNEL_4x1(                                        \
    input, inner_dim, total_elements, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 1>::execute(                              \
      input, inner_dim, total_elements, e2m1_out, e4m3_out, stream)

#define LAUNCH_KERNEL_4x4(                                        \
    input, inner_dim, total_elements, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 4>::execute(                              \
      input, inner_dim, total_elements, e2m1_out, e4m3_out, stream)

#define LAUNCH_KERNEL_4x8(                                        \
    input, inner_dim, total_elements, e2m1_out, e4m3_out, stream) \
  FpConversionKernel<4, 8>::execute(                              \
      input, inner_dim, total_elements, e2m1_out, e4m3_out, stream)
