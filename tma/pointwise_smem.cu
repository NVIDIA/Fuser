#include <cstdio>
#include <iostream>

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (cudaSuccess != err) {                                                 \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                             \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

typedef int64_t nvfuser_index_t;

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

  Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = a[i];
    }
    return *this;
  }
};

template <typename T, int Dims, int AllocDims = Dims>
struct Tensor {
  __device__ T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  Array<nvfuser_index_t, Dims, 1> logical_size;
  Array<nvfuser_index_t, AllocDims, 1> alloc_stride;
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](nvfuser_index_t i) {
    return *data;
  };

  T* data;
};

// Specialization for 0-dim case that's easy to pass in a CPU based tensor.
template <typename T>
struct CpuScalarTensor {
  __device__ T& operator[](int i) {
    return data;
  };

  T data;
};

extern "C" __global__ void pointwise_smem(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T1, Tensor<float, 2, 2> T2) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  nvfuser_index_t i0;
  i0 = ((nvfuser_index_t)threadIdx.x) + (32LL * ((nvfuser_index_t)blockIdx.x));
  bool b1;
  b1 = ((nvfuser_index_t)threadIdx.x) == 0LL;
  float* T4 = reinterpret_cast<float*>(array + smem_offset + ((((T0.logical_size[0LL] * T0.logical_size[1LL]) * 4LL) + 15LL) & -16LL));
  float* T3 = reinterpret_cast<float*>(array + smem_offset + 0LL);
  #pragma unroll 1
  for(nvfuser_index_t i2 = 0; i2 < T0.logical_size[0LL]; ++i2) {
    nvfuser_index_t i3;
    i3 = T0.logical_size[1LL] * i2;
    #pragma unroll 1
    for(nvfuser_index_t i4 = 0; i4 < T0.logical_size[1LL]; ++i4) {
      nvfuser_index_t i5;
      i5 = i3 + i4;
      if (b1) {
        T4[i5]
           = T1[i5];
      }
    }
  }
  #pragma unroll 1
  for(nvfuser_index_t i6 = 0; i6 < T0.logical_size[0LL]; ++i6) {
    nvfuser_index_t i7;
    i7 = T0.logical_size[1LL] * i6;
    #pragma unroll 1
    for(nvfuser_index_t i8 = 0; i8 < T0.logical_size[1LL]; ++i8) {
      nvfuser_index_t i9;
      i9 = i7 + i8;
      if (b1) {
        T3[i9]
           = T0[i9];
      }
    }
  }
  __syncthreads();
  if ((i0 < (T0.logical_size[0LL] * T0.logical_size[1LL]))) {
    T2[i0]
      = T3[i0]
      + T4[i0];
  }
}
