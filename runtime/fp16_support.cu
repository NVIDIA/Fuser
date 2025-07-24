// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#define __NVFUSER_HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __NVFUSER_HALF_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

struct __half;
__device__ __inline__ __half __float2half(const float);

struct __align__(2) __half {
  __half() = default;

  __half(const __half& other) {
    __x = other.__x;
  }

  __half(const __half&& other) {
    __x = other.__x;
  }

  __half(const volatile __half& other) {
    __x = other.__x;
  }

  __half(const volatile __half&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__half::operator=`
  // Doing so would requires us to return `volatile __half&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __half& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __half&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __half& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __half&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half&& other) volatile {
    __x = other.__x;
  }

  __device__ __half(const float f) {
    __x = __float2half(f).__x;
  }

  __device__ uint16_t raw() const {
    return __x;
  }

 protected:
  unsigned short __x;
};

__device__ __inline__ bool __heq(const __half a, const __half b) {
  // From cuda_fp16.hpp
  unsigned short val;
  asm("{ .reg .pred __$temp3;\n"
      "  setp.eq.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(a)), "h"(__NVFUSER_HALF_TO_CUS(b)));
  return (val != 0U) ? true : false;
}

__device__ __inline__ __half operator|(const __half x, const __half y) {
  __half val;
  asm("{  or.b16 %0, %1, %2;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(__NVFUSER_HALF_TO_CUS(x)), "h"(__NVFUSER_HALF_TO_CUS(y)));
  return val;
}

#define __NVFUSER_HAS_HALF__
