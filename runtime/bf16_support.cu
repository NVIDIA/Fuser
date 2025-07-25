// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#define __NVFUSER_BFLOAT_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __NVFUSER_BFLOAT_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

struct __bfloat;
__device__ __inline__ __bfloat __float2bfloat(const float);

struct __align__(2) __bfloat {
  __bfloat() = default;

  __bfloat(const __bfloat& other) {
    __x = other.__x;
  }

  __bfloat(const __bfloat&& other) {
    __x = other.__x;
  }

  __bfloat(const volatile __bfloat& other) {
    __x = other.__x;
  }

  __bfloat(const volatile __bfloat&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__bfloat::operator=`
  // Doing so would requires us to return `volatile __bfloat&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __bfloat& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __bfloat&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __bfloat& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __bfloat&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __bfloat& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __bfloat&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __bfloat& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __bfloat&& other) volatile {
    __x = other.__x;
  }

  __device__ __bfloat(const float f) {
    __x = __float2bfloat(f).__x;
  }

  __device__ uint16_t raw() const {
    return __x;
  }

 protected:
  unsigned short __x;
};

__device__ __inline__ __bfloat operator|(const __bfloat x, const __bfloat y) {
  __bfloat val;
  asm("{  or.b16 %0, %1, %2;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(__NVFUSER_BFLOAT_TO_CUS(x)), "h"(__NVFUSER_BFLOAT_TO_CUS(y)));
  return val;
}

__device__ __inline__ bool __heq(const __bfloat a, const __bfloat b) {
// From cuda_bf16.hpp
#if __CUDA_ARCH__ >= 900
  unsigned short val;
  asm("{ .reg .pred __$temp3;\n"
      "  setp.eq.bf16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(a)), "h"(__NVFUSER_BFLOAT_TO_CUS(b)));
#else
  unsigned int val;
  asm("{.reg .b32 a,b;\n"
      "  mov.b32 a, {0, %1};\n"
      "  mov.b32 b, {0, %2};\n"
      "  set.eq.f32.f32 %0, a, b;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(a)), "h"(__NVFUSER_BFLOAT_TO_CUS(b)));
#endif
  return (val != 0U) ? true : false;
}

#define __NVFUSER_HAS_BFLOAT__
