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

struct __fp8;
__device__ __inline__ __fp8 __float2fp8(const float);

struct __align__(1) __fp8 {
  __fp8() = default;

  __device__ __fp8(const float f) {
    __x = __float2fp8(f).__x;
  }

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

__device__ __inline__ __fp8 __float2fp8(const float f) {
  __fp8 val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "f"(f));
  return val;
}

__device__ __inline__ __fp8 __double2fp8(const double d) {
  __fp8 val;
  asm("{  cvt.rn.f16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "d"(d));
  return val;
}

__device__ __inline__ __fp8 __int2fp8(const int i) {
  __fp8 val;
  asm("{  cvt.rn.f16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __fp8 __int2fp8(const int64_t i64) {
  __fp8 val;
  asm("{  cvt.rn.f16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __fp8 __int2fp8(const uint32_t i) {
  __fp8 val;
  asm("{  cvt.rn.f16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __fp8 __int2fp8(const uint64_t i64) {
  __fp8 val;
  asm("{  cvt.rn.f16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __fp8 __bool2fp8(const bool b) {
  return __int2fp8((int)b);
}

__device__ __inline__ float __fp82float(const __fp8 h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ double __fp82double(const __fp8 h) {
  double val;
  asm("{  cvt.f64.f16 %0, %1;}\n" : "=d"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __fp82int32(const __fp8 h) {
  int val;
  asm("{  cvt.rzi.s32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __fp82int(const __fp8 h) {
  int64_t val;
  asm("{  cvt.rzi.s64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __fp82uint32(const __fp8 h) {
  int val;
  asm("{  cvt.rzi.u32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __fp82uint(const __fp8 h) {
  int64_t val;
  asm("{  cvt.rzi.u64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ void __fp82int(const __fp8 h, int& output) {
  output = __fp82int32(h);
}

__device__ __inline__ void __fp82int(const __fp8 h, int64_t& output) {
  output = __fp82int(h);
}

__device__ __inline__ void __fp82int(const __fp8 h, uint32_t& output) {
  output = __fp82uint32(h);
}

__device__ __inline__ void __fp82int(const __fp8 h, uint64_t& output) {
  output = __fp82uint(h);
}

__device__ __inline__ nvfuser_index_t __fp82index(const __fp8 h) {
  nvfuser_index_t result;
  __fp82int(h, result);
  return result;
}

__device__ __inline__ bool __fp82bool(const __fp8 h) {
  return (bool)__fp82float(h) != 0;
}

__device__ __inline__ __fp8 __real_then_2fp8(const std::complex<float> c) {
  return __float2fp8(std::real(c));
}

__device__ __inline__ __fp8 __real_then_2fp8(const std::complex<double> c) {
  return __double2fp8(std::real(c));
}
