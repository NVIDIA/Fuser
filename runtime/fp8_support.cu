// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

struct __e4m3;
__device__ __inline__ __e4m3 __float2e4m3(const float);
__device__ __inline__ __e4m3 __double2e4m3(const double);

struct __align__(1) __e4m3 {
  __e4m3() = default;

  __e4m3(const __e4m3& other) {
    __x = other.__x;
  }

  __e4m3(const __e4m3&& other) {
    __x = other.__x;
  }

  __e4m3(const volatile __e4m3& other) {
    __x = other.__x;
  }

  __e4m3(const volatile __e4m3&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__e4m3::operator=`
  // Doing so would requires us to return `volatile __e4m3&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __e4m3& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e4m3&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e4m3& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e4m3&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e4m3& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __e4m3&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e4m3& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e4m3&& other) volatile {
    __x = other.__x;
  }

  __device__ __e4m3(const float f) {
    __x = __float2e4m3(f).__x;
  }

  __device__ __e4m3(const double f) {
    __x = __double2e4m3(f).__x;
  }

  __device__ __e4m3(const int x) : __x(x) {}

  __device__ __e4m3(const long long x) : __x(x) {}

  __device__ __e4m3(const uint8_t x) : __x(x) {}

  __device__ __e4m3(const uint16_t x) : __x(x) {}

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

// NOTE [ fp8 cast optimization ]
//
// For simplicity, we only provided fp8 <-> fp32 cast implementation, while
// relying on any other fp cast in the form of target_fp <-> fp32 <-> fp8.
// This avoids the complication of handling hardware specific instructions on
// various compute capabilities.
// But this simplicity could come at the cost of performance. In cuda_fp8.hpp,
// 1. bf16 -> fp8 is done via bf16 -> float -> fp8
// 2. fp16 -> fp8 is done with a conditional
//    # if (> sm_89)
//    fp16 -> fp8
//    # else
//    fp16 -> fp32 -> fp8
//    # endif
// 3. fp64 -> fp8 is handled explicitly as bitwise operations.
// TODO consider cuda_fp8.hpp for performance optimized cast.
__device__ __inline__ __e4m3 __float2e4m3(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e4m32float(const __e4m3 b) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &b, sizeof(uint8_t));
  float val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f32.f16 %0, %1;\n\t"
      "}"
      : "=f"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __double2e4m3(const double d) {
  return __float2e4m3(d);
}

__device__ __inline__ double __e4m32double(const __e4m3 b) {
  return __e4m32float(b);
}

__device__ __inline__ __e4m3 __half2e4m3(const __half h) {
  return __float2e4m3(__half2float(h));
}

__device__ __inline__ __half __e4m32half(const __e4m3 b) {
  return __float2half(__e4m32float(b));
}

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  return __float2e4m3(__bfloat2float(h));
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 b) {
  return __float2bfloat(__e4m32float(b));
}

__device__ __inline__ __e4m3 operator|(const __e4m3 x, const __e4m3 y) {
  unsigned short val;
  unsigned short x_val = x.raw();
  unsigned short y_val = y.raw();
  asm("{  or.b16 %0, %1, %2;}\n" : "=h"(val) : "h"(x_val), "h"(y_val));
  return __e4m3(val);
}

struct __e5m2;
__device__ __inline__ __e5m2 __float2e5m2(const float);
__device__ __inline__ __e5m2 __double2e5m2(const double);

struct __align__(1) __e5m2 {
  __e5m2() = default;

  __e5m2(const __e5m2& other) {
    __x = other.__x;
  }

  __e5m2(const __e5m2&& other) {
    __x = other.__x;
  }

  __e5m2(const volatile __e5m2& other) {
    __x = other.__x;
  }

  __e5m2(const volatile __e5m2&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__e5m2::operator=`
  // Doing so would requires us to return `volatile __e5m2&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __e5m2& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e5m2&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e5m2& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e5m2&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e5m2& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __e5m2&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e5m2& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e5m2&& other) volatile {
    __x = other.__x;
  }

  __device__ __e5m2(const float f) {
    __x = __float2e5m2(f).__x;
  }

  __device__ __e5m2(const double f) {
    __x = __double2e5m2(f).__x;
  }

  __device__ __e5m2(const int x) : __x(x) {}

  __device__ __e5m2(const long long x) : __x(x) {}

  __device__ __e5m2(const uint8_t x) : __x(x) {}

  __device__ __e5m2(const uint16_t x) : __x(x) {}

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

// see NOTE [ fp8 cast optimization ]
__device__ __inline__ __e5m2 __float2e5m2(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e5m22float(const __e5m2 b) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &b, sizeof(uint8_t));
  float val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f32.f16 %0, %1;\n\t"
      "}"
      : "=f"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  return __float2e5m2(f);
}

__device__ __inline__ double __e5m22double(const __e5m2 b) {
  return __e5m22float(b);
}

__device__ __inline__ __e5m2 __half2e5m2(const __half h) {
  return __float2e5m2(__half2float(h));
}

__device__ __inline__ __half __e5m22half(const __e5m2 b) {
  return __float2half(__e5m22float(b));
}

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  return __float2e5m2(__bfloat2float(h));
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 b) {
  return __float2bfloat(__e5m22float(b));
}

__device__ __inline__ __e5m2 operator|(const __e5m2 x, const __e5m2 y) {
  unsigned short val;
  unsigned short x_val = x.raw();
  unsigned short y_val = y.raw();
  asm("{  or.b16 %0, %1, %2;}\n" : "=h"(val) : "h"(x_val), "h"(y_val));
  return __e5m2(val);
}

struct __e8m0;
__device__ __inline__ __e8m0 __float2e8m0(const float);
__device__ __inline__ __e8m0 __double2e8m0(const double);

struct __align__(1) __e8m0 {
  __e8m0() = default;

  __e8m0(const __e8m0& other) {
    __x = other.__x;
  }

  __e8m0(const __e8m0&& other) {
    __x = other.__x;
  }

  __e8m0(const volatile __e8m0& other) {
    __x = other.__x;
  }

  __e8m0(const volatile __e8m0&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__e8m0::operator=`
  // Doing so would requires us to return `volatile __e8m0&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __e8m0& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e8m0&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e8m0& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e8m0&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __e8m0& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __e8m0&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e8m0& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __e8m0&& other) volatile {
    __x = other.__x;
  }

  __device__ __e8m0(const float f) {
    __x = __float2e8m0(f).__x;
  }

  __device__ __e8m0(const double f) {
    __x = __double2e8m0(f).__x;
  }

  __device__ __e8m0(const int x) : __x(x) {}

  __device__ __e8m0(const long long x) : __x(x) {}

  __device__ __e8m0(const uint8_t x) : __x(x) {}

  __device__ __e8m0(const uint16_t x) : __x(x) {}

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

// see NOTE [ fp8 cast optimization ]
__device__ __inline__ __e8m0 __float2e8m0(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e8m0 val;
  asm("{cvt.rn.satfinite.ue8m0x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e8m02float(const __e8m0 b) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &b, sizeof(uint8_t));
  float val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.bf16x2.ue8m0x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f32.bf16 %0, %1;\n\t"
      "}"
      : "=f"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e8m0 __double2e8m0(const double f) {
  return __float2e8m0(f);
}

__device__ __inline__ double __e8m02double(const __e8m0 b) {
  return __e8m02float(b);
}

__device__ __inline__ __e8m0 __half2e8m0(const __half h) {
  return __float2e8m0(__half2float(h));
}

__device__ __inline__ __half __e8m02half(const __e8m0 b) {
  return __float2half(__e8m02float(b));
}

__device__ __inline__ __e8m0 __bfloat2e8m0(const __bfloat h) {
  return __float2e8m0(__bfloat2float(h));
}

__device__ __inline__ __bfloat __e8m02bfloat(const __e8m0 b) {
  return __float2bfloat(__e8m02float(b));
}

__device__ __inline__ __e8m0 operator|(const __e8m0 x, const __e8m0 y) {
  unsigned short val;
  unsigned short x_val = x.raw();
  unsigned short y_val = y.raw();
  asm("{  or.b16 %0, %1, %2;}\n" : "=h"(val) : "h"(x_val), "h"(y_val));
  return __e8m0(val);
}
