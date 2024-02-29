// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

struct __e4m3;
__device__ __inline__ __e4m3 __float2e4m3(const float);

struct __e5m2;
__device__ __inline__ __e5m2 __float2e5m2(const float);

struct __align__(1) __e4m3 {
  __e4m3() = default;

  __device__ __e4m3(const float f) {
    __x = __float2e4m3(f).__x;
  }

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

struct __align__(1) __e5m2 {
  __e5m2() = default;

  __device__ __e5m2(const float f) {
    __x = __float2e5m2(f).__x;
  }

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

__device__ __inline__ __e4m3 __double2e4m3(const double f) {
  constexpr float f_const_zero = 0.f;
  float buffer = 0;
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("cvt.rn.f32.f64 %1, %3;\n\t"
      "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer), "f"(f_const_zero), "d"(f));
  std::memcpy(&tmp, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ double __e4m32double(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  double val;
  asm("{cvt.rn.f16x2.e4m3x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f64.f16 %0, %2;}"
      : "=d"(val), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

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

__device__ __inline__ float __e4m32float(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  float val;
  asm("{cvt.rn.f16x2.e4m3x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f32.f16 %0, %2;}"
      : "=f"(val), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __half2e4m3(const __half h) {
  float buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e4m32half(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  unsigned int _b32_buffer;
  __half val;
  asm("{cvt.rn.f16x2.e4m3x2 %1, %2;}\n\t"
      "cvt.u16.u32 %0, %1;"
      : "=h"(__NVFUSER_HALF_TO_US(val)), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  float buffer;
  memcpy(&buffer, &h, sizeof(__bfloat));
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  unsigned int _b32_buffer;
  __bfloat val;
  asm("{cvt.rn.f16x2.e4m3x2 %1, %2;}\n\t"
      "cvt.u16.u32 %0, %1;"
      "cvt.bf16.f16 %0, %0;"
      : "=h"(__NVFUSER_HALF_TO_US(val)), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  constexpr float f_const_zero = 0.f;
  float buffer = 0;
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("cvt.rn.f32.f64 %1, %3;\n\t"
      "{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer), "f"(f_const_zero), "d"(f));
  std::memcpy(&tmp, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ double __e5m22double(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  double val;
  asm("{cvt.rn.f16x2.e5m2x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f32.f16 %0, %2;}"
      : "=d"(val), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

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

__device__ __inline__ float __e5m22float(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  float val;
  asm("{cvt.rn.f16x2.e5m2x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f32.f16 %0, %2;}"
      : "=f"(val), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __half2e5m2(const __half h) {
  float buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e5m22half(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  unsigned int _b32_buffer;
  __half val;
  asm("{cvt.rn.f16x2.e5m2x2 %1, %2;}\n\t"
      "cvt.u16.u32 %0, %1;"
      : "=h"(__NVFUSER_HALF_TO_US(val)), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  float buffer;
  memcpy(&buffer, &h, sizeof(__bfloat));
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.bf16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "f"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  unsigned int _b32_buffer;
  __bfloat val;
  asm("{cvt.rn.f16x2.e5m2x2 %1, %2;}\n\t"
      "cvt.u16.u32 %0, %1;"
      "cvt.bf16.f16 %0, %0;"
      : "=h"(__NVFUSER_HALF_TO_US(val)), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}
