// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

struct __e4m3;
__device__ __inline__ __e4m3 __float2e4m3(const float);

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

__device__ __inline__ __e4m3 __double2e4m3(const double f) {
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{\n\t"
      ".reg .b16 buf0;\n\t"
      ".reg .b32 buf1;\n\t"
      "cvt.rn.f16.f64 buf0, %1;\n\t"
      "cvt.u32.u16 buf1, buf0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, buf1;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "d"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ double __e4m32double(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  double val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f64.f16 %0, %1;"
      "}"
      : "=d"(val)
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

__device__ __inline__ __e4m3 __half2e4m3(const __half h) {
  uint32_t buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "r"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e4m32half(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __half val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "}"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16.bf16 %1, %1;\n\t"
      "cvt.u32.u16 buf0, %1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, buf0;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __bfloat val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "cvt.bf16.f16 %0, %0;\n\t"
      "}"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

struct __e5m2;
__device__ __inline__ __e5m2 __float2e5m2(const float);

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

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{\n\t"
      ".reg .b16 buf0;\n\t"
      ".reg .b32 buf1;\n\t"
      "cvt.rn.f16.f64 buf0, %1;\n\t"
      "cvt.u32.u16 buf1, buf0;\n\t"
      "cvt.rn.satfinite.e5m2x2.f16x2 %0, buf1;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "d"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ double __e5m22double(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  double val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f64.f16 %0, %1;"
      "}"
      : "=d"(val)
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

__device__ __inline__ __e5m2 __half2e5m2(const __half h) {
  uint32_t buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "r"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e5m22half(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __half val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "}"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16.bf16 %1, %1;\n\t"
      "cvt.u32.u16 buf0, %1;\n\t"
      "cvt.rn.satfinite.e5m2x2.f16x2 %0, buf0;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __bfloat val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "cvt.bf16.f16 %0, %0;\n\t"
      "}"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}
