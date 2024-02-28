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

__device__ __inline__ __e4m3 __float2e4m3(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  std::memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e4m32float(const __e4m3 h) {
  unsigned short _tmp_buffer;
  std::memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  float val;
  asm("{cvt.rn.f16x2.e4m3x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f32.f16 %0, %2;}"
      : "=f"(val), "=r"(_b32_buffer)
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
  std::memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e5m22float(const __e5m2 h) {
  unsigned short _tmp_buffer;
  std::memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __attribute__((unused)) unsigned int _b32_buffer;
  float val;
  asm("{cvt.rn.f16x2.e5m2x2 %1, %2;}\n\t"
      "cvt.u16.u32 %2, %1;\n\t"
      "{cvt.f32.f16 %0, %2;}"
      : "=f"(val), "=r"(_b32_buffer)
      : "h"(_tmp_buffer));

  return val;
}
