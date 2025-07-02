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

__device__ __inline__ __e8m0 operator|(const __e8m0 x, const __e8m0 y) {
  unsigned short val;
  unsigned short x_val = x.raw();
  unsigned short y_val = y.raw();
  asm("{  or.b16 %0, %1, %2;}\n" : "=h"(val) : "h"(x_val), "h"(y_val));
  return __e8m0(val);
}
