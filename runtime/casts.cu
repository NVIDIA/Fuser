// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// clang-format off
// Disable clang-format because it tries to put the _Pragma("unroll")
// and the for loop on the same line, which doesn't make sense.
#define DEFINE_CAST_VECN_WITH_VEC2(name, from_type, to_type)        \
  template <int n, int align>                                       \
  __device__ __inline__ Array<to_type, n, align> name(              \
      const Array<from_type, n, align>& input) {                    \
    Array<to_type, n, align> result;                                \
    _Pragma("unroll")                                               \
    for (int i = 0; i < n; i += 2) {                                \
      if (i + 1 < n) {                                              \
        Array<from_type, 2, align> pair = {input[i], input[i + 1]}; \
        Array<to_type, 2, align> res_pair = name(pair);             \
        result[i] = res_pair[0];                                    \
        result[i + 1] = res_pair[1];                                \
      } else {                                                      \
        result[i] = name(input[i]);                                 \
      }                                                             \
    }                                                               \
    return result;                                                  \
  }

#define DEFINE_CAST_NAIVE_VECN(name, from_type, to_type) \
  template <int n, int align>                            \
  __device__ __inline__ Array<to_type, n, align> name(   \
      const Array<from_type, n, align>& input) {         \
    Array<to_type, n, align> result;                     \
    _Pragma("unroll")                                    \
    for (int i = 0; i < n; i ++) {                       \
      result[i] = name(input[i]);                        \
    }                                                    \
    return result;                                       \
  }

#define DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(name, to_type) \
  template <typename T, int n, int align>                   \
  __device__ __inline__ Array<to_type, n, align> name(      \
      const Array<T, n, align>& input) {                    \
    Array<to_type, n, align> result;                        \
    _Pragma("unroll")                                       \
    for (int i = 0; i < n; i++) {                           \
      result[i] = name(input[i]);                           \
    }                                                       \
    return result;                                          \
  }
// clang-format on

// Wrappers for static casts
template <typename T>
__device__ __inline__ float __to_float(const T f) {
  return static_cast<float>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_float, float);

template <typename T>
__device__ __inline__ float __real_then_to_float(const T f) {
  return __to_float(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_float, float);

template <typename T>
__device__ __inline__ int8_t __to_int8(const T f) {
  return static_cast<int8_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_int8, int8_t);

template <typename T>
__device__ __inline__ int16_t __to_int16(const T f) {
  return static_cast<int16_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_int16, int16_t);

template <typename T>
__device__ __inline__ int32_t __to_int32(const T f) {
  return static_cast<int32_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_int32, int32_t);

template <typename T>
__device__ __inline__ int64_t __to_int64(const T f) {
  return static_cast<int64_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_int64, int64_t);

template <typename T>
__device__ __inline__ int8_t __real_then_to_int8(const T f) {
  return __to_int8(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_int8, int8_t);

template <typename T>
__device__ __inline__ int16_t __real_then_to_int16(const T f) {
  return __to_int16(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_int16, int16_t);

template <typename T>
__device__ __inline__ int32_t __real_then_to_int32(const T f) {
  return __to_int32(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_int32, int32_t);

template <typename T>
__device__ __inline__ int64_t __real_then_to_int64(const T f) {
  return __to_int64(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_int64, int64_t);

template <typename T>
__device__ __inline__ uint8_t __to_uint8(const T f) {
  return static_cast<uint8_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_uint8, uint8_t);

template <typename T>
__device__ __inline__ uint16_t __to_uint16(const T f) {
  return static_cast<uint16_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_uint16, uint16_t);

template <typename T>
__device__ __inline__ uint32_t __to_uint32(const T f) {
  return static_cast<uint32_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_uint32, uint32_t);

template <typename T>
__device__ __inline__ uint64_t __to_uint64(const T f) {
  return static_cast<uint64_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_uint64, uint64_t);

template <typename T>
__device__ __inline__ uint8_t __real_then_to_uint8(const T f) {
  return __to_uint8(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_uint8, uint8_t);

template <typename T>
__device__ __inline__ uint16_t __real_then_to_uint16(const T f) {
  return __to_uint16(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_uint16, uint16_t);

template <typename T>
__device__ __inline__ uint32_t __real_then_to_uint32(const T f) {
  return __to_uint32(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_uint32, uint32_t);

template <typename T>
__device__ __inline__ uint64_t __real_then_to_uint64(const T f) {
  return __to_uint64(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_uint64, uint64_t);

template <typename T>
__device__ __inline__ nvfuser_index_t __to_index(const T f) {
  return static_cast<nvfuser_index_t>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_index, nvfuser_index_t);

template <typename T>
__device__ __inline__ nvfuser_index_t __real_then_to_index(const T f) {
  return __to_index(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_index, nvfuser_index_t);

template <typename T>
__device__ __inline__ double __to_double(const T f) {
  return static_cast<double>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_double, double);

template <typename T>
__device__ __inline__ double __real_then_to_double(const T f) {
  return __to_double(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_double, double);

template <typename T>
__device__ __inline__ bool __to_bool(const T f) {
  return static_cast<bool>(f);
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_bool, bool);

template <typename T>
__device__ __inline__ bool __real_then_to_bool(const T f) {
  return __to_bool(std::real(f));
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__real_then_to_bool, bool);

template <typename T>
__device__ __inline__ std::complex<double> __to_complex_double(const T f) {
  return (std::complex<double>)f;
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_complex_double, std::complex<double>);

template <typename T>
__device__ __inline__ std::complex<float> __to_complex_float(const T f) {
  return (std::complex<float>)f;
}

DEFINE_CAST_NAIVE_VECN_FROM_TEMPLATE(__to_complex_float, std::complex<float>);

// Half casts

__device__ __inline__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "f"(f));
  return val;
}

template <int align>
__device__ __inline__ Array<__half, 2, align> __float2half(
    const Array<float, 2, align>& input) {
  Array<__half, 2, align> result;
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{ cvt.rn.f16x2.f32 %0, %1, %2; }"
      : "=r"(result_scalar)
      : "f"(input[1]), "f"(input[0]));
  return result;
}

DEFINE_CAST_VECN_WITH_VEC2(__float2half, float, __half);

__device__ __inline__ __half __double2half(const double d) {
  __half val;
  asm("{  cvt.rn.f16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "d"(d));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__double2half, double, __half);

__device__ __inline__ __half __int2half(const int i) {
  __half val;
  asm("{  cvt.rn.f16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__int2half, int, __half);

__device__ __inline__ __half __int2half(const int64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__int2half, int64_t, __half);

__device__ __inline__ __half __int2half(const uint32_t i) {
  __half val;
  asm("{  cvt.rn.f16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__int2half, uint32_t, __half);

__device__ __inline__ __half __int2half(const uint64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__int2half, uint64_t, __half);

__device__ __inline__ __half __bool2half(const bool b) {
  return __int2half((int)b);
}

DEFINE_CAST_NAIVE_VECN(__bool2half, bool, __half);

__device__ __inline__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2float, __half, float);

__device__ __inline__ std::complex<float> __half2complex_float(const __half h) {
  return (std::complex<float>)__half2float(h);
}

DEFINE_CAST_NAIVE_VECN(__half2complex_float, __half, std::complex<float>);

__device__ __inline__ double __half2double(const __half h) {
  double val;
  asm("{  cvt.f64.f16 %0, %1;}\n" : "=d"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2double, __half, double);

__device__ __inline__ std::complex<double> __half2complex_double(
    const __half h) {
  return (std::complex<double>)__half2double(h);
}

DEFINE_CAST_NAIVE_VECN(__half2complex_double, __half, std::complex<double>);

__device__ int __half2int32(const __half h) {
  int val;
  asm("{  cvt.rzi.s32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2int32, __half, int);

__device__ __inline__ int64_t __half2int(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.s64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2int, __half, int64_t);

__device__ int __half2uint32(const __half h) {
  int val;
  asm("{  cvt.rzi.u32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2uint32, __half, uint32_t);

__device__ __inline__ int64_t __half2uint(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.u64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2uint, __half, uint64_t);

__device__ __inline__ void __half2int(const __half h, int& output) {
  output = __half2int32(h);
}

__device__ __inline__ void __half2int(const __half h, int64_t& output) {
  output = __half2int(h);
}

__device__ __inline__ void __half2int(const __half h, uint32_t& output) {
  output = __half2uint32(h);
}

__device__ __inline__ void __half2int(const __half h, uint64_t& output) {
  output = __half2uint(h);
}

__device__ __inline__ nvfuser_index_t __half2index(const __half h) {
  nvfuser_index_t result;
  __half2int(h, result);
  return result;
}

DEFINE_CAST_NAIVE_VECN(__half2index, __half, nvfuser_index_t);

__device__ __inline__ bool __half2bool(const __half h) {
  return (bool)__half2float(h) != 0;
}

DEFINE_CAST_NAIVE_VECN(__half2bool, __half, bool);

__device__ __inline__ __half __real_then_2half(const std::complex<float> c) {
  return __float2half(std::real(c));
}

DEFINE_CAST_NAIVE_VECN(__real_then_2half, std::complex<float>, __half);

__device__ __inline__ __half __real_then_2half(const std::complex<double> c) {
  return __double2half(std::real(c));
}

DEFINE_CAST_NAIVE_VECN(__real_then_2half, std::complex<double>, __half);

// Bfloat casts

__device__ __inline__ __bfloat __float2bfloat(const float f) {
  __bfloat val;
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "f"(f));
  return val;
}

template <int align>
__device__ __inline__ Array<__bfloat, 2, align> __float2bfloat(
    const Array<float, 2, align>& input) {
  Array<__bfloat, 2, align> result;
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{ cvt.rn.bf16x2.f32 %0, %1, %2; }"
      : "=r"(result_scalar)
      : "f"(input[1]), "f"(input[0]));
  return result;
}

DEFINE_CAST_VECN_WITH_VEC2(__float2bfloat, float, __bfloat);

__device__ __inline__ __bfloat __double2bfloat(const double d) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "d"(d));
  return val;
#else
  return __float2bfloat(static_cast<float>(d));
#endif
}

DEFINE_CAST_NAIVE_VECN(__double2bfloat, double, __bfloat);

__device__ __inline__ __bfloat __int2bfloat(const int i) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "r"(i));
  return val;
#else
  return __float2bfloat(static_cast<float>(i));
#endif
}

DEFINE_CAST_NAIVE_VECN(__int2bfloat, int, __bfloat);

__device__ __inline__ __bfloat __int2bfloat(const int64_t i64) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "l"(i64));
  return val;
#else
  return __float2bfloat(static_cast<float>(i64));
#endif
}

DEFINE_CAST_NAIVE_VECN(__int2bfloat, int64_t, __bfloat);

__device__ __inline__ __bfloat __int2bfloat(const uint32_t i) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "r"(i));
  return val;
#else
  return __float2bfloat(static_cast<float>(i));
#endif
}

DEFINE_CAST_NAIVE_VECN(__int2bfloat, uint32_t, __bfloat);

__device__ __inline__ __bfloat __int2bfloat(const uint64_t i64) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "l"(i64));
  return val;
#else
  return __float2bfloat(static_cast<float>(i64));
#endif
}

DEFINE_CAST_NAIVE_VECN(__int2bfloat, uint64_t, __bfloat);

__device__ __inline__ __bfloat __bool2bfloat(const bool b) {
  return __int2bfloat((int)b);
}

DEFINE_CAST_NAIVE_VECN(__bool2bfloat, bool, __bfloat);

__device__ __inline__ float __bfloat2float(const __bfloat h) {
  float val;
  asm("{  mov.b32 %0, {0,%1};}\n"
      : "=f"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__bfloat2float, __bfloat, float);

__device__ __inline__ std::complex<float> __bfloat2complex_float(
    const __bfloat h) {
  return (std::complex<float>)__bfloat2float(h);
}

__device__ __inline__ double __bfloat2double(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  double val;
  asm("{  cvt.f64.bf16 %0, %1;}\n"
      : "=d"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<double>(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2double, __bfloat, double);

DEFINE_CAST_NAIVE_VECN(__bfloat2complex_float, __bfloat, std::complex<float>);

__device__ __inline__ std::complex<double> __bfloat2complex_double(
    const __bfloat h) {
  return (std::complex<double>)__bfloat2double(h);
}

DEFINE_CAST_NAIVE_VECN(__bfloat2complex_double, __bfloat, std::complex<double>);

__device__ int __bfloat2int32(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int val;
  asm("{  cvt.rzi.s32.bf16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int>(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2int32, __bfloat, int);

__device__ __inline__ int64_t __bfloat2int(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int64_t val;
  asm("{  cvt.rzi.s64.bf16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int64_t>(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2int, __bfloat, int64_t);

__device__ int __bfloat2uint32(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int val;
  asm("{  cvt.rzi.u32.bf16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int>(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2uint32, __bfloat, uint32_t);

__device__ __inline__ int64_t __bfloat2uint(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int64_t val;
  asm("{  cvt.rzi.u64.bf16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int64_t>(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2uint, __bfloat, uint64_t);

__device__ __inline__ void __bfloat2int(const __bfloat h, int& output) {
  output = __bfloat2int32(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, int64_t& output) {
  output = __bfloat2int(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, uint32_t& output) {
  output = __bfloat2uint32(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, uint64_t& output) {
  output = __bfloat2uint(h);
}

__device__ __inline__ nvfuser_index_t __bfloat2index(
    const __bfloat h,
    bool& output) {
  nvfuser_index_t result;
  __bfloat2int(h, result);
  return result;
}

DEFINE_CAST_NAIVE_VECN(__bfloat2index, __bfloat, nvfuser_index_t);

__device__ __inline__ bool __bfloat2bool(const __bfloat h) {
  return (bool)__bfloat2float(h) != 0;
}

DEFINE_CAST_NAIVE_VECN(__bfloat2bool, __bfloat, bool);

__device__ __inline__ __bfloat __half2bfloat(const __half h) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.f16 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
#else
  return __float2bfloat(__half2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__half2bfloat, __half, __bfloat);

__device__ __inline__ __half __bfloat2half(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  __half val;
  asm("{  cvt.rn.f16.bf16 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return __float2half(__bfloat2float(h));
#endif
}

DEFINE_CAST_NAIVE_VECN(__bfloat2half, __bfloat, __half);

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<float> c) {
  return __float2bfloat(std::real(c));
}

DEFINE_CAST_NAIVE_VECN(__real_then_2bfloat, std::complex<float>, __bfloat);

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<double> c) {
  return __double2bfloat(std::real(c));
}

DEFINE_CAST_NAIVE_VECN(__real_then_2bfloat, std::complex<double>, __bfloat);

// e4m3 casts

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

DEFINE_CAST_VECN_WITH_VEC2(__float2e4m3, float, __e4m3);

__device__ __inline__ __e4m3 __double2e4m3(const double d) {
  return __float2e4m3(d);
}

template <int n, int align>
__device__ __inline__ Array<__e4m3, n, align> __double2e4m3(
    const Array<double, n, align>& input) {
  return __float2e4m3(__to_float(input));
}

template <int align>
__device__ __inline__ Array<__e4m3, 2, align> __half2e4m3(
    const Array<__half, 2, align>& input) {
  Array<__e4m3, 2, align> result;
  const uint32_t& input_scalar = *reinterpret_cast<const uint32_t*>(&input);
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}"
      : "=h"(result_scalar)
      : "r"(input_scalar));
  return result;
}

__device__ __inline__ __e4m3 __half2e4m3(const __half h) {
  Array<__half, 2, 1> input = {h, h};
  return __half2e4m3(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__half2e4m3, __half, __e4m3);

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  return __float2e4m3(__bfloat2float(h));
}

template <int n, int align>
__device__ __inline__ Array<__e4m3, n, align> __bfloat2e4m3(
    const Array<__bfloat, n, align>& input) {
  return __float2e4m3(__bfloat2float(input));
}

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

DEFINE_CAST_VECN_WITH_VEC2(__e4m32half, __e4m3, __half);

__device__ __inline__ float __e4m32float(const __e4m3 b) {
  return __half2float(__e4m32half(b));
}

template <int n, int align>
__device__ __inline__ Array<float, n, align> __e4m32float(
    const Array<__e4m3, n, align>& input) {
  return __half2float(__e4m32half(input));
}

__device__ __inline__ double __e4m32double(const __e4m3 b) {
  return __e4m32float(b);
}

template <int n, int align>
__device__ __inline__ Array<double, n, align> __e4m32double(
    const Array<__e4m3, n, align>& input) {
  return __to_double(__e4m32float(input));
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 b) {
  return __float2bfloat(__e4m32float(b));
}

template <int n, int align>
__device__ __inline__ Array<__bfloat, n, align> __e4m32bfloat(
    const Array<__e4m3, n, align>& input) {
  return __float2bfloat(__e4m32float(input));
}

// e5m2 casts

template <int align>
__device__ __inline__ Array<__e5m2, 2, align> __float2e5m2(
    const Array<float, 2, align>& input) {
  Array<__e5m2, 2, align> result;
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;}"
      : "=h"(result_scalar)
      : "f"(input[1]), "f"(input[0]));
  return result;
}
__device__ __inline__ __e5m2 __float2e5m2(const float f) {
  Array<float, 2, 1> input = {f, f};
  return __float2e5m2(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__float2e5m2, float, __e5m2);

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  return __float2e5m2(f);
}

template <int n, int align>
__device__ __inline__ Array<__e5m2, n, align> __double2e5m2(
    const Array<double, n, align>& input) {
  return __float2e5m2(__to_float(input));
}

template <int align>
__device__ __inline__ Array<__e5m2, 2, align> __half2e5m2(
    const Array<__half, 2, align>& input) {
  Array<__e5m2, 2, align> result;
  const uint32_t& input_scalar = *reinterpret_cast<const uint32_t*>(&input);
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}"
      : "=h"(result_scalar)
      : "r"(input_scalar));
  return result;
}

__device__ __inline__ __e5m2 __half2e5m2(const __half h) {
  Array<__half, 2, 1> input = {h, h};
  return __half2e5m2(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__half2e5m2, __half, __e5m2);

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  return __float2e5m2(__bfloat2float(h));
}

template <int n, int align>
__device__ __inline__ Array<__e5m2, n, align> __bfloat2e5m2(
    const Array<__bfloat, n, align>& input) {
  return __float2e5m2(__bfloat2float(input));
}

template <int align>
__device__ __inline__ Array<__half, 2, align> __e5m22half(
    const Array<__e5m2, 2, align>& input) {
  Array<__half, 2, align> result;
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}"
      : "=r"(result_scalar)
      : "h"(input_scalar));
  return result;
}

__device__ __inline__ __half __e5m22half(const __e5m2 b) {
  Array<__e5m2, 2, 1> input = {b, b};
  return __e5m22half(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__e5m22half, __e5m2, __half);

__device__ __inline__ float __e5m22float(const __e5m2 b) {
  return __half2float(__e5m22half(b));
}

template <int n, int align>
__device__ __inline__ Array<float, n, align> __e5m22float(
    const Array<__e5m2, n, align>& input) {
  return __half2float(__e5m22half(input));
}

__device__ __inline__ double __e5m22double(const __e5m2 b) {
  return __e5m22float(b);
}

template <int n, int align>
__device__ __inline__ Array<double, n, align> __e5m22double(
    const Array<__e5m2, n, align>& input) {
  return __to_double(__e5m22float(input));
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 b) {
  return __float2bfloat(__e5m22float(b));
}

template <int n, int align>
__device__ __inline__ Array<__bfloat, n, align> __e5m22bfloat(
    const Array<__e5m2, n, align>& input) {
  return __float2bfloat(__e5m22float(input));
}

// e8m0 casts

template <int align>
__device__ __inline__ Array<__e8m0, 2, align> __float2e8m0(
    const Array<float, 2, align>& input) {
  Array<__e8m0, 2, align> result;
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;}"
      : "=h"(result_scalar)
      : "f"(input[1]), "f"(input[0]));
  return result;
}

__device__ __inline__ __e8m0 __float2e8m0(const float f) {
  Array<float, 2, 1> input = {f, f};
  return __float2e8m0(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__float2e8m0, float, __e8m0);

__device__ __inline__ __e8m0 __double2e8m0(const double f) {
  return __float2e8m0(f);
}

template <int n, int align>
__device__ __inline__ Array<__e8m0, n, align> __double2e8m0(
    const Array<double, n, align>& input) {
  return __float2e8m0(__to_float(input));
}

__device__ __inline__ __e8m0 __half2e8m0(const __half h) {
  return __float2e8m0(__half2float(h));
}

template <int n, int align>
__device__ __inline__ Array<__e8m0, n, align> __half2e8m0(
    const Array<__half, n, align>& input) {
  return __float2e8m0(__half2float(input));
}

template <int align>
__device__ __inline__ Array<__e8m0, 2, align> __bfloat2e8m0(
    const Array<__bfloat, 2, align>& input) {
  Array<__e8m0, 2, align> result;
  const uint32_t& input_scalar = *reinterpret_cast<const uint32_t*>(&input);
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1;}"
      : "=h"(result_scalar)
      : "r"(input_scalar));
  return result;
}

__device__ __inline__ __e8m0 __bfloat2e8m0(const __bfloat h) {
  Array<__bfloat, 2, 1> input = {h, h};
  return __bfloat2e8m0(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__bfloat2e8m0, __bfloat, __e8m0);

template <int align>
__device__ __inline__ Array<__bfloat, 2, align> __e8m02bfloat(
    const Array<__e8m0, 2, align>& input) {
  Array<__bfloat, 2, align> result;
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{cvt.rn.bf16x2.ue8m0x2 %0, %1;}"
      : "=r"(result_scalar)
      : "h"(input_scalar));
  return result;
}

__device__ __inline__ __bfloat __e8m02bfloat(const __e8m0 b) {
  Array<__e8m0, 2, 1> input = {b, b};
  return __e8m02bfloat(input)[0];
}

DEFINE_CAST_VECN_WITH_VEC2(__e8m02bfloat, __e8m0, __bfloat);

__device__ __inline__ float __e8m02float(const __e8m0 b) {
  return __bfloat2float(__e8m02bfloat(b));
}

template <int n, int align>
__device__ __inline__ Array<float, n, align> __e8m02float(
    const Array<__e8m0, n, align>& input) {
  return __bfloat2float(__e8m02bfloat(input));
}

__device__ __inline__ double __e8m02double(const __e8m0 b) {
  return __bfloat2double(__e8m02bfloat(b));
}

template <int n, int align>
__device__ __inline__ Array<double, n, align> __e8m02double(
    const Array<__e8m0, n, align>& input) {
  return __bfloat2double(__e8m02bfloat(input));
}

__device__ __inline__ __half __e8m02half(const __e8m0 b) {
  return __float2half(__e8m02float(b));
}

template <int n, int align>
__device__ __inline__ Array<__half, n, align> __e8m02half(
    const Array<__e8m0, n, align>& input) {
  return __float2half(__e8m02float(input));
}

// e2m1 casts

// clang-format off
// Disable clang-format because it tries to put the _Pragma("unroll")
// and the for loop on the same line, which doesn't make sense.
#define DEFINE_CAST_VECN_WITH_VEC4(name, from_type, to_type)            \
  template <int n, int align>                                           \
  __device__ __inline__ Array<to_type, n, align> name(                  \
      const Array<from_type, n, align>& input) {                        \
    using InputX2 = Array<from_type, 2, 2>;                             \
    using InputX4 = Array<from_type, 4, 2>;                             \
    static_assert(                                                      \
        sizeof(InputX4) == sizeof(InputX2) * 2,                         \
        "sizeof(InputX4) must be InputX2 * 2");                         \
    using ResultX2 = Array<to_type, 2, 2>;                              \
    using ResultX4 = Array<to_type, 4, 2>;                              \
    static_assert(                                                      \
        sizeof(ResultX4) == sizeof(ResultX2) * 2,                       \
        "sizeof(ResultX4) must be ResultX2 * 2");                       \
    using InputArrayX2 = Array<InputX2, n / 2, align / 2>;              \
    static_assert(                                                      \
        sizeof(InputArrayX2) == sizeof(input),                          \
        "sizeof(InputArrayX2) must be input size");                     \
    using ResultArrayX2 = Array<ResultX2, n / 2, align / 2>;            \
    static_assert(                                                      \
        sizeof(ResultArrayX2) == sizeof(result),                        \
        "sizeof(ResultArrayX2) must be result size");                   \
    const InputArrayX2& inputx2 =                                       \
        reinterpret_cast<const InputArrayX2&>(input);                   \
    Array<to_type, n, align> result;                                    \
    ResultArrayX2& resultx2 = reinterpret_cast<ResultArrayX2&>(result); \
    _Pragma("unroll")                                                   \
    for (int i = 0; i < n / 2; i += 2) {                                \
      if (i + 1 < n / 2) {                                              \
        Array<InputX2, 2, 1> pair = {inputx2[i], inputx2[i + 1]};       \
        static_assert(                                                  \
            sizeof(pair) == sizeof(InputX4),                            \
            "sizeof(pair) must be InputX4 size");                       \
        InputX4& quad = reinterpret_cast<InputX4&>(pair);               \
        ResultX4 res_quad = name(quad);                                 \
        const Array<ResultX2, 2, 1>& res_pair =                         \
            reinterpret_cast<const Array<ResultX2, 2, 1>&>(res_quad);   \
        static_assert(                                                  \
            sizeof(Array<ResultX2, 2, 1>) == sizeof(ResultX4),          \
            "sizeof(Array<ResultX2, 2, 1>) must be ResultX4 size");     \
        resultx2[i] = res_pair[0];                                      \
        resultx2[i + 1] = res_pair[1];                                  \
      } else {                                                          \
        resultx2[i] = name(inputx2[i]);                                 \
      }                                                                 \
    }                                                                   \
    return result;                                                      \
  }
// clang-format on

template <int align>
__device__ __inline__ Array<__e2m1, 2, 2> __float2e2m1(
    const Array<float, 2, align>& input) {
  // Note: Inline PTX can not pass 8-bit register as parameter
  // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  Array<__e2m1, 2, 2> result[2];
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "mov.b16 %0, {byte0, byte0};\n"
      "}"
      : "=h"(result_scalar)
      : "f"(input[0]), "f"(input[1]));
  return result[0];
}

template <int align>
__device__ __inline__ Array<__e2m1, 4, align> __float2e2m1(
    const Array<float, 4, align>& input) {
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

DEFINE_CAST_VECN_WITH_VEC4(__float2e2m1, float, __e2m1);

template <int n, int align>
__device__ __inline__ Array<__e2m1, n, align> __double2e2m1(
    const Array<double, n, align>& input) {
  return __float2e2m1(__to_float(input));
}

template <int n, int align>
__device__ __inline__ Array<__e2m1, n, align> __half2e2m1(
    const Array<__half, n, align>& input) {
  return __float2e2m1(__half2float(input));
}

template <int n, int align>
__device__ __inline__ Array<__e2m1, n, align> __bfloat2e2m1(
    const Array<__bfloat, n, align>& input) {
  return __float2e2m1(__bfloat2float(input));
}

__device__ __inline__ Array<__half, 2, 2> __e2m12half(
    const Array<__e2m1, 2, 2>& input) {
  // Note: Inline PTX can not pass 8-bit register as parameter
  // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  Array<Array<__e2m1, 2, 2>, 2, 1> inputx2 = {input, input};
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&inputx2);
  Array<__half, 2, 2> result;
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1;\n"
      "mov.b16 {byte0, byte1}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "}\n"
      : "=r"(result_scalar)
      : "h"(input_scalar));
  return result;
}

template <int align>
__device__ __inline__ Array<__half, 4, align> __e2m12half(
    const Array<__e2m1, 4, align>& input) {
  // Note: Inline PTX can not pass 8-bit register as parameter
  // https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
  printf("input addr: %lld\n", (int64_t)&input);
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  printf("input scalar: %u\n", input_scalar);
  Array<__half, 4, align> result;
  using HalfX2 = Array<__half, 2, 1>;
  static_assert(sizeof(HalfX2) == 4, "sizeof(HalfX2) must be 4");
  using HalfX2X2 = Array<HalfX2, 2, 1>;
  static_assert(sizeof(HalfX2X2) == 8, "sizeof(HalfX2X2) must be 8");
  HalfX2X2& resultx2 = reinterpret_cast<HalfX2X2&>(result);
  uint32_t& result_scalar0 = *reinterpret_cast<uint32_t*>(&resultx2[0]);
  uint32_t& result_scalar1 = *reinterpret_cast<uint32_t*>(&resultx2[1]);
  asm volatile(
      "{\n"
      ".reg .b8 byte0, byte1;\n"
      "mov.b16 {byte0, byte1}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
      "}\n"
      : "=r"(result_scalar0), "=r"(result_scalar1)
      : "h"(input_scalar));
  return result;
}

DEFINE_CAST_VECN_WITH_VEC4(__e2m12half, __e2m1, __half);

template <int n, int align>
__device__ __inline__ Array<float, n, align> __e2m12float(
    const Array<__e2m1, n, align>& input) {
  return __half2float(__e2m12half(input));
}

template <int n, int align>
__device__ __inline__ Array<double, n, align> __e2m12double(
    const Array<__e2m1, n, align>& input) {
  return __to_double(__e2m12float(input));
}

template <int n, int align>
__device__ __inline__ Array<__bfloat, n, align> __e2m12bfloat(
    const Array<__e2m1, n, align>& input) {
  return __half2bfloat(__e2m12half(input));
}
