// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#define DEFINE_CAST_VEC1(name, from_type, to_type) \
  __device__ __inline__ Array<to_type, 1, 1> name( \
      const Array<from_type, 1, 1>& input) {       \
    Array<to_type, 1, 1> result;                   \
    result[0] = name(input[0]);                    \
    return result;                                 \
  }

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
    _Pragma("unroll") for (int i = 0; i < n; i += 1) {   \
      result[i] = name(input[i]);                        \
    }                                                    \
    return result;                                       \
  }

__device__ __inline__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "f"(f));
  return val;
}

DEFINE_CAST_VEC1(__float2half, float, __half);

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

__device__ __inline__ double __half2double(const __half h) {
  double val;
  asm("{  cvt.f64.f16 %0, %1;}\n" : "=d"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

DEFINE_CAST_NAIVE_VECN(__half2double, __half, double);

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

__device__ __inline__ __bfloat __float2bfloat(const float f) {
  __bfloat val;
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "f"(f));
  return val;
}

DEFINE_CAST_VEC1(__float2bfloat, float, __bfloat);

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

// see NOTE [ fp8 cast optimization ]
__device__ __inline__ __e8m0 __float2e8m0(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e8m0 val;
  asm("{cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;}"
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
