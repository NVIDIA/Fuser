// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

__device__ __inline__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "f"(f));
  return val;
}

__device__ __inline__ __half __double2half(const double d) {
  __half val;
  asm("{  cvt.rn.f16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "d"(d));
  return val;
}

__device__ __inline__ __half __int2half(const int i) {
  __half val;
  asm("{  cvt.rn.f16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __half __int2half(const int64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __half __int2half(const uint32_t i) {
  __half val;
  asm("{  cvt.rn.f16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __half __int2half(const uint64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __half __bool2half(const bool b) {
  return __int2half((int)b);
}

__device__ __inline__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ double __half2double(const __half h) {
  double val;
  asm("{  cvt.f64.f16 %0, %1;}\n" : "=d"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __half2int32(const __half h) {
  int val;
  asm("{  cvt.rzi.s32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __half2int(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.s64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __half2uint32(const __half h) {
  int val;
  asm("{  cvt.rzi.u32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __half2uint(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.u64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

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

__device__ __inline__ bool __half2bool(const __half h) {
  return (bool)__half2float(h) != 0;
}

__device__ __inline__ __half __real_then_2half(const std::complex<float> c) {
  return __float2half(std::real(c));
}

__device__ __inline__ __half __real_then_2half(const std::complex<double> c) {
  return __double2half(std::real(c));
}

__device__ __inline__ __bfloat __float2bfloat(const float f) {
  __bfloat val;
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "f"(f));
  return val;
}

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

__device__ __inline__ __bfloat __bool2bfloat(const bool b) {
  return __int2bfloat((int)b);
}

__device__ __inline__ float __bfloat2float(const __bfloat h) {
  float val;
  asm("{  mov.b32 %0, {0,%1};}\n"
      : "=f"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
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

__device__ __inline__ bool __bfloat2bool(const __bfloat h) {
  return (bool)__bfloat2float(h) != 0;
}

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

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<float> c) {
  return __float2bfloat(std::real(c));
}

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<double> c) {
  return __double2bfloat(std::real(c));
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

__device__ __inline__ Array<__e4m3, 2, 1> __float2e4m3(const Array<float, 2, 1>& input) {
  Array<__e4m3, 2, 1> result;
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

__device__ __inline__ __e4m3 __double2e4m3(const double d) {
  return __float2e4m3(d);
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

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  return __float2e4m3(__bfloat2float(h));
}

template <int align>
__device__ __inline__ Array<__half, 2, align> __e4m32half(
    const Array<__e4m3, 2, align>& input) {
  Array<__half, 2, align> result;
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}"
      : "=f"(result_scalar) 
      : "h"(input_scalar));
  return result;
}

__device__ __inline__ __half __e4m32half(const __e4m3 b) {
  Array<__e4m3, 2, 1> input = {b, b};
  return __e4m32half(input)[0];
}

__device__ __inline__ float __e4m32float(const __e4m3 b) {
  return __half2float(__e4m32half(b));
}

__device__ __inline__ double __e4m32double(const __e4m3 b) {
  return __e4m32float(b);
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 b) {
  return __float2bfloat(__e4m32float(b));
}

__device__ __inline__ Array<__e5m2, 2, 1> __float2e5m2(const Array<float, 2, 1>& input) {
  Array<__e5m2, 2, 1> result;
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

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  return __float2e5m2(f);
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

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  return __float2e5m2(__bfloat2float(h));
}

template <int align>
__device__ __inline__ Array<__e5m2, 2, align> __e5m22half(
    const Array<__e5m2, 2, align>& input) {
  Array<__e5m2, 2, align> result;
  const uint16_t& input_scalar = *reinterpret_cast<const uint16_t*>(&input);
  uint32_t& result_scalar = *reinterpret_cast<uint32_t*>(&result);
  asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}"
      : "=f"(result_scalar) 
      : "h"(input_scalar));
  return result;
}

__device__ __inline__ __half __e5m22half(const __e5m2 b) {
  Array<__e5m2, 2, 1> input = {b, b};
  return __e5m22half(input)[0];
}

__device__ __inline__ float __e5m22float(const __e5m2 b) {
  return __half2float(__e5m22half(b));
}

__device__ __inline__ double __e5m22double(const __e5m2 b) {
  return __e5m22float(b);
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 b) {
  return __float2bfloat(__e5m22float(b));
}

__device__ __inline__ Array<__e8m0, 2, 1> __float2e8m0(const Array<float, 2, 1>& input) {
  Array<__e8m0, 2, 1> result;
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

__device__ __inline__ __e8m0 __double2e8m0(const double f) {
  return __float2e8m0(f);
}

__device__ __inline__ __e8m0 __half2e8m0(const __half h) {
  return __float2e8m0(__half2float(h));
}

__device__ __inline__ Array<__e8m0, 2, 1> __bfloat2e8m0(const Array<__bfloat, 2, 1>& input) {
  Array<__e8m0, 2, 1> result;
  uint16_t& result_scalar = *reinterpret_cast<uint16_t*>(&result);
  asm("{cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1, %2;}"
      : "=h"(result_scalar)
      : "r"(input[1]), "r"(input[0]));
  return result;
}

__device__ __inline__ __e8m0 __bfloat2e8m0(const __bfloat h) {
  Array<__bfloat, 2, 1> input = {h, h};
  return __bfloat2e8m0(input)[0];
}

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

__device__ __inline__ float __e8m02float(const __e8m0 b) {
  return __bfloat2float(__e8m02bfloat(b));
}

__device__ __inline__ double __e8m02double(const __e8m0 b) {
  return __bfloat2double(__e8m02bfloat(b));
}

__device__ __inline__ __half __e8m02half(const __e8m0 b) {
  return __float2half(__e8m02float(b));
}
