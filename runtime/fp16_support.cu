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

struct __half;
__device__ __inline__ __half __float2half(const float);

struct __align__(2) __half {
  __half() = default;

  __half(const __half& other) {
    __x = other.__x;
  }

  __half(const __half&& other) {
    __x = other.__x;
  }

  __half(const volatile __half& other) {
    __x = other.__x;
  }

  __half(const volatile __half&& other) {
    __x = other.__x;
  }

  // Note: not returning reference for `__half::operator=`
  // Doing so would requires us to return `volatile __half&` for the volatile
  // variants, which would trigger a gcc warning `implicit dereference will not
  // access object of type ‘volatile S’ in statement`
  __device__ void operator=(const __half& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __half&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half& other) {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half&& other) {
    __x = other.__x;
  }

  __device__ void operator=(const __half& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const __half&& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half& other) volatile {
    __x = other.__x;
  }

  __device__ void operator=(const volatile __half&& other) volatile {
    __x = other.__x;
  }

  __device__ __half(const float f) {
    __x = __float2half(f).__x;
  }

  __device__ uint16_t raw() const {
    return __x;
  }

 protected:
  unsigned short __x;
};

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

__device__ __inline__ bool __heq(const __half a, const __half b) {
  // From cuda_fp16.hpp
  unsigned short val;
  asm("{ .reg .pred __$temp3;\n"
      "  setp.eq.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(a)), "h"(__NVFUSER_HALF_TO_CUS(b)));
  return (val != 0U) ? true : false;
}

__device__ __inline__ __half operator|(const __half x, const __half y) {
  __half val;
  asm("{  or.b16 %0, %1, %2;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(__NVFUSER_HALF_TO_CUS(x)), "h"(__NVFUSER_HALF_TO_CUS(y)));
  return val;
}

#define __NVFUSER_HAS_HALF__
