// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#define NVFUSER_DEFINE_MAGIC_ZERO          \
  __shared__ int nvfuser_zero_s;           \
  if (threadIdx.x == 0)                    \
    nvfuser_zero_s = 0;                    \
  __syncthreads();                         \
  atomicMin(&nvfuser_zero_s, threadIdx.x); \
  int nvfuser_zero = nvfuser_zero_s;

#define NVFUSER_UPDATE_MAGIC_ZERO \
  do {                            \
    nvfuser_zero <<= 1;           \
  } while (0);

#ifdef __NVCC__
#include <assert.h>
#endif // __NVCC__

__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int b) {
  return ceilDiv(a, (int64_t)b);
}

__device__ constexpr int64_t ceilDiv(int a, int64_t b) {
  return ceilDiv((int64_t)a, b);
}

__device__ constexpr double ceilDiv(double a, double b) {
  return std::ceil(a / b);
}

__device__ constexpr double ceilDiv(double a, int64_t b) {
  return std::ceil(a / b);
}

__device__ constexpr double ceilDiv(int64_t a, double b) {
  return std::ceil(a / b);
}

// Monotonic and precise lerp is described here:
// https://math.stackexchange.com/a/1798323
__device__ double lerp(double start, double end, double weight) {
  if (weight < 0.5) {
    return start + weight * (end - start);
  } else {
    return end - (end - start) * (1.0 - weight);
  }
}

__device__ float lerp(float start, float end, float weight) {
  if (weight < 0.5f) {
    return start + weight * (end - start);
  } else {
    return end - (end - start) * (1.0f - weight);
  }
}

__device__ float lerp(float start, float end, double weight) {
  return lerp(start, end, static_cast<float>(weight));
}

__device__ constexpr int max(int a, int b) {
  return a > b ? a : b;
}

__device__ constexpr int64_t max(int64_t a, int b) {
  return a > (int64_t)b ? a : (int64_t)b;
}

__device__ constexpr int64_t max(int a, int64_t b) {
  return (int64_t)a > b ? (int64_t)a : b;
}

__device__ constexpr int64_t max(int64_t a, int64_t b) {
  return a > b ? a : b;
}

__device__ double fmax(double a, double b) {
  // check and propagate NaN
  if (a != a) {
    return a;
  } else { // If b is nan, it will be returned in the next line
    return a > b ? a : b;
  }
}

__device__ float fmax(float a, float b) {
  // check and propagate NaN
  if (a != a) {
    return a;
  } else { // If b is nan, it will be returned in the next line
    return a > b ? a : b;
  }
}

__device__ constexpr int min(int a, int b) {
  return a > b ? b : a;
}

__device__ constexpr int64_t min(int64_t a, int b) {
  return (int64_t)a > b ? b : (int64_t)a;
}

__device__ constexpr int64_t min(int a, int64_t b) {
  return a > (int64_t)b ? (int64_t)b : a;
}

__device__ constexpr int64_t min(int64_t a, int64_t b) {
  return a > b ? b : a;
}

__device__ double fmin(double a, double b) {
  // check and propagate NaN
  if (b != b) {
    return b;
  } else { // If a is nan, it will be returned in the next line
    return a > b ? b : a;
  }
}

__device__ float fmin(float a, float b) {
  // check and propagate NaN
  if (b != b) {
    return b;
  } else { // If a is nan, it will be returned in the next line
    return a > b ? b : a;
  }
}

__device__ constexpr int alignBufferSize(int buffer, int size) {
  return (buffer + (size - 1)) & ~(size - 1);
}

__device__ double clamp(double x, double minv, double maxv) {
  return fmin(fmax(x, minv), maxv);
}

__device__ float clamp(float x, double minv, double maxv) {
  return fmin(fmax((double)x, minv), maxv);
}

__device__ int clamp(int x, int64_t minv, int64_t maxv) {
  return min(max((int64_t)x, minv), maxv);
}

__device__ int64_t clamp(int64_t x, int64_t minv, int64_t maxv) {
  return min(max(x, minv), maxv);
}

__device__ double frac(double x) {
  return x - trunc(x);
}

__device__ float frac(float x) {
  return x - trunc(x);
}

__device__ double reciprocal(double x) {
  return 1 / x;
}

__device__ float reciprocal(float x) {
  return 1 / x;
}

__device__ double relu(double x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(float x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int64_t x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int x) {
  return x <= 0 ? 0 : x;
}

__device__ double remainder(double a, double b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ float remainder(float a, float b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

__device__ double silu(double x) {
  return x * sigmoid(x);
}

__device__ float silu(float x) {
  return x * sigmoid(x);
}

__device__ double threshold(double x, double t, double v) {
  return x <= t ? v : x;
}

__device__ float threshold(float x, double t, double v) {
  return x <= t ? v : x;
}

__device__ int threshold(int x, int64_t t, int64_t v) {
  return x <= t ? v : x;
}

__device__ int64_t threshold(int64_t x, int64_t t, int64_t v) {
  return x <= t ? v : x;
}

__device__ constexpr int64_t remainder(int64_t a, int64_t b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ constexpr int remainder(int a, int b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ constexpr int64_t fmod(int64_t a, int64_t b) {
  return a % b;
}

__device__ constexpr int fmod(int a, int b) {
  return a % b;
}

__device__ constexpr double fmod(double a, double b) {
  return ::fmod(a, b);
}

__device__ constexpr float fmod(float a, float b) {
  return ::fmod(a, b);
}

__device__ constexpr double nextafter(double a, double b) {
  return ::nextafter(a, b);
}

__device__ constexpr float nextafter(float a, float b) {
  return ::nextafterf(a, b);
}

template <typename T>
__device__ T pow(T a, T b) {
  if (b < 0) {
    if (a == 1) {
      return 1;
    } else if (a == -1) {
      auto negative = (-b) % static_cast<T>(2);
      return negative ? -1 : 1;
    } else {
      return 0;
    }
  } else {
    T result = 1;
    while (b) {
      if (b & 1) {
        result *= a;
      }
      b /= 2;
      a *= a;
    }
    return result;
  }
}

template __device__ int pow<int>(int a, int b);
template __device__ int64_t pow<int64_t>(int64_t a, int64_t b);

template <>
__device__ float pow<float>(float a, float b) {
  return ::pow(a, b);
}

template <>
__device__ double pow<double>(double a, double b) {
  return ::pow(a, b);
}

__device__ float pow(float a, int b) {
  return pow(a, (float)b);
}

__device__ double pow(double a, int b) {
  return pow(a, (double)b);
}

__device__ float pow(float a, int64_t b) {
  return pow(a, (float)b);
}

__device__ double pow(double a, int64_t b) {
  return pow(a, (double)b);
}

__device__ int64_t pow(int64_t a, int b) {
  return pow(a, (int64_t)b);
}

__device__ int64_t pow(int a, int64_t b) {
  return pow((int64_t)a, b);
}

__device__ double rsqrt(double z) {
  return ::rsqrt(z);
}

__device__ float rsqrt(float z) {
  return ::rsqrtf(z);
}

__device__ int rsqrt(int z) {
  return ::rsqrtf((float)z);
}

__device__ int64_t rsqrt(int64_t z) {
  return ::rsqrt((double)z);
}

__device__ double signbit(double a) {
  return ::signbit(a);
}

__device__ float signbit(float a) {
  return ::signbit(a);
}

__device__ int signbit(int a) {
  return a < 0;
}

__device__ int64_t signbit(int64_t a) {
  return a < 0;
}

// Reference:
// https://en.wikipedia.org/wiki/Euclidean_algorithm#Implementations
// https://github.com/pytorch/pytorch/blob/c9f4f01981fd73fcc7c27676cc50230cd1b5bc22/aten/src/ATen/native/Math.h#L1232
template <typename T>
__device__ T gcd(T a, T b) {
  a = abs(a);
  b = abs(b);
  while (b != 0) {
    auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template <int size, int align = size>
struct alignas(align) TypelessData {
  int8_t data[size];

  template <typename T, std::enable_if_t<sizeof(T) == size, int> _ = 0>
  TypelessData(T x) {
    *reinterpret_cast<T*>(data) = x;
  }

  template <typename T, std::enable_if_t<sizeof(T) == size, int> _ = 0>
  operator T() {
    return *reinterpret_cast<T*>(data);
  }
};

template <typename T>
TypelessData<sizeof(T), alignof(T)> erase_type(T x) {
  return x;
}

template <typename T>
bool isfinite(T x) {
  return ::isfinite(x);
}

// ref:
// https://github.com/NVIDIA/cutlass/blob/6fbc0d33800008d3180d3fefed4e1a653e5f72a0/include/cutlass/bfloat16.h#L213
template <>
bool isfinite<__bfloat>(__bfloat x) {
  const auto exponent_biased = int((x.raw() >> 7) & 0x0ff);
  return exponent_biased != 0x0ff;
}

// ref:
// https://github.com/NVIDIA/cutlass/blob/6fbc0d33800008d3180d3fefed4e1a653e5f72a0/include/cutlass/half.h#L511
template <>
bool isfinite<__half>(__half x) {
  const auto exponent_biased = int((x.raw() >> 10) & 0x1f);
  return exponent_biased != 0x1f;
}

template <typename T>
bool isinf(T x) {
  return ::isinf(x);
}

////////////////////////////////////////////////////////////
// TODO: the following overloads are only needed for CUDA //
// 10.2 Please remove when CUDA 10.2 support is dropped   //
////////////////////////////////////////////////////////////

bool isinf(int64_t x) {
  return false;
}

bool isinf(int x) {
  return false;
}

bool isinf(short x) {
  return false;
}

bool isinf(char x) {
  return false;
}

bool isinf(unsigned char x) {
  return false;
}

bool isinf(bool x) {
  return false;
}

bool isfinite(int64_t x) {
  return true;
}

bool isfinite(int x) {
  return true;
}

bool isfinite(short x) {
  return true;
}

bool isfinite(char x) {
  return true;
}

bool isfinite(unsigned char x) {
  return true;
}

bool isfinite(bool x) {
  return true;
}

////////////////////////////////////////////////////////////
//                        End TODO                        //
////////////////////////////////////////////////////////////

template <typename T>
bool isnan(T x) {
  return x != x;
}

template <typename T>
bool isneginf(T x) {
  return x < 0 && isinf(x);
}

template <typename T>
bool isposinf(T x) {
  return x > 0 && isinf(x);
}

template <typename T>
bool isreal(T x) {
  return true;
}

// Return the current value of the cycle counter
__device__ inline int64_t readCycleCounter() {
  // Ensures preceding memory operations are completed. Doing this
  // would make sense for measuring elapsed times enclosed with this
  // function.
  __threadfence();
  return clock64();
}

__device__ float print_impl(const char* name, float value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ double print_impl(const char* name, double value) {
  printf(
      "%s = %lf @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ int print_impl(const char* name, int value) {
  printf(
      "%s = %d @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ int64_t print_impl(const char* name, int64_t value) {
  printf(
      "%s = %ld @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ bool print_impl(const char* name, bool value) {
  printf(
      "%s = %s @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value ? "true" : "false",
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ __half print_impl(const char* name, __half value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      __half2float(value),
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

#if __CUDACC_VER_MAJOR__ >= 11
__device__ __bfloat print_impl(const char* name, __bfloat value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      __bfloat2float(value),
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}
#endif

#define print(...) print_impl(#__VA_ARGS__, (__VA_ARGS__))
