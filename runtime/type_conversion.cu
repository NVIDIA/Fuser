// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on


// This should be specialized for each combination of types that cannot be
// automatically cast by static_cast
template <typename TO, typename FROM>
__device__ __inline__ TO castFloating(FROM x) {
  if constexpr (std::is_same<FROM, TO>::value) {
    return x;
  }
  // Try static cast
  return static_cast<TO>(x);
}

template <> __device__ __inline__ __half castFloating(double x) { return __double2half(x); }
template <> __device__ __inline__ __half castFloating(float x) { return __float2half(x); }
template <> __device__ __inline__ float castFloating(__half x) { return __half2float(x); }
template <> __device__ __inline__ double castFloating(__half x) { return (double)__half2float(x); }

template <> __device__ __inline__ __bfloat castFloating(double x) { return __double2bfloat(x); }
template <> __device__ __inline__ __bfloat castFloating(float x) { return __float2bfloat(x); }
template <> __device__ __inline__ float castFloating(__bfloat x) { return __bfloat2float(x); }
template <> __device__ __inline__ double castFloating(__bfloat x) { return (double)__bfloat2float(x); }

template <> __device__ __inline__ __bfloat castFloating(__half x) { return __half2bfloat(x); }
template <> __device__ __inline__ __half castFloating(__bfloat x) { return __bfloat2half(x); }
