// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// This file defines common helper routines for using CUB.

// CUB uses the CUDA native half-precision types by including these
// files. In addition to the CUB headers themselves, we need them in
// the include path when compiling nvFuser-generated kernels with
// nvrtc. Alternatively, we may want to consider shipping these files
// as part of nvFuser and stop using our own type definitions as it's
// allowed in their license.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nvf {
namespace cub_utils {

// Type utils for interoperability between our own half types and the
// CUDA standard types
template <typename T>
struct CudaType {
  using type = T;

  __device__ inline static T get(const T& t) {
    return t;
  }
};

template <typename T>
struct NvFuserType {
  using type = T;

  __device__ inline static T get(const T& t) {
    return t;
  }
};

#ifdef __NVFUSER_HAS_HALF__
template <>
struct CudaType<__half> {
  using type = __nv_half;

  __device__ inline static type get(const __half& t) {
    return __ushort_as_half(__NVFUSER_HALF_TO_CUS(t));
  }
};
template <>
struct NvFuserType<__half> {
  __device__ inline static __half get(
      const typename CudaType<__half>::type& t) {
    return *(reinterpret_cast<const __half*>(&t));
  }
};
#endif // __NVFUSER_HAS_HALF__

#ifdef __NVFUSER_HAS_BFLOAT__
template <>
struct CudaType<__bfloat> {
  using type = __nv_bfloat16;

  __device__ inline static type get(const __bfloat& t) {
    return __ushort_as_bfloat16(__NVFUSER_BFLOAT_TO_CUS(t));
  }
};
template <>
struct NvFuserType<__bfloat> {
  __device__ inline static __bfloat get(
      const typename CudaType<__bfloat>::type& t) {
    return *(reinterpret_cast<const __bfloat*>(&t));
  }
};
#endif // __NVFUSER_HAS_BFLOAT__

} // namespace cub_utils
} // namespace nvf
