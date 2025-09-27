// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>

// How to lazily load a driver API and invoke it? Just forget about lazy loading
// and write code as if you are using the driver API directly. Magic will
// happen. To understand how the magic works, please refer to the cpp file's doc
// "How does the magic work?"

namespace nvfuser {

// List of driver APIs that you want the magic to happen.
//
// The second argument is the **requested** driver API version.
// It must be lower than or equal to the **actual** driver API version. For max
// compatibility, this requested version should be as low as possible
// while still supporting the capabilities required by nvFuser.
//
// Using the actual driver API version can break forward compatibility.
// Consider the following situation: cuFoo, an imaginary CUDA driver API
// function, is versioned in its symbol name. nvFuser relies on the behavior of
// cuFoo_v2, which is used by CUDA 12. The user upgrades their driver to CUDA
// 13 that introduces cuFoo_v3, which has a different behavior from _v2. In
// this case, `cudaGetDriverEntryPointByVersion(...,13000)` returns cuFoo_v3,
// which breaks nvFuser as is because it still relies on the _v2 behavior.
// Instead, `cudaGetDriverEntryPointByVersion(...,12000)` still returns
// cuFoo_v2 despite of the CUDA 13 upgrade so nvFuser still functions.
//
// nvFuser is expected to support only CUDA_VERSION >= 11000, so I didn't try
// to go lower than that. When we increase the minimum supported version, we
// can accordingly increase the requested versions. However, we don't have to
// unless new driver capabilities are needed.
#define NVF_ALL_DRIVER_API_WRAPPER_VERSION_INDEPENDENT(fn) \
  fn(cuDeviceGetAttribute, 11000);                         \
  fn(cuDeviceGetName, 11000);                              \
  fn(cuFuncGetAttribute, 11000);                           \
  fn(cuFuncSetAttribute, 11000);                           \
  fn(cuGetErrorName, 11000);                               \
  fn(cuGetErrorString, 11000);                             \
  fn(cuLaunchCooperativeKernel, 11000);                    \
  fn(cuLaunchKernel, 11000);                               \
  fn(cuModuleGetFunction, 11000);                          \
  fn(cuModuleLoadData, 11000);                             \
  fn(cuModuleLoadDataEx, 11000);                           \
  fn(cuModuleUnload, 11000);                               \
  fn(cuMemGetAddressRange, 11000);                         \
  fn(cuOccupancyMaxActiveBlocksPerMultiprocessor, 11000);  \
  fn(cuOccupancyAvailableDynamicSMemPerBlock, 11000)

// Stream memory operations (e.g. cuStreamWriteValue32) are specified for both
// 11 and 12+. In CUDA 11, these operations require NVreg_EnableStreamMemOPs=1
// to be explicitly enabled. CUDA 12+ removed this requirement. Therefore, we
// try to request version 12000 whenever it's available.
//
// Details: CUDA 11.7 introduced _v2 of these APIs, which removed the above
// NVreg_EnableStreamMemOPs=1 requirement. In CUDA 12, these _v2 APIs are
// integrated into the vanilla APIs and are therefore removed. Refer to
// https://docs.nvidia.com/cuda/archive/11.7.1/cuda-driver-api/group__CUDA__MEMOP.html
#if (CUDA_VERSION >= 12000)
#define NVF_STREAM_DRIVER_API_WRAPPER(fn) \
  fn(cuStreamWaitValue32, 12000);         \
  fn(cuStreamWriteValue32, 12000)
#elif (CUDA_VERSION >= 11000)
#define NVF_STREAM_DRIVER_API_WRAPPER(fn) \
  fn(cuStreamWaitValue32, 11000);         \
  fn(cuStreamWriteValue32, 11000)
#else
#error "CUDA_VERSION < 11000 isn't supported."
#endif

#if (CUDA_VERSION >= 11080)
#define NVF_DRIVER_API_WRAPPER_CUDA_118(fn) \
  fn(cuOccupancyMaxActiveClusters, 11080);  \
  fn(cuLaunchKernelEx, 11080)
#else
#define NVF_DRIVER_API_WRAPPER_CUDA_118(fn)
#endif

#if (CUDA_VERSION >= 12000)
#define NVF_DRIVER_API_WRAPPER_CUDA_120(fn) fn(cuTensorMapEncodeTiled, 12000)
#else
#define NVF_DRIVER_API_WRAPPER_CUDA_120(fn)
#endif

#define NVF_ALL_DRIVER_API_WRAPPER(fn)                \
  NVF_ALL_DRIVER_API_WRAPPER_VERSION_INDEPENDENT(fn); \
  NVF_STREAM_DRIVER_API_WRAPPER(fn);                  \
  NVF_DRIVER_API_WRAPPER_CUDA_118(fn);                \
  NVF_DRIVER_API_WRAPPER_CUDA_120(fn)

#define DECLARE_DRIVER_API_WRAPPER(funcName, version) \
  extern decltype(::funcName)* funcName

NVF_ALL_DRIVER_API_WRAPPER(DECLARE_DRIVER_API_WRAPPER);

#undef DECLARE_DRIVER_API_WRAPPER

} // namespace nvfuser
