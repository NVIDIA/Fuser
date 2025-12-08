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

// List of driver APIs that you want the magic to happen.
//
// The second argument of `fn` is the **requested** driver API version.
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
// nvFuser is expected to support only CUDA_VERSION >= 12000, so I didn't try
// to go lower than that. When we increase the minimum supported version, we
// can accordingly increase the requested versions. However, we don't have to
// unless new driver capabilities are needed.

#define NVF_MIN_CUDA_FOR_MCAST 13000
#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
#define NVF_FOR_EACH_DRIVER_API_GE_130(fn) \
  /* NVLS multicast */                     \
  fn(cuMulticastCreate, 13000);            \
  fn(cuMulticastAddDevice, 13000);         \
  fn(cuMulticastBindMem, 13000);           \
  fn(cuMulticastGetGranularity, 13000);    \
  fn(cuMulticastUnbind, 13000)
#else
#define NVF_FOR_EACH_DRIVER_API_GE_130(fn)
#endif

#define NVF_FOR_EACH_DRIVER_API(fn)                       \
  fn(cuDeviceGetAttribute, 12000);                        \
  fn(cuDeviceGetName, 12000);                             \
  fn(cuFuncGetAttribute, 12000);                          \
  fn(cuFuncSetAttribute, 12000);                          \
  fn(cuGetErrorName, 12000);                              \
  fn(cuGetErrorString, 12000);                            \
  fn(cuLaunchCooperativeKernel, 12000);                   \
  fn(cuLaunchKernel, 12000);                              \
  fn(cuLaunchKernelEx, 12000);                            \
  fn(cuMemGetAddressRange, 12000);                        \
  fn(cuModuleGetFunction, 12000);                         \
  fn(cuModuleLoadData, 12000);                            \
  fn(cuModuleLoadDataEx, 12000);                          \
  fn(cuModuleUnload, 12000);                              \
  fn(cuOccupancyAvailableDynamicSMemPerBlock, 12000);     \
  fn(cuOccupancyMaxActiveBlocksPerMultiprocessor, 12000); \
  fn(cuOccupancyMaxActiveClusters, 12000);                \
  fn(cuOccupancyMaxPotentialClusterSize, 12000);          \
  fn(cuStreamBatchMemOp, 12000);                          \
  fn(cuStreamWaitValue32, 12000);                         \
  fn(cuStreamWriteValue32, 12000);                        \
  fn(cuTensorMapEncodeTiled, 12000);                      \
  fn(cuDeviceGet, 12000);                                 \
  /* Virtual memory management */                         \
  fn(cuMemAddressReserve, 12000);                         \
  fn(cuMemAddressFree, 12000);                            \
  fn(cuMemMap, 12000);                                    \
  fn(cuMemUnmap, 12000);                                  \
  fn(cuMemSetAccess, 12000);                              \
  fn(cuMemCreate, 12000);                                 \
  fn(cuMemRelease, 12000);                                \
  fn(cuMemExportToShareableHandle, 12000);                \
  fn(cuMemImportFromShareableHandle, 12000);              \
  fn(cuMemGetAllocationGranularity, 12000);               \
  fn(cuMemRetainAllocationHandle, 12000);                 \
  fn(cuMemGetAllocationPropertiesFromHandle, 12000);      \
  fn(cuMemGetAccess, 12000);                              \
  NVF_FOR_EACH_DRIVER_API_GE_130(fn)

#define NVF_DECLARE_DRIVER_API_WRAPPER(fn, requested_version) \
  extern decltype(::fn)* fn

namespace nvfuser {
NVF_FOR_EACH_DRIVER_API(NVF_DECLARE_DRIVER_API_WRAPPER);
} // namespace nvfuser

#undef NVF_DECLARE_DRIVER_API_WRAPPER
