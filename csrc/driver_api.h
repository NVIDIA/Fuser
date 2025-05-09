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

#define DECLARE_DRIVER_API_WRAPPER(funcName, version) \
  extern decltype(::funcName)* funcName

// List of driver APIs that you want the magic to happen.
//
// The second argument is the CUDA_VERSION **requested** for the driver API.
// It's fine if the **actual** CUDA_VERSION is larger than this. For max
// compatibility, this requested CUDA_VERSION should be as low as possible, as
// long as it supports the capabilities that nvFuser requires.
//
// nvFuser is expected to support only CUDA_VERSION >= 11000, so I didn't try
// to go lower than that.
#define ALL_DRIVER_API_WRAPPER_CUDA(fn) \
  fn(cuDeviceGetAttribute, 11000);      \
  fn(cuDeviceGetName, 11000);           \
  fn(cuFuncGetAttribute, 11000);        \
  fn(cuFuncSetAttribute, 11000);        \
  fn(cuGetErrorName, 11000);            \
  fn(cuGetErrorString, 11000);          \
  fn(cuLaunchCooperativeKernel, 11000); \
  fn(cuLaunchKernel, 11000);            \
  fn(cuModuleGetFunction, 11000);       \
  fn(cuModuleLoadDataEx, 11000);        \
  fn(cuModuleUnload, 11000);            \
  fn(cuMemGetAddressRange, 11000);      \
  fn(cuOccupancyMaxActiveBlocksPerMultiprocessor, 11000)

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
#define ALL_DRIVER_API_WRAPPER(fn) \
  ALL_DRIVER_API_WRAPPER_CUDA(fn); \
  fn(cuStreamWaitValue32, 12000);  \
  fn(cuStreamWriteValue32, 12000); \
  fn(cuTensorMapEncodeTiled, 12000)
#elif (CUDA_VERSION >= 11000)
#define ALL_DRIVER_API_WRAPPER(fn) \
  ALL_DRIVER_API_WRAPPER_CUDA(fn); \
  fn(cuStreamWaitValue32, 11000);  \
  fn(cuStreamWriteValue32, 11000)
#else
#error "CUDA_VERSION < 11000 isn't supported."
#endif

ALL_DRIVER_API_WRAPPER(DECLARE_DRIVER_API_WRAPPER);

#undef DECLARE_DRIVER_API_WRAPPER

} // namespace nvfuser
