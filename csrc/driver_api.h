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

#define DECLARE_DRIVER_API_WRAPPER(funcName) \
  extern decltype(::funcName)* funcName;

// List of driver APIs that you want the magic to happen.
#define ALL_DRIVER_API_WRAPPER_CUDA11(fn) \
  fn(cuDeviceGetAttribute);               \
  fn(cuDeviceGetName);                    \
  fn(cuFuncGetAttribute);                 \
  fn(cuFuncSetAttribute);                 \
  fn(cuGetErrorName);                     \
  fn(cuGetErrorString);                   \
  fn(cuLaunchCooperativeKernel);          \
  fn(cuLaunchKernel);                     \
  fn(cuModuleGetFunction);                \
  fn(cuModuleLoadDataEx);                 \
  fn(cuModuleUnload);                     \
  fn(cuOccupancyMaxActiveBlocksPerMultiprocessor)

#if (CUDA_VERSION >= 12000)
#define ALL_DRIVER_API_WRAPPER(fn)   \
  ALL_DRIVER_API_WRAPPER_CUDA11(fn); \
  fn(cuTensorMapEncodeTiled)
#else
#define ALL_DRIVER_API_WRAPPER ALL_DRIVER_API_WRAPPER_CUDA11
#endif

ALL_DRIVER_API_WRAPPER(DECLARE_DRIVER_API_WRAPPER);

#undef DECLARE_DRIVER_API_WRAPPER

} // namespace nvfuser
