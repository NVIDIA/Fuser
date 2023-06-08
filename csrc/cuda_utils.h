// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define NVFUSER_NVRTC_SAFE_CALL(x)               \
  do {                                           \
    nvrtcResult _result = x;                     \
    TORCH_INTERNAL_ASSERT(                       \
        _result == NVRTC_SUCCESS,                \
        "NVRTC error: " #x "failed with error ", \
        nvrtcGetErrorString(_result));           \
  } while (0)

#define NVFUSER_CUDA_SAFE_CALL(x)      \
  do {                                 \
    CUresult _result = x;              \
    if (_result != CUDA_SUCCESS) {     \
      const char* msg;                 \
      const char* name;                \
      cuGetErrorName(_result, &name);  \
      cuGetErrorString(_result, &msg); \
      TORCH_INTERNAL_ASSERT(           \
          _result == CUDA_SUCCESS,     \
          "CUDA error: ",              \
          name,                        \
          " failed with error ",       \
          msg);                        \
    }                                  \
  } while (0)

#define NVFUSER_CUDA_RT_SAFE_CALL(x)  \
  do {                                \
    cudaError_t _result = x;          \
    TORCH_INTERNAL_ASSERT(            \
        _result == cudaSuccess,       \
        "CUDA error: ",               \
        cudaGetErrorName(_result),    \
        " failed with error ",        \
        cudaGetErrorString(_result)); \
  } while (0)
