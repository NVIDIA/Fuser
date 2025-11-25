// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <iostream>

#define CUDA_CHECK(x)                                                     \
  do {                                                                    \
    CUresult _result = x;                                                 \
    if (_result != CUDA_SUCCESS) {                                        \
      const char* msg;                                                    \
      const char* name;                                                   \
      cuGetErrorName(_result, &name);                                     \
      cuGetErrorString(_result, &msg);                                    \
      std::cout << "CUDA error: " << name << " failed with error " << msg \
                << std::endl;                                             \
      exit(_result);                                                      \
    }                                                                     \
  } while (0)

#define CUDA_RUNTIME_CHECK(x)                                             \
  do {                                                                    \
    cudaError_t _result = x;                                              \
    if (_result != cudaSuccess) {                                         \
      const char* name = cudaGetErrorName(_result);                       \
      const char* msg = cudaGetErrorString(_result);                      \
      std::cout << "CUDA error: " << name << " failed with error " << msg \
                << std::endl;                                             \
      exit(_result);                                                      \
    }                                                                     \
  } while (0)

#define NVRTC_CHECK(x)                                                 \
  do {                                                                 \
    nvrtcResult _result = x;                                           \
    if (_result != NVRTC_SUCCESS) {                                    \
      size_t logSize;                                                  \
      nvrtcGetProgramLogSize(prog, &logSize);                          \
      std::vector<char> log(logSize);                                  \
      nvrtcGetProgramLog(prog, log.data());                            \
      std::cerr << "Compilation Failed:\n" << log.data() << std::endl; \
      exit(_result);                                                   \
    }                                                                  \
  } while (0)
