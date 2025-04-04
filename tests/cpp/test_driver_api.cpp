// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <cuda_utils.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using DriverApiTest = NVFuserTest;

TEST_F(DriverApiTest, WriteValue) {
  constexpr cuuint32_t value = 3;
  CUdeviceptr pDevice;
  cudaStream_t stream;

  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc((void**)&pDevice, sizeof(cuuint32_t)));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&stream));
  NVFUSER_CUDA_SAFE_CALL(
      cuStreamWriteValue32(stream, pDevice, value, /*flag=*/0));
}

} // namespace nvfuser
