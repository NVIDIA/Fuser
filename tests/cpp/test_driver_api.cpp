// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <cuda_utils.h>
#include <driver_api.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using DriverApiTest = NVFuserTest;

TEST_F(DriverApiTest, WriteValue) {
  void* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, sizeof(uint32_t)));

  cudaStream_t stream;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&stream));

  constexpr uint32_t kValueToWrite = 3;
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream, reinterpret_cast<CUdeviceptr>(d_ptr), kValueToWrite, /*flag=*/0));

  uint32_t value_received;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      &value_received, d_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));

  EXPECT_EQ(value_received, kValueToWrite);
}

} // namespace nvfuser
