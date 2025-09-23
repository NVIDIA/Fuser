// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <tests/cpp/cluster_runtime_test/cluster_test_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ATen/cuda/CUDAContext.h>

namespace nvfuser {

using ClusterDeviceFuncTest = NVFuserTest;

// Basic functionality test for storeSharedRemote<float>
TEST_F(ClusterDeviceFuncTest, BasicStoreSharedRemoteFloat) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 32;
  constexpr int total_elements = num_blocks * threads_per_block;

  // Create input tensor with sequential values for easy verification
  auto input_tensor = at::arange(
      1.0f,
      total_elements + 1.0f,
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  // Launch test kernel with cluster configuration (2x1x1 cluster)
  launchStoreSharedRemoteTestKernel<float, threads_per_block, num_blocks>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      2, // cluster_x
      1, // cluster_y
      1 // cluster_z
  );
  // Validate the results
  validateClusterStoreResult(input_tensor, output_tensor, 2);
}

// Basic functionality test for storeSharedRemote<double>
TEST_F(ClusterDeviceFuncTest, BasicStoreSharedRemoteDouble) {
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 32;
  constexpr int total_elements = num_blocks * threads_per_block;

  // Create input tensor with sequential values for easy verification
  auto input_tensor = at::arange(
      1.0,
      total_elements + 1.0,
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  // Launch test kernel with cluster configuration (2x1x1 cluster)
  launchStoreSharedRemoteTestKernel<double, threads_per_block, num_blocks>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<double>(),
      output_tensor.data_ptr<double>(),
      2, // cluster_x
      1, // cluster_y
      1 // cluster_z
  );
  // Validate the results
  validateClusterStoreResult(input_tensor, output_tensor, 2);
}
} // namespace nvfuser
