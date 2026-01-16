// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ATen/ATen.h>
#include <ATen/TensorOptions.h>
#include <ATen/cuda/CUDAContext.h>

#include <gtest/gtest.h>

#include "tests/cpp/cluster_runtime_test/cluster_test_helper.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

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
      input_tensor.data_ptr<float>(), output_tensor.data_ptr<float>());
  // Validate the results
  validateClusterStoreResult(input_tensor, output_tensor, 2);
}

// Basic functionality test for storeSharedRemote<double>
TEST_F(ClusterDeviceFuncTest, BasicStoreSharedRemoteDouble) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
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
      input_tensor.data_ptr<double>(), output_tensor.data_ptr<double>());
  // Validate the results
  validateClusterStoreResult(input_tensor, output_tensor, 2);
}

// Cluster reduction test for float
TEST_F(ClusterDeviceFuncTest, ClusterReduceFloatAllReduce) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 128;
  constexpr int total_elements = num_blocks * threads_per_block;

  auto input_tensor = at::arange(
      1.0f,
      total_elements + 1.0f,
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchClusterReduceTestKernel<float, threads_per_block, num_blocks, true>(
      input_tensor.data_ptr<float>(), output_tensor.data_ptr<float>());

  validateClusterReduceResult(input_tensor, output_tensor, true);
}

// Cluster reduction test for double
TEST_F(ClusterDeviceFuncTest, ClusterReduceDoubleAllReduce) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 128;
  constexpr int total_elements = num_blocks * threads_per_block;

  auto input_tensor = at::arange(
      1.0,
      total_elements + 1.0,
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  launchClusterReduceTestKernel<double, threads_per_block, num_blocks, true>(
      input_tensor.data_ptr<double>(), output_tensor.data_ptr<double>());

  validateClusterReduceResult(input_tensor, output_tensor, true);
}

// Cluster reduction test for float - returns single scalar
TEST_F(ClusterDeviceFuncTest, ClusterReduceFloatNotAllReduce) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 128;
  constexpr int total_elements = num_blocks * threads_per_block;

  auto input_tensor = at::arange(
      1.0f,
      total_elements + 1.0f,
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  // Output is now a single scalar value
  auto output_scalar = at::empty(
      {1}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchClusterReduceTestKernel<float, threads_per_block, num_blocks, false>(
      input_tensor.data_ptr<float>(), output_scalar.data_ptr<float>());

  validateClusterReduceResult(input_tensor, output_scalar, false);
}

// Cluster reduction test for double - returns single scalar
TEST_F(ClusterDeviceFuncTest, ClusterReduceDoubleNotAllReduce) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr int num_blocks = 2;
  constexpr int threads_per_block = 128;
  constexpr int total_elements = num_blocks * threads_per_block;

  auto input_tensor = at::arange(
      1.0,
      total_elements + 1.0,
      at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  // Output is now a single scalar value
  auto output_scalar = at::empty(
      {1}, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

  launchClusterReduceTestKernel<double, threads_per_block, num_blocks, false>(
      input_tensor.data_ptr<double>(), output_scalar.data_ptr<double>());

  validateClusterReduceResult(input_tensor, output_scalar, false);
}

} // namespace nvfuser
