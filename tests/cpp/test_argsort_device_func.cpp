// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <tests/cpp/argsort_test_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

namespace nvfuser {

using ArgSortDeviceFuncTest = NVFuserTest;

// Parameterized test fixture for comprehensive validation
using ArgSortComprehensiveTest = NVFuserFixtureParamTest<std::pair<int, int>>;

template <typename DataT>
std::vector<DataT> getVector(at::Tensor tensor) {
  NVF_ERROR_EQ(tensor.dim(), 1);
  if (tensor.dtype() == at::kBFloat16) {
    tensor = tensor.to(at::kFloat);
  }
  auto cpu_tensor = tensor.cpu();
  auto total_elements = tensor.size(0);
  return std::vector<DataT>(
      cpu_tensor.data_ptr<DataT>(),
      cpu_tensor.data_ptr<DataT>() + total_elements);
}

// Helper function to validate sorting correctness
template <typename DataT>
bool validateArgsortOrder(
    const at::Tensor& input_tensor,
    const at::Tensor& indices_tensor,
    bool descending = false) {
  NVF_ERROR_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto input_data = getVector<DataT>(input_tensor);
  auto indices = getVector<int64_t>(indices_tensor);

  int64_t n = input_data.size();

  // Check valid range
  for (int64_t i = 0; i < n; i++) {
    if (indices[i] < 0 || indices[i] >= n) {
      return false;
    }
  }

  // Check permutation
  std::vector<bool> used(n, false);
  for (int64_t i = 0; i < n; i++) {
    if (used[indices[i]]) {
      return false;
    }
    used[indices[i]] = true;
  }

  // Check sorting order
  for (int64_t i = 1; i < n; i++) {
    DataT prev_val = input_data[indices[i - 1]];
    DataT curr_val = input_data[indices[i]];

    if (descending) {
      if (curr_val > prev_val) {
        return false;
      }
    } else {
      if (curr_val < prev_val) {
        return false;
      }
    }
  }

  return true;
}

// Basic functionality test
TEST_F(ArgSortDeviceFuncTest, BasicArgsortFloat) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test ascending
  launchBasicArgsortTestKernel<float>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      ITEMS_PER_THREAD,
      false);

  EXPECT_TRUE(validateArgsortOrder<float>(input_tensor, output_tensor, false));

  // Test descending
  launchBasicArgsortTestKernel<float>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      ITEMS_PER_THREAD,
      true);

  EXPECT_TRUE(validateArgsortOrder<float>(input_tensor, output_tensor, true));
}

// Data type support test
TEST_F(ArgSortDeviceFuncTest, DataTypeSupport) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Test double
  {
    auto input_tensor = at::randn(
        {total_elements},
        at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicArgsortTestKernel<double>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<double>(),
        output_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(
        validateArgsortOrder<double>(input_tensor, output_tensor, false));
  }

  // Test int
  {
    auto input_tensor = at::randint(
        -100,
        100,
        {total_elements},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicArgsortTestKernel<int>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int>(),
        output_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(validateArgsortOrder<int>(input_tensor, output_tensor, false));
  }

  // Test int64_t
  {
    auto input_tensor = at::randint(
        -100,
        100,
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicArgsortTestKernel<int64_t>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int64_t>(),
        output_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(
        validateArgsortOrder<int64_t>(input_tensor, output_tensor, false));
  }

  // Test bfloat16
  {
    auto input_tensor = at::randn(
        {total_elements},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicArgsortTestKernel<__nv_bfloat16>(
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<__nv_bfloat16*>(input_tensor.data_ptr()),
        output_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(
        validateArgsortOrder<float>(input_tensor, output_tensor, false));
  }
}

// Multi-dimensional block tests
TEST_F(ArgSortDeviceFuncTest, MultiDimensionalBlocks) {
  const int ITEMS_PER_THREAD = 2;

  // Test 2D block: 4x2x1 (8 threads total)
  {
    const int total_elements = 8 * ITEMS_PER_THREAD;

    auto input_tensor = at::randn(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchMultiDim2dArgsortTestKernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<int64_t>(),
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(
        validateArgsortOrder<float>(input_tensor, output_tensor, false));
  }

  // Test 3D block: 2x2x2 (8 threads total)
  {
    const int total_elements = 8 * ITEMS_PER_THREAD;

    auto input_tensor = at::randn(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchMultiDim3dArgsortTestKernel<float>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<int64_t>(),
        ITEMS_PER_THREAD,
        false);

    EXPECT_TRUE(
        validateArgsortOrder<float>(input_tensor, output_tensor, false));
  }
}

// Parameterized comprehensive validation test
TEST_P(ArgSortComprehensiveTest, ComprehensiveValidation) {
  auto [BLOCK_SIZE, ITEMS_PER_THREAD] = GetParam();
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Generate random test data using at::randn
  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test ascending
  launchBasicArgsortTestKernel<float>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<nvfuser_index_t>(),
      BLOCK_SIZE,
      ITEMS_PER_THREAD,
      false);

  EXPECT_TRUE(validateArgsortOrder<float>(input_tensor, output_tensor, false));

  // Test descending
  launchBasicArgsortTestKernel<float>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<nvfuser_index_t>(),
      BLOCK_SIZE,
      ITEMS_PER_THREAD,
      true);

  EXPECT_TRUE(validateArgsortOrder<float>(input_tensor, output_tensor, true));
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(
    BlockSizeAndItemsPerThread,
    ArgSortComprehensiveTest,
    testing::Values(
        // Block size 32
        std::make_pair(32, 1),
        std::make_pair(32, 2),
        std::make_pair(32, 3),
        std::make_pair(32, 4),
        std::make_pair(32, 5),
        // Block size 64
        std::make_pair(64, 1),
        std::make_pair(64, 2),
        std::make_pair(64, 3),
        std::make_pair(64, 4),
        std::make_pair(64, 5),
        // Block size 128
        std::make_pair(128, 1),
        std::make_pair(128, 2),
        std::make_pair(128, 3),
        std::make_pair(128, 4),
        std::make_pair(128, 5),
        // Block size 256
        std::make_pair(256, 1),
        std::make_pair(256, 2),
        std::make_pair(256, 3),
        std::make_pair(256, 4),
        std::make_pair(256, 5)),
    [](const testing::TestParamInfo<std::pair<int, int>>& info) {
      return "BlockSize" + std::to_string(info.param.first) +
          "_ItemsPerThread" + std::to_string(info.param.second);
    });

} // namespace nvfuser
