// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <tests/cpp/topk_test_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace nvfuser {

using TopkDeviceFuncTest = NVFuserTest;

// Basic functionality test
TEST_F(TopkDeviceFuncTest, BasicTopkFloat) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const int k = 3;

  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto values_tensor = at::empty(
      {k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto indices_tensor =
      at::empty({k}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test largest
  launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      k,
      true);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, true));

  // Test smallest
  launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      k,
      false);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, false));
}

// Variable k values test
TEST_F(TopkDeviceFuncTest, VariableKValues) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto values_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto indices_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test different k values: 1, 2, 4, 8
  for (int k : {1, 2, 4, 8}) {
    launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        values_tensor.data_ptr<float>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(
        validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, true))
        << "Failed for k=" << k;
  }
}

// Data type support test
TEST_F(TopkDeviceFuncTest, DataTypeSupport) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const int k = 3;

  auto input_tensor = at::randint(
      -100,
      100,
      {total_elements},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test double
  {
    auto input_tensor_cast = input_tensor.to(at::kDouble);
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<double, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor_cast.data_ptr<double>(),
        values_tensor.data_ptr<double>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder(
        input_tensor_cast, values_tensor, indices_tensor, k, true));
  }

  // Test int
  {
    auto input_tensor_cast = input_tensor.to(at::kInt);
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<int, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor_cast.data_ptr<int>(),
        values_tensor.data_ptr<int>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder(
        input_tensor_cast, values_tensor, indices_tensor, k, true));
  }

  // Test int64_t
  {
    auto input_tensor_cast = input_tensor.to(at::kLong);
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<int64_t, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor_cast.data_ptr<int64_t>(),
        values_tensor.data_ptr<int64_t>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder(
        input_tensor_cast, values_tensor, indices_tensor, k, true));
  }

  // Test bfloat16
  {
    auto input_tensor_cast = input_tensor.to(at::kBFloat16);
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<__nv_bfloat16, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<__nv_bfloat16*>(input_tensor_cast.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(values_tensor.data_ptr()),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder(
        input_tensor_cast, values_tensor, indices_tensor, k, true));
  }
}

// Edge cases test
TEST_F(TopkDeviceFuncTest, EdgeCases) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Test all same values
  {
    std::vector<float> test_data(total_elements, 3.0f);
    const int k = 3;

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {k}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        values_tensor.data_ptr<float>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    // Validate correctness and verify all k values should be 3.0f
    EXPECT_TRUE(validateTopkOrder(
        input_tensor, values_tensor, indices_tensor, k, true));

    // Additional validation: all k values should be 3.0f
    EXPECT_TRUE((values_tensor == 3.0f).all().item<bool>()) << "Value mismatch";
  }
}

// Multi-dimensional 2D test
TEST_F(TopkDeviceFuncTest, MultiDim2dTopkFloat) {
  const int ITEMS_PER_THREAD = 2;
  const int total_elements =
      4 * 2 * ITEMS_PER_THREAD; // 4x2 block = 8 threads, 2 items per thread =
                                // 16 elements
  const int k = 5;

  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto values_tensor = at::empty(
      {k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto indices_tensor =
      at::empty({k}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test largest
  launchMultiDim2dTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      k,
      true);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, true));

  // Test smallest
  launchMultiDim2dTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      k,
      false);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, false));
}

// Multi-dimensional 3D test
TEST_F(TopkDeviceFuncTest, MultiDim3dTopkFloat) {
  const int ITEMS_PER_THREAD = 2;
  const int total_elements =
      2 * 2 * 2 * ITEMS_PER_THREAD; // 2x2x2 block = 8 threads, 2 items per
                                    // thread = 16 elements
  const int k = 6;

  auto input_tensor = at::randn(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto values_tensor = at::empty(
      {k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto indices_tensor =
      at::empty({k}, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test largest
  launchMultiDim3dTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      k,
      true);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, true));

  // Test smallest
  launchMultiDim3dTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      k,
      false);

  EXPECT_TRUE(
      validateTopkOrder(input_tensor, values_tensor, indices_tensor, k, false));
}

} // namespace nvfuser
