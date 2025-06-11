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

// Helper function to validate topk correctness
template <typename DataT>
bool validateTopkOrder(
    const at::Tensor& input_tensor,
    const at::Tensor& values_tensor,
    const at::Tensor& indices_tensor,
    int64_t k,
    bool largest = true) {
  NVF_ERROR_EQ(cudaDeviceSynchronize(), cudaSuccess);

  auto input_data = getVector<DataT>(input_tensor);
  auto output_values = getVector<DataT>(values_tensor);
  auto output_indices = getVector<int64_t>(indices_tensor);

  // Check that we have k valid results
  if (static_cast<int64_t>(output_values.size()) < k ||
      static_cast<int64_t>(output_indices.size()) < k) {
    return false;
  }

  // Check valid indices range
  for (int64_t i = 0; i < k; i++) {
    if (output_indices[i] < 0 ||
        output_indices[i] >= static_cast<int64_t>(input_data.size())) {
      return false;
    }
  }

  // Check values match indices
  for (int64_t i = 0; i < k; i++) {
    if (output_values[i] != input_data[output_indices[i]]) {
      return false;
    }
  }

  // Check sorting order of the k elements
  for (int64_t i = 1; i < k; i++) {
    if (largest) {
      // For largest, should be in descending order
      if (output_values[i] > output_values[i - 1]) {
        return false;
      }
    } else {
      // For smallest, should be in ascending order
      if (output_values[i] < output_values[i - 1]) {
        return false;
      }
    }
  }

  return true;
}

// Basic functionality test
TEST_F(TopkDeviceFuncTest, BasicTopkFloat) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const int k = 3;

  std::vector<float> test_data = {
      5.0f, 2.0f, 8.0f, 1.0f, 7.0f, 3.0f, 6.0f, 4.0f};

  auto input_tensor = at::tensor(
      test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto values_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto indices_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  // Test largest
  launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      k,
      true);

  EXPECT_TRUE(validateTopkOrder<float>(
      input_tensor, values_tensor, indices_tensor, k, true));

  // Test smallest
  launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      values_tensor.data_ptr<float>(),
      indices_tensor.data_ptr<int64_t>(),
      BLOCK_SIZE,
      k,
      false);

  EXPECT_TRUE(validateTopkOrder<float>(
      input_tensor, values_tensor, indices_tensor, k, false));
}

// Variable k values test
TEST_F(TopkDeviceFuncTest, VariableKValues) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  std::vector<float> test_data = {
      7.0f, 3.0f, 9.0f, 1.0f, 5.0f, 8.0f, 2.0f, 6.0f};

  auto input_tensor = at::tensor(
      test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
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

    EXPECT_TRUE(validateTopkOrder<float>(
        input_tensor, values_tensor, indices_tensor, k, true))
        << "Failed for k=" << k;
  }
}

// Data type support test
TEST_F(TopkDeviceFuncTest, DataTypeSupport) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const int k = 3;

  // Test double
  {
    std::vector<double> test_data = {5.5, 2.1, 8.3, 1.7, 7.2, 3.9, 6.4, 4.8};

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<double, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<double>(),
        values_tensor.data_ptr<double>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder<double>(
        input_tensor, values_tensor, indices_tensor, k, true));
  }

  // Test int
  {
    std::vector<int> test_data = {5, 2, 8, 1, 7, 3, 6, 4};

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<int, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int>(),
        values_tensor.data_ptr<int>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder<int>(
        input_tensor, values_tensor, indices_tensor, k, true));
  }

  // Test int64_t
  {
    std::vector<int64_t> test_data = {5L, 2L, 8L, 1L, 7L, 3L, 6L, 4L};

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<int64_t, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int64_t>(),
        values_tensor.data_ptr<int64_t>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder<int64_t>(
        input_tensor, values_tensor, indices_tensor, k, true));
  }

  // Test bfloat16
  {
    auto input_tensor = at::randn(
        {total_elements},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<__nv_bfloat16, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<__nv_bfloat16*>(input_tensor.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(values_tensor.data_ptr()),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    EXPECT_TRUE(validateTopkOrder<float>(
        input_tensor, values_tensor, indices_tensor, k, true));
  }
}

// Edge cases test
TEST_F(TopkDeviceFuncTest, EdgeCases) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Test all same values
  {
    std::vector<float> test_data = {
        3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    const int k = 3;

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto values_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto indices_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicTopkTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        values_tensor.data_ptr<float>(),
        indices_tensor.data_ptr<int64_t>(),
        BLOCK_SIZE,
        k,
        true);

    // Validate correctness and verify all k values should be 3.0f
    EXPECT_TRUE(validateTopkOrder<float>(
        input_tensor, values_tensor, indices_tensor, k, true));

    // Additional validation: all k values should be 3.0f
    auto values_cpu = values_tensor.cpu();
    std::vector<float> output_values = getVector<float>(values_tensor);
    for (int i = 0; i < k; i++) {
      EXPECT_EQ(output_values[i], 3.0f) << "Value mismatch at position " << i;
    }
  }
}

} // namespace nvfuser
