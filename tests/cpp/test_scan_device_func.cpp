// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <tests/cpp/scan_test_helper.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

namespace nvfuser {

using ScanDeviceFuncTest = NVFuserTest;

// Basic functionality test for Add operation (cumsum)
TEST_F(ScanDeviceFuncTest, BasicScanAdd) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Add;

  auto input_tensor = at::tensor(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      0.0f, // init_value
      BLOCK_SIZE,
      binary_op_type);

  validateScanResult(input_tensor, output_tensor, binary_op_type);
}

// Basic functionality test for Max operation (cummax)
TEST_F(ScanDeviceFuncTest, BasicScanMax) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Max;

  auto input_tensor = at::tensor(
      {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 4.0f, 7.0f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      std::numeric_limits<float>::lowest(), // init_value
      BLOCK_SIZE,
      binary_op_type);

  validateScanResult(input_tensor, output_tensor, binary_op_type);
}

// Basic functionality test for Min operation (cummin)
TEST_F(ScanDeviceFuncTest, BasicScanMin) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Min;

  auto input_tensor = at::tensor(
      {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f, 4.0f, 7.0f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      std::numeric_limits<float>::max(), // init_value
      BLOCK_SIZE,
      binary_op_type);

  validateScanResult(input_tensor, output_tensor, binary_op_type);
}

// Basic functionality test for Mul operation (cumprod)
TEST_F(ScanDeviceFuncTest, BasicScanMul) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Mul;

  auto input_tensor = at::tensor(
      {1.0f, 2.0f, 0.5f, 3.0f, 0.25f, 4.0f, 0.1f, 2.0f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  auto output_tensor = at::empty(
      {total_elements},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

  launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      1.0f, // init_value
      BLOCK_SIZE,
      binary_op_type);

  validateScanResult(input_tensor, output_tensor, binary_op_type);
}

// Variable block sizes test
TEST_F(ScanDeviceFuncTest, VariableBlockSizes) {
  const int ITEMS_PER_THREAD = 2;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Add;

  // Test different block sizes: 4, 8, 16, 32
  for (int block_size : {4, 8, 16, 32}) {
    const int total_elements = block_size * ITEMS_PER_THREAD;

    // Create sequential input data for easy validation
    std::vector<float> input_data(total_elements);
    std::iota(input_data.begin(), input_data.end(), 1.0f);

    auto input_tensor = at::tensor(
        input_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

    launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        0.0f, // init_value
        block_size,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }
}

// Data type support test
TEST_F(ScanDeviceFuncTest, DataTypeSupport) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Add;

  // Test double
  {
    auto input_tensor = at::arange(
        1,
        total_elements + 1,
        at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));

    launchBasicScanTestKernel<double, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<double>(),
        output_tensor.data_ptr<double>(),
        0.0, // init_value
        BLOCK_SIZE,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }

  // Test int
  {
    auto input_tensor = at::arange(
        1,
        total_elements + 1,
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

    launchBasicScanTestKernel<int, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int>(),
        output_tensor.data_ptr<int>(),
        0, // init_value
        BLOCK_SIZE,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }

  // Test int64_t
  {
    auto input_tensor = at::arange(
        1,
        total_elements + 1,
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

    launchBasicScanTestKernel<int64_t, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<int64_t>(),
        output_tensor.data_ptr<int64_t>(),
        0L, // init_value
        BLOCK_SIZE,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }
}

// Edge cases test
TEST_F(ScanDeviceFuncTest, EdgeCases) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Add;

  // Test all same values
  {
    std::vector<float> test_data(total_elements, 3.0f);

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

    launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        0.0f, // init_value
        BLOCK_SIZE,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }

  // Test zeros
  {
    std::vector<float> test_data(total_elements, 0.0f);

    auto input_tensor = at::tensor(
        test_data, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
    auto output_tensor = at::empty(
        {total_elements},
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));

    launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
        at::cuda::getCurrentCUDAStream(),
        input_tensor.data_ptr<float>(),
        output_tensor.data_ptr<float>(),
        0.0f, // init_value
        BLOCK_SIZE,
        binary_op_type);

    validateScanResult(input_tensor, output_tensor, binary_op_type);
  }
}

} // namespace nvfuser
