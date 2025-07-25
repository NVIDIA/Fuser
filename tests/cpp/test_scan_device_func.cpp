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
#include <limits>
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

TEST_F(ScanDeviceFuncTest, OnlineSoftmax) {
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   (1) First read of gmem input x
  //       one scan op to compute max(m[j-1], x[j])
  //       one scan op to compute d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j]
  //       - m[j])
  //   for j = 0 .. N-1
  //     m[j] = max(m[j-1], x[j])
  //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])

  //   (2) Second read of gmem input x
  //   for j = 0 .. N-1
  //     result[j] = exp(x[j] - m[N-1]) / d[N-1]
  //
  // Final maximum is m[N-1]
  // Final denominator is d[N-1]
  // Final softmax results are exp(x[i] - m[N-1]) / d[N-1]
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Common tensor options for all tensors
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input_tensor = at::tensor({2, 0, 2, 5, 0, 7, 2, 3}, options);

  // (1) m[j] = max(m[j-1], x[j])
  //     m = inclusive_scan(x, max)
  const ScanBinaryOpType binary_op_type = ScanBinaryOpType::Max;
  auto tensor_m_inc = at::empty({total_elements}, options);
  launchBasicScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      tensor_m_inc.data_ptr<float>(),
      0.0f, // init_value
      BLOCK_SIZE,
      binary_op_type);

  // (2) d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
  // (2.1) exclusive_scan to get m[j-1]
  constexpr float neg_infinity = -std::numeric_limits<float>::infinity();
  auto tensor_m_exc = at::empty({total_elements}, options);
  tensor_m_exc[0] = neg_infinity; // Set first element to -infinity
  tensor_m_exc.slice(/*dim=*/0, /*start=*/1, /*end=*/total_elements) =
      tensor_m_inc.slice(/*dim=*/0, /*start=*/0, /*end=*/total_elements - 1);
  // (2.2) tensor_exp_x_m = exp(x[j] - m[j])
  auto tensor_exp_x_m = at::exp(at::sub(input_tensor, tensor_m_inc));
  // (2.3) tensor_discount = exp(m[j-1] - m[j])
  auto tensor_discount = at::exp(at::sub(tensor_m_exc, tensor_m_inc));
  // (2.4) tensor_denominator = discount_scan(tensor_exp_x_m, tensor_discount)
  // d[j] = d[j-1] * tensor_discount[j] + tensor_exp_x_m[j]
  // This implements the softmax attention denominator computation
  auto tensor_denominator = at::empty({total_elements}, options);
  launchDiscountScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      tensor_exp_x_m.data_ptr<float>(),
      tensor_discount.data_ptr<float>(),
      tensor_denominator.data_ptr<float>(),
      0.0f, // init_value
      BLOCK_SIZE);
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

  // (3) Online softmax final step: result[j] = exp(x[j] - m[N-1]) / d[N-1]
  auto tensor_global_max = at::broadcast_to(
      tensor_m_inc.slice(
          /*dim=*/0, /*start=*/total_elements - 1, /*end=*/total_elements),
      {total_elements});
  auto tensor_global_denominator = at::broadcast_to(
      tensor_denominator.slice(
          /*dim=*/0, /*start=*/total_elements - 1, /*end=*/total_elements),
      {total_elements});
  auto tensor_result_online = at::div(
      at::exp(at::sub(input_tensor, tensor_global_max)),
      tensor_global_denominator);

  // (4) validate
  auto tensor_result_pytorch = at::softmax(input_tensor, /*dim=*/0);
  EXPECT_TRUE(at::allclose(
      tensor_result_online,
      tensor_result_pytorch,
      /*rtol=*/1e-5,
      /*atol=*/1e-6));
}

// Test discount scan functionality with simple, predictable inputs
TEST_F(ScanDeviceFuncTest, DiscountScan) {
  const int BLOCK_SIZE = 4;
  const int ITEMS_PER_THREAD = 2;
  const int total_elements = BLOCK_SIZE * ITEMS_PER_THREAD;

  // Common tensor options for all tensors
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input_tensor =
      at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, options);
  auto discount_tensor =
      at::tensor({0.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, options);
  auto output_tensor = at::empty({total_elements}, options);

  // Launch discount scan kernel
  launchDiscountScanTestKernel<float, ITEMS_PER_THREAD>(
      at::cuda::getCurrentCUDAStream(),
      input_tensor.data_ptr<float>(),
      discount_tensor.data_ptr<float>(),
      output_tensor.data_ptr<float>(),
      0.0f, // init_value
      BLOCK_SIZE);
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());

  // Manual validation: d[j] = d[j-1] * discount[j] + input[j]
  std::vector<float> expected = {
      1.0f, // d[0] = input[0] = 1.0
      1.0f * 0.5f + 2.0f, // d[1] = 1.0 * 0.5 + 2.0 = 2.5
      2.5f * 0.5f + 3.0f, // d[2] = 2.5 * 0.5 + 3.0 = 4.25
      4.25f * 0.5f + 4.0f, // d[3] = 4.25 * 0.5 + 4.0 = 6.125
      6.125f * 0.5f + 5.0f, // d[4] = 6.125 * 0.5 + 5.0 = 8.0625
      8.0625f * 0.5f + 6.0f, // d[5] = 8.0625 * 0.5 + 6.0 = 10.03125
      10.03125f * 0.5f + 7.0f, // d[6] = 10.03125 * 0.5 + 7.0 = 12.015625
      12.015625f * 0.5f + 8.0f // d[7] = 12.015625 * 0.5 + 8.0 = 14.0078125
  };
  auto expected_tensor = at::tensor(expected, options);
  // Validate results with tolerance for floating point precision
  EXPECT_TRUE(at::allclose(
      output_tensor, expected_tensor, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "Discount scan results don't match expected values";
}

} // namespace nvfuser
