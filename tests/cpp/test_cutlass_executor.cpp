// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// Test scaled_mm using FusionExecutorCache
TEST_F(NVFuserTest, CutlassExecutor_ScaledMmWithFusionExecutorCache) {
  // Skip if not on SM100 or above (required for scaled_mm)
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Define problem size
  const int64_t M = 128;
  const int64_t N = 128;
  const int64_t K = 128;

  // Create input tensors
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 = makeContigTensor(2, DataType::Float); // Scale A (M x 1)
  auto tv3 = makeContigTensor(2, DataType::Float); // Scale B (1 x N)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation
  auto scaled_out = scaled_mm(tv0, tv1_t, tv2, tv3);

  fusion->addOutput(scaled_out.tv);

  // Create test data
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto a = at::randn({M, K}, options);
  auto b = at::randn({N, K}, options);
  auto scale_a = at::randn({M, 1}, float_options);
  auto scale_b = at::randn({1, N}, float_options);

  std::vector<at::Tensor> inputs = {a, b, scale_a, scale_b};

  // Execute the fusion using FusionExecutorCache
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  // Verify output shape
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[0], M);
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[1], N);
}

// Test scaled_mm with different matrix sizes using FusionExecutorCache
TEST_F(
    NVFuserTest,
    CutlassExecutor_ScaledMmDifferentSizesWithFusionExecutorCache) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Define problem size
  const int64_t M = 256;
  const int64_t N = 512;
  const int64_t K = 128;

  // Create input tensors
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 = makeContigTensor(2, DataType::Float); // Scale A (M x 1)
  auto tv3 = makeContigTensor(2, DataType::Float); // Scale B (1 x N)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation
  auto scaled_out = scaled_mm(tv0, tv1_t, tv2, tv3);

  fusion->addOutput(scaled_out.tv);

  // Create test data
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto a = at::randn({M, K}, options);
  auto b = at::randn({N, K}, options);
  auto scale_a = at::randn({M, 1}, float_options);
  auto scale_b = at::randn({1, N}, float_options);

  std::vector<at::Tensor> inputs = {a, b, scale_a, scale_b};

  // Execute the fusion using FusionExecutorCache
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  // Verify output shape
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[0], M);
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[1], N);
}

// Test scaled_mm with alpha parameter using FusionExecutorCache
TEST_F(NVFuserTest, CutlassExecutor_ScaledMmWithAlphaUsingFusionExecutorCache) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Define problem size
  const int64_t M = 128;
  const int64_t N = 128;
  const int64_t K = 128;

  // Create input tensors
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 = makeContigTensor(2, DataType::Float); // Scale A (M x 1)
  auto tv3 = makeContigTensor(2, DataType::Float); // Scale B (1 x N)
  auto tv4 = makeContigTensor(0, DataType::Float); // Alpha (scalar)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation with alpha
  auto scaled_out = scaled_mm(tv0, tv1_t, tv2, tv3, tv4);

  fusion->addOutput(scaled_out.tv);

  // Create test data
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto a = at::randn({M, K}, options);
  auto b = at::randn({N, K}, options);
  auto scale_a = at::randn({M, 1}, float_options);
  auto scale_b = at::randn({1, N}, float_options);
  auto alpha = at::tensor(2.0, float_options);

  std::vector<at::Tensor> inputs = {a, b, scale_a, scale_b, alpha};

  // Execute the fusion using FusionExecutorCache
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  // Verify output shape
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[0], M);
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[1], N);
}

// Test scaled_mm with bias using FusionExecutorCache
TEST_F(NVFuserTest, CutlassExecutor_ScaledMmWithBiasUsingFusionExecutorCache) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Define problem size
  const int64_t M = 128;
  const int64_t N = 128;
  const int64_t K = 128;

  // Create input tensors
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 = makeContigTensor(2, DataType::Float); // Scale A (M x 1)
  auto tv3 = makeContigTensor(2, DataType::Float); // Scale B (1 x N)
  auto tv4 = makeContigTensor(1, DataType::BFloat16); // Bias (N)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation with bias
  auto scaled_out = scaled_mm(tv0, tv1_t, tv2, tv3, nullptr, tv4);

  fusion->addOutput(scaled_out.tv);

  // Create test data
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto a = at::randn({M, K}, options);
  auto b = at::randn({N, K}, options);
  auto scale_a = at::randn({M, 1}, float_options);
  auto scale_b = at::randn({1, N}, float_options);
  auto bias = at::randn({N}, options);

  std::vector<at::Tensor> inputs = {a, b, scale_a, scale_b, bias};

  // Execute the fusion using FusionExecutorCache
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  // Verify output shape
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[0], M);
  EXPECT_EQ(outputs[0].as<at::Tensor>().sizes()[1], N);
}

} // namespace nvfuser
