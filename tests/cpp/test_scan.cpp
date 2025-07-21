// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ScanTest = NVFuserTest;

// Basic functionality test for scan with Add operation (cumsum)
TEST_F(ScanTest, BasicScanAdd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/1,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test scan with discount factor support - manual validation
TEST_F(ScanTest, ScanAddDiscountFactor) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Create discount factor - a scalar value
  float discount = 0.5f;
  auto discount_factor = IrBuilder::create<Val>(discount, DataType::Float);

  // Call scan with discount factor and return all 3 results: inclusive,
  // exclusive, and reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/0,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/discount_factor,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  // Use fixed input values for manual verification of the validation
  at::Tensor input_tensor = at::tensor(
      {1.0f, 2.0f, 3.0f, 4.0f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input_tensor});
  auto expected_inclusive = at::tensor(
      {1.0f, 2.5f, 4.25f, 6.125f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto expected_exclusive = at::tensor(
      {0.0f, 1.0f, 2.5f, 4.25f},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto expected_reduction = at::tensor(
      {6.125f}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  ASSERT_TRUE(at::allclose(outputs[0].as<at::Tensor>(), expected_inclusive));
  ASSERT_TRUE(at::allclose(outputs[1].as<at::Tensor>(), expected_exclusive));
  ASSERT_TRUE(at::allclose(outputs[2].as<at::Tensor>(), expected_reduction));
  testValidate(
      executor_cache.fusion(), outputs, {input_tensor}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Max operation (cummax)
TEST_F(ScanTest, BasicScanMax) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/1,
      BinaryOpType::Max,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Min operation (cummin)
TEST_F(ScanTest, BasicScanMin) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/1,
      BinaryOpType::Min,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Mul operation (cumprod)
TEST_F(ScanTest, BasicScanMul) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/1,
      BinaryOpType::Mul,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test different tensor shapes and scan dimensions
TEST_F(ScanTest, ScanDifferentDimensions) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({2, 4, 6});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/0,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 4, 6}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test 1D tensor scan
TEST_F(ScanTest, Scan1D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv0,
      /*dim=*/0,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({10}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// NOTE: Complex arithmetic + scan fusion is limited by ExprEval scheduler's
// single expression constraint. For complex fusions with ScanOp, nvFuser would
// need fusion segmentation or different scheduler approaches.

// Test simple ScanOp with just one additional operation
TEST_F(ScanTest, ScanWithSimpleArithmetic) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Single arithmetic operation before scan
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv1,
      /*dim=*/1,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);
  fusion.addOutput(scan_result.inclusive);
  fusion.addOutput(scan_result.exclusive);
  fusion.addOutput(scan_result.reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test ScanOp with multiple arithmetic operations - investigating complex
// fusion behavior
TEST_F(ScanTest, ScanWithArithmeticOps) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Multiple arithmetic operations
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = sub(tv2, IrBuilder::create<Val>(0.5));

  // Call scan once and return all 3 results: inclusive, exclusive, and
  // reduction
  auto scan_result = scan(
      tv3,
      /*dim=*/1,
      BinaryOpType::Add,
      /*init=*/nullptr,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true,
      /*return_reduction=*/true);

  // Additional operations after scan
  auto tv5_inclusive = div(scan_result.inclusive, IrBuilder::create<Val>(3.0));
  auto tv5_exclusive = div(scan_result.exclusive, IrBuilder::create<Val>(3.0));
  auto tv5_reduction = div(scan_result.reduction, IrBuilder::create<Val>(3.0));

  fusion.addOutput(tv5_inclusive);
  fusion.addOutput(tv5_exclusive);
  fusion.addOutput(tv5_reduction);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

} // namespace nvfuser
