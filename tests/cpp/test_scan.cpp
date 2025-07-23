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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Add);
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Max operation (cummax)
TEST_F(ScanTest, BasicScanMax) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Max);
  fusion.addOutput(tv_result);

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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Min);
  fusion.addOutput(tv_result);

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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Mul);
  fusion.addOutput(tv_result);

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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion.addOutput(tv_result);

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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion.addOutput(tv_result);

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

  // Scan operation
  auto tv2 = scan(tv1, /*dim=*/1, BinaryOpType::Add);

  fusion.addOutput(tv2);

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

  // Scan operation
  auto tv4 = scan(tv3, /*dim=*/1, BinaryOpType::Add);

  // Additional operation after scan
  auto tv5 = div(tv4, IrBuilder::create<Val>(3.0));

  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

TEST_F(ScanTest, OnlineSoftmax) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto x = makeSymbolicTensor(1);
  fusion.addInput(x);

  int64_t scan_dim = 0;

  // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
  //
  // Given x[i] for i=0 .. N-1:
  //
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   (1) First read of gmem input x
  //       one scan op to compute max(m[j-1], x[j])
  //       one scan op to compute d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
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
  auto* neg_infty = IrBuilder::create<Val>(
      -std::numeric_limits<double>::infinity(), DataType::Double);
  ScanResult max_scan_result = scan(
      set(x),
      scan_dim,
      BinaryOpType::Max,
      /*init=*/neg_infty,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true); // max x[j] over j = 0 .. i
  TensorView* m = max_scan_result.inclusive;
  TensorView* m_prev = max_scan_result.exclusive;
  TensorView* full_max = max_scan_result.reduction;
  // normalize by running max and exponentiate
  TensorView* exp_x_m = exp(sub(x, m));
  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  TensorView* discount = exp(sub(m_prev, m));

  auto denoms = prefixSum(exp_x_m, scan_dim, discount);

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion.zeroVal(DataType::Float),
      denoms);

  // auto full_max = reductionOp(
  //     BinaryOpType::RHS,
  //     {scan_dim},
  //     /*init=*/neg_infty,
  //     m);

  auto max_bcast = broadcast(full_max, {true});
  auto norm_factor_bcast = broadcast(norm_factor, {true});
  // Recompute numerator
  auto numer = exp(sub(set(x), max_bcast));

  auto result = div(numer, norm_factor_bcast);

  fusion.addOutput(result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}
} // namespace nvfuser
