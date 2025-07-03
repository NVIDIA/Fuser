// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace nvfuser {

using ScanTest = NVFuserTest;

// Basic functionality test for scan with Add operation (cumsum)
TEST_F(ScanTest, BasicScanAdd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Add).inclusive;
  fusion.addOutput(tv_result);

  fusion.printMath();

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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Max).inclusive;
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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Min).inclusive;
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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Mul).inclusive;
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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add).inclusive;
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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add).inclusive;
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
  auto tv2 = scan(tv1, /*dim=*/1, BinaryOpType::Add).inclusive;

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
  auto tv4 = scan(tv3, /*dim=*/1, BinaryOpType::Add).inclusive;

  // Additional operation after scan
  auto tv5 = div(tv4, IrBuilder::create<Val>(3.0));

  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Simple test case for defining a scan
TEST_F(ScanTest, Concrete1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(tv0, /*dim=*/-1, /*discount_factor=*/nullptr);

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScanTest, DiscountFactorScalar1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(
      tv0,
      /*dim=*/-1,
      /*discount_factor=*/IrBuilder::create<Val>(0.8, DataType::Float));

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScanTest, DiscountFactorTensor1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);
  auto tv1 = makeConcreteTensor({32});
  fusion->addInput(tv1);

  auto tv2 = prefixSum(
      tv0,
      /*dim=*/-1,
      /*discount_factor=*/tv1);

  fusion->addOutput(tv2);

  tv0->cacheAfter();
  tv2->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32}, options).exp();

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});

  auto cg_outputs = ke.run({t0, t1});

  testValidate(fusion.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Simple test case for defining a scan
TEST_F(ScanTest, Concrete2D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({4, 32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(tv0, /*dim=*/0, /*discount_factor=*/nullptr);

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// This is similar to what's needed for serial online softmax
TEST_F(ScanTest, OnlineSoftmax) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = makeSymbolicTensor(1);
  fusion->addInput(x);

  int64_t scan_dim = 0;

  // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
  //
  // Given x[i] for i=0 .. N-1:
  //
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   for j = 0 .. N-1
  //     m[j] = max(m[j-1], x[j])
  //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
  //
  // Final denominator is d[N-1]

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
  // normalize by running max and exponentiate
  TensorView* exp_x_m = exp(sub(x, m));
  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  TensorView* discount = exp(sub(m_prev, m));

  auto denoms = prefixSum(exp_x_m, scan_dim, discount);

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion->zeroVal(DataType::Float),
      denoms);

  auto full_max = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/neg_infty,
      m);

  auto max_bcast = broadcast(full_max, {true});
  auto norm_factor_bcast = broadcast(norm_factor, {true});
  // Recompute numerator
  auto numer = exp(sub(set(x), max_bcast));

  auto result = div(numer, norm_factor_bcast);

  fusion->addOutput(result);

  // Don't cache inputs for this fusion because we will need to recompute
  // exp((x-m)) using the final max in a separate loop so caching would mean
  // we'd need to hold the whole tensor in registers. Instead, we manually call
  // set(x) twice in the definition above to give us two separate caches.
  scheduler_utils::cacheAndForkOutputs(fusion.get(), /*unroll=*/true);

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (TensorView* tv : {m, m_prev, denoms}) {
    for (IterDomain* id : tv->getLoopDomain()) {
      uninlineable_ids.insert(id);
    }
  }

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (TensorView* tv : {m, m_prev, denoms}) {
    tv->computeWith(-1);
    for (Val* v : tv->definition()->inputs()) {
      // By using `uninlineable_ids` above, we prevent producers of scan from
      // inlining with the ScanOp past the scan dim, even though this is
      // desired. Here we do this inlining manually instead.
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  auto ref = at::softmax(t0, 0);
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
      << " returned " << cg_outputs[0].as<at::Tensor>().item()
      << " but expected " << ref.item();

  // Test automatic evaluation also
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

//
TEST_F(ScanTest, OnlineSoftmaxOuter) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // TODO: Allow outer dim to be symbolic
  auto x = makeConcreteTensor({-1, 32});
  fusion->addInput(x);

  int64_t scan_dim = 0;

  // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
  //
  // Given x[i] for i=0 .. N-1:
  //
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   for j = 0 .. N-1
  //     m[j] = max(m[j-1], x[j])
  //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
  //
  // Final denominator is d[N-1]

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
  // normalize by running max and exponentiate
  TensorView* exp_x_m = exp(sub(x, m));
  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  TensorView* discount = exp(sub(m_prev, m));

  auto denoms = prefixSum(exp_x_m, scan_dim, discount);

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion->zeroVal(DataType::Float),
      denoms);

  fusion->addOutput(norm_factor);

  scheduler_utils::cacheInputs(fusion.get(), /*unroll=*/true);
  scheduler_utils::cacheAndForkOutputs(fusion.get(), /*unroll=*/true);

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (TensorView* tv : {m, m_prev, denoms}) {
    for (IterDomain* id : tv->getLoopDomain()) {
      uninlineable_ids.insert(id);
    }
  }

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (TensorView* tv : {m, m_prev, denoms}) {
    tv->computeWith(-1);
    for (Val* v : tv->definition()->inputs()) {
      // By using `uninlineable_ids` above, we prevent producers of scan from
      // inlining with the ScanOp past the scan dim, even though this is
      // desired. Here we do this inlining manually instead.
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  at::Tensor ref = (t0 - std::get<0>(t0.max(/*dim=*/0, /*keepdim=*/true)))
                       .exp()
                       .sum(/*dim=*/0);
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
      << " returned " << cg_outputs[0].as<at::Tensor>() << " but expected "
      << ref;

  // Test automatic evaluation also
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// This is a simplified version of FlashAttention that does not circular buffer
// the inputs or use mma instructions, but has the same general computation
// pattern.
//
// Dao et al. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention
// with IO-Awareness. https://arxiv.org/abs/2205.14135
TEST_F(ScanTest, FlashAttentionNoMma) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Inputs are Q, K, V
  // Normally each of these would be 2D, shaped N-by-d
  // Output is softmax(Q@K.T, dim=1)@V
  //
  // Here, I am hard-coding a tiling split and transpose:
  //
  //   Q: N1o, d1o, N1i, d1i
  //   K: N2o, d1o, d1i, N2i
  //   V: N2o, d2o, d2i, N2i
  //
  // For the Q@K.T matmul, we have the following dim roles:
  //
  //   M: N1o, N1i
  //   N: N2o, N2i
  //   K: d1o, d1i
  //
  // For the second matmul S@V the roles are:
  //
  //   M: N1o, N1i
  //   N: d2o, d2i
  //   K: N2o, N2i
  //
  // The overall output should be of size N1o, N1i, d2o, d2i since it is the
  // result of that final matmul
  //
  // The general strategy for ordinary flash attention is to have two main
  // serial loops: over the outer "K" dims d1o and N2o. The inner dims are
  // computed via a tile matmul, equivalent to two more loops d1i and N2i.
  //
  // We grid parallelize across CTAs in the N1o and d2o dimensions and each CTA
  // computes a final output of size N1i by d2i.
  int64_t N1o = 2;
  int64_t N1i = 3;
  int64_t N2o = 4;
  int64_t N2i = 5;
  int64_t d1o = 6;
  int64_t d1i = 7;
  int64_t d2o = 8;
  int64_t d2i = 9;

  // [N1o, N2o, d1o, d2o, N1i, N2i, d1i, d2i]
  auto Q = makeConcreteTensor({N1o, 1, d1o, 1, N1i, 1, d1i, 1});
  auto K = makeConcreteTensor({1, N2o, d1o, 1, 1, N2i, d1i, 1});
  auto V = makeConcreteTensor({1, N2o, 1, d2o, 1, N2i, 1, d2i});
  fusion->addInput(Q);
  fusion->addInput(K);
  fusion->addInput(V);

  // Notation is from Algorithm 1 of Dao et al. 2022

  // TODO: mma
  auto S =
      sum(mul(Q, K),
          /*dims=*/{-2},
          /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, N2i, (1), 1]

  auto m_tilde =
      max(S, {-3}, /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, (1), 1, 1]

  auto* neg_infty = IrBuilder::create<Val>(
      -std::numeric_limits<double>::infinity(), DataType::Double);
  ScanResult max_scan_result = scan(
      m_tilde,
      2,
      BinaryOpType::Max,
      /*init=*/neg_infty,
      /*discount_factor=*/nullptr,
      /*return_exclusive=*/true);
  TensorView* m =
      max_scan_result.inclusive; // [N1o, N2o, (d1o), 1, N1i, 1, 1, 1]
  TensorView* m_prev = max_scan_result.exclusive;

  auto P_tilde = exp(sub(S, m_tilde)); // [N1o, N2o, d1o, 1, N1i, N2i, 1, 1]

  auto l_tilde =
      sum(P_tilde,
          {-3},
          /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, (1), 1, 1]

  auto first_discount = exp(sub(m_prev, m)); // [N1o, N2o, d1o, 1, N1i, 1, 1, 1]

  auto l_tilde_factor =
      exp(sub(m_tilde, m)); // [N1o, N2o, d1o, 1, N1i, 1, 1, d2i]
  auto next_l =
      mul(l_tilde_factor, l_tilde); // [N1o, N2o, d1o, 1, N1i, 1, 1, d2i]

  ScanResult sum_scan_result = scan(
      next_l,
      2,
      BinaryOpType::Add,
      /*init=*/fusion->zeroVal(DataType::Float),
      /*discount_factor=*/first_discount,
      /*return_exclusive=*/true);
  TensorView* l =
      sum_scan_result.inclusive; // [N1o, N2o, (d1o), 1, N1i, 1, 1, 1]
  TensorView* l_prev = sum_scan_result.exclusive;

  auto O_discount =
      mul(div(l_prev, l), first_discount); // [N1o, N2o, d1o, 1, N1i, 1, 1, 1]

  // P_tilde = [N1o, N2o, d1o, 1, N1i, N2i, 1, d2i]
  // V       = [1,   N2o, 1, d2o, 1,   N2i, 1, d2i]
  auto PtildeV =
      sum(mul(P_tilde, V),
          /*dims=*/{-3},
          /*keep_dim=*/true); // [N1o, N2o, d1o, d2o, N1i, (1), 1, d2i]

  auto O = prefixSum(
      mul(div(l_tilde_factor, l), PtildeV),
      2,
      /*discount_factor=*/O_discount); // [N1o, N2o, (d1o), d20, N1i, 1, 1, d2i]

  auto O_final = reductionOp(
      BinaryOpType::RHS,
      {1, 2},
      /*init=*/fusion->zeroVal(DataType::Float),
      set(O), // TODO: is this set really needed to avoid computeWith error on
              // O?
      /*keepdim=*/true); // [N1o, (1), (1), d2o, N1i, 1, 1, d2i]

  fusion->addOutput(O_final);

  fusion->printMath();

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<ScanOp>()) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        for (IterDomain* id : tv->getLoopDomain()) {
          uninlineable_ids.insert(id);
        }
      }
    }
  }

  Q->cacheAfter();
  K->cacheAfter();
  V->cacheAfter();
  O_final->cacheBefore();

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<ScanOp>()) {
      expr->output(0)->as<TensorView>()->computeWith(-1);
      for (Val* v : expr->inputs()) {
        // By using `uninlineable_ids` above, we prevent producers of scan from
        // inlining with the ScanOp past the scan dim, even though this is
        // desired. Here we do this inlining manually instead.
        v->as<TensorView>()->inlineAt(-1);
      }
    }
  }

  O_final->axis(0)->parallelize(ParallelType::BIDx);
  O_final->axis(3)->parallelize(ParallelType::BIDy);
  scheduler_utils::parallelizeAllLike(O_final);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor q = at::randn({N1o, 1, d1o, 1, N1i, 1, d1i, 1}, options);
  at::Tensor k = at::randn({1, N2o, d1o, 1, 1, N2i, d1i, 1}, options);
  at::Tensor v = at::randn({1, N2o, 1, d2o, 1, N2i, 1, d2i}, options);
  std::vector<c10::IValue> inputs{q, k, v};

  auto qorig = q.transpose(2, 4).reshape({N1o * N1i, d1o * d1i}); // 6, 42
  auto korig = k.transpose(2, 5).reshape({N2o * N2i, d1o * d1i}); // 20, 42
  auto vorig = v.transpose(3, 5).reshape({N2o * N2i, d2o * d2i}); // 20, 72
  auto qktref = at::matmul(qorig, korig.t()); // 6, 20
  auto sref = at::softmax(qktref, 1); // 6, 20
  auto ref =
      at::matmul(sref, vorig)
          .reshape({N1o, 1, 1, d2o, N1i, 1, 1, d2i}); // 2, 1, 1, 8, 3, 1, 1, 9

  KernelExecutor ke;
  ke.compile(fusion.get(), inputs);

  auto cg_outputs = ke.run(inputs);

  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>().squeeze(), ref.squeeze()));
  //<< " returned " << cg_outputs[0].as<at::Tensor>()[0] << " but expected "
  //<< ref[0];
}

} // namespace nvfuser
