// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <options.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <limits>

namespace nvfuser {

using ScanTest = NVFuserTest;

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
  TensorView* m;
  TensorView* m_prev;
  std::tie(m, m_prev) = scanWithExclusive(
      set(x),
      scan_dim,
      BinaryOpType::Max,
      /*init=*/neg_infty); // max x[j] over j = 0 .. i
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

  TensorView* m;
  TensorView* m_prev;
  std::tie(m, m_prev) = scanWithExclusive(
      x,
      scan_dim,
      BinaryOpType::Max,
      /*init=*/
      IrBuilder::create<Val>(
          -std::numeric_limits<double>::infinity(),
          DataType::Double)); // max x[j] over j = 0 .. i
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

} // namespace nvfuser
