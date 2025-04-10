// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/arith.h>
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

  fusion->printMath();

  tv0->cacheAfter();
  tv1->cacheBefore();
  // Caching works fine, but once we inline we wind up not allocating the scan
  // ID, meaning the index is just 0, and there's no replacement. This actually
  // gives us the correct result in this test but it's not pretty, so I'd like
  // to handle such cases more gracefully.
  // TODO: Handle cases when the scan ID is inlined away.
  // inlineMost();

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
  // Caching works fine, but once we inline we wind up not allocating the scan
  // ID, meaning the index is just 0, and there's no replacement. This actually
  // gives us the correct result in this test but it's not pretty, so I'd like
  // to handle such cases more gracefully.
  // TODO: Handle cases when the scan ID is inlined away.
  // inlineMost();

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
  // Caching works fine, but once we inline we wind up not allocating the scan
  // ID, meaning the index is just 0, and there's no replacement. This actually
  // gives us the correct result in this test but it's not pretty, so I'd like
  // to handle such cases more gracefully.
  // TODO: Handle cases when the scan ID is inlined away.
  // inlineMost();

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

  // tv0->cacheAfter();
  // tv1->cacheBefore();
  //  Caching works fine, but once we inline we wind up not allocating the scan
  //  ID, meaning the index is just 0, and there's no replacement. This actually
  //  gives us the correct result in this test but it's not pretty, so I'd like
  //  to handle such cases more gracefully.
  //  TODO: Handle cases when the scan ID is inlined away.
  // inlineMost();

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

  auto x = makeConcreteTensor({32});
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
  // Discount factor is exponentiated delta: exp(m[i] - m[i-1])
  TensorView* discount = exp(sub(m, m_prev));

  // What is needed?
  //  - generalize prefixSum to scan (with discount factor) to cover prefixMax
  //    case
  //  - Implement lag1, with similar constraints to scan, for computing the
  //    delta to adjust discount
  //  - How to extract final element of the prefix sum?
  //     - If _all_ we want is the final value, then we don't need to allocate
  //       all of the values for tv1
  //     - We could have a custom reduction type that we apply to grab the last
  //       value in a dimension. This could be accomplished with a new
  //       BinaryOpType::RHS which just returns the rhs

  //
  // lag1(x)[i] = x[i-1]  (in a specified dim)
  // This is related to scan: scan can be defined recursively using lag1:
  //    y := scan(x, f)
  //    y = f(lag1(y), x)
  // We don't represent it like this in our IR because that would require a
  // cyclic graph (see Fold proposal)

  // Note that lag1(scan(x)) is an _exclusive_ scan of x, e.g. sum x[j] for j =
  // 0 .. i-1 One option is for us to produce two outputs from scan: the
  // exclusive and inclusive scans:
  //   mexc, minc = scan(x, scan_dim, BinaryOpType::Max);
  //   exp_x_m = exp(sub(x, minc));
  //   discount = exp(sub(minc, mexc));
  // Note that we don't need to allocate mexc unless it is used, but this might
  // be difficult in our current system because it is a sibling.
  //
  // We could also have a separate node or option where mexc is disabled.

  auto denoms = prefixSum(exp_x_m, scan_dim, discount);

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion->zeroVal(DataType::Float),
      denoms);

  fusion->addOutput(norm_factor);

  // Caching works fine, but once we inline we wind up not allocating the scan
  // ID, meaning the index is just 0, and there's no replacement. This actually
  // gives us the correct result in this test but it's not pretty, so I'd like
  // to handle such cases more gracefully.
  // TODO: Handle cases when the scan ID is inlined away.
  // inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  auto ref = (t0 - t0.max()).exp().sum();
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
      << " returned " << cg_outputs[0].as<at::Tensor>().item()
      << " but expected " << ref.item();

  // Test automatic evaluation also
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
