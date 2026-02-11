// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include "fusion.h"
#include "ops/alias.h"
#include "ops/arith.h"
#include "statement_guard.h"
#include "tests/cpp/utils.h"
#include "validator_utils.h"

namespace nvfuser {

using StatementGuardTest = NVFuserTest;

TEST_F(StatementGuardTest, ExecuteAfterGuard) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  fusion->addInput(in);
  TensorView* out = flatten(in);
  fusion->addOutput(out);

  int64_t num_exprs_before_guard = fusion->numExprs();
  int64_t num_exprs_in_guard = -1;
  {
    FusionGuard fg(fusion.get());
    StatementGuard sg(fusion.get());
    add(in, in);
    num_exprs_in_guard = fusion->numExprs();
  }
  int64_t num_exprs_after_guard = fusion->numExprs();

  EXPECT_LT(num_exprs_before_guard, num_exprs_in_guard);
  EXPECT_GT(num_exprs_in_guard, num_exprs_after_guard);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}, at::device(at::kCUDA));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

// Regression test: special vals lazily created inside a StatementGuard scope
// must not become dangling pointers after the guard rolls back.
TEST_F(StatementGuardTest, LazySpecialValsNotDangling) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(1);
  fusion->addInput(in);
  TensorView* out = set(in);
  fusion->addOutput(out);

  // Force lazy creation of trueVal/falseVal inside a StatementGuard scope.
  // This reproduces the bug where haveDifferentShardings calls simplifyExpr
  // inside a StatementGuard, which can lazily create special vals that then
  // become dangling pointers when the guard rolls back.
  {
    StatementGuard sg(fusion.get());
    // Directly trigger lazy creation of trueVal and falseVal
    fusion->trueVal();
    fusion->falseVal();
    fusion->oneVal();
  }

  // After the guard, the special vals should still be valid (re-created if the
  // originals were destroyed by the guard's rollback).
  Val* z = fusion->zeroVal();
  Val* o = fusion->oneVal();
  Val* t = fusion->trueVal();
  Val* f = fusion->falseVal();
  EXPECT_NE(z, nullptr);
  EXPECT_NE(o, nullptr);
  EXPECT_NE(t, nullptr);
  EXPECT_NE(f, nullptr);
  EXPECT_TRUE(z->isZeroInt());
  EXPECT_TRUE(o->isOneInt());
  EXPECT_TRUE(t->isTrue());
  EXPECT_TRUE(f->isFalse());

  // The fusion should still be executable
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({8}, at::device(at::kCUDA));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  testValidate(
      executor_cache.fusion(),
      {out_tensors[0].as<at::Tensor>()},
      {in_tensor},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
