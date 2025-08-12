// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <statement_guard.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

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

} // namespace nvfuser
