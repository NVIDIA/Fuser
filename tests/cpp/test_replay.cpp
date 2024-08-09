// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>

namespace nvfuser {

using ReplayTest = NVFuserTest;

TEST_F(ReplayTest, HorizontallyMergeReshapeAndPermute) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 5});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* r0 = reshape(s0, {4, 2}, {2, 2, 2});
  TensorView* p0 = permute(r0, {1, 0, 2});

  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  TensorView* r1 = reshape(s1, {4, 3}, {2, 2, 3});
  TensorView* p1 = permute(r1, {1, 0, 2});

  TensorView* out = cat({p0, p1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  Expr* merged = replayExprWithNewInput(r0->definition(), in);
  merged = replayExprWithNewInput(p0->definition(), merged->output(0));
  // To preserve the output allocation domain, we create a Set between
  // `merged`'s output and `out` instead of replacing the fusion output.
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, merged->output(0));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 5}, options);

  FusionExecutorCache fec(std::move(fusion));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0];

  std::vector<at::Tensor> slices = at::split(in_tensor, {2, 3}, /*dim=*/-1);
  at::Tensor expected_out_tensor = at::cat(
      {slices[0].view({2, 2, 2}).permute({1, 0, 2}),
       slices[1].view({2, 2, 3}).permute({1, 0, 2})},
      /*dim=*/-1);

  EXPECT_TRUE(at::equal(out_tensor, expected_out_tensor));
}

TEST_F(ReplayTest, HorizontallyMergeReshapeAndNeg) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 5});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* r0 = reshape(s0, {4, 2}, {2, 2, 2});
  TensorView* n0 = neg(r0);

  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  TensorView* r1 = reshape(s1, {4, 3}, {2, 2, 3});
  TensorView* n1 = neg(r1);

  TensorView* out = cat({n0, n1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  Expr* merged = replayExprWithNewInput(r0->definition(), in);
  merged = replayExprWithNewInput(n0->definition(), merged->output(0));
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out, merged->output(0));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 5}, options);

  FusionExecutorCache fec(std::move(fusion));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0];

  std::vector<at::Tensor> slices = at::split(in_tensor, {2, 3}, /*dim=*/-1);
  at::Tensor expected_out_tensor = at::cat(
      {-slices[0].view({2, 2, 2}), -slices[1].view({2, 2, 3})},
      /*dim=*/-1);

  EXPECT_TRUE(at::equal(out_tensor, expected_out_tensor));
}

} // namespace nvfuser
