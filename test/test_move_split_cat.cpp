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
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

using MoveSplitCatTest = NVFuserTest;

TEST_F(MoveSplitCatTest, Cancellable_Adjacent) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 10});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  TensorView* s2 = slice(in, {0, 5}, {4, 10});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_SliceAmountAndPaddingAmountMismatch) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 10});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  // The window is shifted from [2, 5) to [3, 6), leading to an amount mismatch
  // between this slice and the corresponding pad.
  TensorView* s1 = slice(in, {0, 3}, {4, 6});
  TensorView* s2 = slice(in, {0, 5}, {4, 10});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_CatOnlySubsetOfSplitOutputs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 10});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  // s1 has expands <-2, -5>, and the second pad has padding <2, 0>. This
  // mismatch prevents a false cancellation.
  TensorView* out = cat({s0, s1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_PermuteInBetween) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 10});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 3, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 3, 5});
  TensorView* s2 = slice(in, {0, 0, 5}, {2, 3, 10});
  s0 = permute(s0, {1, 0, 2});
  s1 = permute(s1, {1, 0, 2});
  s2 = permute(s2, {1, 0, 2});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_SomeButNotAllArePermuted) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 10});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 2, 5});
  TensorView* s2 = slice(in, {0, 0, 5}, {2, 2, 10});
  s0 = permute(s0, {1, 0, 2});
  s2 = permute(s2, {1, 0, 2});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 2, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_PermutedDifferently) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 2, 10});
  TensorView* s0 = slice(in, {0, 0, 0, 0}, {2, 2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 0, 2}, {2, 2, 2, 5});
  TensorView* s2 = slice(in, {0, 0, 0, 5}, {2, 2, 2, 10});
  s0 = permute(s0, {2, 1, 0, 3});
  s1 = permute(s1, {1, 0, 2, 3});
  s2 = permute(s2, {2, 1, 0, 3});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 2, 2, 10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_UnsupportedOps) {
  // This test is to verify the optimization correctly bails out on unsupported
  // ops. We could but don't merge a split and a cat when a broadcast is in
  // between.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 4});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 2, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 2, 4});
  s0 = broadcast(s0, {false, true, false, false});
  s1 = broadcast(s1, {false, true, false, false});
  TensorView* out = cat({s0, s1}, /*dim=*/2);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 2, 4}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].is_alias_of(in_tensor));
}

} // namespace nvfuser
