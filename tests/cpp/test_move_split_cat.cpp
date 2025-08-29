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
#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Contains;

using MoveSplitCatTest = NVFuserTest;

TEST_F(MoveSplitCatTest, Cancellable_SplitImmediatelyFollowedByCat) {
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_DifferentOrder) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 6});
  std::vector<TensorView*> slices = chunk(in, /*chunks=*/2, /*dim=*/-1);
  TensorView* out = cat({slices[1], slices[0]}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 6}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_SetWithoutPermute) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 5});
  std::vector<TensorView*> slices = chunk(in, /*chunks=*/2, /*dim=*/-1);
  TensorView* s0 = set(slices[0]);
  TensorView* s1 = set(slices[1]);
  TensorView* out = cat({s0, s1}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

namespace {
MATCHER(IsPermute, "") {
  if (auto* set = dynamic_cast<LoadStoreOp*>(arg)) {
    if (auto* set_out = dynamic_cast<TensorView*>(set->out())) {
      return set_out->hasRoot();
    }
  }
  return false;
}
} // namespace

TEST_F(MoveSplitCatTest, Cancellable_IncompatibleAllocationOrder) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  std::vector<TensorView*> slices = chunk(in, /*chunks=*/2, /*dim=*/-1);
  TensorView* s0 = permute(slices[0], {1, 0, 2});
  TensorView* s1 = permute(slices[1], {1, 0, 2});
  TensorView* out = cat({s0, s1}, /*dim=*/-1);
  out->setAllocationDomain({out->axis(2), out->axis(0), out->axis(1)}, true);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3, 5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  // Check the two permutes are merged to one.
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  Fusion* complete_fusion = runtime->fusionSegments()->completeFusion();
  EXPECT_THAT(complete_fusion->exprs(), Contains(IsPermute()).Times(1));

  // Due to the incompatible output allocation order, the output can't be an
  // alias.
  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_MultiplePermutesInBetween) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 10});
  TensorView* s0 = slice(in, {0, 0, 0}, {2, 3, 2});
  TensorView* s1 = slice(in, {0, 0, 2}, {2, 3, 5});
  TensorView* s2 = slice(in, {0, 0, 5}, {2, 3, 10});
  // dim=2 is the split dimension.
  s0 = permute(s0, {1, 2, 0});
  s1 = permute(s1, {1, 2, 0});
  s2 = permute(s2, {1, 2, 0});
  // dim=1 is the split dimension.
  s0 = permute(s0, {1, 2, 0});
  s1 = permute(s1, {1, 2, 0});
  s2 = permute(s2, {1, 2, 0});
  // dim=0 is the split dimension.
  TensorView* out = cat({s0, s1, s2}, /*dim=*/0);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3, 10}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_WrongAxis) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 2, 4});
  std::vector<TensorView*> slices = chunk(in, /*num_slices=*/2, /*dim=*/-1);
  // dim=2 is the split dimension.
  TensorView* s0 = permute(slices[0], {1, 2, 0});
  TensorView* s1 = permute(slices[1], {1, 2, 0});
  // After permutation, dim=1 is the split dimension. However, the following
  // `cat` is along dim=0.
  TensorView* out = cat({s0, s1}, /*dim=*/0);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 2, 4}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Noncancellable_PermutedDifferently) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 2});
  TensorView* s0 = slice(in, {0, 0}, {2, 2});
  s0 = set(s0);
  s0 = reshape(s0, {2, 2}, {4});

  TensorView* s1 = slice(in, {2, 0}, {4, 2});
  s1 = permute(s1, {1, 0});
  s1 = reshape(s1, {2, 2}, {4});

  TensorView* out = cat({s0, s1}, /*dim=*/0);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 2}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_ReshapeInBetween) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 10});
  TensorView* s0 = slice(in, {0, 0}, {4, 2});
  TensorView* s1 = slice(in, {0, 2}, {4, 5});
  TensorView* s2 = slice(in, {0, 5}, {4, 10});
  s0 = reshape(s0, {4, 2}, {2, 2, 2});
  s1 = reshape(s1, {4, 3}, {2, 2, 3});
  s2 = reshape(s2, {4, 5}, {2, 2, 5});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 10}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_ReshapeAndPermuteInBetween) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({6, 10});
  TensorView* s0 = slice(in, {0, 0}, {6, 2});
  TensorView* s1 = slice(in, {0, 2}, {6, 5});
  TensorView* s2 = slice(in, {0, 5}, {6, 10});
  s0 = reshape(s0, {6, 2}, {2, 3, 2});
  s1 = reshape(s1, {6, 3}, {2, 3, 3});
  s2 = reshape(s2, {6, 5}, {2, 3, 5});
  s0 = permute(s0, {1, 0, 2});
  s1 = permute(s1, {1, 0, 2});
  s2 = permute(s2, {1, 0, 2});
  TensorView* out = cat({s0, s1, s2}, /*dim=*/-1);

  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({6, 10}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, Cancellable_Issue1768) {
  constexpr int b = 16; // batch size
  constexpr int h = 12; // number of heads
  constexpr int s = 128; // sequence length
  constexpr int f = 64; // feature size per head

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* sdpa_backward_out =
      makeContigConcreteTensor({b, h * 3, s, f}, DataType::Half);
  sdpa_backward_out->setAllocationDomain(
      {sdpa_backward_out->axis(0),
       sdpa_backward_out->axis(2),
       sdpa_backward_out->axis(1),
       sdpa_backward_out->axis(3)},
      true);
  TensorView* dq = slice(sdpa_backward_out, {0, 0, 0, 0}, {b, h, s, f});
  TensorView* dk = slice(sdpa_backward_out, {0, h, 0, 0}, {b, h * 2, s, f});
  TensorView* dv = slice(sdpa_backward_out, {0, h * 2, 0, 0}, {b, h * 3, s, f});
  // Swap the head dimension and the sequence length dimension.
  dq = permute(dq, {0, 2, 1, 3});
  dk = permute(dk, {0, 2, 1, 3});
  dv = permute(dv, {0, 2, 1, 3});
  dq = reshape(dq, {b, s, h, f}, {b, s, h * f});
  dk = reshape(dk, {b, s, h, f}, {b, s, h * f});
  dv = reshape(dv, {b, s, h, f}, {b, s, h * f});
  TensorView* cat_out = cat({dq, dk, dv}, /*dim=*/-1);
  TensorView* sum_out = castOp(DataType::Float, cat_out);
  sum_out = sum(sum_out, {0, 1});
  sum_out = castOp(DataType::Half, sum_out);
  TensorView* view_out =
      reshape(cat_out, {b, s, h * f * 3}, {b * s, h * f * 3});
  TensorView* permute_out = permute(view_out, {1, 0});

  fusion->addInput(sdpa_backward_out);
  fusion->addOutput(sum_out);
  fusion->addOutput(view_out);
  fusion->addOutput(permute_out);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor in_tensor =
      at::randn({b * h * 3 * s * f}, options)
          .as_strided({b, h * 3, s, f}, {h * 3 * s * f, f, h * 3 * f, 1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(in_tensor));
  EXPECT_TRUE(out_tensors[2].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, OuterSplit) {
  // A simplified reproducer for #2142.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* merged = makeContigConcreteTensor({4, 6});
  fusion->addInput(merged);

  TensorView* s0 = slice(merged, {0, 0}, {2, 6});
  TensorView* s1 = slice(merged, {2, 0}, {4, 6});
  s0 = reshape(s0, {2, 6}, {4, 3});
  s1 = reshape(s1, {2, 6}, {4, 3});
  merged = cat({s0, s1}, /*dim=*/0);
  fusion->addOutput(merged);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 6}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(MoveSplitCatTest, MultiplePairs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* merged = makeContigConcreteTensor({4, 6});
  fusion->addInput(merged);

  // Region 0. Mergeable because both slices are permuted in the same way and
  // the cat axis matches the split axis.
  TensorView* s0 = slice(merged, {0, 0}, {2, 6});
  TensorView* s1 = slice(merged, {2, 0}, {4, 6});
  s0 = permute(s0, {1, 0});
  s1 = permute(s1, {1, 0});
  merged = cat({s0, s1}, /*dim=*/1);

  // Region 1. Not mergeable because the outer dimension is split and the inner
  // dimension is catted.
  s0 = slice(merged, {0, 0}, {3, 4});
  s1 = slice(merged, {3, 0}, {6, 4});
  s0 = reshape(s0, {3, 4}, {6, 2});
  s1 = reshape(s1, {3, 4}, {6, 2});
  merged = cat({s0, s1}, /*dim=*/1);

  // Region 2. Mergeable because both slices are reshaped in the same way and
  // the outer dimension is split and catted.
  s0 = slice(merged, {0, 0}, {3, 4});
  s1 = slice(merged, {3, 0}, {6, 4});
  s0 = reshape(s0, {3, 4}, {12});
  s1 = reshape(s1, {3, 4}, {12});
  merged = cat({s0, s1}, /*dim=*/0);

  fusion->addOutput(merged);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 6}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  Fusion* complete_fusion = runtime->fusionSegments()->completeFusion();
  std::vector<Expr*> exprs = complete_fusion->exprs();

  // Only region 1 is not mergeable, so we expect to see only that region
  // contains two slices and one cat in the pre-segmenter fusion.
  EXPECT_THAT(exprs, Contains(IsA<SliceOp>()).Times(2));
  EXPECT_THAT(exprs, Contains(IsA<CatOp>()).Times(1));
  // The two permutes in region 0 are expected to be merged.
  EXPECT_THAT(exprs, Contains(IsPermute()).Times(1));
  // The two reshapes in region 1 stay as is and the two reshapes in region 2
  // are merged. Therefore, three reshapes in total.
  EXPECT_THAT(exprs, Contains(IsA<ReshapeOp>()).Times(3));
}

TEST_F(MoveSplitCatTest, MultipleCatsOnSameSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 2});
  TensorView* s0 = slice(in, {0, 0}, {2, 2});
  TensorView* s1 = slice(in, {2, 0}, {4, 2});
  TensorView* non_alias_out = [&]() {
    TensorView* t0 = set(s0);
    t0 = reshape(t0, {2, 2}, {4});
    TensorView* t1 = permute(s1, {1, 0});
    t1 = reshape(t1, {2, 2}, {4});
    // This cat doesn't cancel the split because the above transforms introduce
    // a self-mapping when the catted dimension is mapped.
    return cat({t0, t1}, /*dim=*/0);
  }();
  TensorView* alias_out = [&]() {
    TensorView* t0 = set(s0);
    TensorView* t1 = set(s1);
    // This cat cancels the split.
    return cat({t0, t1}, /*dim=*/0);
  }();

  fusion->addInput(in);
  fusion->addOutput(non_alias_out);
  fusion->addOutput(alias_out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({4, 2}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(in_tensor));
}

} // namespace nvfuser
