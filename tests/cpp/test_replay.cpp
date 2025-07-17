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
#include <transform_replay.h>

namespace nvfuser {

using testing::ContainerEq;
using testing::Each;
using testing::IsTrue;
using testing::Optional;
using testing::Property;
using testing::SizeIs;

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0];

  std::vector<at::Tensor> slices = at::split(in_tensor, {2, 3}, /*dim=*/-1);
  at::Tensor expected_out_tensor = at::cat(
      {slices[0].view({2, 2, 2}).permute({1, 0, 2}),
       slices[1].view({2, 2, 3}).permute({1, 0, 2})},
      /*dim=*/-1);

  EXPECT_TRUE(at::equal(out_tensor.as<at::Tensor>(), expected_out_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  auto out_tensor = out_tensors[0];

  std::vector<at::Tensor> slices = at::split(in_tensor, {2, 3}, /*dim=*/-1);
  at::Tensor expected_out_tensor = at::cat(
      {-slices[0].view({2, 2, 2}), -slices[1].view({2, 2, 3})},
      /*dim=*/-1);

  EXPECT_TRUE(at::equal(out_tensor.as<at::Tensor>(), expected_out_tensor));
}

TEST_F(ReplayTest, ReplaySplitOnReduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(1);
  TensorView* out = sum(in, {0});
  fusion.addInput(in);
  fusion.addOutput(out);

  constexpr int d = 2;
  out->setDeviceMesh(DeviceMesh::createForNumDevices(d));
  out->outer_split(0, d);

  TensorView* new_out = sum(in, {0});
  TransformReplay::selfReplay(
      out->domain(), new_out->domain(), /*ignore_reductions=*/false);
  fusion.replaceOutput(out, new_out);

  std::vector<IterDomain*> out_loop =
      fusion.outputs().at(0)->as<TensorView>()->getLoopDomain();
  EXPECT_THAT(out_loop, SizeIs(2));
  EXPECT_THAT(out_loop, Each(Property(&IterDomain::isReduction, IsTrue())));
}

TEST_F(ReplayTest, IgnoreSplitOnReduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(2);
  TensorView* x = sum(in, {0});
  TensorView* out = set(x);
  fusion.addInput(in);
  fusion.addOutput(out);

  constexpr int d = 2;
  x->setDeviceMesh(DeviceMesh::createForNumDevices(d));
  x->outer_split(0, d);
  x->outer_split(2, d);

  TransformReplay::selfReplay(
      x->domain(), out->domain(), /*ignore_reductions=*/true);

  EXPECT_THAT(
      out->getLoopDomain(),
      ElementsAre(
          Property(&IterDomain::isIteration, IsTrue()),
          Property(&IterDomain::isIteration, IsTrue())));
}

TEST_F(ReplayTest, LoopAndAllocation) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeSymbolicTensor(1);
  TensorView* out = set(in);
  fusion.addInput(in);
  fusion.addOutput(out);

  constexpr int d = 2;
  in->setDeviceMesh(DeviceMesh::createForNumDevices(d));
  in->outer_split(0, d);
  in->setAllocationDomain(in->getLoopDomain(), true);

  fusion.print();

  TransformReplay::selfReplay(in->domain(), out->domain());

  fusion.print();

  
  EXPECT_THAT(out->getLoopDomain(), SizeIs(2));
  EXPECT_THAT(out->getLoopDomain(), ContainerEq(out->getAllocationDomain()));
  EXPECT_THAT(out->getContiguity(), Each(Optional(IsTrue())));
}

} // namespace nvfuser
