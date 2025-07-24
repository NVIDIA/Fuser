// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <iostream>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ExprSortTest = NVFuserTest;

using testing::ElementsAre;
using testing::SizeIs;

// Indirect normalization pattern with zero-dimensional tensors. Originally
// showed up in issue #537.
TEST_F(ExprSortTest, IndirectNormalizationWithZeroDimTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);

  auto tv3 = sum(tv2, {0});

  auto tv4 = set(tv1);
  auto tv5 = set(tv4);
  auto tv6 = set(tv5);
  auto tv7 = set(tv6);

  auto tv8 = add(tv3, tv7);
  fusion.addOutput(tv8);

  auto tv9 = broadcast(tv7, {true});
  auto tv10 = add(tv2, tv9);
  fusion.addOutput(tv10);

  // Here, the fusion graph looks like:
  //
  //         +----------------------------------------+
  //         |                                        |
  // tv0 -> tv2 -> (reduction) -> tv3 -+              |
  //                                   |              |
  // tv1 -> tv4 -> tv5 ->tv6 -> tv7  -+-> tv8         +-> tv10
  //                             |                    |
  //                             +--> (broadcast) -> tv9
  //
  // This fusion may appear to have the persistent pattern, but it
  // isn't the case. The reduction output, tv3, is never used with the
  // reduciton input, tv2. So, ComputeAtLogicalDomainMap detects no
  // domains that should not be inlined, which is correct.
  //
  // However, this could turn into a persistent kernel if tv7 and tv8
  // are grouped first in expression sorting. Since in this case both
  // of them are a 0-dim tensor, there's no constraint to block them
  // from merging. The exact ordering of expression merging depends
  // also on other factors such as the number of preceding
  // expressions, but the above fusion is designed to cause tv7 and
  // tv8 are merged first. And when that happens, notice that there's
  // now the normalization pattern from between tv2 and tv10. Also
  // remember that tv2 is fully inlined, so groups for tv2 and tv10
  // need to be grouped, but since tv7 and tv8 are now a single group,
  // merging groups for tv2 and tv10 would result in a cycle,
  // so they can never be merged.
  //
  // As far as I can see, as long as tv7 and tv8 are merged first,
  // there's no valid way to sort the exprssions. In this case, a
  // valid hierarchical sorting is:
  //
  // tv4
  // tv5
  // tv6
  // tv7
  // tv9
  // for
  //   tv2
  //   tv3
  //   tv10
  // tv8
  //
  // I.e., tv8 needs to come after tv10.
  //
  // Below, the reduction is also rfactor'ed, but the same story applies.

  tv3->split(0, 4);
  auto tv11 = tv3->rFactor({1});

  MaxLogicalDomainInfoSpanningTree tree(tv11);
  TransformPropagator tp(tv11);
  tree.traverse(&tp);

  inlineMost();

  // tv2 should be fully inlined
  ASSERT_TRUE(tv2->getComputeAtPosition() == tv2->nDims())
      << "Unexpected computeAt position of tv2. " << tv2->toString();

  ASSERT_NO_THROW({ GpuLower(&fusion).run(); });
}

// Similar to IndirectNormalizationWithZeroDimTensors, but
// without 0-dim tensors. The same pattern, but because of the CA/PA
// ordering mandated by loopReady(), the problem doesn't happen, i.e.,
// tv7 and t8 are not merged first.
TEST_F(ExprSortTest, IndirectInnerNormalization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);

  auto tv3 = sum(tv2, {1});

  auto tv4 = set(tv1);
  auto tv5 = set(tv4);
  auto tv6 = set(tv5);
  auto tv7 = set(tv6);

  auto tv8 = add(tv3, tv7);
  fusion.addOutput(tv8);

  auto tv9 = broadcast(tv7, {false, true});
  auto tv10 = add(tv2, tv9);
  fusion.addOutput(tv10);

  tv3->split(1, 4);
  auto tv11 = tv3->rFactor({-1});

  MaxLogicalDomainInfoSpanningTree tree(tv11);
  TransformPropagator tp(tv11);
  tree.traverse(&tp);

  inlineMost();

  // tv2 should be fully inlined
  ASSERT_TRUE(tv2->getComputeAtPosition() == tv2->nDims())
      << "Unexpected computeAt position of tv2. " << tv2->toString();

  ASSERT_NO_THROW({ GpuLower(&fusion).run(); });
}

namespace {

MATCHER_P(UnaryOpTypeIs, unary_op_type, "") {
  auto* unary_op = dynamic_cast<UnaryOp*>(arg);
  if (unary_op == nullptr) {
    return false;
  }
  return unary_op->getUnaryOpType() == unary_op_type;
}

} // namespace

TEST_F(ExprSortTest, SegmentedGroup_Unary) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeSymbolicTensor(1);
  TensorView* x = neg(in);
  TensorView* out0 = sin(x);
  TensorView* out1 = cos(x);

  fusion->addInput(in);
  fusion->addOutput(out0);
  fusion->addOutput(out1);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  SegmentedFusion* segmented_fusion = runtime->fusionSegments();
  ASSERT_THAT(segmented_fusion->groups(), SizeIs(1));
  SegmentedGroup* group = segmented_fusion->groups().front();

  EXPECT_THAT(
      group->stablyOrderedExprs(),
      ElementsAre(
          UnaryOpTypeIs(UnaryOpType::Neg),
          UnaryOpTypeIs(UnaryOpType::Sin),
          UnaryOpTypeIs(UnaryOpType::Cos)));
}

TEST_F(ExprSortTest, SegmentedGroup_Binary_SameOperand) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeSymbolicTensor(1);
  TensorView* out = neg(in);
  out = add(out, out);

  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  SegmentedFusion* segmented_fusion = runtime->fusionSegments();
  ASSERT_THAT(segmented_fusion->groups(), SizeIs(1));
  SegmentedGroup* group = segmented_fusion->groups().front();

  EXPECT_THAT(
      group->stablyOrderedExprs(),
      ElementsAre(IsA<UnaryOp>(), IsA<BinaryOp>()));
}

} // namespace nvfuser
