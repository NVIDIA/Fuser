// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser {

using FoldTest = NVFuserTest;

TEST_F(FoldTest, InnerSum) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeConcreteTensor({8, 8});
  fusion.addInput(inp);

  std::vector<std::pair<TensorView*, TensorView*>> fold_tensors =
      beginFold({inp}, {fusion.zeroVal(inp->dtype())}, {1});
  ASSERT_EQ(fold_tensors.size(), 1);
  auto& [prev_fold, next_elem] = fold_tensors.front();

  TensorView* combined = add(prev_fold, next_elem);
  TensorView* out_sum =
      finalizeReductionFold(
          {combined}, /*associative=*/true, /*commutative=*/true)
          .at(0);

  TensorView* out = set(out_sum);

  fusion.addOutput(out);

  fusion.printMath();

  EXPECT_TRUE(out_sum->definition()->isA<EndFoldOp>());

  inlineMost();

  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

// Example using multiple fold tensors at once to compute an argmax function
TEST_F(FoldTest, InnerArgMax) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeConcreteTensor({8, 8});
  fusion.addInput(inp);

  // create a tensor with indices to use as a second return value
  TensorView* idx = broadcast(iota(inp->axis(1)->extent()), {true, false});

  std::vector<std::pair<TensorView*, TensorView*>> fold_tensors = beginFold(
      {inp, idx},
      {IrBuilder::create<Val>(
           std::numeric_limits<double>::infinity(), inp->dtype()),
       fusion.zeroVal(idx->dtype())},
      {1});
  ASSERT_EQ(fold_tensors.size(), 2);
  auto& [prev_max, next_elem] = fold_tensors.front();
  auto& [prev_idx, next_idx] = fold_tensors.back();

  TensorView* pred = gt(next_elem, prev_max);
  TensorView* new_max = where(pred, next_elem, prev_max);
  TensorView* new_idx = where(pred, next_idx, prev_idx);

  // Note this is not commutative since we return the _first_ max index
  std::vector<TensorView*> out_tensors = finalizeReductionFold(
      {new_max, new_idx}, /*associative=*/true, /*commutative=*/false);
  ASSERT_EQ(out_tensors.size(), 2);

  TensorView* out_max = out_tensors.front();
  TensorView* out_idx = out_tensors.back();

  fusion.addOutput(out_max);
  fusion.addOutput(out_idx);

  fusion.printMath();

  EXPECT_TRUE(out_max->definition()->isA<EndFoldOp>());
  EXPECT_TRUE(out_idx->definition()->isA<EndFoldOp>());

  inlineMost();

  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

// Test an outer sum, which is similar to a grouped reduction
TEST_F(FoldTest, OuterSum) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeConcreteTensor({8, 8});
  fusion.addInput(inp);

  std::vector<std::pair<TensorView*, TensorView*>> fold_tensors =
      beginFold({inp}, {fusion.zeroVal(inp->dtype())}, {0});
  ASSERT_EQ(fold_tensors.size(), 1);
  auto& [prev_fold, next_elem] = fold_tensors.front();

  TensorView* combined = add(prev_fold, next_elem);
  TensorView* out_sum =
      finalizeReductionFold(
          {combined}, /*associative=*/true, /*commutative=*/true)
          .at(0);

  TensorView* out = set(out_sum);

  fusion.addOutput(out);

  fusion.printMath();

  EXPECT_TRUE(out_sum->definition()->isA<EndFoldOp>());

  inlineMost();

  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

TEST_F(FoldTest, NestedFolds) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeConcreteTensor({8, 8});
  fusion.addInput(inp);

  // Outer sum
  std::vector<std::pair<TensorView*, TensorView*>> fold_tensors =
      beginFold({inp}, {fusion.zeroVal(inp->dtype())}, {0});
  ASSERT_EQ(fold_tensors.size(), 1);
  auto& [prev_fold, next_elem] = fold_tensors.front();

  // When computing the outer sum, we first compute a max in the inner
  // dimension, then we broadcast and subtract it, then we exponentiate before
  // summing in the _outer_ dimension.
  TensorView* m = nullptr;
  { // Define the inner max
    std::pair<TensorView*, TensorView*> inner_fold_tensors =
        beginFold(
            {next_elem},
            {IrBuilder::create<Val>(
                -std::numeric_limits<double>::infinity(), next_elem->dtype())},
            {1})
            .at(0);
    TensorView* inner_combined = binaryOp(
        BinaryOpType::Max, inner_fold_tensors.first, inner_fold_tensors.second);
    m = finalizeReductionFold(
            {inner_combined},
            /*associative=*/true,
            /*commutative=*/true)
            .at(0);
  }
  fusion.printMath(false);
  TensorView* combined =
      add(prev_fold, exp(sub(next_elem, broadcast(m, {false, true}))));

  TensorView* out_sum =
      finalizeReductionFold(
          {combined}, /*associative=*/true, /*commutative=*/true)
          .at(0);

  TensorView* out = set(out_sum);

  fusion.addOutput(out);

  fusion.printMath();

  EXPECT_TRUE(out_sum->definition()->isA<EndFoldOp>());

  inlineMost();

  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

} // namespace nvfuser
