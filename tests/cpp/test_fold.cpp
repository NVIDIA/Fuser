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
#include <ops/arith.h>
#include <tests/cpp/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser {

using FoldTest = NVFuserTest;

TEST_F(FoldTest, Sum) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeConcreteTensor({8, 8});
  fusion.addInput(inp);

  FoldGroup g =
      FoldGroup::makeFoldGroup({inp}, {fusion.zeroVal(inp->dtype())}, {1});
  TensorView* combined = add(g.prevFoldTensor(), g.nextElementTensor());
  TensorView* out_sum =
      g.finalizeReduction(combined, /*associative=*/true, /*commutative=*/true);

  fusion.addOutput(out_sum);

  fusion.printMath();

  EXPECT_TRUE(out_sum->definition()->isA<FinalizeReductionOp>());
  EXPECT_EQ(
      out_sum->definition()->as<FinalizeReductionOp>()->beginFoldOp(),
      g.beginOp());

  inlineMost();

  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

} // namespace nvfuser
