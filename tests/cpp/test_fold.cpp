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

#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/arith.h>
#include <tests/cpp/utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser {

using FoldTest = NVFuserTest;

TEST_F(FoldTest, CreateNodes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* inp = makeSymbolicTensor(2);
  fusion.addInput(inp);

  FoldGroup g({inp}, {fusion.zeroVal()}, {1});
  TensorView* comb = add(g.prevFoldTensor(), g.nextElementTensor());

  // finalizeReduction returns a std::vector<TensorView*>
  TensorView* s =
      g.finalizeReduction(comb, /*associative=*/true, /*commutative=*/true);

  fusion.addOutput(s);

  fusion.printMath();
}

} // namespace nvfuser
