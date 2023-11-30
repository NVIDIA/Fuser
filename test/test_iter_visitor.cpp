// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <iter_visitor.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>

namespace nvfuser {

using IterVisitorTest = NVFuserTest;

// Quick test of traversing attributes with IterVisitor
TEST_F(IterVisitorTest, IterVisitorTraverseAttributes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(1L),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(1L))}});
  fusion.addOutput(tv1);

  auto tv1_resize = tv1->axis(0)->definition()->as<Resize>();

  auto stmts = StmtSort::getStmts(&fusion, true, true);

  // Make sure the expansion parameters of tv1_resize are visited
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), tv1_resize->leftExpand()) !=
          stmts.end(),
      "Resize left expand parameter not found");
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), tv1_resize->rightExpand()) !=
          stmts.end(),
      "Resize right expand parameter not found");
}

// Test that traversing siblings with IterVisitor visits "orphans", i.e. unused
// outputs of multi-output Exprs.
TEST_F(IterVisitorTest, IterVisitorTraverseSiblings) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto wf = Welford(tv0, {0});
  // wf.var_sum is used, but wf.avg and wf.n are orphaned
  auto tv1 = neg(wf.var_sum);
  fusion.addOutput(tv1);

  auto stmts = StmtSort::getStmts(
      &fusion,
      /*traverse_all_paths*/ false,
      /*traverse_attributes*/ false,
      /*traverse_siblings*/ true);

  // Make sure the expansion parameters of tv1_resize are visited
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), wf.avg) != stmts.end(),
      "Welford avg not traversed");
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), wf.n) != stmts.end(),
      "Welford n not traversed");

  // Test getting statements "to" a tensor with siblings
  stmts = StmtSort::getStmtsTo(
      {wf.n},
      /*traverse_all_paths*/ false,
      /*traverse_attributes*/ false,
      /*traverse_siblings*/ true);
  // Make sure the expansion parameters of tv1_resize are visited
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), wf.avg) != stmts.end(),
      "Welford avg not traversed in getStmtsTo({n})");
  NVF_CHECK(
      std::find(stmts.begin(), stmts.end(), wf.var_sum) != stmts.end(),
      "Welford var_sum not traversed in getStmtsTo({n})");
}

TEST_F(IterVisitorTest, IterVisitorGetInputsTo) {
  // Test that IterVisitor::getInputsTo() will stop further traverse when
  // reaching the target tensors
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto a = makeSymbolicTensor(1);
  auto b = makeSymbolicTensor(1);
  auto c = makeSymbolicTensor(1);

  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addInput(c);

  auto d = add(b, c);
  auto e = add(a, d);

  fusion.addOutput(e);

  auto inputs = IterVisitor::getInputsTo({e}, {a, d});
  std::unordered_set<Val*> inputs_set(inputs.begin(), inputs.end());

  EXPECT_EQ(inputs_set, std::unordered_set<Val*>({a, d}));
}

} // namespace nvfuser
