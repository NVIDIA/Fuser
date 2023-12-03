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
using testing::Contains;
using testing::IsSupersetOf;
using testing::UnorderedElementsAre;

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
  EXPECT_THAT(stmts, Contains(tv1_resize->leftExpand())) << "Resize left expand parameter not found";
  EXPECT_THAT(stmts, Contains(tv1_resize->rightExpand())) << "Resize right expand parameter not found";
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

  EXPECT_THAT(stmts, Contains(wf.avg)) << "Welford avg not traversed";
  EXPECT_THAT(stmts, Contains(wf.n)) << "Welford n not traversed";

  // Test getting statements "to" a tensor with siblings
  stmts = StmtSort::getStmtsTo(
      &fusion,
      {wf.n},
      /*traverse_all_paths=*/false,
      /*traverse_attributes=*/false,
      /*traverse_siblings=*/true);
  EXPECT_THAT(stmts, Contains(wf.avg)) << "Welford avg not traversed in getStmtsTo({n})";
  EXPECT_THAT(stmts, Contains(wf.var_sum)) << "Welford var_sum not traversed in getStmtsTo({n})";
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

  std::vector<Val*> inputs = IterVisitor::getInputsTo({e}, {a, d});
  EXPECT_THAT(inputs, UnorderedElementsAre(a, d));
}

TEST_F(IterVisitorTest, NonTerminatingOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(1);
  TensorView* b = set(a);
  TensorView* c = set(b);
  TensorView* d = set(c);
  TensorView* e = set(d);

  fusion.addInput(a);
  fusion.addOutput(c);
  fusion.addOutput(e);

  // Even though `c` is a non-terminating output, `d` and `e` should still be
  // considered in between. This is because `StmtSort::getExprsBetween`
  // traverses from `to` along use-def chains until it hits `from`.
  EXPECT_THAT(StmtSort::getExprsBetween(&fusion, {a}, {c, e}),
              IsSupersetOf({d->definition(), e->definition()}));
}

} // namespace nvfuser
