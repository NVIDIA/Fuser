// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

// Core nvfuser headers
#include <csrc/bfs.h>
#include <csrc/fusion.h>
#include <csrc/ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/all_ops.h>

// ID model specific headers
#include <csrc/id_model/contiguity.h>
#include <csrc/id_model/id_model.h>
#include <csrc/id_model/utils.h>
#include <csrc/val_graph.h>
#include <csrc/val_graph_nodes.h>

// Testing utilities
#include <tests/cpp/utils.h> // For FusionWrapper and other test helpers

namespace nvfuser {

class ContigIDGroupsTest : public NVFuserTest {
 protected:
  ContigIDGroupsTest() {
    // Common setup for tests, if any
  }

  // Helper to construct a simple backward path for testing
  ExprPath<nvfuser::ExprGroup> buildBackwardExprPath(
      const ValGraph& graph,
      Expr* expr_in_path) {
    nvfuser::ExprGroup group = graph.toGroup(expr_in_path);
    ExprPath<nvfuser::ExprGroup> path;
    path.push_back({group, Direction::Backward});
    return path;
  }
};

// Test Scenario 1: split->in() becomes contiguous in backward pass.
// - split->in() is consistently ordered.
// - split->in() exclusively consumes its alloc domains.
// - The original allocation domains constituting split->in() are contiguous.
// - split->in() has no pre-existing resize_deps_ or non_divisible_deps_.
TEST_F(ContigIDGroupsTest, BackwardSplitInputBecomesContig) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Define original allocation domain (which will also be split->in())
  TensorView* tv0 = makeContigConcreteTensor({100});
  fusion.addInput(tv0);

  // 2. Create a TensorView whose domain will be split.
  // For this test, split->in() is orig_alloc_id itself.
  TensorView* tv1 = reshape(tv0, {100}, {25, 4});

  // 3. Perform a split operation.
  fusion.addOutput(tv1);

  tv1->setLoopDomain(tv1->getRootDomain());

  // --- Validation ---

  // 1. Build ValGraph via IdModel
  IdModel id_model(&fusion, /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const ValGraph& val_graph = id_model.idGraph(IdMappingMode::EXACT);

  // 2. Identify the expression and build backward path
  auto split_expr =
      dynamic_cast<Split*>(tv1->getLogicalDomain().at(0)->definition());
  ASSERT_NE(split_expr, nullptr) << "tv1 should have a split (SplitOp).";

  ExprPath<ExprGroup> backward_path =
      buildBackwardExprPath(val_graph, split_expr);
  ASSERT_FALSE(backward_path.empty()) << "Backward path should not be empty.";

  // 3. Instantiate ContigIDGroups
  // For backward analysis starting from tv1, treat tv1's loop domains as the
  // initial 'alloc_domains'.
  ContigIDGroups contig_finder(
      tv1->getMaybeAllocationDomain(),
      std::vector<bool>(2, true),
      backward_path,
      val_graph);

  // 4. Perform assertion
  // We expect tv0->axis(0) to be marked as contiguous after backward
  // propagation.
  ValGroup tv1_axis0_vg = val_graph.toGroup(tv1->axis(0));
  EXPECT_TRUE(contig_finder.contigIDs().contains(tv1_axis0_vg))
      << "Input to Reshape (tv1->axis(0)) was not found to be contiguous "
      << "after backward analysis from tv1.";
}

// Test Scenario 2: Input of a reshape is not contiguous when the reshape
// output's allocation domains are initially marked as non-contiguous in
// backward pass.
TEST_F(ContigIDGroupsTest, BackwardReshapeInputNotContig) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Define original allocation domain
  TensorView* tv0 = makeContigConcreteTensor({100});
  fusion.addInput(tv0);

  // 2. Create a TensorView via reshape.
  TensorView* tv1 = reshape(tv0, {100}, {25, 4});
  fusion.addOutput(tv1);

  // Set loop domains for tv1 based on its logical structure post-reshape
  tv1->setLoopDomain(tv1->getRootDomain());

  // --- Validation ---

  // 1. Build ValGraph via IdModel
  IdModel id_model(&fusion, /*build_graphs=*/true, /*allow_self_mapping=*/true);
  const ValGraph& val_graph = id_model.idGraph(IdMappingMode::EXACT);

  // 2. Identify the expression (SplitOp underlying tv1's logical domain) and
  // build backward path Reshape internally creates splits/merges. We target the
  // split.
  auto split_expr =
      dynamic_cast<Split*>(tv1->getLogicalDomain().at(0)->definition());
  ASSERT_NE(split_expr, nullptr)
      << "tv1's logical domain should be based on a Split operation after reshape.";

  ExprPath<ExprGroup> backward_path =
      buildBackwardExprPath(val_graph, split_expr);
  ASSERT_FALSE(backward_path.empty()) << "Backward path should not be empty.";

  // 3. Instantiate ContigIDGroups
  // For backward analysis starting from tv1, treat tv1's allocation domains as
  // initially non-contiguous.
  std::vector<IterDomain*> tv1_alloc_domains = tv1->getMaybeAllocationDomain();
  std::vector<bool> tv1_initial_contiguity(
      tv1_alloc_domains.size(), false); // Key difference

  ContigIDGroups contig_finder(
      tv1_alloc_domains, tv1_initial_contiguity, backward_path, val_graph);

  // 4. Perform assertion
  // We expect tv1->axis(0) (which was part of the initial non-contiguous set
  // for tv1's domains) to remain NOT marked as contiguous.
  ValGroup tv1_axis0_vg = val_graph.toGroup(tv1->axis(0));
  EXPECT_FALSE(contig_finder.contigIDs().contains(tv1_axis0_vg))
      << "Output domain tv1->axis(0) was unexpectedly found to be contiguous. ";
}

// Test Scenario: Backward propagation through a split where tv1's allocation
// domains are reordered relative to logical domains and are initially marked
// as contiguous. The test verifies the resulting contiguity of an output domain
// under these specific conditions after the backward pass.
TEST_F(ContigIDGroupsTest, BackwardSplitWithReorderedAllocAndInitialContig) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // 1. Define original allocation domain
  TensorView* tv0 = makeContigConcreteTensor({100});
  fusion.addInput(tv0);

  // 2. Create a TensorView via reshape.
  TensorView* tv1 = reshape(tv0, {100}, {25, 4});
  fusion.addOutput(tv1);

  // Set loop domains for tv1 based on its root domain.
  tv1->setLoopDomain(tv1->getRootDomain());

  // Reorder the allocation domains to test the case where the allocation
  // domains are not in the same order as the logical domains. Mark them as
  // contiguous.
  tv1->setAllocationDomain(
      {tv1->getLogicalDomain()[1], tv1->getLogicalDomain()[0]}, {true, true});

  // --- Validation ---

  // 1. Build ValGraph via IdModel
  IdModel id_model(&fusion, /*build_graphs=*/true);
  const ValGraph& val_graph = id_model.idGraph(IdMappingMode::EXACT);

  // 2. Identify the expression (SplitOp underlying tv1's logical domain) and
  // build backward path Reshape internally creates splits/merges. We target the
  // split.
  auto split_expr =
      dynamic_cast<Split*>(tv1->getLogicalDomain().at(0)->definition());
  ASSERT_NE(split_expr, nullptr)
      << "tv1's logical domain should be based on a Split operation after reshape.";

  ExprPath<ExprGroup> backward_path =
      buildBackwardExprPath(val_graph, split_expr);
  ASSERT_FALSE(backward_path.empty()) << "Backward path should not be empty.";

  // 3. Instantiate ContigIDGroups
  // For backward analysis starting from tv1, treat tv1's allocation domains as
  // initially all contiguous.
  std::vector<IterDomain*> tv1_alloc_domains = tv1->getMaybeAllocationDomain();
  std::vector<bool> tv1_initial_contiguity(
      tv1_alloc_domains.size(), true); // All true

  ContigIDGroups contig_finder(
      tv1_alloc_domains, tv1_initial_contiguity, backward_path, val_graph);

  // 4. Perform assertion
  // Check the contiguity of tv1->axis(0) (an IterDomain from the loop domain).
  // tv1's allocation domains, which are {tv1->getLogicalDomain()[1],
  // tv1->getLogicalDomain()[0]}, were initially marked as {true, true} for
  // contiguity. The expectation here (EXPECT_FALSE for the ValGroup
  // corresponding to tv1->axis(0)) means that after backward propagation
  // through the split, this ValGroup is no longer considered contiguous by the
  // finder. This scenario tests how initial contiguity of allocation domains
  // (which are defined using these logical/loop IterDomains) interacts with
  // reordered allocation domain definitions and backward split handling.
  ValGroup tv1_axis0_vg = val_graph.toGroup(tv1->axis(0));
  EXPECT_FALSE(contig_finder.contigIDs().contains(tv1_axis0_vg))
      << "Output domain tv1->axis(0) was unexpectedly found to be contiguous. ";
}

// Placeholder for future tests

} // namespace nvfuser
