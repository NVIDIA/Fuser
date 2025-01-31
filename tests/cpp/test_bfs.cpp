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

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <val_graph_visitor.h>

#include <fstream>
#include <iostream>

namespace nvfuser {

using BFSTest = NVFuserTest;

// BFS traversal test with a simple exact graph
TEST_F(BFSTest, ValGraphBFS1) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion->addOutput(tv2);

  // tv0: [i0, i1]
  // tv1: [i0, i1]
  // tv2: [i0, i1]

  // Schedule tv0 and tv1 in the same way
  tv0->merge(0, 1)->split(0, 4);
  tv1->merge(0, 1)->split(0, 4);
  // Schedule tv1 similarly but with a reordered merge
  tv2->merge(1, 0)->split(0, 4);

  // tv0: [i0*i1/4, 4]
  // tv1: [i0*i1/4, 4]
  // tv2: [i1*i0/4, 4]

  const IdModel id_model(fusion.get());
  const ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

  ValGroups tv0_loop_groups = graph.toGroups(tv0->getLoopDomain());
  ValGroups tv1_loop_groups = graph.toGroups(tv1->getLoopDomain());
  ValGroups tv2_loop_groups = graph.toGroups(tv2->getLoopDomain());

  // Since the loop domains of tv0 and tv1 are grouped together, the
  // path between them is empty
  ExprPath<ExprGroup> tv1_to_tv0 =
      ValGraphBFS::getExprGroupsBetween(graph, tv1_loop_groups, tv0_loop_groups)
          .first;
  EXPECT_TRUE(tv1_to_tv0.empty());

  // Traversal should fail if not all dependencies are met
  ValGroups incomplete_tv1_loop_groups;
  incomplete_tv1_loop_groups.pushBack(
      graph.toGroup(tv1->getLoopDomain().at(0)));
  EXPECT_THAT(
      [&]() {
        ValGraphBFS::getExprGroupsBetween(
            graph, incomplete_tv1_loop_groups, tv0_loop_groups);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("BFS traversal could not visit some nodes")));

  // On the other hand, the loop domains of tv2 are produced through
  // the reverse merge, so they aren't mapped with the tv1 loop
  // domains. The path between them should look like traversing from
  // tv2 loop domain backward to its root and then forward from tv1 root to
  // tv1 loop domain.
  ExprPath<ExprGroup> tv2_to_tv1 =
      ValGraphBFS::getExprGroupsBetween(graph, tv2_loop_groups, tv1_loop_groups)
          .first;

  ExprPath<ExprGroup> tv2_to_tv1_ref;
  tv2_to_tv1_ref.emplace_back(
      graph.toGroup(tv2->axis(0)->definition()), Direction::Backward);
  tv2_to_tv1_ref.emplace_back(
      graph.toGroup(tv2->axis(0)->definition()->input(0)->definition()),
      Direction::Backward);
  tv2_to_tv1_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()->input(0)->definition()),
      Direction::Forward);
  tv2_to_tv1_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()), Direction::Forward);

  EXPECT_EQ(tv2_to_tv1, tv2_to_tv1_ref);
}

// Traversal to partial reachable nodes. See also the comment in
// ValGraphBFS::getShortestExprPath<ExprGroup>.
TEST_F(BFSTest, ValGraphBFS2) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  // tv1: [i0, i1, i2]
  // tv1: [i0, i1, i2]

  tv1->merge(1, 2)->merge(0, 1);

  // tv0: [i0, i1, i2]
  // tv1: [i0*(i1*i2)]

  const IdModel id_model(fusion.get());
  const ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

  ValGroups tv0_loop_groups = graph.toGroups(tv0->getLoopDomain());
  ValGroups tv1_loop_groups = graph.toGroups(tv1->getLoopDomain());

  // Since the loop domains of tv0 and tv1 are grouped together, the
  // path between them is empty
  ExprPath<ExprGroup> tv1_to_tv0 =
      ValGraphBFS::getExprGroupsBetween(graph, tv1_loop_groups, tv0_loop_groups)
          .first;

  ExprPath<ExprGroup> tv1_to_tv0_ref;
  tv1_to_tv0_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()), Direction::Backward);
  tv1_to_tv0_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()->input(1)->definition()),
      Direction::Backward);

  EXPECT_EQ(tv1_to_tv0, tv1_to_tv0_ref);

  // Grab the path from tv1 to only the i1 and i2 domains of tv0
  // without i0. The path should still be the same.
  ValGroups tv0_partial_groups;
  tv0_partial_groups.pushBack(graph.toGroup(tv0->axis(1)));
  tv0_partial_groups.pushBack(graph.toGroup(tv0->axis(2)));
  ExprPath<ExprGroup> tv1_to_tv0_partial =
      ValGraphBFS::getExprGroupsBetween(
          graph, tv1_loop_groups, tv0_partial_groups)
          .first;

  EXPECT_EQ(tv1_to_tv0_partial, tv1_to_tv0_ref);
}

// Check if a shorter path is taken
TEST_F(BFSTest, ValGraphBFS3) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({16});
  fusion->addInput(tv0);

  // shorter path
  auto tv1 = reshape(tv0, {16}, {4, 4});

  // longer path
  auto tv2 = reshape(tv0, {16}, {8, 2});
  auto tv3 = reshape(tv2, {8, 2}, {4, 4});

  auto tv4 = add(tv1, tv3);

  fusion->addOutput(tv4);

  // tv0: [i0]
  // tv1: [i0/4, 4]
  // tv2: [i0/8, 2]
  // tv3: [i0/8*2/4, 4]
  // tv4: [i0/4, 4]

  // Traversal from tv4 to tv0 can be {tv4 -> tv1 -> tv0} or {tv4 ->
  // tv3 -> tv2 -> tv0}. The former should be seletected as it's shorter

  const IdModel id_model(fusion.get());
  const ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

  ValGroups tv4_groups = graph.toGroups(tv4->getLoopDomain());
  ValGroups tv0_groups = graph.toGroups(tv0->getLoopDomain());

  ExprPath<ExprGroup> tv4_to_tv0 =
      ValGraphBFS::getExprGroupsBetween(graph, tv4_groups, tv0_groups).first;
  ExprPath<ExprGroup> tv4_to_tv0_ref;
  tv4_to_tv0_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()), Direction::Backward);

  ASSERT_EQ(tv4_to_tv0, tv4_to_tv0_ref);
}

// BFS traversal of a graph with a cycle
TEST_F(BFSTest, ValGraphBFS4) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 8});
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {2, 8}, {16});
  auto tv2 = reshape(tv1, {16}, {4, 4});
  auto tv3 = reshape(tv2, {4, 4}, {16});
  auto tv4 = add(tv1, tv3);

  fusion->addOutput(tv4);

  // tv0: [i0, i1]
  // tv1: [i2] // i2 = merge(i0, i1)
  // tv2: [i3, i4] // i3, i4 = split(i2)
  // tv3: [i5] // merge(i3, i4)
  // tv4: [i6] // i6 = i5

  // The tv4 addition makes the sole domain of tv4 mapped with those of
  // tv3 and tv1, i.e., these domains are grouped together:
  //
  // i2, i5, i6
  //
  // Since there's a path from i2 to i5 (a merge and a split), this
  // means the graph is no longer a DAG.

  // Make sure the BFS traversal should still work even with a cycle.

  const IdModel id_model(fusion.get());
  const ValGraph& graph = id_model.idGraph(IdMappingMode::EXACT);

  ValGroups tv4_groups = graph.toGroups(tv4->getLoopDomain());
  ValGroups tv0_groups = graph.toGroups(tv0->getLoopDomain());

  // Traversal from tv4 to tv0 can go through the reshape ops of tv2
  // and tv3, but the shortest path should be just one merge for tv1

  ExprPath<ExprGroup> tv4_to_tv0 =
      ValGraphBFS::getExprGroupsBetween(graph, tv4_groups, tv0_groups).first;

  ExprPath<ExprGroup> tv4_to_tv0_ref;
  tv4_to_tv0_ref.emplace_back(
      graph.toGroup(tv1->axis(0)->definition()), Direction::Backward);

  ASSERT_EQ(tv4_to_tv0, tv4_to_tv0_ref);
}

// Testing IRBFS::getReachableValsFrom with a resize fusion
TEST_F(BFSTest, IRBFSGetReachableValsFrom) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 20});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // Slice the inner domain
  auto tv1 = slice(
      tv0,
      {{fusion.zeroVal(), fusion.zeroVal()},
       {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(99)}});

  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  tv1->setLoopDomain(tv1->getRootDomain());

  auto tv2_loop_id = tv0->getLoopDomain().at(1)->cloneWithoutRFactor();

  IrBuilder::create<Resize>(
      tv2->getLogicalDomain().at(1),
      tv2_loop_id,
      IrBuilder::create<Val>(-1, DataType::Index),
      IrBuilder::create<Val>(-1, DataType::Index));

  tv2->setLoopDomain({tv2->getLogicalDomain().at(0), tv2_loop_id});

  // Just between iter domains in the same tensor. Unlike
  // DependencyCheck, the direction doesn't matter
  {
    auto reachable_vals = getReachableValsFrom<IRBFS>(
        {tv1->getLogicalDomain().begin(), tv1->getLogicalDomain().end()},
        {tv1->getRootDomain().begin(), tv1->getRootDomain().end()});
    std::vector<Val*> ref{
        tv1->getRootDomain().begin(), tv1->getRootDomain().end()};
    EXPECT_EQ(reachable_vals, ref)
        << "Root domain not reachable: " << tv1->toString();
  }

  // The tv2 loop domain is reachable from its logical domain
  {
    auto reachable_vals = getReachableValsFrom<IRBFS>(
        {tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end()},
        {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()});
    std::vector<Val*> ref{
        tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()};
    EXPECT_EQ(reachable_vals, ref)
        << "Loop domain not reachable: " << tv2->toString();
  }

  // If only one of the logical domain is given, only the domain that
  // is dervied from it is returned
  {
    auto reachable_vals = getReachableValsFrom<IRBFS>(
        {tv2->getLogicalDomain().at(0)},
        {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()});
    std::vector<Val*> ref{tv2->getLoopDomain().at(0)};
    EXPECT_EQ(reachable_vals, ref)
        << "Loop domain not reachable: " << tv2->toString();
  }
}

// Testing IRBFS::getValsBetween with a reshape fusion
TEST_F(BFSTest, IRBFSGetValsBetween) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 20});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, shape, {shape[0] * shape[1]});

  auto tv2 = reshape(tv1, {shape[0] * shape[1]}, {shape[1], shape[0]});

  fusion.addOutput(tv2);

  // Use the input 2D domain as the loop domain of all tensors
  tv1->setLoopDomain(tv1->getRootDomain());
  std::vector<IterDomain*> tv2_loop_domain{
      tv0->getLoopDomain().at(0)->cloneWithoutRFactor(),
      tv0->getLoopDomain().at(1)->cloneWithoutRFactor()};

  IrBuilder::create<Merge>(
      tv2->getRootDomain().at(0), tv2_loop_domain[0], tv2_loop_domain[1]);
  tv2->setLoopDomain(tv2_loop_domain);

  // Unlike DependencyCheck::getAllValsBetween, the direction doesn't
  // matter.
  {
    auto all_vals = getValsBetween<IRBFS>(
        {tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end()},
        {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()});
    std::vector<Val*> ref;
    for (auto id : tv2->getLogicalDomain()) {
      ref.push_back(id);
    }
    for (auto id : tv2->getRootDomain()) {
      ref.push_back(id);
    }
    for (auto id : tv2->getLoopDomain()) {
      ref.push_back(id);
    }
    EXPECT_EQ(all_vals, ref);
  }

  // Since only one of the logical domain is given, it doesn't reach
  // anywhere, returning an empty vector
  {
    auto all_vals = getValsBetween<IRBFS>(
        {tv2->getLogicalDomain().at(0)},
        {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()});
    EXPECT_TRUE(all_vals.empty());
  }
}

TEST_F(BFSTest, FindDependencyWithIRBFSGetValsBetween) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  fusion.addOutput(tv1);

  // [i0, i1, i2, i3]
  tv1->merge(0, 2);
  // [i0*i2, i1, i3]
  tv1->merge(1, 2);
  // [i0*i2, i1*i3]
  tv1->reorder({{0, 1}});
  // [i1*i3, i0*i2]

  auto all_deps = getDependenciesTo<IRBFS>(
      {tv1->getLogicalDomain().begin(), tv1->getLogicalDomain().end()},
      {tv1->axis(0)});

  std::vector<Val*> ref{
      tv1->getLogicalDomain().at(1), tv1->getLogicalDomain().at(3)};

  EXPECT_EQ(all_deps, ref);
}

// Test directed getExprsBetween
TEST_F(BFSTest, TraversalDirection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({32});
  fusion.addInput(tv0);

  // Create two reshape paths leading to the same output shape

  auto tv1 = reshape(tv0, {16}, {2, 8});
  auto tv2 = reshape(tv1, {2, 8}, {2, 2, 4});
  auto tv3 = reshape(tv2, {2, 2, 4}, {2, 2, 2, 2});
  auto tv4 = reshape(tv3, {2, 2, 2, 2}, {2, 2, 4});
  auto tv5 = reshape(tv4, {2, 2, 4}, {2, 8});
  auto tv6 = reshape(tv5, {2, 8}, {16});
  auto tv7 = reshape(tv6, {16}, {4, 4});
  auto tv8 = reshape(tv7, {4, 4}, {16});

  auto tv9 = reshape(tv0, {16}, {8, 2});
  auto tv10 = reshape(tv9, {8, 2}, {16});

  // Merge point
  auto tv11 = add(tv8, tv10);
  fusion.addOutput(tv11);

  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  // Shortest path from the input to tv7 should forward the second
  // path and then move one Merge backward
  auto shortest_path = ValGraphBFS::getExprGroupsBetween(
                           exact_graph,
                           exact_graph.toGroups(tv0->getLogicalDomain()),
                           exact_graph.toGroups(tv7->getLogicalDomain()),
                           /*require_all_to_visited=*/true,
                           Direction::Undefined)
                           .first;
  ValGraphBFS::ExprPath shortest_path_reference = {
      {exact_graph.toGroup(tv9->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv10->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv8->axis(-1)->definition()), Direction::Backward}};
  EXPECT_EQ(shortest_path, shortest_path_reference)
      << "Reference: " << shortest_path_reference
      << ". Actual: " << shortest_path;

  // Forward only path should take tv1 through tv7
  auto forward_path = ValGraphBFS::getExprGroupsBetween(
                          exact_graph,
                          exact_graph.toGroups(tv0->getLogicalDomain()),
                          exact_graph.toGroups(tv7->getLogicalDomain()),
                          /*require_all_to_visited=*/true,
                          Direction::Forward)
                          .first;
  ValGraphBFS::ExprPath forward_path_reference = {
      {exact_graph.toGroup(tv1->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv2->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv3->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv4->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv5->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv6->axis(-1)->definition()), Direction::Forward},
      {exact_graph.toGroup(tv7->axis(-1)->definition()), Direction::Forward}};
  EXPECT_EQ(forward_path, forward_path_reference)
      << "Reference: " << forward_path_reference
      << ". Actual: " << forward_path;

  // Backward only path should not find anything
  auto backward_path = ValGraphBFS::getExprGroupsBetween(
                           exact_graph,
                           exact_graph.toGroups(tv0->getLogicalDomain()),
                           exact_graph.toGroups(tv7->getLogicalDomain()),
                           /*require_all_to_visited=*/false,
                           Direction::Backward)
                           .first;
  EXPECT_TRUE(backward_path.empty()) << "Actual: " << backward_path;
}

// A simple test for BFSWithPermissiveDependence
TEST_F(BFSTest, IRBFSPermissiveTraversal) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  // [i0, i1]
  auto tv2 = set(tv0);
  fusion.addOutput(tv2);

  auto tv3 = set(tv1);
  fusion.addOutput(tv3);

  auto i0 = tv2->getLogicalDomain().at(0);
  [[maybe_unused]] auto i1 = tv2->getLogicalDomain().at(0);

  auto i2 = tv3->getLogicalDomain().at(0);

  // i3 = merge(i0, i1)
  tv2->flatten();
  [[maybe_unused]] auto i3 = tv2->axis(0);
  // i4, i5 = split(i3)
  tv2->split(0, 4);
  [[maybe_unused]] auto i4 = tv2->axis(0);
  [[maybe_unused]] auto i5 = tv2->axis(1);

  // from: [i0, i2]
  // to: [i4]
  // -> forward merge, forward split
  {
    auto path = getExprsBetween<IRPermissiveBFS>(
                    {i0, i2}, {i4}, /*require_all_to_visited=*/false)
                    .first;
    EXPECT_EQ(path.size(), 2);
    // fwd merge
    EXPECT_EQ(path.at(0).first, i3->definition());
    EXPECT_EQ(path.at(0).second, Direction::Forward);
    // fwd split
    EXPECT_EQ(path.at(1).first, i4->definition());
    EXPECT_EQ(path.at(1).second, Direction::Forward);
  }

  // from: [i4, i5]
  // to: [i1]
  // -> bwd split, bwd merge
  {
    auto path = getExprsBetween<IRPermissiveBFS>(
                    {i4, i5}, {i1}, /*require_all_to_visited=*/false)
                    .first;
    EXPECT_EQ(path.size(), 2);
    // bwd split
    EXPECT_EQ(path.at(0).first, i4->definition());
    EXPECT_EQ(path.at(0).second, Direction::Backward);
    // bwd merge
    EXPECT_EQ(path.at(1).first, i3->definition());
    EXPECT_EQ(path.at(1).second, Direction::Backward);
  }
}

TEST_F(BFSTest, IRBFSPermissiveTraversal2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->split(0, 4);

  // T1_g_float[iS5{( ceilDiv(( i0 * i2 ), 4) )}, iS6{4}]
  //  logical domain : (iS2{i0}, iS3{i2})
  //  contiguity: t t
  //   Merge: iS2{i0} and iS3{i2} -> iS4{( i0 * i2 )}
  //   Split: iS4{( i0 * i2 )} by factor 4 -> iS5{( ceilDiv(( i0 * i2 ), 4) )},
  //   iS6{4}
  //  loop domain : (iS5{( ceilDiv(( i0 * i2 ), 4) )}, iS6{4})
  fusion.print();

  auto iS5 = tv1->axis(0);
  auto iS6 = tv1->axis(1);

  // When starting with just iS5 witout iS6, the permissive traversal
  // allows to visit the split expr node, even though iS6 is
  // missing. The next set of nodes to visit after the split are its
  // neighbors, which includes iS6. However, it does not seem to make
  // any intuitive sense to allow this visit. The split expr is visited
  // because one of its outputs, iS5, is visited. That in turn allowing to
  // visit the missing split output, iS6, does not seem to make sense.

  // Make sure iS6 is not reachable from iS5
  EXPECT_FALSE(getExprsBetween<IRPermissiveBFS>(
                   {iS5},
                   {iS6},
                   /*require_all_to_visited=*/false)
                   .second);
}

} // namespace nvfuser
