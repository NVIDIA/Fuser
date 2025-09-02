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

#include <iter_visitor.h>
#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fstream>
#include <iostream>

namespace nvfuser {

namespace {

void checkGetAllStmts(Fusion* fusion) {
  // Check if StmtSort can grab all IDS, including those that are
  // producers of root IDs
  auto all_stmts = StmtSort::getAllStmts(fusion, /*traverse_members=*/true);
  std::unordered_set<Statement*> all_stmt_set{
      all_stmts.begin(), all_stmts.end()};
  for (auto tv : fusion->allTvs()) {
    for (auto id_or_expr : tv->domain()->allStatements()) {
      EXPECT_TRUE(all_stmt_set.count(id_or_expr))
          << "Not found: " << id_or_expr->toString() << " of "
          << tv->toString();
    }
  }
}

} // namespace

class LoopDomainSchedulingTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_F(LoopDomainSchedulingTest, ReshapeSplitThenMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // One split reshape and then merge reshape. Schedule the loop
  // domains of all tensors with the initial pre-reshape domain.

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {10}, {2, 5});
  auto tv3 = set(tv2);
  auto tv4 = reshape(tv3, {2, 5}, {10});
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  std::vector<IterDomain*> ref = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref);

  for (auto tv : fusion.allTvs()) {
    tv->split(0, 3);
  }

  inlineMost();

  IdModel id_model(&fusion, /*build_models=*/true);

  ref = tv1->getLoopDomain();
  for (auto tv : fusion.allTvs()) {
    EXPECT_EQ(ref.size(), tv->getLoopDomain().size());

    if (!tv->isFusionInput()) {
      EXPECT_EQ(tv->getComputeAtPosition(), 2) << tv->toString();
    }

    for (const auto i : arange(ref.size())) {
      EXPECT_TRUE(id_model.idGraph(IdMappingMode::EXACT)
                      .disjointValSets()
                      .strictAreMapped(ref.at(i), tv->getLoopDomain().at(i)))
          << "Not mapped: " << ref.at(i)->toString() << ", "
          << tv->getLoopDomain().at(i)->toString() << ", " << tv->toString();
      // Except for the input, they should be mapped in the loop graph too
      if (!tv->isFusionInput()) {
        EXPECT_TRUE(id_model.idGraph(IdMappingMode::LOOP)
                        .disjointValSets()
                        .strictAreMapped(ref.at(i), tv->getLoopDomain().at(i)))
            << "Not mapped: " << ref.at(i)->toString() << ", "
            << tv->getLoopDomain().at(i)->toString() << ", " << tv->toString();
      }
    }
  }

  checkGetAllStmts(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Test loop domain scheduling with slice. More test cases can also be
// found in test_resize.cpp
TEST_F(LoopDomainSchedulingTest, Slice) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 =
      slice(tv0, {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(99)}});

  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : fusion.allTvs()) {
    tv->split(0, 32);
  }

  inlineMost();
  IdModel id_model(&fusion, /*build_models=*/false);
  id_model.buildExactGraph();

  ref_loop = tv0->getLoopDomain();

  for (auto tv : fusion.allTvs()) {
    EXPECT_EQ(ref_loop.size(), tv->getLoopDomain().size());
    for (const auto i : arange(ref_loop.size())) {
      EXPECT_TRUE(
          id_model.idGraph(IdMappingMode::EXACT)
              .disjointValSets()
              .strictAreMapped(ref_loop.at(i), tv->getLoopDomain().at(i)))
          << "Not mapped: " << ref_loop.at(i)->toString() << ", "
          << tv->getLoopDomain().at(i)->toString() << ", " << tv->toString();
    }
  }

  for (auto tv : {tv1, tv2}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  checkGetAllStmts(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0].as<at::Tensor>()));
}

// A test to check that scheduling loop domains can handle the
// case when there is a 0-d TV which is not an input to the fusion.
// The rest of the fusion here is arbitrary.
TEST_F(LoopDomainSchedulingTest, HandleTVsWithNoLogicalDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  auto tv0 = makeConcreteTensor(shape);
  auto tv1 = makeSymbolicTensor(0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 =
      slice(tv0, {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(99)}});

  auto tv3 = set(tv2);
  auto tv4 = set(tv1);

  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  std::vector<IterDomain*> ref_loop = tv2->getLogicalDomain();
  ASSERT_NO_THROW(
      scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  auto t1 = at::tensor(1.00f, options).squeeze();

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0].as<at::Tensor>()));
}

// Iter domain cannot have multiple definitions, whereas ValGroup can.
// This means that the traversal path in ValGraph may not be a valid
// history to construct within a tensor domain. For example, a path
// with a forward expr followed by a backward expr is invalid since it
// requires the output of the forward expr to have both of the exprs
// as its definitions. This test constructs a fusion where the
// ValGraph shortest path of two tensors results in the invalid
// pattern. The loop domain scheduler should be able to find a
// non-shortest but valid path.
TEST_F(LoopDomainSchedulingTest, ReshapeTraversalDirection) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({12});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);

  auto tv2 = reshape(tv1, {12}, {3, 4});
  auto tv3 = reshape(tv2, {3, 4}, {3, 2, 2});
  auto tv4 = reshape(tv3, {3, 2, 2}, {6, 2});
  auto tv5 = set(tv4);
  auto tv6 = reshape(tv5, {6, 2}, {12});

  auto tv7 = reshape(tv1, {12}, {4, 3});
  auto tv8 = reshape(tv7, {4, 3}, {12});

  auto tv9 = add(tv6, tv8);
  fusion.addOutput(tv9);

  // Consider a case where the loop domain of tv5 is scheduled using
  // tv7 as the reference. From the
  // tv7 loop domain to tv5, the shortest path goes
  // through the logical domain of tv8 (and also tv6 and
  // tv9). However, that path cannot be used since that would result
  // in multi-definition iter domains. Instead,
  // scheduleLoopDomainsLike should use the path
  // from tv7 through tv1: backward split (tv7 reshape) -> forward
  // split (tv2 reshape) -> forward split (tv3 reshape) -> forward
  // merge (tv4 reshape).

  std::vector<IterDomain*> ref = tv7->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike({tv5}, ref);

  ASSERT_EQ(tv5->getLogicalDomain().size(), ref.size())
      << "Unexpected number of dimensions: "
      << toDelimitedString(tv5->getLoopDomain());

  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  for (const auto i : arange(tv5->getLoopDomain().size())) {
    EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv5->getLoopDomain().at(i), ref.at(i)))
        << "Expected exact mapping of loop domains: "
        << tv5->getLoopDomain().at(i)->toString() << ", "
        << ref.at(i)->toString();
  }

  // Validate the history of tv5 loop IDs
  auto tv5_loop_to_logical =
      getExprsBetween<IRBFS>(
          {tv5->getLoopDomain().begin(), tv5->getLoopDomain().end()},
          {tv5->getLogicalDomain().begin(), tv5->getLogicalDomain().end()})
          .first;

  // 1. Backward split (tv7 reshape)
  EXPECT_TRUE(
      exact_graph.disjointExprSets().strictAreMapped(
          tv5_loop_to_logical.at(0).first,
          tv7->getLogicalDomain().at(0)->definition()) &&
      tv5_loop_to_logical.at(0).second == Direction::Backward);
  // 2. Forward split (tv2 reshape)
  EXPECT_TRUE(
      exact_graph.disjointExprSets().strictAreMapped(
          tv5_loop_to_logical.at(1).first,
          tv2->getLogicalDomain().at(0)->definition()) &&
      tv5_loop_to_logical.at(1).second == Direction::Forward);
  // 3. Forward split (tv3 reshape)
  EXPECT_TRUE(
      exact_graph.disjointExprSets().strictAreMapped(
          tv5_loop_to_logical.at(2).first,
          tv3->getLogicalDomain().at(1)->definition()) &&
      tv5_loop_to_logical.at(2).second == Direction::Forward);
  // 4. Forward merge (tv4 reshape).
  EXPECT_TRUE(
      exact_graph.disjointExprSets().strictAreMapped(
          tv5_loop_to_logical.at(3).first,
          tv4->getLogicalDomain().at(0)->definition()) &&
      tv5_loop_to_logical.at(3).second == Direction::Forward);

  checkGetAllStmts(&fusion);
}

// Using the same fusion as ReshapeTraversalDirection, try each one of
// the tensors as a reference
TEST_F(LoopDomainSchedulingTest, ManyReshape) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({12});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);

  auto tv2 = reshape(tv1, {12}, {3, 4});
  auto tv3 = reshape(tv2, {3, 4}, {3, 2, 2});
  auto tv4 = reshape(tv3, {3, 2, 2}, {6, 2});
  auto tv5 = set(tv4);
  auto tv6 = reshape(tv5, {6, 2}, {12});

  auto tv7 = reshape(tv1, {12}, {4, 3});
  auto tv8 = reshape(tv7, {4, 3}, {12});

  auto tv9 = add(tv6, tv8);
  fusion.addOutput(tv9);

  // Try each of the tensors as a reference
  for (const auto i : arange(fusion.allTvs().size())) {
    Fusion fusion_copy = fusion;
    FusionGuard fg_copy(&fusion_copy);

    TensorView* ref_tv = fusion_copy.allTvs().at(i);
    std::vector<IterDomain*> ref_loop = ref_tv->getLogicalDomain();
    scheduler_tools::scheduleLoopDomainsLike(fusion_copy.allTvs(), ref_loop);

    IdModel id_model(&fusion_copy, /*build_models=*/false);
    const auto& exact_graph = id_model.buildExactGraph();

    // The new loop domain of each tensor should be exactly mapped
    // with the reference loop domain
    for (const auto tv : fusion_copy.allTvs()) {
      // scheduleLoopDomainsLike skips fusion inputs
      if (tv->isFusionInput()) {
        continue;
      }
      EXPECT_EQ(tv->getLoopDomain().size(), ref_loop.size())
          << "Invalid rank of loop domain: " << tv->toString();
      for (const auto i : arange(ref_loop.size())) {
        EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
            tv->getLoopDomain().at(i), ref_loop.at(i)))
            << "Expected exact mapping of loop domains: "
            << tv->getLoopDomain().at(i)->toString() << ", "
            << ref_loop.at(i)->toString();
      }
    }

    inlineMost();

    // All tensors, except for the inputs, should be fully inlined
    for (const auto tv : fusion_copy.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }
      EXPECT_EQ(tv->getComputeAtPosition(), tv->getLoopDomain().size());
    }

    checkGetAllStmts(&fusion_copy);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({12}, options);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});

    auto ref = t0 * 2;
    EXPECT_TRUE(ref.equal(cg_outputs[0].as<at::Tensor>()));
  }
}

// Testing scheduleLoopDomainsBy with a trivial fusion
TEST_F(LoopDomainSchedulingTest, ScheduleLoopDomainsBy1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = unaryOp(UnaryOpType::Sin, tv0);
  auto tv2 = unaryOp(UnaryOpType::Cos, tv1);
  auto tv3 =
      slice(tv2, {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(99)}});
  auto tv4 = unaryOp(UnaryOpType::Exp, tv3);

  fusion.addOutput(tv4);

  auto resize = tv3->getLogicalDomain().at(0)->definition()->as<Resize>();

  scheduler_tools::scheduleLoopDomainsBy({tv1, tv2, tv4}, {resize});

  // tv1 and tv2 should have the same loop domain as tv3's loop domain
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  EXPECT_EQ(
      exact_graph.toGroups(tv1->getLoopDomain()),
      exact_graph.toGroups(tv3->getLoopDomain()));
  EXPECT_EQ(
      exact_graph.toGroups(tv2->getLoopDomain()),
      exact_graph.toGroups(tv3->getLoopDomain()));
  // In the case of tv4, its logical ID is mapped with the resize
  // output ID, so the resize op is replayed such that the logical ID
  // is produced by the resize op. The loop domain is then set to the
  // input ID of the resize op, so the tv4 loop domain should be equal
  // to the root domain of tv3.
  auto tv4_resize =
      dynamic_cast<Resize*>(tv4->getLogicalDomain().at(0)->definition());
  EXPECT_NE(tv4_resize, nullptr);
  EXPECT_EQ(
      exact_graph.toGroups(tv4->getLoopDomain()),
      exact_graph.toGroups(tv3->getRootDomain()));

  // Reset the loop domains
  tv1->setLoopDomain(tv1->getLogicalDomain());
  tv2->setLoopDomain(tv2->getLogicalDomain());
  tv4->setLoopDomain(tv4->getLogicalDomain());

  // This time, apply some scheduling to tv1 and tv2 before using
  // scheduleLoopDomainsBy. The resize op should not be replayed as
  // their loop domains no longer match with any of the resize input
  // or output IDs.
  tv1->split(0, 4);
  auto tv1_loop_domain = tv1->getLoopDomain();
  tv2->split(0, 2);
  auto tv2_loop_domain = tv2->getLoopDomain();
  scheduler_tools::scheduleLoopDomainsBy({tv1, tv2}, {resize});

  EXPECT_EQ(tv1->getLoopDomain(), tv1_loop_domain);
  EXPECT_EQ(tv2->getLoopDomain(), tv2_loop_domain);

  checkGetAllStmts(&fusion);
}

// Testing scheduleLoopDomainBy on its insertion position of new IDs
TEST_F(LoopDomainSchedulingTest, ScheduleLoopDomainsBy2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // Merge the outermost and innermost IDs, skipping the middle ID.
  tv1->merge(0, 2);

  auto tv1_merge = dynamic_cast<Merge*>(tv1->axis(0)->definition());

  // Propagating the merge to tv2, which should also insert the merge
  // output at the outer position.
  scheduler_tools::scheduleLoopDomainsBy({tv2}, {tv1_merge});
  auto tv2_merge = dynamic_cast<Merge*>(tv2->axis(0)->definition());
  EXPECT_NE(tv2_merge, nullptr);

  // Both tv1 and tv2 should have the same loop domain
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  EXPECT_EQ(
      exact_graph.toGroups(tv1->getLoopDomain()),
      exact_graph.toGroups(tv2->getLoopDomain()));

  checkGetAllStmts(&fusion);
}

// Make sure existing exprs should not be reused if
// update_loop_domain_only is true
TEST_F(LoopDomainSchedulingTest, UpdateLoopDomainOnlyWithExistingExpr) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {IrBuilder::create<Val>(-1L)});
  fusion.addOutput(tv2);

  auto reshape_merge =
      dynamic_cast<Merge*>(tv2->getLogicalDomain().at(0)->definition());
  ASSERT_NE(reshape_merge, nullptr);

  // Cancel the tv2 reshape
  scheduler_tools::scheduleLoopDomainsLike({tv2}, tv1->getLoopDomain());

  // Schedule tv1
  tv1->flatten();

  // Propagate the tv1 schedule to tv2
  scheduler_tools::scheduleLoopDomainsLike(
      {tv2},
      tv1->getLoopDomain(),
      /*update_loop_domain_only=*/true);

  // The merge of tv1, which is propagated to tv2, is exact mapped
  // with the merge for the tv2 reshape. It should not be reused as
  // the update_loop_domain_only flag is true.
  auto propagated_merge =
      dynamic_cast<Merge*>(tv2->getLoopDomain().at(0)->definition());
  ASSERT_NE(propagated_merge, nullptr);

  EXPECT_NE(reshape_merge, propagated_merge);
}

// Testing propagating with broadcast IDs
TEST_F(LoopDomainSchedulingTest, BroadcastRefereceIDs) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = broadcast(tv2, {false, false, true});
  fusion.addOutput(tv3);

  tv3->flatten();
  tv3->split(0, 32);

  // Using tv3 as a reference. tv3 has two broadcast logical IDs,
  // which are merged and split. By using tv3 as a reference, tv1 and
  // tv2 shoud also be scheduled in the same way. tv1 should get two
  // new broadcast IDs, while tv2 should get one.
  scheduler_tools::scheduleLoopDomainsLike({tv1, tv2}, tv3->getLoopDomain());

  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  for (auto tv : {tv1, tv2}) {
    // There must be broadcast IDs that are exact mapped with the two
    // logical broadcast IDs of tv3
    const auto all_ids = tv->domain()->allIDs();
    for (auto ref_logical_broadcast : tv3->getLogicalDomain()) {
      if (!ref_logical_broadcast->isBroadcast()) {
        continue;
      }
      auto it =
          std::find_if(all_ids.begin(), all_ids.end(), [&](IterDomain* id) {
            return id->isBroadcast() &&
                exact_graph.disjointValSets().strictAreMapped(
                    id, ref_logical_broadcast);
          });
      EXPECT_NE(it, all_ids.end())
          << "No matching broadcast ID found in " << tv->toString()
          << ". Missing ref broadcast: " << ref_logical_broadcast->toString();
    }

    // The loop domain should be exact mapped with tv3
    ASSERT_EQ(tv->getLoopDomain().size(), tv3->getLoopDomain().size());
    for (const auto i : arange(tv->getLoopDomain().size())) {
      auto tv_loop_id = tv->getLoopDomain().at(i);
      auto ref_loop_id = tv3->getLoopDomain().at(i);
      EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
          tv_loop_id, ref_loop_id));
    }
  }
}

// Cancelling a reshape to make all tensors ordered as the input
TEST_F(LoopDomainSchedulingTest, CancelReshape1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{16, 32, 2};

  auto tv0 = makeContigConcreteTensor(shape); // [i0, i1, i2]
  fusion.addInput(tv0);
  auto tv1 = permute(tv0, {1, 0, 2}); // [i1, i0, i2]
  auto tv2 =
      reshape(tv1, shape, {shape[1], shape[0] * shape[2]}); // [i1, i0*i2]
  auto tv3 = sin(tv2);
  fusion.addOutput(tv3);

  // Cancel the reshape of tv2
  scheduler_tools::cancelReshapeInLoopDomains(tv0);

  // The loop domain of tv2 should now be the same as its root domain.
  EXPECT_EQ(tv2->getRootDomain(), tv2->getLoopDomain());
  // The loop domain of tv3 should be exact mapped with the tv2 loop
  // domain
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    EXPECT_EQ(
        exact_graph.toGroups(tv3->getLoopDomain()),
        exact_graph.toGroups(tv2->getLoopDomain()));
  }

  // Reorder tv3 as the input
  tv3->reorder({1, 0, 2});
  tv3->flatten();
  tv3->split(0, 128);
  scheduler_tools::scheduleLoopDomainsLike({tv1, tv2}, tv3->getLoopDomain());

  // All loop domains should be exact mapped
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    const auto ref_loop = exact_graph.toGroups(tv3->getLoopDomain());
    for (auto tv : {tv1, tv2}) {
      EXPECT_EQ(exact_graph.toGroups(tv->getLoopDomain()), ref_loop);
    }
  }

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Cancelling chained reshape ops
TEST_F(LoopDomainSchedulingTest, CancelReshape2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{10, 11, 12};

  auto tv0 = makeContigConcreteTensor(shape); // [i0, i1, i2]
  fusion.addInput(tv0);
  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Val>(shape[1]),
       IrBuilder::create<Val>(shape[0] * shape[2])});
  auto tv2 = reshape(
      tv1,
      {IrBuilder::create<Val>(shape[1]),
       IrBuilder::create<Val>(shape[2]),
       IrBuilder::create<Val>(shape[0])});
  auto tv3 = reshape(
      tv2,
      {IrBuilder::create<Val>(shape[0] * shape[1]),
       IrBuilder::create<Val>(shape[2])});
  fusion.addOutput(tv3);

  // Cancel all reshape ops
  scheduler_tools::cancelReshapeInLoopDomains(tv0);

  // All of the tensors should have the same loop domain
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    const auto ref_loop = exact_graph.toGroups(tv0->getLoopDomain());
    for (auto tv : {tv1, tv2, tv3}) {
      EXPECT_EQ(exact_graph.toGroups(tv->getLoopDomain()), ref_loop);
    }
  }

  tv3->flatten();
  tv3->split(0, 32);

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Two reshapes that get merged by a binary op
TEST_F(LoopDomainSchedulingTest, CancelReshape3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{10, 11};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {IrBuilder::create<Val>(-1L)});
  auto tv2 = reshape(tv0, {IrBuilder::create<Val>(-1L)});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  // The cancellation of the second reshape won't do anything as the
  // loop domain is already updated by the first reshape.
  scheduler_tools::cancelReshapeInLoopDomains(tv0);

  // All of the tensors should have the same loop domain
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    const auto ref_loop = exact_graph.toGroups(tv0->getLoopDomain());
    for (auto tv : {tv1, tv2, tv3}) {
      EXPECT_EQ(exact_graph.toGroups(tv->getLoopDomain()), ref_loop);
    }
  }

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Resize should prevent cancellation
TEST_F(LoopDomainSchedulingTest, CancelReshape4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{10, 11, 12};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  // Non-cancellable reshape due to the following slice
  auto tv1 = reshape(
      tv0, {IrBuilder::create<Val>(shape[0]), IrBuilder::create<Val>(-1L)});
  auto tv2 = slice(
      tv1,
      {{fusion.zeroVal(), tv1->axis(0)->extent()},
       {fusion.oneVal(), tv1->axis(1)->extent()}});
  fusion.addOutput(tv2);

  // Cancellable reshape
  auto tv3 = reshape(
      tv0,
      {IrBuilder::create<Val>(shape[0] * shape[1]),
       IrBuilder::create<Val>(-1L)});
  auto tv4 = slice(
      tv3,
      {{fusion.zeroVal(), tv3->axis(0)->extent()},
       {fusion.oneVal(), tv3->axis(1)->extent()}});
  fusion.addOutput(tv4);

  const auto tv1_original_loop = tv1->getLoopDomain();
  const auto tv2_original_loop = tv2->getLoopDomain();

  // tv1 and tv2 should not be modified as the slice depends on the reshaped
  // domain
  scheduler_tools::cancelReshapeInLoopDomains(tv0);

  EXPECT_EQ(tv1->getLoopDomain(), tv1_original_loop);
  EXPECT_EQ(tv2->getLoopDomain(), tv2_original_loop);

  // The tv3 reshape should be cancelled as the slice does not
  // depend on the reshape expr
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    ValGroups ref_loop;
    for (const auto i : arange(2)) {
      ref_loop.pushBack(exact_graph.toGroup(tv0->getLoopDomain().at(i)));
    }
    // The first two loop IDs should be exact mapped with tv0
    for (auto tv : {tv3, tv4}) {
      ASSERT_EQ(tv->getLoopDomain().size(), 3);
      ValGroups tv_loop_groups;
      for (const auto i : arange(2)) {
        tv_loop_groups.pushBack(exact_graph.toGroup(tv->getLoopDomain().at(i)));
      }
      EXPECT_EQ(tv_loop_groups, ref_loop);
    }
  }
}

// Reduction should prevent cancellation
TEST_F(LoopDomainSchedulingTest, CancelReshape5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{10, 11, 12};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  // Non-cancellable reshape due to the following reduction
  auto tv1 = reshape(
      tv0, {IrBuilder::create<Val>(shape[0]), IrBuilder::create<Val>(-1L)});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  // Cancellable reshape
  auto tv3 = reshape(
      tv0,
      {IrBuilder::create<Val>(shape[0] * shape[1]),
       IrBuilder::create<Val>(-1L)});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  const auto tv1_original_loop = tv1->getLoopDomain();
  const auto tv2_original_loop = tv2->getLoopDomain();

  // tv1 and tv2 should not be modified as the tv2 reduction depends on the
  // reshaped domain
  scheduler_tools::cancelReshapeInLoopDomains(tv0);

  EXPECT_EQ(tv1->getLoopDomain(), tv1_original_loop);
  EXPECT_EQ(tv2->getLoopDomain(), tv2_original_loop);

  // The tv3 reshape should be cancelled as the reduction does not
  // depend on the reshape expr
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    const auto ref_loop = exact_graph.toGroups(tv0->getLoopDomain());
    for (auto tv : {tv3, tv4}) {
      EXPECT_EQ(exact_graph.toGroups(tv->getLoopDomain()), ref_loop);
    }
  }
}

// Repro for a vectorization validation bug. It used to do a traversal
// from an allocation domain to its vectorized ID, which may not work
// when a broadcast ID is included in the allocation domain and the
// loop domain is generated by using a concrete ID in place of the
// broadcast ID. That is not how fusions are normally scheduled but
// possible with scheduleLoopDomainsLike.
TEST_F(LoopDomainSchedulingTest, VecValidationRepro) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, 32});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({4, 32});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);

  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->flatten();
  tv4->split(0, 4);

  scheduler_tools::scheduleLoopDomainsLike({tv2, tv3}, tv4->getLoopDomain());

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  // Here, tv2 is scheduled as:
  //
  // T2_l_float[iS13{32}, iS14{4}]
  //  logical domain : (bS4{1}, iS5{32})
  //  contiguity: n t
  //   Merge: iS16{4} and iS5{32} -> iS15{128}
  //   Split: iS15{128} by factor 4 -> iS13{32}, iS14{4}
  //  loop domain : (iS13{32}, iS14{4})
  //
  // Notice that iS16 is used instead of bS4. Thus, traversing from
  // the allocation domain, which is just the logical domain in this
  // case, is not able to reach anything. In
  // https://github.com/NVIDIA/Fuser/pull/3723, the traversal of the
  // vectorization analysis is changed so that it starts from the
  // promoted loop domain to the allocation domain. In this case, the
  // starting nodes are just iS13 and iS14 since there's no
  // inlining, the traversal successfully reaches the innermost ID of
  // the allocation domain, iS5.

  // As long as the lowering completes with no error, the
  // vectorization validation should be fine.
  GpuLower lower(&fusion);
}

} // namespace nvfuser
