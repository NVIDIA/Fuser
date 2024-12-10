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
#include <scheduler/tools/loop_domain_scheduler.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fstream>
#include <iostream>

namespace nvfuser {

class LoopDomainSchedulingTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }

 private:
  EnableOptionsGuard enable_options_guard_;
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

  IdModel id_model(&fusion);

  ref = tv1->getLoopDomain();
  for (auto tv : fusion.allTvs()) {
    EXPECT_EQ(ref.size(), tv->getLoopDomain().size());

    if (!tv->isFusionInput()) {
      EXPECT_EQ(tv->getComputeAtPosition(), 2) << tv->toString();
    }

    for (const auto i : c10::irange(ref.size())) {
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10}, options);
  std::vector<c10::IValue> inputs({t0});

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  auto outputs = ke.run(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
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
  IdModel id_model(&fusion);

  ref_loop = tv0->getLoopDomain();

  for (auto tv : fusion.allTvs()) {
    EXPECT_EQ(ref_loop.size(), tv->getLoopDomain().size());
    for (const auto i : c10::irange(ref_loop.size())) {
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  KernelExecutor ke;
  ke.compile(&fusion, aten_inputs);
  auto cg_outputs = ke.run(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0]));
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

  for (const auto i : c10::irange(tv5->getLoopDomain().size())) {
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
  for (const auto i : c10::irange(fusion.allTvs().size())) {
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
      for (const auto i : c10::irange(ref_loop.size())) {
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

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({12}, options);
    std::vector<c10::IValue> aten_inputs({t0});

    KernelExecutor ke;
    ke.compile(&fusion, aten_inputs);
    auto cg_outputs = ke.run(aten_inputs);

    auto ref = t0 * 2;
    EXPECT_TRUE(ref.equal(cg_outputs[0]));
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

  scheduler_tools::scheduleLoopDomainsBy({tv1, tv2, tv4}, resize);

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
  scheduler_tools::scheduleLoopDomainsBy({tv1, tv2}, resize);

  EXPECT_EQ(tv1->getLoopDomain(), tv1_loop_domain);
  EXPECT_EQ(tv2->getLoopDomain(), tv2_loop_domain);
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
  scheduler_tools::scheduleLoopDomainsBy({tv2}, tv1_merge);
  auto tv2_merge = dynamic_cast<Merge*>(tv2->axis(0)->definition());
  EXPECT_NE(tv2_merge, nullptr);

  // Both tv1 and tv2 should have the same loop domain
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  EXPECT_EQ(
      exact_graph.toGroups(tv1->getLoopDomain()),
      exact_graph.toGroups(tv2->getLoopDomain()));
}

} // namespace nvfuser
