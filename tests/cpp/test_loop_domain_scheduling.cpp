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

#include <inlining.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
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
  scheduler_utils::scheduleLoopDomainsLike(fusion.allTvs(), ref);

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

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
  scheduler_utils::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

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

  // Consider a case where the loop domain of tv4 is scheduled using
  // tv7 as the reference. From the
  // tv7 loop domain to tv4, the shortest path goes
  // through the logical domain of tv8 (and also tv6 and
  // tv9). However, that path cannot be used sicne that would result
  // in multi-definition iter domains. Instead,
  // scheduleLoopDomainsLike should use the path
  // from tv7 through tv1: backward split (tv7 reshape) -> forward
  // split (tv2 reshape) -> forward split (tv3 reshape) -> forward
  // merge (tv4 reshape).

  std::vector<IterDomain*> ref = tv7->getLogicalDomain();
  scheduler_utils::scheduleLoopDomainsLike({tv5}, ref);

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
  auto tv5_loop_to_logical = IRBFS::getExprsBetween(
      {tv5->getLoopDomain().begin(), tv5->getLoopDomain().end()},
      {tv5->getLogicalDomain().begin(), tv5->getLogicalDomain().end()});

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

} // namespace nvfuser
