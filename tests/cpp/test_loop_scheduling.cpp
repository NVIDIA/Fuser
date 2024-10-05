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

class LoopSchedulingTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }

 private:
  EnableOptionsGuard enable_options_guard_;
};

TEST_F(LoopSchedulingTest, ReshapeSplitThenMerge) {
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

TEST_F(LoopSchedulingTest, ReshapeWithBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // One split reshape and then merge reshape. Schedule the loop
  // domains of all tensors with the initial pre-reshape domain.

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  fusion.print();

  std::vector<IterDomain*> ref = tv4->getLogicalDomain();
  scheduler_utils::scheduleLoopDomainsLike(fusion.allTvs(), ref);

  for (auto tv : fusion.allTvs()) {
    tv->flatten();
    tv->split(0, 32);
  }

  inlineMost();

  tv2->domain()->validateLoopDomain(tv2->getLoopDomain());

  IdModel id_model(&fusion);

#if 0
  std::ofstream ofs("exact_graph.dot", std::ofstream::trunc);
  auto dot_string = id_model.idGraph(IdMappingMode::EXACT).toGraphvizDotGraph();
  ofs << dot_string;
  ofs.close();
#endif

  ref = tv4->getLoopDomain();
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
  auto t0 = at::randn({10, 11}, options);
  auto t1 = at::randn({10}, options);
  std::vector<c10::IValue> inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

TEST_F(LoopSchedulingTest, Slice) {
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

} // namespace nvfuser
