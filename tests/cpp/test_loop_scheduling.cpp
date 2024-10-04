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

namespace nvfuser {

class LoopSchedulingTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }

 private:
  EnableOptionsGuard enable_options_guard_;
};

TEST_F(LoopSchedulingTest, Test1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {10}, {2, 5});
  auto tv3 = set(tv2);
  auto tv4 = reshape(tv3, {2, 5}, {10});
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  fusion.printMath();

  std::vector<IterDomain*> ref = tv0->getLogicalDomain();
  scheduler_utils::scheduleLoopDomainsLike(fusion.allTvs(), ref);

  for (auto tv : fusion.allTvs()) {
    tv->split(0, 3);
  }

  inlineMost();
  fusion.print();

  IdModel id_model(&fusion);

  ref = tv1->getLoopDomain();
  for (auto tv : fusion.allTvs()) {
    EXPECT_EQ(ref.size(), tv->getLoopDomain().size());
    for (const auto i : c10::irange(ref.size())) {
      EXPECT_TRUE(id_model.idGraph(IdMappingMode::EXACT)
                      .disjointValSets()
                      .strictAreMapped(ref.at(i), tv->getLoopDomain().at(i)))
          << "Not mapped: " << ref.at(i)->toString() << ", "
          << tv->getLoopDomain().at(i)->toString() << ", " << tv->toString();
      if (!tv->isFusionInput()) {
        EXPECT_TRUE(id_model.idGraph(IdMappingMode::LOOP)
                        .disjointValSets()
                        .strictAreMapped(ref.at(i), tv->getLoopDomain().at(i)))
            << "Not mapped: " << ref.at(i)->toString() << ", "
            << tv->getLoopDomain().at(i)->toString() << ", " << tv->toString();
      }
    }
  }

  std::cerr << id_model.idGraph(IdMappingMode::EXACT).toString();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10}, options);
  std::vector<c10::IValue> inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto outputs = fe.runFusion(inputs);

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

TEST_F(LoopSchedulingTest, TMP1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {10}, {2, 5});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  fusion.printMath();

  // std::vector<IterDomain*> ref = tv0->getLogicalDomain();
  // scheduler_utils::scheduleLoopDomainsLike({tv1, tv2, tv3}, ref);

  // for (auto tv : {tv1, tv2, tv3}) {
  // tv->split(0, 3);
  //}

  // fusion.print();
  // fusion.printKernel();

  tv2->setLoopDomain(tv2->getRootDomain());

  auto clone = tv2->getLoopDomain().at(0)->cloneWithoutRFactor();
  auto tv2_split = tv2->getLogicalDomain().at(0)->definition()->as<Split>();
  auto tv3_split = IrBuilder::create<Split>(
      tv3->getLogicalDomain().at(0),
      tv3->getLogicalDomain().at(1),
      clone,
      tv2_split->factor(),
      tv2_split->innerSplit());
  tv3->setLoopDomain({clone});

  std::cerr << "Original: " << tv2_split->toString();
  std::cerr << "Replayed: " << tv3_split->toString();

  inlineMost();
  fusion.print();

  IdModel id_model(&fusion);

  std::cerr << id_model.idGraph(IdMappingMode::EXACT).toString();

  std::cerr << "LOOP\n" << id_model.idGraph(IdMappingMode::LOOP).toString();

  fusion.printKernel();
}

} // namespace nvfuser
