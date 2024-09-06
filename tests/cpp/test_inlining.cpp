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

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <abstract_tensor.h>
#include <fusion.h>
#include <inlining.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>

namespace nvfuser {

using InliningTest = NVFuserTest;

TEST_F(InliningTest, IsAllowedID) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1({3, 4, 5});
  std::vector<int64_t> shape2({3, 4 * 5});

  // Inner normalization pattern

  // [i0, i1, i2]
  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);

  // [i0, i1*i2]
  auto tv1 = reshape(tv0, shape1, shape2);
  // [i0, r1*r2]
  auto tv2 = sum(tv1, {1});
  // [i0]
  auto tv3 = add(tv2, fusion.oneVal());
  // [i0, b3]
  auto tv4 = broadcast(tv3, {false, true});
  // [i0, i1*i2]
  auto tv5 = sub(tv1, tv4);
  // [i0, i1, i2]
  auto tv6 = reshape(tv5, shape2, shape1);
  fusion.addOutput(tv6);

  fusion.printMath();

  MaxPosCalculator calc;

  // First, check isAllowedID without manipulating loop domains
  EXPECT_TRUE(calc.isAllowedID(
      tv1->getLoopDomain().at(0),
      tv1,
      /*best_effort=*/true,
      false,
      false,
      false));

  EXPECT_FALSE(calc.isAllowedID(
      tv1->getLoopDomain().at(1),
      tv1,
      /*best_effort=*/true,
      false,
      false,
      false));

  // Set [i0, i1, i2] as the loop domain of each of tensors

  {
    tv1->setLoopDomain(tv1->getRootDomain());
    for (const auto i: c10::irange(1)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv1->getLoopDomain().at(i),
          tv1,
          /*best_effort=*/true,
          false,
          false,
          false));
    }

    std::cerr << "Checking unmappable domain\n";
    for (const auto i: c10::irange(1, 3)) {
      EXPECT_FALSE(calc.isAllowedID(
          tv1->getLoopDomain().at(i),
          tv1,
          /*best_effort=*/true,
          false,
          false,
          false)) << i;
    }
  }

  // tv2
  {
    std::vector<IterDomain*> loop_domain {
      tv2->getLogicalDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
      tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv2->getLogicalDomain().at(1), loop_domain[1], loop_domain[2]);
    tv2->setLoopDomain(loop_domain);

    for (const auto i: c10::irange(1)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv2->getLoopDomain().at(i),
          tv2,
          /*best_effort=*/true,
          false,
          false,
          false)) << i << ", " << tv2->getLoopDomain().at(i)->toString();
    }

    for (const auto i: c10::irange(1, 3)) {
      EXPECT_FALSE(calc.isAllowedID(
          tv2->getLoopDomain().at(i),
          tv2,
          /*best_effort=*/true,
          false,
          false,
          false));
    }
  }

  // tv3
  {
    std::vector<IterDomain*> loop_domain{
      tv1->getLoopDomain().at(0)->cloneWithoutRFactor(),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
      tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv3->getLogicalDomain().at(0), loop_domain[1], loop_domain[2]);
    tv3->setLoopDomain(loop_domain);

    for (const auto i: c10::irange(3)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv3->getLoopDomain().at(i),
          tv3,
          /*best_effort=*/true,
          false,
          false,
          false)) << i << ", " << tv3->getLoopDomain().at(i)->toString();
    }
  }


  // tv4
  {
    std::vector<IterDomain*> loop_domain{
        tv1->getLoopDomain().at(0)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv4->getLogicalDomain().at(0), loop_domain[1], loop_domain[2]);
    // Not completely connected
    tv4->setLoopDomain(loop_domain);
    for (const auto i: c10::irange(3)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv4->getLoopDomain().at(i),
          tv4,
          /*best_effort=*/true,
          false,
          false,
          false)) << i << ", " << tv4->getLoopDomain().at(i)->toString();
    }
  }

  // tv5
  {
    std::vector<IterDomain*> loop_domain {
      tv5->getLogicalDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
      tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv5->getLogicalDomain().at(1), loop_domain[1], loop_domain[2]);
    tv5->setLoopDomain(loop_domain);
    for (const auto i: c10::irange(3)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv5->getLoopDomain().at(i),
          tv5,
          /*best_effort=*/true,
          false,
          false,
          false)) << i << ", " << tv5->getLoopDomain().at(i)->toString();
    }
  }

  // tv6
  {
    std::vector<IterDomain*> loop_domain {
      tv6->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv6->getRootDomain().at(1), loop_domain[1], loop_domain[2]);
    tv6->setLoopDomain(loop_domain);
    for (const auto i: c10::irange(3)) {
      EXPECT_TRUE(calc.isAllowedID(
          tv6->getLoopDomain().at(i),
          tv6,
          /*best_effort=*/true,
          false,
          false,
          false)) << i << ", " << tv6->getLoopDomain().at(i)->toString();
    }
  }
}

TEST_F(InliningTest, GetMaxProducerPosFromConsumer) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1({3, 4 * 5});
  std::vector<int64_t> shape2({3, 4, 5});

  // [i0, i1*i2]
  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);

  // [i0, i1, i2]
  auto tv1 = reshape(tv0, shape1, shape2);
  // [i0, i1, i2]  
  auto tv2 = add(tv1, fusion.oneVal());
  fusion.addOutput(tv2);

  std::cout << "Changing loop domains\n";

  auto reshape_split_factor = tv1->axis(1)->definition()->as<Split>()->factor();

  // Set [i0, i1*i2] as the loop domain of each of tensors

  {
    tv1->setLoopDomain(tv1->getRootDomain());
  }

  // tv2
  {
    std::vector<IterDomain*> loop_domain {
      tv2->getLogicalDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv2->getLogicalDomain().at(1),
        tv2->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor, false);
    tv2->setLoopDomain(loop_domain);
  }

  tv1->split(-1, 4);
  tv2->split(-1, 4);

  MaxPosCalculator calc;

  EXPECT_EQ(calc.getMaxProducerPosFromConsumer(tv1, tv2, true), 3);
  
  tv2->axis(-1)->parallelize(ParallelType::Unroll);

  EXPECT_EQ(calc.getMaxProducerPosFromConsumer(tv1, tv2, true), 2);
}

TEST_F(InliningTest, GetMaxPosAllNormalization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1({3, 4 * 5});
  std::vector<int64_t> shape2({3, 4, 5});

  // Inner normalization pattern

  // [i0, i1*i2]
  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);

  // [i0, i1, i2]
  auto tv1 = reshape(tv0, shape1, shape2);
  // [i0, r1, r2]
  auto tv2 = sum(tv1, {1, 2});
  // [i0]
  auto tv3 = add(tv2, fusion.oneVal());
  // [i0, b3, b4]
  auto tv4 = broadcast(tv3, {false, true, true});
  // [i0, i1, i2]
  auto tv5 = sub(tv1, tv4);
  // [i0, i1*i2]
  auto tv6 = reshape(tv5, shape2, shape1);
  fusion.addOutput(tv6);

  fusion.printMath();

  {
    MaxPosCalculator calc;
    // First, check isAllowedID without manipulating loop domains

    std::cerr << calc.getMaxPosAll(tv1, true) << "\n";
  }

  std::cout << "Changing loop domains\n";

  auto reshape_split_factor = tv1->axis(1)->definition()->as<Split>()->factor();

  // Set [i0, i1*i2] as the loop domain of each of tensors

  {
    tv1->setLoopDomain(tv1->getRootDomain());
  }

  // tv2
  {
    std::vector<IterDomain*> loop_domain {
      tv2->getLogicalDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv2->getLogicalDomain().at(1),
        tv2->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor, false);
    tv2->setLoopDomain(loop_domain);
  }

  // tv3
  {
    std::vector<IterDomain*> loop_domain {
      tv3->getLogicalDomain().at(0),      
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    tv3->setLoopDomain(loop_domain);
  }

  // tv4
  {
    std::vector<IterDomain*> loop_domain {
      tv4->getLogicalDomain().at(0),      
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    tv4->setLoopDomain(loop_domain);
  }

  // tv5
  {
    std::vector<IterDomain*> loop_domain {
      tv5->getLogicalDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv5->getLogicalDomain().at(1),
        tv5->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor, false);
    tv5->setLoopDomain(loop_domain);
  }

  // tv6
  {
    std::vector<IterDomain*> loop_domain {
      tv6->getRootDomain().at(0),
      tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv6->getRootDomain().at(1),
        tv6->getRootDomain().at(2),
        loop_domain[1],
        reshape_split_factor, false);
    tv6->setLoopDomain(loop_domain);
  }

  {
    MaxPosCalculator calc;
    // First, check isAllowedID without manipulating loop domains

    EXPECT_EQ(calc.getMaxPosAll(tv1, true), 1);
  }
}

} // namespace nvfuser
