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

#include <fusion.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

namespace nvfuser {

using InliningTest = NVFuserTest;

TEST_F(InliningTest, InliningMismatchedDims1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineMost();

  EXPECT_EQ(tv5->getComputeAtPosition(), 3);
  EXPECT_EQ(tv4->getComputeAtPosition(), 3);
  EXPECT_EQ(tv3->getComputeAtPosition(), 3);
  EXPECT_EQ(tv2->getComputeAtPosition(), 1);
  EXPECT_EQ(tv1->getComputeAtPosition(), 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  testValidate(&fusion, cg_outputs, {input}, __LINE__, __FILE__);
}

TEST_F(InliningTest, InliningMismatchedDims2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineAllAt(tv5, -1, true);

  EXPECT_EQ(tv5->getComputeAtPosition(), 3);
  EXPECT_EQ(tv4->getComputeAtPosition(), 3);
  EXPECT_EQ(tv3->getComputeAtPosition(), 3);
  EXPECT_EQ(tv2->getComputeAtPosition(), 1);
  EXPECT_EQ(tv1->getComputeAtPosition(), 1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  testValidate(&fusion, cg_outputs, {input}, __LINE__, __FILE__);
}

TEST_F(InliningTest, InliningMismatchedDims4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = exp(tv1);
  auto tv3 = relu(tv2);
  auto tv4 = cos(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  tv3->merge(1);
  inlineMost();

  EXPECT_EQ(tv5->getComputeAtPosition(), 3);
  EXPECT_EQ(tv4->getComputeAtPosition(), 3);
  EXPECT_EQ(tv3->getComputeAtPosition(), 1);
  EXPECT_EQ(tv2->getComputeAtPosition(), 1);
  EXPECT_EQ(tv1->getComputeAtPosition(), 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  testValidate(&fusion, cg_outputs, {input}, __LINE__, __FILE__);
}

TEST_F(InliningTest, InliningBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  // broadcasting
  auto tv2 = broadcast(tv1, {false, true, false, true, false, true});
  auto tv3 = cos(tv2);
  auto tv4 = tan(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  EXPECT_EQ(tv4->getComputeAtPosition(), 3);
  EXPECT_EQ(tv3->getComputeAtPosition(), 3);
  EXPECT_EQ(tv2->getComputeAtPosition(), 3);
  EXPECT_EQ(tv1->getComputeAtPosition(), 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  testValidate(&fusion, cg_outputs, {input}, __LINE__, __FILE__);
}

TEST_F(InliningTest, MatchedLeafPosWithoutReplayBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv2 = sin(tv1);
  fusion.addOutput(tv2);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  EXPECT_EQ(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv0, tv1, 3), 3);
  EXPECT_EQ(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv1, tv0, 3), 3);
  EXPECT_EQ(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv1, tv2, 3), 3);
  EXPECT_EQ(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv2, tv1, 3), 3);
}

// Test isAllowedID with setLoopDomain
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

  MaxPosCalculator calc;

  auto isAllowedID = [&calc](TensorView* tv, int64_t axis) -> bool {
    return calc.isAllowedID(
        tv->getLoopDomain().at(axis),
        tv,
        /*best_effort=*/true,
        /*allow_reduction=*/false,
        /*allow_vectorize=*/false,
        /*allow_unmappable=*/false);
  };

  // First, check isAllowedID without manipulating loop domains. tv1
  // is a persistent tensor. The inner domain needs to be persistent.
  EXPECT_TRUE(isAllowedID(tv1, 0)) << tv1->getLoopDomain().at(0)->toString();
  // The inner domain should not be allowed since it's a persistent domain
  EXPECT_FALSE(isAllowedID(tv1, 1)) << tv1->getLoopDomain().at(1)->toString();

  // Set [i0, i1, i2] as the loop domain of each of tensors. i1 and i2
  // become pesistent domains.

  // tv1
  {
    tv1->setLoopDomain(tv1->getRootDomain());
    EXPECT_TRUE(isAllowedID(tv1, 0)) << tv1->getLoopDomain().at(0)->toString();
    // Persistent domain
    EXPECT_FALSE(isAllowedID(tv1, 1)) << tv1->getLoopDomain().at(1)->toString();
    // Persistent domain
    EXPECT_FALSE(isAllowedID(tv1, 2)) << tv1->getLoopDomain().at(2)->toString();
  }

  // tv2
  {
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        IterDomainBuilder(tv1->getLoopDomain().at(1))
            .resetRfactor()
            .iter_type(IterType::Reduction)
            .build(),
        IterDomainBuilder(tv1->getLoopDomain().at(2))
            .resetRfactor()
            .iter_type(IterType::Reduction)
            .build()};
    IrBuilder::create<Merge>(
        tv2->getLogicalDomain().at(1), loop_domain[1], loop_domain[2]);
    tv2->setLoopDomain(loop_domain);

    EXPECT_TRUE(isAllowedID(tv2, 0)) << tv2->getLoopDomain().at(0)->toString();
    // Reduction domain
    EXPECT_FALSE(isAllowedID(tv2, 1)) << tv2->getLoopDomain().at(1)->toString();
    // Reduction domain
    EXPECT_FALSE(isAllowedID(tv2, 2)) << tv2->getLoopDomain().at(2)->toString();
  }

  // tv3
  {
    std::vector<IterDomain*> loop_domain{
        tv3->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    tv3->setLoopDomain(loop_domain);
    for (const auto i : arange(3)) {
      EXPECT_TRUE(isAllowedID(tv3, i))
          << tv3->getLoopDomain().at(i)->toString();
    }
  }

  // tv4
  {
    std::vector<IterDomain*> loop_domain{
        tv4->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    // Note that loop_domain[1] and loop_domain[2] are not connected
    // with the logical domain of tv4
    tv4->setLoopDomain(loop_domain);
    for (const auto i : arange(3)) {
      EXPECT_TRUE(isAllowedID(tv4, i))
          << tv4->getLoopDomain().at(i)->toString();
    }
  }

  // tv5
  {
    std::vector<IterDomain*> loop_domain{
        tv5->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv5->getLogicalDomain().at(1), loop_domain[1], loop_domain[2]);
    tv5->setLoopDomain(loop_domain);
    for (const auto i : arange(3)) {
      EXPECT_TRUE(isAllowedID(tv5, i))
          << tv5->getLoopDomain().at(i)->toString();
    }
  }

  // tv6
  {
    std::vector<IterDomain*> loop_domain{
        tv6->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor(),
        tv1->getLoopDomain().at(2)->cloneWithoutRFactor()};
    IrBuilder::create<Merge>(
        tv6->getRootDomain().at(1), loop_domain[1], loop_domain[2]);
    tv6->setLoopDomain(loop_domain);
    for (const auto i : arange(3)) {
      EXPECT_TRUE(isAllowedID(tv6, i))
          << tv6->getLoopDomain().at(i)->toString();
    }
  }
}

// Test GetMaxProducerPosFromConsumer with setLoopDomain
TEST_F(InliningTest, GetMaxProducerPosFromConsumerWithReshape) {
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

  auto reshape_split_factor = tv1->axis(1)->definition()->as<Split>()->factor();

  // Set [i0, i1*i2] as the loop domain of each of tensors

  tv1->setLoopDomain(tv1->getRootDomain());

  // tv2
  {
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv2->getLogicalDomain().at(1),
        tv2->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv2->setLoopDomain(loop_domain);
  }

  for (auto tv : {tv1, tv2}) {
    tv->split(-1, 4);
  }

  MaxPosCalculator calc;

  // Nothing should prevent tv1 from fully inlined into tv2
  EXPECT_EQ(
      calc.getMaxProducerPosFromConsumer(tv1, tv2, /*best_effort=*/true), 3);

  // Unrolling of tv2 should block the inlining of tv1.
  tv2->axis(-1)->parallelize(ParallelType::Unroll);
  EXPECT_EQ(
      calc.getMaxProducerPosFromConsumer(tv1, tv2, /*best_effort=*/true), 2);
}

TEST_F(InliningTest, GetMaxProducerPosFromConsumerWithBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Use the loop domain of tv4 as the reference.

  // tv2
  {
    // One domain is missing. Can use either a clone of the tv3 inner
    // broadcast domain or the tv4 inner non-broadcast domain.
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        tv3->getLoopDomain().at(1)->cloneWithoutRFactor(
            /*map_with_original=*/true)};
    tv2->setLoopDomain(loop_domain);
  }

  for (auto tv : fusion.allTvs()) {
    tv->flatten();
    tv->split(0, 32);
  }

  // tv3 loop domain does not need to change. Although it has a
  // broadcast iter domain, it is treated as the same as the concrete
  // iter domain of tv4 in the BROADCAST graph

  // Both of the inlining positions of tv2 and tv3 should be the
  // innermost.
  MaxPosCalculator calc;
  EXPECT_EQ(
      calc.getMaxProducerPosFromConsumer(tv2, tv3, /*best_effort=*/true), 2);
  EXPECT_EQ(
      calc.getMaxProducerPosFromConsumer(tv3, tv4, /*best_effort=*/true), 2);
}

// Test MaxPosCalculator.getMaxPosAll with setLoopDomain
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

  // tv1 is the persistent tensor. Its inner two domains are the
  // persistent domains. The outermost domain is the only inlinable
  // domain.

  // First, check getMaxPosAll without manipulating loop domains
  EXPECT_EQ(MaxPosCalculator().getMaxPosAll(tv1, /*best_effort=*/true), 1);

  // Set [i0, i1*i2] as the loop domain of each of tensors

  auto reshape_split_factor = tv1->axis(1)->definition()->as<Split>()->factor();

  tv1->setLoopDomain(tv1->getRootDomain());

  // tv2
  {
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        IterDomainBuilder(tv1->getLoopDomain().at(1))
            .resetRfactor()
            .iter_type(IterType::Reduction)
            .build()};
    IrBuilder::create<Split>(
        tv2->getLogicalDomain().at(1),
        tv2->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv2->setLoopDomain(loop_domain);
  }

  // Set the loop domain of tv4 before tv3. Use a clone of tv4 in
  // tv3.

  // tv4
  {
    std::vector<IterDomain*> loop_domain{
        tv4->getLogicalDomain().at(0),
        IterDomainBuilder(fusion.zeroVal(), fusion.oneVal())
            .iter_type(IterType::Broadcast)
            .build()};
    IrBuilder::create<Split>(
        tv4->getLogicalDomain().at(1),
        tv4->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv4->setLoopDomain(loop_domain);
  }

  // tv3
  {
    std::vector<IterDomain*> loop_domain{
        tv3->getLogicalDomain().at(0),
        tv4->getLoopDomain().at(1)->cloneWithoutRFactor(
            /*map_with_original=*/true)};
    tv3->setLoopDomain(loop_domain);
  }

  // tv5
  {
    std::vector<IterDomain*> loop_domain{
        tv5->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv5->getLogicalDomain().at(1),
        tv5->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv5->setLoopDomain(loop_domain);
  }

  // tv6
  {
    std::vector<IterDomain*> loop_domain{
        tv6->getRootDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv6->getRootDomain().at(1),
        tv6->getRootDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv6->setLoopDomain(loop_domain);
  }

  // Now that the loop domain of each tensor is [i0, i1*i2]. The max
  // position should be 1 for the persistent tensor and the reduction
  // tensor. All other tensors should be 2.
  {
    MaxPosCalculator calc;
    for (auto tv : fusion.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }
      if (tv == tv1 || tv == tv2) {
        EXPECT_EQ(calc.getMaxPosAll(tv, true), 1)
            << "Invalid max position of " << tv->toString();
      } else {
        EXPECT_EQ(calc.getMaxPosAll(tv, true), 2)
            << "Invalid max position of " << tv->toString();
      }
    }
  }

  // Test simple scheduling
  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    tv->split(1, 4);
    tv->split(1, 128);
  }

  // The loop domain is [i0, i1*i2/4/128, 128, 4]. The max
  // position should be still 1 for the persistent tensor and the reduction
  // tensor. All other tensors should be 4.
  {
    MaxPosCalculator calc;
    for (auto tv : fusion.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }
      if (tv == tv1 || tv == tv2) {
        EXPECT_EQ(calc.getMaxPosAll(tv, true), 1)
            << "Invalid max position of " << tv->toString();
      } else {
        EXPECT_EQ(calc.getMaxPosAll(tv, true), 4)
            << "Invalid max position of " << tv->toString();
      }
    }
  }
}

// Test producer position using the same fusion as
// GetMaxProducerPosFromConsumerWithBroadcast
TEST_F(InliningTest, ProducerPosWithBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  {
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        tv3->getLogicalDomain().at(1)->cloneWithoutRFactor(true)};
    tv2->setLoopDomain(loop_domain);
  }

  for (auto tv : fusion.allTvs()) {
    tv->flatten();
    tv->split(0, 32);
  }

  tv2->inlineAt(-1);
  EXPECT_EQ(tv2->getComputeAtPosition(), tv2->getLoopDomain().size());

  EXPECT_EQ(tv3->getMaxProducerPosition(), tv3->getLoopDomain().size())
      << "Invalid producer position of " << tv3->toString();
}

// Test producer position using the same fusion as
// GetMaxPosAllNormalization
TEST_F(InliningTest, ProducerPosNormalization) {
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

  // tv1 is the persistent tensor. Its inner two domains are the
  // persistent domains. The outermost domain is the only inlinable
  // domain.

  // Set [i0, i1*i2] as the loop domain of each of tensors

  auto reshape_split_factor = tv1->axis(1)->definition()->as<Split>()->factor();

  tv1->setLoopDomain(tv1->getRootDomain());

  // tv2
  {
    std::vector<IterDomain*> loop_domain{
        tv2->getLogicalDomain().at(0),
        IterDomainBuilder(tv1->getLoopDomain().at(1))
            .resetRfactor()
            .iter_type(IterType::Reduction)
            .build()};
    IrBuilder::create<Split>(
        tv2->getLogicalDomain().at(1),
        tv2->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv2->setLoopDomain(loop_domain);
  }

  // Set the loop domain of tv4 before tv3. Use a clone of tv4 in
  // tv3.

  // tv4
  {
    std::vector<IterDomain*> loop_domain{
        tv4->getLogicalDomain().at(0),
        IterDomainBuilder(fusion.zeroVal(), fusion.oneVal())
            .iter_type(IterType::Broadcast)
            .build()};
    IrBuilder::create<Split>(
        tv4->getLogicalDomain().at(1),
        tv4->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv4->setLoopDomain(loop_domain);
  }

  // tv3
  {
    std::vector<IterDomain*> loop_domain{
        tv3->getLogicalDomain().at(0),
        tv4->getLoopDomain().at(1)->cloneWithoutRFactor(
            /*map_with_original=*/true)};
    tv3->setLoopDomain(loop_domain);
  }

  // tv5
  {
    std::vector<IterDomain*> loop_domain{
        tv5->getLogicalDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv5->getLogicalDomain().at(1),
        tv5->getLogicalDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv5->setLoopDomain(loop_domain);
  }

  // tv6
  {
    std::vector<IterDomain*> loop_domain{
        tv6->getRootDomain().at(0),
        tv1->getLoopDomain().at(1)->cloneWithoutRFactor()};
    IrBuilder::create<Split>(
        tv6->getRootDomain().at(1),
        tv6->getRootDomain().at(2),
        loop_domain[1],
        reshape_split_factor,
        false);
    tv6->setLoopDomain(loop_domain);
  }

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    tv->split(1, 4);
    tv->split(1, 128);
  }

  inlineMost();

  // All tensors should look like: [i0, i1*i2/128/4, 128, 4]. The
  // computeAt position and max producer position of each tensor
  // should be:
  //      ca_pos   producer_pos
  // tv1:   1          0
  // tv2:   1          1
  // tv3:   1          1
  // tv4:   1          1
  // tv5:   4          1
  // tv6:   4          4

  EXPECT_EQ(tv1->getComputeAtPosition(), 1);
  EXPECT_EQ(tv1->getMaxProducerPosition(), 0);

  EXPECT_EQ(tv2->getComputeAtPosition(), 1);
  EXPECT_EQ(tv2->getMaxProducerPosition(), 1);

  EXPECT_EQ(tv3->getComputeAtPosition(), 1);
  EXPECT_EQ(tv3->getMaxProducerPosition(), 1);

  EXPECT_EQ(tv4->getComputeAtPosition(), 1);
  EXPECT_EQ(tv4->getMaxProducerPosition(), 1);

  EXPECT_EQ(tv5->getComputeAtPosition(), 4);
  EXPECT_EQ(tv5->getMaxProducerPosition(), 1);

  EXPECT_EQ(tv6->getComputeAtPosition(), 4);
  EXPECT_EQ(tv6->getMaxProducerPosition(), 4);
}

} // namespace nvfuser
