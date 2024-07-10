// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <fusion.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ShardingTest = NVFuserFixtureParamTest<bool>;

// TODO: This test checks that isSharded generates an error when a split/merged
// axis is parallelized with DIDx. Update when this restriction is lifted.
TEST_F(ShardingTest, IsSharded) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  a->axis(2)->parallelize(ParallelType::DIDx);
  a->split(0, 4);
  EXPECT_TRUE(isSharded(a));

  TensorView* b = makeSymbolicTensor(3);
  b->split(1, 4);
  b->axis(1)->parallelize(ParallelType::DIDx);
  EXPECT_ANY_THROW(isSharded(b));

  TensorView* c = makeSymbolicTensor(3);
  c->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);
  EXPECT_ANY_THROW(isSharded(c));
}

TEST_F(ShardingTest, PropagateSharding) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = makeSymbolicTensor(3);
  TensorView* c = add(a, b);

  DeviceMesh mesh({0, 1, 2});
  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  a->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(2)->parallelize(ParallelType::DIDx);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(c);

  // Expected behavior: a's shardings propagate to c.
  propagateShardings(&fusion);
  std::vector<TensorView*> tvs = {c};
  EXPECT_TRUE(getTvsWithDifferentSharding(a, tvs).empty());
}

void isContiguous(TensorView* tv) {
  EXPECT_TRUE(tv->hasAllocation());
  auto contiguity = tv->getContiguity();
  auto alloc_domain = tv->getAllocationDomain();
  for (auto i : c10::irange(contiguity.size())) {
    // TODO: This should eventually check that DeviceDim domains also has no
    // value.
    if (alloc_domain[i]->isReduction() || alloc_domain[i]->isBroadcast()) {
      EXPECT_FALSE(contiguity[i].has_value());
    } else {
      EXPECT_TRUE(contiguity[i].value());
    }
  }
}

TEST_F(ShardingTest, ShardedAllocationDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeContigTensor(3);
  TensorView* b = makeContigTensor(3);
  TensorView* c = add(a, b);
  TensorView* d = sum(c, {1});

  DeviceMesh mesh = DeviceMesh::createForNumDevices(3);
  for (auto tv : {a, b, c, d}) {
    tv->setDeviceMesh(mesh);
  }

  int sharded_dim = 1;
  a->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  c->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(d);

  propagateShardings(&fusion);
  insertReshardings(&fusion);
  insertShardedAxisReordering(&fusion);
  setShardedAllocationDomain(&fusion);
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr)) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->inputs())) {
        isContiguous(tv);
      }
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        isContiguous(tv);
      }
    }
  }
}

TEST_P(ShardingTest, ComputeIndex) {
  const bool creates_concrete_tensor = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1, 2});

  TensorView* a = creates_concrete_tensor ? makeConcreteTensor({4, 2, 3, 5})
                                          : makeSymbolicTensor(4);
  TensorView* b = sum(a, {0});
  TensorView* c = add(a, a);
  TensorView* d = permute(a, {{2, 0}});

  fusion->addInput(a);
  fusion->addOutput(b);
  fusion->addOutput(c);
  fusion->addOutput(d);

  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);
  d->setDeviceMesh(mesh);
  a->axis(2)->parallelize(ParallelType::DIDx);
  b->axis(2)->parallelize(ParallelType::DIDx);
  c->axis(2)->parallelize(ParallelType::DIDx);
  d->axis(0)->parallelize(ParallelType::DIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // Dimension 2 has size 1 because that dimension is DIDx parallelized.
  auto a_tensor = at::randn({4, 2, 1, 5}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {a_tensor});
  auto outputs = fe.runFusion({a_tensor});
  testValidate(fusion.get(), outputs, {a_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ShardingTest,
    testing::Bool(),
    testing::PrintToStringParamName());

} // namespace nvfuser
