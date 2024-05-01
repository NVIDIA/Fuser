// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <gtest/gtest.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

#include <csrc/iter_visitor.h>

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
  EXPECT_TRUE(isSharded(b));

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
  propagateShardings(&fusion);

  EXPECT_TRUE(mesh == c->getDeviceMesh());
  EXPECT_TRUE(c->axis(0)->getParallelType() == ParallelType::DIDx);
  EXPECT_TRUE(c->axis(1)->getParallelType() == ParallelType::Serial);
  EXPECT_TRUE(c->axis(2)->getParallelType() == ParallelType::Serial);
}

TEST_P(ShardingTest, ComputeIndex) {
  const bool creates_concrete_tensor = GetParam();
  Fusion fusion;
  FusionGuard fg(&fusion);
  DeviceMesh mesh({0, 1, 2});

  TensorView* a = creates_concrete_tensor ? makeConcreteTensor({4, 2, 3, 5})
                                          : makeSymbolicTensor(4);
  TensorView* b = sum(a, {0});
  TensorView* c = add(a, a);
  TensorView* d = permute(a, {{2, 0}});

  fusion.addInput(a);
  fusion.addOutput(b);
  fusion.addOutput(c);
  fusion.addOutput(d);

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
  fe.compileFusion(&fusion, {a_tensor});
  auto outputs = fe.runFusion({a_tensor});
  testValidate(&fusion, outputs, {a_tensor}, __LINE__, __FILE__);
}

TEST_P(ShardingTest, ShardSplitAxis_Computation) {
  const bool creates_concrete_tensor = GetParam();
  Fusion fusion;
  FusionGuard fg(&fusion);
  int num_devices = 2;
  DeviceMesh mesh({0, 1});

  TensorView* a = creates_concrete_tensor ? makeConcreteTensor({3, 8, 3})
                                          : makeSymbolicTensor(3);
  TensorView* b = sum(a, {2});
  TensorView* c = add(a, a);
  TensorView* d = permute(a, {{1, 0}});

  fusion.addInput(a);
  fusion.addOutput(b);
  fusion.addOutput(c);
  fusion.addOutput(d);

  std::vector<TensorView*> tvs = {a, b, c};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->split(1, num_devices, false);
    tv->axis(1)->parallelize(ParallelType::DIDx);
    tv->setAllocationDomain(tv->getLeafDomain(), true);
  }
  d->setDeviceMesh(mesh);
  d->split(0, num_devices, false);
  d->axis(0)->parallelize(ParallelType::DIDx);
  d->setAllocationDomain(d->getLeafDomain(), true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto a_tensor = at::randn({3, 4, 3}, options); // Input is a sharded tensor

  FusionExecutor fe;
  fe.compileFusion(&fusion, {a_tensor});
  auto outputs = fe.runFusion({a_tensor});
  testValidate(&fusion, outputs, {a_tensor}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ShardingTest,
    testing::Bool(),
    testing::PrintToStringParamName());

} // namespace nvfuser
