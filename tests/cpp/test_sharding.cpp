// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock.h>
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

TEST_F(ShardingTest, DeviceMesh) {
  DeviceMesh mesh({3, 4, 1, 0, 8, 2}, {2, 3});
  // Sizes are not consistent with number of devices
  EXPECT_ANY_THROW(DeviceMesh({1, 2}, {2, 3}));
  // Duplicates in DeviceMesh
  EXPECT_ANY_THROW(DeviceMesh({1, 2, 0, 2}, {2, 3}));

  std::vector<int64_t> local_indices_8 = {1, 1};
  std::vector<int64_t> local_indices_1 = {0, 2};
  EXPECT_EQ(mesh.getIndices(8), local_indices_8);
  EXPECT_EQ(mesh.getIndices(1), local_indices_1);

  std::vector<DeviceIdxType> team_axis1_group0 = {3, 4, 1};
  std::vector<DeviceIdxType> team_axis0_group2 = {1, 2};
  std::vector<DeviceIdxType> team_2_0 = {1, 2};
  EXPECT_EQ(mesh.getTeam(1, 1), team_axis1_group0);
  EXPECT_EQ(mesh.getTeam(1, 0), team_axis0_group2);
  EXPECT_EQ(mesh.getTeam(2, 0), team_axis0_group2);

  DeviceMesh mesh3d = DeviceMesh::createForShape({2, 3, 4});
  std::vector<DeviceIdxType> team_axis0_group1 = {6, 18};
  std::vector<DeviceIdxType> team_axis1_group1 = {14, 18, 22};
  std::vector<DeviceIdxType> team_axis2_group2 = {16, 17, 18, 19};
  EXPECT_EQ(mesh3d.getTeam(18, 0), team_axis0_group1);
  EXPECT_EQ(mesh3d.getTeam(18, 1), team_axis1_group1);
  EXPECT_EQ(mesh3d.getTeam(18, 2), team_axis2_group2);
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
