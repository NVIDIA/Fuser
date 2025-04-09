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
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/make_resharding_contiguous.h>
#include <preseg_passes/propagate_shardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ShardingTest = NVFuserFixtureParamTest<bool>;

TEST_F(ShardingTest, LogicalIsSharded) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(3);
  x->axis(2)->parallelize(ParallelType::DIDx);
  x->split(0, 4);

  EXPECT_TRUE(isSharded(x)) << "DIDx on logical domain:" << std::endl
                            << x->domain()->toString(0, /*loop_only=*/false);
}

TEST_F(ShardingTest, AllocationIsSharded) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(3);
  x->split(1, 4);
  x->axis(1)->parallelize(ParallelType::DIDx);
  x->setAllocationDomain(x->getLoopDomain(), true);

  EXPECT_TRUE(isSharded(x)) << "DIDx on allocation domain:" << std::endl
                            << x->domain()->toString(0, /*loop_only=*/false);
}

TEST_F(ShardingTest, MultipleDIDx) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(1);
  x->split(0, 2);
  x->axis(0)->parallelize(ParallelType::DIDx);
  x->axis(1)->parallelize(ParallelType::DIDx);
  x->setAllocationDomain(x->getLoopDomain(), true);

  EXPECT_ANY_THROW(isSharded(x))
      << "Multiple DIDx:" << std::endl
      << x->domain()->toString(0, /*loop_only=*/false);
}

TEST_F(ShardingTest, ReductionShouldNotBeSharded) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(2);
  TensorView* y = sum(x, {0});

  x->axis(0)->parallelize(ParallelType::DIDx);
  // y->axis(0) is a reduction dimension and shouldn't be sharded. Doing so
  // leads to a multi-DIDx exception.
  y->axis(0)->parallelize(ParallelType::DIDx);
  y->axis(1)->parallelize(ParallelType::DIDx);

  EXPECT_ANY_THROW(isSharded(y)) << "Multiple DIDx";
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
  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(&fusion);
  std::vector<TensorView*> tvs = {c};
  EXPECT_TRUE(getTvsWithDifferentSharding(a, tvs).empty());
}

void isContiguous(TensorView* tv) {
  EXPECT_TRUE(tv->hasAllocation());
  auto contiguity = tv->getContiguity();
  auto alloc_domain = tv->getAllocationDomain();
  for (auto i : arange(contiguity.size())) {
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

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(&fusion);
  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(&fusion);
  preseg_passes::OptimizationPass<
      preseg_passes::MakeReshardingContiguousPass>::runPass(&fusion);
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
  TensorDomain::noReductions(b->getLoopDomain())[1]->parallelize(
      ParallelType::DIDx);
  c->axis(2)->parallelize(ParallelType::DIDx);
  d->axis(0)->parallelize(ParallelType::DIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // Dimension 2 has size 1 because that dimension is DIDx parallelized.
  auto a_tensor = at::randn({4, 2, 1, 5}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {a_tensor});
  auto outputs = ke.run({a_tensor});
  testValidate(fusion.get(), outputs, {a_tensor}, __LINE__, __FILE__);
}

TEST_F(ShardingTest, MultiDimDeviceMesh) {
  DeviceMesh mesh({3, 4, 1, 0, 8, 2}, {2, 3});
  // Shape not consistent with number of devices
  EXPECT_ANY_THROW(DeviceMesh({1, 2}, {2, 3}));
  // Duplicates in DeviceMesh
  EXPECT_ANY_THROW(DeviceMesh({1, 2, 0, 2}, {2, 3}));

  std::vector<int64_t> local_indices_8 = {1, 1};
  std::vector<int64_t> local_indices_1 = {0, 2};
  EXPECT_EQ(mesh.getIndices(8), local_indices_8);
  EXPECT_EQ(mesh.getIndices(1), local_indices_1);

  std::vector<DeviceIdxType> slice_didx_034 = {3, 4, 1};
  std::vector<DeviceIdxType> slice_didy_12 = {1, 2};
  EXPECT_EQ(mesh.getSlice(1, ParallelType::DIDx), slice_didx_034);
  EXPECT_EQ(mesh.getSlice(1, ParallelType::DIDy), slice_didy_12);
  EXPECT_EQ(mesh.getSlice(2, ParallelType::DIDy), slice_didy_12);

  DeviceMesh mesh3d = DeviceMesh::createForShape({2, 3, 4});
  std::vector<DeviceIdxType> slice_didz = {6, 18};
  std::vector<DeviceIdxType> slice_didy = {14, 18, 22};
  std::vector<DeviceIdxType> slice_didx = {16, 17, 18, 19};
  EXPECT_EQ(mesh3d.getSlice(18, ParallelType::DIDz), slice_didz);
  EXPECT_EQ(mesh3d.getSlice(18, ParallelType::DIDy), slice_didy);
  EXPECT_EQ(mesh3d.getSlice(18, ParallelType::DIDx), slice_didx);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ShardingTest,
    testing::Bool(),
    testing::PrintToStringParamName());

} // namespace nvfuser
