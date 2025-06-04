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
#include <preseg_passes/finalize_multidevice_domains.h>
#include <preseg_passes/insert_reshardings.h>
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

TEST_F(ShardingTest, Allreduce) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(2);
  TensorView* y = sum(x, {0});

  x->axis(0)->parallelize(ParallelType::DIDx);
  y->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_FALSE(isSharded(y));
}

TEST_F(ShardingTest, ReductionScatter) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(2);
  TensorView* y = sum(x, {0});

  x->axis(0)->parallelize(ParallelType::DIDx);
  y->axis(0)->parallelize(ParallelType::DIDx);
  y->axis(1)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isSharded(y));
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
      preseg_passes::FinalizeMultideviceDomainsPass>::runPass(&fusion);
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

TEST_F(ShardingTest, ResidualAdd) {
  // This is similar to the residual add after MHA dropout in the transformer.
  // The output of linear following MHA is all-gathered and sharded on the
  // sequence dim. This sharding can be propagated to the linear output through
  // backpropagating the shardings from residual add. This information is not
  // present during forward propagation.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = uniform(
      shape(tv0),
      fusion->zeroVal(DataType::Float),
      fusion->oneVal(DataType::Float),
      DataType::Float);
  TensorView* tv2 = add(tv0, tv1);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(0, mesh.size());
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  // getShardedLogicalAxis uses maybeAllocationDomain to get the sharded axis.
  // Setting allocation domain here manually, which is otherwise done by
  // FinalizeMultideviceDomainsPass, to isolate the test to a single preseg
  // pass.
  for (auto tv : fusion->allTvs()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
  NVF_CHECK(tv1->hasDeviceMesh());
  int64_t expected_sharded_axis =
      getShardedLogicalAxis(tv0, ParallelType::DIDx);
  NVF_CHECK(expected_sharded_axis != -1, "tv0 should have a sharded axis.");
  NVF_CHECK(
      getShardedLogicalAxis(tv1, ParallelType::DIDx) == expected_sharded_axis,
      "Expected tv1 to be sharded like tv0 due to backpropagation of "
      "shardings.");
}

TEST_F(ShardingTest, PropagateParallelTypeOnce) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  TensorView* tv2 = add(tv0, tv1);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(0, mesh.size());
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  tv1->setDeviceMesh(mesh);
  tv1->outer_split(1, mesh.size());
  tv1->axis(1)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto tv : fusion->allTvs()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
  NVF_CHECK(numDeviceDims(tv2) == 1);
  int64_t expected_sharded_axis =
      getShardedLogicalAxis(tv0, ParallelType::DIDx);
  NVF_CHECK(expected_sharded_axis != -1, "tv0 should have a sharded axis.");
  NVF_CHECK(
      getShardedLogicalAxis(tv2, ParallelType::DIDx) == expected_sharded_axis,
      "Expected tv2 to be sharded like tv0 due to forward propagation of "
      "shardings.");
}

TEST_F(ShardingTest, ReductionDIDxIsIgnored) {
  // When propagating shardings, DIDx on reduction dimensions should be ignored.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});

  TensorView* tv0 = makeContigTensor(2); // [M, K]
  TensorView* tv1 = makeContigTensor(2); // [N, K]
  TensorView* tv2 = makeContigTensor(2); // [M, N]
  TensorView* tv3 = linear(tv0, tv1); // [M, N, r{K}]
  TensorView* tv4 = add(tv2, tv3); // [M, N]

  // Shard K in tv0 and tv1.
  // Shard N in tv2.
  for (auto tv : {tv0, tv1, tv2}) {
    tv->setDeviceMesh(mesh);
    tv->outer_split(1, mesh.size());
    tv->axis(1)->parallelize(ParallelType::DIDx);
  }

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  // tv4 will be sharded similarly as tv2 during forward propagation since tv3
  // is parallelized on r{K} Due to backpropagation of shardings, tv3 should be
  // sharded on N since DIDx on r{K} is ignored.
  for (auto tv : fusion->allTvs()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  int64_t expected_sharded_axis =
      getShardedLogicalAxis(tv2, ParallelType::DIDx);
  NVF_CHECK(expected_sharded_axis != -1, "tv2 should have a sharded axis.");
  NVF_CHECK(
      getShardedLogicalAxis(tv3, ParallelType::DIDx) == expected_sharded_axis,
      "Expected tv3 to be sharded like tv2 due to backpropagation of "
      "shardings.");
  NVF_CHECK(
      getShardedLogicalAxis(tv4, ParallelType::DIDx) == expected_sharded_axis,
      "Expected tv4 to be sharded like tv2 due to forward propagation of "
      "shardings.");
}

TEST_F(ShardingTest, ShardedNonDivisibleReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});
  int64_t d = mesh.size();

  TensorView* tv0 = makeContigConcreteTensor({6});
  TensorView* tv1 = reshape(tv0, {6}, {3, 2});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(0, d);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  auto run_propagate_shardings = [&]() {
    preseg_passes::OptimizationPass<
        preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  };

  // tv1 should be sharded on the outer reshape id i.e. {3}
  // but it is not divisible by d=2
  EXPECT_THAT(
      run_propagate_shardings,
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Require the sharded ID to be divisible by the split factor")));
}

TEST_F(ShardingTest, ShardedInnerReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});
  int64_t d = mesh.size();

  TensorView* tv0 = makeContigConcreteTensor({3, 5 * d});
  TensorView* tv1 = reshape(tv0, {3, 5 * d}, {15 * d});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(1, d);
  tv0->axis(1)->parallelize(ParallelType::DIDx);

  auto run_propagate_shardings = [&]() {
    preseg_passes::OptimizationPass<
        preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  };

  EXPECT_THAT(
      run_propagate_shardings,
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Expected the sharding to be on the outer reshaped id")));
}

TEST_F(ShardingTest, ShardedReshapeWithIndependentSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh({0, 1});
  int64_t d = mesh.size();

  TensorView* tv0 = makeContigConcreteTensor({3 * d, 5, 7 * d, 9});
  TensorView* tv1 = reshape(tv0, {3 * d, 5, 7 * d, 9}, {3 * d * 5, 7 * d * 9});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  tv0->setDeviceMesh(mesh);
  tv0->outer_split(0, d);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv0->outer_split(3, d);
  tv0->axis(3)->parallelize(ParallelType::DIDy);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  for (auto tv : fusion->allTvs()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
  EXPECT_EQ(getShardedLogicalAxis(tv1, ParallelType::DIDx), 0);
  EXPECT_EQ(getShardedLogicalAxis(tv1, ParallelType::DIDy), 1);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ShardingTest,
    testing::Bool(),
    testing::PrintToStringParamName());

} // namespace nvfuser
