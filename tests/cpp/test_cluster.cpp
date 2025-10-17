// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format off
#include <gtest/gtest.h>
#include <memory>
#include <tuple>
#include <sstream>

#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include "type.h"

namespace nvfuser {

using TestParam = std::tuple<int64_t, DataType>;

class ClusterReductionTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<TestParam> {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  }
};

TEST_P(ClusterReductionTest, SimpleFusionAllReduce) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  const int64_t vect = 2, bdimx = 128, persistent_batch = 2;
  const auto [cluster_size, dtype] = GetParam();
  const int64_t reduction_size = vect * bdimx * persistent_batch * cluster_size;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  const DataType compute_dtype =
      (dtype == DataType::Double) ? DataType::Double : DataType::Float;
  tv1 = maybeCastOp(compute_dtype, tv1);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv1, tv3);
  tv4 = maybeCastOp(dtype, tv4);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);
  auto unscheduled_fusion_copy = fusion;

  // [I, R]
  tv2->split(1, vect);
  // [I, R/vect, vect]
  tv2->split(1, bdimx);
  // [I, R/vect/bdimx, bdimx, vect]
  tv2->split(1, persistent_batch, false);
  // [I, persistent_batch, R/vect/bdimx/persistent_batch, bdimx, vect]
  // [BIDy, Serial, BIDx(cluster), TIDx, Vectorize or Serial]
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
  // set clustered blocks to use cluster reduction
  tv2->axis(-3)->setClusteredBlocks();
  tv2->axis(0)->parallelize(ParallelType::BIDy);

  auto reference = tv2->rFactor({-1, -4});

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({256, reduction_size}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto outputs = ke.run({t0});
  EXPECT_TRUE(ke.compiledKernel()->kernel()->summary().has_cluster_reduction);
  testValidate(&unscheduled_fusion_copy, outputs, {t0});
}

TEST_P(ClusterReductionTest, SimpleFusionNotAllReduce) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  const int64_t vect = 2, bdimx = 128, serial = 2;
  const auto [cluster_size, dtype] = GetParam();
  const int64_t reduction_size = vect * bdimx * serial * cluster_size;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  const DataType compute_dtype =
      (dtype == DataType::Double) ? DataType::Double : DataType::Float;
  tv1 = maybeCastOp(compute_dtype, tv1);
  auto tv2 = sum(tv1, {1});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);
  auto unscheduled_fusion_copy = fusion;
  // [I, R]
  tv2->split(1, vect);
  // [I, R/vect, vect]
  tv2->split(1, bdimx);
  // [I, R/vect/bdimx, bdimx, vect]
  tv2->split(1, serial, false);
  // [I, serial, R/vect/bdimx/serial, bdimx, vect]
  // [BIDy, Serial, BIDx(cluster), TIDx, Vectorize or Serial]
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
  // set clustered blocks to use cluster reduction
  tv2->axis(-3)->setClusteredBlocks();
  tv2->axis(0)->parallelize(ParallelType::BIDy);

  auto reference = tv2->rFactor({-1, -4});

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({256, reduction_size}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto outputs = ke.run({t0});
  EXPECT_TRUE(ke.compiledKernel()->kernel()->summary().has_cluster_reduction);
  testValidate(&unscheduled_fusion_copy, outputs, {t0});
}
INSTANTIATE_TEST_SUITE_P(
    ,
    ClusterReductionTest,
    ::testing::Combine(
        ::testing::Range<int64_t>(2, 17),
        ::testing::Values(
            DataType::BFloat16,
            DataType::Float,
            DataType::Double)),
    [](const testing::TestParamInfo<TestParam>& info) {
      std::stringstream ss;
      ss << "cluster_" << std::get<0>(info.param);
      ss << "_dtype_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });
} // namespace nvfuser
