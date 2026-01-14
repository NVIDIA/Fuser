// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <memory>
#include <sstream>
#include <tuple>

#include <gtest/gtest.h>

#include "logical_domain_map.h"
#include "ops/all_ops.h"
#include "scheduler/all_schedulers.h"
#include "scheduler/matmul_utils.h"
#include "scheduler/utils.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"
#include "type.h"

namespace nvfuser {

using TestParam = std::tuple<int64_t, DataType>;

class ClusterReductionTest : public NVFuserTest,
                             public ::testing::WithParamInterface<TestParam> {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel);
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

using ClusterReductionTestAutoScheduler = ClusterReductionTest;
TEST_P(ClusterReductionTestAutoScheduler, Softmax) {
  auto [hidden_size, dtype] = GetParam();
  int batch_size = scheduler_utils::safeDiv(deviceSMCount(), 8);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = maybeCastOp(DataType::Float, tv0);
  auto tv2 = softmax(tv1, 1);
  auto tv3 = maybeCastOp(DataType::BFloat16, tv2);
  fusion.addOutput(tv3);
  auto unscheduled_fusion_copy = fusion;

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options).clamp(-2, 2);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  auto runtime = executor_cache.getMostRecentKernelRuntime();
  if (hidden_size * dataTypeSizeBit(dtype) <=
      scheduler_utils::register_file_size_bit *
          scheduler_utils::getMaxClusterSize()) {
    EXPECT_FALSE(runtime->isSegmented());
    EXPECT_TRUE(runtime->schedulerHeuristics()
                    ->heuristicsList()
                    .at(0)
                    ->as<ReductionParams>()
                    ->cross_cluster_reduction);
  }
  testValidate(&unscheduled_fusion_copy, outputs, {t0});
}
INSTANTIATE_TEST_SUITE_P(
    ,
    ClusterReductionTestAutoScheduler,
    ::testing::Combine(
        ::testing::Values(
            129280, // DeepSeek-R1
            128256, //  Llama3
            202048, //  Llama4
            256000, //  Gemma2
            131072, //  Mistral
            152064, //  Qwen2
            100352, //  Phi4
            1024 * 1024 // largest size for fp16 using 16 CTAs per cluster
            ),
        ::testing::Values(DataType::BFloat16, DataType::Float)),
    [](const testing::TestParamInfo<TestParam>& info) {
      std::stringstream ss;
      ss << "_hidden_size_" << std::get<0>(info.param);
      ss << "_dtype_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

TEST_F(ClusterReductionTest, InvalidClusterSize) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());
  const int64_t vect = 8, bdimx = 128, persistent_batch = 2;
  // set an illegal cluster size to trigger the error
  const int64_t cluster_size = scheduler_utils::getMaxClusterSize() + 1;
  const int64_t reduction_size = vect * bdimx * persistent_batch * cluster_size;
  DataType dtype = DataType::BFloat16;
  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = maybeCastOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv1, tv3);
  auto tv5 = maybeCastOp(DataType::BFloat16, tv4);
  fusion.addOutput(tv5);
  auto unscheduled_fusion_copy = fusion;

  // [I, R]
  tv2->split(1, 8);
  // [I, R/8, 8]
  tv2->split(1, 128);
  // [I, R/8/128, 128, 8]
  tv2->split(1, 2, false);
  // [I, 2, R/8/128/2, 128, 8]
  // [BIDy, Serial, BIDx(cluster), TIDx, Vectorize for IO]
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
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
  EXPECT_THAT(
      [&]() { ke.run({t0}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Clustered domain size must be less than or "
                               "equal to max allowed cluster size and larger "
                               "than 1.")));
}

// Test the getMaxActiveClusters utility function
TEST_F(NVFuserTest, GetMaxActiveClusters) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  const int sm_minor = at::cuda::getCurrentDeviceProperties()->minor;
  // Test various cluster configurations from 1 to 17
  for (int64_t cluster_n : arange(1, 18)) {
    MatmulParams::ClusterDims cluster_dims{1, cluster_n};
    int64_t max_active =
        scheduler_utils::getMaxActiveClusters(cluster_dims.m * cluster_dims.n);
    // Our regular CI only covers 8.0, 9.0, 10.0, etc.
    // For other minor versions, max allowed cluster size is not tested.
    if (sm_minor == 0) {
      if (cluster_dims.m * cluster_dims.n > 16) {
        EXPECT_EQ(max_active, 0);
      } else {
        EXPECT_GT(max_active, 0);
      }
    }
  }
}

// Test the getMaxClusterSize utility function
TEST_F(NVFuserTest, GetMaxClusterSize) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  const int sm_minor = at::cuda::getCurrentDeviceProperties()->minor;
  int64_t max_cluster_size = scheduler_utils::getMaxClusterSize();
  // Hopper (9.0) and later devices support clusters
  // Our regular CI only covers 9.0, 10.0, etc.
  // For other minor versions, max allowed cluster size is not tested.
  if (sm_minor == 0) {
    EXPECT_EQ(max_cluster_size, 16);
  }
}

} // namespace nvfuser
