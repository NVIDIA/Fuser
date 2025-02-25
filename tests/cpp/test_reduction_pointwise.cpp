// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
namespace nvfuser {

using PointwiseFusedReductionTest = NVFuserTest;

// inner reduction + non-broadcast epilogue, can't be fused
// outer reduction + non-broadcast epilogue, can be fused
// checked by: SchedulerTopologyChecker::supportedPostReductionFusion
namespace {
void ReductionNonBroadcast(const int reduction_dim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int dim0 = 8192;
  const int dim1 = 1024;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(1, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = sum(tv0, {reduction_dim});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({reduction_dim == 0 ? dim1 : dim0}, options);

  // check whether the fusion can be scheduled as reduction
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0, t1});
  bool can_schedule = Schedule::canSchedule(
      SchedulerType::Reduction, fusion.get(), runtime_info);
  if (reduction_dim == 1) {
    ASSERT_FALSE(can_schedule);
  } else {
    ASSERT_TRUE(can_schedule);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto t_ref = t0.sum({reduction_dim}) + t1;
  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t_ref},
      __LINE__,
      __FILE__);
};
} // namespace
TEST_F(PointwiseFusedReductionTest, InnerReductionNonBroadcast) {
  const int reduction_dim = 1;
  ReductionNonBroadcast(reduction_dim);
}
TEST_F(PointwiseFusedReductionTest, OuterReductionNonBroadcast) {
  const int reduction_dim = 0;
  ReductionNonBroadcast(reduction_dim);
}

// inner reduction + broadcast epilogue, can't be fused
// outer reduction + broadcast epilogue, can't be fused
// checked by: SchedulerTopologyChecker::hasPostReductionBCast
namespace {
void ReductionBroadcast(const int reduction_dim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int dim0 = 8192;
  const int dim1 = 1024;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = sum(tv0, {reduction_dim});
  auto tv3 = broadcast(tv2, {0 == reduction_dim, 1 == reduction_dim});
  auto tv4 = add(tv3, tv1);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0, dim1}, options);

  // check whether the fusion can be scheduled as reduction
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0, t1});
  bool can_schedule = Schedule::canSchedule(
      SchedulerType::Reduction, fusion.get(), runtime_info);
  ASSERT_FALSE(can_schedule);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto t_ref = t0.sum({reduction_dim}).unsqueeze(reduction_dim) + t1;
  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t_ref},
      __LINE__,
      __FILE__);
};
} // namespace
TEST_F(PointwiseFusedReductionTest, InnerReductionBroadcast) {
  const int reduction_dim = 1;
  ReductionBroadcast(reduction_dim);
}
TEST_F(PointwiseFusedReductionTest, OuterReductionBroadcast) {
  const int reduction_dim = 0;
  ReductionBroadcast(reduction_dim);
}

// dtype, group_factor, inner dim size
using IterGroupedParams = std::tuple<DataType, int64_t, int64_t>;
using IterGroupedInnerReduction = NVFuserFixtureParamTest<IterGroupedParams>;
TEST_P(IterGroupedInnerReduction, VariedGroupFactor) {
  auto [dtype, group_factor, dim1] = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, dtype);
  fusion->addInput(tv0);
  tv0 = maybeCastOp(DataType::Float, tv0);
  auto tv1 = sum(tv0, {1});
  fusion->addOutput(tv1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16384, dim1}, options);

  // Generate heuristics, then:
  // (1) enforce unroll on top of vectorization
  // (2) enforce unroll on iter domain, which turns into grouped
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0});
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion.get(), runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();
  EXPECT_TRUE(rparams->vectorize_inner_reduction);

  // must disable TIDy to use grouped reduction
  if (group_factor > 1) {
    rparams->unroll_factor_iter_dom = group_factor;
    rparams->multiple_reds_per_blk = 1;
    rparams->block_dim_iter_dom = ParallelType::Serial;
    rparams->lparams.bindUnsafe(
        LaunchParams::UNINITIALIZED_VAL, ParallelType::TIDy);
  }

  // Schedule, compile, run, validate
  auto fusion_copy = *fusion;
  scheduler_instance->schedule(fusion.get(), rparams);

  // lowering & check iteration grouped reductions
  if (group_factor > 1) {
    GpuLower gpulw(fusion.get());
    // only support grouping pow2 iterations
    if (group_factor & (group_factor - 1)) {
      EXPECT_THAT(
          [&]() { gpulw.run(); },
          testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
              "Iteration grouped reduction only support grouping 2, 4, 8, or 16 iterations")));
      return;
    } else {
      gpulw.run();
      NVF_CHECK(
          gpulw.kernel()->summary().has_iter_grouped_reductions,
          "There must be iter domain grouped reductions.");
      NVF_CHECK(
          gpulw.kernel()->summary().num_grouped_iterations ==
              rparams->unroll_factor_iter_dom,
          "Expected ",
          rparams->unroll_factor_iter_dom,
          " grouped iterations, found ",
          gpulw.kernel()->summary().num_grouped_iterations);
    }
  }

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0}, rparams->lparams);
  auto cg_outputs = ke.run({t0}, {}, rparams->lparams);
  testValidate(&fusion_copy, cg_outputs, {t0}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    IterGroupedInnerReduction,
    ::testing::Combine(
        ::testing::Values(DataType::Half, DataType::Float),
        ::testing::Values(1L, 2L, 3L, 4L, 8L, 16L),
        ::testing::Values(1024L, 2048L, 8192L, 16384L)),
    [](const testing::TestParamInfo<IterGroupedParams>& info) -> std::string {
      std::stringstream ss;
      ss << "dtype_" << std::get<0>(info.param);
      ss << "_factor_" << std::get<1>(info.param);
      ss << "_reduction_size_" << std::get<2>(info.param);
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser
