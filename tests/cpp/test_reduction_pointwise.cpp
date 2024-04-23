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
  bool can_schedule = SchedulerEntry::canSchedule(
      ScheduleHeuristic::Reduction, fusion.get(), runtime_info);
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
  bool can_schedule = SchedulerEntry::canSchedule(
      ScheduleHeuristic::Reduction, fusion.get(), runtime_info);
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
} // namespace nvfuser
