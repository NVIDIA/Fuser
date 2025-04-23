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

TEST_F(NVFuserTest, InnerReductionUnrollVectorization) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({256, 10240}, options);

  // Generate heuristics & enforce unroll on top of vectorization
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0});
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion.get(), runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();
  EXPECT_TRUE(rparams->vectorize_inner_reduction);
  rparams->unroll_factor_top_of_vectorization = 2;

  // Schedule, compile, run, validate
  auto fusion_copy = *fusion;
  scheduler_instance->schedule(fusion.get(), rparams);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0}, rparams->lparams);
  auto cg_outputs = ke.run({t0}, {}, rparams->lparams);
  testValidate(&fusion_copy, cg_outputs, {t0}, __LINE__, __FILE__);
}

// https://github.com/NVIDIA/Fuser/issues/3811
TEST_F(NVFuserTest, ReductionSchedulerWithAdditionalID) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  // tv0 [ b0, i1 ]
  auto tv0 = makeContigConcreteTensor({1, -1});
  fusion.addInput(tv0);
  // tv1 [ i2, i1 ]
  // current scheduler picks tv0 as the reference TV, transformations are
  // propagated to other TVs.
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {0, 1});
  fusion.addOutput(tv2);
  auto tv3 = add(tv0, tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, 100}, options);
  auto t1 = at::randn({5, 100}, options);
  std::vector<c10::IValue> inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(inputs);
  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);

  // checking segmentation
  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen!");
}

// https://github.com/NVIDIA/Fuser/issues/3811
TEST_F(NVFuserTest, ReductionSchedulerWithAdditionalIDInnerNormalization) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigConcreteTensor({-1, -1, 1});
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1, 2}, /*keep_dim=*/true);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);
  auto tv4 = add(tv0, tv1);
  fusion.addOutput(tv4);

  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({100, 20, 1}, options);
  auto t1 = at::randn({100, 20, 128}, options);
  std::vector<c10::IValue> inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(inputs);
  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);

  // checking segmentation
  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen!");
}

// https://github.com/NVIDIA/Fuser/issues/3811
TEST_F(NVFuserTest, ReductionSchedulerWithAdditionalIDOuterNormalization) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigConcreteTensor({1, -1, -1});
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {0, 1}, /*keep_dim=*/true);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);
  auto tv4 = add(tv0, tv1);
  fusion.addOutput(tv4);

  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, 20, 100}, options);
  auto t1 = at::randn({128, 20, 100}, options);
  std::vector<c10::IValue> inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(inputs);
  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);

  // checking segmentation
  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen!");

  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
