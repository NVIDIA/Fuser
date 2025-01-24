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
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
namespace nvfuser {

using PointwiseFusedReductionTest = NVFuserTest;
using InnerReductionTest = NVFuserTest;

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

TEST_F(InnerReductionTest, InnerReductionUnrollVectorization) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({256, 10240}, options);
  std::vector<c10::IValue> runtime_inputs({t0});

  // Generate heuristics & enforce unroll on top of vectorization
  SchedulerRuntimeInfo runtime_info(fusion.get(), runtime_inputs);
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Reduction);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion.get(), runtime_info);
  auto rparams = heuristic_params->as<ReductionParams>();
  EXPECT_TRUE(rparams->vectorize_inner_reduction);
  // rparams->unroll_factor_top_of_vectorization = 2;

  // Schedule, compile, run, validate
  auto fusion_copy = *fusion;
  scheduler_instance->schedule(fusion.get(), rparams);
  KernelExecutor ke;
  ke.compile(fusion.get(), runtime_inputs, rparams->lparams);
  auto cg_outputs = ke.run(runtime_inputs, rparams->lparams);
  testValidate(&fusion_copy, cg_outputs, runtime_inputs, __LINE__, __FILE__);
}

TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBuffer1) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  constexpr int dim0 = 16384, dim1 = 16384;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  int64_t tidx = 128;
  int64_t vect = 4;
  int64_t number_of_stages = 2;
  int64_t prefetch_distance = 0;
  // CircularBufferType circular_buffer_type =
  // WarpSpecialized(ParallelType::TIDx);
  CircularBufferType circular_buffer_type = Pipelined(true);

  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0s->setMemoryType(MemoryType::Shared);
  auto tv1r = tv1->cacheBefore();
  // tv0 --> tv0s --> tv1r ---> tv1
  // Error during best effort replay, a transformation was called that conflicts
  // with an root-to-logical call schedule tma tensors [I,R] --> [I,
  // R/tma/stages, stages, tma] R/tma/stages = 1
  int64_t tma_len = dim1 / number_of_stages;
  tv0s->split(1, tma_len);
  tv0s->split(1, number_of_stages);

  // schedule reduction tensor
  // [I, R] -> [I, R/V, V]
  tv1r->split(1, vect);
  // [I, R/V/TIDx, TIDx, V]
  tv1r->split(1, tidx);
  auto reduction_tv_ref = tv1r->rFactor({1, 3});
  std::cout << "reduction_tv_ref " << reduction_tv_ref->toString() << std::endl;
  std::cout << "dim0 " << dim0 << std::endl;
  std::cout << "prefetch_distance " << prefetch_distance << std::endl;

  // // Set inlineAt before applying circular buffer
  // inlineAllAt(tv1, /*pos=*/2);

  reduction_tv_ref->axis(0)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(reduction_tv_ref);

  // TMA load, [I0*I1/TMA/Stage, Stage, TMA]
  tv0s->axis(-1)->parallelize(ParallelType::Bulk);

  // TIDx for computation, [I, R/V/TIDx, TIDx, V]
  reduction_tv_ref->axis(2)->parallelize(ParallelType::TIDx);

  // TIDx for computation, [I, TIDx]
  tv1r->axis(1)->parallelize(ParallelType::TIDx);

  fusion->print();

  // // Vectorize output
  // tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // Circular buffer
  inlineAllAt(reduction_tv_ref, /*pos=*/2);
  tv0s->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  auto outputs = ke.run({at_tv0});
  auto at_output = at_tv0.sum(-1);
  testValidate(
      fusion.get(), outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}

// 0.465 ms, prefetch_distance = 0
TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBuffer2) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  constexpr int dim0 = 16384, dim1 = 16384;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  int64_t tidx = 128;
  int64_t vect = 2;
  int64_t number_of_stages = 2;
  int64_t prefetch_distance = 1;
  // CircularBufferType circular_buffer_type =
  // WarpSpecialized(ParallelType::TIDx);
  CircularBufferType circular_buffer_type = Pipelined(true);

  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0s->setMemoryType(MemoryType::Shared);
  auto tv1r = tv1->cacheBefore();
  // tv0 --> tv0s --> tv1r ---> tv1

  // scheduler tma tensor
  // [I, R/tma/stages, stages, tma]
  int64_t tma_len = tidx * vect;
  tv0s->split(1, tma_len);
  tv0s->split(1, number_of_stages);

  // schedule reduction tensor
  // [I, R/tma/stages, stages, tma/vect, vect]
  tv1r->split(1, tma_len);
  tv1r->split(1, number_of_stages);
  tv1r->split(-1, vect);
  auto reduction_tv_ref = tv1r->rFactor({1, 2, 4});
  std::cout << "reduction_tv_ref " << reduction_tv_ref->toString() << std::endl;
  std::cout << "dim0 " << dim0 << std::endl;
  std::cout << "prefetch_distance " << prefetch_distance << std::endl;

  // // Set inlineAt before applying circular buffer
  // inlineAllAt(tv1, /*pos=*/2);

  reduction_tv_ref->axis(0)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(reduction_tv_ref);

  // TMA load, [I0, I1/TMA/Stage, Stage, TMA]
  tv0s->axis(-1)->parallelize(ParallelType::Bulk);

  // TIDx for computation, [I, R/tma/stages, stages, tma/vect, vect]
  reduction_tv_ref->axis(-2)->parallelize(ParallelType::TIDx);

  // TIDx for computation, [I, TIDx]
  tv1r->axis(1)->parallelize(ParallelType::TIDx);

  fusion->print();

  // // Vectorize output
  // tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // Circular buffer
  inlineMost();
  // inlineAllAt(reduction_tv_ref, /*pos=*/2);
  tv0s->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  auto outputs = ke.run({at_tv0});
  auto at_output = at_tv0.sum(-1);
  testValidate(
      fusion.get(), outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}

//0.884 ms
TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBuffer) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  constexpr int dim0 = 16384, dim1 = 16384;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  int64_t tidx = 128;
  int64_t number_of_stages = 2;
  int64_t prefetch_distance = 0;
  CircularBufferType circular_buffer_type = Pipelined(true);

  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0s->setMemoryType(MemoryType::Shared);
  auto tv1r = tv1->cacheBefore();
  // tv0 --> tv0s --> tv1r ---> tv1

  // scheduler tma tensor
  // [I, R/tma/stages, stages, tma]
  int64_t tma_len = tidx;
  tv0s->split(1, tma_len);
  tv0s->split(1, number_of_stages);

  // schedule reduction tensor
  // [I, R/tma/stages, stages, tma]
  tv1r->split(1, tma_len);
  tv1r->split(1, number_of_stages);
  auto reduction_tv_ref = tv1r->rFactor({1, 2});

  // // Set inlineAt before applying circular buffer
  // inlineAllAt(tv1, /*pos=*/2);

  reduction_tv_ref->axis(0)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(reduction_tv_ref);

  // TMA load, [I0, I1/TMA/Stage, Stage, TMA]
  tv0s->axis(-1)->parallelize(ParallelType::Bulk);

  // TIDx for computation, [I, R/tma/stages, stages, tma]
  reduction_tv_ref->axis(-1)->parallelize(ParallelType::TIDx);

  // TIDx for computation, [I, TIDx]
  tv1r->axis(1)->parallelize(ParallelType::TIDx);

  fusion->print();

  // // Vectorize output
  // tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // Circular buffer
  inlineMost();
  // inlineAllAt(reduction_tv_ref, /*pos=*/2);
  tv0s->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  auto outputs = ke.run({at_tv0});
  auto at_output = at_tv0.sum(-1);
  testValidate(
      fusion.get(), outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}
} // namespace nvfuser
