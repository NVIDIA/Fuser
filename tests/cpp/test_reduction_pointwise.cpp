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
#include <runtime/executor.h>
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

namespace {
// Function to get the number of CUDA cores per SM.
// convert {major, minor} to hex number and check the map.
int getCoresPerSM(int major, int minor) {
  int sm_version = (major << 4) + minor;
  std::unordered_map<int, int> cores_per_sm_map = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60, 64},
      {0x61, 128},
      {0x62, 128},
      {0x70, 64},
      {0x72, 64},
      {0x75, 64},
      {0x80, 64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128}};
  auto it = cores_per_sm_map.find(sm_version);
  if (it != cores_per_sm_map.end()) {
    return it->second;
  }
  NVF_THROW("Unknown GPU architecture: ", major, ".", minor);
  return 128;
}
} // namespace
// Compute bandwidth flops ratio, return true if it's higher than
// the reference value of 0.07. It returns true for B100/200 and A100.
// Returns false for H100. The reference value is based on test of softmax,
// layer norm, and rms norm. Treating A100 as high bandwidth to flops ratio
// leads to better performance for softmax and dropout fused with layer norm
// or rms norm, but caused minor regressions for layer norm or rms norm alone.
bool isHighBandwidthFlopsRatio() {
  // A100-PCIe-80GB, 1.935e12 B/s, 1.95e13 flops, ratio = 0.0993
  // A100-SXM4-40GB, 1.555e12 B/s, 1.95e13 flops, ratio = 0.0798
  // H100-HBM3-80GB, 3.352e12 B/s, 6.69e13 flops, ratio = 0.0501
  constexpr float reference_ratio = 0.07f;
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // bandwidth
  float hardware_bandwidth = 2.f * (float)dev_prop->memoryBusWidth / 8.f *
      (float)dev_prop->memoryClockRate * 1000.f;
  // fp32 cuda core flops
  const int cuda_core_per_sm = getCoresPerSM(dev_prop->major, dev_prop->minor);
  const int flops_per_cycle = 2;
  float flops = (float)dev_prop->clockRate * 1000.f *
      (float)dev_prop->multiProcessorCount * (float)cuda_core_per_sm *
      (float)flops_per_cycle;
  std::cout << "flops " << flops << std::endl;
  std::cout << "hardware_bandwidth " << hardware_bandwidth << std::endl;
  float bandwidth_flops_ratio = hardware_bandwidth / flops;
  return bandwidth_flops_ratio > reference_ratio;
}

// 0.154 ms, tidx = 256, vect = 4, unroll = 4
TEST_F(InnerReductionTest, MagicScheduler) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int dim0 = 16384, dim1 = 16384;

  std::cout << isHighBandwidthFlopsRatio() << std::endl;

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({at_tv0});

  testValidate(fusion, cg_outputs, {at_tv0}, __LINE__, __FILE__);
}

// umbriel-b200-026
// IO bytes: 16384 x 16385 x 4 = 1.07 GB
// 0.154 ms, main branch, tidx = 256, vect = 4, unroll = 4
// 0.201 ms, stages 2, prefecch 1, tidx = 256

// 0.589 ms, CpAsyncBulkTensorTile, tidx = 256, stages = 2, pf = 1
// rs: BlockDim.x = 256, BlockDim.y = -1, BlockDim.z = -1, GridDim.x = 16384,
// GridDim.y = -1, GridDim.z = -1, Smem Size = 3104
TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBuffer) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  constexpr int dim0 = 16384, dim1 = 16384;

  std::cout << isHighBandwidthFlopsRatio() << std::endl;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  int64_t tidx = 256;
  int64_t number_of_stages = 2;
  int64_t prefetch_distance = 1;
  CircularBufferType circular_buffer_type = Pipelined(false);

  // why sometimes stuck if prefetch_distance = 1
  // ioctl(8, _IOC(_IOC_READ|_IOC_WRITE, 0x46, 0x2a, 0x20), 0x7fffb0420850) = 0
  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  // auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
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
  // inlineSelectedAt();
  inlineSelectedAt({tv0s}, reduction_tv_ref, /*pos=*/2);

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

TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBufferNorm) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  constexpr int dim0 = 16384, dim1 = 16384;

  std::cout << isHighBandwidthFlopsRatio() << std::endl;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = div(tv0, tv2);
  fusion->addOutput(tv3);

  int64_t tidx = 256;
  int64_t vect = 4;
  int64_t tma_len = tidx * vect;
  int64_t number_of_stages = 2;
  int64_t prefetch_distance = 1;
  CircularBufferType circular_buffer_type = Pipelined(false);

  // tma load from gmem to smem
  // [I, R/tma/stages, stages, tma]
  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  // auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0s->setMemoryType(MemoryType::Shared);
  tv0s->split(1, tma_len);
  tv0s->split(1, number_of_stages);

  // vect load from shared memory to register
  // [I, R/tma/stages, stages, tma/vect, vect]
  auto tv0r = tv0s->cacheAfter(LoadStoreOpType::Set);
  tv0r->split(1, tma_len);
  tv0r->split(1, number_of_stages);
  tv0r->split(-1, vect);

  // thread local reduction, pos-3 is left for block reduction
  auto reduction_tv_ref = tv0r->rFactor({1, 2, 4});

  // schedule reduction tensor
  // [I, R/tma/stages, stages, tma]
  auto tv1r = tv1->cacheBefore();
  tv1r->split(1, tma_len);
  tv1r->split(1, number_of_stages);

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
  // inlineSelectedAt();
  inlineSelectedAt({tv0s}, reduction_tv_ref, /*pos=*/2);

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

// bdimx = 512, batch = 16K / 512 / 4 = 8,
// 0.321 ms, bytes = 16K x 16K x 4 x 2 = 2.147 GB, 6.69 TB/s
// 82% SOL
TEST_F(InnerReductionTest, MagicNorm) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int dim0 = 16384, dim1 = 40*1024;

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = mul(tv0, tv2);
  fusion->addOutput(tv3);
  auto fusion_copy = *fusion;

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({at_tv0});

  // testValidate(&fusion_copy, cg_outputs, {at_tv0}, __LINE__, __FILE__);
}

// bdimx = 512, batch = 16K / 512 / 4 = 8,
// 0.321 ms, bytes = 16K x 16K x 4 x 2 = 2.147 GB, 6.69 TB/s
// 82% SOL
// no TMA, bdimx = 512, vect = 4, 0.321 ms, 82% SOL, occupancy = 50%
// nD-TMA, bdimx = 256, stages = 2, 0.370 ms, 71% SOL, occupancy = 12.5%
// nD-TMA, bdimx = 512, stages = 2, 0.383 ms, 68% SOL, occupancy = 25%
// nD-TMA, bdimx = 256, stages = 3, 0.355 ms, 74% SOL, occupancy = 12.5%

// 1D-TMA, bdimx = 256, stages = 2, tma length = 256, 0.385 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 512, 0.350 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 1024, 0.341 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 2048, 0.340 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 4096, 0.339 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 8192, 0.339 ms
// 1D-TMA, bdimx = 256, stages = 2, tma length = 16384, 0.339 ms

TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBufferNorm3) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int dim0 = 16384, dim1 = 16384;
  int64_t number_of_stages = 2, prefetch_distance = 1;
  constexpr int64_t bdimx = 256;
  constexpr int64_t examples_per_cta = 2;
  CircularBufferType circular_buffer_type = Pipelined();

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = mul(tv0, tv2);
  fusion->addOutput(tv3);
  auto fusion_copy = *fusion;

  // Load input from global to shared memory
  // auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0s->setMemoryType(MemoryType::Shared);

  // smem to register
  auto tv0r = tv0s->cacheAfter();

  // Store results in registers
  auto tv3r = tv3->cacheBefore();

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);

  TensorView* reference_tv = tv3r;

  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;

  // boxDim array must be non-zero and less than or equal to 256
  constexpr int64_t vectorize = 4;
  int64_t persistent_batch = dim1 / bdimx / vectorize;

  // Since multi-dim CpAsyncBulk has a size limit of 256 per dimension,
  // we require multiple TMA operations to load the entire example in shared
  // memory for pointwise kernel.
  //
  // Define TMA Box
  // logical domain: [I1, I2]
  tv0s->split(0, examples_per_cta);
  // split: [I0 / 4, 4, I2]
  tv0s->split(-1, 1024);
  // split: [I0/4, 4, I2/256, 256]

  // Schedule reference_tv
  // logical domain: [I1, I2]
  // split: [I1, I2/V (bdimx / tdx), V]
  reference_tv->split(-1, vectorize);
  // split: [I1, EPCT, I2/V/EPCT (tdx), V]
  reference_tv->split(-2, persistent_batch, /*inner_split=*/false);
  // split: [I1, EPCT, I2/V/EPCT (tdx), U, V]
  reference_tv->split(-2, 1);
  // reorder: [I1, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->reorder({{-4, -3}, {-3, -4}});
  // reorder: [I1/EPC, EPC, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->split(0, examples_per_cta);

  TransformPropagator propagator(reference_tv);
  std::vector<TensorView*> all_tvs_except_cache =
      ir_utils::allTvsExcept(fusion, {tv0s});
  SetSelector selector(
      {all_tvs_except_cache.begin(), all_tvs_except_cache.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&propagator);

  std::vector<TensorView*> rfactor_tvs;
  rfactor_tvs.reserve(reduction_tvs.size());
  std::transform(
      reduction_tvs.begin(),
      reduction_tvs.end(),
      std::back_inserter(rfactor_tvs),
      [](TensorView* tv) { return tv->rFactor({-3, -2, -1}); });

  // Define Parallelization Schema
  reference_tv->axis(0)->parallelize(ParallelType::BIDx);
  reference_tv->axis(2)->parallelize(ParallelType::TIDx);
  reference_tv->axis(-2)->parallelize(ParallelType::Unroll);
  scheduler_utils::parallelizeAllLike(reference_tv);

  // Vectorize reg-->gmem and smem-->reg
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0r->axis(-1)->parallelize(ParallelType::Vectorize);

  std::cout << "tv0r: " << tv0r->toString() << std::endl;
  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;

  // InlineMost automatically handles vectorize and tma dimensions
  inlineMost();

  // Handle TMA Tensor
  // Apply circular buffer after computeAt
  tv0s->axis(-1)->parallelize(ParallelType::Bulk);
  if (examples_per_cta > 1) {
    tv0s->circularBuffer(
        number_of_stages, prefetch_distance, circular_buffer_type);
  }

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  // Compile with KernelExecutor directly to avoid scheduling
  KernelExecutor ke;
  ke.compile(fusion, {at_tv0});
  std::vector<at::Tensor> cg_outputs = ke.run({at_tv0});

  // testValidate(&fusion_copy, cg_outputs, {at_tv0}, __LINE__, __FILE__);
}

TEST_F(InnerReductionTest, CpAsyncBulk1DCircularBufferNorm4) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int dim0 = 16384, dim1 = 16384;
  constexpr int64_t bdimx = 256;
  constexpr int64_t vectorize = 4;
  constexpr int64_t tma_length = bdimx * vectorize;
  int64_t persistent_batch = dim1 / bdimx / vectorize;
  int64_t number_of_stages = persistent_batch, prefetch_distance = 1;
  CircularBufferType circular_buffer_type = Pipelined();

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {-1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = mul(tv0, tv2);
  fusion->addOutput(tv3);
  auto fusion_copy = *fusion;

  // Load input from global to shared memory
  // auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  auto tv0s = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0s->setMemoryType(MemoryType::Shared);

  // smem to register
  auto tv0r = tv0s->cacheAfter();

  // Store results in registers
  auto tv3r = tv3->cacheBefore();

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion);

  TensorView* reference_tv = tv3r;

  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;

  // boxDim array must be non-zero and less than or equal to 256

  // Since multi-dim CpAsyncBulk has a size limit of 256 per dimension,
  // we require multiple TMA operations to load the entire example in shared
  // memory for pointwise kernel.
  //
  // TMA, [I, R/tma/stages, stages, tma]
  tv0s->split(1, tma_length);
  tv0s->split(1, number_of_stages);

  // Schedule reference_tv
  // TMA, [I, R/tma/stages, stages, tma/vect, vect]
  reference_tv->split(1, tma_length);
  reference_tv->split(1, number_of_stages);
  reference_tv->split(-1, vectorize);

  TransformPropagator propagator(reference_tv);
  std::vector<TensorView*> all_tvs_except_cache =
      ir_utils::allTvsExcept(fusion, {tv0s});
  SetSelector selector(
      {all_tvs_except_cache.begin(), all_tvs_except_cache.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&propagator);

  std::vector<TensorView*> rfactor_tvs;
  rfactor_tvs.reserve(reduction_tvs.size());
  std::transform(
      reduction_tvs.begin(),
      reduction_tvs.end(),
      std::back_inserter(rfactor_tvs),
      [](TensorView* tv) { return tv->rFactor({-1, -3}); });

  // Define Parallelization Schema
  reference_tv->axis(0)->parallelize(ParallelType::BIDx);
  reference_tv->axis(1)->parallelize(ParallelType::TIDy);
  reference_tv->axis(3)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(reference_tv);

  // Vectorize reg-->gmem and smem-->reg
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0r->axis(-1)->parallelize(ParallelType::Vectorize);

  std::cout << "tv0r: " << tv0r->toString() << std::endl;
  std::cout << "reference_tv: " << reference_tv->toString() << std::endl;

  // InlineMost automatically handles vectorize and tma dimensions
  inlineMost();

  // Handle TMA Tensor
  // Apply circular buffer after computeAt
  tv0s->axis(-1)->parallelize(ParallelType::Bulk);
  tv0s->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  // Compile with KernelExecutor directly to avoid scheduling
  KernelExecutor ke;
  ke.compile(fusion, {at_tv0});
  std::vector<at::Tensor> cg_outputs = ke.run({at_tv0});

  // testValidate(&fusion_copy, cg_outputs, {at_tv0}, __LINE__, __FILE__);
}
} // namespace nvfuser
