// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <string.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <exception>
#include <utility>

namespace nvfuser {

TEST_F(NVFuserTest, BarSyncWarpSpecializedPointwise) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t number_of_stages = 4;
  int64_t prefetch_distance = 1;
  int64_t tensor_outer_dim = 128;
  int64_t tensor_inner_dim = 1024;

  // with bdimx = 256, bdimy = 1,
  // after warp specialization, bdimy = 2,
  // kernel has 256 * 2 = 512 threads, each can use 128 registers.
  // With register sharing, adjust to [64, 192]
  constexpr int64_t bulk_inner_dim = 256;

  CircularBufferType circular_buffer_type = WarpSpecialized(ParallelType::TIDx);

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  TensorView* tv3 = exp(tv2);
  fusion->addOutput(tv3);

  tv2->setMemoryType(MemoryType::Shared);

  // Use TMA to load TV0 into shared memory
  TensorView* tv4 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* tv5 = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv5->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv3;

  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set computeAt position
  inlineAllAt(tv2, /*pos=*/2);

  // Circular Buffer with TMA loads
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::Bulk);
  tv4->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Load TV1 into shared memory
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::Bulk);
  tv5->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Split reference to parallelize TMA tile
  reference->split(-1, bulk_inner_dim);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t2 = at::exp(t0 + t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams());
  auto cg_outputs = ke.run({t0, t1});
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, RegisterSharingCircularBufferingPointwiseCustom) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t number_of_stages = 4;
  int64_t prefetch_distance = 1;
  int64_t tensor_outer_dim = 128;
  int64_t tensor_inner_dim = 1024;

  // with bdimx = 256, bdimy = 1,
  // after warp specialization, bdimy = 2,
  // kernel has 256 * 2 = 512 threads, each can use 128 registers.
  // With register sharing, adjust to [64, 192]
  constexpr int64_t bulk_inner_dim = 256;

  CircularBufferType circular_buffer_type =
      WarpSpecialized(ParallelType::TIDx, std::make_pair(64L, 192L));

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Use TMA to load TV0 into shared memory
  TensorView* tv3 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv3->setMemoryType(MemoryType::Shared);

  TensorView* tv4 = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv2;

  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set computeAt position
  inlineAllAt(tv2, /*pos=*/2);

  // Circular Buffer with TMA loads
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Load TV1 into shared memory
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::Bulk);
  tv4->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Split reference to parallelize TMA tile
  reference->split(-1, bulk_inner_dim);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t2 = t0 + t1;

  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("'setmaxnreg' ignored"), std::string::npos)
      << "'setmaxnreg' ignored!";

  auto cg_outputs = ke.run({t0, t1});
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, RegisterSharingCircularBufferingPointwiseNested) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t number_of_stages = 4;
  int64_t prefetch_distance = 1;
  int64_t tensor_outer_dim = 128;
  int64_t tensor_inner_dim = 1024;
  // with bdimx = 256, bdimy = 1,
  // after warp specialization, bdimy = 2,
  // kernel has 256 * 2 = 512 threads, each can use 128 registers.
  // With register sharing, adjust to [64, 192]
  constexpr int64_t bulk_inner_dim = 256;
  CircularBufferType circular_buffer_type =
      WarpSpecialized(ParallelType::TIDx, std::make_pair(64L, 192L));

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Use TMA to load TV0 into shared memory
  TensorView* tv3 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv3->setMemoryType(MemoryType::Shared);

  TensorView* tv4 = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv2;

  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set computeAt position
  inlineAllAt(tv2, /*pos=*/2);

  // Circular Buffer with TMA loads
  // tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Load TV1 into shared memory
  // tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::Bulk);
  tv4->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Split reference to parallelize TMA tile
  reference->split(-1, bulk_inner_dim);
  // reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t2 = t0 + t1;

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0, t1});
  } catch (const std::exception& e) {
    const char* reference =
        R"(When using register sharing with warp-specialized circular buffering, the circular buffer loop must be the outer-most for-loop.)";
    const char* str_match_pointer = strstr(e.what(), reference);
    ASSERT_TRUE(str_match_pointer != nullptr);
  }
}

using StageAndPrefetch = std::pair<int64_t, int64_t>;

class CircularBufferingTest : public NVFuserFixtureParamTest<StageAndPrefetch> {
 protected:
  int64_t number_of_stages = 1;
  int64_t prefetch_distance = 1;

  void SetUp() override {
    number_of_stages = std::get<0>(GetParam());
    prefetch_distance = std::get<1>(GetParam());
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_P(CircularBufferingTest, SingleDim1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  // I0
  tv3->split(-1, 128);
  // I0/128, 128
  tv3->split(-1, 32);
  // I0/128, 4, 32
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  // Outer for-loop is I0/128
  tv0->computeAt(tv3, 1);

  // Parallelize inner two dimensions
  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis 1, the axis_extent is I0/128.
  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_P(CircularBufferingTest, SingleDim2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // I0
  tv3->split(-1, 128);
  // I0/128, 128
  tv3->split(-1, 32);
  // I0/128, 4, 32
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  // Outer for-loop is I0/128
  tv0->computeAt(tv3, -1);

  // Parallelize inner two dimensions
  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis 1, the axis_extent is I0/128.
  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_P(CircularBufferingTest, SingleDim3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  // I0
  tv3->split(-1, 128);
  // I0/128, 128
  tv3->split(-1, 32);
  // I0/128, 4, 32
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // tv2 is invalid to circular-buffer as its producer, tv1, is
  // computed inside the circular-buffering loop.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv2->circularBuffer(number_of_stages, prefetch_distance));

  // Moving tv2 inner makes tv1 large enough to circular-buffer tv2
  tv2->computeAt(tv3, 2);

  tv2->circularBuffer(number_of_stages, prefetch_distance);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis 2, the axis_extent is 128/32.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 2;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// circular buffering smem to local and unswitch
TEST_P(CircularBufferingTest, SingleDimUnswitch1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  // I0
  tv3->split(-1, 128);
  // I0/128, 128
  tv3->split(-1, 32);
  // I0/128, 4, 32
  tv3->split(-1, 8);
  // I0/128, 4, 4, 8
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis -1 and axis 3 is parallelized with TIDx, the axis
  // extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 2;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// circular buffering gmem to shared and unswitch
TEST_P(CircularBufferingTest, SingleDimUnswitch2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  // I0
  tv2->split(-1, 128);
  // I0/128, 128
  tv2->split(-1, 32);
  // I0/128, 4, 32
  tv2->split(-1, 8);
  // I0/128, 4, 4, 8
  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv2);

  tv1->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis -1 and axis 3 is parallelized with TIDx, the axis
  // extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// circular buffering smem to local and unroll
TEST_P(CircularBufferingTest, SingleDimUnroll) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  // I0
  tv3->split(-1, 128);
  // I0/128, 128
  tv3->split(-1, 16);
  // I0/128, 8, 16
  tv3->split(-2, 4);
  // I0/128, 2, 4, 16
  tv3->split(-2, 2);
  // I0/128, 2, 2, 2, 16
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({199}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis -1 and axis 4 is parallelized with TIDx, the axis
  // extent is 2.
  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 2;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// circular buffering and vectorize
TEST_P(CircularBufferingTest, SingleDimVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  // I0
  tv2->split(-1, 128);
  // I0/128, 128
  tv2->split(-1, 4);
  // I0/128, 32, 4
  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->computeAt(tv2, 2);

  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({200}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis 2 and axis 1 is parallelized with TIDx, the axis
  // extent is I0/128.
  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Multiple tensors to circular-buffer
TEST_P(CircularBufferingTest, MultipleTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // I0
  tv4->split(0, 32);
  // I0/32, 32
  tv4->split(0, 4);
  // I0/32/4, 4, 32
  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->circularBuffer(number_of_stages, prefetch_distance);
  tv3->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({500}, options);
  auto t1 = at::randn({500}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});

  // Given computeAt axis 1, the axis extent is I0/32/4.
  constexpr int64_t axis_extent = 1;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0, t1});
  auto ref = t0 + t1;
  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Nested circular buffering from gmem to smem and smem to register
TEST_P(CircularBufferingTest, NestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto out = tv1;
  fusion.addOutput(out);

  auto tv2 = tv0->cacheAfter();
  auto tv3 = tv2->cacheAfter();

  // I0
  out->split(0, 32);
  // I0/32, 32
  out->split(0, 4);
  // I0/32/4, 4, 32
  TransformPropagatorWithCheck propagator(out);
  MaxLogicalDomainInfoSpanningTree(out).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Shared);

  tv2->computeAt(out, 1);
  tv3->computeAt(out, -1);

  out->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(out);

  tv2->circularBuffer(number_of_stages, prefetch_distance);
  tv3->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1001}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // Given computeAt axis 1 for tv2, the axis extent is I0/32/4 = 8.
  // Given computeAt axis 3 for tv3 and axis 3 is parallelized with TIDx,
  // the axis extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// FusionSmemBlockGemmCache + circular buffering at both smem and local
TEST_P(CircularBufferingTest, SmemBlockGemmCache) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(2); // (M, K)
  TensorView* tv1 = makeSymbolicTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  TensorView* tv6 = tv5->cacheBefore();

  // For smem circular buffering
  auto tv0_cache_local = tv0->cacheAfter();
  auto tv1_cache_local = tv1->cacheAfter();

  // For register circular buffering
  auto tv0_cache_smem = tv0->cacheAfter();
  auto tv1_cache_smem = tv1->cacheAfter();

  const int BSX = 32;
  const int TSX = 8;

  // [M, K, N]
  tv6->split(-1, BSX);
  tv6->split(-1, TSX);
  tv6->split(1, BSX);
  tv6->split(0, BSX);
  tv6->split(1, TSX);
  // [M/BSX, BSX/TSX, TSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 5}, {2, 4}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});
  // [M/BSX, N/BSX, K/BSX, BSX/TSX, BSX/TSX, TSX, TSX, BSX]

  auto tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxLogicalDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv6_rf->computeAt(tv6, -1);
  tv0_cache_local->computeAt(tv6_rf, -1);
  tv1_cache_local->computeAt(tv6_rf, -1);

  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-3)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  tv0_cache_local->circularBuffer(number_of_stages, prefetch_distance);
  tv1_cache_local->circularBuffer(number_of_stages, prefetch_distance);

  tv0_cache_smem->circularBuffer(number_of_stages, prefetch_distance);
  tv1_cache_smem->circularBuffer(number_of_stages, prefetch_distance);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = at::matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});

  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0, t1});
  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
  // The smem cache write in this test case is redundant predicated,
  //   and also circular buffered. Currently we are relying on WAR sync
  //   insertion to ensure ordering of circular buffered tensor access.
  // The check below makes sure that the sync is inserted so that the
  //   test isn't running on a race condition.
  NVF_CHECK(
      ke.compiledKernel()->kernel()->summary().war_hazard_syncs_count > 0);
}

// Vectorized reset test for circular buffered registers
TEST_P(CircularBufferingTest, Vector) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0});
  auto tv2c = tv2->cacheBefore();

  fusion.addOutput(tv2);

  auto tv1cw = tv1->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter();

  tv1cw->split(-1, 32);
  tv1cr->split(-1, 32);
  tv1cr->split(-1, 4);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1cw->computeAt(tv1cr, 1);
  tv0->computeAt(tv1cw, -1);
  tv2c->split(-1, 32);
  tv2c->split(-1, 4);
  tv1cr->computeAt(tv2c, 2);

  tv1cw->setMemoryType(MemoryType::Shared);
  tv1cr->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({200}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  auto ref = (t0 + 1).sum({0});
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: circular buffered
//   circular buffer case 1, both block sync and async wait
//  are needed.
TEST_P(CircularBufferingTest, CpAsync1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeContigConcreteTensor({m, n});
  TensorView* tv1 = makeContigConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 12);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // circular buffer the shared mem tensor.
  tv0_shared->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  KernelExecutor ke;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(ke.compile(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: circular buffered
//   circular buffer case 2, only async wait is needed
TEST_P(CircularBufferingTest, CpAsync2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // circular buffer the shared mem tensor.
  tv0_shared->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  KernelExecutor ke;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(ke.compile(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test for circular buffer in shared mem,
//  where we should not insert redundant syncs when
//  they are not needed.
TEST_P(CircularBufferingTest, NoSync) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // circular buffer the shared mem tensor.
  tv0_shared->circularBuffer(number_of_stages, prefetch_distance);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.run()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  NVF_ERROR(!sync_inserted, "Un-expected block sync inserted");

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

auto StagesAndPrefetches() {
  std::vector<StageAndPrefetch> values;
  for (int64_t i : {2, 5, 9}) {
    for (int64_t j : arange(-i, i)) {
      values.emplace_back(i, j);
    }
  }
  return testing::ValuesIn(values);
}

std::string nonTMAName(const testing::TestParamInfo<StageAndPrefetch>& info) {
  auto prefetch_distance = std::get<1>(info.param);
  std::string prefetch_distance_str;
  if (prefetch_distance < 0) {
    prefetch_distance_str = "neg" + std::to_string(-prefetch_distance);
  } else {
    prefetch_distance_str = std::to_string(prefetch_distance);
  }
  return "stage_" + std::to_string(info.param.first) + "_prefetch_" +
      prefetch_distance_str;
}

INSTANTIATE_TEST_SUITE_P(
    NonTma,
    CircularBufferingTest,
    StagesAndPrefetches(),
    nonTMAName);

using TmaCircularBufferingParams = std::tuple<
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    CircularBufferType,
    LoadStoreOpType>;

class TmaCircularBufferingTest
    : public NVFuserFixtureParamTest<TmaCircularBufferingParams> {
 protected:
  int64_t number_of_stages = 1;
  int64_t prefetch_distance = 1;
  int64_t tensor_outer_dim = 1;
  int64_t tensor_inner_dim = 1;
  CircularBufferType circular_buffer_type;
  LoadStoreOpType tma_load_type;

  void SetUp() override {
    number_of_stages = std::get<0>(GetParam());
    prefetch_distance = std::get<1>(GetParam());
    tensor_outer_dim = std::get<2>(GetParam());
    tensor_inner_dim = std::get<3>(GetParam());
    circular_buffer_type = std::get<4>(GetParam());
    tma_load_type = std::get<5>(GetParam());

    // NOTE: Multiple of 16 required for inner dimension
    NVF_ERROR(tensor_inner_dim % 16 == 0);
    NVFuserTest::SetUp();
  }

  bool testEnablesWarpSpecialization() {
    return std::holds_alternative<WarpSpecialized>(circular_buffer_type);
  }

  bool testEnablesRegisterSharing() {
    return std::holds_alternative<WarpSpecialized>(circular_buffer_type) &&
        std::get<WarpSpecialized>(circular_buffer_type)
            .num_registers.has_value();
  }

  bool testEnablesTIDx() {
    return testEnablesWarpSpecialization() &&
        std::get<WarpSpecialized>(circular_buffer_type).on ==
        ParallelType::TIDx;
  }

  bool testEnablesRegisterSharingTIDx() {
    return testEnablesRegisterSharing() &&
        std::get<WarpSpecialized>(circular_buffer_type).on ==
        ParallelType::TIDx;
  }

  bool testEnablesRegisterSharingTIDy() {
    return testEnablesRegisterSharing() &&
        std::get<WarpSpecialized>(circular_buffer_type).on ==
        ParallelType::TIDy;
  }

  bool invalidCTAShapeException(const std::exception& e) {
    const char* other_active_128_max =
        R"(The # active threads in other thread dimensions > 128 threads.)";
    if (std::get<WarpSpecialized>(circular_buffer_type).on ==
        ParallelType::TIDy) {
      const char* str_match_pointer = strstr(e.what(), other_active_128_max);
      return str_match_pointer != nullptr;
    }
    return false;
  }

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk
  // the memory range [srcMem, srcMem + size - 1] must not overflow the source
  // memory space. Otherwise, the behavior is undefined.
  std::optional<std::string> tma1dPredicate(int64_t bulk_inner_dim) {
    if (tma_load_type != LoadStoreOpType::CpAsyncBulk) {
      return std::nullopt;
    }
    if (tensor_inner_dim % bulk_inner_dim != 0) {
      return std::make_optional(
          "If split output domain is loaded with 1D TMA, the split must be "
          "divisible.");
    }
    if (!std::holds_alternative<WarpSpecialized>(circular_buffer_type)) {
      return std::make_optional(
          "1D TMA load can only be used with WarpSpecialized circular buffer.");
    }
    return std::nullopt;
  }

  template <typename data_type>
  void compare(int64_t tensor_dim, at::Tensor result, at::Tensor reference) {
    at::Tensor reference_cpu_data = reference.cpu();
    at::Tensor result_cpu_data = result.cpu();

    auto reference_cpu = reference_cpu_data.accessor<data_type, 1>();
    auto result_cpu = result_cpu_data.accessor<data_type, 1>();

    constexpr double abs_tolerance = 1e-3;
    constexpr double rel_tolerance = 1e-3;
    for (int64_t pos = 0; pos < tensor_dim; ++pos) {
      double tolerance =
          abs_tolerance + rel_tolerance * fabs((double)reference_cpu[pos]);
      if (fabs((double)result_cpu[pos] - (double)reference_cpu[pos]) >
          tolerance) {
        std::cout << "[" << pos << "] - result: " << result_cpu[pos]
                  << " | reference: " << reference_cpu[pos] << std::endl;
      }
    }
  }

  template <typename data_type>
  void compare(
      int64_t tensor_outer_dim,
      int64_t tensor_inner_dim,
      at::Tensor result,
      at::Tensor reference) {
    at::Tensor reference_cpu_data = reference.cpu();
    at::Tensor result_cpu_data = result.cpu();

    auto reference_cpu = reference_cpu_data.accessor<data_type, 2>();
    auto result_cpu = result_cpu_data.accessor<data_type, 2>();

    constexpr double abs_tolerance = 1e-3;
    constexpr double rel_tolerance = 1e-3;
    for (int64_t out_pos = 0; out_pos < tensor_outer_dim; ++out_pos) {
      for (int64_t in_pos = 0; in_pos < tensor_inner_dim; ++in_pos) {
        double tolerance = abs_tolerance +
            rel_tolerance * fabs((double)reference_cpu[out_pos][in_pos]);
        if (fabs(
                (double)reference_cpu[out_pos][in_pos] -
                (double)result_cpu[out_pos][in_pos]) > tolerance) {
          std::cout << "[" << out_pos << ", " << in_pos
                    << "] - result: " << result_cpu[out_pos][in_pos]
                    << " | ref: " << reference_cpu[out_pos][in_pos]
                    << std::endl;
        }
      }
    }
  }
};

TEST_F(NVFuserTest, ElectSyncCompatibility) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* input = makeContigTensor(3);
  fusion->addInput(input);
  TensorView* output = set(input);
  fusion->addOutput(output);

  TensorView* smem_cache =
      input->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  smem_cache->setMemoryType(MemoryType::Shared);

  // For TMA load, both the shared memory layout and the loop nest and
  // parallelization of TMA are specified by the consumer: smem_cache

  // Step 1: define TMA domain
  // Because we want to treat the entire tensor as 1D, we define the TMA
  // domain as [I0*I1*I2]
  smem_cache->merge(0);
  smem_cache->merge(0);
  // Note that the TMA domain only exist in people's mind, there is no need to
  // set anything here.

  // Step 2: define box
  smem_cache->split(0, 256);
  // [I0*I1*I2/256, 256]
  // partitioned IterDomain: I0*I1*I2
  // coordinate IterDomain: I0*I1*I2/256
  // box IterDomain: 256

  // Step 3: define tile
  // We use dense tile here, so tile == box. Nothing to do here.

  // Step 4: schedule the shared memory tensor
  // By default, the allocation domain is the logical domain, which is already
  // in good shape for this case.

  constexpr int64_t number_of_stages = 2;
  // Step 5: schedule the consumer tensor
  smem_cache->split(0, 4);
  // [I0*I1*I2/256/4, 4, 256]
  smem_cache->split(0, number_of_stages);
  // [I0*I1*I2/256/4/2, 2, 4, 256]

  // [BIDx, 2, TIDx, Bulk]
  smem_cache->axis(0)->parallelize(ParallelType::BIDx);
  smem_cache->axis(2)->parallelize(ParallelType::TIDx);
  smem_cache->axis(3)->parallelize(ParallelType::Bulk);

  // Schedule the smem->gmem part
  output->merge(0);
  output->merge(0);
  output->split(0, 256);
  output->split(0, 4);
  output->split(0, number_of_stages);
  output->axis(0)->parallelize(ParallelType::BIDx);
  output->axis(3)->parallelize(ParallelType::TIDx);

  inlineAllAt(output, /*pos=*/2);
  smem_cache->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  std::vector<int64_t> shape(3, 300);
  auto t0 = at::randn(shape, options);

  // IterDomain 2 for the TMA load is parallelized with TIDx, so we generate
  // (threadIdx.x < 4) predicate. This thread predicate is incompatible with
  // circular buffering because we generate an ElectSync predicate that uses
  // a single thread.
  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    const char* reference =
        R"(This thread-parallelized TensorView T2_s_float[ iblockIdx.x15{( ceilDiv(( ceilDiv(( ceilDiv(( ( ( (( (( getMetaData(T0) )).logical_size ))[0] ) * ( (( (( getMetaData(T0) )).logical_size ))[1] ) ) * ( (( (( getMetaData(T0) )).logical_size ))[2] ) ), 256) ), 4) ), 2) )}, iS16{2}, ithreadIdx.x14{4}, iB12{256} ] ca_pos( 2 ) is incorrectly contained within a If-Then-Else with the ElectSync predicate.)";
    const char* str_match_pointer = strstr(e.what(), reference);
    ASSERT_TRUE(str_match_pointer != nullptr);
  }
}

TEST_P(TmaCircularBufferingTest, SingleDim) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(tma_load_type);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  // Constants
  constexpr size_t bulk_inner_dim = 256;
  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }

  // [M] -> [M/bid, bid]
  reference->split(-1, bulk_inner_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set inlineAt before applying circular buffer
  inlineAllAt(tv1, /*pos=*/1);

  // Parallelization
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Circular Buffer with TMA loads
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_inner_dim}, options);
  at::Tensor t1 = at::exp(t0);

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t1);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, SingleDimUnroll) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(tma_load_type);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  // Constants
  constexpr size_t unroll_dim = 4;
  constexpr size_t bulk_inner_dim = 256;
  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }
  // [M] -> [M/bid, bid]
  reference->split(-1, bulk_inner_dim);
  // [M/bid, bid] -> [M/bid/unroll, unroll, bid]
  reference->split(0, unroll_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set ComputeAt position
  inlineAllAt(tv1, /*pos=*/1);

  // Apply Unroll
  tv1->axis(1)->parallelize(ParallelType::Unroll);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Circular Buffer with TMA loads
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_inner_dim}, options);
  at::Tensor t1 = at::exp(t0);

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  bool is_warp_specialized =
      std::holds_alternative<WarpSpecialized>(circular_buffer_type);
  int64_t axis_extent =
      ceilDiv(ceilDiv(tensor_inner_dim, bulk_inner_dim), unroll_dim);
  if (axis_extent < number_of_stages && !is_warp_specialized) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t1);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, SingleDimUnswitch) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(tma_load_type);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  // Constants
  constexpr size_t unroll_dim = 4;
  constexpr size_t bulk_inner_dim = 256;
  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }
  // [M] -> [M/bid, bid]
  reference->split(-1, bulk_inner_dim);
  // [M/bid, bid] -> [M/bid/unroll, unroll, bid]
  reference->split(0, unroll_dim);

  // Propagate Transformations
  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set ComputeAt position
  inlineAllAt(tv1, /*pos=*/1);

  // Apply Unswitch
  tv1->axis(1)->parallelize(ParallelType::Unswitch);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Circular Buffer with TMA loads
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_inner_dim}, options);
  at::Tensor t1 = at::exp(t0);

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  bool is_warp_specialized =
      std::holds_alternative<WarpSpecialized>(circular_buffer_type);
  int64_t axis_extent =
      ceilDiv(ceilDiv(tensor_inner_dim, bulk_inner_dim), unroll_dim);
  if (axis_extent < number_of_stages && !is_warp_specialized) {
    ASSERT_ANY_THROW(ke.run({t0}));
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t1);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, MultiDim) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  if (tma_load_type == LoadStoreOpType::CpAsyncBulk) {
    GTEST_SKIP() << "LoadStoreOpType::CpAsyncBulk only supports 1D TMA";
    return;
  }

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  TensorView* tv1 = exp(tv0);
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  // Constants
  constexpr int64_t tma_outer_dim = 4;
  constexpr int64_t tma_inner_dim = 256;

  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, tma_inner_dim);
  // [M, N/bid, bid] -> [M/bod, bod, N/bid, bid]
  reference->split(0, tma_outer_dim);
  // [M/bod, bod, N/bid, bid] -> [M/bod, N/bid, bod, bid]
  reference->reorder({{-2, -3}});

  // Propagate TMA transform
  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Apply inlineAt for TMA cache
  inlineAllAt(tv1, /*pos=*/2);

  // Merge TMA tile and Parallelize
  // [M/bod, N/bid, bod, bid] -> [M/bod, N/bid, bod * bid]
  reference->merge(-2, -1);
  // [M/bod, N/bid, bod * bid] -> [M/bod, N/bid, (bod * bid) / 256, 256]
  reference->split(-1, 256);

  // Parallelize
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  // Circular Buffer with TMA loads
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->axis(-2)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::ones({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::exp(t0);

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(
      tensor_outer_dim, tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t1);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, Pointwise) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Use TMA to load TV0 into shared memory
  TensorView* tv3 = tv0->cacheAfter(tma_load_type);
  tv3->setMemoryType(MemoryType::Shared);

  // __syncthreads() is mssing if mixing TMA and non-TMA loading with circular
  TensorView* tv4 = tv1->cacheAfter(tma_load_type);
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv2;

  // Constants
  constexpr int64_t bulk_inner_dim = 256;
  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }
  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set computeAt position
  inlineAllAt(tv2, /*pos=*/2);

  // Circular Buffer with TMA loads
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::Bulk);
  tv4->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Split reference to parallelize TMA tile
  reference->split(-1, bulk_inner_dim);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t2 = t0 + t1;

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0, t1});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0, t1});
  compare<float>(
      tensor_outer_dim, tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t2);
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, PointwiseCpAsync) {
  GTEST_SKIP() << "Needs shared memory predicate, but current "
                  "needsSharedMemoryPredicate() returns false";

  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // Use TMA to load TV0 into shared memory
  TensorView* tv3 = tv0->cacheAfter(tma_load_type);
  tv3->setMemoryType(MemoryType::Shared);

  // Load TV1 into shared memory
  TensorView* tv4 = tv1->cacheAfter(LoadStoreOpType::CpAsync);
  tv4->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv2;

  // Constants
  constexpr int64_t bulk_inner_dim = 256;
  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }
  // [M, N] -> [M, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Set computeAt position
  inlineAllAt(tv2, /*pos=*/2);

  // Circular Buffer with TMA loads
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(2)->parallelize(ParallelType::Bulk);
  tv3->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  // Circular Buffer with set operation
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  // TODO Disable circular buffering for CpAsync
  // Circular buffering handles cpAsync sync logic separate from cloner logic.
  // tv4->circularBuffer(number_of_stages, prefetch_distance);

  // Split reference to parallelize TMA tile
  reference->split(-1, bulk_inner_dim);
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t2 = t0 + t1;

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0, t1});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0, t1});
  compare<float>(
      tensor_outer_dim, tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t2);
  testValidate(fusion.get(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, InnerReduction) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  TensorView* tv1 = sum(tv0, {-1});
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(tma_load_type);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  constexpr int64_t examples_per_cta = 4;
  constexpr int64_t bulk_inner_dim = 256;

  if (auto msg = tma1dPredicate(bulk_inner_dim)) {
    GTEST_SKIP() << msg.value();
    return;
  }

  // [M, N] -> [M/epc, epc, N]
  reference->split(0, examples_per_cta);
  // [M/epc, epc, N] -> [M/epc, epc, N/bid, bid]
  reference->split(-1, bulk_inner_dim);

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // [M/epc, epc, N/bid, bid] -> [M/epc, epc, N]
  reference->merge(-2, -1);
  // [M/epc, epc, N] -> [M/epc, epc, N/tdx, tdx]
  constexpr int64_t tdx = 256;
  reference->split(-1, tdx);

  // Parallelize
  reference->axis(0)->parallelize(ParallelType::BIDx);

  // Use block reduce.
  reference->axis(-1)->parallelize(ParallelType::TIDx);

  // InlineMost automatically handles vectorize and tma dimensions
  inlineMost();

  // Circular Buffer with TMA loads
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = sum(t0, {-1});

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(tensor_outer_dim, cg_outputs[0].as<at::Tensor>(), t1);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, OuterReduction) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  TensorView* tv1 = sum(tv0, {0});
  fusion->addOutput(tv1);

  TensorView* tv2 = tv0->cacheAfter(tma_load_type);
  tv2->setMemoryType(MemoryType::Shared);

  TensorView* reference = tv1;

  constexpr int64_t tile_size = 256;
  if (auto msg = tma1dPredicate(tile_size)) {
    GTEST_SKIP() << msg.value();
    return;
  }

  // [M, N] -> [M, N/bid, bid]
  reference->split(1, tile_size);
  // [M, N/bid, bid] -> [N/bid, M, bid]
  reference->reorder({{1, 0}});

  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Parallelize
  reference->axis(0)->parallelize(ParallelType::BIDx);
  reference->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  inlineMost();

  // Circular Buffer with TMA loads
  tv2->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor t1 = sum(t0, {0});

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    if (!invalidCTAShapeException(e)) {
      throw;
    }
    return;
  }

  auto cg_outputs = ke.run({t0});
  compare<float>(tensor_inner_dim, cg_outputs[0].as<at::Tensor>(), t1);
  // Please note that, serial reduction has larger error than parallel reduction
  // This is the nature of the algorithm, not a bug in the implementation.
  EXPECT_EQ(at::allclose(cg_outputs[0].as<at::Tensor>(), t1, 1e-3, 1e-3), true);
}

TEST_P(TmaCircularBufferingTest, Persistent) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  if (testEnablesWarpSpecialization()) {
    GTEST_SKIP() << "Bdimx is dynamic, Warp Specialization is disabled.";
    return;
  }

  constexpr at::ScalarType dtype = at::ScalarType::Float;
  constexpr int64_t correction = 0;
  constexpr int64_t reduction_axis = 1;
  constexpr bool keepdim = true;

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(x);

  // Algorithm:
  // x_norm = (x - x_mean) / sqrt(x_var)
  Val* num_elem = x->getLoopDomain().at(reduction_axis)->extent();

  TensorView* sum_x = sum(x, {reduction_axis}, /*keepdim=*/false);
  TensorView* mean_x = div(sum_x, num_elem);
  TensorView* bcast_mean = broadcast(mean_x, {false, true});

  TensorView* x_mean_sub = sub(x, bcast_mean);
  TensorView* x_mean_sub_sq = mul(x_mean_sub, x_mean_sub);
  TensorView* sum_x_mean_sub_sq =
      sum(x_mean_sub_sq, {reduction_axis}, /*keepdim=*/false);
  TensorView* var_x = div(sum_x_mean_sub_sq, num_elem);
  TensorView* bcast_var = broadcast(var_x, {false, true});

  TensorView* x_norm = div(sub(x, bcast_mean), sqrt(bcast_var));
  fusion->addOutput(x_norm);

  // Load input from global to shared memory
  TensorView* x_cache_smem = x->cacheAfter(tma_load_type);
  x_cache_smem->setMemoryType(MemoryType::Shared);

  // Load input from shared memory to registers
  x_cache_smem->cacheAfter();

  // Store results in registers
  x_norm->cacheBefore();

  std::vector<TensorView*> reduction_tvs =
      scheduler_utils::getReductionTvs(fusion.get());

  TensorView* reference_tv = x_norm;

  // boxDim array must be non-zero and less than or equal to 256
  constexpr int64_t width = 256;
  constexpr int64_t vectorize = 4;
  int64_t elem_per_compute_thread = tensor_inner_dim / width / vectorize;
  constexpr int64_t examples_per_cta = 4;
  constexpr int64_t tile_size = 256;
  if (auto msg = tma1dPredicate(tile_size)) {
    GTEST_SKIP() << msg.value();
    return;
  }
  // Since multi-dim CpAsyncBulk has a size limit of 256 per dimension,
  // we require multiple TMA operations to load the entire example in shared
  // memory for pointwise kernel.
  //
  // Define TMA Box
  // logical domain: [I1, I2]
  x_cache_smem->split(0, examples_per_cta);
  // split: [I0 / 4, 4, I2]
  x_cache_smem->split(-1, tile_size);
  // split: [I0/4, 4, I2/256, 256]

  // Schedule reference_tv
  //   logical domain: [I1, I2]
  //         split: [I1, I2/V (width / tdx), V]
  reference_tv->split(-1, vectorize);
  //         split: [I1, EPCT, I2/V/EPCT (tdx), V]
  reference_tv->split(-2, elem_per_compute_thread, /*inner_split=*/false);
  //         split: [I1, EPCT, I2/V/EPCT (tdx), U, V]
  reference_tv->split(-2, 1);
  //         reorder: [I1, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->reorder({{-4, -3}, {-3, -4}});
  //         reorder: [I1/EPC, EPC, I2/V/EPCT (tdx), EPCT, U, V]
  reference_tv->split(0, examples_per_cta);

  TransformPropagator propagator(reference_tv);
  std::vector<TensorView*> all_tvs_except_cache =
      ir_utils::allTvsExcept(fusion.get(), {x_cache_smem});
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

  // Vectorize Cache
  reference_tv->axis(-1)->parallelize(ParallelType::Vectorize);

  // InlineMost automatically handles vectorize and tma dimensions
  inlineMost();

  // Handle TMA Tensor
  // Apply circular buffer after computeAt
  x_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);
  if (examples_per_cta > 1) {
    x_cache_smem->circularBuffer(
        number_of_stages, prefetch_distance, circular_buffer_type);
  }

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);
  at::Tensor at_tv1 = at::randn({tensor_outer_dim, tensor_inner_dim}, options);

  // Compile with KernelExecutor directly to avoid scheduling
  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0});
  auto cg_outputs = ke.run({at_tv0});

  std::tuple<at::Tensor, at::Tensor> at_var_mean =
      at::var_mean(at_tv0, {-1}, correction, keepdim);
  at::Tensor at_var = std::get<0>(at_var_mean);
  at::Tensor at_mean = std::get<1>(at_var_mean);
  at::Tensor at_output = (at_tv0 - at_mean) / sqrt(at_var);

  testValidate(
      fusion.get(), cg_outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, Matmul) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  if (testEnablesTIDx()) {
    GTEST_SKIP() << "Warp Specialization with TIDx used for both computation "
                    "and load, requires TIDx to be a multiple of 128.";
    return;
  }

  // There are 512 compute threads and 128 load threads
  // register at entry: 96
  // register for compute: 96 + 32 / 4 = 104
  // register for loading: 96 - 32 = 64
  if (testEnablesRegisterSharingTIDy()) {
    circular_buffer_type =
        WarpSpecialized(ParallelType::TIDy, std::make_pair(64, 104));
  }

  if (tma_load_type == LoadStoreOpType::CpAsyncBulk) {
    GTEST_SKIP() << "LoadStoreOpType::CpAsyncBulk only supports 1D TMA";
    return;
  }

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Algorithm
  TensorView* tv0 = makeContigTensor(2); // (M, K)
  TensorView* tv1 = makeContigTensor(2); // (K, N)
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion->addOutput(tv5);

  // CpAsyncBulk Store
  TensorView* tv6 = tv5->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv6->setMemoryType(MemoryType::Shared);

  // For register circular buffering
  TensorView* tv0_cache_local = tv0->cacheAfter();
  TensorView* tv1_cache_local = tv1->cacheAfter();

  // For shared memory circular buffering
  TensorView* tv0_cache_smem =
      tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  TensorView* tv1_cache_smem =
      tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  constexpr int64_t BSX = 64;
  constexpr int64_t TSX = 32;
  constexpr int64_t TSY = 16;

  // Step 0: [M, K, N]
  // Step 1: [M, K, N/BSX, BSX]
  tv6->split(-1, BSX);

  // Step 2: [M, K, N/BSX, BSX/TSX, TSX]
  tv6->split(-1, TSX);

  // Step 3: [M, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->split(1, BSX);

  // Step 4: [M/BSX, BSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->split(0, BSX);

  // Step 5:[M/BSX, BSX/TSY, TSY, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->split(1, TSY);

  // Step 6: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 4}, {2, 5}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});

  // Step 7a: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX (reduce)]
  // Step 7b: [M/BSX, N/BSX, K/BSX (reduce), BSX/TSY, BSX/TSX, TSY, TSX]
  TensorView* tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxLogicalDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  // Parallelize
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-2)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  // (BSX/TSX * TSX * BSX) = 1024 floats = 4096 bytes * (number of buffers)
  tv0_cache_smem->axis(-3)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-2)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  // (BSX/TSY * TSY * BSX) = 1024 floats = 4096 bytes * (number of buffers)
  tv1_cache_smem->axis(-3)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-2)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  // Apply ParallelType::Bulk to global output tensor.
  tv5->axis(-4)->parallelize(ParallelType::Bulk);
  tv5->axis(-3)->parallelize(ParallelType::Bulk);
  tv5->axis(-2)->parallelize(ParallelType::Bulk);
  tv5->axis(-1)->parallelize(ParallelType::Bulk);

  // IterDomain: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX]
  // Parallelization: BDX, BDY, K/BSX ||, BSX/TSY, BSX/TSX, TSY, TSX, TDX]
  // 4 non-parallelized for-loops
  inlineMost();

  // Apply circular buffering after setting computeAt position
  tv0_cache_local->circularBuffer(number_of_stages, prefetch_distance);
  tv1_cache_local->circularBuffer(number_of_stages, prefetch_distance);

  tv0_cache_smem->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);
  tv1_cache_smem->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  constexpr int64_t K = 1024;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, K}, options);
  at::Tensor t1 = at::randn({K, tensor_inner_dim}, options);
  at::Tensor aten_output =
      (t0.unsqueeze(/*dim=*/-1) * t1.unsqueeze(/*dim=*/0)).sum(/*dim=*/1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});

  auto cg_outputs = ke.run({t0, t1});
  compare<float>(
      tensor_outer_dim,
      tensor_inner_dim,
      cg_outputs[0].as<at::Tensor>(),
      aten_output);
  testValidate(
      fusion.get(), cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_P(TmaCircularBufferingTest, MatmulWithBroadcastedInput) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);

  if (testEnablesTIDx()) {
    GTEST_SKIP() << "Warp Specialization with TIDx used for both computation "
                    "and load, requires TIDx to be a multiple of 128.";
    return;
  }

  // There are 512 compute threads and 128 load threads
  // register at entry: 96
  // register for compute: 96 + 32 / 4 = 104
  // register for loading: 96 - 32 = 64
  if (testEnablesRegisterSharingTIDy()) {
    circular_buffer_type =
        WarpSpecialized(ParallelType::TIDy, std::make_pair(64, 104));
  }

  if (tma_load_type == LoadStoreOpType::CpAsyncBulk) {
    GTEST_SKIP() << "LoadStoreOpType::CpAsyncBulk only supports 1D TMA";
    return;
  }

  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Algorithm
  TensorView* tv0 = makeContigConcreteTensor({-1, -1, 1}); // (M, K, B)
  TensorView* tv1 = makeContigConcreteTensor({1, -1, -1}); // (B, K, N)
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = mul(tv0, tv1); // M, K, N
  TensorView* tv3 = sum(tv2, {1}); // M, R, N
  fusion->addOutput(tv3);

  // CpAsyncBulk Store
  TensorView* tv4 = tv3->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv4->setMemoryType(MemoryType::Shared);

  // For register circular buffering
  TensorView* tv0_cache_local = tv0->cacheAfter();
  TensorView* tv1_cache_local = tv1->cacheAfter();

  // For shared memory circular buffering
  TensorView* tv0_cache_smem =
      tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  TensorView* tv1_cache_smem =
      tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  constexpr int64_t BSX = 64;
  constexpr int64_t TSX = 32;
  constexpr int64_t TSY = 16;

  // Step 0: [M, K, N]
  // Step 1: [M, K, N/BSX, BSX]
  tv4->split(-1, BSX);

  // Step 2: [M, K, N/BSX, BSX/TSX, TSX]
  tv4->split(-1, TSX);

  // Step 3: [M, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv4->split(1, BSX);

  // Step 4: [M/BSX, BSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv4->split(0, BSX);

  // Step 5:[M/BSX, BSX/TSY, TSY, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv4->split(1, TSY);

  // Step 6: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX]
  tv4->reorder(
      {{4, 7}, {7, 6}, {6, 4}, {2, 5}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});

  // Step 7a: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX (reduce)]
  // Step 7b: [M/BSX, N/BSX, K/BSX (reduce), BSX/TSY, BSX/TSX, TSY, TSX]
  TensorView* tv4_rf = tv4->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv4_rf);
  MaxLogicalDomainInfoSpanningTree(tv4_rf).traverse(&propagator);

  // Parallelize
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::BIDy);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv3);

  // (BSX/TSX * TSX * BSX) = 1024 floats = 4096 bytes * (number of buffers)
  tv0_cache_smem->axis(-5)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-3)->parallelize(ParallelType::Bulk);
  tv0_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  // (BSX/TSY * TSY * BSX) = 1024 floats = 4096 bytes * (number of buffers)
  tv1_cache_smem->axis(-4)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-2)->parallelize(ParallelType::Bulk);
  tv1_cache_smem->axis(-1)->parallelize(ParallelType::Bulk);

  // Apply ParallelType::Bulk to global output tensor.
  tv3->axis(-4)->parallelize(ParallelType::Bulk);
  tv3->axis(-3)->parallelize(ParallelType::Bulk);
  tv3->axis(-2)->parallelize(ParallelType::Bulk);
  tv3->axis(-1)->parallelize(ParallelType::Bulk);

  // IterDomain: [M/BSX, N/BSX, K/BSX, BSX/TSY, BSX/TSX, TSY, TSX, BSX]
  // Parallelization: BDX, BDY, K/BSX ||, BSX/TSY, BSX/TSX, TSY, TSX, TDX]
  // 4 non-parallelized for-loops
  inlineMost();

  // Apply circular buffering after setting computeAt position
  tv0_cache_local->circularBuffer(number_of_stages, prefetch_distance);
  tv1_cache_local->circularBuffer(number_of_stages, prefetch_distance);

  tv0_cache_smem->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);
  tv1_cache_smem->circularBuffer(
      number_of_stages, prefetch_distance, circular_buffer_type);

  constexpr int64_t K = 1024;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({tensor_outer_dim, K, 1}, options);
  at::Tensor t1 = at::randn({1, K, tensor_inner_dim}, options);
  at::Tensor aten_output = (t0 * t1).sum(/*dim=*/1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});

  auto cg_outputs = ke.run({t0, t1});
  compare<float>(
      tensor_outer_dim,
      tensor_inner_dim,
      cg_outputs[0].as<at::Tensor>(),
      aten_output);
  testValidate(
      fusion.get(), cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

auto tmaCircularBufferingParams() {
  // A very simple PRNG:
  // https://en.wikipedia.org/wiki/Lehmer_random_number_generator
  uint32_t lcg_parkmiller = 1;
  const std::vector<CircularBufferType> all_types{
      Pipelined(false),
      Pipelined(true),
      WarpSpecialized(ParallelType::TIDx),
      WarpSpecialized(ParallelType::TIDy),
      WarpSpecialized(ParallelType::TIDx, std::make_pair(64, 168)),
      WarpSpecialized(ParallelType::TIDy, std::make_pair(64, 168))};
  const std::vector<LoadStoreOpType> tma_types{
      LoadStoreOpType::CpAsyncBulk, LoadStoreOpType::CpAsyncBulkTensorTile};
  std::vector<TmaCircularBufferingParams> values;
  for (int64_t i : {2, 4}) {
    for (int64_t j : arange(-i, i)) {
      for (int64_t m : {128, 500, 1024}) {
        for (int64_t n : {1024, 2048}) {
          for (auto tma_load_type : tma_types) {
            values.emplace_back(
                i,
                j,
                m,
                n,
                all_types[lcg_parkmiller % all_types.size()],
                tma_load_type);
            lcg_parkmiller = (uint64_t)lcg_parkmiller * 48271 % 0x7fffffff;
          }
        }
      }
    }
  }
  return testing::ValuesIn(values);
}

std::string tmaName(
    const testing::TestParamInfo<TmaCircularBufferingParams>& info) {
  auto prefetch_distance = std::get<1>(info.param);
  std::string prefetch_distance_str;
  if (prefetch_distance < 0) {
    prefetch_distance_str = "neg" + std::to_string(-prefetch_distance);
  } else {
    prefetch_distance_str = std::to_string(prefetch_distance);
  }
  std::stringstream ss;
  ss << "stage_" << std::to_string(std::get<0>(info.param)) << "_prefetch_"
     << prefetch_distance_str << "_M_"
     << std::to_string(std::get<2>(info.param)) << "_N_"
     << std::to_string(std::get<3>(info.param)) << "_"
     << std::get<4>(info.param) << "_" << std::get<5>(info.param);
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    Hopper,
    TmaCircularBufferingTest,
    tmaCircularBufferingParams(),
    tmaName);

namespace {

std::pair<int64_t, int64_t> getNumRegisters(
    int64_t n_computation_threads,
    int64_t n_tma_branch_threads,
    int64_t n_total_threads) {
  // adjust register usage, assuming computation threads increase register
  // usage by 8, then each tma branch threads should reduce by:
  // 8 * n_computation / n_tma_branch_threads
  int64_t initial_reg_count = getRegPerThreadGivenThreadsPerSM(n_total_threads);
  EXPECT_TRUE(initial_reg_count % 8 == 0 || initial_reg_count == 255);
  int64_t compute_reg_count = initial_reg_count + 8;
  int64_t compute_tma_n_threads_ratio =
      n_computation_threads / n_tma_branch_threads;
  int64_t tma_reg_count = initial_reg_count - compute_tma_n_threads_ratio * 8;
  return std::make_pair(tma_reg_count, compute_reg_count);
}

int64_t getOtherActiveThreads(ParallelType ws_pt, dim3 bdim) {
  if (ws_pt == ParallelType::TIDx) {
    return bdim.y * bdim.z;
  } else if (ws_pt == ParallelType::TIDy) {
    return bdim.x * bdim.z;
  } else if (ws_pt == ParallelType::TIDz) {
    return bdim.x * bdim.y;
  } else {
    NVF_THROW("TMA register sharing only supports TIDx, TIDy, and TIDz");
  }
}

// warp specialization with register sharing requires
// all threads in the same warp group execute the same
// register adjustment instruction. So the number of padded
// threads for TMA loading branch depends on CTA shape &
// warp specialization dimension.
// index = TIDx + TIDy * bdimx + TIDz * bdimx * bdimy
// total = bdimx * bdimy * bdimz
// Pad on x: bdimx += 128 / (bdimy * bdimz)
// Pad on y: bdimy += 128 / (bdimx * bdimz)
// Pad on z: bdimz += 128 / (bdimx * bdimy)
int64_t getTmaPadThreads(ParallelType ws_pt, dim3 bdim) {
  return scheduler_utils::safeDiv(128, getOtherActiveThreads(ws_pt, bdim));
}

} // namespace

TEST_F(NVFuserTest, TmaRegisterSharingDynamicShapesExpectFail) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  int64_t gdimx = 2;
  dim3 bdim = dim3(32, 4, 2);
  ParallelType ws_pt = ParallelType::TIDx;
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  auto tv2 = mul(tv1, tv1);
  fusion->addOutput(tv2);

  // [I1, I2] -> [gdimx, I1/gdimx, I2/bdimx/bdimy, bdimy, bdimx]
  tv2->split(0, gdimx, false);
  tv2->split(2, bdim.x);
  tv2->split(2, bdim.y);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-3)->parallelize(ParallelType::TIDz);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  // [I1, I2] -> [gdimx, I1/gdimx, I2]
  tv1->split(0, gdimx, false);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::Bulk);

  // Set inlineAt before applying circular buffer
  inlineAllAt(tv1, /*pos=*/2);

  int64_t n_computation_threads = bdim.x * bdim.y * bdim.z;
  constexpr int64_t n_tma_branch_threads = 128;
  int64_t n_total_threads = n_computation_threads + n_tma_branch_threads;

  CircularBufferType circular_buffer_type = WarpSpecialized(
      ws_pt,
      getNumRegisters(
          n_computation_threads, n_tma_branch_threads, n_total_threads));
  int64_t n_stages = 2;
  tv1->circularBuffer(n_stages, /*prefetch_distance=*/1, circular_buffer_type);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({n_stages * gdimx, n_computation_threads}, options);
  at::Tensor t1 = t0 * t0;
  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    const char* reference =
        R"(Detected dynamic size for parallel type threadIdx.z in warp specialization kernel.)";
    const char* str_match_pointer = strstr(e.what(), reference);
    ASSERT_TRUE(str_match_pointer != nullptr);
    return;
  }
  FAIL() << "Expected exception during compilation";
}

using RegisterSharingParams = std::tuple<dim3, ParallelType>;
using TmaRegisterSharing = NVFuserFixtureParamTest<RegisterSharingParams>;
TEST_P(TmaRegisterSharing, CtaShapeShmoo) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  int64_t gdimx = 2;
  auto [bdim, ws_pt] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);
  auto tv2 = mul(tv1, tv1);

  fusion->addOutput(tv2);

  // [I1, I2] -> [gdimx, I1/gdimx, I2/bdimx/bdimy, I2/bdimx/bdimy/bdimz, bdimz,
  // bdimy, bdimx]
  tv2->split(0, gdimx, false);
  tv2->split(2, bdim.x);
  tv2->split(2, bdim.y);
  tv2->split(2, bdim.z);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-3)->parallelize(ParallelType::TIDz);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  // [I1, I2] -> [gdimx, I1/gdimx, I2]
  tv1->split(0, gdimx, false);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::Bulk);

  // Set inlineAt before applying circular buffer
  inlineAllAt(tv1, /*pos=*/2);

  int64_t n_computation_threads = bdim.x * bdim.y * bdim.z;
  constexpr int64_t n_tma_branch_threads = 128;
  int64_t n_total_threads = n_computation_threads + n_tma_branch_threads;

  constexpr int64_t n_stages = 2;

  // If ws_pt == ParallelType::TIDx and bdim.x == 32, CUDA kernel cannot use
  // register sharing. ncu reports it uses 26 register per thread.
  // getNumRegisters expects 168 registers by default, so the register settings
  // causes nvrtc to hang during compilation.
  if (ws_pt == ParallelType::TIDx && getTmaPadThreads(ws_pt, bdim) < 32) {
    CircularBufferType circular_buffer_type = WarpSpecialized(ws_pt);
    tv1->circularBuffer(
        n_stages, /*prefetch_distance=*/1, circular_buffer_type);
  } else {
    CircularBufferType circular_buffer_type = WarpSpecialized(
        ws_pt,
        getNumRegisters(
            n_computation_threads, n_tma_branch_threads, n_total_threads));
    tv1->circularBuffer(
        n_stages, /*prefetch_distance=*/1, circular_buffer_type);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({n_stages * gdimx, n_computation_threads}, options);
  at::Tensor t1 = t0 * t0;
  KernelExecutor ke;

  try {
    ke.compile(fusion.get(), {t0});
  } catch (const std::exception& e) {
    const char* other_active_128_max =
        R"(The # active threads in other thread dimensions > 128 threads.)";
    if (getOtherActiveThreads(ws_pt, bdim) > n_tma_branch_threads) {
      const char* str_match_pointer = strstr(e.what(), other_active_128_max);
      ASSERT_TRUE(str_match_pointer != nullptr);
      return;
    }
    throw;
  }

  auto cg_outputs = ke.run({t0});
  auto lparams = ke.lastLaunchParams();
  EXPECT_EQ(lparams.nThreads(), n_total_threads);
  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    Hopper,
    TmaRegisterSharing,
    ::testing::Combine(
        ::testing::Values(dim3(32, 4, 2), dim3(128, 2, 1), dim3(256, 1, 1)),
        ::testing::Values(
            ParallelType::TIDx,
            ParallelType::TIDy,
            ParallelType::TIDz)),
    [](const testing::TestParamInfo<RegisterSharingParams>& info) {
      std::stringstream ss;
      ss << "cta_" << std::get<0>(info.param).x;
      ss << "_" << std::get<0>(info.param).y;
      ss << "_" << std::get<0>(info.param).z;
      ss << "_pt_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser
