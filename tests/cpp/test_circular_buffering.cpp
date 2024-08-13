// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class CircularBufferingTest : public NVFuserFixtureParamTest<int> {
 protected:
  int64_t number_of_stages = 1;

  void SetUp() override {
    number_of_stages = GetParam();
    NVFuserTest::SetUp();
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

  tv1->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis 1, the axis_extent is I0/128.
  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv1->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis 1, the axis_extent is I0/128.
  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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
  ASSERT_ANY_THROW(tv2->circularBuffer(number_of_stages));

  // Moving tv2 inner makes tv1 large enough to circular-buffer tv2
  tv2->computeAt(tv3, 2);

  tv2->circularBuffer(number_of_stages);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis 2, the axis_extent is 128/32.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv2->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis -1 and axis 3 is parallelized with TIDx, the axis
  // extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv1->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis -1 and axis 3 is parallelized with TIDx, the axis
  // extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv2->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({199}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis -1 and axis 4 is parallelized with TIDx, the axis
  // extent is 2.
  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv1->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis 2 and axis 1 is parallelized with TIDx, the axis
  // extent is I0/128.
  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv2->circularBuffer(number_of_stages);
  tv3->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({500}, options);
  auto t1 = at::randn({500}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  // Given computeAt axis 1, the axis extent is I0/32/4.
  constexpr int64_t axis_extent = 1;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0, t1});
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

  tv2->circularBuffer(number_of_stages);
  tv3->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1001}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // Given computeAt axis 1 for tv2, the axis extent is I0/32/4 = 8.
  // Given computeAt axis 3 for tv3 and axis 3 is parallelized with TIDx,
  // the axis extent is 4.
  constexpr int64_t axis_extent = 4;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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

  tv0_cache_local->circularBuffer(number_of_stages);
  tv1_cache_local->circularBuffer(number_of_stages);

  tv0_cache_smem->circularBuffer(number_of_stages);
  tv1_cache_smem->circularBuffer(number_of_stages);

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = at::matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  constexpr int64_t axis_extent = 2;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion(aten_inputs);
  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
  // The smem cache write in this test case is redundant predicated,
  //   and also circular buffered. Currently we are relying on WAR sync
  //   insertion to ensure ordering of circular buffered tensor access.
  // The check below makes sure that the sync is inserted so that the
  //   test isn't running on a race condition.
  NVF_CHECK(fe.kernel()->summary().war_hazard_syncs_count > 0);
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
  tv1cr->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({200}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  constexpr int64_t axis_extent = 8;
  if (axis_extent < number_of_stages) {
    ASSERT_ANY_THROW(fe.runFusion({t0}));
    return;
  }

  auto cg_outputs = fe.runFusion({t0});
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
  tv0_shared->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

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
  tv0_shared->circularBuffer(number_of_stages);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

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
  tv0_shared->circularBuffer(number_of_stages);

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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test circular buffer from 2 to 10 stages
INSTANTIATE_TEST_SUITE_P(
    NonTma,
    CircularBufferingTest,
    ::testing::Range(2, 10));

} // namespace nvfuser
