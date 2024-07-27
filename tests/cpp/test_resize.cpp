// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <executor_utils.h>
#include <fusion.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ResizeTest = NVFuserTest;

using testing::Each;
using testing::Not;
using testing::Property;
using testing::UnorderedElementsAre;

// Simple pad test
TEST_F(ResizeTest, Pad1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + split
TEST_F(ResizeTest, Pad2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad, merge + split, inlineMost
TEST_F(ResizeTest, Pad3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9, 11});
  std::vector<int64_t> padded_shape({9, 11 + 2});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = pad(tv2, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  // TransformPropagator and inlineMost do not inline tv2, so it can't
  // be on Local memory. It should be possible to expand tv2 such that
  // it has the same extent as tv3, allowing it to be inlined.
  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// pad + parallelization
TEST_F(ResizeTest, Pad4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  fusion.addOutput(tv1);

  tv1->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + parallelization + RAW sync
TEST_F(ResizeTest, Pad5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  NVF_CHECK(
      tv1->getMemoryType() == MemoryType::Shared,
      "tv1 should be on shared memory: ",
      tv1->getMemoryType());

  GpuLower gpulw(&fusion);
  auto all_lowered_exprs = KernelExprVisitor::getAllExprs(gpulw.run());
  NVF_CHECK(
      std::find_if(
          all_lowered_exprs.begin(),
          all_lowered_exprs.end(),
          [](Expr* expr) { return expr->isA<kir::BlockSync>(); }) !=
          all_lowered_exprs.end(),
      "Block sync not found");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + merge + split parallelization
TEST_F(ResizeTest, Pad6) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({99, 111});
  std::vector<int64_t> padded_shape({shape[0], shape[1] + 2});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(padded_shape);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = pad(tv2, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// pad + unswitch. Having different extents in an unswitched loop nest
// needs a special care (see UnrollPass::canOmitElseClause)
TEST_F(ResizeTest, Pad7) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9, 11});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv3->split(0, 1);
  tv3->split(-1, 4);
  tv3->reorder({{1, 2}});

  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-2)->parallelize(ParallelType::Unswitch);

  scheduler_utils::parallelizeAllLike(tv3);

  scheduler_utils::promoteProducerMemoryTypes(&fusion, {});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Disable for now. Unclear what would be the best way to handle
// when a tensor is resized multiple times. It would likely need a
// different transform propagator.
#if 0
// Stencil-like pattern
TEST_F(ResizeTest, Pad8) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Sort of shift(tv1, {-1});
  auto tv2 = pad(tv1, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(1L)});
  // Sort of shift(tv1, {1});
  auto tv3 = pad(tv1, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(0L)});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 128);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  scheduler_utils::promoteProducerMemoryTypesOfResizedTensors(&fusion, {});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(999, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {0, 1}) + at::pad(t0, {1, 0});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}
#endif

TEST_F(ResizeTest, PadScheduler1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = pad(tv0, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  fusion->addOutput(tv1);

  std::vector<int64_t> shape({99, 111});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, PadScheduler2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({9, 11});
  std::vector<int64_t> padded_shape({9, 11 + 2});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = pad(tv2, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(padded_shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Disabled due to the same reason as Pad8
#if 0
// Auto scheduled version of Pad8
TEST_F(ResizeTest, PadScheduler3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = pad(tv1, {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(1L)});
  auto tv3 = pad(tv1, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(0L)});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(999, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {0, 1}) + at::pad(t0, {1, 0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}
#endif

// Two pad exprs, both using the same symbolic pad widths, segmented
// into two kernels. Make sure the symbolic inputs are available to
// both of the segmented kernels.
TEST_F(ResizeTest, PadScheduler4) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto left_pad = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(left_pad);
  auto right_pad = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(right_pad);

  auto tv1 = pad(tv0, {left_pad, right_pad});
  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto tv3 = pad(tv0, {left_pad, right_pad});
  auto tv4 = sum(tv3, {1});
  fusion->addOutput(tv4);

  std::vector<int64_t> shape({99, 111});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<int64_t> pad_extents{1, 1};
  std::vector<c10::IValue> aten_inputs({t0, 1, 1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Pad a broadcast
// See https://github.com/NVIDIA/Fuser/issues/798
TEST_F(ResizeTest, PadBroadcastInput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // IterTypes are {Broadcast, Iteration}
  auto tv0 = makeConcreteTensor({1, -1});
  fusion->addInput(tv0);

  // trivial pad of broadcast dimension
  auto tv1 =
      pad(tv0,
          {fusion->oneVal(),
           fusion->zeroVal(),
           fusion->zeroVal(),
           fusion->zeroVal()});
  fusion->addOutput(tv1);

  std::vector<int64_t> shape({1, 2});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Trivial cat
TEST_F(ResizeTest, Cat1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2});
  std::vector<int64_t> shape1({3});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial 2D inner cat
TEST_F(ResizeTest, Cat2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial 2D outer cat
TEST_F(ResizeTest, Cat3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({4, 2});
  std::vector<int64_t> shape1({4, 3});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 4);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Cat + merge + split + parallelization + inlineMost
TEST_F(ResizeTest, Cat4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({11, 13});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  fusion.addOutput(tv2);

  tv2->merge(0);
  tv2->split(0, 128);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 1);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Cat + arith op
TEST_F(ResizeTest, Cat5) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor({11, 12});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({11, 13});
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor({11, 25});
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1}, 1);
  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 128);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  inlineMost();

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});
  std::vector<int64_t> shape2({shape0[0], shape0[1] + shape1[1]});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Cat 3 tensors
TEST_F(ResizeTest, Cat6) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({5, 4});
  std::vector<int64_t> shape2({3, 4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor(shape2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(0, 4);
  TransformPropagator propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  inlineMost();

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1, t2}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Cat many tensors
TEST_F(ResizeTest, Cat7) {
  int num_tensors_to_concat = 10;
  std::vector<int64_t> base_shape({11, 13});

  for (int concat_dim : {0, 1}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    std::vector<TensorView*> inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      (void)i;
      // concrete shapes to avoid dynamic Fusion
      auto shape = base_shape;
      shape[concat_dim] = 10 + (i % 5);
      auto tv = makeConcreteTensor(shape);
      fusion.addInput(tv);
      inputs.push_back(tv);
    }

    auto concat_tv = cat(inputs, concat_dim);
    fusion.addOutput(concat_tv);

    concat_tv->merge(0);
    concat_tv->split(0, 128);

    TransformPropagator propagator(concat_tv);
    MaxLogicalDomainInfoSpanningTree(concat_tv).traverse(&propagator);

    inlineMost();

    concat_tv->axis(0)->parallelize(ParallelType::BIDx);
    concat_tv->axis(1)->parallelize(ParallelType::TIDx);
    scheduler_utils::parallelizeAllLike(concat_tv);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    std::vector<at::Tensor> aten_inputs;
    for (const auto i : c10::irange(num_tensors_to_concat)) {
      auto shape = base_shape;
      shape[concat_dim] = 10 + (i % 5);
      aten_inputs.emplace_back(at::randn(shape, options));
    }

    std::vector<c10::IValue> aten_inputs_ivalue(
        {aten_inputs.begin(), aten_inputs.end()});

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs_ivalue);
    auto cg_outputs = fe.runFusion(aten_inputs_ivalue);

    auto ref = at::cat(aten_inputs, concat_dim);

    NVF_CHECK(ref.equal(cg_outputs[0]));
  }
}

// Auto scheduled version of Cat1
TEST_F(ResizeTest, CatScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({2});
  std::vector<int64_t> shape1({3});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Auto scheduled version of Cat5
TEST_F(ResizeTest, CatScheduler2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1}, 1);
  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});
  std::vector<int64_t> shape2({shape0[0], shape0[1] + shape1[1]});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Auto scheduled version of Cat6
TEST_F(ResizeTest, CatScheduler3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({2, 4});
  std::vector<int64_t> shape1({5, 4});
  std::vector<int64_t> shape2({3, 4});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::cat({t0, t1, t2}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Trivial slice
TEST_F(ResizeTest, Slice1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(1L),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(1L))}});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Split a tensor to half and add them up
TEST_F(ResizeTest, Slice2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({11, 30});

  NVF_CHECK(shape[1] % 2 == 0);

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(tv0, {0, 0}, {shape[0], shape[1] / 2});
  auto tv2 = slice(tv0, {0, shape[1] / 2}, {shape[0], shape[1]});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// "Trivial" slice is converted to Set
TEST_F(ResizeTest, Slice3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // These should result in unary set op
  auto tv1 = slice(tv0, {{nullptr, tv0->axis(0)->extent()}});
  auto tv2 = slice(tv0, {Slice()});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  NVF_CHECK(tv1->definition()->isA<LoadStoreOp>());
  NVF_CHECK(tv2->definition()->isA<LoadStoreOp>());
}

// Partition an input, reduce each and concatenate them
TEST_F(ResizeTest, Slice4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({5, 100});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // Consider a fusion of:
  // auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  // auto tv2 = sum(tv1, {1});

  // Reproduce the above fusion with split tensors

  // Split the input to [0:2, :] and [2:, :]
  auto tv1 = slice(
      tv0, {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(2L)}, Slice()});
  auto tv2 = slice(tv0, {{IrBuilder::create<Val>(2L)}, Slice()});

  auto tv3 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv2, IrBuilder::create<Val>(1.0));

  auto tv5 = sum(tv3, {1});
  auto tv6 = sum(tv4, {1});
  auto tv7 = cat({tv5, tv6}, 0);
  fusion.addOutput(tv7);

  // Schedule the two reductions separately
  tv5->split(-1, 32);
  auto tv5_rf = tv5->rFactor({-2});
  tv5_rf->reorder({{-1, -2}});
  auto tv5_cache = tv5->cacheBefore();
  tv5->setMemoryType(MemoryType::Global);
  SetSelector tv5_rf_selector({tv1, tv3, tv5, tv5_cache});
  TransformPropagator tv5_rf_tp(tv5_rf);
  MaxLogicalDomainInfoSpanningTree(tv5_rf, &tv5_rf_selector)
      .traverse(&tv5_rf_tp);
  inlineMost(std::vector<TensorView*>{tv1, tv3, tv5_rf});
  tv5_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv5_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5_rf, {tv1, tv3, tv5, tv5_cache});

  tv6->split(-1, 32);
  auto tv6_rf = tv6->rFactor({-2});
  tv6_rf->reorder({{-1, -2}});
  auto tv6_cache = tv6->cacheBefore();
  tv6->setMemoryType(MemoryType::Global);
  SetSelector tv6_rf_selector({tv2, tv4, tv6, tv6_cache});
  TransformPropagator tv6_rf_tp(tv6_rf);
  MaxLogicalDomainInfoSpanningTree(tv6_rf, &tv6_rf_selector)
      .traverse(&tv6_rf_tp);
  inlineMost(std::vector<TensorView*>{tv2, tv4, tv6_rf});
  tv6_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv6_rf->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv6_rf, {tv2, tv4, tv6, tv6_cache});

  // cat consits of a PadOp and a CatOp. Fully inline the PadOp
  for (auto tv7_inp :
       ir_utils::filterByType<TensorView>(tv7->definition()->inputs())) {
    tv7_inp->inlineAt(-1);
  }

  // Use just one block to concat the two results
  tv7->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = (t0 + 1).to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Multiple slices of the same tensor with the same arguments
TEST_F(ResizeTest, Slice5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({11, 1000});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {Slice(),
       {IrBuilder::create<Val>(1L),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Val>(1L))}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(
      tv0,
      {Slice(),
       {IrBuilder::create<Val>(1L),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Val>(1L))}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  tv2->split(1, 128);

  // tv1 and tv3 are both slice outputs. Propagation should occur from
  // tv1 to tv3 through tv0, which should work as both tensors are
  // sliced in the same way.
  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  inlineMost();

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[1] - 1)});
  auto t2 = t1.to(at::kDouble).sum({1});
  auto t3 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[1] - 1)});
  auto t4 = t3.to(at::kDouble).sum({1});

  testValidate(&fusion, cg_outputs, aten_inputs, {t2, t4}, __LINE__, __FILE__);
}

std::vector<std::pair<int64_t, int64_t>> slice_cases(
    {{0, 5},
     {3, 9},
     {3, 4},
     {7, 5},
     {0, 11},
     {11, 13},
     {-3, 8},
     {-3, -1},
     {-3, -5},
     {13, -1},
     {-11, 9},
     {-11, 0},
     {-13, -11}});

// Test slice with a variety of constant ranges
TEST_F(NVFuserTest, SliceConstantShmoo_CUDA) {
  for (auto [start, stop] : slice_cases) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    std::vector<int64_t> shape({9});

    // concrete shapes to avoid dynamic Fusion
    auto tv0 = makeConcreteTensor(shape);
    fusion.addInput(tv0);

    auto tv1 = slice(
        tv0, {{IrBuilder::create<Val>(start), IrBuilder::create<Val>(stop)}});
    fusion.addOutput(tv1);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    auto t0 = at::randn(shape, options);
    std::vector<c10::IValue> aten_inputs({t0});

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);
    auto cg_outputs = fe.runFusion(aten_inputs);

    testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
  }
}

// Test slice with a variety of non-constant input ranges
TEST_F(NVFuserTest, SliceInputShmoo_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  auto s0 = IrBuilder::create<Val>(DataType::Index);
  auto s1 = IrBuilder::create<Val>(DataType::Index);
  fusion.addInput(tv0);
  fusion.addInput(s0);
  fusion.addInput(s1);

  auto tv1 = slice(tv0, {{s0, s1}});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  {
    // Concretize so that we set output IterType as Iteration. We should now
    // have expressions that work with any input range.
    ExpressionEvaluator expr_eval;

    expr_eval.bind(tv0->axis(0)->extent(), 9);
    expr_eval.bind(s0, 0);
    expr_eval.bind(s1, 9);

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);
    NVF_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto t0 = at::randn(shape, options);
  for (auto [start, stop] : slice_cases) {
    std::vector<c10::IValue> aten_inputs({t0, start, stop});
    auto cg_outputs = fe.runFusion(aten_inputs);

    testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
  }
}

// Same as SliceInputShmoo_CUDA but use FusionExecutorCache, which
// might re-concretize when output sizes change
TEST_F(NVFuserTest, SliceInputShmooFusionExecutorCache_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  std::vector<int64_t> shape({9});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  auto s0 = IrBuilder::create<Val>(DataType::Index);
  auto s1 = IrBuilder::create<Val>(DataType::Index);
  fusion->addInput(tv0);
  fusion->addInput(s0);
  fusion->addInput(s1);

  auto tv1 = slice(tv0, {{s0, s1}});
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto t0 = at::randn(shape, options);
  for (auto [start, stop] : slice_cases) {
    std::vector<c10::IValue> aten_inputs({t0, start, stop});
    auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

    testValidate(fec.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
  }
}

// Auto scheduled version of Slice1
TEST_F(ResizeTest, SliceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(1L),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(1L))}});
  fusion.addOutput(tv1);

  // Make sure all IDs of tv0 and tv1 are mapped in the
  // PERMISSIVE_RESIZE mode.
  ComputeAtMap ca_map(&fusion);
  ASSERT_TRUE(ca_map.areMapped(
      tv1->axis(0), tv0->axis(0), IdMappingMode::PERMISSIVE_RESIZE));
  ASSERT_TRUE(ca_map.areMapped(
      tv1->axis(0),
      tv1->getRootDomain().at(0),
      IdMappingMode::PERMISSIVE_RESIZE));

  std::vector<int64_t> shape({9});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceExtentSimplification) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  // [ i0 ]
  fusion.addInput(tv0);

  auto tv1 =
      slice(tv0, {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(1L)}});
  // By default, the extent of the tv1 domain is:
  //   i0 + ( ( fmax(0, ( fmin(i0, 1) )) ) + ( -i0 ) )
  // This should be simplified to just:
  //   fmax(0, ( fmin(i0, 1) ))

  fusion.addOutput(tv1);

  auto resize_extent = tv1->axis(0)->extent();
  auto bop = dynamic_cast<BinaryOp*>(resize_extent->definition());
  ASSERT_TRUE(bop != nullptr)
      << "Unexpected resize output extent: " << resize_extent->toInlineString();
  EXPECT_EQ(bop->getBinaryOpType(), BinaryOpType::Max)
      << "Unexpected resize output extent: " << resize_extent->toInlineString();
}

TEST_F(ResizeTest, PadReduceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto left_pad0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(left_pad0);
  auto right_pad0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(right_pad0);
  auto left_pad1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(left_pad1);
  auto right_pad1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(right_pad1);

  auto tv1 = pad(tv0, {left_pad0, right_pad0, left_pad1, right_pad1});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> pad_extents{1, 2, 2, 1};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::transform(
      pad_extents.begin(),
      pad_extents.end(),
      std::back_inserter(aten_inputs),
      [](auto pad_extent) { return pad_extent; });

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, SliceReduceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(end0);
  auto start1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(start1);
  auto end1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(end1);

  auto tv1 = slice(tv0, {{start0, end0}, {start1, end1}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> slice_inputs({1, shape[0] - 2, 3, shape[1] - 4});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Multiple slice+reduction. Different slices.
TEST_F(ResizeTest, SliceReduceScheduler2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(end0);
  auto start1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(start1);
  auto end1 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(end1);

  auto tv1 = slice(tv0, {Slice(), {start0, end0}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(tv0, {Slice(), {start1, end1}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 1024});
  std::vector<int64_t> slice_inputs({1, shape[0] - 2, 3, shape[1] - 4});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Multiple slice+reduction. Same slices. Should be segmented at the moment.
TEST_F(ResizeTest, FusionSliceReduceScheduler3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto start0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(start0);
  auto end0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(end0);

  auto tv1 = slice(tv0, {Slice(), {start0, end0}});
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = slice(tv0, {Slice(), {start0, end0}});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> shape({123, 999});
  std::vector<int64_t> slice_inputs({1, shape[1] - 2});

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});
  std::copy(
      slice_inputs.begin(),
      slice_inputs.end(),
      std::back_inserter(aten_inputs));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, CatReduceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  auto tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({11, 12});
  std::vector<int64_t> shape1({shape0[0], 13});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, CatSoftmaxScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 1);
  auto tv3 = softmax(tv2, 1);
  fusion.addOutput(tv3);

  std::vector<int64_t> shape0({11, 99});
  std::vector<int64_t> shape1({shape0[0], 100});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, ReductionSliceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Val>(1L),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Val>(2L))}});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({10, 1234});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Softmax followed by slicing of a non-normalized dimension
TEST_F(ResizeTest, SoftmaxSliceScheduler1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = softmax(tv0, 1);
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Val>(1L),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Val>(2L))},
       Slice()});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({13, 1234});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Softmax followed by slicing of a normalized dimension
TEST_F(ResizeTest, SoftmaxSliceScheduler2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = softmax(tv0, 1);
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(1L),
        sub(tv1->axis(1)->extent(), IrBuilder::create<Val>(2L))}});
  fusion.addOutput(tv2);

  std::vector<int64_t> shape0({110, 12345});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Same as Pad1 but pad by specified value
TEST_F(ResizeTest, PadWithValue) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)},
          IrBuilder::create<Val>(2.0));
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Same as Pad1 but pad by negative value to create an empty tensor
TEST_F(ResizeTest, PadToEmptyTensor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({4, 2});

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(-1L), IrBuilder::create<Val>(-1L)},
          IrBuilder::create<Val>(2.0));
  fusion->addOutput(tv1);
  // set allocation domain to trigger validation check on size/stride
  tv1->setAllocationDomain(tv1->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {-1, -1}, "constant", 2);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Test that padding Half tensor by Double does not promote output
TEST_F(ResizeTest, PadHalfWithDoubleValue) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({9});

  auto tv0 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)},
          IrBuilder::create<Val>(2.5));
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t0 = at::ones(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2.5);

  NVF_CHECK(ref.dtype() == cg_outputs[0].dtype());
  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, FusionSliceForNanoGPT1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> input_shape0{1, 1, 1024, 1024};
  std::vector<int64_t> input_shape1{32, 16, 128, 128};

  auto tv0 = makeContigConcreteTensor({1, 1, -1, -1});
  auto tv1 = makeContigConcreteTensor({-1, -1, -1, -1});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  Slice dim0{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(1L),
      IrBuilder::create<Val>(1L)};
  Slice dim1{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(1L),
      IrBuilder::create<Val>(1L)};
  Slice dim2{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(128L),
      IrBuilder::create<Val>(1L)};
  Slice dim3{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(128L),
      IrBuilder::create<Val>(1L)};
  auto tv2 = slice(tv0, {dim0, dim1, dim2, dim3});

  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  auto tv4 = slice(tv0, {dim0, dim1, dim2, dim3});

  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape0, options);
  auto t1 = at::randn(input_shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto kernel =
      executor_cache.getMostRecentKernelRuntime()->executors().at(0).kernel();
  NVF_CHECK(
      !kernel->summary().has_cooperative_grid_reduction,
      "Grid sync should not be used as slicing input should avoid input caching");

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Similar to FusionSliceForNanoGPT1 but the input to slice is an
// intermediate tensor
TEST_F(ResizeTest, FusionSliceForNanoGPT2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> input_shape0{100, 100};
  std::vector<int64_t> input_shape1{32, 32};

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));

  Slice dim0{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(32L),
      IrBuilder::create<Val>(1L)};
  Slice dim1{
      IrBuilder::create<Val>(0L),
      IrBuilder::create<Val>(32L),
      IrBuilder::create<Val>(1L)};

  auto tv3 = slice(tv2, {dim0, dim1});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto tv5 = slice(tv2, {dim0, dim1});
  auto tv6 = add(tv5, tv1);
  fusion.addOutput(tv6);

  // Another use of tv2. Unlike the above two slice ops, this should
  // not use the copy of tv2
  auto tv7 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape0, options);
  auto t1 = at::randn(input_shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto kernel =
      executor_cache.getMostRecentKernelRuntime()->executors().at(0).kernel();

  // Make sure the slices ops use the same producer
  TensorView* known_slice_producer = nullptr;
  for (auto expr : KernelExprVisitor::getAllExprs(kernel)) {
    if (!ir_utils::isTvOp(expr)) {
      continue;
    }
    auto out_tv = ir_utils::getTvOutput(expr);
    if (out_tv->name() == tv3->name() || out_tv->name() == tv5->name()) {
      NVF_CHECK(
          expr->isA<LoadStoreOp>(),
          "Unexpected defintion of slice output tensor: ",
          out_tv->toString(),
          ", ",
          expr->toString());
      auto producer =
          dynamic_cast<kir::TensorIndex*>(expr->as<LoadStoreOp>()->in());
      if (producer == nullptr) {
        // this could be a default initialization
        continue;
      }
      if (known_slice_producer == nullptr) {
        known_slice_producer = producer->view();
      } else {
        NVF_CHECK(
            known_slice_producer == producer->view(),
            "Expected to have the same tensor is used for the two slice ops. ",
            "Previously found producer: ",
            known_slice_producer->toString(),
            ", new producer: ",
            producer->view()->toString());
      }
    } else if (auto binary_op = dynamic_cast<BinaryOp*>(expr)) {
      // If this is a binary op producing tv7, make sure its producer
      // is tv2
      if (binary_op->getBinaryOpType() == BinaryOpType::Add &&
          binary_op->out()->isA<kir::TensorIndex>() &&
          binary_op->out()->as<kir::TensorIndex>()->view()->name() ==
              tv7->name()) {
        NVF_CHECK(
            binary_op->lhs()->as<kir::TensorIndex>()->view()->name() ==
                tv2->name(),
            "Unexpected tv7 definition: ",
            binary_op->toString());
      }
    }
  }

  NVF_CHECK(known_slice_producer != nullptr, "Slice producer not found");

  // The slice producer must be a copy of tv2
  NVF_CHECK(
      known_slice_producer->definition() &&
          known_slice_producer->definition()->isA<LoadStoreOp>() &&
          known_slice_producer->definition()->as<LoadStoreOp>()->in()->name() ==
              tv2->name(),
      "Unexpected slice producer: ",
      known_slice_producer->toString());

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// C++ version of TestNvFuserFrontend.test_nanogpt_split_mha_linears
TEST_F(ResizeTest, FusionSliceForNanoGPT3) {
  // To verify input caching condition in this test, disable aliasing as that
  // will skip compilation and no kernel will exist.
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<int64_t> input_shape{16, 128, 3072};

  auto tv0 = makeSymbolicTensor(3);

  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(16L)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(128L)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(1024L)}});
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(16L)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(128L)},
       {IrBuilder::create<Val>(1024L), IrBuilder::create<Val>(2048L)}});
  auto tv3 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(16L)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(128L)},
       {IrBuilder::create<Val>(2048L), IrBuilder::create<Val>(3072L)}});

  auto tv4 = reshape(tv1, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv5 = reshape(tv2, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv6 = reshape(tv3, {16, 128, 1024}, {16, 128, 16, 64});

  // TODO: add permute
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto kernel = runtime->executors().at(0).kernel();
  NVF_CHECK(
      !kernel->summary().has_cooperative_grid_reduction,
      "Grid sync should not be used as slicing input should avoid input caching");

  auto at_t1 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(0, 1024)});
  auto at_t2 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(1024, 2048)});
  auto at_t3 = t0.index(
      {at::indexing::Slice(0, 16),
       at::indexing::Slice(0, 128),
       at::indexing::Slice(2048, 3072)});

  auto at_t4 = at_t1.reshape({16, 128, 16, 64});
  auto at_t5 = at_t2.reshape({16, 128, 16, 64});
  auto at_t6 = at_t3.reshape({16, 128, 16, 64});

  NVF_CHECK(cg_outputs.at(0).equal(at_t4));
  NVF_CHECK(cg_outputs.at(1).equal(at_t5));
  NVF_CHECK(cg_outputs.at(2).equal(at_t6));
}

TEST_F(ResizeTest, ResizeReshapeAndSlice) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {4, 8}, {8, 4});
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Val>(0L), IrBuilder::create<Val>(2L)},
       {IrBuilder::create<Val>(0L), IrBuilder::create<Val>(2L)}});
  auto tv3 = add(tv2, tv2);
  fusion->addOutput(tv3);

  std::vector<int64_t> shape({4, 8});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
}

// Make sure resize works with the transpose scheduler
TEST_F(ResizeTest, ResizePermuteAndSlice) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  // Set the problem size so that it can trigger the transpose
  // scheduler. The scheduler selection is validated below.
  auto num_sms =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  std::vector<int64_t> shape({num_sms + 2, 32 * 32 + 10});

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(shape.at(0) - 1)},
       {IrBuilder::create<Val>(2L), IrBuilder::create<Val>(shape.at(1) - 2)}});
  auto tv3 = transpose(tv2, 0, 1);
  auto tv5 = add(tv3, tv3);
  fusion->addOutput(tv5);
  auto tv4 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);

  EXPECT_THAT(
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(ScheduleHeuristic::Transpose)));
}

// When scheduling this test, the pointwise scheduler attempt to replay a Split
// transform on a size-0 dimension, which is not allowed.
TEST_F(ResizeTest, FusionSizeZeroSliceSplitSchedule) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({8});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion->addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(2L),
        IrBuilder::create<Val>(1L)}});
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(2L),
        IrBuilder::create<Val>(4L),
        IrBuilder::create<Val>(1L)}});
  auto tv3 = slice(
      tv0,
      {{IrBuilder::create<Val>(4L),
        IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(1L)}});
  auto tv4 = slice(
      tv0,
      {{IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(1L)}});
  auto tv5 = slice(
      tv0,
      {{IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(1L)}});
  auto tv6 = slice(
      tv0,
      {{IrBuilder::create<Val>(6L),
        IrBuilder::create<Val>(8L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);
  fusion->addOutput(tv4);
  fusion->addOutput(tv5);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);
  FusionExecutor fe;

  auto ref0 = t0.index({at::indexing::Slice(0, 2)});
  auto ref1 = t0.index({at::indexing::Slice(2, 4)});
  auto ref2 = t0.index({at::indexing::Slice(4, 6)});
  auto ref3 = t0.index({at::indexing::Slice(6, 6)});
  auto ref4 = t0.index({at::indexing::Slice(6, 6)});
  auto ref5 = t0.index({at::indexing::Slice(6, 8)});

  NVF_CHECK(ref0.equal(cg_outputs[0]));
  NVF_CHECK(ref1.equal(cg_outputs[1]));
  NVF_CHECK(ref2.equal(cg_outputs[2]));
  NVF_CHECK(ref3.equal(cg_outputs[3]));
  NVF_CHECK(ref4.equal(cg_outputs[4]));
  NVF_CHECK(ref5.equal(cg_outputs[5]));
}

// In this test, we split and merge with size-zero dimensions directly.
TEST_F(ResizeTest, FusionSizeZeroSliceSplit) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({4, 5});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion->addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(2L),
        IrBuilder::create<Val>(2L),
        IrBuilder::create<Val>(1L)},
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(5L),
        IrBuilder::create<Val>(1L)}});
  // tv1 is of shape {0, 5}
  fusion->addOutput(tv1);

  tv1->merge(0, 1); // size 0*5 = 0
  tv1->split(0, 4); // sizes (0, 4)

  FusionExecutor fe;
  fe.compileFusion(fusion.get());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref0 = t0.index({at::indexing::Slice(2, 2), at::indexing::Slice(0, 5)});

  NVF_CHECK(ref0.equal(cg_outputs[0]));
}

// Test squeezing a symbolic dimension
TEST_F(ResizeTest, FusionSqueezeSymbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({4, 5});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto s1 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(s1);
  auto numel_symb = mul(tv0->axis(0)->extent(), tv0->axis(1)->extent());
  auto tv1 = reshape(tv0, {s1, ceilDiv(numel_symb, s1)});
  // tv1 should have symbolic output IterTypes, so squeeze should accept
  // symbolic, and concretization should fail if symbolic squeeze input is not
  // concretized to Broadcast
  // NOTE: squeeze interface should be updated to match reshape and friends,
  // accepting Val inputs
  auto tv2 = squeeze(tv1, std::vector<int64_t>{1});
  // tv1 is of shape {0, 5}
  fusion->addOutput(tv2);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0, 20});

  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto ref0 = t0.flatten();

  NVF_CHECK(ref0.equal(cg_outputs[0]));

  EXPECT_THAT(
      [&]() { fec.runFusionWithInputs({t0, 10}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "must concretize to IterType::Broadcast but found")));
}

// See https://github.com/NVIDIA/Fuser/issues/365
TEST_F(ResizeTest, MultiSliceEmpty) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape({9});
  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion->addInput(tv0);

  // In issue #365, this triggered an error in vectorization when there were
  // multiple slices, and one of them was empty. If this is properly handled in
  // the pre-segmentation RemoveEmptyPass as it should be, then the size-zero
  // slices will be replaced with full(), and vectorization can work properly.
  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv1);
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref0 = t0.index({at::indexing::Slice(0, 1)});
  auto ref1 = t0.index({at::indexing::Slice(0, 0)});

  NVF_CHECK(ref0.equal(cg_outputs[0]));
  NVF_CHECK(ref1.equal(cg_outputs[1]));

  // Check that tv2 is replaced by a FullOp
  const auto runtime = executor_cache.getMostRecentKernelRuntime();
  const auto preseg_fusion = runtime->fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 2);
  EXPECT_NE(preseg_fusion->outputs().at(1), tv1);
  EXPECT_NE(preseg_fusion->outputs().at(1)->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs().at(1)->definition()->isA<FullOp>());
}

TEST_F(ResizeTest, SliceVectorization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int N = 1024 * 1024 * 64;

  auto tv0 = makeContigConcreteTensor({N + 1});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({N});
  fusion.addInput(tv1);

  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(N + 1L),
        IrBuilder::create<Val>(1L)}});

  auto tv3 = add(tv2, tv1);

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(N + 1, options);
  at::Tensor t1 = at::randn(N, options);

  std::vector<c10::IValue> inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, inputs);

  // check that we vectorize 4
  bool found_vectorize = false;
  for (auto id : fusion.outputs().at(0)->as<TensorView>()->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      EXPECT_EQ(id->extent()->evaluate(), 4);
      found_vectorize = true;
      break;
    }
  }
  EXPECT_TRUE(found_vectorize);

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs, lparams);
  auto cg_outputs = fe.runFusion(inputs, lparams);

  auto ref = t0.narrow(0, 1, N) + t1;

  // testValidate does not check that dtypes match
  EXPECT_EQ(cg_outputs[0].dtype(), ref.dtype());
  testValidate(&fusion, cg_outputs, inputs, __LINE__, __FILE__);
}

// Concretize a symbolic pad that results in a broadcast (static pads)
// In this test, the sizes and pad widths are static, so there should be nothing
// to concretize.
TEST_F(NVFuserTest, ResizePadToBroadcastStatic_CUDA) {
  std::vector<int64_t> t0_size = {2, 3, 2, 5, 6};
  std::vector<int64_t> t1_size = {2, 4, 4, 3, 5};
  // Note there are only 8 input scalars for 5D input. Implicit no-pad of dim 0
  std::vector<int64_t> pad_widths = {
      0,
      -1, // dim=4 trim last element
      0,
      -4, // dim=3 pad to broadcast of first element
      1,
      1, // dim=2 pad with zeros on either side
      -1,
      -1, // dim=1 pad to broadcast of second element
      // dim=0 is implicit 0, 0
  };
  std::vector<IterType> expected_itertypes = {
      IterType::Iteration,
      IterType::Broadcast,
      IterType::Iteration,
      IterType::Broadcast,
      IterType::Iteration,
  };

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor(t0_size);
  fusion->addInput(tv0);
  auto tv1 = makeConcreteTensor(t1_size);
  fusion->addInput(tv1);

  std::vector<Val*> pad_width_vals;
  pad_width_vals.reserve(pad_widths.size());
  for (auto w : pad_widths) {
    pad_width_vals.push_back(IrBuilder::create<Val>(w));
  }

  auto tv2 = pad(tv0, pad_width_vals);
  auto tv3 = mul(tv1, tv2);
  fusion->addOutput(tv3);

  EXPECT_FALSE(fusion->hasDynamicTransform());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(t0_size, options);
  auto t1 = at::randn(t1_size, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  auto concretized_fusion = runtime->fusionSegments()->completeFusion();

  auto conc_t2 = concretized_fusion->outputs()[0]
                     ->definition()
                     ->inputs()[1]
                     ->as<TensorView>();
  for (auto i : c10::irange(expected_itertypes.size())) {
    EXPECT_EQ(conc_t2->axis(i)->getIterType(), expected_itertypes.at(i));
  }

  testValidate(concretized_fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Concretize a symbolic pad that results in a broadcast (dynamic pads)
TEST_F(NVFuserTest, ResizePadToBroadcastDynamic_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(5);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(5);
  fusion->addInput(tv1);

  // Note there are only 8 input scalars for 5D input. Implicit no-pad of dim 0
  std::vector<int64_t> pad_widths = {
      0,
      -1, // dim=4 trim last element
      0,
      -4, // dim=3 pad to broadcast of first element
      1,
      1, // dim=2 pad with zeros on either side
      -1,
      -1, // dim=1 pad to broadcast of second element
      // dim=0 is implicit 0, 0
  };
  std::vector<Val*> pad_width_vals;
  pad_width_vals.reserve(pad_widths.size());
  for ([[maybe_unused]] auto _ : pad_widths) {
    auto w_val = IrBuilder::create<Val>(DataType::Int);
    fusion->addInput(w_val);
    pad_width_vals.push_back(w_val);
  }

  auto tv2 = pad(tv0, pad_width_vals);
  auto tv3 = mul(tv1, tv2);
  fusion->addOutput(tv3);

  EXPECT_TRUE(fusion->hasDynamicTransform());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({2, 3, 2, 5, 6}, options);
  auto t1 = at::randn({2, 4, 4, 3, 5}, options);
  // Keep dimension 0, pad to broadcast in dimension 1 and 3. Pad with zero in
  // dimension 2. Trim by one element in dimension 4.
  std::vector<c10::IValue> aten_inputs({
      t0,
      t1,
  });
  aten_inputs.insert(aten_inputs.end(), pad_widths.begin(), pad_widths.end());

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  auto concretized_fusion = runtime->fusionSegments()->completeFusion();

  auto conc_t2 = concretized_fusion->outputs()[0]
                     ->definition()
                     ->inputs()[1]
                     ->as<TensorView>();
  EXPECT_EQ(conc_t2->axis(0)->getIterType(), IterType::Iteration);
  EXPECT_EQ(conc_t2->axis(1)->getIterType(), IterType::Broadcast);
  EXPECT_EQ(conc_t2->axis(2)->getIterType(), IterType::Iteration);
  EXPECT_EQ(conc_t2->axis(3)->getIterType(), IterType::Broadcast);
  EXPECT_EQ(conc_t2->axis(4)->getIterType(), IterType::Iteration);

  testValidate(concretized_fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// See https://github.com/NVIDIA/Fuser/issues/596
TEST_F(NVFuserTest, ResizePadToBroadcastIssue596_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2});
  auto tv1 = makeConcreteTensor({3});
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = pad(tv0, {fusion->zeroVal(), IrBuilder::create<Val>(-1)});
  auto tv3 = mul(tv1, tv2);
  fusion->addOutput(tv3);

  // Fusion is not dynamic
  EXPECT_FALSE(fusion->hasDynamicTransform());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({2}, options);
  auto t1 = at::randn({3}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion), args);
  runtime.compileFusionParallel(args);
  auto cg_outputs = runtime.runWithInputs(args);

  testValidate(
      runtime.fusionSegments()->completeFusion(),
      cg_outputs,
      aten_inputs,
      __LINE__,
      __FILE__);
}

// An input is sliced and then reshaped
TEST_F(ResizeTest, SliceAndReshape1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({1024, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(1L),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Val>(1L))}});
  auto tv2 = reshape(tv1, {IrBuilder::create<Val>(-1L)});
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[0] - 1)});
  auto ref = t1.reshape({-1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// An input is sliced and also separately reshaped
TEST_F(ResizeTest, SliceAndReshape2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({1024, 1024});

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0L), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(1L),
        sub(tv0->axis(1)->extent(), IrBuilder::create<Val>(1L))}});
  auto tv2 = reshape(tv0, {IrBuilder::create<Val>(-1L)});
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.index(
      {at::indexing::Slice(0, at::indexing::None),
       at::indexing::Slice(1, shape[0] - 1)});
  auto t2 = t0.reshape({-1});

  NVF_CHECK(t1.equal(cg_outputs[0]));
  NVF_CHECK(t2.equal(cg_outputs[1]));
}

// Trivial case of slice vectorization. Just slicing a fusion input
TEST_F(ResizeTest, Slice1DVectorizeManual1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const int64_t slice_offset = 4;
  const std::vector<int64_t> shape({1024L * 1024L});

  // Using a concrete tensor to avoid dynamic reshape
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(slice_offset),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(slice_offset))}});
  fusion.addOutput(tv1);

  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref =
      t0.index({at::indexing::Slice(slice_offset, shape[0] - slice_offset)});
  ASSERT_TRUE(ref.equal(cg_outputs[0]));
}

// An input is sliced twice. Both should be vectorizable.
TEST_F(ResizeTest, Slice1DVectorizeManual2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const int64_t slice_offset = 4;
  const std::vector<int64_t> shape({1024L * 1024L});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  // Following two slices are vectorized individually. No cache is introduced
  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(slice_offset),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(slice_offset))}});
  fusion.addOutput(tv1);

  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(slice_offset * 2),
        sub(tv0->axis(0)->extent(),
            IrBuilder::create<Val>(slice_offset * 2))}});
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv1->split(0, 128);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  scheduler_utils::parallelizeAllLike(tv1);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref_t1 =
      t0.index({at::indexing::Slice(slice_offset, shape[0] - slice_offset)});
  auto ref_t2 = t0.index(
      {at::indexing::Slice(slice_offset * 2, shape[0] - slice_offset * 2)});
  ASSERT_TRUE(ref_t1.equal(cg_outputs.at(0)));
  ASSERT_TRUE(ref_t2.equal(cg_outputs.at(1)));
}

// An input is sliced and also entirely read. Both should be vectorizable.
TEST_F(ResizeTest, Slice1DVectorizeManual3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const int64_t slice_offset = 4;
  const std::vector<int64_t> shape({1024L * 1024L});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(slice_offset),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(slice_offset))}});
  fusion.addOutput(tv1);

  auto tv2 = set(tv0);
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv1->split(0, 128);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  scheduler_utils::parallelizeAllLike(tv1);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref =
      t0.index({at::indexing::Slice(slice_offset, shape[0] - slice_offset)});
  ASSERT_TRUE(ref.equal(cg_outputs.at(0)));
  ASSERT_TRUE(t0.equal(cg_outputs.at(1)));
}

// Vectorizing a slice of [1:-3]. It's vectorizable as long as the
// offset at 1 is aligned
TEST_F(ResizeTest, Slice1DVectorizeManual4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({1024L * 1024L});

  auto tv0 = makeContigConcreteTensor({shape[0] - 4});
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(1),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(3))}});
  fusion.addOutput(tv1);

  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0_unaligned = at::randn(shape, options);
  auto t0_aligned = t0_unaligned.index({at::indexing::Slice(3, -1)});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0_aligned});
  auto cg_outputs = fe.runFusion({t0_aligned});

  auto ref_aligned = t0_aligned.index({at::indexing::Slice(1, -3)});

  ASSERT_TRUE(ref_aligned.equal(cg_outputs.at(0)));
}

// Contig merged vectorization with slice
TEST_F(ResizeTest, Slice2DVectorizeManual1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const int64_t slice_offset = 4;

  // The extent of the innermost domain is just 2, and the outer
  // domain is sliced. This slicing should be vectorizable by a
  // factor of 4 as the two domains can be merged and vectorized.
  const std::vector<int64_t> shape({1024L * 1024L, 2});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(slice_offset),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Val>(slice_offset))},
       {IrBuilder::create<Val>(0), tv0->axis(1)->extent()}});
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 128);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.index(
      {at::indexing::Slice(slice_offset, shape[0] - slice_offset),
       at::indexing::Slice(0, at::indexing::None)});
  ASSERT_TRUE(ref.equal(cg_outputs.at(0)));
}

// Fully contiguous tensor, but a sliced domain makes the domain to
// the left non-contiguous
TEST_F(ResizeTest, Slice3DVectorizeManual1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({4, 1025, 3});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(4), IrBuilder::create<Val>(6)},
       {IrBuilder::create<Val>(0), tv0->axis(2)->extent()}});
  fusion.addOutput(tv1);

  // Vectorize tv1 by a factor of 2. The sliced domain and the
  // innermost domain can be contiguous merged, thus producing a
  // domain of extent 6, so vectorization by a factor of 2 appears to
  // be valid, but due to the middle domain being sliced, the
  // outermost domain is no longer contiguous, which means its stride
  // must be divisible by 2, which is not the case here.

  // [4, 2, 3]
  tv1->merge(1);
  // [4, 6]
  tv1->split(1, 2);
  // [4, 3, 2]

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  EXPECT_THAT(
      [&]() { fe.runFusion(aten_inputs); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "with word size 2 not possible due to invalid stride")));
}

// Similar to Slice3DVectorizeManual2 but with a middle broadcast
// domain
TEST_F(ResizeTest, Slice3DVectorizeManual2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({4, 1, 1025, 3});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(0), tv0->axis(1)->extent()},
       {IrBuilder::create<Val>(0), IrBuilder::create<Val>(1024)},
       {IrBuilder::create<Val>(0), tv0->axis(3)->extent()}});
  fusion.addOutput(tv1);

  // [4, 1, 1024, 3]
  tv1->merge(2);
  // [4, 1, 3072]
  tv1->split(2, 4);
  // [4, 1, 768, 4]

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(3)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);

  EXPECT_THAT(
      [&]() { fe.runFusion(aten_inputs); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "with word size 4 not possible due to invalid stride")));
}

// Repro of issue 540 without transpose
TEST_F(ResizeTest, SliceAndReshapeRepro540Manual) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  const std::vector<int64_t> shape({16, 128, 3072});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Val>(0), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(0), tv0->axis(1)->extent()},
       {IrBuilder::create<Val>(0), IrBuilder::create<Val>(1024)}});
  auto tv2 = slice(
      tv0,
      {{IrBuilder::create<Val>(0), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(0), tv0->axis(1)->extent()},
       {IrBuilder::create<Val>(1024), IrBuilder::create<Val>(2048)}});
  auto tv3 = slice(
      tv0,
      {{IrBuilder::create<Val>(0), tv0->axis(0)->extent()},
       {IrBuilder::create<Val>(0), tv0->axis(1)->extent()},
       {IrBuilder::create<Val>(2048), IrBuilder::create<Val>(3072)}});

  auto tv4 = reshape(tv1, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv5 = reshape(tv2, {16, 128, 1024}, {16, 128, 16, 64});
  auto tv6 = reshape(tv3, {16, 128, 1024}, {16, 128, 16, 64});

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv4->cacheBefore();
  tv5->cacheBefore();
  tv6->cacheBefore();

  tv4->merge(0)->merge(0)->merge(0);
  // Vectorize
  tv4->split(0, 4);
  // Unswitch
  tv4->split(0, 1);
  // TIDx
  tv4->split(0, 128);

  tv4->reorder({{1, -1}});

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unswitch);
  tv4->axis(3)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv4);

  for (auto output : fusion.outputs()) {
    output->as<TensorView>()->axis(2)->parallelize(ParallelType::Vectorize);
  }

  for (auto slice_tv : {tv1, tv2, tv3}) {
    slice_tv->axis(2)->parallelize(ParallelType::Vectorize);
  }

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  for (const auto i : c10::irange(3)) {
    auto slice_out_ref = t0.index(
        {at::indexing::Slice(0, at::indexing::None),
         at::indexing::Slice(0, at::indexing::None),
         at::indexing::Slice(i * 1024, (i + 1) * 1024)});
    auto ref = at::native::view(slice_out_ref, {16, 128, 16, 64});
    ASSERT_TRUE(ref.equal(cg_outputs.at(i)));
  }
}

// Test concretizing a pad that follows a reshape. This requires the
// ExpressionEvaluator used in concretization to propagate shapes properly
// across symbolic reshapes in order to infer the size of the downstream pad.
TEST_F(ResizeTest, ReshapeToPad) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto s0 = IrBuilder::create<Val>(DataType::Int);
  auto s1 = IrBuilder::create<Val>(DataType::Int);
  auto s2 = IrBuilder::create<Val>(DataType::Int);
  auto s3 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(s0);
  fusion.addInput(s1);
  fusion.addInput(s2);
  fusion.addInput(s3);

  auto tv1 = reshape(tv0, {s2, s3});
  auto tv2 = pad(tv1, {fusion.zeroVal(), s0, fusion.zeroVal(), s1});
  fusion.addOutput(tv2);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({4, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at_x, 1, 1, 3, 4};
  auto at_y = at::pad(at_x.reshape({3, 4}), {0, 1, 0, 1});

  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  // Assert that we segmented into two segments
  auto seg_fusion =
      fusion_executor_cache.getMostRecentKernelRuntime()->fusionSegments();
  EXPECT_TRUE(seg_fusion->isSegmented());
  EXPECT_EQ(seg_fusion->groups().size(), 2);

  testValidate(
      fusion_executor_cache.fusion(),
      outputs,
      aten_inputs,
      {at_y},
      __LINE__,
      __FILE__);
}

TEST_F(ResizeTest, ReshapeToSlice) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto s0 = IrBuilder::create<Val>(DataType::Int);
  auto s1 = IrBuilder::create<Val>(DataType::Int);
  auto s2 = IrBuilder::create<Val>(DataType::Int);
  auto s3 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(s0);
  fusion.addInput(s1);
  fusion.addInput(s2);
  fusion.addInput(s3);

  auto tv1 = reshape(tv0, {s2, s3});
  auto tv2 = slice(tv1, {{fusion.zeroVal(), s0}, {fusion.zeroVal(), s1}});
  fusion.addOutput(tv2);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({4, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at_x, 3, 2, 3, 4};
  auto at_y = at::slice(at::slice(at_x.reshape({3, 4}), 0, 0, 3), 1, 0, 2);

  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  // Assert that we segmented into two segments
  auto seg_fusion =
      fusion_executor_cache.getMostRecentKernelRuntime()->fusionSegments();
  EXPECT_TRUE(seg_fusion->isSegmented());
  EXPECT_EQ(seg_fusion->groups().size(), 2);

  testValidate(
      fusion_executor_cache.fusion(),
      outputs,
      aten_inputs,
      {at_y},
      __LINE__,
      __FILE__);
}

// Test that we can cat along broadcast dims
// See https://github.com/NVIDIA/Fuser/issues/224
TEST_F(ResizeTest, CatOfBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({1, 2});
  std::vector<int64_t> shape1({3, 2});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({t0, t1}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Test that we can cat along broadcast dims that have been expanded
// See https://github.com/NVIDIA/Fuser/issues/224
TEST_F(ResizeTest, CatOfExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({1, 2});
  std::vector<int64_t> shape0e({4, 2});
  std::vector<int64_t> shape1({3, 2});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor(shape1);
  fusion.addInput(tv1);

  auto tv0e = expand(
      tv0, {IrBuilder::create<Val>(shape0e.at(0)), tv0->axis(1)->extent()});

  auto tv2 = cat({tv0e, tv1}, 0);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat({at::expand_copy(t0, shape0e), t1}, 0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Test that an empty input which is expanded in some non-zero directions can be
// padded in the empty dim as well as the expanded dims.
// This should match test_python_frontend.py::test_pad_expanded_empty
// See https://github.com/NVIDIA/Fuser/issues/870
TEST_F(ResizeTest, PadExpandedEmpty) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto i0 = IrBuilder::create<Val>(DataType::Index);
  auto i1 = IrBuilder::create<Val>(DataType::Index);
  auto i2 = IrBuilder::create<Val>(DataType::Index);

  auto tv0 = TensorViewBuilder()
                 .shape({i0, i1, i2})
                 .expanded({true, false, true})
                 .dtype(DataType::Double)
                 .build();
  fusion.addInput(tv0);

  auto s0 = IrBuilder::create<Val>(-3.70753);

  std::vector<Val*> pad_widths(
      {fusion.zeroVal(DataType::Index),
       fusion.zeroVal(DataType::Index),
       fusion.oneVal(DataType::Index),
       fusion.oneVal(DataType::Index),
       fusion.oneVal(DataType::Index),
       fusion.zeroVal(DataType::Index)});
  auto tv1 = pad(tv0, pad_widths, s0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);

  auto t0 = at::randn({0}, options).as_strided({2, 0, 3}, {0, 0, 0});
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we can pad properly along broadcast dims
// See https://github.com/NVIDIA/Fuser/issues/868
TEST_F(ResizeTest, PadOfBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({1});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv1 = pad(tv0, {fusion.oneVal(), fusion.oneVal()});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we can cat along broadcast dims that have been expanded
// See https://github.com/NVIDIA/Fuser/issues/868
TEST_F(ResizeTest, PadOfExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({1});
  std::vector<int64_t> shape0e({4});

  auto tv0 = makeConcreteTensor(shape0);
  fusion.addInput(tv0);

  auto tv0e = expand(tv0, {IrBuilder::create<Val>(shape0e.at(0))});

  auto tv1 = pad(tv0e, {fusion.oneVal(), fusion.oneVal()});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, dynamicReshapeIssue1393) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(2)
                 .shape({-1, -1})
                 .contiguity({true, std::nullopt})
                 .expanded({false, true})
                 .build();
  auto tv1 = TensorViewBuilder()
                 .ndims(2)
                 .shape({-1, -1})
                 .contiguity({std::nullopt, true})
                 .expanded({true, false})
                 .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);
  auto s0 = IrBuilder::create<Val>(3);
  auto s1 = IrBuilder::create<Val>(4);
  auto s2 = IrBuilder::create<Val>(1);
  auto s3 = IrBuilder::create<Val>(5);
  auto tv3 = reshape(tv2, {s0, s1, s2});
  auto tv4 = expand(tv3, {s0, s1, s3});
  fusion->addOutput(tv4);

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3}, options).as_strided({3, 4}, {1, 0});
  at::Tensor t1 = at::randn({4}, options).as_strided({3, 4}, {0, 1});
  auto ref = t0.add(t1).as_strided({3, 4, 5}, {4, 1, 0});

  std::vector<c10::IValue> aten_inputs({t0, t1});
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  testValidate(fusion, outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test that we can slice a trivially expanded tensor to size 1 then squeeze
// See https://github.com/NVIDIA/Fuser/issues/963
TEST_F(ResizeTest, SqueezeSlicedExpand) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  std::vector<int64_t> shape0({9, 5});

  // dynamic input shape
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  // Note these are Int instead of Index. They will be cast to Index when used
  // as extents.
  auto s0 = IrBuilder::create<Val>(9L);
  auto s1 = IrBuilder::create<Val>(5L);

  // The expand op will create a LoadStoreOp with these values as output
  // extents. This effectively creates a static shape TV from a dynamic shape
  // TV.
  auto tv1 = expand(tv0, {s0, s1});

  auto s2 = IrBuilder::create<Val>(2L);
  auto s3 = IrBuilder::create<Val>(3L);
  auto tv2 =
      slice(tv1, {{nullptr, nullptr, nullptr}, {s2, s3, fusion->oneVal()}});
  std::vector<bool> squeeze_dims({false, true});

  auto tv3 = squeeze(tv2, squeeze_dims);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape0, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(aten_inputs);

  auto ref = at::squeeze(at::slice(t0, 1, 2, 3), 1);

  testValidate(
      fec.fusion(), cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

// Vectorization through resize is not supported yet. Make sure
// vectorization is disabled.
TEST_F(ResizeTest, AvoidVectorization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D tensor with a large enough inner domain. The outer
  // domain will be padded.
  std::vector<int64_t> shape({2, 1000L * 128});
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = pad(
      tv0,
      {fusion.zeroVal(), fusion.zeroVal(), fusion.oneVal(), fusion.oneVal()});
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  std::vector<c10::IValue> inputs({t0});

  // The pointwise scheduler should tell the vectorization factor is
  // 4.
  auto params = getPointwiseHeuristics(&fusion, inputs);
  ASSERT_TRUE(params->vectorize) << "Vectorization is expected to be possible";
  ASSERT_EQ(params->unroll_factor, 4) << "Unexpected factor of vectorization";

  schedulePointwise(&fusion, *params);

  // Make sure tv1 is not vectorized, i.e., no loop IterDomains are vectorized.
  EXPECT_THAT(
      tv1->getLoopDomain(),
      Each(
          Property(&IterDomain::getParallelType, Not(ParallelType::Vectorize))))
      << "Unexpected vectorization: " << tv1;

  // Make sure tv2 should be vectorized, i.e., at least one loop IterDomain is
  // vectorized.
  EXPECT_THAT(
      tv2->getLoopDomain(),
      Contains(Property(&IterDomain::getParallelType, ParallelType::Vectorize)))
      << "Failed to vectorize: " << tv2;

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs, params->lparams);
  auto outputs = fe.runFusion(inputs, params->lparams);
  testValidate(&fusion, outputs, inputs, __LINE__, __FILE__);
}

// MemoryPromotion generates code with volatile T. This test ensures that our
// reduced precision types in runtime file have volatile methods defined
TEST_F(ResizeTest, CatMemoryPromotionReducedFloating) {
  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  std::vector<DataType> dtype_variants({DataType::Half});

  if (deviceMajorMinorCheck(8)) {
    dtype_variants.push_back(DataType::BFloat16);
  }
  if (deviceMajorMinorCheck(9)) {
    dtype_variants.push_back(DataType::Float8_e4m3fn);
    // We cannot set nan to e5m2.
    setFillAllocationWithNan(false);
    dtype_variants.push_back(DataType::Float8_e5m2);
  }

  for (auto dtype : dtype_variants) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    FusionGuard fg(fusion_ptr.get());

    TensorView* tv0 = makeSymbolicTensor(2, dtype);
    fusion_ptr->addInput(tv0);
    TensorView* tv1 = makeSymbolicTensor(2, dtype);
    fusion_ptr->addInput(tv1);

    TensorView* tv2 = castOp(DataType::Float, tv0);
    TensorView* tv3 = neg(tv2);
    TensorView* tv4 = castOp(dtype, tv3);

    TensorView* tv5 = cat({tv4, tv1}, -1);
    fusion_ptr->addOutput(tv5);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    // note randn doesn't support fp8 types. so we cast after initialize
    at::Tensor t0 = at::randn({4, 8}, options).to(data_type_to_aten(dtype));
    at::Tensor t1 = at::randn({4, 12}, options).to(data_type_to_aten(dtype));

    std::vector<c10::IValue> aten_inputs = {t0, t1};

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

    EXPECT_EQ(cg_outputs.size(), 1);
    EXPECT_EQ(cg_outputs[0].dtype(), data_type_to_aten(dtype));

    // note cat doesn't support fp8 types, running reference with floating point
    // instead.
    auto t0_fp32 = t0.to(at::kFloat);
    auto t1_fp32 = t1.to(at::kFloat);
    auto ref = at::cat({-t0_fp32, t1_fp32}, -1);

    testValidate(
        executor_cache.fusion(),
        {cg_outputs[0].to(at::kFloat)},
        aten_inputs,
        {ref},
        __LINE__,
        __FILE__,
        "");
  }
}

TEST_F(ResizeTest, PadDtypes) {
  auto sizes = {0, 10};
  auto dtypes = {
      at::kBool,
      at::kFloat,
      at::kLong,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kComplexFloat,
      at::kComplexDouble};

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* size = IrBuilder::create<Val>(DataType::Int);
  Val* fill_val = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(size);
  fusion->addInput(fill_val);
  for (auto dtype : dtypes) {
    if (!isSupportedTypeByDevice(aten_to_data_type(dtype))) {
      continue;
    }
    auto full_tv = full({size}, fill_val, aten_to_data_type(dtype));
    auto out_tv =
        pad(full_tv, {IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)});
    fusion->addOutput(out_tv);

    auto* pad_value = out_tv->definition()->as<PadOp>()->value();
    EXPECT_TRUE(pad_value->isZero());
    EXPECT_FALSE(pad_value->isOne());
  }

  FusionExecutorCache executor_cache(std::move(fusion));

  for (auto size : sizes) {
    auto cg_outputs = executor_cache.runFusionWithInputs({size, 8});

    testValidate(
        executor_cache.fusion(), cg_outputs, {size, 8}, __LINE__, __FILE__);
  }
}

TEST_F(ResizeTest, Issue2552) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigConcreteTensor({1, 3});
  TensorView* y = makeContigConcreteTensor({1, 3});
  fusion->addInput(x);
  fusion->addInput(y);
  x = expand(x, {IrBuilder::create<Val>(2), x->axis(1)->extent()});
  x = slice(x, /*starts=*/{0, 0}, /*stops=*/{1, 3});
  fusion->addOutput(x);
  TensorView* z = add(x, y);
  fusion->addOutput(z);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor x_tensor = at::randn({1, 3}, options);
  at::Tensor y_tensor = at::randn({1, 3}, options);
  std::vector<at::Tensor> out_tensors =
      fec.runFusionWithInputs({x_tensor, y_tensor});
  testValidate(
      fec.fusion(), out_tensors, {x_tensor, y_tensor}, __LINE__, __FILE__);
}

} // namespace nvfuser
