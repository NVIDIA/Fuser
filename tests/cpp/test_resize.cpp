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

#include <fusion.h>
#include <fusion_profiler.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <val_graph_visitor.h>

namespace nvfuser {

using ResizeTest = NVFuserFixtureParamTest<bool>;

using testing::Each;
using testing::HasSubstr;
using testing::Not;
using testing::Property;
using testing::ThrowsMessage;
using testing::UnorderedElementsAre;

INSTANTIATE_TEST_SUITE_P(
    ,
    ResizeTest,
    testing::Bool(),
    testing::PrintToStringParamName());

// Simple pad test
TEST_P(ResizeTest, Pad1) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + split
TEST_P(ResizeTest, Pad2) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad, merge + split, inlineMost
TEST_P(ResizeTest, Pad3) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// pad + parallelization
TEST_P(ResizeTest, Pad4) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + parallelization + RAW sync
TEST_P(ResizeTest, Pad5) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// pad + merge + split parallelization
TEST_P(ResizeTest, Pad6) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// pad + unswitch. Having different extents in an unswitched loop nest
// needs a special care (see UnrollPass::canOmitElseClause)
TEST_P(ResizeTest, Pad7) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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

TEST_P(ResizeTest, PadScheduler1) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {1, 1});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_P(ResizeTest, PadScheduler2) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_P(ResizeTest, PadScheduler4) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Pad a broadcast
// See https://github.com/NVIDIA/Fuser/issues/798
TEST_P(ResizeTest, PadBroadcastInput) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_F(ResizeTest, SliceConstantShmoo) {
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
TEST_F(ResizeTest, SliceInputShmoo) {
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

// Same as SliceInputShmoo but use FusionExecutorCache, which
// might re-concretize when output sizes change
TEST_F(ResizeTest, SliceInputShmooFusionExecutorCache) {
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

TEST_P(ResizeTest, PadReduceScheduler1) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_P(ResizeTest, PadWithValue) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0, {1, 1}, "constant", 2);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Same as Pad1 but pad by negative value to create an empty tensor
TEST_P(ResizeTest, PadToEmptyTensor) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = at::pad(t0, {-1, -1}, "constant", 2);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Test that padding Half tensor by Double does not promote output
TEST_P(ResizeTest, PadHalfWithDoubleValue) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_F(ResizeTest, SliceForNanoGPT3) {
  // To verify input caching condition in this test, disable aliasing as that
  // will skip compilation and no kernel will exist.
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MemoryPromotion);

  auto* in = makeSymbolicTensor(3);
  fusion->addInput(in);

  std::vector<TensorView*> slices = chunk(in, /*chunks=*/3, /*dim=*/-1);
  for (auto* slice : slices) {
    TensorView* out = reshape(slice, {16, 128, 1024}, {16, 128, 16, 64});
    // TODO: add permute
    fusion->addOutput(out);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in_tensor = at::randn({16, 128, 3072}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
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
  // Without the `add`, the fusion will be accepted by NoOp, defeating the
  // purpose of testing PointWise.
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
      UnorderedElementsAre(HeuristicIs(SchedulerType::Transpose)));
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
      ThrowsMessage<nvfError>(
          HasSubstr("must concretize to IterType::Broadcast but found")));
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

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, inputs).outputs;
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

  auto ref = t0.narrow(0, 1, N) + t1;

  // testValidate does not check that dtypes match
  EXPECT_EQ(cg_outputs[0].dtype(), ref.dtype());
  testValidate(&fusion, cg_outputs, inputs, __LINE__, __FILE__);
}

// Concretize a symbolic pad that results in a broadcast (static pads)
// In this test, the sizes and pad widths are static, so there should be nothing
// to concretize.
TEST_P(ResizeTest, ResizePadToBroadcastStatic) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_P(ResizeTest, ResizePadToBroadcastDynamic) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_P(ResizeTest, ResizePadToBroadcastIssue596) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
      ThrowsMessage<nvfError>(
          HasSubstr("with word size 2 not possible due to invalid stride")));
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
      ThrowsMessage<nvfError>(
          HasSubstr("with word size 4 not possible due to invalid stride")));
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
TEST_P(ResizeTest, ReshapeToPad) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

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
TEST_P(ResizeTest, PadExpandedEmpty) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(), cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we can pad properly along broadcast dims
// See https://github.com/NVIDIA/Fuser/issues/868
TEST_P(ResizeTest, PadOfBroadcast) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we can cat along broadcast dims that have been expanded
// See https://github.com/NVIDIA/Fuser/issues/868
TEST_P(ResizeTest, PadOfExpandedBroadcast) {
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

  EnableOptionsGuard enable_options_guard;
  if (GetParam()) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::IdModel);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, DynamicReshapeIssue1393) {
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
  auto cg_results = scheduleAndRun(&fusion, SchedulerType::PointWise, inputs);
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  ASSERT_EQ(pparams->vectorization_factor, 4)
      << "Unexpected factor of vectorization";

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

  testValidate(&fusion, cg_results.outputs, inputs, __LINE__, __FILE__);
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

TEST_F(ResizeTest, Chunk_NegativeSize) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(1);
  fusion->addInput(in);
  std::vector<TensorView*> outs = chunk(in, /*chunks=*/6, /*dim=*/0);
  for (auto* out : outs) {
    fusion->addOutput(out);
  }

  FusionExecutorCache fec(std::move(fusion));
  EXPECT_THAT(
      [&]() {
        auto in_tensor = at::randn({13}).cuda();
        fec.runFusionWithInputs({in_tensor});
      },
      ThrowsMessage<nvfError>(HasSubstr("Invalid resized domain extent")));
}

TEST_F(ResizeTest, Chunk_SizeZero) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(1);
  fusion->addInput(in);
  std::vector<TensorView*> outs = chunk(in, /*chunks=*/6, /*dim=*/0);
  for (auto* out : outs) {
    fusion->addOutput(out);
  }

  FusionExecutorCache fec(std::move(fusion));
  auto in_tensor = at::randn({15}).cuda();
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensors.back().numel(), 0);
}

TEST_F(ResizeTest, Chunk_Uneven) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(1);
  fusion->addInput(in);
  std::vector<TensorView*> outs = chunk(in, /*chunks=*/6, /*dim=*/0);
  for (auto* out : outs) {
    fusion->addOutput(out);
  }

  FusionExecutorCache fec(std::move(fusion));
  auto in_tensor = at::randn({16}).cuda();
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensors.back().numel(), 1);
}

// Schedule a slice with the loop domain derived from the producer
// domain. See PR #2897.
// Note that the IdModel-based indexing is automatically enabled as
// there are tensors that have non-trivial loop domains as defined by
// requiresIdModel in lower2device.cpp.
TEST_F(ResizeTest, SliceScheduledLikeProducer) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 =
      slice(tv0, {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(99)}});

  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 32);
  }

  inlineMost();

  for (auto tv : {tv1, tv2}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.index({at::indexing::Slice(1, shape[0] - 1)});

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, PadScheduledLikeConsumer) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1));
  auto tv2 = pad(tv1, {IrBuilder::create<Val>(1), IrBuilder::create<Val>(1)});

  auto tv3 = add(tv2, IrBuilder::create<Val>(1));
  fusion.addOutput(tv3);

  std::vector<IterDomain*> ref_loop = tv2->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 32);
  }

  inlineMost();

  for (auto tv : {tv1, tv2, tv3}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(t0 + 1, {1, 1}) + 1;

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

// Slicing the left half and pad it to the original extent
TEST_F(ResizeTest, SliceThenPadLeftHalf) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)}});

  auto tv3 = pad(tv2, {fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)});

  fusion.addOutput(tv3);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 32);
  }

  inlineMost();

  for (auto tv : {tv1, tv2, tv3}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(
      t0.index({at::indexing::Slice(0, shape[0] / 2)}), {0, shape[0] / 2});

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

// Slicing the right half and pad it to the original extent
TEST_F(ResizeTest, SliceThenPadRightHalf) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 = slice(
      tv1,
      {{IrBuilder::create<Val>(shape[0] / 2),
        IrBuilder::create<Val>(shape[0])}});

  auto tv3 = pad(tv2, {IrBuilder::create<Val>(shape[0] / 2), fusion.zeroVal()});

  fusion.addOutput(tv3);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 32);
  }

  inlineMost();

  for (auto tv : {tv1, tv2, tv3}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::pad(
      t0.index({at::indexing::Slice(shape[0] / 2, shape[0])}),
      {shape[0] / 2, 0});

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceThenConcat) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // TODO: Use cat instead of the manual pad + add

  // left half
  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)}});
  auto tv3 = pad(tv2, {fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)});

  // right half
  auto tv4 = slice(
      tv1,
      {{IrBuilder::create<Val>(shape[0] / 2),
        IrBuilder::create<Val>(shape[0])}});
  auto tv5 = pad(tv4, {IrBuilder::create<Val>(shape[0] / 2), fusion.zeroVal()});

  auto tv6 = add(tv3, tv5);

  fusion.addOutput(tv6);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(0, 32);
  }

  inlineMost();

  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  EXPECT_TRUE(t0.equal(cg_outputs[0]));
}

// RoPE pattern except for the rotation
TEST_F(ResizeTest, SliceSliceConcatConcat) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t i0 = 128;
  const int64_t rope_size = 32;

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto zero = fusion.zeroVal();

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor({i0});
  fusion.addInput(tv0);

  // [i0]
  auto tv1 = set(tv0);

  // [rope_size]
  auto tv2 = slice(tv1, {{zero, IrBuilder::create<Val>(rope_size)}});

  auto rope_size_half = IrBuilder::create<Val>(rope_size / 2);

  // first half
  // [0:rope_size/2]
  auto tv3 = slice(tv2, {{zero, rope_size_half}});
  // do some uop
  auto tv4 = add(tv3, IrBuilder::create<Val>(1));
  // Pad back
  // [0:rope_size]
  auto tv5 = pad(tv4, {zero, rope_size_half});

  // second half
  // [rope_size/2:]
  auto tv6 = slice(tv2, {{rope_size_half, IrBuilder::create<Val>(rope_size)}});

  // do some uop
  auto tv7 = add(tv6, IrBuilder::create<Val>(2));
  // Pad back
  // [rope_size]
  auto tv8 = pad(tv7, {rope_size_half, zero});

  // [rope_size]
  auto tv9 = add(tv5, tv8);

  // [i0]
  auto tv10 = pad(tv9, {zero, IrBuilder::create<Val>(i0 - rope_size)});

  // [rope_size:]
  auto tv11 = slice(
      tv1, {{IrBuilder::create<Val>(rope_size), IrBuilder::create<Val>(i0)}});
  // [i0]
  auto tv12 = pad(tv11, {IrBuilder::create<Val>(rope_size), zero});

  auto tv13 = add(tv10, tv12);

  fusion.addOutput(tv13);

  std::vector<IterDomain*> ref_loop = tv0->getLogicalDomain();
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop);

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    tv->split(0, 4);
    tv->split(0, 16);
  }

  inlineMost();

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    EXPECT_EQ(tv->getComputeAtPosition(), 3)
        << "Invalid computeAt position: " << tv->toString();
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({i0}, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::concat(
      {at::slice(t0, 0, 0, rope_size / 2) + 1,
       at::slice(t0, 0, rope_size / 2, rope_size) + 2,
       at::slice(t0, 0, rope_size)},
      0);

  NVF_CHECK(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceConcatAdd) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({8});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // TODO: Use cat instead of the manual pad + add

  // left half
  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)}});
  auto tv3 = pad(tv2, {fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)});

  // right half
  auto tv4 = slice(
      tv1,
      {{IrBuilder::create<Val>(shape[0] / 2),
        IrBuilder::create<Val>(shape[0])}});
  auto tv5 = pad(tv4, {IrBuilder::create<Val>(shape[0] / 2), fusion.zeroVal()});

  auto tv6 = add(tv3, tv5);

  auto tv7 = add(tv1, tv6);

  fusion.addOutput(tv7);

#if 0
  tv2->setLoopDomain(tv2->getRootDomain());

  if (false) {
    std::vector<IterDomain*> tv3_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv3->getRootDomain().at(0),
        tv3_loop.at(0),
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    tv3->setLoopDomain(tv3_loop);
  }

  tv4->setLoopDomain(tv4->getRootDomain());

  if (false) {
    std::vector<IterDomain*> tv5_loop{
        tv4->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv5->getRootDomain().at(0),
        tv5_loop.at(0),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index),
        fusion.zeroVal());
    tv5->setLoopDomain(tv5_loop);
  }

  if (false) {
    std::vector<IterDomain*> tv6_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    auto left_half = IterDomain::resize(
        tv6_loop[0],
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    IrBuilder::create<Resize>(
        tv6->getLogicalDomain().at(0),
        left_half,
        fusion.zeroVal(),
        IrBuilder::create<Val>(shape[0] / 2, DataType::Index));
    tv6->setLoopDomain(tv6_loop);
  }

#if 0
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
#endif
#endif
  // inlineMost();

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0 + t0;

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceThenRotate) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({8});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // TODO: Use cat instead of the manual pad + add

  // left half
  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)}});

  auto tv3 = pad(tv2, {IrBuilder::create<Val>(shape[0] / 2), fusion.zeroVal()});

  // right half
  auto tv4 = slice(
      tv1,
      {{IrBuilder::create<Val>(shape[0] / 2),
        IrBuilder::create<Val>(shape[0])}});
  auto tv5 = pad(tv4, {fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)});

  auto tv6 = add(tv5, tv3);

  fusion.addOutput(tv6);

  if (getenv("LOOP")) {
    tv2->setLoopDomain(tv2->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv3_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv3->getRootDomain().at(0),
        tv3_loop.at(0),
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    tv3->setLoopDomain(tv3_loop);
  }

  if (getenv("LOOP")) {
    tv4->setLoopDomain(tv4->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv5_loop{
        tv4->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv5->getRootDomain().at(0),
        tv5_loop.at(0),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index),
        fusion.zeroVal());
    tv5->setLoopDomain(tv5_loop);
  }

  if (false) {
    std::vector<IterDomain*> tv6_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    auto left_half = IterDomain::resize(
        tv6_loop[0],
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    IrBuilder::create<Resize>(
        tv6->getLogicalDomain().at(0),
        left_half,
        fusion.zeroVal(),
        IrBuilder::create<Val>(shape[0] / 2, DataType::Index));
    tv6->setLoopDomain(tv6_loop);
  }

#if 0
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
#endif

  // inlineMost();

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = at::cat(
      {t0.index({at::indexing::Slice(shape[0] / 2, at::indexing::None)}),
       t0.index({at::indexing::Slice(0, shape[0] / 2)})});

  std::cout << "input:\n" << t0 << "\n";
  std::cout << "ref:\n" << ref << "\n";
  std::cout << "result:\n" << cg_outputs[0] << "\n";

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceRotateThenConcat) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({8});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // TODO: Use cat instead of the manual pad + add

  // left half
  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)}});

  auto tv3 = pad(tv2, {IrBuilder::create<Val>(shape[0] / 2), fusion.zeroVal()});

  // right half
  auto tv4 = slice(
      tv1,
      {{IrBuilder::create<Val>(shape[0] / 2),
        IrBuilder::create<Val>(shape[0])}});
  auto tv5 = pad(tv4, {fusion.zeroVal(), IrBuilder::create<Val>(shape[0] / 2)});

  auto tv6 = add(tv5, tv3);

  auto tv7 = add(tv1, tv6);

  fusion.addOutput(tv7);

  // This doesn't work...
  if (getenv("LOOP")) {
    tv2->setLoopDomain(tv2->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv3_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv3->getRootDomain().at(0),
        tv3_loop.at(0),
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    tv3->setLoopDomain(tv3_loop);
  }

  if (getenv("LOOP")) {
    tv4->setLoopDomain(tv4->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv5_loop{
        tv4->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv5->getRootDomain().at(0),
        tv5_loop.at(0),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index),
        fusion.zeroVal());
    tv5->setLoopDomain(tv5_loop);
  }

  if (false) {
    std::vector<IterDomain*> tv6_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    auto left_half = IterDomain::resize(
        tv6_loop[0],
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    IrBuilder::create<Resize>(
        tv6->getLogicalDomain().at(0),
        left_half,
        fusion.zeroVal(),
        IrBuilder::create<Val>(shape[0] / 2, DataType::Index));
    tv6->setLoopDomain(tv6_loop);
  }

#if 0
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
#endif

  // inlineMost();

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref =
      t0 +
      at::cat(
          {t0.index({at::indexing::Slice(shape[0] / 2, at::indexing::None)}),
           t0.index({at::indexing::Slice(0, shape[0] / 2)})});

  std::cout << "input:\n" << t0 << "\n";
  std::cout << "ref:\n" << ref << "\n";
  std::cout << "result:\n" << cg_outputs[0] << "\n";

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, SliceSliceRotateConcat) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({32});
  const int64_t rope_size = 8;

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 =
      slice(tv1, {{fusion.zeroVal(), IrBuilder::create<Val>(rope_size)}});

  // left half
  auto tv3 =
      slice(tv2, {{fusion.zeroVal(), IrBuilder::create<Val>(rope_size / 2)}});

  auto tv4 =
      pad(tv3, {IrBuilder::create<Val>(rope_size / 2), fusion.zeroVal()});

  // right half
  auto tv5 = slice(
      tv2,
      {{IrBuilder::create<Val>(rope_size / 2),
        IrBuilder::create<Val>(rope_size)}});
  auto tv6 = pad(tv5, {fusion.zeroVal(), IrBuilder::create<Val>(rope_size)});

  // concat
  auto tv7 = add(tv6, tv4);

  // auto tv7 = add(tv1, tv6);

  fusion.addOutput(tv7);

  // This doesn't work...
  if (getenv("LOOP")) {
    tv2->setLoopDomain(tv2->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv3_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv3->getRootDomain().at(0),
        tv3_loop.at(0),
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    tv3->setLoopDomain(tv3_loop);
  }

  if (getenv("LOOP")) {
    tv4->setLoopDomain(tv4->getRootDomain());
  }

  if (false) {
    std::vector<IterDomain*> tv5_loop{
        tv4->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    IrBuilder::create<Resize>(
        tv5->getRootDomain().at(0),
        tv5_loop.at(0),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index),
        fusion.zeroVal());
    tv5->setLoopDomain(tv5_loop);
  }

  if (false) {
    std::vector<IterDomain*> tv6_loop{
        tv2->getRootDomain()[0]->cloneWithoutRFactor(),
    };
    auto left_half = IterDomain::resize(
        tv6_loop[0],
        fusion.zeroVal(),
        IrBuilder::create<Val>(-shape[0] / 2, DataType::Index));
    IrBuilder::create<Resize>(
        tv6->getLogicalDomain().at(0),
        left_half,
        fusion.zeroVal(),
        IrBuilder::create<Val>(shape[0] / 2, DataType::Index));
    tv6->setLoopDomain(tv6_loop);
  }

#if 0
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
#endif

  // inlineMost();

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref =
      t0 +
      at::cat(
          {t0.index({at::indexing::Slice(shape[0] / 2, at::indexing::None)}),
           t0.index({at::indexing::Slice(0, shape[0] / 2)})});

  std::cout << "input:\n" << t0 << "\n";
  std::cout << "ref:\n" << ref << "\n";
  std::cout << "result:\n" << cg_outputs[0] << "\n";

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, ReshapeSliceSliceRotateConcat) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({16});
  const int64_t rope_size = 4;
  std::vector<int64_t> shape2({shape[0] / rope_size, rope_size});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  // [32]
  auto tv1 = set(tv0);

  // [4, 4]
  auto tv2 = reshape(tv1, shape, shape2);

  // [1, 4]
  auto tv3 = slice(
      tv2,
      {{fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])}});

  // left half
  // [1, 2]
  auto tv4 = slice(
      tv3,
      {{fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1] / 2)}});
  // [1, 4]
  auto tv5 =
      pad(tv4, {IrBuilder::create<Val>(shape2[1] / 2), fusion.zeroVal()});

  // right half
  // [1, 2]
  auto tv6 = slice(
      tv3,
      {{fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {IrBuilder::create<Val>(shape2[1] / 2),
        IrBuilder::create<Val>(shape2[1])}});
  // [1, 4]
  auto tv7 =
      pad(tv6, {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1] / 2)});

  // concat
  // [1, 4]
  auto tv8 = add(tv7, tv5);

  // [4, 4]
  auto tv9 =
      pad(tv8,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(3)});

  // [3, 4]
  auto tv10 = slice(
      tv2,
      {{IrBuilder::create<Val>(1), IrBuilder::create<Val>(shape2[0])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])}});
  auto tv11 =
      pad(tv10,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(1),
           fusion.zeroVal()});

  auto tv12 = add(tv9, tv11);

  auto tv13 = reshape(tv12, shape2, shape);

  fusion.addOutput(tv13);

  fusion.print();

  if (getenv("MANUAL")) {
    tv1->split(0, 4, false);

    tv3->setLoopDomain(tv3->getRootDomain());

    // tv4
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv4->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      if (getenv("INNER")) {
        // This doesn't work
        tv4->setLoopDomain({outer_root, tv4->getRootDomain().at(1)});
      } else {
        tv4->setLoopDomain({outer_root, tv4->getLogicalDomain().at(1)});
      }
    }

    // tv5
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv5->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv5->setLoopDomain({outer_root, tv5->getLogicalDomain().at(1)});
    }

    // tv6
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv6->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv6->setLoopDomain({outer_root, tv6->getLogicalDomain().at(1)});
    }

    // tv7
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv7->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv7->setLoopDomain({outer_root, tv7->getLogicalDomain().at(1)});
    }

    // tv8
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv8->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv8->setLoopDomain({outer_root, tv8->getLogicalDomain().at(1)});
    }

    // tv9
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv9->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv9->setLoopDomain({outer_root, tv9->getLogicalDomain().at(1)});
    }

    // tv10
    tv10->setLoopDomain(tv10->getRootDomain());

    // tv11
    {
      auto outer_root = tv10->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv10 root-to-logical resize
      IrBuilder::create<Resize>(
          tv11->getMaybeRootDomain().at(0),
          outer_root,
          IrBuilder::create<Val>(-1, DataType::Index),
          fusion.zeroVal());
      tv11->setLoopDomain({outer_root, tv11->getLogicalDomain().at(1)});
    }

    // tv12
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      IrBuilder::create<Resize>(
          tv12->getMaybeRootDomain().at(0),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(shape2[0] - 1, DataType::Index));
      tv12->setLoopDomain({outer_root, tv12->getLogicalDomain().at(1)});
    }

    // tv13
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      IrBuilder::create<Resize>(
          tv13->getMaybeRootDomain().at(0),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(shape2[0] - 1, DataType::Index));
      tv13->setLoopDomain({outer_root, tv13->getRootDomain().at(1)});
    }
  } else {
    std::vector<IterDomain*> ref_loop = tv2->getLogicalDomain();
    scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, 1);
  }

  fusion.printMath();
#if 1
  for (auto tv : fusion.allTvs()) {
    if (tv->nDims() == 2) {
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }
  }

  if (getenv("INLINE")) {
    // inlineMost();
    for (auto tv : fusion.allTvs()) {
      if (tv->nDims() == 2) {
        tv->inlineAt(1);
      }
    }
  }

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
#endif
  // fusion.print();
  // fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index({at::indexing::Slice(0, rope_size)});
  auto t2 = t1.index({at::indexing::Slice(0, rope_size / 2)});
  auto t3 = t1.index({at::indexing::Slice(rope_size / 2, rope_size)});
  auto t4 = at::cat({t3, t2});
  auto t5 = t0.index({at::indexing::Slice(rope_size)});
  auto t6 = at::cat({t4, t5});
  auto ref = t6;

  std::cout << "input:\n" << t0 << "\n";
  std::cout << "ref:\n" << ref << "\n";
  std::cout << "result:\n" << cg_outputs[0] << "\n";

  EXPECT_TRUE(ref.equal(cg_outputs[0]));
}

TEST_F(ResizeTest, ReshapeSliceSliceRotateConcat16) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t rope_size = 16;
  std::vector<int64_t> shape1({rope_size * 4});
  std::vector<int64_t> shape2({shape1[0] / rope_size, 2, rope_size / 2});

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  // [rope_size * 4]
  auto tv1 = set(tv0);

  // [4, 2, rope_size / 2]
  auto tv2 = reshape(tv1, shape1, shape2);

  // [1, 2, rope_size / 2]
  auto tv3 = slice(
      tv2,
      {
          {fusion.zeroVal(), IrBuilder::create<Val>(1)},
          {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
          {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
      });

  // left half
  // [1, 1, rope_size / 2]
  auto tv4 = slice(
      tv3,
      {{fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])}});
  // [1, 2, rope_size / 2]
  auto tv5 =
      pad(tv4,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(1),
           fusion.zeroVal()});

  // right half
  // [1, 1, rope_size / 2]
  auto tv6 = slice(
      tv3,
      {{fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {IrBuilder::create<Val>(1), IrBuilder::create<Val>(shape2[1])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])}});
  // [1, 2, rope_size / 2]
  auto tv7 =
      pad(tv6,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(1)});

  // concat
  // [1, 2, rope_size / 2]
  auto tv8 = add(tv7, tv5);

  // [4, 2, rope_size / 2]
  auto tv9 =
      pad(tv8,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(3)});

  // [3, 2, rope_size / 2]
  auto tv10 = slice(
      tv2,
      {
          {IrBuilder::create<Val>(1), IrBuilder::create<Val>(shape2[0])},
          {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
          {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
      });
  // [4, 2, rope_size / 2]
  auto tv11 =
      pad(tv10,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(1),
           fusion.zeroVal()});

  auto tv12 = add(tv9, tv11);

  auto tv13 = reshape(tv12, shape2, shape1);

  fusion.addOutput(tv13);

  if (getenv("MANUAL")) {
    tv1->split(0, 4, false);
    tv1->split(1, 2, false);

    tv3->setLoopDomain(tv3->getRootDomain());

    // tv4
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv4->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv4->setLoopDomain({
          outer_root,
          tv4->getLogicalDomain().at(1),
          tv4->getLogicalDomain().at(2),
      });
    }

    // tv5
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv5->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv5->setLoopDomain(
          {outer_root,
           tv5->getLogicalDomain().at(1),
           tv5->getLogicalDomain().at(2)});
    }

    // tv6
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv6->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv6->setLoopDomain(
          {outer_root,
           tv6->getLogicalDomain().at(1),
           tv6->getLogicalDomain().at(2)});
    }

    // tv7
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv7->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv7->setLoopDomain(
          {outer_root,
           tv7->getLogicalDomain().at(1),
           tv7->getLogicalDomain().at(2)});
    }

    // tv8
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv8->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv8->setLoopDomain(
          {outer_root,
           tv8->getLogicalDomain().at(1),
           tv8->getLogicalDomain().at(2)});
    }

    // tv9
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      IrBuilder::create<Resize>(
          tv9->getMaybeRootDomain().at(0),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      tv9->setLoopDomain(
          {outer_root,
           tv9->getLogicalDomain().at(1),
           tv9->getLogicalDomain().at(2)});
    }

    // tv10
    tv10->setLoopDomain(tv10->getRootDomain());

    // tv11
    {
      auto outer_root = tv10->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv10 root-to-logical resize
      IrBuilder::create<Resize>(
          tv11->getMaybeRootDomain().at(0),
          outer_root,
          IrBuilder::create<Val>(-1, DataType::Index),
          fusion.zeroVal());
      tv11->setLoopDomain(
          {outer_root,
           tv11->getLogicalDomain().at(1),
           tv11->getLogicalDomain().at(2)});
    }

    // tv12
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      IrBuilder::create<Resize>(
          tv12->getMaybeRootDomain().at(0),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(shape2[0] - 1, DataType::Index));
      tv12->setLoopDomain(
          {outer_root,
           tv12->getLogicalDomain().at(1),
           tv12->getLogicalDomain().at(2)});
    }

    // tv13
    {
      auto outer_root = tv3->getRootDomain().at(0)->cloneWithoutRFactor();
      // Replay tv3 root-to-logical resize
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(-shape2[0] + 1, DataType::Index));
      IrBuilder::create<Resize>(
          tv13->getMaybeRootDomain().at(0),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(shape2[0] - 1, DataType::Index));
      tv13->setLoopDomain(
          {outer_root,
           tv13->getRootDomain().at(1),
           tv13->getRootDomain().at(2)});
    }
  } else {
    std::vector<IterDomain*> ref_loop = tv2->getLogicalDomain();
    // Don't want to change the loop domain of the rotated axis
    std::swap(ref_loop[1], ref_loop[2]);
    scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, -2);
    for (auto tv : fusion.allTvs()) {
      // Reverse the reordering
      tv->reorder({{-2, -1}});
    }

    fusion.printMath();
    fusion.print();
  }

  for (auto tv : fusion.allTvs()) {
    if (tv->nDims() == 3) {
      tv->split(-1, 4);
      tv->merge(0, -2);
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }
  }

  if (getenv("INLINE")) {
    // inlineMost();
    for (auto tv : fusion.allTvs()) {
      if (tv->nDims() == 2) {
        tv->inlineAt(1);
      }
    }
  }

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto t1 = t0.index({at::indexing::Slice(0, rope_size)});
  auto t2 = t1.index({at::indexing::Slice(0, rope_size / 2)});
  auto t3 = t1.index({at::indexing::Slice(rope_size / 2, rope_size)});
  auto t4 = at::cat({t3, t2});
  auto t5 = t0.index({at::indexing::Slice(rope_size)});
  auto t6 = at::cat({t4, t5});
  auto ref = t6;

  EXPECT_TRUE(ref.equal(cg_outputs[0])) << "input:\n"
                                        << t0
                                        << "\n"
                                           "ref:\n"
                                        << ref << "\n"
                                        << "result:\n"
                                        << cg_outputs[0] << "\n";
}

// def rope_one_entry(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
// rope_n_elem: int) -> torch.Tensor:
//     x_rope = x[..., : rope_n_elem]
//     x1 = x_rope[..., : rope_n_elem // 2]  # (B, nh, T, hs/2)
//     x2 = x_rope[..., rope_n_elem // 2 :]  # (B, nh, T, hs/2)
//     rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
//     roped = (x_rope * cos) + (rotated * sin)
//     roped.to(dtype=x.dtype)
//     return torch.cat((roped, x[..., rope_n_elem :]), dim=-1)
TEST_F(ResizeTest, RoPE4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // The innermost dimension should be 32, but for now, fp32 is used
  // instead of fp16 for simplicity, and thus the innermost dimension
  // is also reduced by half
  std::vector<int64_t> shape1({2, 16, 1024, 32 / 2});
  // std::vector<int64_t> shape1({2, 2, 16, 32 / 2});
  const int64_t rope_size = 4;

  if (auto env = getenv("BSZ")) {
    shape1[0] = std::atoi(env);
  }

  if (auto env = getenv("N_HEAD")) {
    shape1[1] = std::atoi(env);
  }

  if (auto env = getenv("BLOCK_SIZE")) {
    shape1[2] = std::atoi(env);
  }

  if (auto env = getenv("HEAD_SIZE")) {
    shape1[3] = std::atoi(env) / 2;
  }

  std::vector<int64_t> shape2(
      {shape1[0], shape1[1], shape1[2], shape1[3] / rope_size, rope_size});

  std::cerr << "shape1: " << shape1 << "\n";
  std::cerr << "shape2: " << shape2 << "\n";

  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  // concrete shapes to avoid dynamic Fusion

  // TODO: Use bfloat16

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);
  // cos
  auto tv1 = makeContigConcreteTensor({shape1[2], rope_size});
  fusion.addInput(tv1);
  // sin
  auto tv2 = makeContigConcreteTensor({shape1[2], rope_size});
  fusion.addInput(tv2);

  std::cerr << "Inputs: " << tv0->toString() << ", " << tv1->toString() << ", "
            << tv2->toString() << "\n";

  auto tv3 = set(tv0);

  auto tv4 = reshape(tv3, shape1, shape2);
  auto x = tv4;

  // x_rope
  auto tv5 = slice(
      x,
      {{fusion.zeroVal(), IrBuilder::create<Val>(shape2[0])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
       {fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[4])}});
  auto x_rope = tv5;

  std::cerr << "x_rope: " << x_rope->toString() << "\n";

  // x1
  auto tv6 = slice(
      x_rope,
      {{fusion.zeroVal(), IrBuilder::create<Val>(shape2[0])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
       {fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[4] / 2)}});
  std::cerr << "x1: " << tv6->toString() << "\n";
  auto tv7 =
      pad(tv6, {IrBuilder::create<Val>(shape2[4] / 2), fusion.zeroVal()});

  // x2
  auto tv8 = slice(
      x_rope,
      {{fusion.zeroVal(), IrBuilder::create<Val>(shape2[0])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
       {fusion.zeroVal(), IrBuilder::create<Val>(1)},
       {IrBuilder::create<Val>(shape2[4] / 2),
        IrBuilder::create<Val>(shape2[4])}});
  std::cerr << "x2: " << tv8->toString() << "\n";
  auto tv9 =
      pad(tv8, {fusion.zeroVal(), IrBuilder::create<Val>(shape2[4] / 2)});

  // rotated
  auto tv10 = add(tv9, tv7);
  auto rotated = tv10;

  std::cerr << "rotated: " << rotated->toString() << "\n";

  // x_rope * cos
  auto tv11 = broadcast(tv1, {true, true, false, true, false});
  auto tv12 = mul(x_rope, tv11);

  // rotated * sin
  auto tv13 = broadcast(tv2, {true, true, false, true, false});
  auto tv14 = mul(rotated, tv13);

  // roped
  auto tv15 = add(tv12, tv14);
  auto roped = tv15;

  auto tv16 =
      pad(roped,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(shape2[3] - 1)});

  // x[..., rope_n_elem :]
  auto tv17 = slice(
      x,
      {{fusion.zeroVal(), IrBuilder::create<Val>(shape2[0])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[1])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[2])},
       {IrBuilder::create<Val>(1), IrBuilder::create<Val>(shape2[3])},
       {fusion.zeroVal(), IrBuilder::create<Val>(shape2[4])}});
  auto tv18 =
      pad(tv17,
          {fusion.zeroVal(),
           fusion.zeroVal(),
           IrBuilder::create<Val>(1),
           fusion.zeroVal()});

  auto tv19 = add(tv16, tv18);

  auto tv20 = reshape(tv19, shape2, shape1);

  auto tv21 = set(tv20);

  fusion.addOutput(tv21);

  fusion.printMath();

  if (getenv("MANUAL")) {
    tv3->split(-1, shape2[3], false);

    int64_t x_rope_slice_dim = 3;

    auto ref_id = x->getLogicalDomain().at(x_rope_slice_dim);

    tv5->setLoopDomain(tv5->getRootDomain());

    // tv6
    {
      auto tv = tv6;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv7
    {
      auto tv = tv7;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv8
    {
      auto tv = tv8;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv9
    {
      auto tv = tv9;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv10
    {
      auto tv = tv10;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv11, tv12, tv13, tv14, tv15
    for (auto tv : {tv11, tv12, tv13, tv14, tv15}) {
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      // Use concrete domain for broadcast
      if (tv == tv11 || tv == tv13) {
        for (const auto i : c10::irange(2)) {
          ASSERT_TRUE(loop_domain.at(i)->isBroadcast());
          loop_domain.at(i) =
              x->getLoopDomain().at(i)->cloneWithoutRFactor(true);
        }
      }
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv16
    {
      auto tv = tv16;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    tv17->setLoopDomain(tv17->getRootDomain());

    // tv18
    {
      auto tv = tv18;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          outer_root,
          IrBuilder::create<Val>(-1, DataType::Index),
          fusion.zeroVal());
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv19
    {
      auto tv = tv19;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              shape2[x_rope_slice_dim] - 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getLoopDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv20
    {
      auto tv = tv20;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      auto resize = IrBuilder::create<Resize>(
          tv->getMaybeRootDomain().at(x_rope_slice_dim),
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              shape2[x_rope_slice_dim] - 1, DataType::Index));
      std::cerr << resize->toString();
      auto loop_domain = tv->getRootDomain();
      loop_domain.at(x_rope_slice_dim) = outer_root;
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }

    // tv21
    {
      auto tv = tv21;
      auto outer_root = ref_id->cloneWithoutRFactor();
      auto sliced_root = IterDomain::resize(
          outer_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              -shape2[x_rope_slice_dim] + 1, DataType::Index));
      auto padded_root = IterDomain::resize(
          sliced_root,
          fusion.zeroVal(),
          IrBuilder::create<Val>(
              shape2[x_rope_slice_dim] - 1, DataType::Index));

      auto inner_root = tv20->getRootDomain().at(4)->cloneWithoutRFactor();
      auto merge = IrBuilder::create<Merge>(
          tv->getMaybeRootDomain().at(3), padded_root, inner_root);
      std::cerr << merge->toString();
      auto loop_domain = tv->getLogicalDomain();
      loop_domain.at(3) = outer_root;
      loop_domain.push_back(inner_root);
      tv->setLoopDomain(loop_domain);
      std::cout << tv->toString() << "\n";
      tv->printTransforms();
      std::cout << std::endl;
    }
  } else {
    std::vector<IterDomain*> ref_loop = x->getLogicalDomain();
    scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, -2);
  }

  fusion.print();

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    // [i0, i1, i2, i3, i4]
    ASSERT_EQ(tv->getLoopDomain().size(), 5);

    // [i0*i1*i2*i3, i4]
    tv->merge(0)->merge(0)->merge(0);

    // Let i4 as is since it's resized
    tv->split(0, 256);
  }

  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }
    tv->inlineAt(-2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv21->axis(-1)->parallelize(ParallelType::Vectorize);

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn({shape1[2], rope_size}, options);
  auto t2 = at::randn({shape1[2], rope_size}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  if (getenv("BENCHMARK")) {
    int64_t mem_size = sizeof(float);
    for (const auto s : shape1) {
      mem_size *= s;
    }
    mem_size *= 2;

    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
      cg_outputs = fe.runFusion(aten_inputs);
      FusionProfiler::stop();
      auto t = FusionProfiler::profile().kernel_time_ms;
      std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      std::cout << "Bandwidth (GB/s): "
                << ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
                << "\n";
    }
  }
}

TEST_F(ResizeTest, RoPEFull) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Llama-2-7b-hf
  int64_t n_head = 32;
  int64_t head_size = 128;
  int64_t n_query_groups = 32;
  int64_t rope_n_elem = 128;
  int64_t batches = 2;
  int64_t seq_length = 4096;

  int64_t q_per_kv = n_head / n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  if (getenv("SMALL")) {
    n_head = 4;
    n_query_groups = 4;
    seq_length = 8;
    head_size = 16;
    rope_n_elem = 16;
  }

  std::vector<int64_t> shape1{
      batches, n_query_groups, total_qkv, seq_length, head_size};

  // head_size is assumed to be divisible by rope_n_elem, but it's
  // likely this can be lifted
  std::vector<int64_t> shape2{
      batches,
      n_query_groups,
      total_qkv,
      seq_length,
      head_size / rope_n_elem,
      rotation_num_splits,
      rope_n_elem / rotation_num_splits};

  bool has_rope_reshape = head_size != rope_n_elem;

  if (!has_rope_reshape) {
    shape2.erase(shape2.begin() + 4);
  }

  std::cerr << "shape1: " << shape1 << "\n";
  std::cerr << "shape2: " << shape2 << "\n";

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 = makeContigConcreteTensor({seq_length, rope_n_elem});
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 = makeContigConcreteTensor({seq_length, rope_n_elem});
  fusion.addInput(tv2);
  auto sin = tv2;

  auto tv3 = set(tv0);
  auto tv4 = reshape(tv3, shape1, shape2);
  // Due to the broadcast, there will be tv5
  auto tv5 = tv4;

  auto qkv = tv5;

  std::cerr << "qkv: " << qkv->toString() << "\n";

  cos = reshape(
      cos,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "cos: " << cos->toString() << "\n";

  sin = reshape(
      sin,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "sin: " << sin->toString() << "\n";

  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape2.size());
  for (const auto s : shape2) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  // tv5 (q)
  TensorView* tv6 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);
    tv6 = slice(qkv, slice_arg);
  }
  auto q = tv6;

  // tv6 (k)
  TensorView* tv7 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);
    tv7 = slice(qkv, slice_arg);
  }
  auto k = tv7;

  // tv6 (v)
  TensorView* tv8 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv + 1);
    tv8 = slice(qkv, slice_arg);
  }
  auto v = tv8;

  TensorView* tv9 = nullptr;
  {
    auto cur_shape = shape2;
    cur_shape[qkv_slice_dim] = q_per_kv;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape2.begin() + 3, shape2.end());
    tv9 = reshape(q, cur_shape, new_shape);
  }
  q = tv9;

  TensorView* tv10 = nullptr;
  {
    auto cur_shape = shape2;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape2.begin() + 3, shape2.end());
    tv10 = reshape(k, cur_shape, new_shape);
  }
  k = tv10;

  TensorView* tv11 = nullptr;
  {
    auto cur_shape = shape2;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape2.begin() + 3, shape2.end());
    tv11 = reshape(v, cur_shape, new_shape);
  }
  v = tv11;

  // x: q, k or v
  // 1. take [..., :rope_n_elem]
  // 2. apply_rope
  // 3. concat apply_rope and [..., rope_n_elem:]
  auto apply_rope = [&](TensorView* x) -> TensorView* {
    const int64_t rope_n_elem_slice_dim = (int64_t)x->nDims() - 3;
    const int64_t rotation_dim = (int64_t)x->nDims() - 2;

    std::vector<Slice> slice_arg;
    slice_arg.reserve(x->nDims());
    for (const auto id : x->getLogicalDomain()) {
      Slice arg;
      arg.start = zero;
      arg.stop = id->extent();
      slice_arg.push_back(arg);
    }

    TensorView* x_rope = x;
    if (has_rope_reshape) {
      auto x_rope_slice_arg = slice_arg;
      x_rope_slice_arg.at(rope_n_elem_slice_dim).stop = one;
      x_rope = slice(x, x_rope_slice_arg);
    }

    // x1
    NVF_ERROR(rotation_num_splits == 2);
    auto x1_slice_arg = slice_arg;
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x1_slice_arg.at(rotation_dim).stop = one;
    auto x1 = slice(x_rope, x1_slice_arg);
    std::cerr << "x1: " << x1->toString() << "\n";
    auto x1_padded = pad(x1, {zero, zero, one, zero});

    // x2
    auto x2_slice_arg = slice_arg;
    NVF_ERROR(rotation_num_splits == 2);
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x2_slice_arg.at(rotation_dim).start = one;
    auto x2 = slice(x_rope, x2_slice_arg);
    std::cerr << "x2: " << x2->toString() << "\n";
    auto x2_padded = pad(x2, {zero, zero, zero, one});

    auto rotated = add(x2_padded, x1_padded);
    std::vector<bool> bcast_flags(x->nDims(), false);
    bcast_flags[0] = true;
    bcast_flags[1] = true;
    if (has_rope_reshape) {
      bcast_flags[3] = true;
    }
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);
    std::cerr << "x_rope: " << x_rope->toString() << "\n";
    std::cerr << "cos_bc: " << cos_broadcast->toString() << "\n";
    auto apply_rope_result =
        add(mul(x_rope, cos_broadcast), mul(rotated, sin_broadcast));
    std::cerr << "apply_rope_result: " << apply_rope_result->toString() << "\n";
    TensorView* padded_apply_rope_result = apply_rope_result;
    if (has_rope_reshape) {
      padded_apply_rope_result =
          pad(apply_rope_result,
              {zero,
               zero,
               zero,
               zero,
               zero,
               IrBuilder::create<Val>(shape2.at(4) - 1)});
    }
    std::cerr << "Padded: " << padded_apply_rope_result->toString() << "\n";

    auto out = padded_apply_rope_result;

    if (has_rope_reshape) {
      // [..., rope_n_elem:]
      auto x_remaining_slice_arg = slice_arg;
      x_remaining_slice_arg.at(rope_n_elem_slice_dim).start = one;
      auto x_remaining = slice(x, x_remaining_slice_arg);
      auto padded_x_remaining =
          pad(x_remaining, {zero, zero, zero, zero, one, zero});
      out = add(out, padded_x_remaining);
    }

    return out;
  };

  q = apply_rope(q);
  k = apply_rope(k);
  // Not used but just for clarity
  // v = apply_rope(v);

  std::vector<int64_t> reverse_reshape_input_shape;
  reverse_reshape_input_shape.reserve(q->nDims());
  reverse_reshape_input_shape.push_back(batches);
  reverse_reshape_input_shape.push_back(n_query_groups * q_per_kv);
  reverse_reshape_input_shape.push_back(seq_length);
  if (has_rope_reshape) {
    reverse_reshape_input_shape.push_back(head_size / rope_n_elem);
  }
  reverse_reshape_input_shape.push_back(rotation_num_splits);
  reverse_reshape_input_shape.push_back(rope_n_elem / rotation_num_splits);

  auto q_original_shape = reshape(
      q,
      reverse_reshape_input_shape,
      {batches, n_query_groups * q_per_kv, seq_length, -1});

  reverse_reshape_input_shape[1] = n_query_groups;
  [[maybe_unused]] auto k_original_shape = reshape(
      k,
      reverse_reshape_input_shape,
      {batches, n_query_groups, seq_length, -1});

  [[maybe_unused]] auto v_original_shape = reshape(
      v,
      reverse_reshape_input_shape,
      {batches, n_query_groups, seq_length, -1});

  fusion.addOutput(q_original_shape);
  // Disabled for now
  // fusion.addOutput(k_original_shape);
  // fusion.addOutput(v_original_shape);

  fusion.printMath();

  std::vector<IterDomain*> ref_loop = qkv->getLogicalDomain();
  std::swap(ref_loop.at(ref_loop.size() - 1), ref_loop.at(ref_loop.size() - 2));
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, -2);

  fusion.printMath();

  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  for (auto tv : fusion.allTvs()) {
    for (const auto i : c10::irange(ref_loop.size() - 1)) {
      auto loop_id = tv->getLoopDomain().at(i);
      EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
          loop_id, ref_loop.at(i)));
    }
  }

  // Reorder back to the original order
#if 0
  for (auto tv : fusion.allTvs()) {
    tv->reorder({{-2, -1}});
  }
#endif
  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn({seq_length, rope_n_elem}, options);
  auto t2 = at::randn({seq_length, rope_n_elem}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  std::cerr << "Finished running\n";

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  // auto qkv = at::randn(shape1, options);
}

TEST_F(ResizeTest, RoPEFull2) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Llama-2-7b-hf
  int64_t n_head = 32;
  int64_t head_size = 128;
  int64_t n_query_groups = 32;
  int64_t rope_n_elem = 128;
  int64_t batches = 2;
  int64_t seq_length = 4096;

  int64_t q_per_kv = n_head / n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  bool use_cat = getenv("CAT");

  if (getenv("SMALL")) {
    n_head = 4;
    n_query_groups = 4;
    seq_length = 8;
    head_size = 16;
    rope_n_elem = 16;
  }

  std::vector<int64_t> shape1{
      batches, n_query_groups, total_qkv, seq_length, head_size};

  const bool has_rope_reshape = head_size != rope_n_elem;

  std::cerr << "shape1: " << shape1 << "\n";

  std::unordered_set<TensorView*> tvs_to_vectorize;

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 = makeContigConcreteTensor({seq_length, rope_n_elem});
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 = makeContigConcreteTensor({seq_length, rope_n_elem});
  fusion.addInput(tv2);
  auto sin = tv2;

  auto qkv = tv0;

  std::cerr << "qkv: " << qkv->toString() << "\n";

  cos = reshape(
      cos,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "cos: " << cos->toString() << "\n";

  sin = reshape(
      sin,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "sin: " << sin->toString() << "\n";

  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape1.size());
  for (const auto s : shape1) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  // tv5 (q)
  TensorView* tv6 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);
    tv6 = slice(qkv, slice_arg);
    std::cerr << "q slice: " << tv6->definition()->toString();
    tvs_to_vectorize.emplace(tv6);
  }
  auto q = tv6;

  // tv6 (k)
  TensorView* tv7 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);
    tv7 = slice(qkv, slice_arg);
    tvs_to_vectorize.emplace(tv7);
  }
  auto k = tv7;

  // tv6 (v)
  TensorView* tv8 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv + 1);
    tv8 = slice(qkv, slice_arg);
  }
  auto v = tv8;

  TensorView* tv9 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = q_per_kv;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv9 = reshape(q, cur_shape, new_shape);
  }
  q = tv9;

  TensorView* tv10 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv10 = reshape(k, cur_shape, new_shape);
  }
  k = tv10;

  TensorView* tv11 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv11 = reshape(v, cur_shape, new_shape);
  }
  v = tv11;

  // x: q, k or v
  // 1. take [..., :rope_n_elem]
  // 2. apply_rope
  // 3. concat apply_rope and [..., rope_n_elem:]
  auto apply_rope = [&](TensorView* x, bool is_q) -> TensorView* {
    // Insert reshape
    std::vector<int64_t> current_shape{
        batches, n_query_groups, seq_length, head_size};
    if (is_q) {
      current_shape[1] *= q_per_kv;
    }

    auto rope_n_elem_reshape_factor = head_size / rope_n_elem;
    std::vector<int64_t> new_shape = has_rope_reshape
        ? std::vector<
              int64_t>{batches, n_query_groups, seq_length, rope_n_elem_reshape_factor, rotation_num_splits, rope_n_elem / rotation_num_splits}
        : std::vector<int64_t>{
              batches,
              n_query_groups,
              seq_length,
              rotation_num_splits,
              rope_n_elem / rotation_num_splits};

    x = reshape(x, current_shape, new_shape);

    const int64_t rope_n_elem_slice_dim = (int64_t)x->nDims() - 3;
    const int64_t rotation_dim = (int64_t)x->nDims() - 2;

    std::vector<Slice> slice_arg;
    slice_arg.reserve(x->nDims());
    for (const auto id : x->getLogicalDomain()) {
      Slice arg;
      arg.start = zero;
      arg.stop = id->extent();
      slice_arg.push_back(arg);
    }

    TensorView* x_rope = x;
    if (has_rope_reshape) {
      auto x_rope_slice_arg = slice_arg;
      x_rope_slice_arg.at(rope_n_elem_slice_dim).stop = one;
      x_rope = slice(x, x_rope_slice_arg);
    }

    // x1
    NVF_ERROR(rotation_num_splits == 2);
    auto x1_slice_arg = slice_arg;
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x1_slice_arg.at(rotation_dim).stop = one;
    auto x1 = slice(x_rope, x1_slice_arg);
    std::cerr << "x1: " << x1->toString() << "\n";

    TensorView* x1_padded = nullptr;
    if (!use_cat) {
      pad(x1, {zero, zero, one, zero});
    }

    // x2
    auto x2_slice_arg = slice_arg;
    NVF_ERROR(rotation_num_splits == 2);
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x2_slice_arg.at(rotation_dim).start = one;
    auto x2 = slice(x_rope, x2_slice_arg);
    std::cerr << "x2: " << x2->toString() << "\n";
    TensorView* x2_padded = nullptr;
    if (!use_cat) {
      x2_padded = pad(x2, {zero, zero, zero, one});
    }

    TensorView* rotated = nullptr;
    if (!use_cat) {
      rotated = add(x2_padded, x1_padded);
    } else {
      rotated = cat({x2, x1}, -2);
    }

    std::vector<bool> bcast_flags(x->nDims(), false);
    bcast_flags[0] = true;
    bcast_flags[1] = true;
    if (has_rope_reshape) {
      bcast_flags[3] = true;
    }
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);
    std::cerr << "x_rope: " << x_rope->toString() << "\n";
    std::cerr << "cos_bc: " << cos_broadcast->toString() << "\n";
    auto apply_rope_result =
        add(mul(x_rope, cos_broadcast), mul(rotated, sin_broadcast));
    std::cerr << "apply_rope_result: " << apply_rope_result->toString() << "\n";
    TensorView* padded_apply_rope_result = apply_rope_result;
    if (has_rope_reshape) {
      padded_apply_rope_result =
          pad(apply_rope_result,
              {zero,
               zero,
               zero,
               zero,
               zero,
               IrBuilder::create<Val>(rope_n_elem_reshape_factor - 1)});
    }
    std::cerr << "Padded: " << padded_apply_rope_result->toString() << "\n";

    auto out = padded_apply_rope_result;

    if (has_rope_reshape) {
      // [..., rope_n_elem:]
      auto x_remaining_slice_arg = slice_arg;
      x_remaining_slice_arg.at(rope_n_elem_slice_dim).start = one;
      auto x_remaining = slice(x, x_remaining_slice_arg);
      auto padded_x_remaining =
          pad(x_remaining, {zero, zero, zero, zero, one, zero});
      out = add(out, padded_x_remaining);
    }

    // Reverse reshape
    out = reshape(out, new_shape, current_shape);

    return out;
  };

  auto q_out = apply_rope(q, true);
  q_out = set(q_out);
  [[maybe_unused]] auto k_out = apply_rope(k, false);
  k_out = set(k_out);
  // Not used but just for clarity
  [[maybe_unused]] auto v_out = apply_rope(v, false);

  fusion.addOutput(q_out);
  tvs_to_vectorize.emplace(q_out);
  // Disabled for now
  fusion.addOutput(k_out);
  tvs_to_vectorize.emplace(k_out);
  // fusion.addOutput(v_original_shape);

  fusion.printMath();

  NVF_ERROR(q->uses().size() == 1);
  NVF_ERROR(q->uses().at(0)->isA<ViewOp>());
  auto ref_tv = q->uses().at(0)->output(0)->as<TensorView>();

  std::vector<IterDomain*> ref_loop = ref_tv->getLogicalDomain();
  std::swap(ref_loop.at(ref_loop.size() - 1), ref_loop.at(ref_loop.size() - 2));
  // for (const auto i : c10::irange(2, ref_loop.size() - 2)) {
  // std::swap(ref_loop.at(i), ref_loop.at(i + 1));
  // }
  std::cerr << "Ref domain: " << toDelimitedString(ref_loop) << "\n";
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, -2);

  fusion.printMath();

  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  for (auto tv : fusion.allTvs()) {
    for (const auto i : c10::irange(ref_loop.size() - 2)) {
      auto loop_id = tv->getLoopDomain().at(i);
      EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
          loop_id, ref_loop.at(i)));
    }
  }

  // Reorder back to the original order
  for (auto tv : fusion.allTvs()) {
    std::cerr << "Before: " << tv->toString() << "\n";
    tv->reorder({{3, -1}});
    std::cerr << "Reordered: " << tv->toString() << "\n";

    // Parallelize the innermost domain
    tv->split(-1, 4);
    std::cerr << "Vec split: " << tv->toString() << "\n";
    tv->axis(-2)->parallelize(ParallelType::TIDx);
    std::cerr << "TIDx parallelized: " << tv->toString() << "\n";

    if (tvs_to_vectorize.find(tv) != tvs_to_vectorize.end()) {
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    // Schedule the outermost three loops
    tv->merge(0)->merge(0);
    // If TIDx is small, use TIDy as well
    int64_t vec_factor = 4;
    int64_t bdimx = rope_n_elem / 2 / vec_factor;
    if (bdimx < 128) {
      tv->split(0, ceilDiv(128, bdimx));
      tv->axis(1)->parallelize(ParallelType::TIDy);
    }
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn({seq_length, rope_n_elem}, options);
  auto t2 = at::randn({seq_length, rope_n_elem}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  std::cerr << "Finished running\n";

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  if (getenv("BENCHMARK")) {
    int64_t mem_size = sizeof(float);
    for (const auto s : shape1) {
      mem_size *= s;
    }
    mem_size *= 2;

    // Only q and k are computed
    mem_size = mem_size / total_qkv * (q_per_kv + 1);

    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
      cg_outputs = fe.runFusion(aten_inputs);
      FusionProfiler::stop();
      auto t = FusionProfiler::profile().kernel_time_ms;
      std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      std::cout << "Bandwidth (GB/s): "
                << ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
                << "\n";
    }
  }
}

TEST_F(ResizeTest, RoPEFullBF16) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Llama-2-7b-hf
  int64_t n_head = 32;
  int64_t head_size = 128;
  int64_t n_query_groups = 32;
  int64_t rope_n_elem = 128;
  int64_t batches = 2;
  int64_t seq_length = 4096;

  int64_t q_per_kv = n_head / n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  bool use_cat = getenv("CAT");

  if (getenv("SMALL")) {
    n_head = 4;
    n_query_groups = 4;
    seq_length = 8;
    head_size = 16;
    rope_n_elem = 16;
  }

  std::vector<int64_t> shape_before_permutation{
      batches, seq_length, n_query_groups, total_qkv, head_size};
  std::vector<int64_t> shape1{
      batches, n_query_groups, total_qkv, seq_length, head_size};

  const bool has_rope_reshape = head_size != rope_n_elem;

  std::cerr << "shape1: " << shape1 << "\n";

  std::unordered_set<TensorView*> tvs_to_vectorize;

  // qkv after permutation
  // TODO: Add real permutation
  std::vector<std::optional<bool>> tv0_contig(shape1.size(), false);
  tv0_contig.back() = true;
  auto tv0 = TensorViewBuilder()
                 .shape(shape1)
                 .dtype(DataType::BFloat16)
                 .contiguity(tv0_contig)
                 .build();
  // auto tv0 = makeContigConcreteTensor(shape1, DataType::BFloat16);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv2);
  auto sin = tv2;

  auto qkv = tv0;

  std::cerr << "qkv: " << qkv->toString() << "\n";

  cos = reshape(
      cos,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "cos: " << cos->toString() << "\n";

  sin = reshape(
      sin,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "sin: " << sin->toString() << "\n";

  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape1.size());
  for (const auto s : shape1) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  // tv5 (q)
  TensorView* tv6 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);
    tv6 = slice(qkv, slice_arg);
    std::cerr << "q slice: " << tv6->definition()->toString();
    tvs_to_vectorize.emplace(tv6);
  }
  auto q = tv6;

  // tv6 (k)
  TensorView* tv7 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);
    tv7 = slice(qkv, slice_arg);
    tvs_to_vectorize.emplace(tv7);
  }
  auto k = tv7;

  // tv6 (v)
  TensorView* tv8 = nullptr;
  {
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv + 1);
    tv8 = slice(qkv, slice_arg);
  }
  auto v = tv8;

  TensorView* tv9 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = q_per_kv;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv9 = reshape(q, cur_shape, new_shape);
  }
  q = tv9;

  TensorView* tv10 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv10 = reshape(k, cur_shape, new_shape);
  }
  k = tv10;

  TensorView* tv11 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv11 = reshape(v, cur_shape, new_shape);
  }
  v = tv11;

  // x: q, k or v
  // 1. take [..., :rope_n_elem]
  // 2. apply_rope
  // 3. concat apply_rope and [..., rope_n_elem:]
  auto apply_rope = [&](TensorView* x, bool is_q) -> TensorView* {
    // Insert reshape
    std::vector<int64_t> current_shape{
        batches, n_query_groups, seq_length, head_size};
    if (is_q) {
      current_shape[1] *= q_per_kv;
    }

    auto rope_n_elem_reshape_factor = head_size / rope_n_elem;
    std::vector<int64_t> new_shape = has_rope_reshape
        ? std::vector<
              int64_t>{batches, n_query_groups, seq_length, rope_n_elem_reshape_factor, rotation_num_splits, rope_n_elem / rotation_num_splits}
        : std::vector<int64_t>{
              batches,
              n_query_groups,
              seq_length,
              rotation_num_splits,
              rope_n_elem / rotation_num_splits};

    x = reshape(x, current_shape, new_shape);

    const int64_t rope_n_elem_slice_dim = (int64_t)x->nDims() - 3;
    const int64_t rotation_dim = (int64_t)x->nDims() - 2;

    std::vector<Slice> slice_arg;
    slice_arg.reserve(x->nDims());
    for (const auto id : x->getLogicalDomain()) {
      Slice arg;
      arg.start = zero;
      arg.stop = id->extent();
      slice_arg.push_back(arg);
    }

    TensorView* x_rope = x;
    if (has_rope_reshape) {
      auto x_rope_slice_arg = slice_arg;
      x_rope_slice_arg.at(rope_n_elem_slice_dim).stop = one;
      x_rope = slice(x, x_rope_slice_arg);
    }

    // x1
    NVF_ERROR(rotation_num_splits == 2);
    auto x1_slice_arg = slice_arg;
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x1_slice_arg.at(rotation_dim).stop = one;
    auto x1 = slice(x_rope, x1_slice_arg);
    std::cerr << "x1: " << x1->toString() << "\n";

    TensorView* x1_padded = nullptr;
    if (!use_cat) {
      x1_padded = pad(x1, {zero, zero, one, zero});
    }

    // x2
    auto x2_slice_arg = slice_arg;
    NVF_ERROR(rotation_num_splits == 2);
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x2_slice_arg.at(rotation_dim).start = one;
    auto x2 = slice(x_rope, x2_slice_arg);
    std::cerr << "x2: " << x2->toString() << "\n";
    TensorView* x2_padded = nullptr;
    if (!use_cat) {
      x2_padded = pad(x2, {zero, zero, zero, one});
    }

    TensorView* rotated = nullptr;
    if (!use_cat) {
      rotated = add(x2_padded, x1_padded);
    } else {
      rotated = cat({x2, x1}, -2);
    }

    std::vector<bool> bcast_flags(x->nDims(), false);
    bcast_flags[0] = true;
    bcast_flags[1] = true;
    if (has_rope_reshape) {
      bcast_flags[3] = true;
    }
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);
    std::cerr << "x_rope: " << x_rope->toString() << "\n";
    std::cerr << "cos_bc: " << cos_broadcast->toString() << "\n";
    auto apply_rope_result =
        add(mul(x_rope, cos_broadcast), mul(rotated, sin_broadcast));
    std::cerr << "apply_rope_result: " << apply_rope_result->toString() << "\n";
    TensorView* padded_apply_rope_result = apply_rope_result;
    if (has_rope_reshape) {
      padded_apply_rope_result =
          pad(apply_rope_result,
              {zero,
               zero,
               zero,
               zero,
               zero,
               IrBuilder::create<Val>(rope_n_elem_reshape_factor - 1)});
    }
    std::cerr << "Padded: " << padded_apply_rope_result->toString() << "\n";

    auto out = padded_apply_rope_result;

    if (has_rope_reshape) {
      // [..., rope_n_elem:]
      auto x_remaining_slice_arg = slice_arg;
      x_remaining_slice_arg.at(rope_n_elem_slice_dim).start = one;
      auto x_remaining = slice(x, x_remaining_slice_arg);
      auto padded_x_remaining =
          pad(x_remaining, {zero, zero, zero, zero, one, zero});
      out = add(out, padded_x_remaining);
    }

    // Reverse reshape
    out = reshape(out, new_shape, current_shape);

    return out;
  };

  auto q_out = apply_rope(q, true);
  q_out = castOp(DataType::BFloat16, q_out);
  q_out = set(q_out);
  [[maybe_unused]] auto k_out = apply_rope(k, false);
  k_out = castOp(DataType::BFloat16, k_out);
  k_out = set(k_out);
  // Not used but just for clarity
  [[maybe_unused]] auto v_out = apply_rope(v, false);

  fusion.addOutput(q_out);
  tvs_to_vectorize.emplace(q_out);
  // Disabled for now
  fusion.addOutput(k_out);
  tvs_to_vectorize.emplace(k_out);
  // fusion.addOutput(v_original_shape);

  fusion.printMath();

  NVF_ERROR(q->uses().size() == 1);
  NVF_ERROR(q->uses().at(0)->isA<ViewOp>());
  auto ref_tv = q->uses().at(0)->output(0)->as<TensorView>();

  std::vector<IterDomain*> ref_loop = ref_tv->getLogicalDomain();
  std::swap(ref_loop.at(ref_loop.size() - 1), ref_loop.at(ref_loop.size() - 2));
  std::cerr << "Ref domain: " << toDelimitedString(ref_loop) << "\n";
  scheduler_tools::scheduleLoopDomainsLike(fusion.allTvs(), ref_loop, -2);

  fusion.printMath();

  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();

  for (auto tv : fusion.allTvs()) {
    for (const auto i : c10::irange(ref_loop.size() - 2)) {
      auto loop_id = tv->getLoopDomain().at(i);
      EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
          loop_id, ref_loop.at(i)));
    }
  }

  // Reorder back to the original order
  for (auto tv : fusion.allTvs()) {
    std::cerr << "Before: " << tv->toString() << "\n";
    tv->reorder({{3, -1}});
    std::cerr << "Reordered: " << tv->toString() << "\n";

    // Parallelize the innermost domain
    tv->split(-1, 8);
    std::cerr << "Vec split: " << tv->toString() << "\n";
    tv->axis(-2)->parallelize(ParallelType::TIDx);

    if (tvs_to_vectorize.find(tv) != tvs_to_vectorize.end()) {
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    // Schedule the outermost three loops
    tv->merge(0)->merge(0);
    // If TIDx is small, use TIDy as well
    int64_t vec_factor = 4;
    int64_t bdimx = rope_n_elem / 2 / vec_factor;
    if (bdimx < 128) {
      tv->split(0, ceilDiv(128, bdimx));
      tv->axis(1)->parallelize(ParallelType::TIDy);
    }
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape_before_permutation, options);
  t0 = at::permute(t0, {0, 2, 3, 1, 4});
  auto t1 = at::randn({seq_length, rope_n_elem}, options);
  auto t2 = at::randn({seq_length, rope_n_elem}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  if (getenv("BENCHMARK")) {
    int64_t mem_size = 1;
    for (const auto s : shape1) {
      mem_size *= s;
    }
    // read and write
    mem_size *= 2;
    // Only q and k are computed
    mem_size = mem_size / total_qkv * (q_per_kv + 1);
    // sin and cos
    mem_size += seq_length * rope_n_elem * 2;
    // BFloat16
    mem_size *= 2;

    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
      cg_outputs = fe.runFusion(aten_inputs);
      FusionProfiler::stop();
      auto t = FusionProfiler::profile().kernel_time_ms;
      std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      std::cout << "Bandwidth (GB/s): "
                << ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
                << "\n";
    }
  }
}

TEST_F(ResizeTest, RoPEFullBF16Permute) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Llama-2-7b-hf
  int64_t n_head = 32;
  int64_t head_size = 128;
  int64_t n_query_groups = 32;
  int64_t rope_n_elem = 128;
  int64_t batches = 2;
  int64_t seq_length = 4096;

  int64_t q_per_kv = n_head / n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  bool use_cat = getenv("CAT");

  if (getenv("SMALL")) {
    n_head = 4;
    n_query_groups = 4;
    seq_length = 8;
    head_size = 16;
    rope_n_elem = 16;
  }

  std::vector<int64_t> input_shape{
      batches, seq_length, head_size * (n_head + 2 * n_query_groups)};
  std::vector<int64_t> shape_before_permutation{
      batches, seq_length, n_query_groups, total_qkv, head_size};
  std::vector<int64_t> shape1{
      batches, n_query_groups, total_qkv, seq_length, head_size};

  const bool has_rope_reshape = head_size != rope_n_elem;

  std::cerr << "shape1: " << shape1 << "\n";

  std::unordered_set<TensorView*> tvs_to_vectorize;

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(input_shape, DataType::BFloat16);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv2);
  auto sin = tv2;

  cos = reshape(
      cos,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "cos: " << cos->toString() << "\n";

  sin = reshape(
      sin,
      {seq_length, rope_n_elem},
      {seq_length, rotation_num_splits, rope_n_elem / rotation_num_splits});

  std::cerr << "sin: " << sin->toString() << "\n";

  auto zero = fusion.zeroVal();
  auto one = fusion.oneVal();

  auto get_qkv = [&]() {
    TensorView* qkv = nullptr;
    // Permute and reshape
    auto tv = tv0;
    tv = set(tv); // vectorization
    tvs_to_vectorize.emplace(tv);
    tv = reshape(tv, input_shape, shape_before_permutation);
    tv = permute(tv, {0, 2, 3, 1, 4});
    qkv = tv;
    std::cerr << "qkv: " << qkv->toString() << "\n";
    return qkv;
  };

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(shape1.size());
  for (const auto s : shape1) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 2;

  // tv5 (q)
  TensorView* tv6 = nullptr;
  {
    auto qkv = get_qkv();
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);
    tv6 = slice(qkv, slice_arg);
    std::cerr << "q slice: " << tv6->definition()->toString();
  }
  const auto initial_q = tv6;
  auto q = tv6;

  // tv6 (k)
  TensorView* tv7 = nullptr;
  {
    auto qkv = get_qkv();
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);
    tv7 = slice(qkv, slice_arg);
  }
  const auto initial_k = tv7;
  auto k = tv7;

  // tv6 (v)
  TensorView* tv8 = nullptr;
  {
    auto qkv = get_qkv();
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv + 1);
    tv8 = slice(qkv, slice_arg);
  }
  const auto initial_v = tv8;
  auto v = tv8;

  TensorView* tv9 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = q_per_kv;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv9 = reshape(q, cur_shape, new_shape);
  }
  q = tv9;

  TensorView* tv10 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv10 = reshape(k, cur_shape, new_shape);
  }
  k = tv10;

  TensorView* tv11 = nullptr;
  {
    auto cur_shape = shape1;
    cur_shape[qkv_slice_dim] = 1;
    std::vector<int64_t> new_shape;
    new_shape.push_back(batches);
    new_shape.push_back(-1);
    new_shape.insert(new_shape.end(), shape1.begin() + 3, shape1.end());
    tv11 = reshape(v, cur_shape, new_shape);
  }
  v = tv11;

  // x: q, k or v
  // 1. take [..., :rope_n_elem]
  // 2. apply_rope
  // 3. concat apply_rope and [..., rope_n_elem:]
  auto apply_rope = [&](TensorView* x, bool is_q) -> TensorView* {
    // Insert reshape
    std::vector<int64_t> current_shape{
        batches, n_query_groups, seq_length, head_size};
    if (is_q) {
      current_shape[1] *= q_per_kv;
    }

    auto rope_n_elem_reshape_factor = head_size / rope_n_elem;
    std::vector<int64_t> new_shape = has_rope_reshape
        ? std::vector<
              int64_t>{batches, n_query_groups, seq_length, rope_n_elem_reshape_factor, rotation_num_splits, rope_n_elem / rotation_num_splits}
        : std::vector<int64_t>{
              batches,
              n_query_groups,
              seq_length,
              rotation_num_splits,
              rope_n_elem / rotation_num_splits};

    x = reshape(x, current_shape, new_shape);

    const int64_t rope_n_elem_slice_dim = (int64_t)x->nDims() - 3;
    const int64_t rotation_dim = (int64_t)x->nDims() - 2;

    std::vector<Slice> slice_arg;
    slice_arg.reserve(x->nDims());
    for (const auto id : x->getLogicalDomain()) {
      Slice arg;
      arg.start = zero;
      arg.stop = id->extent();
      slice_arg.push_back(arg);
    }

    TensorView* x_rope = x;
    if (has_rope_reshape) {
      auto x_rope_slice_arg = slice_arg;
      x_rope_slice_arg.at(rope_n_elem_slice_dim).stop = one;
      x_rope = slice(x, x_rope_slice_arg);
    }

    // x1
    NVF_ERROR(rotation_num_splits == 2);
    auto x1_slice_arg = slice_arg;
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x1_slice_arg.at(rotation_dim).stop = one;
    auto x1 = slice(x_rope, x1_slice_arg);
    std::cerr << "x1: " << x1->toString() << "\n";

    TensorView* x1_padded = nullptr;
    if (!use_cat) {
      x1_padded = pad(x1, {zero, zero, one, zero});
    }

    // x2
    auto x2_slice_arg = slice_arg;
    NVF_ERROR(rotation_num_splits == 2);
    if (has_rope_reshape) {
      x1_slice_arg.at(rope_n_elem_slice_dim).stop = one;
    }
    x2_slice_arg.at(rotation_dim).start = one;
    auto x2 = slice(x_rope, x2_slice_arg);
    std::cerr << "x2: " << x2->toString() << "\n";
    TensorView* x2_padded = nullptr;
    if (!use_cat) {
      x2_padded = pad(x2, {zero, zero, zero, one});
    }

    TensorView* rotated = nullptr;
    if (!use_cat) {
      rotated = add(x2_padded, x1_padded);
    } else {
      rotated = cat({x2, x1}, -2);
    }

    std::vector<bool> bcast_flags(x->nDims(), false);
    bcast_flags[0] = true;
    bcast_flags[1] = true;
    if (has_rope_reshape) {
      bcast_flags[3] = true;
    }
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);
    std::cerr << "x_rope: " << x_rope->toString() << "\n";
    std::cerr << "cos_bc: " << cos_broadcast->toString() << "\n";
    auto apply_rope_result =
        add(mul(x_rope, cos_broadcast), mul(rotated, sin_broadcast));
    std::cerr << "apply_rope_result: " << apply_rope_result->toString() << "\n";
    TensorView* padded_apply_rope_result = apply_rope_result;
    if (has_rope_reshape) {
      padded_apply_rope_result =
          pad(apply_rope_result,
              {zero,
               zero,
               zero,
               zero,
               zero,
               IrBuilder::create<Val>(rope_n_elem_reshape_factor - 1)});
    }
    std::cerr << "Padded: " << padded_apply_rope_result->toString() << "\n";

    auto out = padded_apply_rope_result;

    if (has_rope_reshape) {
      // [..., rope_n_elem:]
      auto x_remaining_slice_arg = slice_arg;
      x_remaining_slice_arg.at(rope_n_elem_slice_dim).start = one;
      auto x_remaining = slice(x, x_remaining_slice_arg);
      auto padded_x_remaining =
          pad(x_remaining, {zero, zero, zero, zero, one, zero});
      out = add(out, padded_x_remaining);
    }

    // Reverse reshape
    out = reshape(out, new_shape, current_shape);

    return out;
  };

  auto q_out = apply_rope(q, true);
  q_out = castOp(DataType::BFloat16, q_out);
  q_out = set(q_out);
  [[maybe_unused]] auto k_out = apply_rope(k, false);
  k_out = castOp(DataType::BFloat16, k_out);
  k_out = set(k_out);
  // Not used but just for clarity
  [[maybe_unused]] auto v_out = apply_rope(v, false);

  fusion.addOutput(q_out);
  tvs_to_vectorize.emplace(q_out);
  // Disabled for now
  fusion.addOutput(k_out);
  tvs_to_vectorize.emplace(k_out);
  // fusion.addOutput(v_original_shape);

  fusion.printMath();

  {
    NVF_ERROR(q->uses().size() == 1);
    NVF_ERROR(q->uses().at(0)->isA<ViewOp>());
    auto ref_tv = q->uses().at(0)->output(0)->as<TensorView>();

    std::cerr << "Reference tensor: " << ref_tv->toString() << "\n";

    std::vector<IterDomain*> ref_loop = ref_tv->getLogicalDomain();
    std::swap(
        ref_loop.at(ref_loop.size() - 1), ref_loop.at(ref_loop.size() - 2));
    std::cerr << "Ref domain: " << toDelimitedString(ref_loop) << "\n";
    std::vector<TensorView*> tvs;
    auto excluded_tvs = DependencyCheck::getAllValsBetween(
        {fusion.inputs().begin(), fusion.inputs().end()},
        {initial_q, initial_k, initial_v});
    std::cerr << "Excluded: " << toDelimitedString(excluded_tvs) << "\n";
    for (auto tv : fusion.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }

      if (std::find(excluded_tvs.begin(), excluded_tvs.end(), tv) !=
              excluded_tvs.end() &&
          tv != initial_q && tv != initial_k && tv != initial_v) {
        continue;
      }

      tvs.push_back(tv);
    }

    scheduler_tools::scheduleLoopDomainsLike(tvs, ref_loop, -2);

    fusion.printMath();

    // q inputs
    {
      auto inputs = DependencyCheck::getAllValsBetween(
          {fusion.inputs().begin(), fusion.inputs().end()}, {initial_q});
      auto input_tvs = ir_utils::filterByType<TensorView>(inputs).vector();
      std::vector<IterDomain*> ref_loop = initial_q->getLoopDomain();
      std::cerr << "Ref for Q: " << toDelimitedString(ref_loop) << "\n";
      scheduler_tools::scheduleLoopDomainsLike(input_tvs, ref_loop, -1);

      for (auto tv : input_tvs) {
        if (tv->isFusionInput()) {
          continue;
        }
        std::cerr << "After: " << tv->toString() << "\n";
      }
    }

    // k inputs
    {
      auto inputs = DependencyCheck::getAllValsBetween(
          {fusion.inputs().begin(), fusion.inputs().end()}, {initial_k});
      auto input_tvs = ir_utils::filterByType<TensorView>(inputs).vector();
      std::vector<IterDomain*> ref_loop = initial_k->getLoopDomain();
      std::cerr << "Ref for K: " << toDelimitedString(ref_loop) << "\n";
      scheduler_tools::scheduleLoopDomainsLike(input_tvs, ref_loop, -1);

      for (auto tv : input_tvs) {
        if (tv->isFusionInput()) {
          continue;
        }
        std::cerr << "After: " << tv->toString() << "\n";
      }
    }
  }

  fusion.printMath();

#if 0
  IdModel id_model(&fusion, /*build_models=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  for (auto tv : fusion.allTvs()) {
    for (const auto i : c10::irange(ref_loop.size() - 2)) {
      auto loop_id = tv->getLoopDomain().at(i);
      EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
          loop_id, ref_loop.at(i)));
    }
  }
#endif

  // Reorder back to the original order
  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    std::cerr << "Before: " << tv->toString() << "\n";
    tv->reorder({{3, -1}});
    std::cerr << "Reordered: " << tv->toString() << "\n";

    // Parallelize the innermost domain
    tv->split(-1, 8);
    std::cerr << "Vec split: " << tv->toString() << "\n";
    tv->axis(-2)->parallelize(ParallelType::TIDx);

    if (tvs_to_vectorize.find(tv) != tvs_to_vectorize.end()) {
      tv->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    // Schedule the outermost three loops
    tv->merge(0)->merge(0);
    // If TIDx is small, use TIDy as well
    int64_t vec_factor = 4;
    int64_t bdimx = rope_n_elem / 2 / vec_factor;
    if (bdimx < 128) {
      tv->split(0, ceilDiv(128, bdimx));
      tv->axis(1)->parallelize(ParallelType::TIDy);
    }
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({seq_length, rope_n_elem}, options);
  auto t2 = at::randn({seq_length, rope_n_elem}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);

  if (getenv("BENCHMARK")) {
    int64_t mem_size = 1;
    for (const auto s : shape1) {
      mem_size *= s;
    }
    // read and write
    mem_size *= 2;
    // Only q and k are computed
    mem_size = mem_size / total_qkv * (q_per_kv + 1);
    // sin and cos
    mem_size += seq_length * rope_n_elem * 2;
    // BFloat16
    mem_size *= 2;

    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
      cg_outputs = fe.runFusion(aten_inputs);
      FusionProfiler::stop();
      auto t = FusionProfiler::profile().kernel_time_ms;
      std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      std::cout << "Bandwidth (GB/s): "
                << ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
                << "\n";
    }
  }
}

TEST_F(ResizeTest, FP16) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({12}, DataType::BFloat16);
  fusion.addInput(tv0);

  auto tv1 =
      slice(tv0, {{IrBuilder::create<Val>(1L), IrBuilder::create<Val>(1L)}});

  auto tv2 = pad(tv0, {fusion.oneVal(), fusion.oneVal()});

  auto tv4 = add(tv2, tv2);

  auto tv3 = cat({tv2, tv2}, 0);

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  fusion.printMath();
  fusion.printKernel();
}

TEST_F(ResizeTest, Permute) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto zero = fusion.zeroVal();
  [[maybe_unused]] auto one = fusion.oneVal();

  std::vector<int64_t> shape{3, 4, 5};
  std::vector<int64_t> shape_permute{4, 3, 5};
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = permute(tv1, {1, 0, 2});
  auto tv3 = set(tv2);

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(tv3->nDims());
  for (const auto s : shape_permute) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  slice_default_arg.back().start = one;
  slice_default_arg.back().stop = IrBuilder::create<Val>(3L);

  auto tv4 = slice(tv3, slice_default_arg);
  fusion.addOutput(tv4);

  fusion.printMath();
  fusion.printKernel();

  scheduler_tools::scheduleLoopDomainsLike(
      fusion.allTvs(), tv4->getLoopDomain(), -1);

  inlineMost();

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, RotationInSmem) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32});
  fusion.addInput(tv0);

  auto tv1 = set(tv0); // shmem cache

  auto zero = fusion.zeroVal();
  auto c16 = IrBuilder::create<Val>(16L);
  auto c32 = IrBuilder::create<Val>(32L);

  auto tv2 = slice(tv1, {{zero, c16}});
  auto tv3 = pad(tv2, {c16, zero});

  auto tv4 = slice(tv1, {{c16, c32}});
  auto tv5 = pad(tv4, {zero, c16});

  auto tv6 = add(tv5, tv3);
  auto tv7 = add(tv1, tv6);

  fusion.addOutput(tv7);

  fusion.printMath();

  scheduler_tools::scheduleLoopDomainsLike({tv2}, tv3->getLoopDomain(), -1);

  scheduler_tools::scheduleLoopDomainsLike({tv4}, tv5->getLoopDomain(), -1);

  fusion.print();

  // tv1 and the rest need to have different loop domains to trigger
  // synchronization.

  tv1->split(0, 4);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  for (auto tv : {tv2, tv3, tv4, tv5, tv6, tv7}) {
    tv->split(0, 2);
    tv->axis(-2)->parallelize(ParallelType::TIDx);
  }

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsync);
  tv1->definition()->as<LoadStoreOp>()->setCacheOp(CacheOp::Global);

  fusion.printMath();

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(ResizeTest, RoPEFullBF16PermuteShmem) {
  EnableOptionsGuard enable_options_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Llama-2-7b-hf
  int64_t n_head = 32;
  int64_t head_size = 128;
  int64_t n_query_groups = 32;
  int64_t rope_n_elem = 128;
  int64_t batches = 2;
  int64_t seq_length = 4096;

  int64_t q_per_kv = n_head / n_query_groups;
  int64_t total_qkv = q_per_kv + 2;

  int64_t rotation_num_splits = 2;

  if (getenv("SMALL")) {
    n_head = 4;
    n_query_groups = 4;
    seq_length = 8;
    head_size = 16;
    rope_n_elem = 16;
  }

  // Not possible to have this reshape yet. This is necessary before
  // the x1/x2 slice. This could be solved by a native alias support.
  // std::vector<int64_t> input_shape{
  // batches, seq_length, head_size * (n_head + 2 * n_query_groups)};
  std::vector<int64_t> shape_before_permutation{
      batches, seq_length, n_query_groups, total_qkv, head_size};
  std::vector<int64_t> shape1{
      batches, n_query_groups, total_qkv, seq_length, head_size};

  std::vector<int64_t> input_shape = shape_before_permutation;

  std::cerr << "input shape: " << input_shape << "\n";

  std::unordered_set<TensorView*> tvs_to_vectorize;

  // qkv after permutation
  auto tv0 = makeContigConcreteTensor(input_shape, DataType::BFloat16);
  fusion.addInput(tv0);

  std::cerr << "Input: " << tv0->toString() << "\n";

  // cos
  auto tv1 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv1);
  auto cos = tv1;

  // sin
  auto tv2 =
      makeContigConcreteTensor({seq_length, rope_n_elem}, DataType::BFloat16);
  fusion.addInput(tv2);
  auto sin = tv2;

  auto zero = fusion.zeroVal();
  [[maybe_unused]] auto one = fusion.oneVal();

  std::vector<Slice> slice_default_arg;
  slice_default_arg.reserve(input_shape.size());
  for (const auto s : input_shape) {
    slice_default_arg.push_back(Slice{zero, IrBuilder::create<Val>(s)});
  }

  int64_t qkv_slice_dim = 3;

  // tv5 (q)p
  TensorView* tv6 = nullptr;
  {
    auto qkv = tv0;
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 2);
    tv6 = slice(qkv, slice_arg);
    std::cerr << "q slice: " << tv6->definition()->toString();
    tvs_to_vectorize.emplace(tv6);
  }
  auto q = tv6;

  // tv6 (k)
  TensorView* tv7 = nullptr;
  {
    auto qkv = tv0;
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv);
    slice_arg[qkv_slice_dim].stop = IrBuilder::create<Val>(total_qkv - 1);
    tv7 = slice(qkv, slice_arg);
    tvs_to_vectorize.emplace(tv7);
  }
  auto k = tv7;

#if 0
  // tv6 (v)
  TensorView* tv8 = nullptr;
  {
    auto qkv = tv0;
    auto slice_arg = slice_default_arg;
    slice_arg[qkv_slice_dim].start = IrBuilder::create<Val>(q_per_kv + 1);
    tv8 = slice(qkv, slice_arg);
  }
  [[maybe_unused]] auto v = tv8;
#endif

  // x: q, k or v
  // 1. take [..., :rope_n_elem]
  // 2. apply_rope
  // 3. concat apply_rope and [..., rope_n_elem:]
  auto apply_rope = [&](TensorView* x, bool is_q) -> TensorView* {
    std::vector<Slice> slice_arg;
    slice_arg.reserve(x->nDims());
    for (const auto id : x->getLogicalDomain()) {
      Slice arg;
      arg.start = zero;
      arg.stop = id->extent();
      slice_arg.push_back(arg);
    }

    TensorView* x_rope = x;

    // x1
    auto x1_slice_arg = slice_arg;
    x1_slice_arg.back().stop =
        IrBuilder::create<Val>(rope_n_elem / rotation_num_splits);
    auto x1 = slice(x_rope, x1_slice_arg);
    std::cerr << "x1: " << x1->definition()->toString() << "\n";

    // x2
    auto x2_slice_arg = slice_arg;
    x2_slice_arg.back().start =
        IrBuilder::create<Val>(rope_n_elem / rotation_num_splits);
    auto x2 = slice(x_rope, x2_slice_arg);
    std::cerr << "x2: " << x2->definition()->toString() << "\n";

    TensorView* rotated = cat({x2, x1}, -1);

    std::vector<bool> bcast_flags{true, false, true, true, false};
    auto cos_broadcast = broadcast(cos, bcast_flags);
    auto sin_broadcast = broadcast(sin, bcast_flags);
    std::cerr << "x_rope: " << x_rope->toString() << "\n";
    std::cerr << "cos_bc: " << cos_broadcast->toString() << "\n";
    auto out = add(mul(x_rope, cos_broadcast), mul(rotated, sin_broadcast));
    std::cerr << "apply_rope_result: " << out->toString() << "\n";

    std::vector<int64_t> cur_shape = input_shape;
    cur_shape[qkv_slice_dim] = is_q ? q_per_kv : 1;
    std::vector<int64_t> new_shape{batches, seq_length, -1, rope_n_elem};
    out = reshape(out, cur_shape, new_shape);
    out = permute(out, {0, 2, 1, 3});
    out = castOp(DataType::BFloat16, out);
    return out;
  };

  auto q_out = apply_rope(q, true);
  q_out = set(q_out);
  fusion.addOutput(q_out);
  tvs_to_vectorize.emplace(q_out);

  [[maybe_unused]] auto k_out = apply_rope(k, false);
  k_out = set(k_out);
  fusion.addOutput(k_out);
  tvs_to_vectorize.emplace(k_out);

  // Not used but just for clarity
  //[[maybe_unused]] auto v_out = apply_rope(v, false);

  // fusion.addOutput(v_original_shape);

  fusion.printMath();

  q->setMemoryType(MemoryType::Shared);

  if (!getenv("DISABLE_SCHEDULE")) {
    for (auto tv : {q, k}) {
      for (const auto tv_use : tv->uses()) {
        SliceOp* slice = dynamic_cast<SliceOp*>(tv_use);
        if (slice == nullptr) {
          continue;
        }
        TensorView* slice_out = slice->output(0)->as<TensorView>();
        auto padded_tv =
            slice->output(0)->uses().at(0)->output(0)->as<TensorView>();
        NVF_ERROR(padded_tv->definition()->isA<PadOp>());
        auto ref_tv = padded_tv;
        std::cerr << "Reference tensor: " << ref_tv->toString() << "\n";
        std::vector<IterDomain*> ref_loop = ref_tv->getLogicalDomain();
        std::cerr << "Ref domain: " << toDelimitedString(ref_loop) << "\n";
        scheduler_tools::scheduleLoopDomainsLike({slice_out}, ref_loop, -1);
        std::cerr << "Slice out: " << slice_out->toString() << "\n";
      }
    }

    for (auto tv : fusion.allTvs()) {
      ASSERT_EQ(tv->getLoopDomain().back()->extent()->evaluate(), rope_n_elem);
    }

    // Reorder
    scheduler_tools::scheduleLoopDomainsLike(
        fusion.allTvs(), q->getLogicalDomain(), 3);

    fusion.printMath();

#if 0
    IdModel id_model(&fusion, /*build_models=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    for (auto tv : fusion.allTvs()) {
      for (const auto i : c10::irange(ref_loop.size() - 2)) {
        auto loop_id = tv->getLoopDomain().at(i);
        EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
            loop_id, ref_loop.at(i)));
      }
    }
#endif

    for (auto tv : fusion.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }

      std::cerr << "Before: " << tv->toString() << "\n";

      // Parallelize the innermost domain
      tv->split(-1, 8);

      if (tvs_to_vectorize.find(tv) != tvs_to_vectorize.end()) {
        std::cerr << "Vec split: " << tv->toString() << "\n";
        tv->axis(-1)->parallelize(ParallelType::Vectorize);
        if (!tv->isFusionOutput()) {
          tv->setMemoryType(MemoryType::Shared);
          tv->definition()->as<SliceOp>()->setOpType(LoadStoreOpType::CpAsync);
          tv->definition()->as<SliceOp>()->setCacheOp(CacheOp::Global);

          // Dummy split
          tv->split(-2, 2);
          tv->merge(-3, -2);
        }
      }

      tv->axis(-2)->parallelize(ParallelType::TIDx);

      // Schedule the outermost three loops
      tv->merge(0)->merge(0);

      std::cerr << "After: " << tv->toString() << "\n";
    }

    // Inlining
    for (auto tv : fusion.allTvs()) {
      if (tv->isFusionInput()) {
        continue;
      }
      // This doesn't work. TODO: Investigate
      // tv->inlineAt(1);

      // If TIDx is small, use TIDy as well
      int64_t vec_factor = 8;
      int64_t bdimx = rope_n_elem / vec_factor;
      if (bdimx < 128) {
        tv->split(0, ceilDiv(128, bdimx));
        tv->axis(1)->parallelize(ParallelType::TIDy);
      }
      tv->axis(0)->parallelize(ParallelType::BIDx);
    }
  }

  fusion.printMath();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn({seq_length, rope_n_elem}, options);
  auto t2 = at::randn({seq_length, rope_n_elem}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  if (getenv("BENCHMARK")) {
    int64_t mem_size = 1;
    for (const auto s : shape1) {
      mem_size *= s;
    }
    // read and write
    mem_size *= 2;
    // Only q and k are computed
    mem_size = mem_size / total_qkv * (q_per_kv + 1);
    // sin and cos
    mem_size += seq_length * rope_n_elem * 2;
    // BFloat16
    mem_size *= 2;

    ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
    for (int i = 0; i < 10; ++i) {
      clearL2Cache();
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
      cg_outputs = fe.runFusion(aten_inputs);
      FusionProfiler::stop();
      auto t = FusionProfiler::profile().kernel_time_ms;
      std::cout << "Elapsed time (us): " << (t * 1000) << "\n";
      std::cout << "Bandwidth (GB/s): "
                << ((float)mem_size * 0.001 * 0.001 * 0.001 / (t * 0.001))
                << "\n";
    }
  }

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
