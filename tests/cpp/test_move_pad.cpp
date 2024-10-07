// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::Contains;
using testing::IsTrue;
using testing::Property;
using testing::UnorderedElementsAre;

using MovePadTest = NVFuserTest;

TEST_F(MovePadTest, UnaryCat) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({2, 10});
  TensorView* tv2 = relu(tv0);
  TensorView* tv3 = cat({tv1, tv2}, /*dim=*/0);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({2, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 1);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, BinaryCat) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({4, 10});
  TensorView* tv2 = makeContigConcreteTensor({2, 10});
  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = cat({tv2, tv3}, /*dim=*/0);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({4, 10}, options);
  at::Tensor t2 = at::randn({2, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 1);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, BinaryBroadcastOnNonCatDim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({10});
  // broadcast on non-pad dimension should propagate across binary operations
  TensorView* tv1_b = broadcast(tv1, {true, false});
  TensorView* tv2 = makeContigConcreteTensor({4, 5});
  // tv1_b [bS, iS]. pad axes [1,] is not on broadcast dimension, we should be
  // able to propagate pad
  TensorView* tv3 = add(tv0, tv1_b);
  TensorView* tv4 = cat({tv2, tv3}, /*dim=*/1);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({10}, options);
  at::Tensor t2 = at::randn({4, 5}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  // ensure that we propagate the pad across binary operation and the first
  // segment is no-op
  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::NoOp),
          HeuristicIs(SchedulerType::PointWise)));

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, BinaryBroadcastOnCatDim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({10});
  TensorView* tv1_b = broadcast(tv1, {true, false});
  TensorView* tv2 = makeContigConcreteTensor({2, 10});
  // tv1_b [bS, iS]. pad axes [0,] is includes broadcast dimension, pad
  // propagation stops here
  TensorView* tv3 = add(tv0, tv1_b);
  TensorView* tv4 = cat({tv2, tv3}, /*dim=*/0);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({10}, options);
  at::Tensor t2 = at::randn({2, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, PadReplayOnMultipleUsesCase0) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({1, 10});
  // pad on tv5 will propagation back to tv0, since all its uses are traversed
  //
  //   tv0 --> tv2 --> tv3 --> tv5
  //               \        /
  //                -> tv4 -
  TensorView* tv2 = relu(tv0);
  TensorView* tv3 = neg(tv2);
  TensorView* tv4 = sin(tv2);
  TensorView* tv5 = add(tv3, tv4);
  TensorView* tv6 = cat({tv5, tv1}, /*dim=*/0);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({1, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 1);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, PadReplayOnMultipleUsesCase1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 = makeContigConcreteTensor({4, 10});
  // pad on tv5 will NOT propagation back to tv0, since there's one path
  //   tv0 --> tv2 --> tv7
  // which is not covered by reverse traversing through producers of tv5.
  // So the pad propagation would stop at tv2
  TensorView* tv2 = relu(tv0);
  TensorView* tv3 = neg(tv2);
  TensorView* tv4 = sin(tv2);
  TensorView* tv5 = add(tv3, tv4);
  TensorView* tv6 = cat({tv5, tv1}, /*dim=*/0);
  TensorView* tv7 = sin(tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv6);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  at::Tensor t1 = at::randn({4, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, CascadePadCase0) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // all pad operations should be merged together
  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(2L),
           IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(1L)});
  TensorView* tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(0L)});
  TensorView* tv3 =
      pad(tv2,
          {IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L)},
          IrBuilder::create<Val>(0.0));
  auto s4 = IrBuilder::create<Val>(4.0);
  auto s5 = IrBuilder::create<Val>(4.0);
  auto s6 = sub(s4, s5);
  TensorView* tv7 =
      pad(tv3,
          {IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L)},
          s6);

  fusion->addInput(tv0);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  Fusion* complete_fusion = runtime->fusionSegments()->completeFusion();
  std::vector<Expr*> exprs = complete_fusion->exprs();
  EXPECT_THAT(exprs, Contains(Property(&Expr::isA<PadOp>, IsTrue())).Times(1));

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, CascadePadCase1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(2L),
           IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(1L)});
  // PadOp with different pad value cannot be merged
  TensorView* tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L)},
          IrBuilder::create<Val>(1.0));

  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  Fusion* complete_fusion = runtime->fusionSegments()->completeFusion();
  std::vector<Expr*> exprs = complete_fusion->exprs();
  EXPECT_THAT(exprs, Contains(Property(&Expr::isA<PadOp>, IsTrue())).Times(2));

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, CascadePadCase2) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // some pad outputs are output of the fusion, we cannot merge pad in this
  // instance.
  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(2L),
           IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(1L)});
  TensorView* tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(0L)});
  TensorView* tv3 =
      pad(tv2,
          {IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L)},
          IrBuilder::create<Val>(0.0));
  auto s4 = IrBuilder::create<Val>(4.0);
  auto s5 = IrBuilder::create<Val>(4.0);
  auto s6 = sub(s4, s5);
  TensorView* tv7 =
      pad(tv3,
          {IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L)},
          s6);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, NotMergeNegativePad) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10});
  TensorView* tv1 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(-2L),
           IrBuilder::create<Val>(1L),
           IrBuilder::create<Val>(1L)});
  TensorView* tv2 =
      pad(tv1,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(3L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(0L)});
  TensorView* tv3 = relu(tv2);

  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  testValidate(fec.fusion(), out_tensors, aten_inputs, __LINE__, __FILE__);
}

TEST_F(MovePadTest, BooleanCat) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigConcreteTensor({4, 10}, DataType::Bool);
  TensorView* tv1 = makeContigConcreteTensor({4, 10}, DataType::Bool);
  TensorView* tv2 = makeContigConcreteTensor({2, 10}, DataType::Bool);
  TensorView* tv3 = logical_and(tv0, tv1);
  TensorView* tv4 = cat({tv3, tv2}, /*dim=*/0);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 10}, options) > 0.5;
  at::Tensor t1 = at::randn({4, 10}, options) > 0.5;
  at::Tensor t2 = at::randn({2, 10}, options) > 0.5;
  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs(aten_inputs);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 1);

  // ExpressionEvaluator is hitting an assert with dynamic value.
  // https://github.com/NVIDIA/Fuser/issues/2697 testValidate(fec.fusion(),
  // out_tensors, aten_inputs, __LINE__, __FILE__);
  at::Tensor ref = at::cat({at::bitwise_and(t0, t1), t2}, 0);
  testValidate(
      fec.fusion(), out_tensors, aten_inputs, {ref}, __LINE__, __FILE__);
}

} // namespace nvfuser
