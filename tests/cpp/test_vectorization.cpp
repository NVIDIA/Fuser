
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/interface_nodes.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using VectorizationTest = NVFuserTest;

TEST_F(VectorizationTest, InnerPersistent_fp16_32_64) {
  const std::vector<int64_t> input_shape = {256, 2048};
  // Fusion with 3 inputs and 3 outputs
  // inputs: half, float, double
  // outputs: half, float, double
  // all are vectorized as 16 bytes, so the vectorization factors are 8, 4,
  // and 2.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigConcreteTensor(input_shape, DataType::Half);
  TensorView* tv1 = makeContigConcreteTensor(input_shape, DataType::Float);
  TensorView* tv2 = makeContigConcreteTensor(input_shape, DataType::Double);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Double, tv0);
  auto tv4 = castOp(DataType::Double, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = add(tv5, tv2);
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 = div(tv6, tv8);
  auto tv10 = castOp(DataType::Half, tv9);
  auto tv11 = castOp(DataType::Float, tv9);
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);

  at::Tensor at_x_fp16 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0));
  at::Tensor at_x_fp32 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor at_x_fp64 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
  std::vector<c10::IValue> aten_inputs = {at_x_fp16, at_x_fp32, at_x_fp64};

  const int expected_vect_factor = 8;
  auto hp = getInnerPersistentHeuristics(fusion.get(), aten_inputs);
  NVF_CHECK(hp, "getInnerPersistentHeuristics failed!");
  EXPECT_EQ(hp->unroll_factor_inner_reduction, expected_vect_factor);
  scheduleInnerPersistentKernel(fusion.get(), *hp);
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    int64_t expected_val = 16L / dataTypeSize(tv->getDataType().value());
    EXPECT_TRUE(tv->axis(-1)->getParallelType() == ParallelType::Vectorize);
    EXPECT_TRUE(tv->axis(-1)->extent()->isConst());
    EXPECT_EQ(tv->axis(-1)->extent()->value(), expected_val);
  }
  FusionExecutor fe;
  fe.compileFusion(fusion.get(), aten_inputs, hp->lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, hp->lparams);
  auto t1 = at_x_fp16.to(at::kDouble) + at_x_fp32.to(at::kDouble) + at_x_fp64;
  auto t2 = t1.sum(-1).unsqueeze(-1);
  auto t3 = t1 / t2;
  auto t4 = t3.to(at::kHalf);
  auto t5 = t3.to(at::kFloat);
  testValidate(
      fusion.get(), cg_outputs, aten_inputs, {t3, t4, t5}, __LINE__, __FILE__);
}


TEST_F(VectorizationTest, OuterPersistent_fp16_32_64) {
  const std::vector<int64_t> input_shape = {16384, 8192};
  // Fusion with 3 inputs and 3 outputs
  // inputs: half, float, double
  // outputs: half, float, double
  // all are vectorized as 16 bytes, so the vectorization factors are 8, 4,
  // and 2.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigConcreteTensor(input_shape, DataType::Half);
  TensorView* tv1 = makeContigConcreteTensor(input_shape, DataType::Float);
  TensorView* tv2 = makeContigConcreteTensor(input_shape, DataType::Double);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Double, tv0);
  auto tv4 = castOp(DataType::Double, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = add(tv5, tv2);
  auto tv7 = sum(tv6, {0});
  auto tv8 = broadcast(tv7, {true, false});
  auto tv9 = div(tv6, tv8);
  auto tv10 = castOp(DataType::Half, tv9);
  auto tv11 = castOp(DataType::Float, tv9);
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);
  fusion->printMath();
  at::Tensor at_x_fp16 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0));
  at::Tensor at_x_fp32 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor at_x_fp64 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
  std::vector<c10::IValue> aten_inputs = {at_x_fp16, at_x_fp32, at_x_fp64};

  const int expected_vect_factor = 8;
  auto hp = getOuterPersistentHeuristics(fusion.get(), aten_inputs);
  NVF_CHECK(hp, "getInnerPersistentHeuristics failed!");
  EXPECT_EQ(hp->unroll_factor_iter_dom, expected_vect_factor);
  scheduleOuterPersistentKernel(fusion.get(), *hp);
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    int64_t expected_val = 16L / dataTypeSize(tv->getDataType().value());
    EXPECT_TRUE(tv->axis(-1)->getParallelType() == ParallelType::Vectorize);
    EXPECT_TRUE(tv->axis(-1)->extent()->isConst());
    EXPECT_EQ(tv->axis(-1)->extent()->value(), expected_val);
  }
  FusionExecutor fe;
  fe.compileFusion(fusion.get(), aten_inputs, hp->lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, hp->lparams);
  auto t1 = at_x_fp16.to(at::kDouble) + at_x_fp32.to(at::kDouble) + at_x_fp64;
  auto t2 = t1.sum(0).unsqueeze(0);
  auto t3 = t1 / t2;
  auto t4 = t3.to(at::kHalf);
  auto t5 = t3.to(at::kFloat);
  testValidate(
      fusion.get(), cg_outputs, aten_inputs, {t3, t4, t5}, __LINE__, __FILE__);
}


TEST_F(VectorizationTest, InnerOuterPersistent_fp16_32_64) {
  const std::vector<int64_t> input_shape = {256, 1024};
  // Fusion with 3 inputs and 3 outputs
  // inputs: half, float, double
  // outputs: half, float, double, double
  // all are vectorized as 16 bytes, so the vectorization factors are 8, 4,
  // and 2.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeContigConcreteTensor(input_shape, DataType::Half);
  TensorView* tv1 = makeContigConcreteTensor(input_shape, DataType::Float);
  TensorView* tv2 = makeContigConcreteTensor(input_shape, DataType::Double);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Double, tv0);
  auto tv4 = castOp(DataType::Double, tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = add(tv5, tv2);
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 = div(tv6, tv8);
  auto tv10 = castOp(DataType::Half, tv9);
  auto tv11 = castOp(DataType::Float, tv9);
  auto tv12 = sum(tv6, {0});
  fusion->addOutput(tv9);
  fusion->addOutput(tv10);
  fusion->addOutput(tv11);
  fusion->addOutput(tv12);

  at::Tensor at_x_fp16 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0));
  at::Tensor at_x_fp32 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor at_x_fp64 = at::randn(
      input_shape, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
  std::vector<c10::IValue> aten_inputs = {at_x_fp16, at_x_fp32, at_x_fp64};

  const int expected_vect_factor = 8;
  auto hp = getInnerOuterPersistentHeuristics(fusion.get(), aten_inputs);
  NVF_CHECK(hp, "getInnerPersistentHeuristics failed!");
  EXPECT_EQ(hp->unroll_factor_inner_reduction, expected_vect_factor);
  scheduleInnerOuterPersistentKernel(fusion.get(), *hp);
  for (auto tv : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    int64_t expected_val = 16L / dataTypeSize(tv->getDataType().value());
    EXPECT_TRUE(tv->axis(-1)->getParallelType() == ParallelType::Vectorize);
    EXPECT_TRUE(tv->axis(-1)->extent()->isConst());
    EXPECT_EQ(tv->axis(-1)->extent()->value(), expected_val);
  }
  FusionExecutor fe;
  fe.compileFusion(fusion.get(), aten_inputs, hp->lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, hp->lparams);
  auto t1 = at_x_fp16.to(at::kDouble) + at_x_fp32.to(at::kDouble) + at_x_fp64;
  auto t2 = t1.sum(-1).unsqueeze(-1);
  auto t3 = t1 / t2;
  auto t4 = t3.to(at::kHalf);
  auto t5 = t3.to(at::kFloat);
  auto t6 = t1.sum(0);
  testValidate(
      fusion.get(),
      cg_outputs,
      aten_inputs,
      {t3, t4, t5, t6},
      __LINE__,
      __FILE__,
      "",
      hp->lparams);
}
} // namespace nvfuser
