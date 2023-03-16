// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <dynamic_transform.h>
#include <expr_evaluator.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {

TEST_F(NVFuserTest, DynamicTransform1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape0);
  auto reshape_shape1 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape1);

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  auto tv3 = add(tv1, tv2);

  fusion.addOutput(tv3);

  fusion.printMath();

  // tv2 has symbolic axes as reshape is dynamic
  TORCH_CHECK(
      tv2->domain()->hasSymbolicAxis(),
      "Expected to have symbolic axes: ",
      tv2->toString());

  // The symbolic axes of tv2 should not be propagated to tv3 as tv1
  // is fully concrete
  TORCH_CHECK(
      !tv3->domain()->hasSymbolicAxis(),
      "Not expected to have symbolic axes: ",
      tv3->toString());

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4);
    expr_eval.bind(tv0->axis(1)->extent(), 3);
    expr_eval.bind(reshape_shape0, 3);
    expr_eval.bind(reshape_shape1, 4);

    auto info = DynamicTransformInfo::get(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, -1
    expr_eval.bind(tv0->axis(0)->extent(), 4);
    expr_eval.bind(tv0->axis(1)->extent(), 3);
    expr_eval.bind(reshape_shape0, 3);
    expr_eval.bind(reshape_shape1, -1);

    auto info = DynamicTransformInfo::get(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 5, -1
    expr_eval.bind(tv0->axis(0)->extent(), 4);
    expr_eval.bind(tv0->axis(1)->extent(), 3);
    expr_eval.bind(reshape_shape0, 5);
    expr_eval.bind(reshape_shape1, -1);

    // This should fail as (4 * 3) is not evenly divisible by 5
    EXPECT_THAT(
        [&]() { DynamicTransformInfo::get(&fusion, &expr_eval); },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Cannot infer")));
  }
}

TEST_F(NVFuserTest, DynamicTransform2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // All tensors are 2D symbolic tensors. tv1 and tv2 have the same shape

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  // Reshape to the same shape as tv1
  auto tv3 = reshape(tv0, {tv1->axis(0)->extent(), tv1->axis(1)->extent()});
  auto tv4 = add(tv1, tv2);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  fusion.printMath();

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4);
    expr_eval.bind(tv0->axis(1)->extent(), 3);
    // Bind only tv2 extents. It should be enough as tv1 has the same
    // shape
    expr_eval.bind(tv2->axis(0)->extent(), 3);
    expr_eval.bind(tv2->axis(1)->extent(), 4);

    auto info = DynamicTransformInfo::get(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;
  }
}

TEST_F(NVFuserTest, DynamicTransform3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape0);
  auto reshape_shape1 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape1);

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  auto tv3 = add(tv1, tv2);

  fusion.addOutput(tv3);

  fusion.printMath();

  ExpressionEvaluator expr_eval;

  // input: 4, 3
  // output: 3, 4
  expr_eval.bind(tv0->axis(0)->extent(), 4);
  expr_eval.bind(tv0->axis(1)->extent(), 3);
  expr_eval.bind(reshape_shape0, 3);
  expr_eval.bind(reshape_shape1, 4);

  auto info = DynamicTransformInfo::get(&fusion, &expr_eval);

  std::cerr << info.toString() << std::endl;

  DynamicTransformConcretizer::concretizeFusion(&fusion, info);

  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  fusion.printMath();
}

TEST_F(NVFuserTest, DynamicTransform4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape0);
  auto reshape_shape1 = IrBuilder::create<Int>();
  fusion.addInput(reshape_shape1);

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  // tv3 will also have symbolic axes
  auto tv3 = set(tv2);
  auto tv4 = add(tv1, tv3);

  fusion.addOutput(tv4);

  fusion.printMath();

  ExpressionEvaluator expr_eval;

  // input: 4, 3
  // output: 3, 4
  expr_eval.bind(tv0->axis(0)->extent(), 4);
  expr_eval.bind(tv0->axis(1)->extent(), 3);
  // Bind only tv2 extents. It should be enough as tv1 has the same
  // shape
  expr_eval.bind(tv2->axis(0)->extent(), 3);
  expr_eval.bind(tv2->axis(1)->extent(), 4);

  auto info = DynamicTransformInfo::get(&fusion, &expr_eval);

  std::cerr << info.toString() << std::endl;

  DynamicTransformConcretizer::concretizeFusion(&fusion, info);

  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  fusion.printMath();
}

} // namespace nvfuser
