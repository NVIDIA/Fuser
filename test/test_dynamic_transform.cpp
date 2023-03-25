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

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

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

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

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
        [&]() { DynamicTransform::getConcretizationInfo(&fusion, &expr_eval); },
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

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

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

  auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

  std::cerr << info.toString() << std::endl;

  DynamicTransform::concretizeFusion(&fusion, info);

  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  fusion.printMath();
}

TEST_F(NVFuserTest, DynamicTransform4_CUDA) {
#if 1
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {3, 4}},
          {{4, 3}, {12, 1}},
      };
#else
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {12, 1}},
      };
#endif
  for (auto before_after : before_after_shapes) {
    std::cerr << "Before: " << before_after.first << std::endl;
    std::cerr << "After: " << before_after.second << std::endl;
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

    expr_eval.bind(tv0->axis(0)->extent(), before_after.first.at(0));
    expr_eval.bind(tv0->axis(1)->extent(), before_after.first.at(1));
    expr_eval.bind(tv2->axis(0)->extent(), before_after.second.at(0));
    expr_eval.bind(tv2->axis(1)->extent(), before_after.second.at(1));

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;

    std::cout << "Before fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    DynamicTransform::concretizeFusion(&fusion, info);

    std::cout << "After fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

    fusion.printMath();
  }
}

// Dynamic reshape followed by static resize
TEST_F(NVFuserTest, DynamicTransform5_CUDA) {
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {3, 4}},
          //{{4, 3}, {12, 1}}, not possible to do pad a broadcast domain yet
      };

  for (auto before_after : before_after_shapes) {
    std::cerr << "Before: " << before_after.first << std::endl;
    std::cerr << "After: " << before_after.second << std::endl;
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
    auto tv3 =
        pad(tv2,
            {IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1)});
    auto tv4 = set(tv3);

    fusion.addOutput(tv4);

    fusion.printMath();

    ExpressionEvaluator expr_eval;

    expr_eval.bind(tv0->axis(0)->extent(), before_after.first.at(0));
    expr_eval.bind(tv0->axis(1)->extent(), before_after.first.at(1));
    expr_eval.bind(tv2->axis(0)->extent(), before_after.second.at(0));
    expr_eval.bind(tv2->axis(1)->extent(), before_after.second.at(1));

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;

    std::cout << "Before fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    DynamicTransform::concretizeFusion(&fusion, info);

    std::cout << "After fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

    fusion.printMath();
  }
}

// Reshape of reshape
TEST_F(NVFuserTest, DynamicTransform6_CUDA) {
  std::vector<std::vector<std::vector<int64_t>>> reshape_lists = {
      {{4, 3}, {3, 4}},
      {{4, 3}, {3, 4}, {12}},
      {{4, 3}, {3, 1, 4}, {12, 1}},
      {{4, 3}, {12}, {3, 4}},
      {{4, 3}, {1, 2, 1, 3, 2}, {3, 4}},
  };

  for (auto reshape_list : reshape_lists) {
    std::cerr << "Reshape list: " << reshape_list << std::endl;
    std::vector<TensorView*> reshape_tvs;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(reshape_list.at(0).size());
    fusion.addInput(tv0);

    reshape_tvs.push_back(tv0);

    for (auto it = reshape_list.begin() + 1; it != reshape_list.end(); ++it) {
      auto shape = *it;
      std::vector<Val*> shape_arg;
      for (const auto i : c10::irange(shape.size())) {
        (void)i;
        shape_arg.push_back(IrBuilder::create<Int>());
      }

      auto tv = reshape(reshape_tvs.back(), shape_arg);
      reshape_tvs.push_back(tv);
    }
    fusion.addOutput(reshape_tvs.back());

    fusion.printMath();

    ExpressionEvaluator expr_eval;

    for (const auto i : c10::irange(reshape_list.size())) {
      const auto& shape = reshape_list.at(i);
      for (const auto j : c10::irange(shape.size())) {
        expr_eval.bind(reshape_tvs.at(i)->axis(j)->extent(), shape.at(j));
      }
    }

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    std::cerr << info.toString() << std::endl;

    std::cout << "Before fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    DynamicTransform::concretizeFusion(&fusion, info);

    std::cout << "After fusion concretization\n";
    fusion.printMath();
    std::cout << std::endl;

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

    fusion.printMath();
  }
}

} // namespace nvfuser
