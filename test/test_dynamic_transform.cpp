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
#include <test/utils.h>
#include <test/validator.h>

#include <functional>

namespace nvfuser {

// Simple test of analyzing dynamic reshape
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
    TORCH_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
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
    TORCH_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
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

// Reshape a tensor like another tensor
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

    TORCH_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
  }
}

// Analyze dynamic reshape and concretize
TEST_F(NVFuserTest, DynamicTransform3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto reshape_shape0 = IrBuilder::create<Int>();
  auto reshape_shape1 = IrBuilder::create<Int>();

  auto tv2 = reshape(tv0, {reshape_shape0, reshape_shape1});
  auto tv3 = add(tv1, tv2);

  fusion.addOutput(tv3);

  std::vector<int64_t> shape_before({4, 3});
  std::vector<int64_t> shape_after({3, 4});

  ExpressionEvaluator expr_eval;

  // input: 4, 3
  // output: 3, 4
  expr_eval.bind(tv0->axis(0)->extent(), shape_before.at(0));
  expr_eval.bind(tv0->axis(1)->extent(), shape_before.at(1));
  expr_eval.bind(tv1->axis(0)->extent(), shape_after.at(0));
  expr_eval.bind(tv1->axis(1)->extent(), shape_after.at(1));

  auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, info);
  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn(shape_before, options);
  at::Tensor t1 = at::randn(shape_after, options);
  std::vector<c10::IValue> inputs = {t0, t1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);

  auto ref = t1 + t0.reshape(shape_after);

  testValidate(fec.fusion(), cg_outputs, inputs, {ref}, __LINE__, __FILE__);
}

// Test multiple patterns of reshape
TEST_F(NVFuserTest, DynamicTransform4_CUDA) {
  std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>>
      before_after_shapes = {
          {{4, 3}, {3, 4}},
          {{4, 3}, {12, 1}},
          {{4, 3}, {4, 3}},
          {{4, 6}, {4, 2, 3}},
      };
  for (const auto& before_after : before_after_shapes) {
    const auto& before_shape = before_after.first;
    const auto& after_shape = before_after.second;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(before_shape.size());
    fusion.addInput(tv0);
    auto tv1 = makeSymbolicTensor(after_shape.size());
    fusion.addInput(tv1);

    std::vector<Val*> shape_arg;
    for (const auto i : c10::irange(after_shape.size())) {
      (void)i;
      shape_arg.push_back(IrBuilder::create<Int>());
    }

    auto tv2 = reshape(tv0, shape_arg);

    // tv3 will also have symbolic axes
    auto tv3 = set(tv2);
    auto tv4 = add(tv1, tv3);

    fusion.addOutput(tv4);

    ExpressionEvaluator expr_eval;

    for (const auto i : c10::irange(before_shape.size())) {
      expr_eval.bind(tv0->axis((int)i)->extent(), before_shape.at(i));
    }

    for (const auto i : c10::irange(after_shape.size())) {
      expr_eval.bind(tv2->axis((int)i)->extent(), after_shape.at(i));
    }

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, info);

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
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
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(2);
    fusion.addInput(tv0);

    auto reshape_shape0 = IrBuilder::create<Int>();
    fusion.addInput(reshape_shape0);
    auto reshape_shape1 = IrBuilder::create<Int>();
    fusion.addInput(reshape_shape1);

    auto tv1 = reshape(tv0, {reshape_shape0, reshape_shape1});
    auto tv2 =
        pad(tv1,
            {IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1),
             IrBuilder::create<Int>(1)});
    auto tv3 = set(tv2);

    fusion.addOutput(tv3);

    ExpressionEvaluator expr_eval;

    expr_eval.bind(tv0->axis(0)->extent(), before_after.first.at(0));
    expr_eval.bind(tv0->axis(1)->extent(), before_after.first.at(1));
    expr_eval.bind(tv1->axis(0)->extent(), before_after.second.at(0));
    expr_eval.bind(tv1->axis(1)->extent(), before_after.second.at(1));

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, info);

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
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

    ExpressionEvaluator expr_eval;

    for (const auto i : c10::irange(reshape_list.size())) {
      const auto& shape = reshape_list.at(i);
      for (const auto j : c10::irange(shape.size())) {
        expr_eval.bind(reshape_tvs.at(i)->axis((int)j)->extent(), shape.at(j));
      }
    }

    auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, info);

    TORCH_CHECK(
        !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
  }
}

// Test equality of DynamicTransformInfo
TEST_F(NVFuserTest, DynamicTransform7_CUDA) {
  // Represents a series of reshapes
  struct TransformList {
    std::vector<std::vector<int64_t>> shapes;
  };

  struct ShapeInfo {
    TransformList ref_transform;
    std::vector<TransformList> equal_transforms;
    std::vector<TransformList> different_transforms;
  };

  std::vector<ShapeInfo> patterns;

  patterns.push_back(ShapeInfo{
      .ref_transform = {{{3, 4}, {4, 3}}},
      .equal_transforms =
          {{{{3, 4}, {4, 3}}}, {{{2, 8}, {4, 4}}}, {{{3, 8}, {4, 6}}}},
      .different_transforms = {{{{3, 4}, {2, 6}}}}});

  patterns.push_back(ShapeInfo{
      .ref_transform = {{{3, 4}, {12}, {1, 4, 3}}},
      .equal_transforms =
          {
              {{{3, 4}, {12}, {1, 4, 3}}},
              {{{5, 8}, {40}, {1, 4, 10}}},
          },
      .different_transforms = {
          {{{3, 4}, {12}, {4, 1, 3}}},
          {{{3, 4}, {12}, {4, 3, 1}}},
      }});

  for (const auto& pattern : patterns) {
    const auto& ref_transform = pattern.ref_transform;
    std::vector<TensorView*> reshape_tvs;

    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(ref_transform.shapes.at(0).size());
    fusion.addInput(tv0);

    reshape_tvs.push_back(tv0);

    for (auto it = ref_transform.shapes.begin() + 1;
         it != ref_transform.shapes.end();
         ++it) {
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

    ExpressionEvaluator ref_expr_eval;

    for (const auto i : c10::irange(ref_transform.shapes.size())) {
      const auto& shape = ref_transform.shapes.at(i);
      for (const auto j : c10::irange(shape.size())) {
        ref_expr_eval.bind(
            reshape_tvs.at(i)->axis((int)j)->extent(), shape.at(j));
      }
    }

    auto ref_info =
        DynamicTransform::getConcretizationInfo(&fusion, &ref_expr_eval);

    for (const auto& transform : pattern.equal_transforms) {
      TORCH_CHECK(transform.shapes.size() == ref_transform.shapes.size());
      ExpressionEvaluator expr_eval;
      for (const auto i : c10::irange(transform.shapes.size())) {
        const auto& shape = transform.shapes.at(i);
        for (const auto j : c10::irange(shape.size())) {
          expr_eval.bind(
              reshape_tvs.at(i)->axis((int)j)->extent(), shape.at(j));
        }
      }

      auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

      TORCH_CHECK(
          ref_info == info,
          "Expected to be equal: ",
          ref_info.toString(),
          "\n",
          info.toString());
    }

    for (const auto& transform : pattern.different_transforms) {
      TORCH_CHECK(transform.shapes.size() == ref_transform.shapes.size());
      ExpressionEvaluator expr_eval;
      for (const auto i : c10::irange(transform.shapes.size())) {
        const auto& shape = transform.shapes.at(i);
        for (const auto j : c10::irange(shape.size())) {
          expr_eval.bind(
              reshape_tvs.at(i)->axis((int)j)->extent(), shape.at(j));
        }
      }

      auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

      TORCH_CHECK(
          ref_info != info,
          "Expected to be different: ",
          ref_info.toString(),
          "\n",
          info.toString());
    }
  }
}

// Make sure non-dynamic reshape op is created when possible
TEST_F(NVFuserTest, DynamicTransform8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({3, 4});
  fusion.addInput(tv0);

  auto tv1 =
      reshape(tv0, {IrBuilder::create<Int>(4), IrBuilder::create<Int>(3)});
  fusion.addOutput(tv1);

  // Make sure the reshape is recognized as a static reshape
  TORCH_CHECK(
      !tv1->domain()->hasSymbolicAxis(),
      "Not expected to have symbolic axes: ",
      tv1->toString());
}

// Mix of static and dynamic reshape. Make sure only dynamic reshape
// is handled by the dynamic transform concretizer.
TEST_F(NVFuserTest, DynamicTransform9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, {3, 4}, {4, 3});

  auto reshape_shape0 = IrBuilder::create<Int>();

  auto tv2 = reshape(tv1, {reshape_shape0});
  fusion.addOutput(tv2);

  // The first reshape is static
  TORCH_CHECK(
      !tv1->domain()->hasSymbolicAxis(),
      "Unexpected to have symblic axes: ",
      tv1->toString());
  // The second reshape is static
  TORCH_CHECK(
      tv2->domain()->hasSymbolicAxis(),
      "Expected to have symblic axes: ",
      tv2->toString());

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 3);
  expr_eval.bind(tv0->axis(1)->extent(), 4);
  expr_eval.bind(reshape_shape0, 12);

  auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

  // There must be only one dynamic reshape entry, and that must be
  // for tv2.
  TORCH_CHECK(
      info.getReshapeTransforms().size() == 1,
      info.getReshapeTransforms().at(0).first == tv2,
      "Unexpected dynamic transform info:",
      info.toString());
}

// Make sure inherited symbolic IDs are concretized through rfactor exprs
TEST_F(NVFuserTest, DynamicTransform10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, {IrBuilder::create<Int>(), IrBuilder::create<Int>()});
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Int>(1),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Int>(1))}});
  fusion.addOutput(tv2);

  // tv2 has an rfactor expr (i.e., resize). The input to the expr is
  // symbolic, so is the output. When concretized, both of the input
  // and output must be concretized.

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 3);
  expr_eval.bind(tv0->axis(1)->extent(), 4);
  expr_eval.bind(tv1->axis(0)->extent(), 4);
  expr_eval.bind(tv1->axis(1)->extent(), 3);

  auto info = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, info);

  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");
}

// Simple test of hashing. Create concretization info objects with two
// similar but different reshape sizes and see if their hashes are different.
TEST_F(NVFuserTest, DynamicTransform11_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Int>(),
       IrBuilder::create<Int>(),
       IrBuilder::create<Int>()});
  fusion.addOutput(tv1);

  ExpressionEvaluator expr_eval1;
  // input: 4, 3
  // output: 2, 2, 3
  expr_eval1.bind(tv0->axis(0)->extent(), 4);
  expr_eval1.bind(tv0->axis(1)->extent(), 3);
  expr_eval1.bind(tv1->axis(0)->extent(), 2);
  expr_eval1.bind(tv1->axis(1)->extent(), 2);
  expr_eval1.bind(tv1->axis(2)->extent(), 3);

  auto info1 = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval1);

  ExpressionEvaluator expr_eval2;
  ;
  // input: 4, 3
  // output: 3, 2, 2
  expr_eval2.bind(tv0->axis(0)->extent(), 4);
  expr_eval2.bind(tv0->axis(1)->extent(), 3);
  expr_eval2.bind(tv1->axis(0)->extent(), 3);
  expr_eval2.bind(tv1->axis(1)->extent(), 2);
  expr_eval2.bind(tv1->axis(2)->extent(), 2);

  auto info2 = DynamicTransform::getConcretizationInfo(&fusion, &expr_eval2);

  // Generally different concretizations doesn't always mean different
  // hashes, but in this case they should be different
  auto hash1 = std::hash<DynamicTransformConcretizationInfo>{}(info1);
  auto hash2 = std::hash<DynamicTransformConcretizationInfo>{}(info2);
  TORCH_CHECK(
      hash1 != hash2,
      "Unexpected hash collision: ",
      hash1,
      " for\n",
      info1.toString(),
      "and\n",
      info2.toString());
}

} // namespace nvfuser
