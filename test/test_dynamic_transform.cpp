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

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(reshape_shape0);
  auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);
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
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 3L);
    expr_eval.bind(reshape_shape1, 4L);
    // We cannot infer the shape of tv1 from the above bound values, since
    // either axis of tv2 might be broadcast against one from tv1.
    expr_eval.bind(tv1->axis(0)->extent(), 3L);
    expr_eval.bind(tv1->axis(1)->extent(), 4L);

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
    TORCH_CHECK(
        info.getReshapeTransforms().size() == 1,
        "Expected to have one reshape transform: ",
        info.toString());
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 3, -1
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 3L);
    expr_eval.bind(reshape_shape1, -1L);

    // This should throw an exception since any reshape size of -1 must be
    // specified as a definition-time constant, as opposed to an input scalar.
    EXPECT_THAT(
        [&]() {
          auto initial_info = DynamicTransform::getInitialInfo(&fusion);
          auto info =
              DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
        },
        ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
            "Values of -1 passed to reshape must be constant at definition")));
  }

  {
    ExpressionEvaluator expr_eval;

    // input: 4, 3
    // output: 5, 4
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    expr_eval.bind(reshape_shape0, 5L);
    expr_eval.bind(reshape_shape1, 4L);

    // This should fail as (4 * 3) is not equal to (5 * 4)
    EXPECT_THAT(
        [&]() {
          auto initial_info = DynamicTransform::getInitialInfo(&fusion);
          auto info =
              DynamicTransformConcretizationInfo(&initial_info, &expr_eval);
        },
        ::testing::ThrowsMessage<c10::Error>(::testing::HasSubstr(
            "Total element counts across view operation must match.")));
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
    expr_eval.bind(tv0->axis(0)->extent(), 4L);
    expr_eval.bind(tv0->axis(1)->extent(), 3L);
    // Bind only tv2 extents. It should be enough as tv1 has the same
    // shape
    expr_eval.bind(tv2->axis(0)->extent(), 3L);
    expr_eval.bind(tv2->axis(1)->extent(), 4L);

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

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

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
  auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);

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
  // We cannot infer reshape_shape0 and reshape_shape1 from tv0's and tv1's
  // extents alone, since either of these reshaped extents could either match
  // that of tv1 or be 1, resulting in a broadcast.
  expr_eval.bind(reshape_shape0, shape_after.at(0));
  expr_eval.bind(reshape_shape1, shape_after.at(1));

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, &info);
  TORCH_CHECK(
      !fusion.hasDynamicTransform(), "Expected to have no dynamic transform");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
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
      shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
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
      // We must bind tv1's extents, since they cannot be inferred until after
      // concretization. Because tv2 is a dynamic reshape both its IterDomains
      // are Symbolic, which means both of tv3's IterDomains are also Symbolic.
      // tv1 has both IterDomains of type Iteration, but it since we add tv3 to
      // it to get tv4, we do not know whether this will resolve broadcasts from
      // tv3 or not until concretization.
      expr_eval.bind(tv1->axis((int)i)->extent(), after_shape.at(i));
    }

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

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

    auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(reshape_shape0);
    auto reshape_shape1 = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(reshape_shape1);

    auto tv1 = reshape(tv0, {reshape_shape0, reshape_shape1});
    auto tv2 =
        pad(tv1,
            {IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L),
             IrBuilder::create<Val>(1L)});
    auto tv3 = set(tv2);

    fusion.addOutput(tv3);

    ExpressionEvaluator expr_eval;

    expr_eval.bind(tv0->axis(0)->extent(), before_after.first.at(0));
    expr_eval.bind(tv0->axis(1)->extent(), before_after.first.at(1));
    expr_eval.bind(tv1->axis(0)->extent(), before_after.second.at(0));
    expr_eval.bind(tv1->axis(1)->extent(), before_after.second.at(1));

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

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
        shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
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

    auto initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

    DynamicTransform::concretizeFusion(&fusion, &info);

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
      const auto& shape = *it;
      std::vector<Val*> shape_arg;
      for (const auto i : c10::irange(shape.size())) {
        (void)i;
        shape_arg.push_back(IrBuilder::create<Val>(DataType::Int));
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

    auto ref_initial_info = DynamicTransform::getInitialInfo(&fusion);
    auto ref_info =
        DynamicTransformConcretizationInfo(&ref_initial_info, &ref_expr_eval);

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

      auto initial_info = DynamicTransform::getInitialInfo(&fusion);
      auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

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

      auto initial_info = DynamicTransform::getInitialInfo(&fusion);
      auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

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
      reshape(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(3L)});
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

  auto reshape_shape0 = IrBuilder::create<Val>(DataType::Int);

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

  expr_eval.bind(tv0->axis(0)->extent(), 3L);
  expr_eval.bind(tv0->axis(1)->extent(), 4L);
  expr_eval.bind(reshape_shape0, 12L);

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  // There must be only one dynamic reshape entry, and that must be
  // for tv2.
  TORCH_CHECK(
      info.getReshapeTransforms().size() == 1,
      info.getReshapeTransforms().at(0).first == 0, // first and only reshape
      "Unexpected dynamic transform info:",
      info.toString());
}

// Make sure inherited symbolic IDs are concretized through rfactor exprs
TEST_F(NVFuserTest, DynamicTransform10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = reshape(
      tv0,
      {IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int)});
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(1L),
        sub(tv1->axis(0)->extent(), IrBuilder::create<Val>(1L))}});
  fusion.addOutput(tv2);

  // tv2 has an rfactor expr (i.e., resize). The input to the expr is
  // symbolic, so is the output. When concretized, both of the input
  // and output must be concretized.

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 3L);
  expr_eval.bind(tv0->axis(1)->extent(), 4L);
  expr_eval.bind(tv1->axis(0)->extent(), 4L);
  expr_eval.bind(tv1->axis(1)->extent(), 3L);

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, &info);

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
      {IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int)});
  fusion.addOutput(tv1);

  ExpressionEvaluator expr_eval1;
  // input: 4, 3
  // output: 2, 2, 3
  expr_eval1.bind(tv0->axis(0)->extent(), 4L);
  expr_eval1.bind(tv0->axis(1)->extent(), 3L);
  expr_eval1.bind(tv1->axis(0)->extent(), 2L);
  expr_eval1.bind(tv1->axis(1)->extent(), 2L);
  expr_eval1.bind(tv1->axis(2)->extent(), 3L);

  auto initial_info1 = DynamicTransform::getInitialInfo(&fusion);
  auto info1 = DynamicTransformConcretizationInfo(&initial_info1, &expr_eval1);

  ExpressionEvaluator expr_eval2;
  ;
  // input: 4, 3
  // output: 3, 2, 2
  expr_eval2.bind(tv0->axis(0)->extent(), 4L);
  expr_eval2.bind(tv0->axis(1)->extent(), 3L);
  expr_eval2.bind(tv1->axis(0)->extent(), 3L);
  expr_eval2.bind(tv1->axis(1)->extent(), 2L);
  expr_eval2.bind(tv1->axis(2)->extent(), 2L);

  auto initial_info2 = DynamicTransform::getInitialInfo(&fusion);
  auto info2 = DynamicTransformConcretizationInfo(&initial_info2, &expr_eval2);

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

// Test FusionExecutorCache with dynamic reshapes
TEST_F(NVFuserTest, DynamicTransformFusionExecutorCache_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv1);

  auto tv2 = reshape(tv0, {tv1->axis(0)->extent(), tv1->axis(1)->extent()});
  auto tv3 = add(tv1, tv2);

  fusion->addOutput(tv3);

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

  FusionExecutorCache executor_cache(std::move(fusion));

  TORCH_CHECK(
      executor_cache.countRuntimes() == 0, "Expect to start with no runtimes");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  { // trivial reshape
    auto t0 = at::randn({3, 4}, options);
    auto t1 = at::randn({3, 4}, options);
    std::vector<c10::IValue> inputs = {t0, t1};
    auto cg_outputs = executor_cache.runFusionWithInputs(inputs);
    auto ref = t0 + t1;
    testValidate(
        executor_cache.fusion(), cg_outputs, inputs, {ref}, __LINE__, __FILE__);
    TORCH_CHECK(
        executor_cache.countRuntimes() == 1,
        "Expect to create a single runtime");
  }
  { // non-trivial reshape: merge and split
    auto t0 = at::randn({3, 4}, options);
    auto t1 = at::randn({4, 3}, options);
    std::vector<c10::IValue> inputs = {t0, t1};
    auto cg_outputs = executor_cache.runFusionWithInputs(inputs);
    auto ref = t0.view({4, 3}) + t1;
    testValidate(
        executor_cache.fusion(), cg_outputs, inputs, {ref}, __LINE__, __FILE__);
    auto num_rts = executor_cache.countRuntimes();
    auto num_concs = executor_cache.countConcretizations();
    TORCH_CHECK(num_rts == 2, "Non-trivial reshape should create new runtime");
    TORCH_CHECK(
        num_concs == 2,
        "Non-trivial reshape should create new concretization cache level");
  }
  { // different non-trivial reshape
    auto t0 = at::randn({2, 6}, options);
    auto t1 = at::randn({4, 3}, options);
    std::vector<c10::IValue> inputs = {t0, t1};
    auto cg_outputs = executor_cache.runFusionWithInputs(inputs);
    auto ref = t0.view({4, 3}) + t1;
    testValidate(
        executor_cache.fusion(), cg_outputs, inputs, {ref}, __LINE__, __FILE__);
    auto num_rts = executor_cache.countRuntimes();
    auto num_concs = executor_cache.countConcretizations();
    TORCH_CHECK(
        num_rts == 2,
        "Second non-trivial reshape should not create new runtime");
    TORCH_CHECK(
        num_concs == 2,
        "Second non-trivial reshape should not create new concretization cache level");
  }
}

using shape_t = std::vector<int64_t>;
using dynamic_view_invocation = std::tuple<
    shape_t, // input_shape
    shape_t, // output_shape
    bool // expect miss
    >;

//! Given a collection of input/output shapes test that FusionExecutorCache
//! properly caches concretized Fusions. The first argument is a vector of
//! input/output shape pairs. Each of these shape pairs will be run using the
//! same FusionExecutorCache. The argument expect_miss indicates whether we
//! expect a cache hit or miss at the concretization level.
//! reshape_before_reduction has the same meaning as in reductionViewAddFusion
//! in test_gpu_view.cpp.
void reductionDynamicViewAddFusion(
    std::vector<dynamic_view_invocation>& invocations,
    bool reshape_before_reduction) {
  constexpr int kReductionAxis = -1;

  auto input_dims = std::get<0>(invocations[0]).size();
  auto output_dims = std::get<1>(invocations[0]).size();

  auto bias_dims = (reshape_before_reduction) ? input_dims : output_dims;

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_dims);
  TensorView* bias = makeSymbolicTensor(bias_dims);
  fusion.addInput(x);
  fusion.addInput(bias);

  auto tv1 =
      (reshape_before_reduction) ? add(x, bias) : sum(x, {kReductionAxis});
  // create vectors of input scalars describing this reshape
  std::vector<Val*> output_shape(output_dims);
  for (size_t i : c10::irange(output_dims)) {
    output_shape[i] = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(output_shape[i]);
  }
  auto x_reshape = reshape(tv1, output_shape);
  auto y = (reshape_before_reduction) ? sum(x_reshape, {kReductionAxis})
                                      : add(x_reshape, bias);
  fusion.addOutput(y);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  size_t num_concretizations = fusion_executor_cache.countConcretizations();
  // Check that concretizations and runtimes are cache misses only when they
  // should be
  auto checkCache = [&](bool expect_miss) {
    auto current = fusion_executor_cache.countConcretizations();
    ASSERT_EQ(current, num_concretizations + (size_t)expect_miss);
    num_concretizations = current;
  };

  for (auto& inv : invocations) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();

    auto input_shape = std::get<0>(inv);
    auto output_shape = std::get<1>(inv);
    auto expect_miss = std::get<2>(inv);

    TORCH_INTERNAL_ASSERT(input_shape.size() == input_dims);
    TORCH_INTERNAL_ASSERT(output_shape.size() == output_dims);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor at_x = at::randn(input_shape, options);
    auto bias_shape = (reshape_before_reduction) ? input_shape : output_shape;
    if (!reshape_before_reduction) {
      // When bias_shape = output_shape, it may contain -1s
      // concretize bias_shape so that we can properly initialize at_bias
      size_t other_numel = 1;
      ssize_t negone_dim = -1; // negative if no -1 shape is provided
      for (auto i : c10::irange(bias_shape.size())) {
        if (bias_shape[i] == -1) {
          ASSERT_EQ(negone_dim, -1); // test cases should not have multiple -1s
          negone_dim = -1;
        } else {
          other_numel *= bias_shape[i];
        }
      }
      if (negone_dim >= 0) {
        bias_shape[negone_dim] = (int64_t)at_x.numel() / (int64_t)other_numel;
      }
    }
    at::Tensor at_bias = at::randn(bias_shape, options);
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};
    // Add input scalars describing the reshape size for concretization
    for (size_t i : c10::irange(output_dims)) {
      aten_inputs.emplace_back(output_shape[i]);
    }

    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
    checkCache(expect_miss);

    auto at_tv1 = (reshape_before_reduction) ? (at_x + at_bias)
                                             : at::sum(at_x, kReductionAxis);
    auto at_x_reshape = at::native::view(at_tv1, output_shape);
    auto at_y = (reshape_before_reduction)
        ? at::sum(at_x_reshape, kReductionAxis)
        : at::add(at_x_reshape, at_bias);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionDynamicReshapeReductionShmoo_CUDA) {
  auto invocations = std::vector<dynamic_view_invocation>{
      {{8, 3 * 4, 7, 9}, {8, 3 * 4, 7, 9}, true}, // trivial
      {{8, 3 * 4, 7, 5}, {8, 3 * 4, 7, 5}, false}, // trivial
      {{8, 3 * 4, 7, 9}, {8, 3, 4, 7 * 9}, true}, // merge(2) osplit(1, 3)
      {{8, 3 * 4, 7, 9},
       {8, 3, 4 * 7, 9},
       true}, // merge(1) merge(2) osplit(1, 3)
      {{8, 3 * 4, 7, 5},
       {8, 3, 4 * 7, 5},
       false}, // merge(1) merge(2) osplit(1, 3)
      {{8, 3 * 5, 7, 9}, {8, 3, 5 * 7, 9}, false}, // merge(1) osplit(1, 3)

      // test passing -1 dynamically for dimension size
      // This is unsupported. See https://github.com/NVIDIA/Fuser/issues/249
      // Values of -1 must be passed as constants instead of input-dependent
      // scalars.
      //{{8, 3 * 5, 7, 9}, {8, 3, -1, 9}, false} // merge(1) osplit(1, 3)
  };
  reductionDynamicViewAddFusion(
      invocations, true /* reshape_before_reduction */);
}

using dynamic_pad_invocation = std::tuple<
    std::vector<int64_t>, // input_shape
    std::vector<int64_t>, // pad_widths
    bool // expect miss
    >;

void reductionDynamicPadAddFusion(
    std::vector<dynamic_pad_invocation>& invocations) {
  constexpr int kReductionAxis = -1;

  auto input_dims = std::get<0>(invocations[0]).size();
  auto num_pad_widths = std::get<1>(invocations[0]).size();

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_dims);
  fusion.addInput(x);

  std::vector<Val*> pad_width_vals(num_pad_widths);
  for (auto i : c10::irange(num_pad_widths)) {
    pad_width_vals[i] = IrBuilder::create<Val>(DataType::Int);
    fusion.addInput(pad_width_vals[i]);
  }
  auto x_pad = pad(x, pad_width_vals);
  auto y = sum(x_pad, {kReductionAxis});
  fusion.addOutput(y);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  // Check that concretizations and runtimes are cache misses only when they
  // should be
  size_t num_concretizations = fusion_executor_cache.getKernelRuntimes().size();
#define CHECK_CACHE(expect_miss, ...)                              \
  auto current = fusion_executor_cache.getKernelRuntimes().size(); \
  auto expected = num_concretizations + (size_t)expect_miss;       \
  TORCH_CHECK(                                                     \
      current == expected,                                         \
      "Expected cache size ",                                      \
      expected,                                                    \
      " but found ",                                               \
      current,                                                     \
      ". ",                                                        \
      __VA_ARGS__);                                                \
  num_concretizations = current;

  for (auto& inv : invocations) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();

    auto input_shape = std::get<0>(inv);
    auto pad_widths = std::get<1>(inv);
    auto expect_miss = std::get<2>(inv);

    TORCH_INTERNAL_ASSERT(input_shape.size() == input_dims);
    TORCH_INTERNAL_ASSERT(pad_widths.size() == num_pad_widths);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

    at::Tensor at_x = at::randn(input_shape, options);
    std::vector<c10::IValue> aten_inputs = {at_x};
    // Add input scalars describing the reshape size for concretization
    for (size_t i : c10::irange(pad_widths.size())) {
      aten_inputs.emplace_back(pad_widths[i]);
    }

    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
    CHECK_CACHE(
        expect_miss, "Input shape=", input_shape, " pad_widths=", pad_widths);

    auto at_x_pad = at::pad(at_x, pad_widths);
    auto at_y = at::sum(at_x_pad, kReductionAxis);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}
#undef CHECK_CACHE

// Test dynamic pad for various inputs
TEST_F(NVFuserTest, DynamicPadShmoo_CUDA) {
  // NOLINTBEGIN(bugprone-implicit-widening-of-multiplication-result)
  auto invocations = std::vector<dynamic_pad_invocation>{
      {{3, 5}, {0, 0}, true}, // trivial

      {{3, 5}, {2, 1}, false}, // simple pad of both sides
      {{3, 5}, {-1, 1}, false}, // shift by one
      // The following fails with a SIGFPE in innerReductionHeuristic
      // See https://github.com/NVIDIA/Fuser/issues/264
      //{{3, 5}, {-3, -2}, false}, // output is zero-dimensional

      // Output has size 1 so is set to broadcast.
      // This was previously "working" by concretizing the size-1 pad to
      // Iteration, even though it should be Broadcast. When set properly to
      // Broadcast, it fails with an error in ConcretizedBroadcastDomains.
      //{{3, 5}, {0, -4}, true},

      // Test full negative shifts, so output doesn't overlap input
      {{3, 5}, {-5, 2}, false},
      {{3, 5}, {2, -5}, false}, // full shift the other direction, re-use

      // The following reuses the schedule of {3, 5} inputs, and does not set
      // broadcast on the second input dimension.
      {{3, 1}, {1, 1}, false},

      // Test zero-dimensional input
      //{{3, 0}, {0, 0}, false}, // SIGFPE (see #264 above)
      {{3, 0}, {1, 1}, true}, // zero-dimensional concretizes differently
      //{{3, 0}, {-1, 1}, false}, // SIGFPE (see #264 above)
  };
  // NOLINTEND(bugprone-implicit-widening-of-multiplication-result)
  reductionDynamicPadAddFusion(invocations);
}

// Test that a Symbolic root/Broadcast rfactor is not  concretized to
// Iteration/Iteration
TEST_F(NVFuserTest, FusionDynamicSliceToBroadcast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  // tv0[:2] introduces symbolic IterDomain
  auto tv1 = slice(
      tv0, {{fusion.zeroVal(), IrBuilder::create<Val>(2L), fusion.oneVal()}});
  // tv1 has Broadcast rfactor, Iteration root
  auto tv2 = slice(tv1, {{fusion.zeroVal(), fusion.oneVal(), fusion.oneVal()}});
  // tv2 has a Symbolic root related to a Broadcast rfactor through a Resize op
  fusion.addOutput(tv2);

  // At concretization, tv1's rfactor will be set to Iteration, which will
  // propagate to tv2s root. This test will test that when tv2 root is
  // concretized to Iteration, it does not wind up overwriting the Broadcast
  // rfactor.

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  std::vector<c10::IValue> aten_inputs = {at0};
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
  auto at1 = at::slice(at0, 0, 0, 2);
  auto at2 = at::slice(at1, 0, 0, 1);
  testValidate(&fusion, outputs, aten_inputs, {at2}, __LINE__, __FILE__);
}

// Test that empty input to cat is concretized away
TEST_F(NVFuserTest, FusionDynamicEmptyCat1_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = cat({tv0, tv1, tv2}, 0);

  fusion.addOutput(tv3);

  // Check correctness
  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  at::Tensor at1 = at::randn({0}, options);
  at::Tensor at2 = at::randn({3}, options);
  std::vector<c10::IValue> aten_inputs = {at0, at1, at2};
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
  auto at3 = at::cat({at0, at1, at2}, 0);
  testValidate(&fusion, outputs, aten_inputs, {at3}, __LINE__, __FILE__);
}

// Test that empty input to cat is concretized away
TEST_F(NVFuserTest, FusionDynamicEmptyCat2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = cat({tv0, tv1}, 0);

  fusion.addOutput(tv2);

  // Check correctness
  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({5}, options);
  at::Tensor at1 = at::randn({0}, options);
  std::vector<c10::IValue> aten_inputs = {at0, at1};
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
  auto at2 = at::cat({at0, at1}, 0);
  testValidate(&fusion, outputs, aten_inputs, {at2}, __LINE__, __FILE__);

  // Check that fusion consists only of tv2 = set(tv0)
  auto fkr = fusion_executor_cache.getMostRecentKernelRuntime();
  auto seg_fusion = fkr->fusionSegments();
  auto output_def = seg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(output_def->isA<LoadStoreOp>());
  EXPECT_EQ(output_def->as<LoadStoreOp>()->opType(), LoadStoreOpType::Set);
  EXPECT_EQ(output_def->input(0), seg_fusion->inputs()[0]);
}

// Repro of https://github.com/NVIDIA/Fuser/issues/418
TEST_F(NVFuserTest, DynamicTransformIssue418_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  auto s0 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(s0);

  auto sh = tensor_sizes(tv0);
  auto tv1 = reshape(tv0, {sh[0], div(sh[1], s0), s0, sh[2], sh[3]});
  // Reducing along axis 2 in tv1 is equivalent to a partial reduction across
  // axis 1 of tv0.
  auto vm = variance_mean(tv1, {2, 3, 4}, 0, true);
  fusion->addOutput(vm.mean);
  fusion->addOutput(vm.var);

  FusionExecutorCache fusion_executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({256, 128, 28, 28}, options);
  std::vector<c10::IValue> aten_inputs = {at0, 32};
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);
  auto at1 = at0.reshape({256, 4, 32, 28, 28});
  auto atmean = at1.mean({2, 3, 4}, /*keepdim*/ true);
  auto atvar = at1.var({2, 3, 4}, /*unbiased*/ true, /*keepdim*/ true);

  testValidate(
      fusion_executor_cache.fusion(),
      outputs,
      aten_inputs,
      {atmean, atvar},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, Issue249_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, tv0);
  auto tv2 = reshape(
      tv1,
      {tv1->axis(0)->extent(),
       tv1->axis(2)->extent(),
       IrBuilder::create<Val>(-1L)});
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({2, 3, 4, 5}, options);
  auto at_y = (at_x + at_x).reshape({2, 4, -1});
  auto at_z = at_y + at_y;

  auto outputs = fusion_executor_cache.runFusionWithInputs({at_x});

  testValidate(
      fusion_executor_cache.fusion(),
      outputs,
      {at_x},
      {at_z},
      __LINE__,
      __FILE__);
}

// This is just like the test above, but uses an input scalar with value -1
TEST_F(NVFuserTest, Issue249InputNegative1_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  auto s0 = IrBuilder::create<Val>(DataType::Int);
  auto s1 = IrBuilder::create<Val>(DataType::Int);
  auto s2 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(s0);
  fusion.addInput(s1);
  fusion.addInput(s2);

  auto tv1 = add(tv0, tv0);
  auto tv2 = reshape(tv1, {s0, s1, s2});
  auto tv3 = add(tv2, tv2);
  fusion.addOutput(tv3);

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({2, 3, 4, 5}, options);
  auto at_y = (at_x + at_x).reshape({2, 4, -1});
  auto at_z = at_y + at_y;

  // Dynamic reshape sizes that are not constant at definition must be explicit:
  // no -1 allowed
  EXPECT_THROW(
      fusion_executor_cache.runFusionWithInputs({at_x, 2, 4, -1}),
      std::exception);

  // Passing explicit sizes works fine
  auto outputs = fusion_executor_cache.runFusionWithInputs({at_x, 2, 4, 15});

  testValidate(
      fusion_executor_cache.fusion(),
      outputs,
      {at_x, 2, 4, 15},
      {at_z},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
