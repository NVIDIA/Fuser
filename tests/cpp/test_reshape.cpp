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

#include <codegen.h>
#include <device_lower/analysis/divisible_split.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/registry_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>
#include <transform_view.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using testing::UnorderedElementsAre;

class ReshapeTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_F(ReshapeTest, ViewDtypeSameSizeOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};

  TensorView* x = makeSymbolicTensor(input_shape.size(), DataType::Float);
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_view = view(x_add_bias, DataType::Int32);
  fusion.addOutput(x_view);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias})
          .outputs;
  testValidate(&fusion, cg_outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ViewDtypeFailMismatchSize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};

  TensorView* x = makeSymbolicTensor(input_shape.size(), DataType::Float);
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(view(x_add_bias, DataType::Int));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(view(x_add_bias, DataType::Half));
}

TEST_F(ReshapeTest, ViewAsRealOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // TODO: We should modify our schedulers to correctly handle
  // view_as_real. And test these schedulers.
  std::vector<int64_t> input_shape{512};
  std::vector<int64_t> output_shape{512, 2};

  TensorView* x =
      makeSymbolicTensor(input_shape.size(), DataType::ComplexFloat);
  TensorView* bias =
      makeSymbolicTensor(input_shape.size(), DataType::ComplexFloat);
  fusion.addInput(x);
  fusion.addInput(bias);

  TensorView* y = makeSymbolicTensor(output_shape.size());
  fusion.addInput(y);

  auto y_plus_1 = add(y, IrBuilder::create<Val>(1.0));

  auto x_add_bias = add(x, bias);
  auto x_view = view_as_real(x_add_bias);
  auto out = add(y_plus_1, x_view);
  fusion.addOutput(out);

  out->axis(0)->parallelize(ParallelType::TIDx);
  x_add_bias->computeAt(out, -1);
  y->computeAt(out, -1);

  auto in_options =
      at::TensorOptions().dtype(at::kComplexFloat).device(at::kCUDA, 0);
  auto out_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, in_options);
  at::Tensor at_bias = at::randn(input_shape, in_options);
  at::Tensor at_y = at::randn(output_shape, out_options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_x, at_bias, at_y});
  auto outputs = ke.run({at_x, at_bias, at_y});

  testValidate(&fusion, outputs, {at_x, at_bias, at_y}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeRfactorExtentReplacement) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion->addInput(tv1);

  auto tv2 = reshape(tv0, {12, 8}, {4, 3, 8});
  auto tv3 = sum(tv2, {-1});
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  auto tv5 = add(tv1, tv4);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({12, 8}, options);
  auto t1 = at::randn({4, 3}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, 10, 4, 10};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_reshape = reshape(x_add_bias, input_shape, output_shape);
  fusion.addOutput(x_reshape);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias})
          .outputs;
  testValidate(&fusion, cg_outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeFailMismatchSize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // The number of elements in input and output shapes do not match,
  // so this reshape transformation is invalid.
  // 2 * 10 * 40 != 2 * 50 * 4 * 10

  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, 50, 4, 10};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(reshape(x_add_bias, input_shape, output_shape));
}

TEST_F(ReshapeTest, ReshapeFailMulitDimInference) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Only one dimension can be inferred in the output shape.
  // Otherwise, the size of the dimensions is ambiguous.
  std::vector<int64_t> input_shape{2, 10, 40};
  std::vector<int64_t> output_shape{2, -1, 4, -1};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(reshape(x_add_bias, input_shape, output_shape));
}

void reductionViewAddFusion(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& output_shape,
    const bool has_implicit_broadcast,
    const bool reshape_before_reduction) {
  constexpr int kReductionAxis = -1;

  // Drop size for reduction axis from reshape_shape
  std::vector<int64_t> reshape_shape;
  {
    const auto kAxis = (kReductionAxis < 0)
        ? (kReductionAxis + input_shape.size())
        : kReductionAxis;
    for (auto i : arange(input_shape.size())) {
      if (reshape_before_reduction || i != kAxis) {
        reshape_shape.push_back(input_shape[i]);
      }
    }
  }

  auto bias_shape = (reshape_before_reduction) ? input_shape : output_shape;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = (has_implicit_broadcast)
      ? makeConcreteTensor(input_shape)
      : makeSymbolicTensor(input_shape.size());
  TensorView* bias = (has_implicit_broadcast)
      ? makeConcreteTensor(bias_shape)
      : makeSymbolicTensor(bias_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto tv1 =
      (reshape_before_reduction) ? add(x, bias) : sum(x, {kReductionAxis});
  auto x_reshape = reshape(tv1, reshape_shape, output_shape);
  auto y = (reshape_before_reduction) ? sum(x_reshape, {kReductionAxis})
                                      : add(x_reshape, bias);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(bias_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({at_x, at_bias});

  testValidate(&fusion, outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

typedef std::vector<int64_t> shape_t;
using ReshapeExample = std::pair<shape_t, shape_t>;
// TODO: View examples with just 333 elements are failing validation in
// normalization. This might just be because our tolerances aren't tuned well
// for small sizes and the parallelization could be limited which could be
// detected as a validation issue, though it might not actually be a correctness
// issue. Using 3333 instead of 333 in those cases but should validate what's
// going on in the 333 case.
std::vector<ReshapeExample> all_reshape_examples = {
    {{1, 19, 1, 3 * 4, 7, 1, 99}, {1, 19, -1, 3, 4 * 7 * 99}},
    {{1, 19, 1, 3 * 4, 7, 1, 99}, {1, 19, 1, 3, 4 * 7 * 99}},
    {{19, 3 * 4, 7, 99}, {19, 3, 4 * 7 * 99}},

    {{3, 17, 2 * 4 * 10, 1}, {3 * 17, 1, 2, 4, -1}},
    {{3, 17, 2 * 4 * 10, 1}, {3 * 17, 1, 2, 4, 10}},
    {{3, 17, 2 * 4 * 10, 1}, {3 * 17, 2, 4, 1, 10}},

    {{3, 17, 2 * 4 * 10, 1, 9}, {-1, 1, 2, 4, 10, 9}},
    {{3, 17, 2 * 4 * 10, 1, 9}, {3 * 17, 1, 2, 4, 10, 9}},
    {{3, 17, 2 * 4 * 10, 1, 9}, {3 * 17, 2, 4, 1, 10, 9}},

    {{2, 3, 2 * 2, 5}, {1, 2 * 3, 1, -1, 2, 5, 1}},

    {{22, 11 * 2, 2}, {22, -1, 1, 1, 2 * 2}},
    {{22, 1, 22, 1}, {-1}},
    {{22, 11 * 2, 2}, {22, 11, 1, 1, 2 * 2}},
    {{22, 1, 22, 1}, {22 * 22}},

    {{37, 9, 7, 3 * 2, 5 * 2}, {37 * 9, 2, -1, 3, 7 * 5}},
    {{37, 9, 7, 3 * 2, 5 * 2}, {37 * 9, 2, 2, 3, 7 * 5}},

    {{1, 1, 3333, 1}, {1, 1, -1, 1}},
    // Disabled for now due to non-deterministic nan issue (#1920)
    // {{1, 1111 * 3}, {1, 1, 1, -1, 1, 3}},
    {{1, 3333, 1}, {-1}},
    {{1, 1, 3333, 1}, {1, 1, 3333, 1}},
    {{1, 303 * 11, 1}, {1, 303, -1, 1}},
    {{1, 3333, 1}, {1, 303, 11, 1}},
    // Disabled for now due to non-deterministic nan issue (#1920)
    // {{1, 3333}, {1, 1, 1, 1111, 1, 3}},
    {{1, 3333, 1}, {3333}},

    {{1, 3922 * 7, 1, 2}, {1, 3922 * 2, 1, -1}},
    {{1, 3922 * 2, 1, 7}, {1, -1, 2}},
    {{1, 3922 * 7, 2}, {1, 3922 * 2, 7}},
    {{1, 3922 * 2, 1, 7}, {1, 3922 * 7, 2}},
    {{1, 3922 * 7, 1, 2}, {1, 3922 * 2, 1, 7}},

    {{8, 1, 1, 2 * 4, 1, 8}, {8, 2, 4, 1, -1}},
    {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1, 8}},

    {{2, 3, 2 * 2, 5}, {1, 6, 1, 2, 2, 5, 1}},

    // Empty tensor reshapes
    {{2, 3, 0, 5}, {6, 0, 5}},
    {{2, 3, 0, 5}, {0, 7, 0}},
    {{2, 3, 0, 5}, {0, -1, 0}},
};

std::vector<ReshapeExample> reshape_after_reduce_examples = {
    {{19, 12, 7, 99}, {19, 3, 28}},
    {{1, 19, 1, 12, 7, 1, 99}, {1, 19, 1, 3, 28}},
    {{3, 17, 80, 1}, {51, 1, 2, 4, 10}},
    {{3, 17, 80, 1, 9}, {51, 1, 2, 4, 10}},
    {{2, 3, 4, 5}, {1, 6, 1, 2, 2, 1}},
    {{22, 22, 2}, {22, 11, 1, 1, 2}},
    {{37, 9, 7, 6, 10}, {333, 2, 21}},
    {{1, 1, 333, 1}, {1, 1, 333, 1}},
    {{8, 1, 1, 8, 1, 8}, {8, 2, 4, 1}},
    {{1, 333, 1}, {1, 37, 9, 1}},
    {{22, 1, 22, 1}, {484}},
    {{1, 333, 1}, {333}},
    {{1, 27454, 1, 2}, {1, 3922, 1, 7}},
    {{1, 7844, 1, 7}, {1, 1961, 4}}};

namespace ReshapeReduction {

struct ReshapeReductionParam {
  ReshapeExample reshape_example;
  bool has_implicit_broadcast;
  bool reshape_before_reduction;
};

std::vector<ReshapeReductionParam> generateReshapeReductionParams() {
  // For each reshape, test with and without implicit broadcast
  int total_tests =
      2 * (all_reshape_examples.size() + reshape_after_reduce_examples.size());
  std::vector<ReshapeReductionParam> params;
  params.reserve(total_tests);
  for (auto reshape_before_reduction : {true, false}) {
    const auto& examples = reshape_before_reduction
        ? all_reshape_examples
        : reshape_after_reduce_examples;
    for (auto has_implicit_broadcast : {false, true}) {
      for (const auto& re : examples) {
        params.push_back(
            {re, has_implicit_broadcast, reshape_before_reduction});
      }
    }
  }
  return params;
}

class ReshapeReduction : public NVFuserFixtureParamTest<ReshapeReductionParam> {
 protected:
  void SetUp() override {
    NVFuserFixtureParamTest<ReshapeReductionParam>::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_P(ReshapeReduction, FusionReshapeReduction) {
  const auto& param = GetParam();
  const auto& [input_shape, output_shape] = param.reshape_example;
  maybeClearAllocator(); // Shmoo tests can occupy a lot of memory
  reductionViewAddFusion(
      input_shape,
      output_shape,
      param.has_implicit_broadcast,
      param.reshape_before_reduction);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ReshapeReduction,
    ::testing::ValuesIn(generateReshapeReductionParams()));

} // namespace ReshapeReduction

void persistentViewAddFusion(
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape,
    bool reshape_before_persistent) {
  constexpr int kAxis = -1;

  // Support -1 sizes in the inputs
  auto inferred_shapes = inferViewShapes(input_shape, output_shape);
  auto inferred_input = inferred_shapes.first;
  auto inferred_output = inferred_shapes.second;

  auto bias_shape =
      reshape_before_persistent ? inferred_input : inferred_output;
  for (auto has_implicit_broadcast : {false, true}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    TensorView* x = (has_implicit_broadcast)
        ? makeConcreteTensor(inferred_input)
        : makeSymbolicTensor(inferred_input.size());
    TensorView* bias = (has_implicit_broadcast)
        ? makeConcreteTensor(bias_shape)
        : makeSymbolicTensor(bias_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto tv1 = (reshape_before_persistent) ? add(x, bias) : softmax(x, kAxis);
    auto x_reshape = reshape(tv1, inferred_input, inferred_output);
    auto y = (reshape_before_persistent) ? softmax(x_reshape, kAxis)
                                         : add(x_reshape, bias);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(inferred_input, options);
    at::Tensor at_bias = at::randn(bias_shape, options);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto outputs = executor_cache.runFusionWithInputs({at_x, at_bias});

    testValidate(&fusion, outputs, {at_x, at_bias}, __LINE__, __FILE__);
  }
}

TEST_F(ReshapeTest, ReshapePersistentShmoo) {
  for (auto e : all_reshape_examples) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();
    persistentViewAddFusion(
        e.first, e.second, true /* reshape_before_persistent */);
  }

  for (auto e : all_reshape_examples) {
    maybeClearAllocator(); // see above
    persistentViewAddFusion(
        e.first, e.second, false /* reshape_before_persistent */);
  }
}

void addViewGeluFusion(
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape) {
  for (auto has_implicit_broadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    TensorView* bias = (has_implicit_broadcast)
        ? makeConcreteTensor(input_shape)
        : makeSymbolicTensor(input_shape.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_add_bias = add(x, bias);
    auto x_reshape = reshape(x_add_bias, input_shape, output_shape);
    auto y = gelu(x_reshape);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape, options);
    at::Tensor at_bias = at::randn(input_shape, options);

    auto cg_outputs =
        scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias})
            .outputs;
    testValidate(&fusion, cg_outputs, {at_x, at_bias}, __LINE__, __FILE__);
  }
}

TEST_F(ReshapeTest, ReshapeSplit) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{2, 4, 10};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(ReshapeTest, ReshapeBroadcast) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{1, 80};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(ReshapeTest, ReshapeMerge) {
  std::vector<int64_t> input_shape{2, 40, 7};
  std::vector<int64_t> output_shape{560};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(ReshapeTest, ReshapeAllShmoo) {
  for (auto e : all_reshape_examples) {
    // Shmoo tests can occupy a lot of memory due to allocating many
    // different tensor sizes. So in order to avoid an OOM during this
    // test, we manually clear the allocator after it's reached a certain
    // threshold.
    maybeClearAllocator();
    addViewGeluFusion(e.first, e.second);
  }
}

void geluViewAddFusion(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> output_shape) {
  // Support -1 sizes in the inputs
  auto inferred_shapes = inferViewShapes(input_shape, output_shape);
  auto inferred_input = inferred_shapes.first;
  auto inferred_output = inferred_shapes.second;

  for (auto hasImplicitBroadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (hasImplicitBroadcast)
        ? makeConcreteTensor(inferred_input)
        : makeSymbolicTensor(inferred_input.size());
    TensorView* bias = (hasImplicitBroadcast)
        ? makeConcreteTensor(inferred_output)
        : makeSymbolicTensor(inferred_output.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_gelu = gelu(x);
    auto x_reshape = reshape(x_gelu, inferred_input, inferred_output);
    auto y = add(x_reshape, bias);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(inferred_input, options);
    at::Tensor at_bias = at::randn(inferred_output, options);

    auto cg_outputs =
        scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias})
            .outputs;
    testValidate(&fusion, cg_outputs, {at_x, at_bias}, __LINE__, __FILE__);
  }
}

TEST_F(ReshapeTest, ReshapeStride) {
  for (const auto& e : all_reshape_examples) {
    geluViewAddFusion(e.first, e.second);
  }
}

void geluViewBinaryAddFusion(
    std::vector<int64_t> input_shape1,
    std::vector<int64_t> input_shape2,
    std::vector<int64_t> output_shape) {
  for (auto hasImplicitBroadcast : {false, true}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* x = (hasImplicitBroadcast)
        ? makeConcreteTensor(input_shape1)
        : makeSymbolicTensor(input_shape1.size());
    TensorView* bias = (hasImplicitBroadcast)
        ? makeConcreteTensor(input_shape2)
        : makeSymbolicTensor(input_shape2.size());
    fusion.addInput(x);
    fusion.addInput(bias);

    auto x_gelu = gelu(x);
    auto x_reshape = reshape(x_gelu, input_shape1, output_shape);
    auto bias_reshape = reshape(bias, input_shape2, output_shape);
    auto y = add(x_reshape, bias_reshape);
    fusion.addOutput(y);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::randn(input_shape1, options);
    at::Tensor at_bias = at::randn(input_shape2, options);

    auto cg_outputs =
        scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias})
            .outputs;
    testValidate(&fusion, cg_outputs, {at_x, at_bias}, __LINE__, __FILE__);
  }
}

TEST_F(ReshapeTest, ReshapeBinary) {
  geluViewBinaryAddFusion({27454, 2}, {54908}, {7844, 7});
}

// Repro of issue #1493
TEST_F(ReshapeTest, ReshapeConcreteDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = reshape(tv0, {2, 3}, {6});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  auto tv4 = broadcast(tv3, {true, false});
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(0);
  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3}, options);
  auto t1 = at::randn({1, 6}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeConcreteDomain2) {
  constexpr int kAxis = -1;
  std::vector<int64_t> input_shape = {19, 12, 7, 99};
  std::vector<int64_t> output_shape = {19, 3, 2772};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(output_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto tv1 = softmax(x, kAxis);
  auto x_reshape = reshape(tv1, input_shape, output_shape);
  auto y = add(x_reshape, bias);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(output_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({at_x, at_bias});

  testValidate(&fusion, outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

// Repro of issue #1608
TEST_F(ReshapeTest, ReshapeConcreteDomain3) {
  std::vector<int64_t> input_shape = {14, 12, 8, 100};
  std::vector<int64_t> bcast_shape = {14, 12, 8, 1};
  std::vector<int64_t> other_shape = {14, 100, 96};
  std::vector<int64_t> output_shape = {14, 3, 3200};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* y = makeConcreteTensor(bcast_shape);
  TensorView* z = makeSymbolicTensor(other_shape.size());
  fusion.addInput(x);
  fusion.addInput(y);
  fusion.addInput(z);

  auto tv1 = add(x, y);
  auto tv2 = reshape(tv1, input_shape, output_shape);
  auto tv3 = reshape(z, other_shape, output_shape);
  auto output = add(tv2, tv3);
  fusion.addOutput(output);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(bcast_shape, options);
  at::Tensor at_z = at::randn(other_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({at_x, at_y, at_z});

  testValidate(&fusion, outputs, {at_x, at_y, at_z}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeConcreteDomain4) {
  std::vector<int64_t> shape1 = {3, 4, 5};
  std::vector<int64_t> shape2 = {3 * 4 * 5};

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(shape1.size() - 1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(shape1.size());
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false, false});
  auto tv3 = add(tv1, tv2);
  auto tv4 = reshape(tv3, shape1, shape2);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  NVF_CHECK(tv5->nDims() == 1);

  // The concrete domain of tv5, which is 1D, with permissive or loop mapping
  // needs to be either the domain of tv4 or tv5, both of which have the three
  // concrete root domains of tv1. In other words, it must map with tv4 and tv5
  // with the exact mapping.
  ComputeAtMap map(&fusion);
  auto concrete_id =
      map.getConcreteMappedID(tv5->axis(0), IdMappingMode::PERMISSIVE);
  NVF_CHECK(
      map.areMapped(concrete_id, tv5->axis(0), IdMappingMode::EXACT),
      "Invalid concrete ID: ",
      concrete_id->toString());
  NVF_CHECK(
      map.areMapped(concrete_id, tv4->axis(0), IdMappingMode::EXACT),
      "Invalid concrete ID: ",
      concrete_id->toString());
}

TEST_F(ReshapeTest, ReshapeConcreteDomain5) {
  const std::vector<int64_t> shape1 = {12};
  const std::vector<int64_t> shape2 = {4, 3};
  const std::vector<int64_t> shape3 = {12, 5};
  const std::vector<int64_t> shape4 = {4, 3, 5};

  for (auto order : {true, false}) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(1);
    fusion.addInput(tv0);

    auto tv1 = makeSymbolicTensor(2);
    fusion.addInput(tv1);

    auto tv0_cache = set(tv0);

    auto path1 = [&]() {
      auto reshape_2d = reshape(tv0_cache, shape1, shape2);
      auto reshape_2d_copy = set(reshape_2d);
      fusion.addOutput(reshape_2d_copy);
      return reshape_2d_copy;
    };

    auto path2 = [&]() {
      auto tv0_bc = broadcast(tv0_cache, {false, true});
      auto tv0_bc_plus_tv1 = add(tv0_bc, tv1);
      auto reshape_3d = reshape(tv0_bc_plus_tv1, shape3, shape4);
      auto reshape_3d_copy = set(reshape_3d);
      fusion.addOutput(reshape_3d_copy);
      return reshape_3d_copy;
    };

    TensorView* path1_out = nullptr;
    TensorView* path2_out = nullptr;

    if (order) {
      // Fails before #1544. Concrete ID is picked from path1_out, which
      // doesn't have the second root domain of tv1
      path2_out = path2();
      path1_out = path1();
    } else {
      // Works fine
      path1_out = path1();
      path2_out = path2();
    }

    path2_out->merge(-2, -1);
    path2_out->merge(-2, -1);

    tv0->computeAt(path2_out, -1);
    tv1->computeAt(path2_out, -1);

    NVF_CHECK(path1_out->nDims() == 1);
    NVF_CHECK(path2_out->nDims() == 1);

    ComputeAtMap map(&fusion);

    // Make sure the two output tensors are mapped. Note both are 1D.
    NVF_CHECK(map.areMapped(
        path1_out->axis(0), path2_out->axis(0), IdMappingMode::LOOP));

    auto concrete_id =
        map.getConcreteMappedID(path2_out->axis(0), IdMappingMode::LOOP);
    NVF_CHECK(
        path2_out->axis(0) == concrete_id,
        "Incorrect concrete ID: ",
        concrete_id->toString());
  }
}

TEST_F(ReshapeTest, FlattenAfterUnsqueezeOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{512};

  TensorView* x = makeSymbolicTensor(input_shape.size(), DataType::Double);
  TensorView* bias = makeSymbolicTensor(input_shape.size(), DataType::Double);
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_unsqueeze = unsqueeze(x_add_bias, -1);
  auto x_reshape = flatten(x_unsqueeze);
  fusion.addOutput(x_reshape);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);

  x_reshape->split(0, 4);
  x_add_bias->computeAt(x_reshape, 1);
  x_reshape->axis(0)->parallelize(ParallelType::TIDx);

  KernelExecutor ke;
  ke.compile(&fusion, {at_x, at_bias});
  auto outputs = ke.run({at_x, at_bias});

  testValidate(&fusion, outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ComputeAtLogicalDomainMapWithView) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> input_shape1{10, 12};
  const std::vector<int64_t> input_shape2{10, 3, 4};

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));

  // reduction followed by broadcast
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true, true});

  // Path with a reshape
  auto tv4 = reshape(tv1, input_shape1, input_shape2);

  // Join the reduciton+broadcast and reshape paths together
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  ComputeAtLogicalDomainMap map;
  map.build();

  // It's not possible to compute tv1 at the -1 position of
  // t2. ComputeAtLogicalDomainMap should tell that by not mapping the
  // second axis.
  auto tv1_tv2_mappable_dims =
      map.getMappableDims(tv1->domain(), tv2->domain());
  NVF_CHECK(
      tv1_tv2_mappable_dims.find(tv1->axis(1)) == tv1_tv2_mappable_dims.end(),
      "Invalid ComputeAtLogicalDomainMap. Domain should not be mappable: ",
      tv1->axis(1)->toString());
}

TEST_F(ReshapeTest, ExpandRepro) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> input_shape1{4, 1, 1};
  const std::vector<int64_t> input_shape2{4, 3, 2};

  auto tv0 = makeConcreteTensor({-1, 1, 1});
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(3);
  fusion.addInput(tv1);

  auto tv2 = expand_as(tv0, tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape1, options);
  at::Tensor at_y = at::randn(input_shape2, options);

  KernelExecutor ke;
  ke.compile(&fusion);
  LaunchParams l_params;
  auto outputs = ke.run({at_x, at_y}, {}, l_params, {});

  testValidate(&fusion, outputs, {at_x, at_y}, __LINE__, __FILE__);

  // second run to verify cached output allocation
  outputs = ke.run({at_x, at_y}, {}, l_params, {});
  testValidate(&fusion, outputs, {at_x, at_y}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ExpandView1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({4, 1, 8});
  fusion->addInput(tv0);

  auto tv1 = makeConcreteTensor({12, 8});
  fusion->addInput(tv1);

  auto tv2 = expand(
      tv0,
      {IrBuilder::create<Val>(4L),
       IrBuilder::create<Val>(3L),
       IrBuilder::create<Val>(8L)});

  auto tv3 = reshape(tv2, {4, 3, 8}, {12, 8});
  auto tv4 = add(tv3, tv1);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 1, 8}, options);
  auto t1 = at::randn({12, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ExpandView2) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 8});
  fusion->addInput(tv0);

  auto tv1 = makeConcreteTensor({3, 4, 8});
  fusion->addInput(tv1);

  auto tv2 =
      expand(tv0, {IrBuilder::create<Val>(12L), IrBuilder::create<Val>(8L)});

  auto tv3 = reshape(tv2, {12, 8}, {3, 4, 8});
  auto tv4 = add(tv3, tv1);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, 8}, options);
  auto t1 = at::randn({3, 4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeTransformCache) {
  auto assert_matches = [](ReshapeExample example_0, ReshapeExample example_1) {
    NVF_ERROR(
        analyzeViewConstraint(example_0.first, example_0.second) ==
            analyzeViewConstraint(example_1.first, example_1.second),
        "View: ",
        example_0.first,
        " -> ",
        example_0.second,
        "  Does not match:",
        example_1.first,
        " -> ",
        example_1.second);
  };

  auto assert_does_not_match = [](ReshapeExample example_0,
                                  ReshapeExample example_1) {
    NVF_ERROR(
        !(analyzeViewConstraint(example_0.first, example_0.second) ==
          analyzeViewConstraint(example_1.first, example_1.second)),
        "View: ",
        example_0.first,
        " -> ",
        example_0.second,
        "  Should not match:",
        example_1.first,
        " -> ",
        example_1.second);
  };

  // Splits are done as splitting out left hand side, so left hand side
  // split changes can't reuse reshape, but right hand side split changes can.
  // Merges, since they don't bury hard values in can always be reshared.
  // Need to make sure squeeze and broadcast changes don't try to reuse reshape.
  // What matches and what doesn't is very specific to the implementation of how
  // the splits/merges are generated. This could be changed over time as there
  // isn't a single set of transformations to potentially make a reshape. For
  // example we could always merge all dimensions, then split out all
  // dimensions. This would always be valid but would not be efficient for
  // indexing.

  // "Same"
  assert_matches(
      {{1, 1, 3333, 1}, {1, 1, 3333, 1}}, {{1, 1, 3333, 1}, {1, 1, -1, 1}});
  assert_matches(
      {{8, 1, 1, 2 * 4, 1, 8}, {8, 2, 4, 1, 8}},
      {{8, 1, 1, 2 * 4, 1, 8}, {8, 2, 4, 1, -1}});

  // Trivial reduce matching
  assert_matches({{1, 3333, 1}, {-1}}, {{1, 24, 1}, {-1}});

  // Trivial reduce not matching
  assert_does_not_match({{1, 3333, 1}, {-1}}, {{1, 3333}, {-1}});

  // Broadcast matching
  assert_matches({{3333}, {1, -1, 1}}, {{24}, {1, -1, 1}});

  // Broadcast not matching
  assert_does_not_match({{3333}, {1, -1, 1}}, {{24}, {1, -1}});

  // RHS split
  assert_matches(
      {{3, 17, 2 * 4 * 10, 1}, {3 * 17, 1, 2, 4, -1}},
      {{3, 17, 2 * 4 * 10 * 7, 1}, {3 * 17, 1, 2, 4, -1}});
  assert_matches(
      {{1, 303 * 11, 1}, {1, 303, -1, 1}},
      {{1, 303 * 11 * 4, 1}, {1, 303, -1, 1}});
  assert_matches(
      {{2, 3, 2 * 2 * 3, 5}, {1, 2 * 3, 1, 2, -1, 5, 1}},
      {{2, 3, 2 * 2 * 4, 5}, {1, 2 * 3, 1, 2, -1, 5, 1}});
  assert_matches(
      {{22, 11 * 2, 2}, {22, 11, 1, 1, -1}},
      {{22, 11 * 2 * 4, 2 * 3}, {22, 11, 1, 1, -1}});
  assert_matches(
      {{1, 1111 * 3}, {1, 1, 1, 1111, 1, -1}},
      {{1, 1111 * 3 * 7}, {1, 1, 1, 1111, 1, -1}});
  assert_matches(
      {{1, 303 * 11 * 2, 1}, {1, 303, -1, 1}},
      {{1, 303 * 11 * 3, 1}, {1, 303, -1, 1}});
  assert_matches(
      {{8, 1, 1, 2 * 4, 1, 8}, {8, 2, -1, 1, 8}},
      {{8, 1, 1, 2 * 4 * 6, 1, 8}, {8, 2, -1, 1, 8}});

  // LHS split not matching
  assert_does_not_match(
      {{3, 17, 2 * 4 * 10, 1}, {3 * 17, 1, 2, -1, 10}},
      {{3, 17, 2 * 4 * 3 * 10, 1}, {3 * 17, 1, 2, -1, 10}});
  assert_does_not_match(
      {{1, 303 * 11, 1}, {1, -1, 11, 1}},
      {{1, 303 * 11 * 2, 1}, {1, -1, 11, 1}});
  assert_does_not_match(
      {{2, 3, 2 * 2, 5}, {1, 2 * 3, 1, -1, 2, 5, 1}},
      {{2, 3, 3 * 2, 5}, {1, 2 * 3, 1, -1, 2, 5, 1}});
  assert_does_not_match(
      {{22, (11 + 1) * 2, 2}, {22, -1, 1, 1, 2 * 2}},
      {{22, 11 * 2, 2}, {22, -1, 1, 1, 2 * 2}});
  assert_does_not_match(
      {{1, 1111 * 3}, {1, 1, 1, -1, 1, 3}},
      {{1, 1111 * 2 * 3}, {1, 1, 1, -1, 1, 3}});
  assert_does_not_match(
      {{1, 303 * 11, 1}, {1, -1, 11, 1}},
      {{1, (303 + 1) * 11, 1}, {1, -1, 11, 1}});
  assert_does_not_match(
      {{8, 1, 1, 2 * 4, 1, 8}, {8, -1, 4, 1, 8}},
      {{8, 1, 1, 3 * 4, 1, 8}, {8, -1, 4, 1, 8}});

  // Merge matching
  assert_matches(
      {{3, 17, 2 * 4 * 10, 1, 9}, {-1, 1, 2, 4, 10, 9}},
      {{4, 18, 2 * 4 * 10, 1, 9}, {-1, 1, 2, 4, 10, 9}});
  assert_matches({{22, 1, 23, 1}, {-1, 1}}, {{23, 1, 22, 1}, {-1, 1}});

  // Merge not matching
  assert_does_not_match({{2, 3, 4}, {-1, 4}}, {{2, 3, 4}, {2, -1}});
  assert_does_not_match(
      {{22, 1, 23, 1, 24}, {-1, 24}}, {{22, 1, 23, 1, 24}, {22, -1}});

  // Split->Merge matching
  assert_matches(
      {{22, 11 * 2, 3}, {22, 11, 1, 1, -1}},
      {{22, 11 * 3, 2}, {22, 11, 1, 1, -1}});
  assert_matches(
      {{1, 3922 * 3 * 7, 1, 2 * 2}, {1, 3922 * 2, 1, -1}},
      {{1, 3922 * 7, 1, 2}, {1, 3922 * 2, 1, -1}});

  // Split->Merge not matching
  assert_does_not_match(
      {{22, 11 * 2, 2}, {22, -1, 1, 1, 4}},
      {{22, 11 * 2 * 3, 2}, {22, -1, 1, 1, 4}});
  assert_does_not_match(
      {{1, 3922 * 7, 1, 2}, {1, -1, 1, 7}},
      {{1, 3922 * 7 * 2, 1, 2}, {1, -1, 1, 7}});

  // Merge->Split matching
  assert_matches(
      {{1, 3922 * 2, 1, 7}, {1, 3922 * 7, -1}},
      {{1, 3922 * 2 * 3, 1, 7}, {1, 3922 * 7, -1}});
  assert_matches(
      {{19, 3 * 4, 7, 99}, {19, 3, -1}}, {{19, 3 * 3, 8, 10}, {19, 3, -1}});

  // Merge->Split not matching
  assert_does_not_match(
      {{1, 3922 * 2, 1, 7}, {1, -1, 2}}, {{1, 3922, 1, 7}, {1, -1, 2}});
  assert_does_not_match(
      {{19, 3 * 4, 7, 99}, {19, -1, 3}}, {{19, 3 * 5, 7, 99}, {19, -1, 3}});
}

TEST_F(ReshapeTest, ReshapeIdGraph) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 2, x = 3, y = 4, z = 5;

  auto tv0 = makeConcreteTensor({w, x, y, z});
  fusion.addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {w, x, y, z}, {w, y, x * z});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({w, x, y, z});
  fusion.addInput(tv3);

  auto tv4 = reshape(tv3, {w, x, y, z}, {w, y, x * z});
  fusion.addOutput(tv4);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv5 = add(tv0, tv3);
  fusion.addOutput(tv5);

  auto tv6 = makeConcreteTensor({w, x, x, y, z});

  auto tv7 = sum(tv6, {2});
  auto tv8 = broadcast(tv7, {false, true, false, true, false, false});

  auto tv9 = makeConcreteTensor({w, 6, x, 7, y, z});
  fusion.addInput(tv9);
  auto tv10 = add(tv8, tv9);
  fusion.addOutput(tv10);

  auto tv12 = reshape(tv8, {w, 1, x, 1, y, z}, {w, y, x * z});
  fusion.addOutput(tv12);

  // Link the reshapes after the reshapes happen
  auto t13 = add(tv12, tv4);
  fusion.addOutput(t13);

  // Start from the exact iter domain graph of the fusion
  IterDomainGraph id_graph(&fusion);
  auto disjoint_reshape_ids = id_graph.exactNodes();

  NVF_CHECK(id_graph.exactNodes().strictAreMapped(tv2->axis(1), tv4->axis(1)));
  NVF_CHECK(id_graph.exactNodes().strictAreMapped(tv2->axis(2), tv4->axis(2)));

  NVF_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[1], tv12->getRootDomain()[1]));
  NVF_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[2], tv12->getRootDomain()[2]));
  NVF_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[3], tv12->getRootDomain()[3]));
}

TEST_F(ReshapeTest, ReshapeVectorize) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = flatten(tv0, 1, 2);
  auto tv2 = flatten(tv0, 1, 2);
  auto tv3 = sin(tv1);
  auto tv4 = sin(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input = at::randn({256, 256, 256}, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {input}).outputs;
  testValidate(&fusion, cg_outputs, {input}, __LINE__, __FILE__);

  auto hasVectorization = [](TensorView* tv) -> bool {
    for (auto i : tv->getLoopDomain()) {
      if (i->getParallelType() == ParallelType::Vectorize) {
        return true;
      }
    }
    return false;
  };

  for (auto o : fusion.outputs()) {
    NVF_CHECK(hasVectorization(o->as<TensorView>()));
  }
  for (auto i : fusion.inputs()) {
    for (auto c : ir_utils::consumerTvsOf(i->as<TensorView>())) {
      NVF_CHECK(hasVectorization(c));
    }
  }
}

TEST_F(ReshapeTest, ExpandFlatten) {
#ifdef FBCODE_CAFFE2
  GTEST_SKIP() << "Fails accuracy on V100 32gb";
#endif
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({-1, -1, 1});
  fusion->addInput(tv0);
  auto tv1 = expand(
      tv0,
      {tv0->axis(0)->extent(),
       tv0->axis(1)->extent(),
       IrBuilder::create<Val>(8L)});
  auto tv2 = flatten(tv1, 1, 2);
  auto tv3 = sum(tv2, {1});
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({input});

  testValidate(
      executor_cache.fusion(), cg_outputs, {input}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, IllegalReductionFlatten) {
  EXPECT_THAT(
      []() {
        auto fusion = std::make_unique<Fusion>();
        FusionGuard fg(fusion.get());

        auto tv0 = makeConcreteTensor({2, 3});
        fusion->addInput(tv0);

        auto tv1 = sum(tv0, {1});
        auto tv2 = flatten(tv1, 0, 1);
        fusion->addOutput(tv2);
      },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Invalid end_dim")));
}

TEST_F(ReshapeTest, ReductionFlatten1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 3, 5});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = flatten(tv1, 0, 1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3, 5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, PwiseViewSchedule) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int x = 31, y = 65, z = 103;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv3);

  auto tv4 = reshape(tv3, {x, y, z}, {x, y * z});
  fusion.addOutput(tv4);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv5 = add(tv0, tv3);
  fusion.addOutput(tv5);

  {
    TransformPropagator propagator(tv4);
    MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);
  }

  for (auto i : arange(tv5->nDims() - 1)) {
    (void)i; // Suppress unused variable warning
    tv5->merge(0);
  }
  tv5->split(0, 32);
  tv5->split(0, 4);
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::Unroll);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  {
    TransformPropagator propagator(tv5);
    MaxLogicalDomainInfoSpanningTree spanning_tree(tv5);
    spanning_tree.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(tv5);

    // Inline the schedule
    inlineMost();
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto cg_outputs = ke.run({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, SumViewSchedule) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int x = 31, y = 65, z = 103;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv3);

  auto tv4 = reshape(tv3, {x, y, z}, {x, y * z});
  auto tv5 = sum(tv4, {1});
  fusion.addOutput(tv5);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv6 = add(tv0, tv3);
  fusion.addOutput(tv6);

  {
    TransformPropagator propagator(tv4);
    MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);
  }

  tv5->split(1, 128);
  tv5->split(1, 4);

  auto tv5_rf = tv5->rFactor({1, 2});
  tv5_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv5_rf->axis(2)->parallelize(ParallelType::Unroll);
  tv5_rf->axis(3)->parallelize(ParallelType::TIDx);

  {
    TransformPropagator propagator(tv5_rf);
    MaxLogicalDomainInfoSpanningTree spanning_tree(tv5_rf);
    spanning_tree.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(tv5_rf);

    // Inline the schedule
    inlineMost();
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {x, y * z});
  auto t4 = at::native::view(t3, {x, y * z});
  auto t5 = t4.sum({1});
  auto t6 = t0 + t3;

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto cg_outputs = ke.run({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, {t2, t5, t6}, __LINE__, __FILE__);
}

// Make sure matching reshapes are segmented into the same kernel
TEST_F(ReshapeTest, ReshapeMagicSchedule1) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int x = 31, y = 65, z = 103;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  // Without tv6, the complete fusion would be segmented to many pointwise
  // kernels.
  //
  // This is for two reasons:
  //
  // 1. Without tv6, tv2's definition, reshape, would have to be segmented out
  // by MarkAliasesPrepare. Therefore, the complete fusion can't be accepted as
  // one kernel.
  //
  // 2. Because of that, the complete fusion would be
  // decomposed into singletons, which the segmenter attempts to merge.
  // The segmenter can't yet horizontally merge `tv1`, `tv4` and `tv5`, leading
  // to many pointwise kernels.
  //
  // A similar trick is applied to several other FusionReshapeMagicSchedule
  // tests to work around this segmenter limitation.
  auto tv6 = add(tv2, tv2);
  fusion->addOutput(tv6);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv3);

  auto tv4 = reshape(tv3, {x, y, z}, {x, y * z});
  fusion->addOutput(tv4);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv5 = add(tv0, tv3);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});
  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

// Make sure reshapes of reshapes are correct
TEST_F(ReshapeTest, ReshapeMagicSchedule2) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int x = 31, y = 65, z = 103;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  auto tv3 = reshape(tv2, {x, y * z}, {x * y, z});
  auto tv4 = reshape(tv3, {x * y, z}, {y, x * z});
  auto tv5 = reshape(tv4, {y, x * z}, {x, y, z});
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);

  // For now pointwise scheduler only accepts a single reshape at a time, so
  // this will be broken up into multiple kernels. This is due to the reference
  // check looking for all mappings to all input IDs.
  // TODO: Fix the reference check for this case
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// Make sure broadcasts not on the reshape path that don't interfere with
// reshape are segmented in one kernel and correctly trigger 2D pointwise
// scheduling
TEST_F(ReshapeTest, ReshapeMagicSchedule3) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int w = 15, x = 31, y = 49, z = 65;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  auto tv9 = add(tv2, tv2);
  fusion->addOutput(tv9);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv3);

  auto tv4 = reshape(tv3, {x, y, z}, {x, y * z});
  fusion->addOutput(tv4);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv5 = add(tv0, tv3);
  fusion->addOutput(tv5);

  // Broadcast on another branch to drive the pointwise reference to not be on
  // the reshape paths.

  auto tv6 = makeConcreteTensor({w, x, y, z});
  fusion->addInput(tv6);
  auto tv7 = broadcast(tv0, {true, false, false, false});
  auto tv8 = add(tv6, tv7);
  // tv8 should be the reference for the pointwise fusion. This broadcast
  // pattern doesn't interfere with the reshapes, so this should also be
  // scheduled as 2D.
  fusion->addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  at::Tensor t6 = at::randn({w, x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t6});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<PointwiseParams>());
  auto pparams =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  NVF_CHECK(pparams->break_point == 1);

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t3, t6}, __LINE__, __FILE__);
}

// Make sure broadcasts through reshapes when not conflicting with reshape are
// segmented into one kernel and trigger 2D pointwise scheduler.
TEST_F(ReshapeTest, ReshapeMagicSchedule4) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int x = 31, y = 49, z = 65;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  auto tv8 = add(tv2, tv2);
  fusion->addOutput(tv8);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion->addInput(tv3);

  auto tv4 = makeConcreteTensor({x, 1, 1});
  fusion->addInput(tv4);

  auto tv5 = add(tv4, tv3);

  auto tv6 = reshape(tv5, {x, y, z}, {x, y * z});
  auto tv9 = add(tv6, tv6);
  fusion->addOutput(tv9);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv7 = add(tv0, tv3);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  at::Tensor t4 = at::randn({x, 1, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t4});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<PointwiseParams>());
  auto pparams =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  NVF_CHECK(pparams->break_point == 1);

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t3, t4}, __LINE__, __FILE__);
}

// Make sure different reshapes that are consumed by the reference are segmented
// into a single kernel.
TEST_F(ReshapeTest, ReshapeMagicSchedule5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int w = 15, x = 31, y = 49, z = 65;

  auto tv0 = makeConcreteTensor({w, x, y * z});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {w, x, y * z}, {z, y, x, w});

  auto tv3 = makeConcreteTensor({w, x * y, z});
  fusion.addInput(tv3);
  auto tv4 = cos(tv3);
  auto tv5 = reshape(tv4, {w, x * y, z}, {z, y, x, w});

  auto tv6 = add(tv2, tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, x, y * z}, options);
  at::Tensor t3 = at::randn({w, x * y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<PointwiseParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

// Test reshape/transpose and its impact on vectorization
TEST_F(ReshapeTest, ReshapeMagicSchedule6) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // pointwise heuristics will avoid vectorization if can't achieve a full wave.
  // use a large size to make sure we can achieve a full wave, e.g. x * y >= 128
  // * sm_count
  int x = 1024, y = 1024;

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = reshape(tv0, {x, y}, {x, y / 2, 2});
  auto tv2 = transpose(tv1, 0, 1);

  auto tv3 = makeContigTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options);
  at::Tensor t3 = at::randn({y / 2, x, 2}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<PointwiseParams>());
  NVF_CHECK(
      executor_cache.getMostRecentExecutorInfo()
          .params->as<PointwiseParams>()
          ->vectorization_factor > 1);

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

// View with 3D reduction scheduling
TEST_F(ReshapeTest, ReshapeMagicSchedule7) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int v = 3, w = 5, x = 42, y = 7, z = 9;

  auto tv0 = makeConcreteTensor({w, v, x, y, z});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {w, v, x, y, z}, {v * w, x, y * z});

  auto tv3 = makeConcreteTensor({v, w, x, z, y});
  fusion.addInput(tv3);
  auto tv4 = cos(tv3);
  auto tv5 = reshape(tv4, {v, w, x, z, y}, {v * w, x, y * z});

  auto tv6 = add(tv2, tv5);
  auto tv7 = sum(tv6, {0, 2});
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, v, x, y, z}, options);
  at::Tensor t3 = at::randn({v, w, x, z, y}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<ReductionParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

// View with 3D normalization scheduling
TEST_F(ReshapeTest, ReshapeMagicSchedule8) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int v = 3, w = 5, x = 42, y = 7, z = 9;

  auto tv0 = makeConcreteTensor({w, v, x, y, z});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {w, v, x, y, z}, {v * w, x, y * z});

  auto tv3 = makeConcreteTensor({v, w, x, z, y});
  fusion.addInput(tv3);
  auto tv4 = cos(tv3);
  auto tv5 = reshape(tv4, {v, w, x, z, y}, {v * w, x, y * z});

  auto tv6 = add(tv2, tv5);
  auto tv7 = sum(tv6, {0, 2});
  auto tv8 = broadcast(tv7, {true, false, true});
  auto tv9 = add(tv6, tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, v, x, y, z}, options);
  // This might trigger transpose kernel.
  at::Tensor t3 = at::randn({v, w, x, z, y}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  NVF_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  NVF_CHECK(executor_cache.getMostRecentExecutorInfo()
                .params->isA<ReductionParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

// AlbertForMaskedLM repro https://github.com/csarofeen/pytorch/issues/2066
TEST_F(ReshapeTest, ReshapeMagicSchedule9) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 2, y = 512, z = 128;

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  auto tv2 = makeContigTensor(1);
  auto tv3 = makeContigTensor(2);
  auto tv4 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addInput(tv4);

  auto tv5 = broadcast(tv0, {true, true, false});
  auto tv6 = broadcast(tv1, {false, false, true});
  auto tv7 = broadcast(tv2, {true, true, false});
  auto tv8 = broadcast(tv3, {false, false, true});
  auto tv9 = set(tv6);

  auto s10 = IrBuilder::create<Val>(1e-12);
  auto tv11 = add(abs(tv8), s10);

  auto tv12 = sub(tv4, tv9);
  auto tv13 = rsqrt(tv11);
  auto tv14 = broadcast(tv13, {false, false, false});
  auto tv15 = mul(tv12, tv14);
  auto tv16 = mul(tv15, tv5);
  auto tv17 = add(tv16, tv7);
  auto tv18 = castOp(DataType::Float, tv17);
  auto tv19 = reshape(tv18, {x, y, z}, {x * y, z});
  fusion.addOutput(tv6);
  fusion.addOutput(tv13);
  fusion.addOutput(tv19);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);
  auto t1 = at::randn({2, 512}, options);
  auto t2 = at::randn({128}, options);
  auto t3 = at::randn({2, 512}, options);
  auto t4 = at::randn({2, 512, 128}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});

  testValidate(&fusion, cg_outputs, {t0, t1, t2, t3, t4}, __LINE__, __FILE__);
}

// Simpler version of FusionReshapeMagicSchedule9_CUDA
TEST_F(ReshapeTest, ReshapeMagicSchedule10) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 2, y = 512, z = 128;

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  auto tv2 = makeContigTensor(3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv0, {true, true, false});
  auto tv4 = broadcast(tv1, {false, false, true});

  auto tv5 = add(tv2, tv4);
  auto tv6 = add(tv5, tv3);
  auto tv7 = reshape(tv6, {x, y, z}, {x * y, z});
  fusion.addOutput(tv4);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);
  auto t1 = at::randn({2, 512}, options);
  auto t2 = at::randn({2, 512, 128}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
}

// CamemBert repro
TEST_F(ReshapeTest, ReshapeMagicSchedule11) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 512, y = 12, z = 64;

  auto tv0 = makeContigConcreteTensor({1, -1, -1, -1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {1, x, y, z}, {1, x, y * z});
  auto tv3 = reshape(tv2, {1, x, y * z}, {x, y * z});
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Make sure different reshapes that are consumed by the reference are segmented
// into a single kernel.
TEST_F(ReshapeTest, ReshapeMapping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int w = 15, x = 31, y = 49, z = 65;

  auto tv0 = makeConcreteTensor({w, x, y * z});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {w, x, y * z}, {z, y, x, w});

  auto tv3 = makeConcreteTensor({w, x * y, z});
  fusion.addInput(tv3);
  auto tv4 = cos(tv3);
  auto tv5 = reshape(tv4, {w, x * y, z}, {z, y, x, w});

  auto tv6 = add(tv2, tv5);
  fusion.addOutput(tv6);

  tv6->merge(0);
  tv6->merge(0);
  tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);
  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::Unroll);
  tv6->axis(2)->parallelize(ParallelType::TIDx);

  TransformPropagator propagator(tv6);
  MaxLogicalDomainInfoSpanningTree spanning_tree(tv6);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv6);

  // Inline the schedule
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, x, y * z}, options);
  at::Tensor t3 = at::randn({w, x * y, z}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto cg_outputs = ke.run({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, LowerDivisibleSplits) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int w = 15, x = 31, y = 49, z = 65;

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {w, x, y, z}, {z, y, x, w});

  fusion.addOutput(tv2);

  tv2->merge(0)->merge(0)->merge(0)->split(0, 4)->split(0, 8, false);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree spanning_tree(tv2);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv2);

  // Inline the schedule
  inlineMost();

  auto divisible_splits = getAllDivisibleSplits(&fusion);

  // Operations on all tensors are basically:
  // [10] merge(0)          [9]->outer->definition
  // [9] merge(0)           [8]->outer->definition
  // [8] merge(0)           [7]->in->definition
  // [7] split(0, z, false) [6]->in->definition
  // [6] split(1, y, false) [5]->in->definition
  // [5] split(2, x, false) [3]->inner->definition
  // RFactor of tv2
  // [4] merge(0)           [3]->outer->definition
  // [3] merge(0)           [2]->outer->definition
  // [2] merge(0)           [1]->in->definition
  // [1] split(0, 4)        [0]->in->definition
  // [0] split(0, 8, false) tv->axis(0)->definition

  for (auto tv : std::vector<TensorView*>({tv2, tv1, tv0})) {
    auto transform_0 = tv->axis(0)->definition()->as<Split>();
    auto transform_1 = transform_0->in()->definition()->as<Split>();
    auto transform_2 = transform_1->in()->definition()->as<Merge>();
    auto transform_3 = transform_2->outer()->definition()->as<Merge>();

    auto transform_5 = transform_3->inner()->definition()->as<Split>();
    auto transform_6 = transform_5->in()->definition()->as<Split>();
    auto transform_7 = transform_6->in()->definition()->as<Split>();

    NVF_CHECK(
        divisible_splits.find(transform_5) != divisible_splits.end(),
        "Expecting: ",
        transform_5->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
    NVF_CHECK(
        divisible_splits.find(transform_6) != divisible_splits.end(),
        "Expecting: ",
        transform_6->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
    NVF_CHECK(
        divisible_splits.find(transform_7) != divisible_splits.end(),
        "Expecting: ",
        transform_7->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
  }
}

TEST_F(ReshapeTest, Issue2076) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // torch.randn(4, 128, 1, 128, device='cuda').transpose(1,2),
  // sizes[4, 128, 1, 128] strides[128*128, 128, 128, 1]
  // sizes[4, 1, 128, 128] strides[128*128, 128, 128, 1]
  auto tv0 = TensorViewBuilder()
                 .shape({-1, 1, -1, -1})
                 .dtype(DataType::Bool)
                 .contiguity({true, std::nullopt, true, true})
                 .build();
  fusion.addInput(tv0);

  // torch.randn(48, 128, 128, device='cuda'),
  auto tv1 = makeContigTensor(3);
  fusion.addInput(tv1);

  // torch.randn(4, 1, 128, 128, device='cuda'),
  auto tv2 = makeContigConcreteTensor({-1, 1, -1, -1});
  fusion.addInput(tv2);

  auto tv3 = castOp(DataType::Float, tv0);
  auto tv4 = reshape(tv1, {48, 128, 128}, {4, 12, 128, 128});

  auto tv5 = mul(tv3, IrBuilder::create<Val>(1.0));
  auto tv6 = sub(IrBuilder::create<Val>(1.0), tv5);
  auto tv7 = castOp(DataType::Bool, tv6);
  auto tv8 = where(tv7, IrBuilder::create<Val>(-3.4028200000000001e+38), tv6);
  auto tv9 = add(tv8, tv2);
  auto tv10 = set(tv9);
  auto tv11 = expand(
      tv10,
      {tv10->axis(0)->extent(),
       IrBuilder::create<Val>(12L),
       tv10->axis(2)->extent(),
       tv10->axis(3)->extent()});

  auto tv12 = add(tv4, tv11);
  auto tv13 = reshape(tv12, {4, 12, 128, 128}, {48, 128, 128});
  auto tv14 = max(tv13, {2});
  auto tv15 = broadcast(tv14, {false, false, true});
  auto tv16 = set(tv15);
  auto tv17 = expand(
      tv16,
      {tv16->axis(0)->extent(),
       tv16->axis(1)->extent(),
       IrBuilder::create<Val>(128L)});
  auto tv18 = sub(tv13, tv17);
  auto tv19 = exp(tv18);
  auto tv20 = sum(tv19, {2});
  auto tv21 = broadcast(tv20, {false, false, true});
  auto tv22 = set(tv21);
  auto tv23 = expand(
      tv22,
      {tv22->axis(0)->extent(),
       tv22->axis(1)->extent(),
       IrBuilder::create<Val>(128L)});
  auto tv24 = div(tv19, tv23);

  fusion.addOutput(tv9);
  fusion.addOutput(tv24);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 =
      at::randn({4, 128, 1, 128}, options).transpose(1, 2).to(at::kBool);
  at::Tensor t1 = at::randn({48, 128, 128}, options);
  at::Tensor t2 = at::randn({4, 1, 128, 128}, options);

  auto t3 = t0.to(at::kFloat);
  auto t4 = t1.reshape({4, 12, 128, 128});

  // [4, 1, 128, 128]
  auto t5 = t3 * 1;
  auto t6 = 1 - t5;
  auto t7 = t6.to(at::kBool);
  auto t8 = at::where(t7, -3.4028200000000001e+38, t6);
  auto t9 = t8 + t2;
  auto t10 = t9;
  // [4, 1, 128, 128]
  auto t11 = t10.expand({4, 12, 128, 128});

  auto t12 = t4 + t11;
  auto t13 = t12.reshape({48, 128, 128});
  // 48, 128, 128
  auto t14 = std::get<0>(t13.max(2));
  auto t15 = t14.unsqueeze(-1);
  // 48, 128, 1
  auto t16 = t15;
  auto t17 = t16.expand({48, 128, 128});
  auto t18 = t13 - t17;
  auto t19 = t18.exp();
  auto t20 = t19.sum({2});
  auto t21 = t20.unsqueeze(-1);
  auto t22 = t21;
  auto t23 = t22.expand({48, 128, 128});
  auto t24 = t19 / t23;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      &fusion, cg_outputs, {t0, t1, t2}, {t9, t24}, __LINE__, __FILE__);
}

// Simplify first to reproduce compute at issue
TEST_F(ReshapeTest, Issue2076_v2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // torch.randn(4, 128, 1, device='cuda').transpose(1,2)
  // sizes[4, 128, 1] strides[128, 1, 1]
  // sizes[4, 1, 128] strides[128, 128, 1]
  auto tv0 = TensorViewBuilder()
                 .shape({-1, 1, -1})
                 .contiguity({true, std::nullopt, true})
                 .build();
  fusion.addInput(tv0);

  // torch.randn(48, 128, device='cuda')
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  // torch.randn(4, 1, 128, device='cuda'),
  auto tv2 = makeContigConcreteTensor({-1, 1, -1});
  fusion.addInput(tv2);

  auto tv3 = reshape(tv1, {48, 128}, {4, 12, 128});
  auto tv4 = add(tv0, tv2);

  auto tv5 = add(tv3, tv4);
  auto tv6 = reshape(tv5, {4, 12, 128}, {48, 128});

  fusion.addOutput(tv4);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({4, 128, 1}, options).transpose(1, 2);
  at::Tensor t1 = at::randn({48, 128}, options);
  at::Tensor t2 = at::randn({4, 1, 128}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeZeroDimInput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor({}, DataType::Float);
  fusion.addInput(x);
  TensorView* y = makeSymbolicTensor(3, DataType::Float);
  fusion.addInput(y);

  auto x_rsh = reshape(x, {}, {1, 1, 1});

  auto prod = mul(x_rsh, y);
  fusion.addOutput(prod);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x =
      at::randn({1}).to(options)[0]; // indexing to get zero-dim tensor
  NVF_ERROR(at_x.ndimension() == 0);

  at::Tensor at_y = at::randn({2, 3, 4}).to(options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_y}).outputs;
  testValidate(&fusion, cg_outputs, {at_x, at_y}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeZeroDimOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Make a concrete tensor so that all dims are set as broadcast
  TensorView* x = makeContigConcreteTensor({1, 1, 1}, DataType::Float);
  TensorView* y = makeContigConcreteTensor({1, 1}, DataType::Float);
  TensorView* z = makeSymbolicTensor(0, DataType::Float);
  fusion.addInput(x);
  fusion.addInput(y);
  fusion.addInput(z);

  auto x_rsh = reshape(x, {1, 1, 1}, {});
  // test mixed broadcast and concrete 1-dimensional iter-domains
  auto y_mixed = reshape(y, {1, 1}, {1, 1, 1});
  auto y_rsh = reshape(y_mixed, {1, 1, 1}, {});

  auto prod = mul(add(x_rsh, y_rsh), z);
  fusion.addOutput(prod);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x = at::randn({1, 1, 1}).to(options);
  at::Tensor at_y = at::randn({1, 1}).to(options);
  at::Tensor at_z =
      at::randn({1}).to(options)[0]; // indexing to get zero-dim tensor
  NVF_ERROR(at_z.ndimension() == 0);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_y, at_z})
          .outputs;
  testValidate(&fusion, cg_outputs, {at_x, at_y, at_z}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeZeroDimInputOutput) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeSymbolicTensor({}, DataType::Float);
  fusion.addInput(x);
  TensorView* y = makeSymbolicTensor({}, DataType::Float);
  fusion.addInput(y);

  auto x_rsh = reshape(x, {}, {});
  // test broadcasting then squeezing
  x_rsh = reshape(x, {}, {1, 1, 1});
  x_rsh = reshape(x_rsh, {1, 1, 1}, {});

  auto prod = mul(x_rsh, y);
  fusion.addOutput(prod);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x =
      at::randn({1}).to(options)[0]; // indexing to get zero-dim tensor
  at::Tensor at_y = at::randn({1}).to(options)[0];
  NVF_ERROR(at_x.ndimension() == 0 && at_y.ndimension() == 0);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_y}).outputs;
  testValidate(&fusion, cg_outputs, {at_x, at_y}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, ReshapeOfReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {4, 8}, {8, 4});
  auto tv2 = reshape(tv1, {8, 4}, {32});
  fusion->addOutput(tv2);

  std::vector<int64_t> shape({4, 8});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto ref = t0.reshape({8, 4}).reshape({32});

  NVF_CHECK(ref.equal(cg_outputs[0].as<at::Tensor>()));
}

// A reproducer for #1116.
TEST_F(ReshapeTest, ExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({4, 5});
  fusion.addInput(in);
  TensorView* out = broadcast(in, {false, false, true});
  out = expand(
      out,
      {IrBuilder::create<Val>(4),
       IrBuilder::create<Val>(5),
       IrBuilder::create<Val>(6)});
  // tryStaticReshape failed to get the expanded extent, which is 6.
  out = reshape(out, {IrBuilder::create<Val>(40), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  at::Tensor in_tensor =
      at::randn({4, 5}, at::dtype(at::kFloat).device(at::kCUDA, 0));

  KernelExecutor ke;
  ke.compile(&fusion, {in_tensor});
  auto cg_outputs = ke.run({in_tensor});

  testValidate(&fusion, cg_outputs, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, SplitMergePointwiseSplitMerge) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const std::vector<int64_t> input_shape = {12, 20};
  DataType dtype = DataType::Float;
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  // root domain : (i0, i2)
  // logical domain : (3, i0/3, 4, i2/4)
  auto tv2 = reshape(tv1, {12, 20}, {3, 4, 4, 5});
  // root domain : (3, i0/3, 4, i2/4)
  // logical domain : (3, i0/3*4, i2/4)
  auto tv3 = reshape(tv2, {3, 4, 4, 5}, {3, 16, 5});
  // root domain : (3, i0/3*4, i2/4)
  auto tv4 = mul(tv3, tv3);
  // root domain : (i0, i2)
  // logical domain : (3, i0/3, 4, i2/4)
  auto tv5 = reshape(tv1, {12, 20}, {3, 4, 4, 5});
  // root domain : (3, i0/3, 4, i2/4)
  // logical domain : (3, i0/3*4, i2/4)
  auto tv6 = reshape(tv5, {3, 4, 4, 5}, {3, 16, 5});
  fusion->addOutput(tv4);
  fusion->addOutput(tv6);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), {cg_outputs}, {t0}, __LINE__, __FILE__);
}

// segmented into 2 kernels: pointwise and reduction
TEST_F(ReshapeTest, GroupNormOriginal) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int64_t N = 2, C = 128, H = 16, W = 16, G = 32;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const std::vector<int64_t> group_shape = {N, G, C / G, H, W};
  const std::vector<int64_t> input_shape_wb = {C};
  const std::vector<int64_t> group_shape_wb = {G, C / G};
  DataType dtype = DataType::Half;
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  auto tv1 = makeContigTensor(input_shape_wb.size(), DataType::Float);
  auto tv2 = makeContigTensor(input_shape_wb.size(), DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  // pointwise ops, e.g. cast
  auto tv3 = castOp(DataType::Float, tv0);
  // reshape from {N, C, H, W} to {N, G, C / G, H, W}
  auto tv4 = reshape(tv3, input_shape, group_shape);
  // normalization
  auto tv5 = sum(tv4, {-1, -2, -3});
  auto tv6 = broadcast(tv5, {false, false, true, true, true});
  auto tv7 = div(tv4, tv6);
  // reshape back to {N, C, H, W}
  auto tv8 = reshape(tv7, group_shape, input_shape);
  // pointwise ops, e.g. scale, bias, cast
  auto tv9 = broadcast(tv1, {true, false, true, true});
  auto tv10 = broadcast(tv2, {true, false, true, true});
  auto tv11 = mul(tv8, tv9);
  auto tv12 = add(tv11, tv10);
  auto tv13 = castOp(dtype, tv12);
  fusion->addOutput(tv13);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_wb = at::TensorOptions()
                        .dtype(data_type_to_aten(DataType::Float))
                        .device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto tw = at::randn(input_shape_wb, options_wb);
  auto tb = at::randn(input_shape_wb, options_wb);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, tw, tb});
  // should expect 1 after adding a pre-segment pass to move reshape to input
  // and output.
  EXPECT_THAT(
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::PointWise),
          HeuristicIs(SchedulerType::Reduction)));

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, tw, tb}, __LINE__, __FILE__);
}

using ReductionAxes = std::vector<int64_t>;
class ViewReductionTest : public NVFuserFixtureParamTest<ReductionAxes> {};

TEST_P(ViewReductionTest, ReductionReshapeInputNoMergedIds) {
  auto reduction_axes = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int64_t N = 2, C = 128, H = 16, W = 16, G = 32;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const std::vector<int64_t> group_shape = {N, G, C / G, H, W};
  DataType dtype = DataType::Half;
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = reshape(tv1, input_shape, group_shape);
  auto tv3 = sum(tv2, {reduction_axes});
  fusion->addOutput(tv3);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = t0.reshape(group_shape).to(at::kFloat);
  auto ref = t1.sum({reduction_axes});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  // should have only 1 segment group
  auto seg_groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();
  EXPECT_EQ(seg_groups.size(), 1);
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ViewReductionTest,
    ::testing::Values(
        std::vector<int64_t>{0},
        std::vector<int64_t>{0, 1},
        std::vector<int64_t>{0, 1, 2},
        std::vector<int64_t>{-1},
        std::vector<int64_t>{-1, -2},
        std::vector<int64_t>{-1, -2, -3},
        std::vector<int64_t>{0, 2, 4},
        std::vector<int64_t>{1, 3}));

class ViewNormalizationTest : public NVFuserFixtureParamTest<ReductionAxes> {};

TEST_P(ViewNormalizationTest, NormalizationReshapeInputNoMergedIds) {
  auto reduction_axes = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const int64_t N = 2, C = 128, H = 16, W = 16, G = 32;
  const std::vector<int64_t> input_shape = {N, C, H, W};
  const std::vector<int64_t> group_shape = {N, G, C / G, H, W};
  int ndims = (int)group_shape.size();
  std::vector<bool> broadcast_tags(ndims, false);
  for (auto axis : reduction_axes) {
    int idx = axis < 0 ? ndims + axis : axis;
    broadcast_tags[idx] = true;
  }
  DataType dtype = DataType::Half;
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = reshape(tv1, input_shape, group_shape);
  auto tv3 = sum(tv2, {reduction_axes});
  auto tv4 = broadcast(tv3, broadcast_tags);
  auto tv5 = div(tv2, tv4);
  auto tv6 = castOp(dtype, tv5);
  fusion->addOutput(tv6);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = t0.reshape(group_shape).to(at::kFloat);
  auto t2 = t1.sum(reduction_axes);
  for (int idx = 0; idx < ndims; idx++) {
    if (broadcast_tags[idx]) {
      t2 = t2.unsqueeze(idx);
    }
  }
  auto t3 = t1 / t2;
  auto ref = t3.to(at::kHalf);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  // should have only 1 segment group
  auto seg_groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();
  EXPECT_EQ(seg_groups.size(), 1);
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ViewNormalizationTest,
    ::testing::Values(
        std::vector<int64_t>{0},
        std::vector<int64_t>{0, 1},
        std::vector<int64_t>{0, 1, 2},
        std::vector<int64_t>{-1},
        std::vector<int64_t>{-1, -2},
        std::vector<int64_t>{-1, -2, -3},
        std::vector<int64_t>{0, 2, 4},
        std::vector<int64_t>{1, 3}));

// GroupNorm with the last reshape moved to output tensors
// first 3 reshapes are fused in InnerPersistent
// last reshape is NoOp
TEST_F(ReshapeTest, GroupNormReshapeMovedToOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  constexpr int64_t n = 2, c = 128, h = 16, w = 16, g = 32;
  const std::vector<int64_t> input_shape = {n, c, h, w};
  const std::vector<int64_t> group_shape = {n, g, c / g, h, w};
  const std::vector<int64_t> input_shape_wb = {c};
  const std::vector<int64_t> group_shape_wb = {g, c / g};
  DataType dtype = DataType::Half;
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  auto tv1 = makeContigTensor(input_shape_wb.size(), DataType::Float);
  auto tv2 = makeContigTensor(input_shape_wb.size(), DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Float, tv0);
  // reshape from {N, C, H, W} to {N, G, C / G, H, W}
  auto tv4 = reshape(tv3, input_shape, group_shape);
  // normalization
  auto tv5 = sum(tv4, {-1, -2, -3});
  auto tv6 = broadcast(tv5, {false, false, true, true, true});
  auto tv7 = div(tv4, tv6);
  // reshape scale and bias
  auto tv8 = reshape(tv1, input_shape_wb, group_shape_wb);
  auto tv9 = reshape(tv2, input_shape_wb, group_shape_wb);
  // apply scale and bias
  auto tv10 = broadcast(tv8, {true, false, false, true, true});
  auto tv11 = broadcast(tv9, {true, false, false, true, true});
  auto tv12 = mul(tv7, tv10);
  auto tv13 = add(tv12, tv11);
  auto tv14 = castOp(dtype, tv13);
  // reshape back to input shape, segmented as a NoOp
  auto tv15 = reshape(tv14, group_shape, input_shape);
  fusion->addOutput(tv15);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_wb = at::TensorOptions()
                        .dtype(data_type_to_aten(DataType::Float))
                        .device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto tw = at::randn(input_shape_wb, options_wb);
  auto tb = at::randn(input_shape_wb, options_wb);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, tw, tb});
  auto seg_groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  EXPECT_THAT(
      seg_groups,
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::InnerPersistent),
          HeuristicIs(SchedulerType::ExprEval)));

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, tw, tb}, __LINE__, __FILE__);
}

TEST_F(ReshapeTest, MismatchingReshape) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // TODO: use symbolic sizes. Currently, this is not working because of
  // failures in replaceSymbolicSizes
  TensorView* tv0 = makeContigConcreteTensor({2, 3, 5}, DataType::Float);
  // TensorView* tv0 = makeContigTensor(3, DataType::Float);
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = reshape(tv1, {2, 3, 5}, {2 * 3, 5});
  auto tv3 = reshape(tv1, {2, 3, 5}, {2, 3 * 5});
  auto tv4 = cos(tv2);
  auto tv5 = exp(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  // For this fusion, because tv4 and tv5 have different views, and these views
  // are not compatible, traditionally, we were not able to support this fusion.
  // However, with the advanced feature that domains in a TensorDomain can be
  // connected by both forward and backward, we can schedule the entire fusion
  // like tv4 as follows:

  // First, the most difficult part is to schedule tv5. The TensorDomain of
  // tv5 only contains 2 IDs, and non of them is mapped to anything in tv4.
  // So, in order to schedule tv5 like tv4, we need to reconstruct the entire ID
  // graph into tv5.

  // Before schedule, tv5 is:
  //
  //  logical domain: [I0, I1]
  //
  // Now, we want to make tv5 as:
  //
  //                    I2   I3
  //                      \ /
  //  logical domain: [I0, I1]
  //
  // so that [I0, I2, I3] are mapped to the logical domain of fusion input.
  std::vector<IterDomain*> tv5_root{
      // Topological root of tv5, not the root domain of tv5.
      // TODO: rename root domain as producer domain
      tv5->getLogicalDomain()[0],
      tv0->getLogicalDomain()[1]->cloneWithoutRFactor(),
      tv0->getLogicalDomain()[2]->cloneWithoutRFactor(),
  };
  IrBuilder::create<Merge>(tv5->axis(1), tv5_root[1], tv5_root[2]);

  // Now, except for tv4, all tensors contain all IDs that are exact mapped to
  // the logical domain of the fusion input. tv4 not containing IDs exact mapped
  // to the logical domain of the fusion input is not a problem, because tv4 is
  // the reference tensor and the entire fusion will be scheduled like tv4.

  // Now, let's schedule tv1, tv3, and tv5 to be like tv4's logical domain:
  auto schedule = AbstractTensor::zip(
      {tv1->getLogicalDomain(), tv3->getRootDomain(), tv5_root});
  schedule.merge(0);

  // Now, tv5 looks like:
  //
  //                         I2  I3
  //                        / | /
  //  logical domain: [I0  /, I1]
  //                    | /
  //                    I4
  //
  // and `schedule` contains [I4, I3]

  // Now, `schedule` is like the logical domain of tv2 and tv4. So let's append
  // tv2 and tv4 to it so we can parallelizing all of them all together.
  schedule.addRow(tv2->getLogicalDomain()).addRow(tv4->getLogicalDomain());

  // Parallelize all tensors as [BIDx, TIDx]
  schedule.merge(0);
  schedule.split(0, 128);

  schedule.parallelize(0, ParallelType::BIDx);
  schedule.parallelize(1, ParallelType::TIDx);

  // Now, tv5 looks like:
  //
  //                         I2  I3
  //                        / | / |
  //  logical domain: [I0  /, I1] |
  //                    | /       /
  //                    I4       /
  //                      \     /
  //                       \   /
  //                        \ /
  //                        I5
  //                        / \.
  //                    BIDx   TIDx
  //
  // and `schedule` contains [BIDx, TIDx]

  // TODO: make inlining work
  // inlineMost();

  // Set the loop domain of all tensors
  auto uz = schedule.unzip();
  tv1->setLoopDomain(uz[0].as<IterDomain*>());
  tv3->setLoopDomain(uz[1].as<IterDomain*>());
  tv5->setLoopDomain(uz[2].as<IterDomain*>());
  tv2->setLoopDomain(uz[3].as<IterDomain*>());
  tv4->setLoopDomain(uz[4].as<IterDomain*>());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // TODO: use larger tensor size once we are able to successfully parallelize
  // this fusion.
  at::Tensor t0 = at::randn({2, 3, 5}).to(options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test that replacement of a scalar in a reshaped ID that is a product of a
// split gets replaced properly. In this test, the reshape splits the i0=128
// axis into 32, i0 / 32. When we replace extents in the replaceSymbolicSizes
// pass, the i0/32 scalar, but not the constant 32. The original error came from
// replacing just one of those split axes, leaving the logical domain with one
// output of a Split and another IterDomain with no definition, so the logical
// domain was not a simple transformation of the root domain.
// See https://github.com/NVIDIA/Fuser/issues/2671
TEST_F(ReshapeTest, ReplacedScalarInSplitOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType dtype = DataType::Half;
  const std::vector<int64_t> input_shape = {128};
  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  auto tv1 = makeContigTensor(input_shape.size(), dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto S32 = IrBuilder::create<Val>(32L);
  auto S04 = div(tv0->axis(0)->extent(), IrBuilder::create<Val>(32L));
  auto tv2 = reshape(tv0, {S32, S04});
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = segment_set(tv3);
  auto tv5 = reshape(tv1, {S32, S04});
  auto tv6 = castOp(DataType::Float, tv5);
  auto tv7 = mul(tv4, tv6);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// This is a more specific version of the above test. Here we are directly
// replacing one of the Split outputs' extents and testing that we successfully
// perform the replacement and that the definition is intact.
TEST_F(ReshapeTest, ReplaceScalarInSplitOutputManually) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = reshape(tv0, {128}, {32, 4});
  auto tv3 = mul(tv1, tv2);

  fusion->addOutput(tv3);

  ASSERT_EQ(tv2->nDims(), 2);
  IterDomain* old_id = tv2->axis(1);
  Val* old_ext = old_id->extent();
  EXPECT_FALSE(old_ext->isConst());
  ASSERT_FALSE(old_id->definition() == nullptr);
  EXPECT_TRUE(old_id->definition()->isA<Split>());

  Val* replacement = IrBuilder::create<Val>(4, DataType::Index);
  std::unordered_map<Val*, Val*> replacement_map{{old_ext, replacement}};
  ir_utils::replaceValue(fusion.get(), replacement_map);

  ASSERT_EQ(tv2->nDims(), 2);
  IterDomain* new_id = tv2->axis(1);
  Val* new_ext = tv2->axis(1)->extent();
  EXPECT_EQ(new_ext, replacement);
  ASSERT_TRUE(new_ext->isConst());
  EXPECT_EQ(new_ext->value(), 4);
  ASSERT_FALSE(new_id->definition() == nullptr);
  EXPECT_TRUE(new_id->definition()->isA<Split>());
  EXPECT_FALSE(tv2->axis(0)->definition() == nullptr);
  EXPECT_TRUE(tv2->axis(0)->definition() == new_id->definition());
}

TEST_F(ReshapeTest, CyclicReshape) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{4, 8};
  std::vector<int64_t> shape2{32};

  // Create a fusion with a cyclic reshape pattern
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = reshape(tv0, shape1, shape2);
  auto tv2 = reshape(tv1, shape2, shape1);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  EXPECT_TRUE(
      registry_utils::SchedulerTopologyChecker::hasCyclicReshape(&fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Make sure there's no cycle in each segment
  auto segmented_fusion =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments();
  for (const auto group : segmented_fusion->groups()) {
    auto segment_fusion = segmented_fusion->makeFusion(group).second;
    EXPECT_FALSE(
        registry_utils::SchedulerTopologyChecker::hasCyclicReshape(
            segment_fusion.get()));
  }
}

} // namespace nvfuser
