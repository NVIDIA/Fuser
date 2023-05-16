// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/analysis/divisible_split.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <inlining.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_graphviz.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <mutator.h>
#include <ops/all_ops.h>
#include <register_interface.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>
#include <transform_view.h>

// fuser and IR parser
#include <parser.h>
#include <torch/csrc/jit/ir/irparser.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, FusionViewDtypeSameSizeOutput_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_view = at_x_add_bias.view(at::ScalarType::Int);

  testValidate(&fusion, outputs, aten_inputs, {at_x_view}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionViewDtypeFailMismatchSize_CUDA) {
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

TEST_F(NVFuserTest, FusionViewAsRealOutput_CUDA) {
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

  auto y_plus_1 = add(y, IrBuilder::create<Double>(1));

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
  std::vector<c10::IValue> aten_inputs = {at_x, at_bias, at_y};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_view = at::view_as_real(at_x_add_bias);
  auto at_y_plus_1 = at_y + 1.0;
  auto at_out = at_y_plus_1 + at_x_view;

  testValidate(&fusion, outputs, aten_inputs, {at_out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeRfactorExtentReplacement_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion->addInput(tv1);

  auto tv2 = reshape(tv0, {12, 8}, {4, 3, 8});
  auto tv3 = sum(tv2, {-1});
  auto tv4 = add(tv3, IrBuilder::create<Double>(1));
  auto tv5 = add(tv1, tv4);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({12, 8}, options);
  auto t1 = at::randn({4, 3}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = at::native::view(t0, {4, 3, 8}).sum({-1}) + 1 + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeOutput_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_reshape = at::native::view(at_x_add_bias, output_shape);

  testValidate(
      &fusion, outputs, aten_inputs, {at_x_reshape}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeFailMismatchSize_CUDA) {
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

TEST_F(NVFuserTest, FusionReshapeFailMulitDimInference_CUDA) {
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
    std::vector<int64_t>& input_shape,
    std::vector<int64_t>& output_shape,
    bool reshape_before_reduction) {
  constexpr int kReductionAxis = -1;

  // Drop size for reduction axis from reshape_shape
  std::vector<int64_t> reshape_shape;
  {
    const auto kAxis = (kReductionAxis < 0)
        ? (kReductionAxis + input_shape.size())
        : kReductionAxis;
    for (auto i : c10::irange(input_shape.size())) {
      if (reshape_before_reduction || i != kAxis) {
        reshape_shape.push_back(input_shape[i]);
      }
    }
  }

  auto bias_shape = (reshape_before_reduction) ? input_shape : output_shape;
  for (auto has_implicit_broadcast : {false, true}) {
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
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

    FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

    auto at_tv1 = (reshape_before_reduction) ? (at_x + at_bias)
                                             : at::sum(at_x, kReductionAxis);
    auto at_x_reshape = at::native::view(at_tv1, output_shape);
    auto at_y = (reshape_before_reduction)
        ? at::sum(at_x_reshape, kReductionAxis)
        : at::add(at_x_reshape, at_bias);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

typedef std::vector<int64_t> shape;
typedef std::pair<shape, shape> reshape_example;

// TODO: View examples with just 333 elements are failing validation in
// normalization. This might just be because our tolerances aren't tuned well
// for small sizes and the parallelization could be limited which could be
// detected as a validation issue, though it might not actually be a correctness
// issue. Using 3333 instead of 333 in those cases but should validate what's
// going on in the 333 case.
std::vector<reshape_example> all_reshape_examples = {
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
};

TEST_F(NVFuserTest, FusionReshapeReductionShmoo_CUDA) {
  for (auto e : all_reshape_examples) {
    reductionViewAddFusion(
        e.first, e.second, true /* reshape_before_reduction */);
  }
  std::vector<reshape_example> reshape_after_reduce_examples = {
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

  for (auto e : reshape_after_reduce_examples) {
    reductionViewAddFusion(
        e.first, e.second, false /* reshape_before_reduction */);
  }
}

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
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

    FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
    auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

    auto at_tv1 = (reshape_before_persistent)
        ? (at_x + at_bias)
        : at::_softmax(at_x, kAxis, false /* half_to_float */);
    auto at_x_reshape = at::native::view(at_tv1, inferred_output);
    auto at_y = (reshape_before_persistent)
        ? at::_softmax(at_x_reshape, kAxis, false /* half_to_float */)
        : at::add(at_x_reshape, at_bias);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionReshapePersistentShmoo_CUDA) {
  for (auto e : all_reshape_examples) {
    persistentViewAddFusion(
        e.first, e.second, true /* reshape_before_persistent */);
  }

  for (auto e : all_reshape_examples) {
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
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_add_bias = at_x + at_bias;
    auto at_x_reshape = at::native::view(at_x_add_bias, output_shape);
    auto at_y = at::gelu(at_x_reshape);

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionReshapeSplit_CUDA) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{2, 4, 10};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionReshapeBroadcast_CUDA) {
  std::vector<int64_t> input_shape{80};
  std::vector<int64_t> output_shape{1, 80};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionReshapeMerge_CUDA) {
  std::vector<int64_t> input_shape{2, 40, 7};
  std::vector<int64_t> output_shape{560};
  addViewGeluFusion(input_shape, output_shape);
}

TEST_F(NVFuserTest, FusionReshapeAllShmoo_CUDA) {
  for (auto e : all_reshape_examples) {
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
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_gelu = at::gelu(at_x);
    auto at_x_reshape = at::native::view(at_x_gelu, inferred_output);
    auto at_y = at_x_reshape + at_bias;

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionReshapeStride_CUDA) {
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
    std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

    auto lparams = schedulePointwise(&fusion, aten_inputs);

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs, lparams);
    auto outputs = fe.runFusion(aten_inputs, lparams);

    auto at_x_gelu = at::gelu(at_x);
    auto at_x_reshape = at::native::view(at_x_gelu, output_shape);
    auto at_bias_reshape = at::native::view(at_bias, output_shape);
    auto at_y = at_x_reshape + at_bias_reshape;

    testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionReshapeBinary_CUDA) {
  geluViewBinaryAddFusion({27454, 2}, {54908}, {7844, 7});
}

// Repro of issue #1493
TEST_F(NVFuserTest, FusionReshapeConcreteDomain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = reshape(tv0, {2, 3}, {6});
  auto tv3 = add(tv2, IrBuilder::create<Double>(1));
  auto tv4 = broadcast(tv3, {true, false});
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(0);
  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({2, 3}, options);
  auto t1 = at::randn({1, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = (at::native::view(t0, {6}) + 1).unsqueeze(0) + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeConcreteDomain2_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  auto at_tv1 = at::_softmax(at_x, kAxis, false /* half_to_float */);
  auto at_x_reshape = at::native::view(at_tv1, output_shape);
  auto at_y = at::add(at_x_reshape, at_bias);

  testValidate(&fusion, outputs, aten_inputs, {at_y}, __LINE__, __FILE__);
}

// Repro of issue #1608
TEST_F(NVFuserTest, FusionReshapeConcreteDomain3_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_y, at_z};

  FusionExecutorCache fusion_executor_cache(std::move(fusion_ptr));
  auto outputs = fusion_executor_cache.runFusionWithInputs(aten_inputs);

  auto at_tv1 = at::add(at_x, at_y);
  auto at_tv2 = at::native::view(at_tv1, output_shape);
  auto at_tv3 = at::native::view(at_z, output_shape);
  auto at_output = at::add(at_tv2, at_tv3);

  testValidate(&fusion, outputs, aten_inputs, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeConcreteDomain4_CUDA) {
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

  TORCH_CHECK(tv5->nDims() == 1);

  // The concrete domain of tv5, which is 1D, with permissive or loop mapping
  // needs to be either the domain of tv4 or tv5, both of which have the three
  // concrete root domains of tv1. In other words, it must map with tv4 and tv5
  // with the exact mapping.
  ComputeAtMap map(&fusion);
  auto concrete_id =
      map.getConcreteMappedID(tv5->axis(0), IdMappingMode::PERMISSIVE);
  TORCH_CHECK(
      map.areMapped(concrete_id, tv5->axis(0), IdMappingMode::EXACT),
      "Invalid concrete ID: ",
      concrete_id->toString());
  TORCH_CHECK(
      map.areMapped(concrete_id, tv4->axis(0), IdMappingMode::EXACT),
      "Invalid concrete ID: ",
      concrete_id->toString());
}

TEST_F(NVFuserTest, FusionReshapeConcreteDomain5_CUDA) {
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

    TORCH_CHECK(path1_out->nDims() == 1);
    TORCH_CHECK(path2_out->nDims() == 1);

    ComputeAtMap map(&fusion);

    // Make sure the two output tensors are mapped. Note both are 1D.
    TORCH_CHECK(map.areMapped(
        path1_out->axis(0), path2_out->axis(0), IdMappingMode::LOOP));

    auto concrete_id =
        map.getConcreteMappedID(path2_out->axis(0), IdMappingMode::LOOP);
    TORCH_CHECK(
        path2_out->axis(0) == concrete_id,
        "Incorrect concrete ID: ",
        concrete_id->toString());
  }
}

TEST_F(NVFuserTest, FusionFlattenAfterUnsqueezeOutput_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_bias};

  x_reshape->split(0, 4);
  x_add_bias->computeAt(x_reshape, 1);
  x_reshape->axis(0)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto at_x_add_bias = at_x + at_bias;
  auto at_x_reshape = at_x_add_bias.unsqueeze(-1).flatten();

  testValidate(
      &fusion, outputs, aten_inputs, {at_x_reshape}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionComputeAtRootDomainMapWithView_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> input_shape1{10, 12};
  const std::vector<int64_t> input_shape2{10, 3, 4};

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));

  // reduction followed by broadcast
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true, true});

  // Path with a reshape
  auto tv4 = reshape(tv1, input_shape1, input_shape2);

  // Join the reduciton+broadcast and reshape paths together
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  ComputeAtRootDomainMap map;
  map.build();

  // It's not possible to compute tv1 at the -1 position of
  // t2. ComputeAtRootDomainMap should tell that by not mapping the
  // second axis.
  auto tv1_tv2_mappable_dims =
      map.getMappableDims(tv1->domain(), tv2->domain());
  TORCH_CHECK(
      tv1_tv2_mappable_dims.find(tv1->axis(1)) == tv1_tv2_mappable_dims.end(),
      "Invalid ComputeAtRootDomainMap. Domain should not be mappable: ",
      tv1->axis(1)->toString());
}

TEST_F(NVFuserTest, FusionExpandRepro_CUDA) {
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
  std::vector<c10::IValue> aten_inputs = {at_x, at_y};

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  LaunchParams l_params;
  auto outputs = fe.runFusion(aten_inputs, {}, l_params, {});

  auto out = at_x.expand_as(at_y);

  testValidate(&fusion, outputs, aten_inputs, {out}, __LINE__, __FILE__);

  // second run to verify cached output allocation
  outputs = fe.runFusion(aten_inputs, {}, l_params, {});
  testValidate(&fusion, outputs, aten_inputs, {out}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpandView1_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({4, 1, 8});
  fusion->addInput(tv0);

  auto tv1 = makeConcreteTensor({12, 8});
  fusion->addInput(tv1);

  auto tv2 = expand(
      tv0,
      {IrBuilder::create<Int>(4),
       IrBuilder::create<Int>(3),
       IrBuilder::create<Int>(8)});

  auto tv3 = reshape(tv2, {4, 3, 8}, {12, 8});
  auto tv4 = add(tv3, tv1);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({4, 1, 8}, options);
  auto t1 = at::randn({12, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = at::reshape(t0.expand({4, 3, 8}), {12, 8}) + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpandView2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 8});
  fusion->addInput(tv0);

  auto tv1 = makeConcreteTensor({3, 4, 8});
  fusion->addInput(tv1);

  auto tv2 =
      expand(tv0, {IrBuilder::create<Int>(12), IrBuilder::create<Int>(8)});

  auto tv3 = reshape(tv2, {12, 8}, {3, 4, 8});
  auto tv4 = add(tv3, tv1);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1, 8}, options);
  auto t1 = at::randn({3, 4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = at::reshape(t0.expand({12, 8}), {3, 4, 8}) + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeTransformCache_CUDA) {
  auto assert_matches = [](reshape_example example_0,
                           reshape_example example_1) {
    TORCH_INTERNAL_ASSERT(
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

  auto assert_does_not_match = [](reshape_example example_0,
                                  reshape_example example_1) {
    TORCH_INTERNAL_ASSERT(
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

TEST_F(NVFuserTest, FusionReshapeIdGraph_CUDA) {
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

  // Grab the trivial reduced tensor from t12's reshape.
  ir_utils::producerTvsOf(tv12)[0];

  // Start from the exact iter domain graph of the fusion
  IterDomainGraph id_graph(&fusion);
  auto disjoint_reshape_ids = id_graph.exactNodes();

  TORCH_CHECK(
      id_graph.exactNodes().strictAreMapped(tv2->axis(1), tv4->axis(1)));
  TORCH_CHECK(
      id_graph.exactNodes().strictAreMapped(tv2->axis(2), tv4->axis(2)));

  TORCH_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[1], tv12->getRootDomain()[1]));
  TORCH_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[2], tv12->getRootDomain()[2]));
  TORCH_CHECK(id_graph.exactNodes().strictAreMapped(
      tv2->getRootDomain()[3], tv12->getRootDomain()[3]));
}

TEST_F(NVFuserTest, FusionReshapeVectorize_CUDA) {
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
  at::Tensor input = at::randn({256, 1024, 1024}, options);

  auto lparams = schedulePointwise(&fusion, {input});

  auto hasVectorization = [](TensorView* tv) -> bool {
    for (auto i : tv->getLeafDomain()) {
      if (i->getParallelType() == ParallelType::Vectorize) {
        return true;
      }
    }
    return false;
  };

  for (auto o : fusion.outputs()) {
    TORCH_CHECK(hasVectorization(o->as<TensorView>()));
  }
  for (auto i : fusion.inputs()) {
    for (auto c : ir_utils::consumerTvsOf(i->as<TensorView>())) {
      TORCH_CHECK(hasVectorization(c));
    }
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input}, lparams);
  auto outputs = fe.runFusion({input}, lparams);

  auto tv_ref = input.flatten(1, 2).sin();

  testValidate(&fusion, outputs, {input}, {tv_ref, tv_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpandFlatten_CUDA) {
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
       IrBuilder::create<Int>(8)});
  auto tv2 = flatten(tv1, 1, 2);
  auto tv3 = sum(tv2, {1});
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({256, 1024, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({input});

  auto aten_out = input.expand({256, 1024, 8}).flatten(1, 2).sum(1);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {input},
      {aten_out},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionIllegalReductionFlatten_CUDA) {
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
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Invalid end_dim")));
}

TEST_F(NVFuserTest, FusionReductionFlatten1_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({2, 3, 5});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = flatten(tv1, 0, 1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({2, 3, 5}, options);
  auto ref = t0.sum({1}).flatten(0, 1);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPwiseViewSchedule_CUDA) {
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
    MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);
  }

  for (auto i : c10::irange(tv5->nDims() - 1)) {
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
    MaxRootDomainInfoSpanningTree spanning_tree(tv5);
    spanning_tree.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(tv5);

    // Inline the schedule
    inlineMost();
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {x, y * z});
  auto t4 = at::native::view(t3, {x, y * z});
  auto t5 = t0 + t3;

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t3});
  auto cg_outputs = fe.runFusion({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, {t2, t4, t5}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSumViewSchedule_CUDA) {
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
    MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);
  }

  tv5->split(1, 128);
  tv5->split(1, 4);

  auto tv5_rf = tv5->rFactor({1, 2});
  tv5_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv5_rf->axis(2)->parallelize(ParallelType::Unroll);
  tv5_rf->axis(3)->parallelize(ParallelType::TIDx);

  {
    TransformPropagator propagator(tv5_rf);
    MaxRootDomainInfoSpanningTree spanning_tree(tv5_rf);
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

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t3});
  auto cg_outputs = fe.runFusion({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, {t2, t5, t6}, __LINE__, __FILE__);
}

// Make sure matching reshapes are segmented into the same kernel
TEST_F(NVFuserTest, FusionReshapeMagicSchedule1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {x, y * z});
  auto t4 = at::native::view(t3, {x, y * z});
  auto t5 = t0 + t3;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});
  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());

  testValidate(&fusion, cg_outputs, {t0, t3}, {t2, t4, t5}, __LINE__, __FILE__);
}

// Make sure reshapes of reshapes are correct
TEST_F(NVFuserTest, FusionReshapeMagicSchedule2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 31, y = 65, z = 103;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  auto tv3 = reshape(tv2, {x, y * z}, {x * y, z});
  auto tv4 = reshape(tv3, {x * y, z}, {y, x * z});
  auto tv5 = reshape(tv4, {y, x * z}, {x, y, z});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  auto aten_out = sin(t0);

  // For now pointwise scheduler only accepts a single reshape at a time, so
  // this will be broken up into multiple kernels. This is due to the reference
  // check looking for all mappings to all input IDs.
  // TODO: Fix the reference check for this case
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(&fusion, cg_outputs, {t0}, {aten_out}, __LINE__, __FILE__);
}

// Make sure broadcasts not on the reshape path that don't interfere with
// reshape are segmented in one kernel and correctly trigger 2D pointwise
// scheduling
TEST_F(NVFuserTest, FusionReshapeMagicSchedule3_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int w = 15, x = 31, y = 49, z = 65;

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

  // Broadcast on another branch to drive the pointwise reference to not be on
  // the reshape paths.

  auto tv6 = makeConcreteTensor({w, x, y, z});
  fusion.addInput(tv6);
  auto tv7 = broadcast(tv0, {true, false, false, false});
  auto tv8 = add(tv6, tv7);
  // tv8 should be the reference for the pointwise fusion. This broadcast
  // pattern doesn't interfere with the reshapes, so this should also be
  // scheduled as 2D.
  fusion.addOutput(tv8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {x, y * z});
  auto t4 = at::native::view(t3, {x, y * z});
  auto t5 = t0 + t3;
  at::Tensor t6 = at::randn({w, x, y, z}, options);
  auto t8 = t6.add(t0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t6});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<PointwiseParams>());
  auto pparams =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  TORCH_CHECK(pparams->break_point == 1);

  testValidate(
      &fusion, cg_outputs, {t0, t3, t6}, {t2, t4, t5, t8}, __LINE__, __FILE__);
}

// Make sure broadcasts through reshapes when not conflicting with reshape are
// segmented into one kernel and trigger 2D pointwise scheduler.
TEST_F(NVFuserTest, FusionReshapeMagicSchedule4_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 31, y = 49, z = 65;

  auto tv0 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv0);

  auto tv1 = sin(tv0);

  auto tv2 = reshape(tv1, {x, y, z}, {x, y * z});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({x, y, z});
  fusion.addInput(tv3);

  auto tv4 = makeConcreteTensor({x, 1, 1});
  fusion.addInput(tv4);

  auto tv5 = add(tv4, tv3);

  auto tv6 = reshape(tv5, {x, y, z}, {x, y * z});
  fusion.addOutput(tv6);

  // Link 0 and 3 together for reshape analysis done based on before the
  // reshapes actually happened.
  auto tv7 = add(tv0, tv3);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t3 = at::randn({x, y, z}, options);
  at::Tensor t4 = at::randn({x, 1, 1}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {x, y * z});
  auto t5 = t4 + t3;
  auto t6 = at::native::view(t5, {x, y * z});
  auto t7 = t0 + t3;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t4});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<PointwiseParams>());
  auto pparams =
      executor_cache.getMostRecentExecutorInfo().params->as<PointwiseParams>();
  TORCH_CHECK(pparams->break_point == 1);

  testValidate(
      &fusion, cg_outputs, {t0, t3, t4}, {t2, t6, t7}, __LINE__, __FILE__);
}

// Make sure different reshapes that are consumed by the reference are segmented
// into a single kernel.
TEST_F(NVFuserTest, FusionReshapeMagicSchedule5_CUDA) {
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
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {z, y, x, w});
  at::Tensor t3 = at::randn({w, x * y, z}, options);
  auto t4 = cos(t3);
  auto t5 = at::native::view(t4, {z, y, x, w});
  auto t6 = add(t2, t5);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<PointwiseParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, {t6}, __LINE__, __FILE__);
}

// Test reshape/transpose and its impact on vectorization
TEST_F(NVFuserTest, FusionReshapeMagicSchedule6_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 128, y = 128;

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
  auto t1 = at::native::view(t0, {x, y / 2, 2});

  auto t2 = t1.transpose(0, 1);
  at::Tensor t3 = at::randn({y / 2, x, 2}, options);
  auto t4 = add(t2, t3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<PointwiseParams>());
  TORCH_CHECK(
      executor_cache.getMostRecentExecutorInfo()
          .params->as<PointwiseParams>()
          ->vectorize &&
      executor_cache.getMostRecentExecutorInfo()
          .params->as<PointwiseParams>()
          ->unroll_factor);

  testValidate(&fusion, cg_outputs, {t0, t3}, {t4}, __LINE__, __FILE__);
}

// View with 3D reduction scheduling
TEST_F(NVFuserTest, FusionReshapeMagicSchedule7_CUDA) {
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
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {v * w, x, y * z});
  at::Tensor t3 = at::randn({v, w, x, z, y}, options);
  auto t4 = cos(t3);
  auto t5 = at::native::view(t4, {v * w, x, y * z});
  auto t7 = add(t2, t5).sum(2).sum(0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<ReductionParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, {t7}, __LINE__, __FILE__);
}

// View with 3D normalization scheduling
TEST_F(NVFuserTest, FusionReshapeMagicSchedule8_CUDA) {
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
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {v * w, x, y * z});
  // This might trigger transpose kernel.
  at::Tensor t3 = at::randn({v, w, x, z, y}, options);
  auto t4 = cos(t3);
  auto t5 = at::native::view(t4, {v * w, x, y * z});
  auto t6 = add(t2, t5);
  auto t7 = t6.sum(2).sum(0);
  auto t8 = t7.unsqueeze(-1).unsqueeze(0);
  auto t9 = t6 + t8;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  // Collect the heuristic params
  executor_cache.profile(true);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3});

  TORCH_CHECK(!executor_cache.getMostRecentKernelRuntime()->isSegmented());
  TORCH_CHECK(executor_cache.getMostRecentExecutorInfo()
                  .params->isA<ReductionParams>());

  testValidate(&fusion, cg_outputs, {t0, t3}, {t9}, __LINE__, __FILE__);
}

// AlbertForMaskedLM repro https://github.com/csarofeen/pytorch/issues/2066
TEST_F(NVFuserTest, FusionReshapeMagicSchedule9_CUDA) {
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

  auto s10 = IrBuilder::create<Double>(1e-12);
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

  auto t5 = t0.unsqueeze(0).unsqueeze(0);
  auto t6 = t1.unsqueeze(-1);
  auto t7 = t2.unsqueeze(0).unsqueeze(0);
  auto t8 = t3.unsqueeze(-1);
  auto t9 = t6;

  auto t11 = t8.abs().add(1.e-12);
  auto t12 = t4.sub(t9);
  auto t13 = t11.rsqrt();
  auto t14 = t13;
  auto t15 = t12.mul(t14);
  auto t16 = t15.mul(t5);
  auto t17 = t16.add(t7);
  auto t18 = t17.to(at::kFloat);
  auto t19 = at::native::view(t18, {x * y, z});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});

  testValidate(
      &fusion,
      cg_outputs,
      {t0, t1, t2, t3, t4},
      {t6, t13, t19},
      __LINE__,
      __FILE__);
}

// Simpler version of FusionReshapeMagicSchedule9_CUDA
TEST_F(NVFuserTest, FusionReshapeMagicSchedule10_CUDA) {
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
TEST_F(NVFuserTest, FusionReshapeMagicSchedule11_CUDA) {
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
  auto t2 = at::native::view(t0, {1, x, y * z});
  auto t3 = at::native::view(t2, {x, y * z});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t3}, __LINE__, __FILE__);
}

// Make sure different reshapes that are consumed by the reference are segmented
// into a single kernel.
TEST_F(NVFuserTest, FusionReshapeMapping_CUDA) {
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
  MaxRootDomainInfoSpanningTree spanning_tree(tv6);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv6);

  // Inline the schedule
  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, x, y * z}, options);
  auto t1 = sin(t0);
  auto t2 = at::native::view(t1, {z, y, x, w});
  at::Tensor t3 = at::randn({w, x * y, z}, options);
  auto t4 = cos(t3);
  auto t5 = at::native::view(t4, {z, y, x, w});
  auto t6 = add(t2, t5);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t3});
  auto cg_outputs = fe.runFusion({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, {t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLowerDivisibleSplits_CUDA) {
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
  MaxRootDomainInfoSpanningTree spanning_tree(tv2);
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

    TORCH_CHECK(
        divisible_splits.find(transform_5) != divisible_splits.end(),
        "Expecting: ",
        transform_5->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
    TORCH_CHECK(
        divisible_splits.find(transform_6) != divisible_splits.end(),
        "Expecting: ",
        transform_6->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
    TORCH_CHECK(
        divisible_splits.find(transform_7) != divisible_splits.end(),
        "Expecting: ",
        transform_7->toString(),
        "\nFrom TV: ",
        tv,
        "\nTo be a divisible split.");
  }
}

TEST_F(NVFuserTest, FusionIssue2076_CUDA) {
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

  auto tv5 = mul(tv3, IrBuilder::create<Double>(1));
  auto tv6 = sub(IrBuilder::create<Double>(1), tv5);
  auto tv7 = castOp(DataType::Bool, tv6);
  auto tv8 =
      where(tv7, IrBuilder::create<Double>(-3.4028200000000001e+38), tv6);
  auto tv9 = add(tv8, tv2);
  auto tv10 = set(tv9);
  auto tv11 = expand(
      tv10,
      {tv10->axis(0)->extent(),
       IrBuilder::create<Int>(12),
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
       IrBuilder::create<Int>(128)});
  auto tv18 = sub(tv13, tv17);
  auto tv19 = exp(tv18);
  auto tv20 = sum(tv19, {2});
  auto tv21 = broadcast(tv20, {false, false, true});
  auto tv22 = set(tv21);
  auto tv23 = expand(
      tv22,
      {tv22->axis(0)->extent(),
       tv22->axis(1)->extent(),
       IrBuilder::create<Int>(128)});
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
TEST_F(NVFuserTest, FusionIssue2076_v2_CUDA) {
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

  auto t3 = t1.reshape({4, 12, 128});

  // [4, 1, 128]
  auto t4 = t0.add(t2);
  auto t5 = t3.add(t4);
  auto t6 = t5.reshape({48, 128});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(&fusion, cg_outputs, {t0, t1, t2}, {t4, t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeZeroDimInput_CUDA) {
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
  TORCH_INTERNAL_ASSERT(at_x.ndimension() == 0);

  at::Tensor at_y = at::randn({2, 3, 4}).to(options);

  std::vector<c10::IValue> aten_inputs = {at_x, at_y};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_prod = at_x * at_y;

  testValidate(&fusion, outputs, aten_inputs, {at_prod}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeZeroDimOutput_CUDA) {
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
  TORCH_INTERNAL_ASSERT(at_z.ndimension() == 0);

  std::vector<c10::IValue> aten_inputs = {at_x, at_y, at_z};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_prod = (at_x.squeeze() + at_y.squeeze()) * at_z;

  testValidate(&fusion, outputs, aten_inputs, {at_prod}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReshapeZeroDimInputOutput_CUDA) {
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
  TORCH_INTERNAL_ASSERT(at_x.ndimension() == 0 && at_y.ndimension() == 0);

  std::vector<c10::IValue> aten_inputs = {at_x, at_y};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto outputs = fe.runFusion(aten_inputs, lparams);

  auto at_prod = at_x * at_y;

  testValidate(&fusion, outputs, aten_inputs, {at_prod}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, ReshapeOfReshape_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = reshape(tv0, {4, 8}, {8, 4});
  auto tv2 = reshape(tv1, {8, 4}, {32});
  fusion->addOutput(tv2);

  std::vector<int64_t> shape({4, 8});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(!runtime->isSegmented(), "Segmentation not expected");

  auto ref = t0.reshape({8, 4}).reshape({32});

  TORCH_CHECK(ref.equal(cg_outputs.at(0)));
}

} // namespace nvfuser
