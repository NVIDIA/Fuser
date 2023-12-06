// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>


namespace nvfuser {
using SegmentationTest = NVFuserTest;

TEST_F(SegmentationTest, Issue1284_Repro1) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {10, 20};
  std::vector<int64_t> input_shape_1 = {15};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  fusion.addInput(in_0);
  fusion.addInput(in_1);

  TensorView* out_0 = add(in_0, IrBuilder::create<Val>(0.f));
  TensorView* out_1 = add(in_1, IrBuilder::create<Val>(2.f));

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  std::vector<c10::IValue> aten_inputs = {at_in_0, at_in_1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t1 = at_in_1 + 2;

  auto runtime = fec.getMostRecentKernelRuntime();
  NVF_ERROR(runtime->isSegmented());
  NVF_ERROR(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion, outputs, {at_in_0, at_in_1}, {at_in_0, t1}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, Issue1284_Repro2) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {4, 4};
  std::vector<int64_t> input_shape_1 = {3, 4, 4};
  std::vector<int64_t> input_shape_2 = {2, 8, 4, 4};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  TensorView* in_2 = makeSymbolicTensor(input_shape_2.size());

  fusion.addInput(in_0);
  fusion.addInput(in_1);
  fusion.addInput(in_2);

  TensorView* out_0 = add(in_0, in_1);
  TensorView* out_1 = add(in_0, in_2);

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  at::Tensor at_in_2 = at::randn(input_shape_2, options);

  std::vector<c10::IValue> aten_inputs = {at_in_0, at_in_1, at_in_2};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t0 = at_in_0 + at_in_1;
  auto t1 = at_in_0 + at_in_2;

  auto runtime = fec.getMostRecentKernelRuntime();
  NVF_ERROR(runtime->isSegmented());
  NVF_ERROR(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion,
      outputs,
      {at_in_0, at_in_1, at_in_2},
      {t0, t1},
      __LINE__,
      __FILE__);
}


// Test forced segmentation hint
TEST_F(SegmentationTest, SegmenterHint) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> input_shape{32, 64, 8, 128};
  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();
  fusion->addInput(tv0);
  auto tv1 = relu(tv0);
  auto tv2 = segment_set(tv1);
  auto tv3 = neg(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({at_x});
  auto ref_out = at_x.clone().relu().neg();

  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();

  NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen");
  auto groups = optimized_fusion->fusionSegments()->groups();
  NVF_CHECK(groups.size() == 2, "segmentation hint isn't working as expected");
  // with the hint, segment_set should be grouped with its producer
  // [relu, segment_set], [neg]
  for (auto& group : groups) {
    // we only check the group with a single node
    if (group->exprs().size() == 1) {
      auto relu_expr = group->exprs()[0];
      NVF_CHECK(
          relu_expr->isA<UnaryOp>() &&
              relu_expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Neg,
          "segmentation result is not expected");
    }
  }
  testValidate(
      executor_cache.fusion(), outputs, {at_x}, {ref_out}, __LINE__, __FILE__);
}


} // namespace nvfuser
