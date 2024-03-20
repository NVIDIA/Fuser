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
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

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

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);

  testValidate(&fusion, outputs, {at_in_0, at_in_1}, __LINE__, __FILE__);
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

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);

  testValidate(
      &fusion, outputs, {at_in_0, at_in_1, at_in_2}, __LINE__, __FILE__);
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

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  const std::vector<SegmentedGroup*>& groups =
      runtime->fusionSegments()->groups();
  EXPECT_EQ(groups.size(), 2) << "Segmentation hint isn't working as expected";

  // with the hint, segment_set should be grouped with its producer
  // [relu, segment_set], [neg]
  for (auto& group : groups) {
    // we only check the group with a single node
    if (group->exprs().size() == 1) {
      auto relu_expr = group->exprs()[0];
      EXPECT_TRUE(
          relu_expr->isA<UnaryOp>() &&
          relu_expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Neg)
          << "segmentation result is not expected";
    }
  }
  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, SegmentHintOnNonTerminatingOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* add_out = add(in, in);
  add_out = segment_set(add_out);
  TensorView* mul_out = mul(add_out, add_out);

  fusion->addInput(in);
  fusion->addOutput(add_out);
  fusion->addOutput(mul_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  // Segment 1: in -> add_out (defined by segment_set)
  // Segment 2: add_out -> mul_out
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);
}

TEST_F(SegmentationTest, EnforceSegmentationByCachingBeforeAndAfter) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = sum(tv0, {0});
  TensorView* tv2 = div(tv0, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  // A fake proxy for the real isSharding check.
  auto is_sharding = [](Expr* expr) -> bool {
    return expr->isA<ReductionOp>();
  };

  // I'd put this in a pre-segmenter pass.
  std::vector<Expr*> sharding_exprs;
  for (Expr* expr : fusion->exprs()) {
    if (is_sharding(expr)) {
      sharding_exprs.push_back(expr);
    }
  }
  for (Expr* sharding_expr : sharding_exprs) {
    for (TensorView* in_tv :
         ir_utils::filterByType<TensorView>(sharding_expr->inputs())) {
      if (!in_tv->isFusionInput()) {
        in_tv->cacheBefore(LoadStoreOpType::SegmenterSet);
      }
    }
    for (TensorView* out_tv :
         ir_utils::filterByType<TensorView>(sharding_expr->outputs())) {
      if (!out_tv->isFusionOutput()) {
        out_tv->cacheAfter(LoadStoreOpType::SegmenterSet);
      }
    }
  }

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(
      fec.fusion(),
      out_tensors,
      {in_tensor},
      {in_tensor / in_tensor.sum({0})},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);
}

TEST_F(SegmentationTest, SetAllocationDomainOnSegmentBoundary) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* add_out = add(in, in);
  add_out = segment_set(add_out);
  TensorView* reshape_out = reshape(add_out, {2, 3, 5}, {6, 5});

  fusion->addInput(in);
  fusion->addOutput(reshape_out);

  add_out->setAllocationDomain(
      {add_out->axis(0), add_out->axis(1), add_out->axis(2)}, false);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, InputForwardingUntilBinary) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigConcreteTensor({2, 3}, DataType::Half);
  TensorView* y = makeContigConcreteTensor({2, 3}, DataType::Half);
  fusion->addInput(x);
  fusion->addInput(y);

  x = castOp(DataType::Float, x);
  x = neg(x);
  x = sin(x);

  y = castOp(DataType::Float, y);
  y = sin(y);
  y = neg(y);

  TensorView* z = add(x, y);
  // This `segment_set` is needed to trigger input forwarding. Otherwise, the
  // whole fusion will be accepted by pointwise.
  z = segment_set(z);
  fusion->addOutput(z);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  std::vector<at::Tensor> out_tensors =
      fec.runFusionWithInputs({in_tensor, in_tensor});
  testValidate(
      fec.fusion(), out_tensors, {in_tensor, in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 1);
}

TEST_F(SegmentationTest, InputForwardingUntilOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create two chains of ops so the test triggers input forwarding. With one
  // chain, the whole fusion will be accepted by pointwise.
  TensorView* in0 = makeContigTensor(2);
  TensorView* in1 = makeContigTensor(2);
  TensorView* out0 = tanh(in0);
  out0 = sin(out0);
  TensorView* out1 = sin(in1);
  out1 = tanh(out1);

  fusion->addInput(in0);
  fusion->addInput(in1);
  fusion->addOutput(out0);
  fusion->addOutput(out1);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  std::vector<at::Tensor> out_tensors =
      fec.runFusionWithInputs({in_tensor, in_tensor});
  testValidate(
      fec.fusion(), out_tensors, {in_tensor, in_tensor}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, ForwardedExprsAreNotMergeable) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = neg(tv0);
  auto tv2 = slice(tv1, {0}, {5});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in_tensor = at::randn({10}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, ForwardedExprsAreReplicated) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = neg(tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = sum(tv1, {1});
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in_tensor = at::randn({10, 20}, options);

  FusionExecutorCache fec(std::move(fusion));
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, ForceFp16Simple) {
  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IoToLowerPrecision);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});

  // Group 2
  auto tv4 = add(tv3, tv1); // Edge: tv3: expect cast
  auto tv5 = castOp(DataType::Half, tv4);

  fusion->addOutput(tv5);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  // Check the segmented edge is fp16
  SegmentedFusion* segmented_fusion =
      fec.getMostRecentKernelRuntime()->fusionSegments();
  for (SegmentedEdge* edge : segmented_fusion->edges()) {
    auto* edge_tv = edge->val->as<TensorView>();
    EXPECT_EQ(edge_tv->getDataType(), DataType::Half);
  }
}

TEST_F(SegmentationTest, ForceBf16Simple) {
#if !defined(CUDA_VERSION) || CUDA_VERSION < 11000
  GTEST_SKIP() << "requires cuda 11.0 or newer toolkit";
#endif

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IoToLowerPrecision);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv2 = sum(tv0, {1});
  auto tv3 = broadcast(tv2, {false, true});

  // Group 2
  auto tv4 = add(tv3, tv1); // Edge: tv3: expect cast
  auto tv5 = castOp(DataType::BFloat16, tv4);

  fusion->addOutput(tv5);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  // Check the segmented edge is bf16
  SegmentedFusion* segmented_fusion =
      fec.getMostRecentKernelRuntime()->fusionSegments();
  for (SegmentedEdge* edge : segmented_fusion->edges()) {
    auto* edge_tv = edge->val->as<TensorView>();
    EXPECT_EQ(edge_tv->getDataType(), DataType::BFloat16);
  }
}

TEST_F(SegmentationTest, ForceFp16NotAllCast) {
  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IoToLowerPrecision);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv3 = sum(tv0, {1});
  auto tv4 = broadcast(tv3, {false, true, false});
  auto tv5 = sum(tv0, {1});

  // Group 2
  auto tv6 = add(tv4, tv1); // edge tv4, expect cast
  auto tv7 = castOp(DataType::Half, tv6);

  // Group 3
  auto tv8 = sum(tv5, {1}); // edge tv5, don't expect cast

  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  SegmentedFusion* segmented_fusion =
      fec.getMostRecentKernelRuntime()->fusionSegments();
  Fusion* complete_fusion = segmented_fusion->completeFusion();

  // Check that the edge that wasn't fp16 is the producer of the
  //  reduction op, i.e. tv8 = sum(tv5,{1});.
  for (SegmentedEdge* edge : segmented_fusion->edges()) {
    auto* edge_tv = edge->val->as<TensorView>();
    if (edge_tv->getDataType() == DataType::Float) {
      Expr* consumer = *(complete_fusion->unordered_uses(edge_tv).begin());
      EXPECT_TRUE(consumer->isA<ReductionOp>());
    }
  }
}

TEST_F(SegmentationTest, ForceBf16NotAllCast) {
#if !defined(CUDA_VERSION) || CUDA_VERSION < 11000
  GTEST_SKIP() << "requires cuda 11.0 or newer toolkit";
#endif

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IoToLowerPrecision);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Group 1
  auto tv3 = sum(tv0, {1});
  auto tv4 = broadcast(tv3, {false, true, false});
  auto tv5 = sum(tv0, {1});

  // Group 2
  auto tv6 = add(tv4, tv1); // edge tv4, expect cast
  auto tv7 = castOp(DataType::BFloat16, tv6);

  // Group 3
  auto tv8 = sum(tv5, {1}); // edge tv5, don't expect cast

  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  fec.runFusionWithInputs({in0, in1});

  SegmentedFusion* segmented_fusion =
      fec.getMostRecentKernelRuntime()->fusionSegments();
  Fusion* complete_fusion = segmented_fusion->completeFusion();

  // Check that the edge that wasn't fp16 is the producer of the
  //  reduction op, i.e. tv8 = sum(tv5,{1});.
  for (SegmentedEdge* edge : segmented_fusion->edges()) {
    auto* edge_tv = edge->val->as<TensorView>();
    if (edge_tv->getDataType() == DataType::Float) {
      Expr* consumer = *(complete_fusion->unordered_uses(edge_tv).begin());
      EXPECT_TRUE(consumer->isA<ReductionOp>());
    }
  }
}

// Test that a segment with a slice does not introduce a cast
// See https://github.com/NVIDIA/Fuser/pull/1936
TEST_F(SegmentationTest, SliceSegmentCasts) {
  EnableOptionsGuard opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::IoToLowerPrecision);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::Half);

  fusion->addInput(tv0);

  // Group 1
  auto tv1 = mul(tv0, tv0);
  // Group 2
  auto tv2 = slice(tv1, {0}, {3}, {1});
  auto tv3 = add(tv2, tv2);

  fusion->addOutput(tv3);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto in0 = at::randn({5}, options);
  auto outputs = fec.runFusionWithInputs({in0});

  SegmentedFusion* segmented_fusion =
      fec.getMostRecentKernelRuntime()->fusionSegments();

  ASSERT_EQ(segmented_fusion->edges().size(), 1);

  SegmentedEdge* slice_edge = segmented_fusion->edges().at(0);

  // Expect edge to be half-precision
  // TODO: Change this rhs to DataType::Half once we have addressed
  // https://github.com/NVIDIA/Fuser/issues/1902
  EXPECT_EQ(slice_edge->val->getDataType(), DataType::Float);

  // There should be no cast before the slice
  EXPECT_TRUE(slice_edge->val->uses().at(0)->isA<SliceOp>());

  testValidate(fec.fusion(), outputs, {in0}, __LINE__, __FILE__);
}

TEST_F(SegmentationTest, codeGenSupportedMergeIssue1970) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3, DataType::Half);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Half);
  fusion->addInput(tv1);

  auto* tv1 = castOp(DataType::Float, tv0);
  auto* tv2 = sum(tv1, {0, 2});
  auto* tv3 = castOp(DataType::Float, tv1);
  auto* tv4 = add(tv2, tv3);
  auto* tv5 = castOp(DataType::Half, tv4);
  fusion->addOutput(tv5);
  auto* tv6 = add(tv5, tv5);
  auto* tv7 = castOp(DataType::Half, tv6);
  fusion->addOutput(tv7);

  FusionExecutorCache fec(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto in0 = at::randn({3, 4, 3}, options);
  auto in1 = at::randn({4}, options);
  auto outputs = fec.runFusionWithInputs({in0, in1});

  testValidate(fec.fusion(), outputs, {in0}, __LINE__, __FILE__);
}

} // namespace nvfuser
