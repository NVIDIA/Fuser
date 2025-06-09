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
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::SizeIs;

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
  at::Tensor t0 = at::randn(input_shape_0, options);
  at::Tensor t1 = at::randn(input_shape_1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn(input_shape_0, options);
  at::Tensor t1 = at::randn(input_shape_1, options);
  at::Tensor t2 = at::randn(input_shape_2, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);

  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor},
      {in_tensor / in_tensor.sum({0})},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor, in_tensor});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor, in_tensor},
      __LINE__,
      __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
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

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor, in_tensor});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {in_tensor, in_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(SegmentationTest, ForwardedExprsAreNotMergeable) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = neg(tv0);
  auto tv2 = slice(tv1, std::vector<int64_t>({0}), std::vector<int64_t>({5}));
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in_tensor = at::randn({10}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  executor_cache.runFusionWithInputs({in0, in1});

  // Check the segmented edge is fp16
  SegmentedFusion* segmented_fusion =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments();
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

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<int64_t> shape{15, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  executor_cache.runFusionWithInputs({in0, in1});

  // Check the segmented edge is bf16
  SegmentedFusion* segmented_fusion =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments();
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

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  executor_cache.runFusionWithInputs({in0, in1});

  SegmentedFusion* segmented_fusion =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments();
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

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<int64_t> shape{16, 16, 16};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn(shape, options);
  auto in1 = at::randn(shape, options);
  executor_cache.runFusionWithInputs({in0, in1});

  SegmentedFusion* segmented_fusion =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments();
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

TEST_F(SegmentationTest, codeGenSupportedMergeIssue1970) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3, DataType::Float);
  fusion->addInput(tv0);

  auto* tv1 = neg(tv0);
  // two uses of forwarded non scalar input leads to duplicated merge of the
  // same consumer
  auto* tv2 = add(tv1, tv1);
  auto* tv3 = segment_set(tv2);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({3, 4, 3}, options);
  auto outputs = executor_cache.runFusionWithInputs({in0});

  testValidate(executor_cache.fusion(), outputs, {in0}, __LINE__, __FILE__);
}

// Test that Reduction axes are removed in segmentation edges
// https://github.com/NVIDIA/Fuser/pull/2487
TEST_F(SegmentationTest, EraseReductionsInSegmentationEdges) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigConcreteTensor({3, 32, 17}, DataType::Float);
  fusion->addInput(tv0);

  auto* tv1 = sum(tv0, {2});
  tv1->setAllocationDomain(
      {tv1->axis(1), tv1->axis(2), tv1->axis(0)}, {true, std::nullopt, true});
  auto* tv2 = sum(tv1, {0});
  auto* tv3 = sum(tv2, {0});

  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({3, 32, 17}, options);
  auto outputs = executor_cache.runFusionWithInputs({in0});

  testValidate(executor_cache.fusion(), outputs, {in0}, __LINE__, __FILE__);

  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();
  ASSERT_TRUE(runtime != nullptr);

  SegmentedFusion* segmented_fusion = runtime->fusionSegments();

  EXPECT_EQ(segmented_fusion->groups().size(), 3);

  for (SegmentedGroup* group : segmented_fusion->groups()) {
    std::unique_ptr<Fusion> segment_fusion =
        segmented_fusion->makeFusion(group).second;

    auto* segment_input = segment_fusion->inputs().at(0)->as<TensorView>();

    EXPECT_FALSE(segment_input->domain()->hasReduction());
  }
}

TEST_F(SegmentationTest, AliasedOutputOnSegmentation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigConcreteTensor({2, 3, 4}, DataType::Float);
  fusion->addInput(tv0);

  auto* tv1 = neg(tv0);
  auto* seg_out = segment_set(tv1);
  // validating segmentation on aliased output is not affecting how outputs are
  // hidden.
  fusion->aliasOutputToInput(seg_out, tv0, AllocationType::ReuseBuffer);
  auto* tv2 = relu(seg_out);
  fusion->addOutput(tv2);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 3, 4}, options);
  auto in0_ref = in0.clone();

  auto outputs = executor_cache.runFusionWithInputs({in0});
  auto in0_neg = in0_ref.neg();
  EXPECT_TRUE(in0_neg.allclose(in0));

  testValidate(
      executor_cache.fusion(),
      outputs,
      {in0.clone()},
      {in0_neg.relu()},
      __LINE__,
      __FILE__);
}

TEST_F(SegmentationTest, MultipleSegmentSetsInOneSegment) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto* in = makeContigTensor(1);
  auto* t = add(in, in);
  auto* seg_out_0 = segment_set(t);
  auto* seg_out_1 = segment_set(t);
  auto* out = add(seg_out_0, seg_out_1);
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor in_tensor = at::randn({10}, options);
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(2));
}

TEST_F(SegmentationTest, ForwardInputsToSegmenterSetIssue2658) {
  // Disable mark aliases prepare pass, which might insert more segment_set
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* vanilla_in = makeContigConcreteTensor({2, 3});
  TensorView* in = relu(vanilla_in);
  TensorView* seg_in = segment_set(in);
  TensorView* permute_out = permute(seg_in, {1, 0});
  TensorView* compute_out = mul(in, in);
  compute_out = add(compute_out, in);
  fusion->addInput(vanilla_in);
  fusion->addOutput(in);
  fusion->addOutput(permute_out);
  fusion->addOutput(compute_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

// Test to verify an upcast is replicated between different segments
TEST_F(SegmentationTest, PrivatizeUpcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);

  auto tv1 = segment_set(tv0);
  auto tv2 = castOp(DataType::Float, tv1);

  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  auto tv4 = sum(tv2, {1});
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 32}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  // There must be three segments, one with ExprEvalExecutor and two
  // with KernelExecutor.
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(3));

  for (const auto& executor : runtime->executors()) {
    // Ignore the one taken care by ExprEvalExecutor
    if (executor.get()->isA<ExprEvalExecutor>()) {
      continue;
    }
    // This segment should corresponds to each of the reductions. Both
    // of them should use tv1.
    auto ke = dynamic_cast<KernelExecutor*>(executor.get());
    ASSERT_NE(ke, nullptr);
    kir::Kernel* kernel = ke->compiledKernel()->kernel();
    EXPECT_EQ(kernel->inputs().size(), 1);
    EXPECT_EQ(kernel->inputs().at(0)->name(), 1);
  }
}

TEST_F(SegmentationTest, PrivatizeUpcastAndSqueeze) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});
  auto tv2 = segment_set(tv1);
  auto tv3 = squeeze(tv2, {0});
  auto tv4 = castOp(DataType::Float, tv3);

  auto tv5 = sum(tv4, {0});
  fusion.addOutput(tv5);

  auto tv6 = sum(tv4, {1});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 32}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  // There must be three segments, one with ExprEvalExecutor and two
  // with KernelExecutor.
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(3));

  for (const auto& executor : runtime->executors()) {
    // Ignore the one taken care by ExprEvalExecutor
    if (executor.get()->isA<ExprEvalExecutor>()) {
      continue;
    }
    // This segment should corresponds to each of the reductions. Both
    // of them should use tv2.
    auto ke = dynamic_cast<KernelExecutor*>(executor.get());
    ASSERT_NE(ke, nullptr);
    kir::Kernel* kernel = ke->compiledKernel()->kernel();
    EXPECT_EQ(kernel->inputs().size(), 1);
    EXPECT_EQ(kernel->inputs().at(0)->name(), 2);
  }
}

// Unlike PrivatizeUpcast, verify replicated upcast ops are
// consolidated back as they are grouped into the same segment
TEST_F(SegmentationTest, RevertPrivatizedUpcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);

  auto tv1 = segment_set(tv0);

  auto tv2 = set(tv1);
  auto tv3 = castOp(DataType::Float, tv2);

  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto tv5 = sum(tv3, {1});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 32}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  KernelArgumentHolder outputs;

  // Make sure NVFUSER_DUMP=segmented_fusion works
  {
    DebugDumpOptionsGuard options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::FusionSegments);
    std::ostringstream tmp_buf;
    DebugStreamGuard debug_stream_guard(tmp_buf);
    outputs = executor_cache.runFusionWithInputs({t0});
  }

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  // There must be two segments, one with ExprEvalExecutor and another
  // with KernelExecutor.
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(2));

  for (const auto& executor : runtime->executors()) {
    // Ignore the one taken care by ExprEvalExecutor
    if (executor.get()->isA<ExprEvalExecutor>()) {
      continue;
    }
    // This segment should have the two reductions. There must be only
    // one upcast op with tv1 as its producer.
    auto ke = dynamic_cast<KernelExecutor*>(executor.get());
    ASSERT_NE(ke, nullptr);
    kir::Kernel* kernel = ke->compiledKernel()->kernel();
    int64_t num_upcast_ops = 0;
    for (auto expr : KernelExprVisitor::getAllExprs(kernel)) {
      auto uop = dynamic_cast<UnaryOp*>(expr);
      if (uop == nullptr || uop->getUnaryOpType() != UnaryOpType::Cast) {
        continue;
      }

      EXPECT_EQ(uop->in()->as<kir::TensorIndex>()->view()->name(), 2);

      ++num_upcast_ops;
    }
    EXPECT_EQ(num_upcast_ops, 1);
  }
}

// Unlike PrivatizeUpcast, verify replicated upcast ops are
// consolidated back as they are grouped into the same segment
/**
 * Test case for verifying the correct segmentation and operation reversion
 * in the presence of privatized upcast and squeeze operations in
 * NVFuser.
 *
 * This test constructs a fusion graph with a sequence of tensor operations:
 * - Symbolic tensor creation and broadcasting
 * - Type casting (upcast) and squeeze
 * - Reduction (sum) operations
 *
 * The test validates:
 * - The fusion produces correct outputs for given random input tensors.
 * - The segmented fusion produces exactly two segments.
 * - Each segment contains either both an upcast and a squeeze operation, or
 * neither; privatizaion will double these ops but they should be removed
 * during privatization.
 *
 * This ensures that privatized upcast and squeeze operations are correctly
 * reverted and handled during segmentation, maintaining correctness and
 * expected segmentation behavior.
 */
TEST_F(SegmentationTest, RevertPrivatizedUpcastAndSqueeze) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {true, false, false});

  auto tv2 = segment_set(tv1);

  auto tv3 = set(tv2);
  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = squeeze(tv4, {0});

  auto tv6 = sum(tv5, {1});
  fusion.addOutput(tv6);

  auto tv7 = sum(tv5, {1});
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 32}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  KernelArgumentHolder outputs;

  // Make sure NVFUSER_DUMP=segmented_fusion works
  {
    DebugDumpOptionsGuard options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::FusionSegments);
    std::ostringstream tmp_buf;
    DebugStreamGuard debug_stream_guard(tmp_buf);
    outputs = executor_cache.runFusionWithInputs({t0});
  }

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  // There must be two segments
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(2));

  for (const auto group : runtime->fusionSegments()->groups()) {
    int64_t num_upcast_ops = 0;
    int64_t num_squeeze_ops = 0;
    for (auto expr : group->exprs()) {
      if (expr->isA<UnaryOp>() &&
          expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Cast) {
        // EXPECT_EQ(expr->input(0)->as<kir::TensorIndex>()->view()->name(), 3);
        ++num_upcast_ops;
      } else if (expr->isA<SqueezeOp>()) {
        ++num_squeeze_ops;
      } else {
        continue;
      }
    }
    EXPECT_TRUE(
        (num_upcast_ops == 1 && num_squeeze_ops == 1) ||
        (num_upcast_ops == 0 && num_squeeze_ops == 0));
  }
}

TEST_F(SegmentationTest, ForwardFull) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // FullOp that is used in two segments
  auto tv1 = full({tv0->axis(0)->extent()}, fusion.oneVal(), DataType::Float);

  auto tv2 = add(tv0, tv1);
  auto tv3 = segment_set(tv2);

  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(runtime->fusionSegments()->groups(), SizeIs(2));

  // Make sure the full output should not be a segment input
  for (const auto& executor : runtime->executors()) {
    auto ke = dynamic_cast<KernelExecutor*>(executor.get());
    ASSERT_NE(ke, nullptr);
    kir::Kernel* kernel = ke->compiledKernel()->kernel();
    bool full_op_found = false;
    for (auto expr : KernelExprVisitor::getAllExprs(kernel)) {
      auto out_tv = ir_utils::getTvOutput(expr);
      if (out_tv == nullptr) {
        continue;
      }
      auto full_op = dynamic_cast<FullOp*>(out_tv->definition());
      if (full_op == nullptr) {
        continue;
      }
      full_op_found = true;
      auto output_it =
          std::ranges::find_if(kernel->outputs(), [&](Val* output) {
            return output->isA<TensorView>() &&
                output->name() == out_tv->name();
          });
      EXPECT_EQ(output_it, kernel->outputs().end())
          << "FullOp ouput should not be a segment output";
    }
    EXPECT_TRUE(full_op_found) << "Each segment has its own FullOp";
  }
}

} // namespace nvfuser
