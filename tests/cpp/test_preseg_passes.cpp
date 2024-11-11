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

#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/optimization_pass.h>
#include <preseg_passes/pre_segmenter.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser::preseg_passes {

using PresegTest = NVFuserTest;

TEST_F(PresegTest, FusionTestOptimizationPassFlag) {
  class DerivedPass : public OptimizationPass<DerivedPass> {
    friend class OptimizationPass<DerivedPass>;

   protected:
    static void runPass(Fusion* fusion) {
      throw std::runtime_error("running DerivedPass");
    };
    static std::string name() {
      return "DerivedPass";
    }
  };

  auto fusion = std::make_unique<Fusion>();

  {
    // disabling the flag explicitly
    OptimizationPassGuard<DerivedPass> guard(false);
    OptimizationPass<DerivedPass>::runPass(fusion.get());
  }

  // the flag should be default on
  EXPECT_THAT(
      [&]() { OptimizationPass<DerivedPass>::runPass(fusion.get()); },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("running DerivedPass")));
}

TEST_F(PresegTest, FusionCyclicGraph) {
  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   // .ndims(input_shape.size())
                   .dtype(DataType::Double)
                   .build();
    fusion->addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    auto tv3 = set(tv2);
    auto tv4 = set(tv3);
    fusion->addOutput(tv4);

    NVF_CHECK(
        ir_utils::checkCycle(fusion.get()).empty(),
        "no cycle should be detected in fusion");
    // manually creating a cycle on the an active branch
    auto expr = tv2->definition();
    ir_utils::replaceValInExprInputs(expr, tv1, tv4);
    NVF_CHECK(
        ir_utils::checkCycle(fusion.get()).size() == 6,
        "cycle of size 6 should be detected in fusion");
    EXPECT_THAT(
        [&]() { StmtSort::getStmtsBetween({}, fusion->outputs()); },
        ::testing::ThrowsMessage<nvfuser::nvfError>(
            ::testing::HasSubstr("Statements found in the cycle")));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    int M = 64, N = 128;
    bool keepdim = false;
    int correction = 1;
    at::ScalarType dtype = at::kFloat;

    auto tv0 = makeSymbolicTensor(2, aten_to_data_type(dtype));
    fusion->addInput(tv0);
    auto tvs = variance_mean(tv0, {1}, correction, keepdim);
    auto tv_var = tvs.var;
    fusion->addOutput(tv_var);
    auto tv_mean = tvs.mean;
    nvfuser::Val* s0 = IrBuilder::create<Val>(1.0, DataType::Double);
    auto tv1 = add(tv_mean, s0);
    auto tv2 = set(tv1);

    // manually creating a cycle on dead branch
    auto expr = tv1->definition();
    ir_utils::replaceValInExprInputs(expr, s0, tv2);

    // cycle on dead branch shouldn't be picked up by default
    NVF_CHECK(
        ir_utils::checkCycle(fusion.get()).empty(),
        "cycle on dead branch shouldn't be detected");

    // adding dead branch to destination
    auto to = fusion->outputs();
    to.push_back(tv1);
    // cycle should be detected, since dead branch is in our check path
    EXPECT_THAT(
        [&]() { StmtSort::getStmtsBetween({}, to); },
        ::testing::ThrowsMessage<nvfuser::nvfError>(
            ::testing::HasSubstr("Statements found in the cycle")));

    // check for proper size of cycle detected
    NVF_CHECK(
        ir_utils::checkCycle(fusion.get(), {}, to).size() == 4,
        "cycle with size 4 before `to` should be detected");

    // adding `tv2` to `from` to hide cycle from detection
    std::unordered_set<Statement*> from;
    from.insert(tv2);
    NVF_CHECK(
        ir_utils::checkCycle(fusion.get(), from, to).empty(),
        "cycle after `from` shouldn't be detected");

    // running the unmodified fusion should succeed. cycle on dead branch
    // shouldn't have any real impact
    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({M, N}, options);

    FusionExecutorCache executor_cache(std::move(fusion));
    auto outputs = executor_cache.runFusionWithInputs({t0});

    testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
  }
}

// Test cast optimization
TEST_F(PresegTest, FusionTestCastOptimization) {
  std::vector<int64_t> input_shape{3, 7, 8};
  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Double)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Float, tv0);
    tv = castOp(DataType::Half, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    // (input)double -> float -> half -> float -> double
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)double -> half -> double
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::Double, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Double)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Float, tv0);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Half, tv);
    // (input)double -> float -> double -> half
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)double -> half
    auto ref_tv = castOp(DataType::Half, tv0);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Double)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Float, tv0);
    tv = castOp(DataType::Half, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Half, tv);
    // (input)double -> float -> half -> float -> double -> half
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)double -> half
    auto ref_tv = castOp(DataType::Half, tv0);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    tv = castOp(DataType::Float, tv);
    // (input)float -> double -> float
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // TODO: should I have copied the tensor to avoid an aliased output?!
    // simplified as (input)
    ASSERT_TRUE(tv0->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    tv = castOp(DataType::Half, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Float, tv);
    // (input)float -> double -> half -> float -> double -> float -> double ->
    // float -> double -> float
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)float -> half -> float
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::Float, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    tv = castOp(DataType::Half, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::BFloat16, tv);
    tv = castOp(DataType::Float, tv);
    // (input)float -> double -> half -> float -> bfloat16 -> float
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)float -> half -> bfloat16 -> float
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::BFloat16, ref_tv);
    ref_tv = castOp(DataType::Float, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Int32)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    tv = castOp(DataType::ComplexDouble, tv);
    tv = castOp(DataType::Int, tv);
    tv = castOp(DataType::BFloat16, tv);
    tv = castOp(DataType::Float, tv);
    tv = castOp(DataType::Double, tv);
    // (input)int32 -> double -> complex double -> int64 -> bfloat16 -> float ->
    // double
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)int32 -> bfloat16 -> double
    auto ref_tv = castOp(DataType::BFloat16, tv0);
    ref_tv = castOp(DataType::Double, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    fusion->addOutput(tv);
    tv = castOp(DataType::Half, tv);
    tv = castOp(DataType::Double, tv);
    tv = castOp(DataType::Float, tv);
    // (input)float -> double(output0) -> half -> double -> float(output1)
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)float -> double(output0) -> half -> float(output1)
    auto ref_tv = castOp(DataType::Double, tv0);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
    ref_tv = castOp(DataType::Half, ref_tv);
    ref_tv = castOp(DataType::Float, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[1]));
  }

  {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    auto tv0 = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Half, tv0);
    tv = castOp(DataType::BFloat16, tv);
    tv = castOp(DataType::Half, tv);
    // (input)float -> half -> bfloat16 -> half
    fusion->addOutput(tv);
    OptimizationPass<PreSegmenter>::runPass(fusion.get());
    // simplified as (input)float -> half -> bfloat -> half
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::BFloat16, ref_tv);
    ref_tv = castOp(DataType::Half, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }
}

// Test that we remove empty output branch before segmentation
TEST_F(PresegTest, FusionRemoveEmptyOutput) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({0, 3});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({0, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at0};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  // In the FusionKernelRuntime, before segmentation a number of optimization
  // passes are performed. One of those is RemoveEmptyPass, which should replace
  // the empty output tv1 with a new TensorView defined by `full({0, 3})` in
  // this case.
  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);
  EXPECT_NE(preseg_fusion->outputs()[0], tv1);
  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, {at0}, __LINE__, __FILE__);
}

// Test that we replace empty reduction with full
TEST_F(PresegTest, FusionRemoveEmptyReduction) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({0, 3});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({0, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at0};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);
  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

// In this test, a reduction over a non-empty axis occurs first, followed by a
// reduction over the remaining empty axis. The output is actually not empty,
// even though the first reduction results in an empty tensor.
TEST_F(PresegTest, FusionRemoveEmptyReductionWithNonReduction) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({0, 3, 2});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({0, 3, 2}, options);
  std::vector<c10::IValue> aten_inputs = {at0};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);
  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we replace empty Welford with full
TEST_F(PresegTest, FusionRemoveEmptyWelford) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({0, 3});
  fusion.addInput(tv0);
  auto w = Welford(tv0, {0});
  fusion.addOutput(w.avg);
  auto var = div(w.var_sum, fusion_ptr->zeroVal(DataType::Float));
  fusion.addOutput(var);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({0, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at0};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 2);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  EXPECT_NE(var->definition(), nullptr);
  EXPECT_TRUE(var->definition()->isA<BinaryOp>());
  // We divide in the fusion to normalize the variance, so here we have to peel
  // that back
  auto var_sum = var->definition()->inputs()[0]->as<TensorView>();
  EXPECT_TRUE(var_sum->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(
      preseg_fusion,
      outputs,
      aten_inputs,
      {at::mean(at0, 0), at::var(at0, 0)},
      __LINE__,
      __FILE__);
}

// Test that we replace empty tensors in cat properly
TEST_F(PresegTest, FusionRemoveEmptyCat) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({0, 3});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({2, 3});
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor({4, 3});
  fusion.addInput(tv2);

  // equivalent to cat({tv1, tv2}, 0)
  auto tv3 = cat({tv0, tv1, tv2}, 0);
  fusion.addOutput(tv3);
  // set(tv1)
  auto tv4 = cat({tv0, tv1}, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({0, 3}, options);
  at::Tensor at1 = at::randn({2, 3}, options);
  at::Tensor at2 = at::randn({4, 3}, options);
  std::vector<c10::IValue> aten_inputs = {at0, at1, at2};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 2);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<CatOp>());
  EXPECT_EQ(preseg_fusion->outputs()[0]->definition()->inputs().size(), 2);

  EXPECT_NE(preseg_fusion->outputs()[1]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[1]->definition()->isA<LoadStoreOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we replace empty tensors in pad properly
TEST_F(PresegTest, FusionRemoveEmptyPad) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());
  // Concrete tensor with zero for one extent, so that we can prove the output
  // is empty
  auto tv0 = makeConcreteTensor({3, 0});
  fusion.addInput(tv0);

  // Use a non-zero pad value to verify that it is used in the rewritten fill
  auto pad_val = IrBuilder::create<Val>(3.14, DataType::Float);

  // equivalent to full({3, 2}, pad_val, DataType::Float)
  auto tv1 = pad(tv0, {fusion.oneVal(), fusion.oneVal()}, pad_val);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({3, 0}, options);
  std::vector<c10::IValue> aten_inputs = {at0};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  auto rewritten_def = preseg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(rewritten_def->isA<FullOp>());
  EXPECT_TRUE(rewritten_def->as<FullOp>()->getFillValue()->sameAs(pad_val));

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

// Test that we replace empty tensors in matmuls properly
TEST_F(PresegTest, FusionRemoveEmptyMatmul) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 0}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({0, 8}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  // [M, K, N]
  auto tv1b = broadcast(tv1, {true, false, false});
  // [M, N, K]
  auto tv1t = transpose(tv1b, 1, 2);

  auto tv2 = fusedMultiplySum(tv0b, tv1t, {2});
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor at0 = at::randn({16, 0}, options);
  at::Tensor at1 = at::randn({0, 8}, options);
  std::vector<c10::IValue> aten_inputs = {at0, at1};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  auto rewritten_def = preseg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(rewritten_def->isA<FullOp>());
  EXPECT_EQ(rewritten_def->as<FullOp>()->getFillValue()->evaluate(), 0.0);

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(PresegTest, ReplaceOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigTensor(1);
  fusion->addInput(x);
  fusion->addOutput(x);

  TensorView* y = add(x, x);
  fusion->replaceOutput(x, y);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({10}, at::device(at::kCUDA));
  at::Tensor out_tensor = executor_cache.runFusionWithInputs({in_tensor})[0];

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(PresegTest, ExtentSubstitution) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const std::vector<int64_t> input_shape = {128};
  const std::vector<int64_t> group_shape = {32, 4};
  auto tv0 = makeContigTensor(input_shape.size());
  auto tv1 = makeContigTensor(input_shape.size());
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = reshape(tv0, input_shape, group_shape);
  auto tv3 = reshape(tv1, input_shape, group_shape);
  auto tv4 = add(tv2, tv3);
  fusion->addOutput(tv4);

  OptimizationPass<PreSegmenter>::runPass(fusion.get());
  // two inputs should be same after ExactMappedExtentSubstitutionPass in
  // OptimizationPass
  const auto& inputs = fusion.get()->inputs();
  TensorView* input1 = dynamic_cast<TensorView*>(inputs.at(0));
  TensorView* input2 = dynamic_cast<TensorView*>(inputs.at(1));
  auto extend1 = input1->getLogicalDomain().at(0)->extent();
  auto extend2 = input2->getLogicalDomain().at(0)->extent();
  EXPECT_EQ(extend1, extend2);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  auto t1 = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PresegTest, DisjointSetsOfExtents) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  auto lds = tv1->getLogicalDomain();
  std::vector<Val*> shape{lds.at(1)->extent(), lds.at(0)->extent()};
  auto tv2 = TensorViewBuilder()
                 .ndims(2)
                 .shape(shape)
                 .contiguity({true, true})
                 .dtype(DataType::Float)
                 .build();

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = add(tv0, tv1);
  auto tv4 = add(tv2, tv3);
  fusion->addOutput(tv4);

  // This fusion has disjoint sets of Ids
  // disjoint sets of ids{
  //   { iS8{i4}; iS4{i4}; iS6{i0}; iS0{i0}; iS2{i3} }
  //   { iS9{i3}; iS5{i3}; iS7{i2}; iS1{i2}; iS3{i4} }
  // }
  // ExactMappedExtentSubstitutionPass generates disjoint sets of extents
  // disjoint sets of extents{
  //   { i4; i0; i3; i2 }
  // }
  // It works as follows:
  // { iS8{i4}; iS4{i4}; iS6{i0}; iS0{i0}; iS2{i3} } --> {i4; i0; i3}
  // then processing {iS9{i3}; iS5{i3}; iS7{i2}; iS1{i2}; iS3{i4}},
  // it notices the existence of {i3} in disjoint sets of extents and will
  // directly add all the non-existing extents to the set that contains {i3}
  // this leads to the final disjoint sets of extents { i4; i0; i3; i2 }
  // After extent substitution, all extents are replaced with {i0}.
  OptimizationPass<PreSegmenter>::runPass(fusion.get());
  auto ref_extent = tv0->getLogicalDomain().at(0)->extent();
  for (auto tv : {tv0, tv1, tv2}) {
    for (auto id : tv->getLogicalDomain()) {
      EXPECT_EQ(id->extent(), ref_extent);
    }
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto t1 = at::randn({32, 32}, options);
  auto t2 = at::randn({32, 32}, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(PresegTest, DisjointSetsOfExtentsConcreteSymbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigConcreteTensor({32, 32});

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // id_sets: disjoint sets{
  //   { iS4{32}; iS0{i0}; iS2{32} }
  //   { iS5{32}; iS1{i2}; iS3{32} }
  // }

  // ExactMappedExtentSubstitutionPass generates disjoint sets of extents:

  // Extent sets: disjoint sets{
  //   { 32; i0 }
  //   { 32; i2 }
  // }

  OptimizationPass<PreSegmenter>::runPass(fusion.get());
  // all extents are consolidated to 32
  for (auto tv : {tv0, tv1}) {
    for (auto id : tv->getLogicalDomain()) {
      EXPECT_EQ(id->extent()->evaluate().as<int64_t>(), 32);
    }
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32, 32}, options);
  auto t1 = at::randn({32, 32}, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}
} // namespace nvfuser::preseg_passes
