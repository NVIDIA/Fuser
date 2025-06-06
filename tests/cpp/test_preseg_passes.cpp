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
#include <preseg_passes/consecutive_cast.h>
#include <preseg_passes/move_gather.h>
#include <preseg_passes/optimization_pass.h>
#include <preseg_passes/pre_segmenter.h>
#include <preseg_passes/reuse_expensive_computation_results.h>
#include <preseg_passes/translate_no_reduction_matmul_to_mul_squeeze.h>
#include <preseg_passes/translate_repeat_to_expand.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser::preseg_passes {

using testing::ElementsAre;

using PresegTest = NVFuserTest;

TEST_F(PresegTest, FusionTestOptimizationPassFlag) {
  class DerivedPass : public OptimizationPass<DerivedPass> {
    friend class OptimizationPass<DerivedPass>;

   protected:
    static void runPass(Fusion* fusion) {
      throw std::runtime_error("running DerivedPass");
    };
    static constexpr std::string_view name() {
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
    // simplified as (input)float -> half -> bfloat16 -> half
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
  at::Tensor t0 = at::randn({0, 3}, options);

  KernelArgumentHolder args({t0});
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

  testValidate(preseg_fusion, outputs, {t0}, {t0}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({0, 3}, options);

  KernelArgumentHolder args({t0});
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);
  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, {t0}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({0, 3, 2}, options);

  KernelArgumentHolder args({t0});
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);
  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  EXPECT_TRUE(preseg_fusion->outputs()[0]->definition()->isA<FullOp>());

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, {t0}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({0, 3}, options);

  KernelArgumentHolder args({t0});
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
      {t0},
      {at::mean(t0, 0), at::var(t0, 0)},
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
  at::Tensor t0 = at::randn({0, 3}, options);
  at::Tensor t1 = at::randn({2, 3}, options);
  at::Tensor t2 = at::randn({4, 3}, options);

  KernelArgumentHolder args({t0, t1, t2});
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

  testValidate(preseg_fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({3, 0}, options);

  KernelArgumentHolder args({t0});
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  auto rewritten_def = preseg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(rewritten_def->isA<FullOp>());
  EXPECT_TRUE(rewritten_def->as<FullOp>()->getFillValue()->sameAs(pad_val));

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, {t0}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({16, 0}, options);
  at::Tensor t1 = at::randn({0, 8}, options);

  KernelArgumentHolder args({t0, t1});
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  EXPECT_EQ(preseg_fusion->outputs().size(), 1);

  EXPECT_NE(preseg_fusion->outputs()[0]->definition(), nullptr);
  auto rewritten_def = preseg_fusion->outputs()[0]->definition();
  EXPECT_TRUE(rewritten_def->isA<FullOp>());
  EXPECT_EQ(rewritten_def->as<FullOp>()->getFillValue()->evaluate(), 0.0);

  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  testValidate(preseg_fusion, outputs, {t0, t1}, __LINE__, __FILE__);
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
  auto cg_outputs = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), cg_outputs, {in_tensor}, __LINE__, __FILE__);
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

// Trivial repeat pattern
TEST_F(PresegTest, TranslateRepeatToExpand1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32});
  fusion.addInput(tv0);

  auto tv1 = cat({tv0, tv0}, -1);
  fusion.addOutput(tv1);

  {
    // Make sure pad and cat no longer exist
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
}

// Consecutive repetitions with the same IDs
TEST_F(PresegTest, TranslateRepeatToExpand2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32});
  fusion.addInput(tv0);

  auto tv1 = cat({tv0, tv0}, -1);
  auto tv2 = cat({tv1, tv1}, -1);

  fusion.addOutput(tv2);

  {
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
}

// Consecutive repetitions with different IDs
TEST_F(PresegTest, TranslateRepeatToExpand3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8});
  fusion.addInput(tv0);

  auto tv1 = cat({tv0, tv0}, 1);
  auto tv2 = cat({tv1, tv1}, 0);

  fusion.addOutput(tv2);

  {
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
}

// Repeat the same ID of the same tensor multiple times. While the
// repetitions are the same, there's nothing to allow the output IDs
// to be mapped, so the translated fusion will be segmented. This is a
// downside compared to the original fusion, where all IDs are
// connected, so it's relatively straightforward to fuse them together
// without segmentation.
TEST_F(PresegTest, TranslateRepeatToExpand4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Consecutive repetitions with the same IDs
  auto tv1 = cat({tv0, tv0}, 1);
  auto tv2 = cat({tv0, tv0}, 1);

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  {
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be segmented to two pointwise kernels
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  const auto& heuristic_list = runtime->schedulerHeuristics()->heuristicsList();
  ASSERT_EQ(heuristic_list.size(), 2);
  EXPECT_EQ(heuristic_list.at(0)->scheduler_type, SchedulerType::PointWise);
  EXPECT_EQ(heuristic_list.at(1)->scheduler_type, SchedulerType::PointWise);
}

// Repeating more than two times
TEST_F(PresegTest, TranslateRepeatToExpand5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32});
  fusion.addInput(tv0);

  auto tv1 = cat({tv0, tv0, tv0, tv0}, -1);
  fusion.addOutput(tv1);

  {
    // Make sure pad and cat no longer exist
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
}

// Repeating a broadcast ID. Repro of
// https://github.com/NVIDIA/Fuser/issues/3682.
TEST_F(PresegTest, TranslateRepeatToExpand6) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 1});
  fusion.addInput(tv0);

  auto tv1 = cat({tv0, tv0}, -1);
  fusion.addOutput(tv1);

  {
    // Make sure pad and cat no longer exist
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateRepeatToExpand>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isOneOf<PadOp, CatOp>(); }),
        new_exprs.end());
    // RepeatOp should be used
    EXPECT_NE(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isA<RepeatOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({32, 1}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      ElementsAre(HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp0) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 4});
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = reshape(tv1, {2, 4}, {8});
  auto tv3 = castOp(DataType::Float, tv2);
  fusion.addOutput(tv3);

  {
    // Make sure cast no longer exists
    Fusion fusion_copy = fusion;
    OptimizationPass<ConsecutiveCastPass>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) {
              return new_expr->isA<UnaryOp>() &&
                  new_expr->as<UnaryOp>()->getUnaryOpType() ==
                  UnaryOpType::Cast;
            }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 4});
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = reshape(tv1, {2, 4}, {8});
  fusion.addOutput(tv2);
  // tv2 is fusion output, we should not replay the reshape operation after the
  // cast op
  auto tv3 = castOp(DataType::Float, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  bool is_segmented =
      executor_cache.getMostRecentKernelRuntime()->isSegmented();
  NVF_CHECK(!is_segmented, "Fusion should not be segmented");

  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 4});
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = reshape(tv1, {2, 4}, {8});
  // tv2 has another use in relu, we should not replay the reshape operation
  // after the cast op
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = relu(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  bool is_segmented =
      executor_cache.getMostRecentKernelRuntime()->isSegmented();
  NVF_CHECK(!is_segmented, "Fusion should not be segmented");

  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 4});
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = sum(tv1, {0});
  auto tv3 = reshape(tv2, {4}, {2, 2});
  auto tv4 = castOp(DataType::Float, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = reshape(tv1, {2, 3, 4}, {2, 3, 2, 2});
  auto tv3 = castOp(DataType::Float, tv2);
  std::vector<IterDomain*> tv3_nhwc = {
      tv3->axis(0), tv3->axis(2), tv3->axis(3), tv3->axis(1)};
  tv3->setAllocationDomain(tv3_nhwc, true);
  fusion.addOutput(tv3);

  {
    // Make sure cast no longer exists
    Fusion fusion_copy = fusion;
    OptimizationPass<ConsecutiveCastPass>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) {
              return new_expr->isA<UnaryOp>() &&
                  new_expr->as<UnaryOp>()->getUnaryOpType() ==
                  UnaryOpType::Cast;
            }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  ASSERT_TRUE(outputs[0].as<at::Tensor>().is_contiguous(
      at::MemoryFormat::ChannelsLast));
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp5) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  // multiple segements with cast should be merged
  auto tv1 = castOp(DataType::Double, tv0);
  auto tv2 = reshape(tv1, {2, 3, 4}, {2, 3, 2, 2});
  auto tv3 = castOp(DataType::Float, tv2);
  auto tv4 = reshape(tv3, {2, 3, 2, 2}, {6, 2, 2});
  auto tv5 = castOp(DataType::Half, tv4);
  fusion.addOutput(tv5);

  {
    // Make sure we merge all cast together
    Fusion fusion_copy = fusion;
    OptimizationPass<ConsecutiveCastPass>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::count_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) {
              return new_expr->isA<UnaryOp>() &&
                  new_expr->as<UnaryOp>()->getUnaryOpType() ==
                  UnaryOpType::Cast;
            }),
        1);
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PresegTest, FusionTestCastOptimizationMetaOp6) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 4}, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = reshape(tv1, {2, 3, 4}, {6, 4});
  // casting to higher precision dtype, we shouldn't propagate this cast to
  // producer, since it would result in increase in intermediate buffer size.
  auto tv3 = castOp(DataType::Double, tv2);
  fusion.addOutput(tv3);

  {
    // Make sure we merge all cast together
    Fusion fusion_copy = fusion;
    OptimizationPass<ConsecutiveCastPass>::runPass(&fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::count_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) {
              return new_expr->isA<UnaryOp>() &&
                  new_expr->as<UnaryOp>()->getUnaryOpType() ==
                  UnaryOpType::Cast;
            }),
        2);
    auto expr_iter =
        std::find_if(new_exprs.begin(), new_exprs.end(), [](Expr* new_expr) {
          return new_expr->isA<ViewOp>();
        });
    EXPECT_TRUE(
        expr_iter != new_exprs.end() &&
        (*(*expr_iter)->input(0)->getDataType() == DataType::Float));
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf);
  auto t0 = at::randn({2, 3, 4}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

struct MatmulInputShape {
  std::vector<int64_t> shape_a;
  std::vector<int64_t> shape_b;
  std::string toString() const {
    std::stringstream ss;
    ss << toDelimitedString(shape_a, "x") << "_"
       << toDelimitedString(shape_b, "x");
    return ss.str();
  }
};

TEST_F(PresegTest, MoveGatherOverCast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = castOp(DataType::Float, tv0);
  auto s3 = IrBuilder::create<Val>(-100.0, DataType::Int);
  auto s4 = IrBuilder::create<Val>(0, DataType::Int);

  auto tv3 = ne(tv1, s3);
  auto tv4 = where(tv3, tv1, s4);
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = takeAlongAxis(tv2, tv5, -1);
  auto tv7 = squeeze(tv6, {1});
  auto tv8 = max(tv2, {-1});
  auto tv9 = sub(tv7, tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto options_2 = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn({8, 16}, options);
  auto t1 = at::randint(0, 5, {8}, options_2);

  KernelArgumentHolder outputs_no_pass, outputs_with_pass;
  {
    auto new_fusion = std::make_unique<Fusion>(fusion);
    OptimizationPassGuard<MoveGatherPass> guard(false);
    FusionExecutorCache executor_cache(std::move(new_fusion));
    outputs_no_pass = executor_cache.runFusionWithInputs({t0, t1});
  }

  // Now run with the pass and check outputs match.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  outputs_with_pass = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(),
      outputs_with_pass,
      {t0, t1},
      {outputs_no_pass[0].as<at::Tensor>()},
      __LINE__,
      __FILE__);

  auto exprs = executor_cache.getMostRecentKernelRuntime()
                   ->fusionSegments()
                   ->completeFusion()
                   ->exprs();
  // Has take Along axis
  // lookupTv is now a fusion input and of type bfloat16
  auto gather_ops = ir_utils::filterByType<GatherOp>(exprs);
  EXPECT_EQ(gather_ops.size(), 1);
  auto gather_op = gather_ops.vector().at(0);
  auto lookupTv = gather_op->lookupTv();
  EXPECT_TRUE(lookupTv->isFusionInput());
  EXPECT_EQ(lookupTv->getDataType(), DataType::BFloat16);

  std::vector<UnaryOp*> all_cast_ops;
  std::vector<UnaryOp*> new_cast_ops;
  auto unary_ops = ir_utils::filterByType<UnaryOp>(exprs).vector();

  // We should have two cast ops
  std::copy_if(
      unary_ops.begin(),
      unary_ops.end(),
      std::back_inserter(all_cast_ops),
      [](UnaryOp* op) {
        std::cout << op->toString() << std::endl;
        return op->getUnaryOpType() == UnaryOpType::Cast;
      });
  EXPECT_EQ(all_cast_ops.size(), 2);

  // The cast op after gather op should be a new cast op
  // with a broadcast domain and dtype of float
  std::copy_if(
      unary_ops.begin(),
      unary_ops.end(),
      std::back_inserter(new_cast_ops),
      [gather_op](UnaryOp* op) {
        return op->getUnaryOpType() == UnaryOpType::Cast &&
            op->input(0) == gather_op->output(0) &&
            op->output(0)->getDataType() == DataType::Float &&
            op->output(0)->as<TensorView>()->hasBroadcast();
      });
  EXPECT_EQ(new_cast_ops.size(), 1);
}

TEST_F(PresegTest, MoveGatherOverSqueezeAndCast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion.addInput(tv1);

  tv0 = broadcast(tv0, {true, false, false});
  tv1 = broadcast(tv1, {true, false});

  auto tv2_f = castOp(DataType::Float, tv0);
  auto tv2 = squeeze(tv2_f, {0});
  auto tv3 = squeeze(tv1, {0});
  auto s3 = IrBuilder::create<Val>(-100.0, DataType::Int);
  auto s4 = IrBuilder::create<Val>(0, DataType::Int);

  auto tv4 = ne(tv3, s3);
  auto tv5 = where(tv4, tv3, s4);
  auto tv6 = broadcast(tv5, {false, true});
  auto tv7 = takeAlongAxis(tv2, tv6, -1);
  auto tv9 = squeeze(tv7, {1});

  auto tv10 = max(tv2_f, {-1});
  auto tv11 = sub(tv9, tv10);

  fusion.addOutput(tv11);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto options_2 = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randn({8, 16}, options);
  auto t1 = at::randint(0, 5, {8}, options_2);

  KernelArgumentHolder outputs_no_pass, outputs_with_pass;
  {
    auto new_fusion = std::make_unique<Fusion>(fusion);
    OptimizationPassGuard<MoveGatherPass> guard(false);
    FusionExecutorCache executor_cache(std::move(new_fusion));
    outputs_no_pass = executor_cache.runFusionWithInputs({t0, t1});
  }

  // Now run with the pass and check outputs match.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  outputs_with_pass = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(),
      outputs_with_pass,
      {t0, t1},
      {outputs_no_pass[0].as<at::Tensor>()},
      __LINE__,
      __FILE__);

  auto exprs = executor_cache.getMostRecentKernelRuntime()
                   ->fusionSegments()
                   ->completeFusion()
                   ->exprs();
  // Has take Along axis
  // lookupTv is now a fusion input and of type bfloat16
  auto gather_ops = ir_utils::filterByType<GatherOp>(exprs);
  EXPECT_EQ(gather_ops.size(), 1);
  auto gather_op = gather_ops.vector().at(0);
  auto lookupTv = gather_op->lookupTv();

  // The Def of lookUp should be the broadcast Op
  EXPECT_TRUE(lookupTv->definition()->isA<BroadcastOp>());

  auto squeeze_ops = ir_utils::filterByType<SqueezeOp>(exprs).vector();
  std::vector<SqueezeOp*> new_squeeze_ops;

  // There should still be two squeeze ops
  EXPECT_EQ(squeeze_ops.size(), 2);

  // The new squeeze op after gather op should be a new cast op
  // with a broadcast domain and dtype of float and after
  // the cast op
  std::copy_if(
      squeeze_ops.begin(),
      squeeze_ops.end(),
      std::back_inserter(new_squeeze_ops),
      [gather_op](SqueezeOp* op) {
        // take_along_axis->cast->squeeze
        return op->input(0)->definition()->input(0) == gather_op->output(0);
      });
  EXPECT_EQ(new_squeeze_ops.size(), 1);
}

using TranslateNoReductionMatmulTest =
    NVFuserFixtureParamTest<MatmulInputShape>;

INSTANTIATE_TEST_SUITE_P(
    ,
    TranslateNoReductionMatmulTest,
    testing::Values(
        MatmulInputShape{{8, 1}, {1, 16}},
        MatmulInputShape{{8, 1}, {1}},
        MatmulInputShape{{1}, {1, 8}},
        MatmulInputShape{{2, 3, 1}, {2, 1, 4}},
        MatmulInputShape{{2, 3, 4, 1}, {2, 3, 1, 5}},
        MatmulInputShape{{4, 1}, {2, 1, 5}},
        MatmulInputShape{{4, 1}, {2, 3, 1, 5}},
        MatmulInputShape{{1}, {2, 3, 1, 5}},
        MatmulInputShape{{3, 4, 1}, {1}},
        MatmulInputShape{{2, 3, 4, 1}, {1}}),
    [](const testing::TestParamInfo<MatmulInputShape>& info) {
      return info.param.toString();
    });

// Test the translation of MatmulOp when K=1
TEST_P(TranslateNoReductionMatmulTest, Test) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  const auto config = GetParam();

  auto tv0 = makeContigConcreteTensor(config.shape_a);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(config.shape_b);
  fusion.addInput(tv1);

  auto tv2 = matmul(tv0, tv1);
  fusion.addOutput(tv2);

  {
    // Make sure MatmulOp no longer exists
    Fusion fusion_copy = fusion;
    OptimizationPass<TranslateNoReductionMatmulToMulSqueeze>::runPass(
        &fusion_copy);
    auto new_exprs = fusion_copy.exprs();
    EXPECT_EQ(
        std::find_if(
            new_exprs.begin(),
            new_exprs.end(),
            [](Expr* new_expr) { return new_expr->isA<MatmulOp>(); }),
        new_exprs.end());
  }

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn(config.shape_a, options);
  auto t1 = at::randn(config.shape_b, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);

  // Should be scheduled as a pointwise kernel
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      ElementsAre(HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(PresegTest, ReuseExpensiveComputationResults) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({128});
  fusion.addInput(tv0);
  auto tv1 = exp(tv0);
  auto tv2 = exp(tv0);
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  OptimizationPass<ReuseExpensiveComputationResultsPass>::runPass(&fusion);

  // There is only one exp operation after the pass
  int unary_ops = 0;
  for (auto expr : fusion.exprs()) {
    if (expr->isA<UnaryOp>() &&
        expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Exp) {
      unary_ops++;
    }
  }
  EXPECT_EQ(unary_ops, 1);
}
} // namespace nvfuser::preseg_passes
