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

TEST_F(NVFuserTest, FusionTestOptimizationPassFlag_CUDA) {
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

TEST_F(NVFuserTest, FusionCyclicGraph_CUDA) {
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
            ::testing::HasSubstr("cycle detected")));
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
            ::testing::HasSubstr("cycle detected")));

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
TEST_F(NVFuserTest, FusionTestCastOptimization_CUDA) {
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
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
    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        fusion.get());
    // simplified as (input)float -> half -> bfloat -> half
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::BFloat16, ref_tv);
    ref_tv = castOp(DataType::Half, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }
}

// Test that we remove empty output branch before segmentation
TEST_F(NVFuserTest, FusionRemoveEmptyOutput_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyReduction_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyReductionWithNonReduction_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyWelford_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyCat_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyPad_CUDA) {
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
TEST_F(NVFuserTest, FusionRemoveEmptyMatmul_CUDA) {
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

TEST_F(NVFuserTest, FusionFactorAmax_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* tv0 = makeContigTensor(2, DataType::Float);
  TensorView* tv1 = makeContigTensor(0, DataType::Float);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Partial Reduction
  TensorView* tv2 = sum(tv0, {1}, /*keepdim=*/false);
  TensorView* tv3 = broadcast(tv2, {false, true});
  TensorView* tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Full Amax Reduction
  TensorView* tv5 = abs(tv4);
  TensorView* tv6 = max(tv5, {0, 1}, /*keepdim=*/false);

  // Amax Aliased Output
  fusion.aliasOutputToInput(tv6, tv1, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor x = at::randn({32, 1228}, options);
  at::Tensor fp8_amax_history = at::zeros({}, options);
  std::vector<c10::IValue> aten_inputs = {x, fp8_amax_history};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), 2);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* first_fusion = segments.front().get();
  EXPECT_EQ(first_fusion->outputs().size(), 2);
  Val* last_output = first_fusion->outputs().back();

  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, 1);
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), 2);

  // Aten reference
  at::Tensor at_t1 = at::sum(x, {1}, /*keepdim=*/true);
  at::Tensor at_t2 = x + at_t1;
  at::Tensor at_t3 = at::abs(at_t2);
  at::Tensor at_t4 = at::max(at_t3);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  testValidate(
      preseg_fusion, outputs, aten_inputs, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue2258_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* x = makeContigTensor(2, DataType::BFloat16);
  TensorView* bias = makeContigTensor(1, DataType::BFloat16);
  TensorView* residual = makeContigTensor(2, DataType::BFloat16);
  TensorView* ln_weight = makeContigTensor(1, DataType::BFloat16);
  TensorView* ln_bias = makeContigTensor(1, DataType::BFloat16);
  TensorView* fp8_scale = makeContigTensor(0, DataType::Float);
  TensorView* fp8_amax = makeContigTensor(0, DataType::Float);
  Val* scalar_eps = IrBuilder::create<Val>(DataType::Double);

  fusion.addInput(x);
  fusion.addInput(bias);
  fusion.addInput(residual);
  fusion.addInput(ln_weight);
  fusion.addInput(ln_bias);
  fusion.addInput(fp8_scale);
  fusion.addInput(fp8_amax);
  fusion.addInput(scalar_eps);

  TensorView* x_cast = castOp(DataType::Float, x);
  TensorView* bias_cast = castOp(DataType::Float, bias);
  TensorView* residual_cast = castOp(DataType::Float, residual);
  TensorView* ln_weight_cast = castOp(DataType::Float, ln_weight);
  TensorView* ln_bias_cast = castOp(DataType::Float, ln_bias);

  TensorView* t0 = broadcast(bias_cast, {true, false});
  TensorView* t1 = add(x_cast, t0);
  TensorView* t2 = add(t1, residual_cast);
  fusion.addOutput(t2);

  TensorView* gamma_centered_ln_weight =
      add(ln_weight_cast, IrBuilder::create<Val>(1.0f, DataType::Float));

  ForwardNormResult t3 = layer_norm(
      t2,
      /*kNormShapeNumDims=*/1,
      gamma_centered_ln_weight,
      ln_bias_cast,
      scalar_eps);
  TensorView* t4 = mul(t3.output, fp8_scale);
  TensorView* t5 = castOp(DataType::BFloat16, t4);
  fusion.addOutput(t5);

  // Full Amax Reduction
  TensorView* t6 = abs(t3.output);
  TensorView* t7 = max(t6, {0, 1}, /*keepdim=*/false);

  // Amax Aliased Output
  fusion.aliasOutputToInput(t7, fp8_amax, AllocationType::ReuseBuffer);

  const float eps = 1e-5;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn({32, 1228}, options);
  at::Tensor at_bias = at::randn({1228}, options);
  at::Tensor at_residual = at::randn({32, 1228}, options);
  at::Tensor at_ln_weight = at::randn({1228}, options);
  at::Tensor at_ln_bias = at::randn({1228}, options);

  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_fp8_scale = at::randn({}, fp32_options);
  at::Tensor at_fp8_amax = at::zeros({}, fp32_options);
  std::vector<c10::IValue> aten_inputs = {
      at_x,
      at_bias,
      at_residual,
      at_ln_weight,
      at_ln_bias,
      at_fp8_scale,
      at_fp8_amax,
      eps};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), 2);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* first_fusion = segments.front().get();
  EXPECT_EQ(first_fusion->outputs().size(), 3);

  Val* last_output = first_fusion->outputs().back();
  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, 1);
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), 2);

  // Aten reference
  at::Tensor at_x_cast = at_x.to(at::kFloat);
  at::Tensor at_bias_cast = at_bias.to(at::kFloat);
  at::Tensor at_residual_cast = at_residual.to(at::kFloat);
  at::Tensor at_ln_weight_cast = at_ln_weight.to(at::kFloat);
  at::Tensor at_ln_bias_cast = at_ln_bias.to(at::kFloat);
  at::Tensor at_t1 = at_x_cast + at_bias_cast.unsqueeze(0);
  at::Tensor at_t2 = at_t1 + at_residual_cast;
  at::Tensor at_gamma_centered_ln_weight = at_ln_weight_cast + 1.0f;
  at::Tensor at_t3 = at::layer_norm(
      at_t2, {1228}, at_gamma_centered_ln_weight, at_ln_bias_cast, 1e-5);
  at::Tensor at_t4 = at_t3 * at_fp8_scale;
  at::Tensor at_t5 = at_t4.to(at::kBFloat16);
  at::Tensor at_t6 = at::abs(at_t3);
  at::Tensor at_t7 = at::max(at_t6);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  testValidate(
      preseg_fusion,
      outputs,
      aten_inputs,
      {at_t2, at_t5, at_t7},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionFactorAmaxHorizontalMultiplePartial_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* tv0 = makeContigTensor(2, DataType::Float);
  TensorView* tv1 = makeContigTensor(0, DataType::Float);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Add Inner and Outer Partial Reductions together
  TensorView* tv2 = sum(tv0, {1}, /*keepdim=*/true);
  TensorView* tv3 = sum(tv0, {0}, /*keepdim=*/false);
  TensorView* tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Full Amax Reduction
  TensorView* tv5 = abs(tv4);
  TensorView* tv6 = max(tv5, {0, 1}, /*keepdim=*/false);

  // Amax Aliased Output
  fusion.aliasOutputToInput(tv6, tv1, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor x = at::randn({32, 1228}, options);
  at::Tensor fp8_amax_history = at::zeros({}, options);
  std::vector<c10::IValue> aten_inputs = {x, fp8_amax_history};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Four segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), 4);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* third_fusion = segments.at(2).get();
  EXPECT_EQ(third_fusion->outputs().size(), 2);
  Val* last_output = third_fusion->outputs().back();

  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, 1);
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), 2);

  // Aten reference
  at::Tensor at_t1 = at::sum(x, {1}, /*keepdim=*/true);
  at::Tensor at_t2 = at::sum(x, {0}, /*keepdim=*/true);
  at::Tensor at_t3 = at_t1 + at_t2;
  at::Tensor at_t4 = at::abs(at_t3);
  at::Tensor at_t5 = at::max(at_t4);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  testValidate(
      preseg_fusion, outputs, aten_inputs, {at_t3, at_t5}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionFactorAmaxBroadcast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* tv0 = makeContigTensor(2, DataType::Float);
  TensorView* tv1 = makeContigTensor(2, DataType::Float);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Partial Reduction
  TensorView* tv2 = sum(tv0, {1}, /*keepdim=*/false);
  TensorView* tv3 = broadcast(tv2, {false, true});
  TensorView* tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Full Amax Reduction
  TensorView* tv5 = abs(tv4);
  TensorView* tv6 = max(tv5, {0, 1}, /*keepdim=*/true);

  // Amax Aliased Output
  fusion.aliasOutputToInput(tv6, tv1, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor x = at::randn({32, 1228}, options);
  at::Tensor fp8_amax_history = at::zeros({1, 1}, options);
  std::vector<c10::IValue> aten_inputs = {x, fp8_amax_history};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), 2);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* first_fusion = segments.front().get();
  EXPECT_EQ(first_fusion->outputs().size(), 2);
  Val* last_output = first_fusion->outputs().back();

  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, 1);
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), 2);

  // Aten reference
  at::Tensor at_t1 = at::sum(x, {1}, /*keepdim=*/true);
  at::Tensor at_t2 = x + at_t1;
  at::Tensor at_t3 = at::abs(at_t2);
  at::Tensor at_t4 = at::max(at_t3);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  testValidate(
      preseg_fusion, outputs, aten_inputs, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionFactorAmaxCast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* tv0 = makeContigTensor(2, DataType::Half);
  TensorView* tv1 = makeContigTensor(0, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Partial Reduction
  TensorView* tv0_cast = castOp(DataType::Float, tv0);
  TensorView* tv2 = sum(tv0_cast, {1}, /*keepdim=*/false);
  TensorView* tv3 = broadcast(tv2, {false, true});
  TensorView* tv4 = add(tv0_cast, tv3);
  TensorView* tv4_cast = castOp(DataType::Half, tv4);
  fusion.addOutput(tv4_cast);

  // Full Amax Reduction
  TensorView* tv5 = abs(tv4);
  TensorView* tv6 = max(tv5, {0, 1}, /*keepdim=*/false);
  TensorView* tv6_cast = castOp(DataType::Half, tv6);

  // Amax Aliased Output
  fusion.aliasOutputToInput(tv6_cast, tv1, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor x = at::randn({32, 1228}, options);
  at::Tensor fp8_amax_history = at::zeros({}, options);
  std::vector<c10::IValue> aten_inputs = {x, fp8_amax_history};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), 2);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* first_fusion = segments.front().get();
  EXPECT_EQ(first_fusion->outputs().size(), 2);
  Val* last_output = first_fusion->outputs().back();

  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, 1);
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), 2);

  // Aten reference
  at::Tensor x_cast = x.to(at::kFloat);
  at::Tensor at_t1 = at::sum(x_cast, {1}, /*keepdim=*/true);
  at::Tensor at_t2 = x_cast + at_t1;
  at::Tensor at_t2_cast = at_t2.to(at::kHalf);
  at::Tensor at_t3 = at::abs(at_t2);
  at::Tensor at_t4 = at::max(at_t3);
  at::Tensor at_t4_cast = at_t4.to(at::kHalf);

  auto preseg_fusion = runtime.fusionSegments()->completeFusion();
  testValidate(
      preseg_fusion,
      outputs,
      aten_inputs,
      {at_t2_cast, at_t4_cast},
      __LINE__,
      __FILE__);
}

void checkAmaxSegmentation(FusionKernelRuntime& runtime,
	                   int64_t number_of_segments,
	                   int64_t fusion_index,
	                   int64_t number_of_outputs_in_fusion,
	                   int64_t number_of_iterdomains,
	                   int64_t expected_number_of_reduction_axes) {
  // Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  std::vector<std::unique_ptr<Fusion>> segments = runtime.getFusionSegments();
  EXPECT_EQ(segments.size(), number_of_segments);

  // Expect partial reduction for amax to be saved as output of first fusion
  Fusion* selected_fusion = segments.at(fusion_index).get();

  EXPECT_EQ(selected_fusion->outputs().size(), number_of_outputs_in_fusion);
  Val* last_output = selected_fusion->outputs().back();

  EXPECT_TRUE(last_output->isA<TensorView>());
  TensorView* partial_amax = last_output->as<TensorView>();

  EXPECT_TRUE(
      partial_amax->definition()->isA<ReductionOp>() &&
      partial_amax->definition()->as<ReductionOp>()->getReductionOpType() ==
          BinaryOpType::Max);

  // Check that there is a single reduction axis
  std::vector<IterDomain*> logical_domain = partial_amax->getLogicalDomain();
  EXPECT_EQ(partial_amax->getLogicalDomain().size(), number_of_iterdomains);

  int64_t num_reduction_axes = std::count_if(
      partial_amax->getLogicalDomain().begin(),
      partial_amax->getLogicalDomain().end(),
      [](IterDomain* id) { return id->isReduction(); });
  EXPECT_EQ(num_reduction_axes, expected_number_of_reduction_axes);
}

TEST_F(NVFuserTest, FusionFactorAmaxBroadcastCast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  TensorView* tv0 = makeContigTensor(2, DataType::Half);
  TensorView* tv1 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Partial Reduction
  TensorView* tv0_cast = castOp(DataType::Float, tv0);
  TensorView* tv2 = sum(tv0_cast, {1}, /*keepdim=*/false);
  TensorView* tv3 = broadcast(tv2, {false, true});
  TensorView* tv4 = add(tv0_cast, tv3);
  TensorView* tv4_cast = castOp(DataType::Half, tv4);
  fusion.addOutput(tv4_cast);

  // Full Amax Reduction
  TensorView* tv5 = abs(tv4);
  TensorView* tv6 = max(tv5, {0, 1}, /*keepdim=*/true);
  TensorView* tv6_cast = castOp(DataType::Half, tv6);

  // Amax Aliased Output
  fusion.aliasOutputToInput(tv6_cast, tv1, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor x = at::randn({32, 1228}, options);
  at::Tensor fp8_amax_history = at::zeros({1, 1}, options);
  std::vector<c10::IValue> aten_inputs = {x, fp8_amax_history};

  auto args = KernelArgumentHolder::createKernelArgumentHolder(aten_inputs);
  FusionKernelRuntime runtime(std::move(fusion_ptr), args);
  runtime.compileFusionParallel(args);
  auto outputs = runtime.runWithInputs(args);

  // Expected result of factorAmaxReduction pass:
  // * Two segments are created because a partial reduction and a full reduction
  // cannot be in the same fusion.
  // * Expect partial reduction for amax to be saved as last output of first
  // fusion
  // The partial amax reduction has a single reduction axis.
  checkAmaxSegmentation(runtime,
	                /*number_of_segments=*/2,
	                /*fusion_index=*/0,
	                /*number_of_outputs_in_fusion=*/2,
	                /*number_of_iterdomains=*/2,
	                /*expected_number_of_reduction_axes=*/1);

  // Aten reference
  at::Tensor x_cast = x.to(at::kFloat);
  at::Tensor at_t1 = at::sum(x_cast, {1}, /*keepdim=*/true);
  at::Tensor at_t2 = x_cast + at_t1;
  at::Tensor at_t3 = at::abs(at_t2);
  at::Tensor at_t4 = at::max(at_t3);
  at::Tensor at_t2_cast = at_t2.to(at::kHalf);
  at::Tensor at_t4_cast = at_t4.to(at::kHalf);

  testValidate(
      runtime.fusionSegments()->completeFusion(),
      outputs,
      aten_inputs,
      {at_t2_cast, at_t4_cast},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser::preseg_passes
