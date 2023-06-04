// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <optimization/optimization_pass.h>
#include <optimization/pre_segmenter.h>
#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser::optimization {

TEST_F(NVFuserTest, FusionTestOptimizationPassFlag_CUDA) {
  class DerivedPass : public OptimizationPass<DerivedPass> {
    friend class OptimizationPass<DerivedPass>;

   protected:
    static void runPass(Fusion* fusion) {
      throw std::runtime_error("running DerivedPass");
    };
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

    auto cycle = ir_utils::checkCycle(fusion.get());
    TORCH_CHECK(cycle.empty(), "no cycle should be detected in fusion");

    auto expr = tv2->definition();
    ir_utils::replaceValInExpr(expr, tv1, tv4); // manually creating a cycle
    TORCH_CHECK(cycle.size() == 6, "cycle should be detected in fusion");
    EXPECT_THAT([&]() {StmtSort::getStmtsBetween(fusion.get(), {}, fusion->outputs()); },
          ::testing::ThrowsMessage<c10::Error>(
              ::testing::HasSubstr("cycle detected")));
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
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
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
        fusion.get());
    // simplified as (input)float -> half -> bfloat -> half
    auto ref_tv = castOp(DataType::Half, tv0);
    ref_tv = castOp(DataType::BFloat16, ref_tv);
    ref_tv = castOp(DataType::Half, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }
}

} // namespace nvfuser::optimization
