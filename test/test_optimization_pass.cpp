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

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

namespace nvfuser::optimization {

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
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv = castOp(DataType::Double, tv0);
    tv = castOp(DataType::Float, tv);
    // (input)float -> double -> float
    fusion->addOutput(tv);
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
        fusion.get());
    // TODO: should I have copied the tensor to avoid an alised output?!
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
    // TODO: should I have copied the tensor to avoid an alised output?!
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
    // TODO: should I have copied the tensor to avoid an alised output?!
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
    // (input)float -> double -> half -> float -> bfloat16 -> float
    fusion->addOutput(tv);
    optimization::OptimizationPass<optimization::PreSegmenter>::runPass(
        fusion.get());
    // TODO: should I have copied the tensor to avoid an alised output?!
    // simplified as (input)float -> half -> bfloat16 -> float
    auto ref_tv = castOp(DataType::BFloat16, tv0);
    ref_tv = castOp(DataType::Double, ref_tv);
    ASSERT_TRUE(ref_tv->sameAs(fusion->outputs()[0]));
  }
}

}
