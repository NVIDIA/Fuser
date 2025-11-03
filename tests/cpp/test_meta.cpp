// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <fusion.h>
#include <gtest/gtest.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using MetaTest = NVFuserTest;

TEST_F(MetaTest, ScanRowMajor) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // Build a simple scan fusion: out = cumsum(in, dim=1)
  auto tv0 = makeContigConcreteTensor({4, 8}, DataType::Float);
  fusion_ptr->addInput(tv0);
  auto tv_out = scan(tv0, /*dim=*/1, BinaryOpType::Add);
  fusion_ptr->addOutput(tv_out);

  // Create a real input to also get a concrete reference layout
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion_ptr->inputs().at(0), input);
  auto real_out =
      ee_cuda.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  // Meta path via ExpressionEvaluator
  ExpressionEvaluator ee_meta;
  auto meta_in = at::empty_strided(
      input.sizes(), input.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion_ptr->inputs().at(0), meta_in);
  auto meta_out =
      ee_meta.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

TEST_F(MetaTest, ScanColMajor) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // Build a simple scan fusion: out = cumsum(in, dim=0)
  auto tv0 = makeConcreteTensor({4, 8}, DataType::Float);
  fusion_ptr->addInput(tv0);
  auto tv_out = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion_ptr->addOutput(tv_out);

  // Create a real input to also get a concrete reference layout
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 4}, options).t();

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion_ptr->inputs().at(0), input);
  auto real_out =
      ee_cuda.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_in = at::empty_strided(
      input.sizes(), input.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion_ptr->inputs().at(0), meta_in);
  auto meta_out =
      ee_meta.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

// Test GroupedMmaOp with mat1=[m, k], mat2=[k, n] -> out=[g, m, n]
TEST_F(MetaTest, GroupedMma2D2D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // Build a grouped mm fusion: out = grouped_mm(mat1, mat2, offsets)
  // mat1: [m, k] = [4, 8]
  // mat2: [k, n] = [8, 6]
  // offsets: [g] = [3] with values [2, 4, 6]
  // output: [g, m, n] = [3, 4, 6]
  auto mat1 = makeContigConcreteTensor({4, 8}, DataType::Float);
  auto mat2 = makeContigConcreteTensor({8, 6}, DataType::Float);
  auto offsets = makeContigConcreteTensor({3}, DataType::Int);
  fusion_ptr->addInput(mat1);
  fusion_ptr->addInput(mat2);
  fusion_ptr->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion_ptr->addOutput(result.tv);

  // Create real inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor mat1_input = at::randn({4, 8}, options);
  at::Tensor mat2_input = at::randn({8, 6}, options);
  at::Tensor offsets_input =
      at::tensor({2, 4, 6}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion_ptr->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion_ptr->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion_ptr->inputs().at(2), offsets_input);
  auto real_out =
      ee_cuda.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_mat1 = at::empty_strided(
      mat1_input.sizes(), mat1_input.strides(), options.device(at::kMeta));
  auto meta_mat2 = at::empty_strided(
      mat2_input.sizes(), mat2_input.strides(), options.device(at::kMeta));
  auto meta_offsets = at::empty_strided(
      offsets_input.sizes(),
      offsets_input.strides(),
      at::TensorOptions().dtype(at::kInt).device(at::kMeta));
  ee_meta.bind(fusion_ptr->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion_ptr->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion_ptr->inputs().at(2), meta_offsets);
  auto meta_out =
      ee_meta.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

// Test GroupedMmaOp with mat1=[g, m, k], mat2=[k, n] -> out=[m, n]
TEST_F(MetaTest, GroupedMma3D2D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // Build a grouped mm fusion: out = grouped_mm(mat1, mat2, offsets)
  // mat1: [g, m, k] = [3, 4, 8]
  // mat2: [k, n] = [8, 6]
  // offsets: [g] = [3] with values [2, 4, 6]
  // output: [m, n] = [4, 6]
  auto mat1 = makeContigConcreteTensor({3, 4, 8}, DataType::Float);
  auto mat2 = makeContigConcreteTensor({8, 6}, DataType::Float);
  auto offsets = makeContigConcreteTensor({3}, DataType::Int);
  fusion_ptr->addInput(mat1);
  fusion_ptr->addInput(mat2);
  fusion_ptr->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion_ptr->addOutput(result.tv);

  // Create real inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor mat1_input = at::randn({3, 4, 8}, options);
  at::Tensor mat2_input = at::randn({8, 6}, options);
  at::Tensor offsets_input =
      at::tensor({2, 4, 6}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion_ptr->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion_ptr->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion_ptr->inputs().at(2), offsets_input);
  auto real_out =
      ee_cuda.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_mat1 = at::empty_strided(
      mat1_input.sizes(), mat1_input.strides(), options.device(at::kMeta));
  auto meta_mat2 = at::empty_strided(
      mat2_input.sizes(), mat2_input.strides(), options.device(at::kMeta));
  auto meta_offsets = at::empty_strided(
      offsets_input.sizes(),
      offsets_input.strides(),
      at::TensorOptions().dtype(at::kInt).device(at::kMeta));
  ee_meta.bind(fusion_ptr->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion_ptr->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion_ptr->inputs().at(2), meta_offsets);
  auto meta_out =
      ee_meta.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

// Test GroupedMmaOp with mat1=[m, k], mat2=[g, k, n] -> out=[m, n]
TEST_F(MetaTest, GroupedMma2D3D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // Build a grouped mm fusion: out = grouped_mm(mat1, mat2, offsets)
  // mat1: [m, k] = [4, 8]
  // mat2: [g, k, n] = [3, 8, 6]
  // offsets: [g] = [3] with values [2, 4, 6] -> group sizes = [2, 2, 2]
  // output: [m, n] = [4, 6]
  auto mat1 = makeContigConcreteTensor({4, 8}, DataType::Float);
  auto mat2 = makeContigConcreteTensor({3, 8, 6}, DataType::Float);
  auto offsets = makeContigConcreteTensor({3}, DataType::Int);
  fusion_ptr->addInput(mat1);
  fusion_ptr->addInput(mat2);
  fusion_ptr->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion_ptr->addOutput(result.tv);

  // Create real inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor mat1_input = at::randn({4, 8}, options);
  at::Tensor mat2_input = at::randn({3, 8, 6}, options);
  at::Tensor offsets_input =
      at::tensor({2, 4, 6}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion_ptr->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion_ptr->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion_ptr->inputs().at(2), offsets_input);
  auto real_out =
      ee_cuda.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_mat1 = at::empty_strided(
      mat1_input.sizes(), mat1_input.strides(), options.device(at::kMeta));
  auto meta_mat2 = at::empty_strided(
      mat2_input.sizes(), mat2_input.strides(), options.device(at::kMeta));
  auto meta_offsets = at::empty_strided(
      offsets_input.sizes(),
      offsets_input.strides(),
      at::TensorOptions().dtype(at::kInt).device(at::kMeta));
  ee_meta.bind(fusion_ptr->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion_ptr->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion_ptr->inputs().at(2), meta_offsets);
  auto meta_out =
      ee_meta.evaluate(fusion_ptr->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

} // namespace nvfuser
