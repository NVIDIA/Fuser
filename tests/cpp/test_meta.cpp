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

} // namespace nvfuser
