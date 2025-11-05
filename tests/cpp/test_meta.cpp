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
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Build a simple scan fusion: out = cumsum(in, dim=1)
  auto tv0 = makeContigConcreteTensor({4, 8}, DataType::Float);
  fusion->addInput(tv0);
  auto tv_out = scan(tv0, /*dim=*/1, BinaryOpType::Add);
  fusion->addOutput(tv_out);

  // Create a real input to also get a concrete reference layout
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  // Meta path via ExpressionEvaluator
  ExpressionEvaluator ee_meta;
  auto meta_in = at::empty_strided(
      input.sizes(), input.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion->inputs().at(0), meta_in);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

TEST_F(MetaTest, ScanColMajor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Build a simple scan fusion: out = cumsum(in, dim=0)
  auto tv0 = makeConcreteTensor({4, 8}, DataType::Float);
  fusion->addInput(tv0);
  auto tv_out = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion->addOutput(tv_out);

  // Create a real input to also get a concrete reference layout
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({8, 4}, options).t();

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_in = at::empty_strided(
      input.sizes(), input.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion->inputs().at(0), meta_in);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

// Parameterized test for EmbeddingFwd with different memory layouts
class EmbeddingFwdMetaTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
  // Parameters: (input_is_row_major, weight_is_row_major)
};

TEST_P(EmbeddingFwdMetaTest, MemoryLayouts) {
  auto [input_is_row_major, weight_is_row_major] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Build embedding fusion with appropriate memory layout
  TensorView* tv_input = input_is_row_major
      ? makeContigConcreteTensor({2, 4}, DataType::Int)
      : makeConcreteTensor({2, 4}, DataType::Int);
  TensorView* tv_weight = weight_is_row_major
      ? makeContigConcreteTensor({10, 8}, DataType::Float)
      : makeConcreteTensor({10, 8}, DataType::Float);
  fusion->addInput(tv_input);
  fusion->addInput(tv_weight);

  auto tv_out = embedding_fwd(
      tv_input,
      tv_weight,
      /*padding_idx=*/nullptr,
      /*max_norm=*/nullptr,
      /*norm_type=*/nullptr,
      /*scale_grad_by_freq=*/nullptr,
      /*sparse=*/nullptr);
  fusion->addOutput(tv_out);

  // Create real inputs with specified memory layout
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = input_is_row_major
      ? at::randint(0, 10, {2, 4}, options.dtype(at::kLong)).to(at::kInt)
      : at::randint(0, 10, {4, 2}, options.dtype(at::kLong)).to(at::kInt).t();
  at::Tensor weight = weight_is_row_major ? at::randn({10, 8}, options)
                                          : at::randn({8, 10}, options).t();

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), input);
  ee_cuda.bind(fusion->inputs().at(1), weight);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_input = at::empty_strided(
      input.sizes(),
      input.strides(),
      options.device(at::kMeta).dtype(at::kInt));
  auto meta_weight = at::empty_strided(
      weight.sizes(), weight.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion->inputs().at(0), meta_input);
  ee_meta.bind(fusion->inputs().at(1), meta_weight);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

INSTANTIATE_TEST_SUITE_P(
    EmbeddingFwdMemoryFormats,
    EmbeddingFwdMetaTest,
    ::testing::Values(
        std::make_tuple(true, true), // input: row-major, weight: row-major
        std::make_tuple(true, false), // input: row-major, weight: col-major
        std::make_tuple(false, true), // input: col-major, weight: row-major
        std::make_tuple(false, false)), // input: col-major, weight: col-major
    [](const testing::TestParamInfo<std::tuple<bool, bool>>& info) {
      auto [input_is_row_major, weight_is_row_major] = info.param;
      return std::string("input_") +
          (input_is_row_major ? "RowMajor" : "ColMajor") + "_weight_" +
          (weight_is_row_major ? "RowMajor" : "ColMajor");
    })

} // namespace nvfuser
