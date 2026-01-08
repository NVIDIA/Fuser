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

#include <array>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
      ? makeContigConcreteTensor({2, 4}, DataType::Int32)
      : makeConcreteTensor({2, 4}, DataType::Int32);
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
      bool input_is_row_major = std::get<0>(info.param);
      bool weight_is_row_major = std::get<1>(info.param);
      return std::string("input_") +
          (input_is_row_major ? "RowMajor" : "ColMajor") + "_weight_" +
          (weight_is_row_major ? "RowMajor" : "ColMajor");
    });

// Parameterized tests for GroupedMmaOp with different memory formats
enum class MemoryFormat2D { Contiguous, Transposed };

enum class MemoryFormat3D {
  Perm012, // [0, 1, 2] - Contiguous
  Perm021, // [0, 2, 1]
  Perm102, // [1, 0, 2]
  Perm120, // [1, 2, 0]
  Perm201, // [2, 0, 1]
  Perm210, // [2, 1, 0]
};

// All memory format values for parameterized tests
constexpr std::array<MemoryFormat2D, 2> all2DMemoryFormats = {
    MemoryFormat2D::Contiguous,
    MemoryFormat2D::Transposed};

constexpr std::array<MemoryFormat3D, 6> all3DMemoryFormats = {
    MemoryFormat3D::Perm012,
    MemoryFormat3D::Perm021,
    MemoryFormat3D::Perm102,
    MemoryFormat3D::Perm120,
    MemoryFormat3D::Perm201,
    MemoryFormat3D::Perm210};

// Helper functions to convert memory format enums to strings for test naming
std::string memoryFormat2DToString(MemoryFormat2D format) {
  switch (format) {
    case MemoryFormat2D::Contiguous:
      return "Contiguous";
    case MemoryFormat2D::Transposed:
      return "Transposed";
  }
  std::unreachable();
}

std::string memoryFormat3DToString(MemoryFormat3D format) {
  switch (format) {
    case MemoryFormat3D::Perm012:
      return "Perm012";
    case MemoryFormat3D::Perm021:
      return "Perm021";
    case MemoryFormat3D::Perm102:
      return "Perm102";
    case MemoryFormat3D::Perm120:
      return "Perm120";
    case MemoryFormat3D::Perm201:
      return "Perm201";
    case MemoryFormat3D::Perm210:
      return "Perm210";
  }
  std::unreachable();
}

// Helper function to create tensor with specified memory format for 2D tensors
at::Tensor createTensor2D(
    const std::vector<int64_t>& sizes,
    MemoryFormat2D format,
    const at::TensorOptions& options) {
  switch (format) {
    case MemoryFormat2D::Contiguous:
      return at::randn(sizes, options);
    case MemoryFormat2D::Transposed:
      return at::randn({sizes[1], sizes[0]}, options).t();
  }
  std::unreachable();
}

// Helper function to create tensor with specified memory format for 3D tensors
at::Tensor createTensor3D(
    const std::vector<int64_t>& sizes,
    MemoryFormat3D format,
    const at::TensorOptions& options) {
  // sizes = [d0, d1, d2]
  switch (format) {
    case MemoryFormat3D::Perm012: // [0, 1, 2] - Contiguous
      return at::randn(sizes, options);
    case MemoryFormat3D::Perm021: // [0, 2, 1]
      return at::randn({sizes[0], sizes[2], sizes[1]}, options)
          .permute({0, 2, 1});
    case MemoryFormat3D::Perm102: // [1, 0, 2]
      return at::randn({sizes[1], sizes[0], sizes[2]}, options)
          .permute({1, 0, 2});
    case MemoryFormat3D::Perm120: // [1, 2, 0]
      return at::randn({sizes[1], sizes[2], sizes[0]}, options)
          .permute({2, 0, 1});
    case MemoryFormat3D::Perm201: // [2, 0, 1]
      return at::randn({sizes[2], sizes[0], sizes[1]}, options)
          .permute({1, 2, 0});
    case MemoryFormat3D::Perm210: // [2, 1, 0]
      return at::randn({sizes[2], sizes[1], sizes[0]}, options)
          .permute({2, 1, 0});
  }
  std::unreachable();
}

// Test GroupedMmaOp with mat1=[m, k], mat2=[k, n] -> out=[g, m, n]
class MetaTestGroupedMma2D2D : public NVFuserTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<MemoryFormat2D, MemoryFormat2D>> {
};

TEST_P(MetaTestGroupedMma2D2D, MemoryFormats) {
  auto [mat1_format, mat2_format] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // mat1: [m, k] = [128, 128], mat2: [k, n] = [128, 128]
  // output: [g, m, n] = [4, 128, 128]
  auto mat1 = makeConcreteTensor({128, 128}, DataType::BFloat16);
  auto mat2 = makeConcreteTensor({128, 128}, DataType::BFloat16);
  auto offsets = makeContigConcreteTensor({4}, DataType::Index);
  fusion->addInput(mat1);
  fusion->addInput(mat2);
  fusion->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion->addOutput(result.tv);

  // Create real inputs with specified memory formats
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor mat1_input = createTensor2D({128, 128}, mat1_format, options);
  at::Tensor mat2_input = createTensor2D({128, 128}, mat2_format, options);
  at::Tensor offsets_input = at::tensor(
      {32, 64, 96, 128},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion->inputs().at(2), offsets_input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

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
  ee_meta.bind(fusion->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion->inputs().at(2), meta_offsets);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kBFloat16);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryFormatCombinations,
    MetaTestGroupedMma2D2D,
    ::testing::Combine(
        ::testing::ValuesIn(all2DMemoryFormats),
        ::testing::ValuesIn(all2DMemoryFormats)),
    [](const ::testing::TestParamInfo<
        std::tuple<MemoryFormat2D, MemoryFormat2D>>& info) {
      return "Mat1" + memoryFormat2DToString(std::get<0>(info.param)) +
          "_Mat2" + memoryFormat2DToString(std::get<1>(info.param));
    });

// Test GroupedMmaOp with mat1=[g, m, k], mat2=[k, n] -> out=[m, n]
class MetaTestGroupedMma3D2D : public NVFuserTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<MemoryFormat3D, MemoryFormat2D>> {
};

TEST_P(MetaTestGroupedMma3D2D, MemoryFormats) {
  auto [mat1_format, mat2_format] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // mat1: [g, m, k] = [4, 128, 128], mat2: [k, n] = [128, 128]
  // output: [m, n] = [128, 128]
  auto mat1 = makeConcreteTensor({4, 128, 128}, DataType::BFloat16);
  auto mat2 = makeConcreteTensor({128, 128}, DataType::BFloat16);
  auto offsets = makeContigConcreteTensor({4}, DataType::Index);
  fusion->addInput(mat1);
  fusion->addInput(mat2);
  fusion->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion->addOutput(result.tv);

  // Create real inputs with specified memory formats
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor mat1_input = createTensor3D({4, 128, 128}, mat1_format, options);
  at::Tensor mat2_input = createTensor2D({128, 128}, mat2_format, options);
  at::Tensor offsets_input = at::tensor(
      {32, 64, 96, 128},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion->inputs().at(2), offsets_input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

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
  ee_meta.bind(fusion->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion->inputs().at(2), meta_offsets);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kBFloat16);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryFormatCombinations,
    MetaTestGroupedMma3D2D,
    ::testing::Combine(
        ::testing::ValuesIn(all3DMemoryFormats),
        ::testing::ValuesIn(all2DMemoryFormats)),
    [](const ::testing::TestParamInfo<
        std::tuple<MemoryFormat3D, MemoryFormat2D>>& info) {
      return "Mat1" + memoryFormat3DToString(std::get<0>(info.param)) +
          "_Mat2" + memoryFormat2DToString(std::get<1>(info.param));
    });

// Test GroupedMmaOp with mat1=[m, k], mat2=[g, k, n] -> out=[m, n]
class MetaTestGroupedMma2D3D : public NVFuserTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<MemoryFormat2D, MemoryFormat3D>> {
};

TEST_P(MetaTestGroupedMma2D3D, MemoryFormats) {
  auto [mat1_format, mat2_format] = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // mat1: [m, k] = [128, 128], mat2: [g, k, n] = [4, 128, 128]
  // output: [m, n] = [128, 128]
  auto mat1 = makeConcreteTensor({128, 128}, DataType::BFloat16);
  auto mat2 = makeConcreteTensor({4, 128, 128}, DataType::BFloat16);
  auto offsets = makeContigConcreteTensor({4}, DataType::Index);
  fusion->addInput(mat1);
  fusion->addInput(mat2);
  fusion->addInput(offsets);

  auto result = grouped_mm(mat1, mat2, offsets);
  fusion->addOutput(result.tv);

  // Create real inputs with specified memory formats
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor mat1_input = createTensor2D({128, 128}, mat1_format, options);
  at::Tensor mat2_input = createTensor3D({4, 128, 128}, mat2_format, options);
  at::Tensor offsets_input = at::tensor(
      {32, 64, 96, 128},
      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  // CUDA path
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion->inputs().at(2), offsets_input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

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
  ee_meta.bind(fusion->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion->inputs().at(2), meta_offsets);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kBFloat16);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryFormatCombinations,
    MetaTestGroupedMma2D3D,
    ::testing::Combine(
        ::testing::ValuesIn(all2DMemoryFormats),
        ::testing::ValuesIn(all3DMemoryFormats)),
    [](const ::testing::TestParamInfo<
        std::tuple<MemoryFormat2D, MemoryFormat3D>>& info) {
      return "Mat1" + memoryFormat2DToString(std::get<0>(info.param)) +
          "_Mat2" + memoryFormat3DToString(std::get<1>(info.param));
    });

// Test for MatmulOp with 1D @ 1D (dot product) on meta device
// This tests the special case where aten::dot does not support meta device
TEST_F(MetaTest, Matmul1D) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Build 1D @ 1D matmul fusion (dot product)
  auto tv_a = makeContigConcreteTensor({128}, DataType::Float);
  auto tv_b = makeContigConcreteTensor({128}, DataType::Float);
  fusion->addInput(tv_a);
  fusion->addInput(tv_b);

  auto tv_out = matmul(tv_a, tv_b);
  fusion->addOutput(tv_out);

  // Create real inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor a_input = at::randn({128}, options);
  at::Tensor b_input = at::randn({128}, options);

  // CUDA path via ExpressionEvaluator
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), a_input);
  ee_cuda.bind(fusion->inputs().at(1), b_input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation - this is where the special handling for 1D @ 1D kicks in
  ExpressionEvaluator ee_meta;
  auto meta_a = at::empty_strided(
      a_input.sizes(), a_input.strides(), options.device(at::kMeta));
  auto meta_b = at::empty_strided(
      b_input.sizes(), b_input.strides(), options.device(at::kMeta));
  ee_meta.bind(fusion->inputs().at(0), meta_a);
  ee_meta.bind(fusion->inputs().at(1), meta_b);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks: tensor is meta, dtype/size/stride match
  // For 1D @ 1D, the output should be a scalar (0-dimensional tensor)
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kFloat);
  EXPECT_EQ(meta_out.dim(), 0); // scalar output
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
}

// Test CutlassNvfp4GroupedMmaOp with meta device
TEST_F(MetaTest, CutlassNvfp4GroupedMma) {
#if NVFUSER_CUTLASS_KERNEL_ENABLED
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Choose an example where all M, N, K, and K/2 are different:
  //   M = 128, N = 80, K = 192, K/2 = 96
  // Shapes:
  //   mat1: [M, K/2]       = [128, 96]   (packed FP4)
  //   mat2: [G, N, K/2]    = [4, 80, 96] (packed FP4)
  //   output: [M, N]       = [128, 80]
  // Note: Use unpacked type Float4_e2m1fn for fusion definition
  auto mat1 = makeContigConcreteTensor({128, 96}, DataType::Float4_e2m1fn);
  auto mat2 =
      makeContigConcreteTensor({4, 80, 96}, DataType::Float4_e2m1fn);
  // Block-scaling factors have last dim K / 16 = 192 / 16 = 12
  auto scale1 = makeContigConcreteTensor({128, 12}, DataType::Float8_e4m3fn);
  auto scale2 = makeContigConcreteTensor({4, 80, 12}, DataType::Float8_e4m3fn);
  auto alpha = makeContigConcreteTensor({4}, DataType::Float);
  auto problem_sizes = makeContigConcreteTensor({4, 3}, DataType::Index);
  auto expert_offsets = makeContigConcreteTensor({4}, DataType::Index);
  auto sf_offsets = makeContigConcreteTensor({4}, DataType::Index);

  fusion->addInput(mat1);
  fusion->addInput(mat2);
  fusion->addInput(scale1);
  fusion->addInput(scale2);
  fusion->addInput(alpha);
  fusion->addInput(problem_sizes);
  fusion->addInput(expert_offsets);
  fusion->addInput(sf_offsets);

  auto result = cutlass_nvfp4_grouped_mm(
      mat1,
      mat2,
      scale1,
      scale2,
      alpha,
      problem_sizes,
      expert_offsets,
      sf_offsets,
      DataType::BFloat16);
  fusion->addOutput(result);

  // Create real inputs with appropriate data types
  auto options_uint8 =
      at::TensorOptions().dtype(torch::kUInt8).device(at::kCUDA, 0);
  auto options_fp4 =
      at::TensorOptions().dtype(at::kFloat4_e2m1fn_x2).device(at::kCUDA, 0);
  auto options_fp8 =
      at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(at::kCUDA, 0);
  auto options_fp32 =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0);

  // FP4 tensors must be created as UInt8 and viewed as Float4
  at::Tensor mat1_input =
      at::randint(0, 256, {128, 96}, options_uint8).view(at::kFloat4_e2m1fn_x2);
  at::Tensor mat2_input = at::randint(0, 256, {4, 80, 96}, options_uint8)
                              .view(at::kFloat4_e2m1fn_x2);
  // FP8 tensors can be created from FP32 tensors
  at::Tensor scale1_input =
      at::randn({128, 12}, options_fp32).to(at::kFloat8_e4m3fn);
  at::Tensor scale2_input =
      at::randn({4, 80, 12}, options_fp32).to(at::kFloat8_e4m3fn);
  at::Tensor alpha_input = at::ones({4}, options_fp32);
  at::Tensor problem_sizes_input = at::tensor(
      {32, 80, 192, 32, 80, 192, 32, 80, 192, 32, 80, 192},
      options_int).reshape({4, 3});
  at::Tensor expert_offsets_input = at::tensor({0, 32, 64, 96}, options_int);
  at::Tensor sf_offsets_input = at::tensor({0, 32, 64, 96}, options_int);

  // CUDA path
  ExpressionEvaluator ee_cuda;
  ee_cuda.bind(fusion->inputs().at(0), mat1_input);
  ee_cuda.bind(fusion->inputs().at(1), mat2_input);
  ee_cuda.bind(fusion->inputs().at(2), scale1_input);
  ee_cuda.bind(fusion->inputs().at(3), scale2_input);
  ee_cuda.bind(fusion->inputs().at(4), alpha_input);
  ee_cuda.bind(fusion->inputs().at(5), problem_sizes_input);
  ee_cuda.bind(fusion->inputs().at(6), expert_offsets_input);
  ee_cuda.bind(fusion->inputs().at(7), sf_offsets_input);
  auto real_out = ee_cuda.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Meta evaluation
  ExpressionEvaluator ee_meta;
  auto meta_mat1 = at::empty_strided(
      mat1_input.sizes(), mat1_input.strides(), options_fp4.device(at::kMeta));
  auto meta_mat2 = at::empty_strided(
      mat2_input.sizes(), mat2_input.strides(), options_fp4.device(at::kMeta));
  auto meta_scale1 = at::empty_strided(
      scale1_input.sizes(),
      scale1_input.strides(),
      options_fp8.device(at::kMeta));
  auto meta_scale2 = at::empty_strided(
      scale2_input.sizes(),
      scale2_input.strides(),
      options_fp8.device(at::kMeta));
  auto meta_alpha = at::empty_strided(
      alpha_input.sizes(),
      alpha_input.strides(),
      options_fp32.device(at::kMeta));
  auto meta_problem_sizes = at::empty_strided(
      problem_sizes_input.sizes(),
      problem_sizes_input.strides(),
      options_int.device(at::kMeta));
  auto meta_expert_offsets = at::empty_strided(
      expert_offsets_input.sizes(),
      expert_offsets_input.strides(),
      options_int.device(at::kMeta));
  auto meta_sf_offsets = at::empty_strided(
      sf_offsets_input.sizes(),
      sf_offsets_input.strides(),
      options_int.device(at::kMeta));

  ee_meta.bind(fusion->inputs().at(0), meta_mat1);
  ee_meta.bind(fusion->inputs().at(1), meta_mat2);
  ee_meta.bind(fusion->inputs().at(2), meta_scale1);
  ee_meta.bind(fusion->inputs().at(3), meta_scale2);
  ee_meta.bind(fusion->inputs().at(4), meta_alpha);
  ee_meta.bind(fusion->inputs().at(5), meta_problem_sizes);
  ee_meta.bind(fusion->inputs().at(6), meta_expert_offsets);
  ee_meta.bind(fusion->inputs().at(7), meta_sf_offsets);
  auto meta_out = ee_meta.evaluate(fusion->outputs().at(0)).as<at::Tensor>();

  // Checks
  EXPECT_TRUE(meta_out.is_meta());
  EXPECT_EQ(meta_out.scalar_type(), at::kBFloat16);
  EXPECT_EQ(meta_out.sizes(), real_out.sizes());
  EXPECT_EQ(meta_out.strides(), real_out.strides());
#else
  GTEST_SKIP() << "Test requires CUTLASS support";
#endif
}

} // namespace nvfuser
