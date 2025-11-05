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

  // mat1: [m, k] = [128, 128], mat2: [k, n] = [128, 128], output: [g, m, n] = [4, 128, 128]
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
      {32, 64, 96, 128}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

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

  // mat1: [g, m, k] = [4, 128, 128], mat2: [k, n] = [128, 128], output: [m, n] = [128, 128]
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
      {32, 64, 96, 128}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

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

  // mat1: [m, k] = [128, 128], mat2: [g, k, n] = [4, 128, 128], output: [m, n] = [128, 128]
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
      {32, 64, 96, 128}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

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

} // namespace nvfuser
