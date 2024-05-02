// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <exceptions.h>
#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>

namespace nvfuser {

namespace debugging {

// Utilities for debugging MMA ops

// Set a tensor as identity, for example
//  [1, 0, 0]
//  [0, 1, 0]
//  [0, 0, 1]
//  [0, 0, 0]
//  [0, 0, 0]
// This is helpful for debugging because mathematically, an identity matrix
// multiplies any matrix to itself. For example, if you are seeing a wrong
// result, but you don't know if it's because of the input B's memory format is
// not scheduled correctly, you can set the input A to identity and print the
// output. By reading the output, you can tell how the memory layout of input B
// looks like.
void setAsIdentity(at::Tensor tensor) {
  tensor.zero_();
  for (auto i : c10::irange(tensor.size(0))) {
    for (auto j : c10::irange(tensor.size(1))) {
      if (i == j) {
        tensor[i][j] = 1;
      }
    }
  }
}

// Set a tensor as a range, for example
//  [0, 1, 2]
//  [3, 4, 5]
//  [6, 7, 8]
// This makes the tensor easier to read if you print it out.
void setAsARange(at::Tensor tensor) {
  tensor.zero_();
  for (auto i : c10::irange(tensor.size(0))) {
    for (auto j : c10::irange(tensor.size(1))) {
      tensor[i][j] = i * tensor.size(1) + j;
    }
  }
}

} // namespace debugging

using MmaTestParams = std::tuple<MmaMacro, PrimDataType>;

class MmaTest : public NVFuserFixtureParamTest<MmaTestParams> {
 protected:
  MmaMacro macro;
  PrimDataType dtype;

  void SetUp() override {
    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());

    if (isTuring(macro) && cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }

    if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    }

    NVFuserTest::SetUp();
  }
};

TEST_P(MmaTest, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(
      getM(macro), getN(macro), getK(macro), MmaLayout::TN);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, 1, K]
  // Just doing a gmem->register copy
  tv0 = set(tv0);

  // [1, N, K]
  // Just doing a gmem->register copy
  tv1 = set(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] -> [N, M, K]
  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv1->applyMmaSwizzle(MmaOperand::B);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);
  tv1->merge(1);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput3DTuring(
      getM(macro),
      getN(macro),
      getK(macro),
      MmaLayout::TN,
      data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      MmaLayout::TN);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

auto all_dtypes = testing::Values(DataType::Half, DataType::BFloat16);

std::string testName(const testing::TestParamInfo<MmaTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  os << toString(macro) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    Turing,
    MmaTest,
    testing::Combine(
        testing::Values(
            MmaMacro::Turing_16_8_8,
            MmaMacro::Turing_16_8_16,
            MmaMacro::Turing_16_16_16),
        testing::Values(DataType::Half)),
    testName);

INSTANTIATE_TEST_SUITE_P(
    Ampere,
    MmaTest,
    testing::Combine(
        testing::Values(MmaMacro::Ampere_16_8_16, MmaMacro::Ampere_16_16_16),
        all_dtypes),
    testName);

class HopperBase : public NVFuserTest {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// For smem mma input tensors, the schedule does not matter, we just naively
// parallelize it so the test runs faster.
void naivelyParallelize(TensorView* tv) {
  while (tv->nDims() > 1) {
    tv->merge(0);
  }
  tv->split(0, 128);
  tv->axis(1)->parallelize(ParallelType::TIDx);
}

auto all_mma_layouts =
    testing::Values(MmaLayout::TT, MmaLayout::TN, MmaLayout::NT, MmaLayout::NN);

auto all_hopper_macros = testing::Values(
    MmaMacro::Hopper_64_8_16,
    MmaMacro::Hopper_64_16_16,
    MmaMacro::Hopper_64_24_16,
    MmaMacro::Hopper_64_32_16,
    MmaMacro::Hopper_64_40_16,
    MmaMacro::Hopper_64_48_16,
    MmaMacro::Hopper_64_56_16,
    MmaMacro::Hopper_64_64_16,
    MmaMacro::Hopper_64_72_16,
    MmaMacro::Hopper_64_80_16,
    MmaMacro::Hopper_64_88_16,
    MmaMacro::Hopper_64_96_16,
    MmaMacro::Hopper_64_104_16,
    MmaMacro::Hopper_64_112_16,
    MmaMacro::Hopper_64_120_16,
    MmaMacro::Hopper_64_128_16,
    MmaMacro::Hopper_64_136_16,
    MmaMacro::Hopper_64_144_16,
    MmaMacro::Hopper_64_152_16,
    MmaMacro::Hopper_64_160_16,
    MmaMacro::Hopper_64_168_16,
    MmaMacro::Hopper_64_176_16,
    MmaMacro::Hopper_64_184_16,
    MmaMacro::Hopper_64_192_16,
    MmaMacro::Hopper_64_200_16,
    MmaMacro::Hopper_64_208_16,
    MmaMacro::Hopper_64_216_16,
    MmaMacro::Hopper_64_224_16,
    MmaMacro::Hopper_64_232_16,
    MmaMacro::Hopper_64_240_16,
    MmaMacro::Hopper_64_248_16,
    MmaMacro::Hopper_64_256_16);

using HopperMmaRSTestParams =
    std::tuple<MmaMacro, PrimDataType, MmaLayout, MmaInputSmemSwizzle>;

class HopperRS : public HopperBase,
                 public ::testing::WithParamInterface<HopperMmaRSTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_b;

  void SetUp() override {
    HopperBase::SetUp();

    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_b = std::get<3>(GetParam());
  }
};

std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DHopperRS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

std::pair<at::Tensor, at::Tensor> matmulAtInput3DHopperRS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DHopperRS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

TEST_P(HopperRS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->register copy
  tv0 = set(tv0);
  // Just doing a gmem->smem copy
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = fusedMultiplySum(tv0, tv1, {layout == MmaLayout::TT ? 1 : 2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  moveInnerBroadcastLeft(tv0);
  tv0->applyMmaSwizzle(MmaOperand::A);

  tv0->merge(1);
  tv0->merge(1);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  tv1->applyMmaSwizzle(swizzle_b, layout == MmaLayout::TN);

  naivelyParallelize(tv1);

  if (layout == MmaLayout::TT) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput3DHopperRS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

std::string testNameHopperRS(
    const testing::TestParamInfo<HopperMmaRSTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  auto swizzle_b = std::get<3>(info.param);
  os << toString(macro) << "_" << toString(layout) << "_" << toString(swizzle_b)
     << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperRS,
    testing::Combine(
        all_hopper_macros,
        all_dtypes,
        testing::Values(MmaLayout::TT, MmaLayout::TN),
        kAllSmemSwizzleModes),
    testNameHopperRS);

using HopperMmaSSTestParams = std::tuple<
    MmaMacro,
    PrimDataType,
    MmaLayout,
    MmaInputSmemSwizzle,
    MmaInputSmemSwizzle>;

class HopperSS : public HopperBase,
                 public ::testing::WithParamInterface<HopperMmaSSTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;
  MmaInputSmemSwizzle swizzle_a;
  MmaInputSmemSwizzle swizzle_b;

  void SetUp() override {
    HopperBase::SetUp();

    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());
    swizzle_a = std::get<3>(GetParam());
    swizzle_b = std::get<4>(GetParam());
  }
};

std::pair<std::vector<int64_t>, std::vector<int64_t>>
matmulAtInputShape3DHopperSS(int M, int N, int K, MmaLayout layout) {
  switch (layout) {
    case MmaLayout::TT:
      return {{M, K, 1}, {1, K, N}};
    case MmaLayout::TN:
      return {{M, 1, K}, {1, N, K}};
    case MmaLayout::NT:
      return {{K, M, 1}, {K, 1, N}};
    case MmaLayout::NN:
      return {{1, K, M}, {N, K, 1}};
    default:
      NVF_CHECK(false, "unsupported data layout.");
  }
}

std::pair<at::Tensor, at::Tensor> matmulAtInput3DHopperSS(
    int M,
    int N,
    int K,
    MmaLayout layout,
    c10::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  auto shapes = matmulAtInputShape3DHopperSS(M, N, K, layout);
  return std::make_pair(
      at::randn(shapes.first, options), at::randn(shapes.second, options));
}

TEST_P(HopperSS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  bool transpose_a = (layout == MmaLayout::NT || layout == MmaLayout::NN);
  bool transpose_b = (layout == MmaLayout::TN || layout == MmaLayout::NN);

  auto shapes = matmulAtInputShape3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout);

  auto tv0 = makeConcreteTensor(shapes.first, dtype);
  auto tv1 = makeConcreteTensor(shapes.second, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Just doing a gmem->smem copy
  tv0 = set(tv0);
  tv0->setMemoryType(MemoryType::Shared);
  tv1 = set(tv1);
  tv1->setMemoryType(MemoryType::Shared);

  int axes = 0;
  switch (layout) {
    case MmaLayout::NT:
      axes = 0;
      break;
    case MmaLayout::TT:
    case MmaLayout::NN:
      axes = 1;
      break;
    case MmaLayout::TN:
      axes = 2;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  auto tv2 = fusedMultiplySum(tv0, tv1, {axes});

  // Reorder the accumulator as [M, N, K]
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N] -> [M, N, K]
      tv2->reorder({{-2, -1}});
      break;
    case MmaLayout::TN:
      // [M, N, K]
      break;
    case MmaLayout::NT:
      // [K, M, N] -> [M, N, K]
      tv2->reorder({{-3, -1}});
      break;
    case MmaLayout::NN:
      // [N, K, M] -> [M, N, K]
      tv2->reorder({{-1, -3}});
      break;
    default:
      NVF_ERROR("Invalid layout");
  }
  tv2->commitLeafToRFactor();

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // Bring related dims to innermost, that is:
  // - Reorder tv0 as [1, M, K] or [1, K, M]
  // - Reorder tv1 as [1, N, K] or [1, K, N]
  moveInnerBroadcastLeft(tv0);
  moveInnerBroadcastLeft(tv1);

  // Hopper tensor core assumes K major, so we are using !transpose_a here.
  tv0->applyMmaSwizzle(swizzle_a, !transpose_a);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->applyMmaSwizzle(swizzle_b, transpose_b ^ transpose_a);

  naivelyParallelize(tv0);
  naivelyParallelize(tv1);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput3DHopperSS(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.squeeze().to(at::kFloat),
      inputs.second.squeeze().to(at::kFloat),
      layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

std::string testNameHopperSS(
    const testing::TestParamInfo<HopperMmaSSTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  auto swizzle_a = std::get<3>(info.param);
  auto swizzle_b = std::get<4>(info.param);
  os << toString(macro) << "_" << toString(layout) << "_" << toString(swizzle_a)
     << "_" << toString(swizzle_b) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperSS,
    testing::Combine(
        all_hopper_macros,
        all_dtypes,
        all_mma_layouts,
        kAllSmemSwizzleModes,
        kAllSmemSwizzleModes),
    testNameHopperSS);

} // namespace nvfuser
