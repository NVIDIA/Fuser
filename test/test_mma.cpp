// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <test/utils.h>
#include <test/validator.h>

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

using MmaTestParams = std::tuple<MmaMacro, PrimDataType, MmaLayout>;

class MmaTest : public NVFuserFixtureParamTest<MmaTestParams> {
 protected:
  MmaLayout layout;
  MmaMacro macro;
  PrimDataType dtype;

  void SetUp() override {
    macro = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    layout = std::get<2>(GetParam());

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

  bool transpose_a = (layout == MmaLayout::NT || layout == MmaLayout::NN);
  bool transpose_b = (layout == MmaLayout::TT || layout == MmaLayout::NT);

  std::vector<int64_t> A_shape{getM(macro), getK(macro)},
      B_shape{getN(macro), getK(macro)};

  if (transpose_a) {
    std::swap(A_shape[0], A_shape[1]);
  }

  if (transpose_b) {
    std::swap(B_shape[0], B_shape[1]);
  }

  auto tv0 = makeConcreteTensor(A_shape, dtype);
  auto tv1 = makeConcreteTensor(B_shape, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, K]
  if (transpose_a) {
    tv0 = transpose(tv0, 0, 1);
  }

  // [N, K]
  if (transpose_b) {
    tv1 = transpose(tv1, 0, 1);
  }

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  // [M, N, K] -> [N, M, K]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  tv0b->merge(1);
  tv0b->merge(1);
  tv0b->axis(1)->parallelize(ParallelType::TIDx);
  tv1b->merge(1);
  tv1b->axis(1)->parallelize(ParallelType::TIDx);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

auto all_mma_layouts =
    testing::Values(MmaLayout::TT, MmaLayout::TN, MmaLayout::NT, MmaLayout::NN);

auto all_dtypes = testing::Values(DataType::Half, DataType::BFloat16);

std::string testName(const testing::TestParamInfo<MmaTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  os << getM(macro) << "_" << getN(macro) << "_" << getK(macro) << "_"
     << toString(layout) << dtype;
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
        testing::Values(DataType::Half),
        all_mma_layouts),
    testName);

INSTANTIATE_TEST_SUITE_P(
    Ampere,
    MmaTest,
    testing::Combine(
        testing::Values(MmaMacro::Ampere_16_8_16, MmaMacro::Ampere_16_16_16),
        all_dtypes,
        all_mma_layouts),
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

auto all_smem_swizzle_modes = testing::Values(
    MmaInputSmemSwizzle::None,
    MmaInputSmemSwizzle::B128,
    MmaInputSmemSwizzle::B64,
    MmaInputSmemSwizzle::B32);

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

TEST_P(HopperRS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  bool transpose_a = (layout == MmaLayout::NT || layout == MmaLayout::NN);
  bool transpose_b = (layout == MmaLayout::TN || layout == MmaLayout::NN);

  std::vector<int64_t> A_shape{getM(macro), getK(macro)},
      B_shape{getK(macro), getN(macro)};

  if (transpose_a) {
    std::swap(A_shape[0], A_shape[1]);
  }

  if (transpose_b) {
    std::swap(B_shape[0], B_shape[1]);
  }

  auto tv0 = makeConcreteTensor(A_shape, dtype);
  auto tv1 = makeConcreteTensor(B_shape, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, K]
  if (transpose_a) {
    tv0 = transpose(tv0, 0, 1);
  }

  TensorView* tv0b = nullptr;
  int axes = 0;
  if (transpose_b) {
    // [M, N, K]
    tv0b = broadcast(tv0, {false, true, false});
    axes = 2;
  } else {
    // [M, K, N]
    tv0b = broadcast(tv0, {false, false, true});
    axes = 1;
  }
  auto tv1b = broadcast(tv1, {true, false, false});

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {axes});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  auto tv2c = tv2->cacheBefore();

  if (transpose_b) {
    // [M, N, K] -> [N, M, K]
    tv0b->reorder({{-2, -3}});
  } else {
    // [M, K, N] -> [N, M, K]
    tv0b->reorder({{-1, -3}});
  }
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv0b->merge(1);
  tv0b->merge(1);
  tv0b->axis(1)->parallelize(ParallelType::TIDx);

  tv1b->setMemoryType(MemoryType::Shared);
  tv1b->applyMmaSwizzle(swizzle_b, transpose_b);

  naivelyParallelize(tv1b);

  if (!transpose_b) {
    // [M, K, N] -> [M, N, K]
    tv2c->reorder({{-1, -2}});
  }

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

std::string testNameHopperRS(
    const testing::TestParamInfo<HopperMmaRSTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto layout = std::get<2>(info.param);
  auto swizzle_b = std::get<3>(info.param);
  os << getM(macro) << "_" << getN(macro) << "_" << getK(macro) << "_"
     << toString(layout) << "_" << toString(swizzle_b) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperRS,
    testing::Combine(
        all_hopper_macros,
        all_dtypes,
        all_mma_layouts,
        all_smem_swizzle_modes),
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

TEST_P(HopperSS, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  bool transpose_a = (layout == MmaLayout::NT || layout == MmaLayout::NN);
  bool transpose_b = (layout == MmaLayout::TN || layout == MmaLayout::NN);

  std::vector<int64_t> A_shape{getM(macro), getK(macro)},
      B_shape{getK(macro), getN(macro)};

  if (transpose_a) {
    std::swap(A_shape[0], A_shape[1]);
  }

  if (transpose_b) {
    std::swap(B_shape[0], B_shape[1]);
  }

  auto tv0 = makeConcreteTensor(A_shape, dtype);
  auto tv1 = makeConcreteTensor(B_shape, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv0b = nullptr;
  TensorView* tv1b = nullptr;
  int axes = 0;
  switch (layout) {
    case MmaLayout::TT:
      // [M, K, N]
      tv0b = broadcast(tv0, {false, false, true});
      tv0b->reorder({{-1, -3}});
      tv1b = broadcast(tv1, {true, false, false});
      axes = 1;
      break;
    case MmaLayout::TN:
      // [M, N, K]
      tv0b = broadcast(tv0, {false, true, false});
      tv0b->reorder({{-2, -3}});
      tv1b = broadcast(tv1, {true, false, false});
      axes = 2;
      break;
    case MmaLayout::NT:
      // [K, M, N]
      tv0b = broadcast(tv0, {false, false, true});
      tv0b->reorder({{-1, -3}});
      tv1b = broadcast(tv1, {false, true, false});
      tv1b->reorder({{-2, -3}});
      axes = 0;
      break;
    case MmaLayout::NN:
      // [N, K, M]
      tv0b = broadcast(tv0, {true, false, false});
      tv1b = broadcast(tv1, {false, false, true});
      tv1b->reorder({{-1, -3}});
      axes = 1;
      break;
    default:
      NVF_ERROR("Invalid layout");
  }

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {axes});

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

  tv0b->setMemoryType(MemoryType::Shared);
  // Hopper tensor core assumes K major, so we are using !transpose_a here.
  tv0b->applyMmaSwizzle(swizzle_a, !transpose_a);
  tv1b->setMemoryType(MemoryType::Shared);
  tv1b->applyMmaSwizzle(swizzle_b, transpose_b, transpose_a);

  naivelyParallelize(tv0b);
  naivelyParallelize(tv1b);

  tv2c->applyMmaSwizzle(MmaOperand::Accumulator);
  tv2->applyMmaSwizzle(MmaOperand::Accumulator);

  auto inputs = matmulAtInput(
      getM(macro), getN(macro), getK(macro), layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
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
  os << getM(macro) << "_" << getN(macro) << "_" << getK(macro) << "_"
     << toString(layout) << "_" << toString(swizzle_a) << "_"
     << toString(swizzle_b) << dtype;
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    HopperSS,
    testing::Combine(
        all_hopper_macros,
        all_dtypes,
        all_mma_layouts,
        all_smem_swizzle_modes,
        all_smem_swizzle_modes),
    testNameHopperSS);

} // namespace nvfuser
