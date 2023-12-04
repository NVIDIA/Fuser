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

#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>

namespace nvfuser {

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

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
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
  testValidate(
      &fusion,
      cg_outputs,
      {inputs.first, inputs.second},
      {tref},
      __LINE__,
      __FILE__);
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

} // namespace nvfuser
