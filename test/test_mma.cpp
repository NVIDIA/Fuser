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

using TuringAmpereMmaTestParams = std::tuple<MmaMacro, DataType, MmaLayout>;

class TuringAmpere : public NVFuserFixtureParamTest<TuringAmpereMmaTestParams> {
  void SetUp() override {
    // requires Hopper or newer
    if (cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// MMA unit test on Turing
TEST_P(TuringAmpere, SingleTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto dtype = std::get<1>(GetParam());
  auto layout = std::get<2>(GetParam());

  if (isAmpere(macro) && cudaArchGuardShouldSkip(8, 0)) {
    GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
  }

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

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(A_shape, options);
  auto t1 = at::randn(B_shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  at::Tensor t0t = t0, t1t = t1;

  if (transpose_a) {
    t0t = t0.t();
  }

  if (!transpose_b) {
    t1t = t1.t();
  }

  auto tref = t0t.to(at::kFloat).matmul(t1t.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    MmaTest,
    TuringAmpere,
    testing::Values(
        std::make_tuple(MmaMacro::Turing_16_8_8, DataType::Half, MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Turing_16_8_16,
            DataType::Half,
            MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Turing_16_16_16,
            DataType::Half,
            MmaLayout::TT),
        std::make_tuple(MmaMacro::Turing_16_8_8, DataType::Half, MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Turing_16_8_16,
            DataType::Half,
            MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Turing_16_16_16,
            DataType::Half,
            MmaLayout::TN),
        std::make_tuple(MmaMacro::Turing_16_8_8, DataType::Half, MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Turing_16_8_16,
            DataType::Half,
            MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Turing_16_16_16,
            DataType::Half,
            MmaLayout::NT),
        std::make_tuple(MmaMacro::Turing_16_8_8, DataType::Half, MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Turing_16_8_16,
            DataType::Half,
            MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Turing_16_16_16,
            DataType::Half,
            MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::Half,
            MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::Half,
            MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::BFloat16,
            MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::BFloat16,
            MmaLayout::TT),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::Half,
            MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::Half,
            MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::BFloat16,
            MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::BFloat16,
            MmaLayout::TN),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::Half,
            MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::Half,
            MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::BFloat16,
            MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::BFloat16,
            MmaLayout::NT),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::Half,
            MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::Half,
            MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Ampere_16_8_16,
            DataType::BFloat16,
            MmaLayout::NN),
        std::make_tuple(
            MmaMacro::Ampere_16_16_16,
            DataType::BFloat16,
            MmaLayout::NN)),
    [](const testing::TestParamInfo<TuringAmpereMmaTestParams>& info) {
      std::ostringstream os;
      auto macro = std::get<0>(info.param);
      auto dtype = std::get<1>(info.param);
      auto layout = std::get<2>(info.param);
      os << toString(macro) << "_" << toString(layout) << dtype;
      return os.str();
    });

} // namespace nvfuser
