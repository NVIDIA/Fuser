// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <options.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul_heuristic_plugin.h>
#include <scheduler/matmul_heuristic_plugin_api.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <memory>

namespace nvfuser {

namespace {
class MatmulSchedulerTest : public NVFuserTest {
 protected:
  MatmulSchedulerTest() : optimization_guard_(false) {}

 private:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
};

class MatmulSchedulerTestWithLayout
    : public MatmulSchedulerTest,
      public ::testing::WithParamInterface<MmaLayout> {
 protected:
  MmaLayout layout;
  void SetUp() override {
    layout = GetParam();
    MatmulSchedulerTest::SetUp();
  }
};

using PrecisionsDesc = std::tuple<PrimDataType, PrimDataType, PrimDataType>;

using AbsoluteError = double;
using RelariveError = double;
using ErrorThresholds = std::pair<AbsoluteError, RelariveError>;
using TestCaseErrorThresholds = std::map<PrecisionsDesc, ErrorThresholds>;
class PrecisionParametrizedTest
    : public NVFuserFixtureParamTest<PrecisionsDesc> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  PrecisionParametrizedTest() : optimization_guard_(false) {}

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
};

[[nodiscard]] auto get_type_letter(const PrimDataType& type) {
  switch (type) {
    case PrimDataType::Half:
      return "H";
    case PrimDataType::Float:
      return "S";
    case PrimDataType::BFloat16:
      return "T";
    default:
      break;
  }
  NVF_THROW("Unsupported conversion of PrimDataType");
  return "*";
}

static const PrecisionsDesc HSH = std::make_tuple(
    PrimDataType::Half,
    PrimDataType::Float,
    PrimDataType::Half);
static const PrecisionsDesc HSS = std::make_tuple(
    PrimDataType::Half,
    PrimDataType::Float,
    PrimDataType::Float);
static const PrecisionsDesc TST = std::make_tuple(
    PrimDataType::BFloat16,
    PrimDataType::Float,
    PrimDataType::BFloat16);
static const PrecisionsDesc TSS = std::make_tuple(
    PrimDataType::BFloat16,
    PrimDataType::Float,
    PrimDataType::Float);

void checkUnsegmentedVectorization(
    const FusionExecutorCache& executor_cache,
    int64_t expected_vec_A,
    int64_t expected_vec_B,
    int64_t expected_vec_epilogue) {
  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();

  ASSERT_NE(runtime, nullptr);

  // expected to match whole fusion with single segment
  EXPECT_FALSE(runtime->isSegmented());

  ASSERT_TRUE(isSchedulerInUse(runtime, SchedulerType::Matmul));

  // Check that supported_vec_size matches expected.
  const MatmulParams* params = runtime->schedulerHeuristics()
                                   ->heuristicsList()
                                   .front()
                                   ->as<MatmulParams>();

  EXPECT_EQ(params->supported_vec_size.a, expected_vec_A);
  EXPECT_EQ(params->supported_vec_size.b, expected_vec_B);
  EXPECT_EQ(params->supported_vec_size.epilogue, expected_vec_epilogue);
}

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBias) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.0001, 0.0001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.0001, 0.0001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto accu_type = DataType(accu_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_accu_type = data_type_to_aten(accu_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := cast(bias)
  auto tv4 = maybeCastOp(accu_type, tv2);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv3, tv4);
  // tv6 := cast(tv5)
  auto tv6 = maybeCastOp(out_type, tv5);

  fusion->addOutput(tv6);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_accu_type);

  auto t5 = atBiasEpilogue(t3, t4);
  auto t6 = t5.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t6, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = relu(A x B)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueRelu) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.0001, 0.0001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.0001, 0.0001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  auto tv3 = relu(tv2);
  auto tv4 = maybeCastOp(out_type, tv3);

  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::relu(t2);
  auto t4 = t3.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t4, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = relu((A x B) + bias)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasRelu) {
  // NOTE: test skips Turing arch, the relative error was too big
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto accu_type = DataType(accu_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_accu_type = data_type_to_aten(accu_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});

  // tv4 := cast(bias)
  auto tv4 = maybeCastOp(accu_type, tv2);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv3, tv4);

  // tv6 := relu((A x B) + bias)
  auto tv6 = relu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_accu_type);
  auto t5 = atBiasEpilogue(t3, t4);
  auto t6 = at::relu(t5);
  auto t7 = t6.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference D tensor results
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t7, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = A x B;
//   Aux = relu(D)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueReluAux) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = maybeCastOp(out_type, tv2);
  auto tv4 = relu(tv2);
  auto tv5 = maybeCastOp(out_type, tv4);

  fusion->addOutput(tv3);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = t2.to(at_out_type);
  auto t4 = at::relu(t2);
  auto t5 = t4.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // D tensor results
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t3, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(
      at::allclose(outputs[1].as<at::Tensor>(), t5, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//   Aux = relu(D)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasReluAux) {
  // NOTE: test skips Turing arch, the relative error was too big
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto accu_type = DataType(accu_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_accu_type = data_type_to_aten(accu_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := cast(bias)
  auto tv4 = maybeCastOp(accu_type, tv2);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv3, tv4);

  // tv6 := cast((A x B) + bias)
  auto tv6 = maybeCastOp(out_type, tv5);

  // tv7 := relu((A x B) + bias)
  auto tv7 = relu(tv5);
  auto tv8 = maybeCastOp(out_type, tv7);

  fusion->addOutput(tv6);
  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_accu_type);
  auto t5 = atBiasEpilogue(t3, t4);
  auto t6 = t5.to(at_out_type);
  auto t7 = at::relu(t5);
  auto t8 = t7.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference D tensor results
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t6, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(
      at::allclose(outputs[1].as<at::Tensor>(), t8, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = gelu(A x B)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueGelu) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = gelu(tv2);
  auto tv4 = maybeCastOp(out_type, tv3);

  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::gelu(t2);
  auto t4 = t3.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t4, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = A x B
//   Aux = gelu(D)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueGeluAux) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  // TODO: Make these tests work with Hopper as well as Ampere
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = maybeCastOp(out_type, tv2);
  auto tv4 = gelu(tv2);
  auto tv5 = maybeCastOp(out_type, tv4);

  fusion->addOutput(tv3);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = t2.to(at_out_type);
  auto t4 = at::gelu(t2);
  auto t5 = t4.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // D tensor results
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t3, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(
      at::allclose(outputs[1].as<at::Tensor>(), t5, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion for Ampere:
//   D = gelu((A x B) + bias)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasGelu) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.01, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto accu_type = DataType(accu_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_accu_type = data_type_to_aten(accu_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := cast(bias)
  auto tv4 = maybeCastOp(accu_type, tv2);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv3, tv4);

  // tv6 := gelu((A x B) + bias)
  auto tv6 = gelu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_accu_type);
  auto t5 = atBiasEpilogue(t3, t4);
  auto t6 = at::gelu(t5);
  auto t7 = t6.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t7, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//   Aux = gelu(D)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasGeluAux) {
  // TODO: Make these tests work with Hopper as well as Ampere
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MmaLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.01, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, accu_prim_type, out_prim_type] = GetParam();
  const auto [abs_err_thr, rel_err_thr] = errs[GetParam()];

  const auto in_type = DataType(in_prim_type);
  const auto accu_type = DataType(accu_prim_type);
  const auto out_type = DataType(out_prim_type);
  const auto at_in_type = data_type_to_aten(in_prim_type);
  const auto at_accu_type = data_type_to_aten(accu_prim_type);
  const auto at_out_type = data_type_to_aten(out_prim_type);

  // NOTE: bfloat16 is not supported on pre-Ampere archs
  if (DataType::BFloat16 == in_type || DataType::BFloat16 == out_type) {
    NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := cast(bias)
  auto tv4 = maybeCastOp(accu_type, tv2);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv3, tv4);
  // tv6 := cast((A x B) + bias)
  auto tv6 = maybeCastOp(out_type, tv5);

  // tv7 := gelu((A x B) + bias)
  auto tv7 = gelu(tv5);
  auto tv8 = maybeCastOp(out_type, tv7);

  fusion->addOutput(tv6);
  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_accu_type);
  auto t5 = atBiasEpilogue(t3, t4);
  auto t6 = t5.to(at_out_type);
  auto t7 = at::gelu(t5);
  auto t8 = t7.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(
      executor_cache,
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(in_type),
      16l / dataTypeSize(out_type));

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference D tensor results
  NVF_CHECK(
      at::allclose(outputs[0].as<at::Tensor>(), t6, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(
      at::allclose(outputs[1].as<at::Tensor>(), t8, abs_err_thr, rel_err_thr));
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    MatmulSchedulerTest,
    PrecisionParametrizedTest,
    ::testing::Values(HSS, HSH, TSS, TST),
    [](const testing::TestParamInfo<PrecisionsDesc>& info) {
      std::ostringstream os;
      os << get_type_letter(std::get<0>(info.param));
      os << get_type_letter(std::get<1>(info.param));
      os << get_type_letter(std::get<2>(info.param));
      return os.str();
    });

TEST_F(MatmulSchedulerTest, FusedMultiplySumOnly) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 10, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t M = 128, N = 256, K = 512;
  TensorView* x = makeContigConcreteTensor({M, 1, K}, DataType::Half);
  TensorView* y = makeContigConcreteTensor({1, N, K}, DataType::Half);
  TensorView* z = fusedMultiplySum(x, y, {-1});

  fusion->addInput(x);
  fusion->addInput(y);
  fusion->addOutput(z);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto x_ref = at::randn({M, 1, K}, options);
  auto y_ref = at::randn({1, N, K}, options);
  auto z_ref = atMatmul(x_ref, y_ref, MmaLayout::TN);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto out_tensors = executor_cache.runFusionWithInputs({x_ref, y_ref});

  checkUnsegmentedVectorization(executor_cache, 8l, 8l, 4l);

  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {x_ref, y_ref},
      {z_ref},
      __LINE__,
      __FILE__);
}

// Matmul test that uses segmenter for 'C = A x B' fusion,
//   for Ampere with strict ref check, hence single layout check
TEST_F(MatmulSchedulerTest, BasicMatmulStrictCheckTT) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  const int M = 128, N = 256, K = 512;
  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0, t1, layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// Matmul test that reslies on segmenter for 'C = A x B' fusion, for Ampere
TEST_P(MatmulSchedulerTestWithLayout, BasicMatmulRelaxedCheck) {
  // TODO: Make these tests work with Hopper as well as Ampere
  // skip until we have Hopper support
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 2048;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test that reslies on segmenter for 'C = A x B' fusion, for Ampere
//  MMA first input is passed as second fusion parameter.
//  MMA second input is passed as first fusion parameter.
TEST_F(MatmulSchedulerTest, BasicMatmulInputShuffledTT) {
  // TODO: Make these tests work with Hopper as well as Ampere
  // skip until we have Hopper support
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 2048;
  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv1);
  fusion->addInput(tv0);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t1, t0});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = float2half(A x B)' fusion, for
//  Ampere
TEST_F(MatmulSchedulerTest, EpilogueOutputCast) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addOutput(tv3);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto tref = t2.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 8);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = alpha * (A x B)' fusion, for
//  Ampere
TEST_F(MatmulSchedulerTest, EpilogueAlpha) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, A - tv0, B - tv1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(s0);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = mul(s0, tv2);

  fusion->addOutput(tv3);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto tref = at::mul(t2, alpha).to(at::kFloat);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, alpha});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = float2half(alpha * (A x B))'
//  fusion, for Ampere
TEST_F(MatmulSchedulerTest, EpilogueAlphaOutputCast) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, A - tv0, B - tv1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(s0);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = mul(s0, tv2);
  auto tv4 = castOp(DataType::Half, tv3);

  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::mul(t2, alpha).to(at::kFloat);
  auto tref = t3.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, alpha});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 8);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = (A x B) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueBeta) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // beta - s0
  auto s0 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});

  // tv4 := beta * C
  auto tv4 = mul(s0, tv2);
  // tv5 := A x B + beta * C
  auto tv5 = add(tv3, tv4);

  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double beta = 2.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  auto t4 = at::mul(t2, beta).to(at::kFloat);
  auto t5 = at::add(t3, t4);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, beta});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t5, 0.01, 0.04));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = alpha * (A x B) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueAlphaBeta) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := alpha * (A x B)
  auto tv4 = mul(s0, tv3);

  // tv5 := beta * C
  auto tv5 = mul(s1, tv2);
  // tv6 := alpha * (A x B) + beta * C
  auto tv6 = add(tv4, tv5);

  fusion->addOutput(tv6);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);

  auto t5 = at::mul(t2, beta).to(at::kFloat);
  auto t6 = at::add(t4, t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t6, 0.001, 0.004));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = gelu(alpha * (A x B) + beta * C)
TEST_F(MatmulSchedulerTest, EpilogueAlphaBetaGeluOutputCast) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := alpha * (A x B)
  auto tv4 = mul(s0, tv3);

  // tv5 := beta * C
  auto tv5 = mul(s1, tv2);
  // tv6 := alpha * (A x B) + beta * C
  auto tv6 = add(tv4, tv5);
  // tv7 := gelu(alpha * (A x B) + beta * C)
  auto tv7 = gelu(tv6);
  // tv8 := half(gelu(alpha * (A x B) + beta * C))
  auto tv8 = castOp(DataType::Half, tv7);

  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);

  auto t5 = at::mul(t2, beta).to(at::kFloat);
  auto t6 = at::add(t4, t5);

  auto t7 = at::gelu(t6);
  auto t8 = t7.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 8);

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t8, 0.01, 0.06));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = alpha * ((A x B) + bias) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueAlphaBetaBias) {
  // TODO: Make these tests work with Hopper as well as Ampere
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2, bias - tv3
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);
  auto tv3 = makeContigTensor(1, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(s0);
  fusion->addInput(s1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv4 = fusedMultiplySum(tv0, tv1, {-1});

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv3);
  // tv6 := alpha * ((A x B) + bias)
  auto tv6 = mul(s0, tv5);
  // tv7 := beta * C
  auto tv7 = mul(s1, tv2);
  // tv8 := (alpha * ((A x B) + bias)) + (beta * C)
  auto tv8 = add(tv6, tv7);

  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kHalf, M, N, K);
  auto t3 = matmulAtInput2D(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K);

  auto t4 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  // t5 := (A x B) + bias
  auto t5 = atBiasEpilogue(t4, t3);
  // t6 := alpha * ((A x B) + bias)
  auto t6 = at::mul(t5, alpha).to(at::kFloat);
  // t7 := beta * C
  auto t7 = at::mul(t2, beta).to(at::kFloat);
  // t8 := (alpha * ((A x B) + bias)) + (beta * C)
  auto t8 = at::add(t6, t7);

  KernelArgumentHolder args = {t0, t1, t2, t3, alpha, beta};
  auto outputs = executor_cache.runFusionWithInputs(args);

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t8, 0.01, 0.01));
}

// Strided batch gemm test taht uses matmul scheduler, for Ampere:
//   D = (A x B)
TEST_P(MatmulSchedulerTestWithLayout, StridedBatch) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 248, B = 2;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // tv2 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
  auto t2 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative
  // verification
  //       caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t2, 0.0001, 0.0001));
}

// Strided batch gemm test with alpha and beta that uses matmul scheduler,
//  for Ampere architecture:
//   D = alpha * (A x B) + beta * C
TEST_P(MatmulSchedulerTestWithLayout, StridedBatchEpilogueAlphaBeta) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 248, B = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);
  auto tv2 = makeContigTensor(3, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := alpha * (A x B)
  auto tv4 = mul(s0, tv3);
  // tv5 := beta * C
  auto tv5 = mul(s1, tv2);
  // tv6 := alpha * (A x B) + beta * C
  auto tv6 = add(tv4, tv5);

  fusion->addOutput(tv6);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kFloat, M, N, K, B);

  auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);
  auto t5 = at::mul(t2, beta).to(at::kFloat);
  auto t6 = at::add(t4, t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative
  //  verification caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t6, 0.0001, 0.0001));
}

// Strided batch gemm test with alpha and beta scaling that uses matmul
// scheduler,
//  there is only single C tensor for whole batch; test for Ampere architecture:
//   D = alpha * (A x B) + beta * C
TEST_P(MatmulSchedulerTestWithLayout, StridedBatchEpilogueAlphaSingleBeta) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 248, B = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, C - tv2
  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := alpha * (A x B)
  auto tv4 = mul(s0, tv3);
  // tv5 := beta * C
  auto tv5 = mul(s1, tv2);
  // tv6 := bcast(beta * C)
  // [M, N] -> [B, M, N], with B as bcast
  auto tv6 = broadcast(tv5, {true, false, false});
  // tv7 := alpha * (A x B) + beta * C
  auto tv7 = add(tv4, tv6);

  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  at::manual_seed(0);
  const double alpha = 1.5;
  const double beta = 2.5;

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
  auto t2 = matmulAtInput2D(layout, TensorMatmulPos::C, at::kFloat, M, N, K);

  auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);
  auto t5 = at::mul(t2, beta).to(at::kFloat);
  // NOTE: t6, a result of adding an outer-most broadcast dimension to
  //  the result of scaling C with beta
  auto t6 = at::unsqueeze(t5, 0);
  auto t7 = at::add(t4, t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative
  //  verification caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t7, 0.0001, 0.0001));
}

// Strided batch gemm test with bias that uses matmul scheduler, for Ampere:
//   D = (A x B) + bias
TEST_P(MatmulSchedulerTestWithLayout, StridedBatchEpilogueBias) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 248, B = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, bias - tv2
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := (A x B) + bias
  auto tv4 = biasEpilogue(tv3, tv2);

  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K, B);

  auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = atBiasEpilogue(t3, t2).to(at::kFloat);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative
  //  verification caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t4, 0.0001, 0.0001));
}

// Strided batch gemm test with single bias vector that uses matmul
// scheduler, for Ampere:
//   D = (A x B) + bias
TEST_P(MatmulSchedulerTestWithLayout, StridedBatchEpilogueSingleBias) {
  // TODO: Make these tests work with Hopper as well as Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  const int M = 504, N = 136, K = 248, B = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1, bias - tv2
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);
  auto tv2 = makeContigTensor(1, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  // tv3 := A x B
  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // tv4 := (A x B) + bias
  auto tv4 = biasEpilogue(tv3, tv2);

  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const auto fusion_layout = getMatmulProblemLayout(fusion.get());
  NVF_CHECK(
      fusion_layout == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  at::manual_seed(0);
  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
  // Explicitly make bias tensor a single dim by passing 0 for batch
  auto t2 =
      matmulAtInput2D(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K, 0);

  auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = atBiasEpilogue(t3, t2).to(at::kFloat);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  // NOTE: increasted absolute tolerance to silence false negative
  //  verification caused by different way of calculating reference
  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), t4, 0.0001, 0.0001));
}

// Test matmul with contiguous inputs but sizes that are not divisible by 8 and
// with misaligned input pointers
TEST_P(MatmulSchedulerTestWithLayout, MisalignedVectorization) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  // TODO: parametrized test instead of nested loops (still use a loop over
  // sizes and re-use FusionExecutorCache)
  for (bool add_2d_bias : {false, true}) {
    for (bool downcast_output : {false, true}) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());

      auto tv0 = makeContigTensor(2, DataType::Half);
      auto tv1 = makeContigTensor(2, DataType::Half);
      fusion->addInput(tv0);
      fusion->addInput(tv1);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

      if (add_2d_bias) {
        auto bias = makeContigTensor(2, DataType::Half);
        fusion->addInput(bias);
        tv2 = add(tv2, bias);
      }

      if (downcast_output) {
        tv2 = castOp(DataType::Half, tv2);
      }

      fusion->addOutput(tv2);

      const auto fusion_layout = getMatmulProblemLayout(fusion.get());
      NVF_CHECK(
          fusion_layout == layout,
          "mismatch between test layout (",
          toString(layout),
          ") and layout inferred from fusion definition (",
          toString(fusion_layout),
          ")");

      FusionExecutorCache executor_cache(std::move(fusion));

      auto run = [&](int M,
                     int N,
                     int K,
                     // Pointer alignment
                     int align_A,
                     int align_B,
                     int align_bias,
                     int expected_vec_A,
                     int expected_vec_B,
                     int expected_vec_epilogue) {
        const auto maybeUnalign = [](const at::Tensor& t, int offset) {
          if (offset == 16 / t.element_size()) {
            // Already fully aligned
            return t;
          }
          return at::pad(t.ravel(), {{0, offset}})
              .index({at::indexing::Slice(offset, t.numel() + offset, 1)})
              .view({t.size(0), t.size(1)});
        };

        auto t0 = maybeUnalign(
            matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K),
            align_A);
        auto t1 = maybeUnalign(
            matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K),
            align_B);

        auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

        KernelArgumentHolder args = {t0, t1};

        if (add_2d_bias) {
          const auto options =
              at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
          auto bias = maybeUnalign(at::randn({M, N}, options), align_bias);
          tref = tref + bias;
          args.push(bias);
        }

        if (downcast_output) {
          tref = tref.to(at::kHalf);
        }

        auto outputs = executor_cache.runFusionWithInputs(args);

        FusionKernelRuntime* runtime =
            executor_cache.getMostRecentKernelRuntime();

        ASSERT_NE(runtime, nullptr);

        // expected to match whole fusion with single segment
        EXPECT_FALSE(runtime->isSegmented());

        ASSERT_TRUE(isSchedulerInUse(runtime, SchedulerType::Matmul));

        // Check that supported_vec_size matches expected.
        const MatmulParams* params = runtime->schedulerHeuristics()
                                         ->heuristicsList()
                                         .front()
                                         ->as<MatmulParams>();

        EXPECT_EQ(params->supported_vec_size.a, expected_vec_A);
        EXPECT_EQ(params->supported_vec_size.b, expected_vec_B);
        EXPECT_EQ(params->supported_vec_size.epilogue, expected_vec_epilogue);

        EXPECT_TRUE(
            at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
      };

      [[maybe_unused]] bool contig_K_A =
          layout == MmaLayout::TT || layout == MmaLayout::TN;
      [[maybe_unused]] bool contig_K_B =
          layout == MmaLayout::TN || layout == MmaLayout::NN;

      // When not downcasting, outputs are Float
      [[maybe_unused]] int max_vec_epi = downcast_output ? 8 : 4;

      // all fully vectorizable in all layouts
      run(504, 136, 248, 8, 8, 8, 8, 8, max_vec_epi);
      // odd K. Operands vectorizable when K is not the contiguous axis.
      // Output always fully vectorizable
      run(504,
          136,
          249,
          8,
          8,
          8,
          contig_K_A ? 1 : 8,
          contig_K_B ? 1 : 8,
          max_vec_epi);
      // Odd N. Output not vectorizable. A fully vectorizable. B fully
      // vectorizable unless N is the contiguous dim.
      run(504, 137, 248, 8, 8, 8, 8, contig_K_B ? 8 : 1, 1);
      // Odd M. Output fully vectorizable. B fully vectorizable. A fully
      // vectorizable unless M is the contiguous dim.
      run(505, 136, 248, 8, 8, 8, contig_K_A ? 8 : 1, 8, max_vec_epi);
      // Odd M and N. Output not vectorizable. A and B fully vectorizable
      // unless K is not the contiguous dim.
      run(505, 137, 248, 8, 8, 8, contig_K_A ? 8 : 1, contig_K_B ? 8 : 1, 1);
      // Odd M, N, K. None vectorizable.
      run(505, 137, 249, 8, 8, 8, 1, 1, 1);
      // Cases with vectorizable strides but misaligned base pointers
      // A not vectorizable due to pointer offset
      run(504, 136, 248, 2, 8, 8, 2, 8, max_vec_epi);
      // B not vectorizable due to pointer offset
      run(504, 136, 248, 8, 2, 8, 8, 2, max_vec_epi);
      run(504, 136, 248, 8, 8, 2, 8, 8, add_2d_bias ? 2 : max_vec_epi);
    }
  }
}

// Test matmul with strided inputs. This tests that vectorization is properly
// computed.
TEST_P(MatmulSchedulerTestWithLayout, StridedInputs) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  for (bool add_2d_bias : {false, true}) {
    for (bool downcast_output : {false, true}) {
      auto run = [&](int M,
                     int N,
                     int K,
                     // Pointer alignment
                     int align_A,
                     int align_B,
                     int align_bias,
                     // Whether to mark the TensorViews as contiguous
                     bool contiguous_inner_dim_A,
                     bool contiguous_inner_dim_B,
                     bool contiguous_inner_dim_bias,
                     // Padding
                     int pad_A,
                     int pad_B,
                     int pad_bias,
                     int expected_vec_A,
                     int expected_vec_B,
                     int expected_vec_epilogue) {
        auto fusion = std::make_unique<Fusion>();
        FusionGuard fg(fusion.get());

        // Inputs are contiguous in their inner dimension but discontiguous
        // in the outer dim.
        auto tv0 = TensorViewBuilder()
                       .ndims(2)
                       .contiguity({false, contiguous_inner_dim_A})
                       .dtype(DataType::Half)
                       .build();
        auto tv1 = TensorViewBuilder()
                       .ndims(2)
                       .contiguity({false, contiguous_inner_dim_B})
                       .dtype(DataType::Half)
                       .build();

        fusion->addInput(tv0);
        fusion->addInput(tv1);

        tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
        tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
        auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

        if (add_2d_bias) {
          auto bias = TensorViewBuilder()
                          .ndims(2)
                          .contiguity({false, contiguous_inner_dim_bias})
                          .dtype(DataType::Half)
                          .build();
          fusion->addInput(bias);
          tv2 = add(tv2, bias);
        }

        if (downcast_output) {
          tv2 = castOp(DataType::Half, tv2);
        }

        fusion->addOutput(tv2);

        const auto fusion_layout = getMatmulProblemLayout(fusion.get());
        NVF_CHECK(
            fusion_layout == layout,
            "mismatch between test layout (",
            toString(layout),
            ") and layout inferred from fusion definition (",
            toString(fusion_layout),
            ")");

        FusionExecutorCache executor_cache(std::move(fusion));

        // stride to introduce pad in the inner-most dimension, and shift
        // data pointer by offset
        const auto padAndUnalign2D =
            [](const at::Tensor& t, int pad_to, int offset) {
              // Determine new strides. We pad the contiguous axis by
              // increasing the other stride to the next highest multiple of 8
              std::vector<int64_t> new_strides(t.ndimension(), 0);
              int64_t linear_size = 1;
              for (size_t i : arange(t.ndimension())) {
                new_strides[i] = t.stride((int64_t)i);
                if (new_strides[i] != 1) {
                  // Pad contiguous dimension by modifying other stride. This
                  // only works for 2D tensors.
                  new_strides[i] += pad_to;
                }
                // Use strides to determine space needed for padded tensor
                linear_size += t.size((int64_t)i) * new_strides[i];
              }

              at::Tensor out = at::as_strided(
                  at::empty({linear_size}, t.options())
                      .index({at::indexing::Slice(
                          offset, linear_size + offset, 1)}),
                  t.sizes(),
                  new_strides);
              out.copy_(t);
              return out;
            };

        auto t0 = padAndUnalign2D(
            matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K),
            pad_A,
            align_A);
        auto t1 = padAndUnalign2D(
            matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K),
            pad_B,
            align_B);

        auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

        KernelArgumentHolder args = {t0, t1};

        if (add_2d_bias) {
          const auto options =
              at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
          auto bias =
              padAndUnalign2D(at::randn({M, N}, options), pad_bias, align_bias);
          tref = tref + bias;
          args.push(bias);
        }

        if (downcast_output) {
          tref = tref.to(at::kHalf);
        }

        auto outputs = executor_cache.runFusionWithInputs(args);

        FusionKernelRuntime* runtime =
            executor_cache.getMostRecentKernelRuntime();

        ASSERT_NE(runtime, nullptr);

        // expected to match whole fusion with single segment
        EXPECT_FALSE(runtime->isSegmented());

        ASSERT_TRUE(isSchedulerInUse(runtime, SchedulerType::Matmul));

        // Check that supported_vec_size matches expected.
        const MatmulParams* params = runtime->schedulerHeuristics()
                                         ->heuristicsList()
                                         .front()
                                         ->as<MatmulParams>();

        EXPECT_EQ(params->supported_vec_size.a, expected_vec_A);
        EXPECT_EQ(params->supported_vec_size.b, expected_vec_B);
        EXPECT_EQ(params->supported_vec_size.epilogue, expected_vec_epilogue);

        EXPECT_TRUE(
            at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
      };

      [[maybe_unused]] bool contig_K_A =
          layout == MmaLayout::TT || layout == MmaLayout::TN;
      [[maybe_unused]] bool contig_K_B =
          layout == MmaLayout::TN || layout == MmaLayout::NN;

      // When not downcasting, outputs are Float
      [[maybe_unused]] int max_vec_epi = downcast_output ? 8 : 4;

      // Pad outer stride of A by 1. M and K are even, so no vectorization of
      // A is possible despite compatible sizes.
      run(504, 136, 248, 8, 8, 8, true, true, true, 1, 0, 0, 1, 8, max_vec_epi);
      // Pad outer stride of B by 1. N and K are even, so no vectorization of
      // B is possible despite compatible sizes.
      run(504, 136, 248, 8, 8, 8, true, true, true, 0, 1, 0, 8, 1, max_vec_epi);
      // Padding by 2 from a multiple of 8 means we can only vectorize at
      // width 2
      run(504,
          136,
          248,
          8,
          8,
          8,
          true,
          true,
          true,
          2,
          2,
          2,
          2,
          2,
          add_2d_bias ? 2 : max_vec_epi);
      // Incompatible sizes are not vectorized despite padding to compatible
      // strides
      run(505,
          136,
          249,
          8,
          8,
          8,
          true,
          true,
          true,
          1,
          0,
          0,
          1,
          contig_K_B ? 1 : 8,
          max_vec_epi);
      run(504,
          137,
          249,
          8,
          8,
          8,
          true,
          true,
          true,
          0,
          1,
          0,
          contig_K_A ? 1 : 8,
          1,
          1);

      // Test that declaring a tensor's inner dimension discontiguous in the
      // Fusion means we don't hit an error even if the inputs would support
      // vectorization.
      run(504,
          136,
          248,
          8,
          8,
          8,
          false,
          true,
          true,
          0,
          0,
          0,
          1,
          8,
          max_vec_epi);
      run(504,
          136,
          248,
          8,
          8,
          8,
          true,
          false,
          true,
          0,
          0,
          0,
          8,
          1,
          max_vec_epi);
      run(504,
          136,
          248,
          8,
          8,
          8,
          true,
          true,
          false,
          0,
          0,
          0,
          8,
          8,
          add_2d_bias ? 1 : max_vec_epi);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulSchedulerTestWithLayout,
    testing::ValuesIn(kAllSupportedMmaLayout),
    mmaLayoutName);

class TestKernelConfig : public matmul_heuristic_plugin::KernelConfig {
  void configure() override {
    // Set load_stages to 0, which is an allowed value (with a warning), but not
    // one that will be set by our default scheduler. This lets us use it as a
    // sentinel to check that this heuristic was run.
    load_stages = (uint8_t)0;
  }
};

std::unique_ptr<matmul_heuristic_plugin::KernelConfig> testConfigFactory() {
  return std::unique_ptr<matmul_heuristic_plugin::KernelConfig>(
      new TestKernelConfig);
}

class MatmulSchedulerPluginTest : public NVFuserTest {
 protected:
  MatmulSchedulerPluginTest()
      : optimization_guard_(false), factory_guard_(testConfigFactory) {}

 private:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
  matmul_heuristic_plugin::KernelConfigFactoryGuard factory_guard_;
};

// Test that our fake plugin works to override the default heuristic
TEST_F(MatmulSchedulerPluginTest, BasicMatmul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  const int M = 128, N = 256, K = 512;
  const auto layout = MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion->addOutput(tv2);

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0, t1, layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  // enable profiling so that executor logs are captured
  executor_cache.profile(true);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  checkUnsegmentedVectorization(executor_cache, 8, 8, 4);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();

  ASSERT_NE(runtime, nullptr);

  NVF_CHECK(
      !runtime->isSegmented(),
      "fusion got segmented, expected to match whole fusion with single "
      "segment");

  NVF_CHECK(
      isSchedulerInUse(runtime, SchedulerType::Matmul),
      "matmul scheduler was not used to handle prepared fusion");

  HeuristicParams* heur = runtime->getMostRecentExecutorLog().params.get();
  ASSERT_NE(heur, nullptr);
  ASSERT_TRUE(heur->isA<MatmulParams>());
  MatmulParams* mmheur = heur->as<MatmulParams>();
  EXPECT_EQ(mmheur->circular_buffer_options.smem_circular_buffer_stage, 0);

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// Test that we can segment a fusion that has a MatmulOp with epilogue
// TODO: Once we can control the ExprEval and Matmul schedulers via options, run
// this test with all three combinations (with and without each scheduler, but
// at least one enabled).
TEST_F(MatmulSchedulerTest, SegmentMatmulOpPrologue) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Prologue prevents ExprEval scheduler from accepting. If Matmul scheduler
  // rejects, then Pointwise must not accept this unsegmented fusion.
  tv1 = castOp(DataType::Half, sin(tv1));

  auto tv2 = matmul(tv0, tv1);

  fusion->addOutput(tv2);

  NVF_CHECK(
      ir_utils::getOpsOfType<MatmulOp>(fusion.get()).size() == 1,
      "matmul fusion must have at least one MatmulOp");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({K, N}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  // TODO: check vectorization if fusion is enabled
  // checkUnsegmentedVectorization(executor_cache, 8, 8, 8);

  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);
}

// This is just like the above test but with LinearOp instead of MatmulOp
TEST_F(MatmulSchedulerTest, SegmentLinearOpPrologue) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Prologue prevents ExprEval scheduler from accepting. If Matmul scheduler
  // rejects, then Pointwise must not accept this unsegmented fusion.
  tv1 = castOp(DataType::Half, sin(tv1));

  auto tv2 = linear(tv0, tv1);

  fusion->addOutput(tv2);

  NVF_CHECK(
      ir_utils::getOpsOfType<LinearOp>(fusion.get()).size() == 1,
      "matmul fusion must have at least one MmaOp");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  // TODO: check vectorization if fusion is enabled
  // checkUnsegmentedVectorization(executor_cache, 8, 8, 8);

  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test that the matmul scheduler refuses to translate a matmul that is not
// Half or BFloat16
TEST_F(MatmulSchedulerTest, SegmentMatmulOpUnsupportedDtype) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Float);
  auto tv1 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // Prologue prevents ExprEval scheduler from accepting. If Matmul scheduler
  // rejects, then Pointwise must not accept this unsegmented fusion.
  tv1 = castOp(DataType::Float, sin(tv1));

  auto tv2 = matmul(tv0, tv1);

  fusion->addOutput(tv2);

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({K, N}, options);

  // Enable MatmulOp fusion, which should reject because float operands are not
  // supported.
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMatmul);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();

  EXPECT_TRUE(runtime->isSegmented());

  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(MatmulSchedulerTest, PreBroadcastMmaBiasNeg) {
  // TODO: fix up params or switch to FusionExecutorCache when ready, then
  // enable Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, DataType::Half);
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, DataType::Half);
  TensorView* tv2 = makeContigConcreteTensor({-1}, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
  // We add these computations to test
  // scheduling (with epilogue) when the ouptut of mma is not
  // cast to half.
  auto tv4 = maybeCastOp(DataType::Float, tv2);
  auto tv5 = biasEpilogue(tv3, tv4);
  auto tv6 = neg(tv5);

  fusion->addOutput(tv6);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto a = at::randn({M, K}, options);
  auto b = at::randn({N, K}, options);
  auto c = at::randn({M}, options);
  auto t0 = a.unsqueeze(1);
  auto t1 = b.unsqueeze(0);
  auto tref =
      atBiasEpilogue(
          at::matmul(a.to(at::kFloat), b.to(at::kFloat).t()), c.to(at::kFloat))
          .neg_();

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Hopper_64_64_16;
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 16);
  gemm_tile.warp_tile = GemmTile(64, 64, 16);
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 2;
  mparams.use_smem_epilogue = true;
  mparams.promote_prologue_smem_reuse = false;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1, c}, LaunchParams(), matmul_cparams);
  auto outputs = ke.run({t0, t1, c});

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Test automatically scheduling a fusion that requires 64-bit indexing
TEST_F(MatmulSchedulerTest, EpilogueFusionInt64Indexing) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMatmul);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigConcreteTensor({-1, -1}, DataType::Half);
  auto tv1 = makeContigConcreteTensor({-1, -1}, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);
  auto tv3 = maybeCastOp(DataType::Float, tv2);
  auto tv4 = neg(tv3);
  auto tv5 = castOp(DataType::Half, tv4);

  fusion->addOutput(tv5);

  // We use an input size large enough to require 64-bit indexing to index some
  // elements of the inputs.
  // NOTE: This size requires 64-bit indexing in the epilogue only.
  const int M = 1 << 16, N = 1 << 15, K = 1 << 8;

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  at::Tensor tref = -at::matmul(t0, t1);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

class MatmulFusionTest
    : public MatmulSchedulerTest,
      public ::testing::WithParamInterface<std::pair<bool, bool>> {
 protected:
  void SetUp() override {
    if (fusion_enabled) {
      EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMatmul);
    }
    if (horizontal_fusion_enabled) {
      EnableOptionsGuard::getCurOptions().set(
          EnableOption::FuseMultipleMatmuls);
    }
  }

  bool fusion_enabled = GetParam().first;
  bool horizontal_fusion_enabled = GetParam().second;
};

// Test that we can segment a Fusion containing two matmuls
TEST_P(MatmulFusionTest, Llama2FFN) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = matmul(tv0, tv1);
  auto tv4 = matmul(tv0, tv2);

  // silu
  auto tv5 = mul(sigmoid(tv3), tv3);

  auto tv6 = mul(tv5, tv4);

  fusion->addOutput(tv6);

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({K, N}, options);
  auto t2 = at::randn({K, N}, options);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  at::Tensor t3, t4;
  if (fusion_enabled) {
    // More accurate reference, which is what nvfuser computes
    t3 = at::matmul(t0.to(at::kFloat), t1.to(at::kFloat));
    t4 = at::matmul(t0.to(at::kFloat), t2.to(at::kFloat));
  } else {
    // at::matmul downcasts to half-precision, and we cast back up, which loses
    // precision. Computing a reference as above leads to allclose failure, so
    // we use the less accurate reference in this case.
    t3 = at::matmul(t0, t1).to(at::kFloat);
    t4 = at::matmul(t0, t2).to(at::kFloat);
  }
  at::Tensor tref = ((at::sigmoid(t3) * t3) * t4);

  NVF_CHECK(at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));

  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();

  size_t expected_kernels =
      fusion_enabled ? (horizontal_fusion_enabled ? 1 : 2) : 3;

  EXPECT_EQ(runtime->fusionSegments()->groups().size(), expected_kernels);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulFusionTest,
    ::testing::ValuesIn(std::vector<std::pair<bool, bool>>{
        {false, false},
        {true, false},
        {true, true}}),
    [](const testing::TestParamInfo<std::pair<bool, bool>>& info) {
      bool fuse = info.param.first;
      bool horiz_fuse = info.param.second;
      if (horiz_fuse) {
        NVF_ERROR(
            fuse, "Horizontal fusion enabled but overall fusion disabled");
      }
      return fuse ? (horiz_fuse ? "fuse_horizontal" : "fuse_single")
                  : "dontfuse";
    });

// This test can be used to check that an external plugin has been loaded. It
// is DISABLED_ so that the test suite will pass even if the user has not
// provided a plugin via NVFUSER_MATMUL_HEURISTIC_PLUGIN. To check that a
// plugin can be loaded properly, invoke the test suite like so:
//
//   export NVFUSER_MATMUL_HEURISTIC_PLUGIN=/path/to/plugin.so
//   build/test_matmul --gtest_also_run_disabled_tests
//
TEST_F(MatmulSchedulerTest, DISABLED_RequireExternalPlugin) {
  EXPECT_TRUE(matmul_heuristic_plugin::hasPlugin());

  MatmulParams params;
}

class AllocationDomainTest
    : public NVFuserFixtureParamTest<std::tuple<bool, bool>> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  AllocationDomainTest() : optimization_guard_(false) {
    MatMulTileOptions gemm_tile;

    if (cudaArchGuardShouldSkip(7, 5, 9, 0)) {
      // Hopper or above
      gemm_tile.cta_tile = GemmTile(128, 256, 64);
      gemm_tile.warp_tile = GemmTile(64, 128, 64);
      mparams.mma_macro = MmaMacro::Hopper_64_128_16;
    } else {
      // Ampere
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    }
    mparams.supported_vec_size = {8, 8, 4};
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  }

  std::pair<TensorView*, TensorView*> getInputTVs(
      int M,
      int N,
      int K,
      bool a_m_inner,
      bool b_k_inner) {
    auto tv0 = makeContigConcreteTensor({M, K}, DataType::Half);
    auto tv1 = makeContigConcreteTensor({K, N}, DataType::Half);

    if (a_m_inner) {
      tv0->setAllocationDomain({tv0->axis(1), tv0->axis(0)}, true);
    }

    if (b_k_inner) {
      tv1->setAllocationDomain({tv1->axis(1), tv1->axis(0)}, true);
    }
    return {tv0, tv1};
  }

  std::pair<at::Tensor, at::Tensor> getInputTensors(
      int M,
      int N,
      int K,
      bool a_m_inner,
      bool b_k_inner) {
    const auto options =
        at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0 /*device*/);
    auto t0 = a_m_inner ? at::randn({M, K}, options).as_strided({M, K}, {1, M})
                        : at::randn({M, K}, options);
    auto t1 = b_k_inner ? at::randn({K, N}, options).as_strided({K, N}, {1, K})
                        : at::randn({K, N}, options);
    return {t0, t1};
  }

  MatmulParams mparams;

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
};

// This tests fusions where inputs to a Matmul will have the root domains
// [M, K] and [K, N], and all possible combinations of allocation domains.
// Please note that inpout in B is transposed prior to creating a Mma op.
TEST_P(AllocationDomainTest, BasicMatmul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 10, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 512, N = 128, K = 256;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  // This has rfactor: {N, K}
  auto tv1t = transpose(tv1);
  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1t, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});
  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Same as above but without the the input tv1 being transposed.
TEST_P(AllocationDomainTest, BasicMatmulNoTranspose) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 512, N = 128, K = 256;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // [M, K, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});
  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(AllocationDomainTest, BasicMatmulWithPrologueSet) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 128, N = 256, K = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tvSet = set(tv1);
  // K, N -> N, K
  auto tv1t = transpose(tvSet);

  // M, N, K
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1t, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});
  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(AllocationDomainTest, BasicMatmulWithPrologueSetCastSin) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 128, N = 256, K = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tvSet = set(tv1);
  // K, N -> N, K
  auto tv1t = transpose(tvSet);
  auto tv1tc = castOp(DataType::Half, sin(tv1t));

  // M, N, K
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1tc, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.sin().to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test for Ampere MMA: across supported layouts
TEST_P(AllocationDomainTest, BasicMatmulWithPrologueSetCastSinNoTranspose) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 128, N = 256, K = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tvSet = set(tv1);
  auto tv1tc = castOp(DataType::Half, sin(tvSet));

  // M, K, N
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1tc, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.sin().to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(AllocationDomainTest, BasicMatmulWithPrologueSetCastSinSetNoTranspose) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 128, N = 256, K = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tvSet = set(tv1);
  auto tv1tc = castOp(DataType::Half, sin(tvSet));
  auto tv1tcs = set(tv1tc);

  // M, K, N
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1tcs, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.sin().to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(AllocationDomainTest, MatmulWithPrologueSetCastSinTranspose) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  bool a_m_inner = std::get<0>(GetParam());
  bool b_k_inner = std::get<1>(GetParam());

  const int M = 128, N = 256, K = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto [tv0, tv1] = getInputTVs(M, N, K, a_m_inner, b_k_inner);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tvSet = set(tv1);
  auto tv1tc = castOp(DataType::Half, sin(tvSet));
  auto tv1tct = transpose(tv1tc);

  // M, N, K
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1tct, {true, false, false});
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion->addOutput(tv2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(fusion.get(), &mparams);

  auto [t0, t1] = getInputTensors(M, N, K, a_m_inner, b_k_inner);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = ke.run({t0, t1});
  auto tref = t0.to(at::kFloat).matmul(t1.sin().to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

INSTANTIATE_TEST_SUITE_P(
    MatmulSchedulerTest,
    AllocationDomainTest,
    testing::Combine(testing::Bool(), testing::Bool()));

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

TEST_F(MatmulSchedulerTest, OperandOrderIssue2434) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  int M = 32, N = 64, K = 128;

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* b = makeContigConcreteTensor({N, K}, DataType::BFloat16);
  TensorView* a = makeContigConcreteTensor({M, K}, DataType::BFloat16);

  TensorView* bbcast = broadcast(b, {true, false, false});
  TensorView* abcast = broadcast(a, {false, true, false});
  TensorView* ab = mul(abcast, bbcast);
  TensorView* mm = sum(ab, {-1});

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(mm);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  auto tref = at::linear(t0.to(at::kFloat), t1.to(at::kFloat));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test for Hopper+ (Hopper, Blackwell)

using HopperPlusMatmulSchedulerTestParams = std::tuple<
    bool, // use_smem_epilogue
    bool, // a_k_inner
    bool, // b_k_inner
    int64_t, // M
    int64_t, // N
    int64_t, // K
    MmaMacro,
    int64_t // SplitK Factor
    >;

std::string hopperPlusTestName(
    const testing::TestParamInfo<HopperPlusMatmulSchedulerTestParams>& info) {
  std::ostringstream os;
  bool use_smem_epilogue;
  bool a_k_inner, b_k_inner;
  int64_t M, N, K;
  MmaMacro mma_macro;
  int64_t splitk_factor;
  std::tie(
      use_smem_epilogue,
      a_k_inner,
      b_k_inner,
      M,
      N,
      K,
      mma_macro,
      splitk_factor) = info.param;
  os << (a_k_inner ? "K" : "M");
  os << (b_k_inner ? "K" : "N");
  os << "_" << M << "_" << N << "_" << K;
  os << "_MmaMacro_" << macroToString(mma_macro);
  if (use_smem_epilogue) {
    os << "_tma_store";
  }
  if (splitk_factor > 1) {
    os << "_splitk_" << splitk_factor;
  }
  return os.str();
}

std::string hopperPlusTestNameSwizzle(
    const testing::TestParamInfo<HopperPlusMatmulSchedulerTestParams>& info) {
  std::unordered_map<MmaMacro, std::string> mma_macro_to_swizzle_str_map = {
      {MmaMacro::Hopper_64_256_16, "128BSwizzle"},
      {MmaMacro::Hopper_64_128_16, "128BSwizzle"},
      {MmaMacro::Hopper_64_64_16, "128BSwizzle"},
      {MmaMacro::Hopper_64_32_16, "64BSwizzle"},
      {MmaMacro::Hopper_64_16_16, "32BSwizzle"},
      {MmaMacro::Blackwell1CTA_128_256_16, "128BSwizzle"},
      {MmaMacro::Blackwell1CTA_128_128_16, "128BSwizzle"},
      {MmaMacro::Blackwell1CTA_128_64_16, "128BSwizzle"},
      {MmaMacro::Blackwell1CTA_128_32_16, "64BSwizzle"},
      {MmaMacro::Blackwell1CTA_128_16_16, "32BSwizzle"}};
  MmaMacro mma_macro = std::get<6>(info.param);
  std::ostringstream os;
  os << hopperPlusTestName(info);
  os << "_" << mma_macro_to_swizzle_str_map.at(mma_macro);
  return os.str();
}

class HopperPlusMatmulSchedulerTest
    : public NVFuserFixtureParamTest<HopperPlusMatmulSchedulerTestParams> {
 protected:
  void SetUp() {
    std::tie(
        use_smem_epilogue,
        a_k_inner,
        b_k_inner,
        M,
        N,
        K,
        mma_macro,
        splitk_factor) = GetParam();

    if (isHopper(mma_macro)) {
      NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);
    } else {
      NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(10, 0, 11, 0);
      if (use_smem_epilogue) {
        GTEST_SKIP() << "TMA store is not supported for Blackwell yet.";
      }
    }

    if (a_k_inner) {
      layout = b_k_inner ? MmaLayout::TN : MmaLayout::TT;
    } else {
      layout = b_k_inner ? MmaLayout::NN : MmaLayout::NT;
    }

    fusion_up = std::make_unique<Fusion>();
    fusion = fusion_up.get();
    fusion_guard = std::make_unique<FusionGuard>(fusion);

    // Create custom Matmul Params
    MatMulTileOptions gemm_tile;
    // TODO cta tile is a multiple of mma macro for hopper.
    // Default cta_tile configuration is 2-CTA.
    gemm_tile.cta_tile =
        GemmTile(2 * getM(mma_macro), getN(mma_macro), 2 * getK(mma_macro));

    // TODO warp tile is (macroM, macroN, macroK) for hopper.
    gemm_tile.warp_tile =
        GemmTile(getM(mma_macro), getN(mma_macro), 2 * getK(mma_macro));

    mparams.supported_vec_size = {8, 8, 8};

    mparams.mma_macro = mma_macro;

    mparams.use_smem_epilogue = use_smem_epilogue;

    mparams.splitk_factor = splitk_factor;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 2;
  }

  void TearDown() {
    if (testing::Test::IsSkipped() || testing::Test::HasFailure()) {
      return;
    }

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion).size(),
        "matmul fusion must have exactly one MmaOp");

    // Schedule matmul fusion using custom parameters
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(fusion, &mparams);

    KernelExecutor ke;
    ke.compile(fusion, inputs, LaunchParams(), matmul_cparams);
    auto nvf_out = ke.run(inputs);
    EXPECT_TRUE(at::allclose(nvf_out[0].as<at::Tensor>(), tref, 1e-2, 1e-2));
  }

 protected:
  bool use_smem_epilogue;
  bool a_k_inner, b_k_inner;
  int64_t M, N, K;
  MmaMacro mma_macro;
  int64_t splitk_factor;
  std::unique_ptr<Fusion> fusion_up;
  Fusion* fusion;
  std::unique_ptr<FusionGuard> fusion_guard;
  DataType dtype = DataType::Half;

  MmaLayout layout;

  MatmulParams mparams;

  KernelArgumentHolder inputs;

  // Tests should place the reference tensor here
  at::Tensor tref;
};

TEST_P(HopperPlusMatmulSchedulerTest, FusedMultiplySum) {
  const auto& [A, B] =
      matmulAtInput3DSS(M, N, K, layout, data_type_to_aten(dtype));
  inputs = {A, B};

  TensorView* tv0 = nullptr;
  TensorView* tv1 = nullptr;
  std::unordered_map<int64_t, int64_t> old2new;
  int64_t k_axis = 0;

  switch (layout) {
    case MmaLayout::TT:
      // Inner dims KN, order is MKN
      tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      tv1 = makeContigConcreteTensor({1, -1, -1}, dtype);
      old2new = {{-2, -1}, {-1, -2}};
      k_axis = -2;
      break;
    case MmaLayout::TN:
      // Inner dims KK, order is MNK
      tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype);
      tv1 = makeContigConcreteTensor({1, -1, -1}, dtype);
      old2new = {};
      k_axis = -1;
      break;
    case MmaLayout::NT:
      // Inner dims MN, order is KMN
      tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype);
      old2new = {{-3, -1}};
      k_axis = -3;
      break;
    case MmaLayout::NN:
      // Inner dims MK, order is NKM
      tv0 = makeContigConcreteTensor({1, -1, -1}, dtype);
      tv1 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      old2new = {{-1, -3}};
      k_axis = -2;
      break;
  }

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {k_axis});

  // Reorder the accumulator as [M, N, K]
  tv2->reorder(old2new);
  tv2->commitLeafToLogical();

  auto tv3 = castOp(dtype, tv2);
  fusion->addOutput(tv3);

  tref = atMatmul(A.squeeze(), B.squeeze(), layout);
}

// TODO: Remove this test once the architecture agnostic can be
// run on hopper.
TEST_P(HopperPlusMatmulSchedulerTest, FusedMultiplySumBiasNeg) {
  const auto& [A, B] =
      matmulAtInput3DSS(M, N, K, layout, data_type_to_aten(dtype));
  const auto& C = matmulAtInput2D(
      layout, TensorMatmulPos::Bias, data_type_to_aten(dtype), M, N, K);
  inputs = {A, B, C};

  TensorView* tv0 = nullptr;
  TensorView* tv1 = nullptr;
  std::unordered_map<int64_t, int64_t> old2new;
  int64_t k_axis = 0;

  switch (layout) {
    case MmaLayout::TT:
      // Inner dims KN, order is MKN
      tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      tv1 = makeContigConcreteTensor({1, -1, -1}, dtype);
      old2new = {{-2, -1}, {-1, -2}};
      k_axis = -2;
      break;
    case MmaLayout::TN:
      // Inner dims KK, order is MNK
      tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype);
      tv1 = makeContigConcreteTensor({1, -1, -1}, dtype);
      old2new = {};
      k_axis = -1;
      break;
    case MmaLayout::NT:
      // Inner dims MN, order is KMN
      tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype);
      old2new = {{-3, -1}};
      k_axis = -3;
      break;
    case MmaLayout::NN:
      // Inner dims MK, order is NKM
      tv0 = makeContigConcreteTensor({1, -1, -1}, dtype);
      tv1 = makeContigConcreteTensor({-1, -1, 1}, dtype);
      old2new = {{-1, -3}};
      k_axis = -2;
      break;
  }
  TensorView* tv2 = makeContigConcreteTensor({-1}, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {k_axis});

  // Reorder the accumulator as [M, N, K]
  tv3->reorder(old2new);
  tv3->commitLeafToLogical();

  auto* tv4 = maybeCastOp(DataType::Float, tv2);
  auto* tv5 = biasEpilogue(tv3, tv4);
  auto* tv6 = neg(tv5);
  auto* tv7 = castOp(dtype, tv6);
  fusion->addOutput(tv7);

  tref = atBiasEpilogue(
             atMatmul(A.squeeze(), B.squeeze(), layout),
             C.to(data_type_to_aten(DataType::Float)))
             .neg_()
             .to(data_type_to_aten(DataType::Half));
}

INSTANTIATE_TEST_SUITE_P(
    General,
    HopperPlusMatmulSchedulerTest,
    testing::Combine(
        testing::Bool(), // use_smem_epilogue
        testing::Bool(), // a_k_inner
        testing::Bool(), // b_k_inner
        testing::Values(512), // M
        testing::Values(256), // N
        testing::Values(128), // K
        testing::Values(
            MmaMacro::Hopper_64_128_16,
            MmaMacro::Blackwell1CTA_128_128_16), // mma_macros
        testing::Values(1, 2) // SplitK Factor
        ),
    hopperPlusTestName);

INSTANTIATE_TEST_SUITE_P(
    Swizzle,
    HopperPlusMatmulSchedulerTest,
    testing::Combine(
        testing::Values(true), // use_smem_epilogue
        testing::Bool(), // a_k_inner
        testing::Bool(), // b_k_inner
        testing::Values(512), // M
        testing::Values(256), // N
        testing::Values(64), // K
        testing::Values(
            MmaMacro::Hopper_64_256_16,
            MmaMacro::Hopper_64_128_16,
            MmaMacro::Hopper_64_64_16,
            MmaMacro::Hopper_64_32_16,
            MmaMacro::Hopper_64_16_16,
            MmaMacro::Blackwell1CTA_128_256_16,
            MmaMacro::Blackwell1CTA_128_128_16,
            MmaMacro::Blackwell1CTA_128_64_16,
            MmaMacro::Blackwell1CTA_128_32_16,
            MmaMacro::Blackwell1CTA_128_16_16), // mma_macros
        testing::Values(1) // SplitK Factor
        ),
    hopperPlusTestNameSwizzle);

} // namespace nvfuser
