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
#include <scheduler/all_schedulers.h>
#include <scheduler/mma_utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include "ops/arith.h"
#include "type.h"

namespace nvfuser {

namespace {
class MatmulSchedulerTest : public NVFuserTest {};

using PrecisionsDesc = std::pair<PrimDataType, PrimDataType>;

using AbsoluteError = double;
using RelariveError = double;
using ErrorThresholds = std::pair<AbsoluteError, RelariveError>;
using TestCaseErrorThresholds = std::map<PrecisionsDesc, ErrorThresholds>;
class PrecisionParametrizedTest
    : public NVFuserFixtureParamTest<PrecisionsDesc> {};

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
  NVF_ERROR(false, "Unsupported conversion of PrimDataType");
  return "*";
}

static const PrecisionsDesc HSH =
    std::make_pair(PrimDataType::Half, PrimDataType::Half);
static const PrecisionsDesc HSS =
    std::make_pair(PrimDataType::Half, PrimDataType::Float);
static const PrecisionsDesc TST =
    std::make_pair(PrimDataType::BFloat16, PrimDataType::BFloat16);
static const PrecisionsDesc TSS =
    std::make_pair(PrimDataType::BFloat16, PrimDataType::Float);

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBias) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.0001, 0.0001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.0001, 0.0001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);
  // tv4 := cast(tv3)
  auto tv4 = maybeCastOp(out_type, tv3);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t2.to(at_out_type);

  auto t5 = atBiasEpilogue(t3, t2);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t5, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = relu(A x B)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueRelu) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.0001, 0.0001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.0001, 0.0001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = relu(tv2);
  auto tv4 = maybeCastOp(out_type, tv3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::relu(t2);
  auto t4 = t3.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  NVF_CHECK(outputs[0].allclose(t4, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = relu((A x B) + bias)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasRelu) {
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);
  auto tv4 = maybeCastOp(out_type, tv3);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv2);

  // tv6 := relu((A x B) + bias)
  auto tv6 = relu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t3.to(at_out_type);
  auto t5 = atBiasEpilogue(t4, t2);
  auto t7 = at::relu(t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  // D tensor results
  NVF_CHECK(outputs[0].allclose(t7, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = A x B;
//   Aux = relu(D)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, DISABLED_EpilogueReluAux) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.001, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = maybeCastOp(out_type, tv2);
  auto tv4 = relu(tv3);
  auto tv5 = maybeCastOp(out_type, tv4);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = t2.to(at_out_type);
  auto t4 = at::relu(t2);
  auto t5 = t4.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // D tensor results
  NVF_CHECK(outputs[0].allclose(t3, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(outputs[1].allclose(t5, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//   Aux = relu(D)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, DISABLED_EpilogueBiasReluAux) {
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.001, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, in_type);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);
  auto tv4 = maybeCastOp(out_type, tv3);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv2);

  // tv6 := relu((A x B) + bias)
  auto tv6 = relu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv5);
  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t3.to(at_out_type);
  auto t5 = atBiasEpilogue(t4, t2);
  auto t7 = at::relu(t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  // D tensor results
  NVF_CHECK(outputs[0].allclose(t5, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(outputs[1].allclose(t7, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = gelu(A x B)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueGelu) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = gelu(tv2);
  auto tv4 = maybeCastOp(out_type, tv3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::gelu(t2);
  auto t4 = t3.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  NVF_CHECK(outputs[0].allclose(t4, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = A x B
//   Aux = gelu(D)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, DISABLED_EpilogueGeluAux) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.001, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.001, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = maybeCastOp(out_type, tv2);
  auto tv4 = gelu(tv2);
  auto tv5 = maybeCastOp(out_type, tv4);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = t2.to(at_out_type);
  auto t4 = at::gelu(t2);
  auto t5 = t4.to(at_out_type);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // D tensor results
  NVF_CHECK(outputs[0].allclose(t3, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(outputs[1].allclose(t5, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion for Ampere:
//   D = gelu((A x B) + bias)
//  Target architectures: Turing, Ampere
TEST_P(PrecisionParametrizedTest, EpilogueBiasGelu) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.01, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.01, 0.01)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);
  auto tv4 = maybeCastOp(out_type, tv3);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv2);

  // tv6 := gelu((A x B) + bias)
  auto tv6 = gelu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t3.to(at_out_type);
  auto t5 = atBiasEpilogue(t4, t2);
  auto t7 = at::gelu(t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t7, abs_err_thr, rel_err_thr));
}

// Matmul test that uses segmenter for fusion:
//   D = (A x B) + bias
//   Aux = gelu(D)
//  Target architectures: Ampere
TEST_P(PrecisionParametrizedTest, DISABLED_EpilogueBiasGeluAux) {
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  const auto layout = MatmulLayout::TT;

  static TestCaseErrorThresholds errs = {
      {HSS, std::make_pair(0.001, 0.001)},
      {HSH, std::make_pair(0.01, 0.001)},
      {TSS, std::make_pair(0.001, 0.001)},
      {TST, std::make_pair(0.001, 0.001)},
  };

  NVF_CHECK(
      errs.count(GetParam()) != 0,
      "Undefined error thresholds for requested precisions");

  const auto [in_prim_type, out_prim_type] = GetParam();
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

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, in_type);
  auto tv1 = makeContigTensor(2, in_type);
  auto tv2 = makeContigTensor(1, out_type);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);
  auto tv4 = maybeCastOp(out_type, tv3);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv2);

  // tv6 := gelu((A x B) + bias)
  auto tv6 = gelu(tv5);
  auto tv7 = maybeCastOp(out_type, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addOutput(tv5);
  fusion->addOutput(tv7);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 248;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at_in_type, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at_in_type, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::Bias, at_out_type, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = t3.to(at_out_type);
  auto t5 = atBiasEpilogue(t4, t2);
  auto t7 = at::gelu(t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  // D tensor results
  NVF_CHECK(outputs[0].allclose(t5, abs_err_thr, rel_err_thr));
  // Aux tensor results
  NVF_CHECK(outputs[1].allclose(t7, abs_err_thr, rel_err_thr));
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    MatmulSchedulerTest,
    PrecisionParametrizedTest,
    ::testing::Values(HSS, HSH, TSS, TST),
    [](const testing::TestParamInfo<PrecisionsDesc>& info) {
      std::ostringstream os;
      os << get_type_letter(std::get<0>(info.param));
      os << get_type_letter(PrimDataType::Float);
      os << get_type_letter(std::get<1>(info.param));
      return os.str();
    });

// Matmul test that uses segmenter for 'C = A x B' fusion,
//   for Ampere with strict ref check, hence single layout check
TEST_F(MatmulSchedulerTest, BasicMatmulStrictCheckTT) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 8, 9);
  const int M = 128, N = 256, K = 512;
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = matmul(tv0, tv1, layout, true);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must be always TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0, t1, layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "fusion got segmented, expected to match whole fusion with single segment");

  NVF_CHECK(
      isSchedulerInUse(
          executor_cache.getMostRecentKernelRuntime(),
          ScheduleHeuristic::Matmul),
      "matmul scheduler was not used to handle prepared fusion");

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// Matmul test that reslies on segmenter for 'C = A x B' fusion, for Ampere
TEST_F(MatmulSchedulerTest, BasicMatmulRelaxedCheck) {
  // skip until we have Hopper support
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 2048;
  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(2, DataType::Half);
    auto tv1 = makeContigTensor(2, DataType::Half);
    auto tv2 = matmul(tv0, tv1, layout, true);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addOutput(tv2);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must be always TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
    auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

    FusionExecutorCache executor_cache(std::move(fusion));

    auto outputs = executor_cache.runFusionWithInputs({t0, t1});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "fusion got segmented, expected to match whole fusion with single segment");

    NVF_CHECK(
        isSchedulerInUse(
            executor_cache.getMostRecentKernelRuntime(),
            ScheduleHeuristic::Matmul),
        "matmul scheduler was not used to handle prepared fusion");

    NVF_CHECK(outputs[0].allclose(tref, 0.001, 0.001));
  }
}

// Matmul test that reslies on segmenter for 'C = A x B' fusion, for Ampere
//  MMA first input is passed as second fusion parameter.
//  MMA second input is passed as first fusion parameter.
TEST_F(MatmulSchedulerTest, BasicMatmulInputShuffledTT) {
  // skip until we have Hopper support
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 2048;
  const auto layout = MmaOptions::MmaLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = matmul(tv0, tv1, layout, true);

  fusion->addInput(tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must be always TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t1, t0});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "fusion got segmented, expected to match whole fusion with single segment");

  NVF_CHECK(
      isSchedulerInUse(
          executor_cache.getMostRecentKernelRuntime(),
          ScheduleHeuristic::Matmul),
      "matmul scheduler was not used to handle prepared fusion");

  NVF_CHECK(outputs[0].allclose(tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = float2half(A x B)' fusion, for
//  Ampere
TEST_F(MatmulSchedulerTest, EpilogueOutputCast) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // A - tv0, B - tv1
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto tref = t2.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  NVF_CHECK(outputs[0].allclose(tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = alpha * (A x B)' fusion, for
//  Ampere
TEST_F(MatmulSchedulerTest, EpilogueAlpha) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, A - tv0, B - tv1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = mul(s0, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(s0);
  fusion->addOutput(tv3);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto tref = at::mul(t2, alpha).to(at::kFloat);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, alpha});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  NVF_CHECK(outputs[0].allclose(tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for 'C = float2half(alpha * (A x B))'
//  fusion, for Ampere
TEST_F(MatmulSchedulerTest, EpilogueAlphaOutputCast) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, A - tv0, B - tv1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  auto tv2 = matmul(tv0, tv1, layout, true);
  auto tv3 = mul(s0, tv2);
  auto tv4 = castOp(DataType::Half, tv3);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(s0);
  fusion->addOutput(tv4);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t3 = at::mul(t2, alpha).to(at::kFloat);
  auto tref = t3.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, alpha});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  NVF_CHECK(outputs[0].allclose(tref, 0.001, 0.001));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = (A x B) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueBeta) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // beta - s0
  auto s0 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);

  // tv3 := A x B
  auto tv3 = matmul(tv0, tv1, layout, true);

  // tv4 := beta * C
  auto tv4 = mul(s0, tv2);
  // tv5 := A x B + beta * C
  auto tv5 = add(tv3, tv4);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addOutput(tv5);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double beta = 2.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

  auto t4 = at::mul(t2, beta).to(at::kFloat);
  auto t5 = at::add(t3, t4);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, beta});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t5, 0.01, 0.04));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = alpha * (A x B) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueAlphaBeta) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);

  auto tv3 = matmul(tv0, tv1, layout, true);
  // tv4 := alpha * (A x B)
  auto tv4 = mul(s0, tv3);

  // tv5 := beta * C
  auto tv5 = mul(s1, tv2);
  // tv6 := alpha * (A x B) + beta * C
  auto tv6 = add(tv4, tv5);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);
  fusion->addOutput(tv6);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);

  auto t5 = at::mul(t2, beta).to(at::kFloat);
  auto t6 = at::add(t4, t5);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t6, 0.001, 0.004));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = gelu(alpha * (A x B) + beta * C)
TEST_F(MatmulSchedulerTest, EpilogueAlphaBetaGeluOutputCast) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const auto layout = MatmulLayout::TT;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // alpha - s0, beta - s1
  auto s0 = IrBuilder::create<Val>(DataType::Double);
  auto s1 = IrBuilder::create<Val>(DataType::Double);

  // A - tv0, B - tv1, C - tv2
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  auto tv2 = makeContigTensor(2, DataType::Half);

  auto tv3 = matmul(tv0, tv1, layout, true);
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

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);
  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kHalf, M, N, K);

  auto t3 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  auto t4 = at::mul(t3, alpha).to(at::kFloat);

  auto t5 = at::mul(t2, beta).to(at::kFloat);
  auto t6 = at::add(t4, t5);

  auto t7 = at::gelu(t6);
  auto t8 = t7.to(at::kHalf);

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t8, 0.01, 0.06));
}

// Matmul test that uses segmenter for fusion for Ampere:
//  D = alpha * ((A x B) + bias) + beta * C
TEST_F(MatmulSchedulerTest, EpilogueAlphaBetaBias) {
  // NOTE: test skips Turing arch, the relative error was too big
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  const auto layout = MatmulLayout::TT;
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

  auto tv4 = matmul(tv0, tv1, layout, true);

  // tv5 := (A x B) + bias
  auto tv5 = biasEpilogue(tv4, tv3);
  // tv6 := alpha * ((A x B) + bias)
  auto tv6 = mul(s0, tv5);
  // tv7 := beta * C
  auto tv7 = mul(s1, tv2);
  // tv8 := (alpha * ((A x B) + bias)) + (beta * C)
  auto tv8 = add(tv6, tv7);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(s0);
  fusion->addInput(s1);
  fusion->addOutput(tv8);

  NVF_CHECK(
      1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
      "matmul fusion must have at least one MmaOp");
  NVF_CHECK(
      ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().has_value(),
      "input layout has not be set for MmaOp");
  NVF_CHECK(
      MatmulLayout::TN ==
          ir_utils::getOpsOfType<MmaOp>(fusion.get()).front()->layout().value(),
      "the MmaOp layout of Ampere MMA must always be TN");

  const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
  NVF_CHECK(
      fusion_layout.isValid(),
      "failed to get decide matmul layout through fusion definition");
  NVF_CHECK(
      fusion_layout.getData() == layout,
      "mismatch between test layout (",
      toString(layout),
      ") and layout inferred from fusion definition (",
      toString(fusion_layout.getData()),
      ")");

  FusionExecutorCache executor_cache(std::move(fusion));

  const int M = 504, N = 136, K = 1024;

  at::manual_seed(0);
  const double alpha = 2.5;
  const double beta = 1.5;
  auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kHalf, M, N, K);
  auto t3 = matmulAtInput(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K);

  auto t4 = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  // t5 := (A x B) + bias
  auto t5 = atBiasEpilogue(t4, t3);
  // t6 := alpha * ((A x B) + bias)
  auto t6 = at::mul(t5, alpha).to(at::kFloat);
  // t7 := beta * C
  auto t7 = at::mul(t2, beta).to(at::kFloat);
  // t8 := (alpha * ((A x B) + bias)) + (beta * C)
  auto t8 = at::add(t6, t7);

  auto outputs =
      executor_cache.runFusionWithInputs({t0, t1, t2, t3, alpha, beta});

  NVF_CHECK(
      !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation did happen");

  // NOTE: increasted absolute tolerance to silence false negative verification
  //       caused by different way of calculating reference
  NVF_CHECK(outputs[0].allclose(t8, 0.01, 0.01));
}

// Strided batch gemm test taht uses matmul scheduler, for Ampere:
//   D = (A x B)
TEST_F(MatmulSchedulerTest, StridedBatch) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 248, B = 2;
  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    // A - tv0, B - tv1
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);

    // tv2 := A x B
    auto tv2 = splitkLikeBatchedMatmul(tv0, tv1, layout);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addOutput(tv2);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must always be TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    FusionExecutorCache executor_cache(std::move(fusion));

    at::manual_seed(0);
    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
    auto t2 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

    auto outputs = executor_cache.runFusionWithInputs({t0, t1});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "segmentation did happen");

    // NOTE: increasted absolute tolerance to silence false negative
    // verification
    //       caused by different way of calculating reference
    NVF_CHECK(outputs[0].allclose(t2, 0.0001, 0.0001));
  }
}

// Strided batch gemm test with alpha and beta that uses matmul scheduler,
//  for Ampere architecture:
//   D = alpha * (A x B) + beta * C
TEST_F(MatmulSchedulerTest, StridedBatchEpilogueAlphaBeta) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 248, B = 2;

  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    // A - tv0, B - tv1, C - tv2
    // alpha - s0, beta - s1
    auto s0 = IrBuilder::create<Val>(DataType::Double);
    auto s1 = IrBuilder::create<Val>(DataType::Double);
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);
    auto tv2 = makeContigTensor(3, DataType::Float);

    // tv3 := A x B
    auto tv3 = splitkLikeBatchedMatmul(tv0, tv1, layout);
    // tv4 := alpha * (A x B)
    auto tv4 = mul(s0, tv3);
    // tv5 := beta * C
    auto tv5 = mul(s1, tv2);
    // tv6 := alpha * (A x B) + beta * C
    auto tv6 = add(tv4, tv5);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addInput(s0);
    fusion->addInput(s1);
    fusion->addOutput(tv6);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must always be TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    FusionExecutorCache executor_cache(std::move(fusion));

    at::manual_seed(0);
    const double alpha = 2.5;
    const double beta = 1.5;

    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
    auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kFloat, M, N, K, B);

    auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
    auto t4 = at::mul(t3, alpha).to(at::kFloat);
    auto t5 = at::mul(t2, beta).to(at::kFloat);
    auto t6 = at::add(t4, t5);

    auto outputs =
        executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "segmentation did happen");

    // NOTE: increasted absolute tolerance to silence false negative
    //  verification caused by different way of calculating reference
    NVF_CHECK(outputs[0].allclose(t6, 0.0001, 0.0001));
  }
}

// Strided batch gemm test with alpha and beta scaling that uses matmul
// scheduler,
//  there is only single C tensor for whole batch; test for Ampere architecture:
//   D = alpha * (A x B) + beta * C
TEST_F(MatmulSchedulerTest, StridedBatchEpilogueAlphaSingleBeta) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 248, B = 2;

  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    // A - tv0, B - tv1, C - tv2
    // alpha - s0, beta - s1
    auto s0 = IrBuilder::create<Val>(DataType::Double);
    auto s1 = IrBuilder::create<Val>(DataType::Double);
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);
    auto tv2 = makeContigTensor(2, DataType::Float);

    // tv3 := A x B
    auto tv3 = splitkLikeBatchedMatmul(tv0, tv1, layout);
    // tv4 := alpha * (A x B)
    auto tv4 = mul(s0, tv3);
    // tv5 := beta * C
    auto tv5 = mul(s1, tv2);
    // tv6 := bcast(beta * C)
    // [M, N] -> [B, M, N], with B as bcast
    auto tv6 = broadcast(tv5, {true, false, false});
    // tv7 := alpha * (A x B) + beta * C
    auto tv7 = add(tv4, tv6);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addInput(s0);
    fusion->addInput(s1);
    fusion->addOutput(tv7);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must always be TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    FusionExecutorCache executor_cache(std::move(fusion));

    at::manual_seed(0);
    const double alpha = 1.5;
    const double beta = 2.5;

    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
    auto t2 = matmulAtInput(layout, TensorMatmulPos::C, at::kFloat, M, N, K);

    auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
    auto t4 = at::mul(t3, alpha).to(at::kFloat);
    auto t5 = at::mul(t2, beta).to(at::kFloat);
    // NOTE: t6, a result of adding an outer-most broadcast dimension to
    //  the result of scaling C with beta
    auto t6 = at::unsqueeze(t5, 0);
    auto t7 = at::add(t4, t5);

    auto outputs =
        executor_cache.runFusionWithInputs({t0, t1, t2, alpha, beta});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "segmentation did happen");

    // NOTE: increasted absolute tolerance to silence false negative
    //  verification caused by different way of calculating reference
    NVF_CHECK(outputs[0].allclose(t7, 0.0001, 0.0001));
  }
}

// Strided batch gemm test with bias that uses matmul scheduler, for Ampere:
//   D = (A x B) + bias
TEST_F(MatmulSchedulerTest, StridedBatchEpilogueBias) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 248, B = 2;

  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    // A - tv0, B - tv1, bias - tv2
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);
    auto tv2 = makeContigTensor(2, DataType::Float);

    // tv3 := A x B
    auto tv3 = splitkLikeBatchedMatmul(tv0, tv1, layout);
    // tv4 := (A x B) + bias
    auto tv4 = biasEpilogue(tv3, tv2);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addOutput(tv4);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must always be TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    FusionExecutorCache executor_cache(std::move(fusion));

    at::manual_seed(0);
    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
    auto t2 =
        matmulAtInput(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K, B);

    auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
    auto t4 = atBiasEpilogue(t3, t2).to(at::kFloat);

    auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "segmentation did happen");

    // NOTE: increasted absolute tolerance to silence false negative
    //  verification caused by different way of calculating reference
    NVF_CHECK(outputs[0].allclose(t4, 0.0001, 0.0001));
  }
}

// Strided batch gemm test with single bias vector that uses matmul
// scheduler, for Ampere:
//   D = (A x B) + bias
TEST_F(MatmulSchedulerTest, StridedBatchEpilogueSingleBias) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  const int M = 504, N = 136, K = 248, B = 2;

  for (auto layout : kAllSupportedMatmulLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    // A - tv0, B - tv1, bias - tv2
    auto tv0 = makeContigTensor(3, DataType::Half);
    auto tv1 = makeContigTensor(3, DataType::Half);
    auto tv2 = makeContigTensor(1, DataType::Float);

    // tv3 := A x B
    auto tv3 = splitkLikeBatchedMatmul(tv0, tv1, layout);
    // tv4 := (A x B) + bias
    auto tv4 = biasEpilogue(tv3, tv2);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addOutput(tv4);

    NVF_CHECK(
        1 == ir_utils::getOpsOfType<MmaOp>(fusion.get()).size(),
        "matmul fusion must have at least one MmaOp");
    NVF_CHECK(
        ir_utils::getOpsOfType<MmaOp>(fusion.get())
            .front()
            ->layout()
            .has_value(),
        "input layout has not be set for MmaOp");
    NVF_CHECK(
        MatmulLayout::TN ==
            ir_utils::getOpsOfType<MmaOp>(fusion.get())
                .front()
                ->layout()
                .value(),
        "the MmaOp layout of Ampere MMA must always be TN");

    const auto fusion_layout = mma_utils::getMatmulLayout(fusion.get());
    NVF_CHECK(
        fusion_layout.isValid(),
        "failed to get decide matmul layout through fusion definition");
    NVF_CHECK(
        fusion_layout.getData() == layout,
        "mismatch between test layout (",
        toString(layout),
        ") and layout inferred from fusion definition (",
        toString(fusion_layout.getData()),
        ")");

    FusionExecutorCache executor_cache(std::move(fusion));

    at::manual_seed(0);
    auto t0 = matmulAtInput(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
    auto t1 = matmulAtInput(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
    // Explicitly make bias tensor a single dim by passing 0 for batch
    auto t2 =
        matmulAtInput(layout, TensorMatmulPos::Bias, at::kFloat, M, N, K, 0);

    auto t3 = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
    auto t4 = atBiasEpilogue(t3, t2).to(at::kFloat);

    auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

    NVF_CHECK(
        !executor_cache.getMostRecentKernelRuntime()->isSegmented(),
        "segmentation did happen");

    // NOTE: increasted absolute tolerance to silence false negative
    //  verification caused by different way of calculating reference
    NVF_CHECK(outputs[0].allclose(t4, 0.0001, 0.0001));
  }
}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

} // namespace nvfuser
