// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>

#include <cutlass/codegen.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using CutlassExecutorTest = NVFuserTest;

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(CutlassExecutorTest, Nvfp4ScaledGemm_CodeGen) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Float4_e2m1fn);
  auto tv1 = makeContigTensor(2, DataType::Float4_e2m1fn);
  // B has K inner
  tv1->setAllocationDomain(
      {tv1->axis(1), tv1->axis(0)}, /*new_contiguity=*/true);
  auto tv2 = makeContigTensor(2, DataType::Float8_e4m3fn);
  auto tv3 = makeContigTensor(2, DataType::Float8_e4m3fn);
  auto tv4 = makeContigTensor(0, DataType::Float); // alpha

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  // TODO: support more output dtypes, specifically nvfp4
  auto smm = scaled_mm(
      tv0,
      tv1,
      tv2,
      tv3,
      tv4,
      /*bias=*/nullptr,
      /*beta=*/nullptr,
      /*dtype=*/DataType::BFloat16);

  fusion->addOutput(smm.tv);

  const std::string code = cutlass_codegen::generateCode(fusion.get());

  EXPECT_THAT(
      code,
      ::testing::HasSubstr("using MmaTileShape = Shape<_256, _256, _256>;"));
}

} // namespace nvfuser
