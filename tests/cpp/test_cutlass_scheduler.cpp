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

  TensorView* a = makeContigTensor(2, DataType::Float4_e2m1fn);
  TensorView* b = makeContigTensor(2, DataType::Float4_e2m1fn);
  // B has K inner
  b->setAllocationDomain({b->axis(1), b->axis(0)}, /*new_contiguity=*/true);
  TensorView* a_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* b_sf = makeContigTensor(2, DataType::Float8_e4m3fn);
  TensorView* alpha = makeContigTensor(0, DataType::Float);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addInput(a_sf);
  fusion->addInput(b_sf);
  fusion->addInput(alpha);

  // TODO: support more output dtypes, specifically nvfp4
  auto smm = scaled_mm(
      a,
      b,
      a_sf,
      b_sf,
      alpha,
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
