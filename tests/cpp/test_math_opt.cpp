// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include "fusion.h"
#include "fusion_guard.h"
#include "ops/all_ops.h"
#include "runtime/executor.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

namespace nvfuser {

using MathOptTest = NVFuserTest;

TEST_F(MathOptTest, FastMathTanh) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = tanh(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    EnableOptionsGuard enable_opt_guard;
    EnableOptionsGuard::getCurOptions().set(EnableOption::FastMath);
    ke.compile(fusion.get(), {t0});
  }

  // Verify PTX, result validation is skipped since reference won't use fast
  // math.
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx_string(
      compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  EXPECT_TRUE(ptx_string.find("tanh.approx.f32") != std::string::npos);
}

using NanReductionTest = NVFuserFixtureParamTest<BinaryOpType>;

TEST_P(NanReductionTest, Test) {
  // Check NAN reduction behavior for several cases:
  // 1. No NAN input -> no NAN output
  // 2. Single NAN input -> NAN output only for min/max (not fmin/fmax)
  // 3. All NAN input -> NAN output

  BinaryOpType opType = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  Val* init = ops::binOpIdentity(opType, tv0->dtype());
  auto tv1 = reductionOp(opType, {0}, init, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  // No-NAN input
  auto cg_outputs = ke.run({t0});
  EXPECT_FALSE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());

  // Single NAN input
  t0[0] = std::numeric_limits<float>::quiet_NaN();
  cg_outputs = ke.run({t0});
  bool any_nan =
      at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>();
  if (opType == BinaryOpType::FMax || opType == BinaryOpType::FMin) {
    EXPECT_FALSE(any_nan);
  } else {
    EXPECT_TRUE(any_nan);
  }

  // All NAN input
  t0 = at::full({32}, std::numeric_limits<float>::quiet_NaN(), options);
  cg_outputs = ke.run({t0});
  EXPECT_TRUE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    MathOptTest,
    NanReductionTest,
    ::testing::Values(
        BinaryOpType::Max,
        BinaryOpType::FMax,
        BinaryOpType::Min,
        BinaryOpType::FMin),
    [](const testing::TestParamInfo<BinaryOpType>& info) -> std::string {
      std::stringstream ss;
      ss << info.param;
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser
