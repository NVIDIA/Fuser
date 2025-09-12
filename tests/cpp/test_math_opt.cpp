// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <fusion.h>
#include <fusion_guard.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

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

TEST_F(MathOptTest, IntrinsicFp16OpsHmax) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2, DataType::BFloat16);
  fusion->addInput(tv0);
  auto tv1 = max(tv0, {1});
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    ke.compile(fusion.get(), {t0});
  }
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx_string(
      compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  EXPECT_TRUE(ptx_string.find("max.bf16") != std::string::npos);
}

TEST_F(MathOptTest, IntrinsicFp16OpsHmaxAutoSchedule) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::BFloat16);
  fusion->addInput(tv0);
  auto tv1 = max(tv0, {1});
  fusion->addOutput(tv1);
  auto unscheduled_fusion = *fusion;

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({1024, 8192}, options);

  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::CudaToFile);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&unscheduled_fusion, outputs, {t0}, __LINE__, __FILE__);
}
} // namespace nvfuser
