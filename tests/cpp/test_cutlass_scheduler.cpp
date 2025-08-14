// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/cutlass_executor.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using CutlassExecutorTest = NVFuserTest;

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(CutlassExecutorTest, SimpleNvfp4ScaledGemm) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  constexpr int64_t M = 256;
  constexpr int64_t N = 128;
  constexpr int64_t K = 64;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Float4_e2m1fn);
  auto tv1 = makeContigTensor(2, DataType::Float4_e2m1fn);
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

  fusion->printMath();

  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_TRUE(scheduler->canScheduleCompileTime(fusion.get()));

  // Create actual tensor data for inputs
  at::manual_seed(0);
  auto fp4_options =
      at::TensorOptions().dtype(torch::kFloat4_e2m1fn_x2).device(at::kCUDA, 0);
  auto fp8_options =
      at::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(torch::kFloat).device(at::kCUDA, 0);

  // Create nvfp4 tensors by creating uint8 tensors and viewing them as
  // Float4_e2m1fn_x2
  auto a_fp4 = at::empty({M, K}, fp4_options);
  auto b_fp4 = at::empty({N, K}, fp4_options);

  // Create scale tensors in Float format (as expected by the fusion)
  auto a_scale = at::empty({M, K / 8}, fp8_options);
  auto b_scale = at::empty({N, K / 8}, fp8_options);

  // Create scalar tensors
  auto alpha = at::scalar_tensor(1.5f, float_options);

  KernelArgumentHolder args;
  args.push(a_fp4);
  args.push(b_fp4);
  args.push(a_scale);
  args.push(b_scale);
  args.push(alpha);

  auto runtime_info = SchedulerRuntimeInfo(fusion.get(), args, nullptr, {});

  auto params = scheduler->computeHeuristics(fusion.get(), runtime_info);
  EXPECT_NE(params, nullptr);

  auto cutlass_params = dynamic_cast<CutlassParams*>(params.get());
  EXPECT_NE(cutlass_params, nullptr);

  // Create CutlassExecutor and compile the fusion
  CutlassExecutor executor;
  executor.compile(fusion.get(), args);
  EXPECT_TRUE(executor.isCompiled());

  // Run the fusion
  auto outputs = executor.run(args);
  EXPECT_EQ(outputs.size(), 1);

  // Check that the output is a tensor with correct properties
  auto output_tensor = outputs[0].as<at::Tensor>();
  EXPECT_EQ(output_tensor.sizes(), at::IntArrayRef({M, N}));
  EXPECT_EQ(output_tensor.dtype(), at::kBFloat16);
  EXPECT_EQ(output_tensor.device().type(), at::kCUDA);

  ExpressionEvaluator expr_eval;
  expr_eval.bind(tv0, a_fp4);
  expr_eval.bind(tv1, b_fp4);
  expr_eval.bind(tv2, a_scale);
  expr_eval.bind(tv3, b_scale);
  expr_eval.bind(tv4, alpha);
  PolymorphicValue eval_smm = expr_eval.evaluate(smm.tv);

  EXPECT_TRUE(
      at::allclose(output_tensor, eval_smm.as<at::Tensor>(), .0001, .0001));
}

} // namespace nvfuser
