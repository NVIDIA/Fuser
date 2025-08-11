// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <cutlass/nvf_cutlass.h>
#include <runtime/cutlass_executor.h>

namespace nvfuser {

using CutlassExecutorTest = NVFuserTest;

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(CutlassExecutorTest, SimpleNvfp4ScaledGemm) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  constexpr int64_t M = 8192;
  constexpr int64_t N = 8192;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2, DataType::Float4_e2m1fn);
  auto tv1 = makeContigTensor(2, DataType::Float4_e2m1fn);
  auto tv2 = makeContigTensor(2, DataType::Float);
  auto tv3 = makeContigTensor(2, DataType::Float);
  auto tv4 = makeContigTensor(0, DataType::Float); // scale
  auto tv5 = makeContigTensor(0, DataType::Float); // alpha
  auto tv6 = makeContigTensor(0, DataType::Float); // beta

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);
  fusion->addInput(tv5);
  fusion->addInput(tv6);

  auto tv7 = scaled_mm(tv0, tv1, tv2, tv3, tv4, tv5, tv6);

  fusion->addOutput(tv7.tv);

  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_TRUE(scheduler->canScheduleCompileTime(fusion.get()));

  // Create actual tensor data for inputs
  at::manual_seed(0);
  auto uint8_options = at::TensorOptions().dtype(torch::kUInt8).device(at::kCUDA, 0);
  auto float_options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // Create nvfp4 tensors by creating uint8 tensors and viewing them as Float4_e2m1fn_x2
  auto a_fp4 = at::randint(0, 256, {M, N / 2}, uint8_options).view(at::kFloat4_e2m1fn_x2);
  auto b_fp4 = at::randint(0, 256, {N, M / 2}, uint8_options).view(at::kFloat4_e2m1fn_x2);
  
  // Create scale tensors in Float format (as expected by the fusion)
  auto a_scale = at::randn({M, 1}, float_options);
  auto b_scale = at::randn({1, N}, float_options);
  
  // Create scalar tensors
  auto scale = at::scalar_tensor(1.0f, float_options);
  auto alpha = at::scalar_tensor(1.0f, float_options);
  auto beta = at::scalar_tensor(0.0f, float_options);

  KernelArgumentHolder args;
  args.push(a_fp4);
  args.push(b_fp4);
  args.push(a_scale);
  args.push(b_scale);
  args.push(scale);
  args.push(alpha);
  args.push(beta);

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
}

} // namespace nvfuser
