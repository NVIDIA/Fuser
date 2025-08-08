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
#include <runtime/executor_kernel_arg.h>
#include <scheduler/cutlass.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// Test basic Cutlass scheduler canSchedule functionality
TEST_F(NVFuserTest, CutlassScheduler_CanSchedule) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create a simple nvfp4 scaled matmul fusion
  // Create input tensors
  // Note: nvfp4 types are not exposed in C++ API, using BFloat16 for testing
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 = makeContigTensor(2, DataType::BFloat16); // Scale A
  auto tv3 = makeContigTensor(2, DataType::BFloat16); // Scale B
  auto tv4 = makeContigTensor(0, DataType::Float); // Alpha (scalar)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation
  auto scaled_out = scaled_mm(tv0, tv1_t, tv2, tv3, tv4);

  fusion->addOutput(scaled_out.tv);

  // Check if Cutlass scheduler can handle this fusion
  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_TRUE(scheduler->canScheduleCompileTime(fusion.get()));
}

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(NVFuserTest, CutlassScheduler_SimpleNvfp4ScaledGemm) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Define problem size
  const int64_t M = 512;
  const int64_t N = 512;
  const int64_t K = 512;
  const int64_t block_size = 128; // Block size for scaling factors

  // Create input tensors
  // Note: nvfp4 types are not exposed in C++ API, using BFloat16 for testing
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A (M x K)
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B (N x K)
  auto tv2 =
      makeContigTensor(2, DataType::BFloat16); // Scale A (M x K/block_size)
  auto tv3 =
      makeContigTensor(2, DataType::BFloat16); // Scale B (N x K/block_size)
  auto tv4 = makeContigTensor(0, DataType::Float); // Alpha (scalar)

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);

  // Transpose B for column major layout expected by CUTLASS
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul operation
  auto scaled_out = scaled_mm(
      tv0, // mat1
      tv1_t, // mat2
      tv2, // scale1
      tv3, // scale2
      tv4, // alpha
      nullptr, // bias
      nullptr, // beta
      DataType::BFloat16, // output dtype
      0, // output_block_scale_size
      DataType::BFloat16, // output_block_scale_dtype
      false // output_gamma
  );

  fusion->addOutput(scaled_out.tv);

  // Test that the scheduler can handle this fusion
  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_TRUE(scheduler->canScheduleCompileTime(fusion.get()));

  // Create dummy inputs for heuristic computation
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);
  auto t2 = at::randn({M, K / block_size}, options);
  auto t3 = at::randn({N, K / block_size}, options);
  auto t4 = at::ones({}, float_options);

  KernelArgumentHolder args;
  args.push(t0);
  args.push(t1);
  args.push(t2);
  args.push(t3);
  args.push(t4);

  auto runtime_info = SchedulerRuntimeInfo(fusion.get(), args, nullptr, {});

  auto params = scheduler->computeHeuristics(fusion.get(), runtime_info);
  EXPECT_NE(params, nullptr);

  auto cutlass_params = dynamic_cast<CutlassParams*>(params.get());
  EXPECT_NE(cutlass_params, nullptr);

  // Verify heuristic parameters
  EXPECT_TRUE(cutlass_params->use_nvfp4);
  EXPECT_GT(cutlass_params->tile_m, 0);
  EXPECT_GT(cutlass_params->tile_n, 0);
  EXPECT_GT(cutlass_params->tile_k, 0);

  // Note: Actual execution will happen in CutlassExecutor once JIT compilation
  // is implemented. For now, we're just testing the scheduling logic.
}

// Test Cutlass scheduler rejects non-scaled matmul
TEST_F(NVFuserTest, CutlassScheduler_RejectsNonScaledMatmul) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create a regular matmul (not scaled)
  auto tv0 = makeContigTensor(2, DataType::BFloat16);
  auto tv1 = makeContigTensor(2, DataType::BFloat16);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);
  fusion->addOutput(tv2);

  // Cutlass scheduler should reject non-scaled matmul
  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_FALSE(scheduler->canScheduleCompileTime(fusion.get()));
}

// Test Cutlass scheduler with epilogue fusion
TEST_F(NVFuserTest, CutlassScheduler_WithEpilogue) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create input tensors
  // Note: nvfp4 types are not exposed in C++ API, using BFloat16 for testing
  auto tv0 = makeContigTensor(2, DataType::BFloat16); // Matrix A
  auto tv1 = makeContigTensor(2, DataType::BFloat16); // Matrix B
  auto tv2 = makeContigTensor(2, DataType::BFloat16); // Scale A
  auto tv3 = makeContigTensor(2, DataType::BFloat16); // Scale B
  auto tv4 = makeContigTensor(0, DataType::Float); // Alpha
  auto tv5 = makeContigTensor(1, DataType::BFloat16); // Bias vector

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);
  fusion->addInput(tv5);

  // Transpose B for column major layout
  auto tv1_t = transpose(tv1, 0, 1);

  // Create scaled matmul with bias
  auto scaled_out = scaled_mm(
      tv0, // mat1
      tv1_t, // mat2
      tv2, // scale1
      tv3, // scale2
      tv4, // alpha
      tv5, // bias
      nullptr, // beta
      DataType::BFloat16, // output dtype
      0, // output_block_scale_size
      DataType::BFloat16, // output_block_scale_dtype
      false // output_gamma
  );

  // Add a simple epilogue operation
  auto tv_relu = relu(scaled_out.tv);

  fusion->addOutput(tv_relu);

  // Test that the scheduler can handle this fusion with epilogue
  auto scheduler = std::make_unique<CutlassScheduler>();
  EXPECT_TRUE(scheduler->canScheduleCompileTime(fusion.get()));

  // Create dummy inputs for heuristic computation
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto float_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({512, 512}, options);
  auto t1 = at::randn({512, 512}, options);
  auto t2 = at::randn({512, 4}, options);
  auto t3 = at::randn({512, 4}, options);
  auto t4 = at::ones({}, float_options);
  auto t5 = at::randn({512}, options);

  KernelArgumentHolder args;
  args.push(t0);
  args.push(t1);
  args.push(t2);
  args.push(t3);
  args.push(t4);
  args.push(t5);

  auto runtime_info = SchedulerRuntimeInfo(fusion.get(), args, nullptr, {});

  auto params = scheduler->computeHeuristics(fusion.get(), runtime_info);
  EXPECT_NE(params, nullptr);

  auto cutlass_params = dynamic_cast<CutlassParams*>(params.get());
  EXPECT_NE(cutlass_params, nullptr);
  EXPECT_TRUE(cutlass_params->has_epilogue_fusion);
}

} // namespace nvfuser
