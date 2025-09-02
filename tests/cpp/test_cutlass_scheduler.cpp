// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/cutlass_compiled_kernel.h>
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

  constexpr int64_t M = 8192, N = 8192, K = 8192;

  // Create actual tensor data for inputs
  auto options = at::TensorOptions().dtype(torch::kFloat).device(at::kCUDA, 0);

  // Create nvfp4 tensors by creating uint8 tensors and viewing them as
  // Float4_e2m1fn_x2
  at::Tensor a_fp4 = at::empty({M, K}, options.dtype(at::kFloat4_e2m1fn_x2));
  at::Tensor b_fp4 =
      at::empty({N, K}, options.dtype(at::kFloat4_e2m1fn_x2)).t();

  // Create scale tensors in Float format (as expected by the fusion)
  at::Tensor a_scale = at::empty({M, K / 8}, options.dtype(at::kFloat8_e4m3fn));
  at::Tensor b_scale = at::empty({N, K / 8}, options.dtype(at::kFloat8_e4m3fn));

  // Create scalar tensors
  at::Tensor alpha = at::scalar_tensor(1.5f, options);

  KernelArgumentHolder args;
  args.push(a_fp4);
  args.push(b_fp4);
  args.push(a_scale);
  args.push(b_scale);
  args.push(alpha);

  // We have to allocate the outputs ourself and add those to args. This will
  // eventually be the responsibility of the CutlassExecutor

  at::Tensor output_tensor = at::empty({M, N}, options.dtype(at::kBFloat16));
  args.push(output_tensor);

  CutlassCompiledKernel kernel(
      fusion.get(), c10::Device(c10::DeviceType::CUDA, 0));

  kernel.compile();
  EXPECT_TRUE(kernel.isCompiled());

  // Run the fusion

  c10::DeviceGuard dg(kernel.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  at::cuda::jit::initializeCudaContext();

  kernel.run(args, stream);

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
