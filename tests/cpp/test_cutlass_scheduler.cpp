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
#include <runtime/cutlass_executor.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/runtime_info.h>
#include <scheduler/scheduler_types.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <cstdlib>

namespace nvfuser {

using CutlassExecutorTest = NVFuserTest;

// Test Cutlass scheduler with simple nvfp4 block-scaled GEMM
TEST_F(CutlassExecutorTest, Nvfp4ScaledGemm_CompiledKernel) {
  // Skip if not on SM100 or above
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    GTEST_SKIP() << "Skipping test on pre-SM100 GPUs";
  }

  if (!std::getenv("CUTLASS_PATH")) {
    GTEST_SKIP() << "The CUTLASS_PATH environment variable must be set in "
                 << "order to run this test";
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
  at::Tensor at_a = at::empty({M, K}, options.dtype(at::kFloat4_e2m1fn_x2));
  at::Tensor at_b = at::empty({N, K}, options.dtype(at::kFloat4_e2m1fn_x2)).t();

  // Create scale tensors in Float format (as expected by the fusion)
  at::Tensor at_a_sf = at::empty({M, K / 8}, options.dtype(at::kFloat8_e4m3fn));
  at::Tensor at_b_sf = at::empty({N, K / 8}, options.dtype(at::kFloat8_e4m3fn));

  // Create scalar tensors
  at::Tensor at_alpha = at::scalar_tensor(1.5f, options);

  std::vector<c10::IValue> inputs{at_a, at_b, at_a_sf, at_b_sf, at_alpha};

  CutlassParams params;

  CutlassExecutor ce;
  ce.compile(fusion.get(), params);

  KernelArgumentHolder outputs = ce.run(inputs);

  testValidate(fusion.get(), outputs, inputs, __LINE__, __FILE__);
}

} // namespace nvfuser
