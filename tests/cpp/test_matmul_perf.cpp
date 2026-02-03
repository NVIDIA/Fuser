// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <chrono>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "exceptions.h"
#include "fusion.h"
#include "fusion_guard.h"
#include "mma_type.h"
#include "ops/all_ops.h"
#include "preseg_passes/pre_segmenter.h"
#include "runtime/fusion_executor_cache.h"
#include "runtime/matmul_tma.h"
#include "scheduler/matmul.h"
#include "tests/cpp/utils.h"

#if defined(NVFUSER_ENABLE_CUTLASS)
#include "cutlass/arch/config.h"
#endif

namespace nvfuser {

namespace {

struct MatmulProblem {
  int64_t m;
  int64_t n;
  int64_t k;
};

std::unique_ptr<Fusion> buildMatmulFusion(DataType dtype) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto a = makeContigTensor(2, dtype);
  auto b = makeContigTensor(2, dtype);
  fusion->addInput(a);
  fusion->addInput(b);

  auto layout = MmaLayout::TT;
  auto a_canon = canonicalizeInputToBMNK(a, layout, MmaOperand::A);
  auto b_canon = canonicalizeInputToBMNK(b, layout, MmaOperand::B);
  auto c = fusedMultiplySum(a_canon, b_canon, {-1});
  auto d = castOp(dtype, c);

  fusion->addOutput(d);
  OptimizationPass<preseg_passes::PreSegmenter>::runPass(fusion.get());
  return fusion;
}

template <typename Fn>
double timeMs(int warmup_iters, int iters, Fn&& fn) {
  for (int i = 0; i < warmup_iters; ++i) {
    fn();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  NVFUSER_CUDA_RT_SAFE_CALL(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iters);
}

void printResult(
    const std::string& label,
    const MatmulProblem& problem,
    double ms_per_iter) {
  const double flops = 2.0 * static_cast<double>(problem.m) *
      static_cast<double>(problem.n) * static_cast<double>(problem.k);
  const double gflops = flops / (ms_per_iter * 1.0e6);
  std::cout << label << " M=" << problem.m << " N=" << problem.n
            << " K=" << problem.k << " : " << ms_per_iter << " ms, "
            << gflops << " GFLOPs" << std::endl;
}

} // namespace

TEST(MatmulPerfTest, CompareImplementations) {
  if (!deviceMajorMinorCheck(9, 0)) {
    GTEST_SKIP() << "Requires SM90 (Hopper).";
  }
#if !defined(NVFUSER_ENABLE_CUTLASS)
  GTEST_SKIP() << "CUTLASS support is disabled.";
#endif
#if defined(NVFUSER_ENABLE_CUTLASS) && !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  GTEST_SKIP() << "CUTLASS SM90 support is unavailable.";
#endif

  constexpr int warmup_iters = 100;
  constexpr int iters = 1000;

  std::vector<MatmulProblem> problems{
      {1024, 1024, 1024},
      {2048, 2048, 2048},
      {4096, 4096, 4096},
  };

  std::vector<at::ScalarType> dtypes{
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
  };

  for (auto dtype : dtypes) {
    for (const auto& problem : problems) {
      at::cuda::CUDAGuard device_guard{0};
      auto options =
          at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
      auto a = at::randn({problem.m, problem.k}, options);
      auto b = at::randn({problem.k, problem.n}, options);

      auto ref = at::matmul(a, b);

      auto tma_out = matmulTma(a, b);
      EXPECT_TRUE(at::allclose(tma_out, ref, 1e-2, 1e-2));

      auto fusion = buildMatmulFusion(
          dtype == at::ScalarType::Half ? DataType::Half : DataType::BFloat16);
      FusionExecutorCache executor_cache(std::move(fusion));
      auto fuser_out = executor_cache.runFusionWithInputs({a, b});
      auto fuser_tensor = fuser_out[0].as<at::Tensor>();
      EXPECT_TRUE(at::allclose(fuser_tensor, ref, 1e-2, 1e-2));

      double torch_ms = timeMs(warmup_iters, iters, [&]() {
        auto out = at::matmul(a, b);
        (void)out;
      });
      printResult("torch", problem, torch_ms);

      double fuser_ms = timeMs(warmup_iters, iters, [&]() {
        auto out = executor_cache.runFusionWithInputs({a, b});
        (void)out;
      });
      printResult("nvfuser", problem, fuser_ms);

      double tma_ms = timeMs(warmup_iters, iters, [&]() {
        auto out = matmulTma(a, b);
        (void)out;
      });
      printResult("tma", problem, tma_ms);
    }
  }
}

} // namespace nvfuser
