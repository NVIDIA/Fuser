// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

static void NvFuserScheduler_LayerNormBackward_HeuristicCache(
    benchmark::State& benchmark_state) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // PreAllocate
  std::unique_ptr<FusionExecutorCache> executor_cache;
  KernelArgumentHolder args;

  std::vector<int64_t> shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  auto runtime = getLayerBackwardNormRuntime(
      std::move(fusion_ptr), executor_cache, args, shape, norm_shape);

  NVF_ERROR(runtime->getMaybeHeuristicsFor(args).has_value());

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    runtime->getMaybeHeuristicsFor(args);
  }
}

static void NvFuserScheduler_LayerNormForward_HeuristicCache(
    benchmark::State& benchmark_state) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // PreAllocate
  std::unique_ptr<FusionExecutorCache> executor_cache;
  KernelArgumentHolder args;

  std::vector<int64_t> shape{20, 100, 35, 67};
  std::vector<int64_t> norm_shape{67};

  auto runtime = getLayerForwardNormRuntime(
      std::move(fusion_ptr), executor_cache, args, shape, norm_shape);

  NVF_ERROR(runtime->getMaybeHeuristicsFor(args).has_value());

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    runtime->getMaybeHeuristicsFor(args);
  }
}

BENCHMARK(NvFuserScheduler_LayerNormBackward_HeuristicCache)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(NvFuserScheduler_LayerNormForward_HeuristicCache)
    ->Unit(benchmark::kMicrosecond);
