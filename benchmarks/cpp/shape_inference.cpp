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

void LayerNormBackward_ShapeInference_Base(
    benchmark::State& benchmark_state,
    bool disable_launch_parameter_cache) {
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

  executor_cache->profile(true);
  executor_cache->disableKernelLaunch();
  executor_cache->runFusionWithInputs(args);
  if (disable_launch_parameter_cache) {
    executor_cache->disableLaunchParamCache();
  }

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    executor_cache->runFusionWithInputs(args);
  }
}

static void NvFuserScheduler_LayerNormBackward_ShapeInference(
    benchmark::State& benchmark_state) {
  LayerNormBackward_ShapeInference_Base(benchmark_state, true);
}

static void NvFuserScheduler_LayerNormBackward_NoShapeInferenceCachedBaseline(
    benchmark::State& benchmark_state) {
  LayerNormBackward_ShapeInference_Base(benchmark_state, false);
}

void LayerNormForward_ShapeInferenceBase(
    benchmark::State& benchmark_state,
    bool disable_launch_param_cache) {
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

  executor_cache->profile(true);
  executor_cache->disableKernelLaunch();
  executor_cache->runFusionWithInputs(args);

  if (disable_launch_param_cache) {
    executor_cache->disableLaunchParamCache();
  }

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    executor_cache->runFusionWithInputs(args);
  }
}

static void NvFuserScheduler_LayerNormForward_NoShapeInferenceCachedBaseline(
    benchmark::State& benchmark_state) {
  LayerNormForward_ShapeInferenceBase(benchmark_state, false);
}

static void NvFuserScheduler_LayerNormForward_ShapeInference(
    benchmark::State& benchmark_state) {
  LayerNormForward_ShapeInferenceBase(benchmark_state, true);
}

BENCHMARK(NvFuserScheduler_LayerNormBackward_ShapeInference)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(NvFuserScheduler_LayerNormForward_ShapeInference)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(NvFuserScheduler_LayerNormBackward_NoShapeInferenceCachedBaseline)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(NvFuserScheduler_LayerNormForward_NoShapeInferenceCachedBaseline)
    ->Unit(benchmark::kMicrosecond);
