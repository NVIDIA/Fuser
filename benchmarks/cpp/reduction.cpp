// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

// Return reduction tensor view and output of reduction
static void setupReduction(Fusion* fusion, DataType dtype, int red_axis) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = makeContigTensor(2, dtype);
  fusion->addInput(tv0);

  TensorView* tv0_cast = tv0;
  if (is_fp16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }

  TensorView* tv1 = sum(tv0_cast, {red_axis});

  TensorView* tv1_cast = tv1;
  if (is_fp16) {
    tv1_cast = castOp(DataType::Half, tv1);
  }

  fusion->addOutput(tv1_cast);
}

static void NvFuserScheduler_Reduction(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype,
    int reduction_dim) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
                     : at::randn({reduction_size, iter_size}, options));

  KernelArgumentHolder inputs({aten_input});

  runBenchmarkIterations(benchmark_state, executor_cache, inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size) * dataTypeSizeByte(dtype));
}

static void Baseline_Reduction(
    benchmark::State& benchmark_state,
    DataType dtype,
    int reduction_dim) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
                     : at::randn({reduction_size, iter_size}, options));

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = aten_input.sum({reduction_dim});
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size) * dataTypeSizeByte(dtype));
}

//------------------------------------------------------------------------------

static void Baseline_Reduction_Outer_fp32(benchmark::State& benchmark_state) {
  Baseline_Reduction(benchmark_state, DataType::Float, 0);
}

static void Baseline_Reduction_Outer_fp16(benchmark::State& benchmark_state) {
  Baseline_Reduction(benchmark_state, DataType::Half, 0);
}

static void Baseline_Reduction_Inner_fp32(benchmark::State& benchmark_state) {
  Baseline_Reduction(benchmark_state, DataType::Float, 1);
}

static void Baseline_Reduction_Inner_fp16(benchmark::State& benchmark_state) {
  Baseline_Reduction(benchmark_state, DataType::Half, 1);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction_Outer_fp32,
    setupReduction,
    NvFuserScheduler_Reduction,
    DataType::Float,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction_Outer_fp16,
    setupReduction,
    NvFuserScheduler_Reduction,
    DataType::Half,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction_Inner_fp32,
    setupReduction,
    NvFuserScheduler_Reduction,
    DataType::Float,
    1);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction_Inner_fp16,
    setupReduction,
    NvFuserScheduler_Reduction,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1024, 1024 * 512}, {2, 4 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 4 * 1024}, {1024, 1024 * 512}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1024, 1024 * 1024}, {2, 4 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 4 * 1024}, {1024, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Reduction_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
