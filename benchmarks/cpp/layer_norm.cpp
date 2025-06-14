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

//------------------------------------------------------------------------------

static void setupLayerNorm(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const float kEps = 1e-5;

  Val* eps_ptr = IrBuilder::create<Val>(kEps);

  // setup fusion
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto layer_norm_results = layer_norm(input, 1, weight, bias, eps_ptr);

  auto output = layer_norm_results.output;
  auto mean = layer_norm_results.mean;
  auto invstd = layer_norm_results.invstd;

  if (dtype != DataType::Float) {
    output = castOp(dtype, output);
  }

  fusion->addOutput(output);
  fusion->addOutput(mean);
  fusion->addOutput(invstd);
}

static void NvFuserScheduler_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  KernelArgumentHolder args = {input, weight, bias};

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      dataTypeSizeByte(dtype));
}

//------------------------------------------------------------------------------

static void Baseline_LayerNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};
  const size_t kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (auto idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      dataTypeSizeByte(dtype));
}

static void Baseline_LayerNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Float);
}

static void Baseline_LayerNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Half);
}

static void NvFuserScheduler_TIMM_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  // NHWC, norm on C
  std::vector<int64_t> input_shape{
      benchmark_state.range(0) * benchmark_state.range(2) *
          benchmark_state.range(2),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  KernelArgumentHolder args = {input, weight, bias};

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      dataTypeSizeByte(dtype));
}

static void Baseline_TIMM_LayerNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  // NHWC, norm on C
  std::vector<int64_t> input_shape{
      benchmark_state.range(0) * benchmark_state.range(2) *
          benchmark_state.range(2),
      benchmark_state.range(1)};
  const size_t kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (auto idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      dataTypeSizeByte(dtype));
}
//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_fp32,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// GPT-2
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    ->Args({8192, 768})
    ->Args({8192, 1024})
    ->Args({8192, 1280})
    ->Args({8192, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp32)
    ->Args({16384, 768})
    ->Args({16384, 1024})
    ->Args({16384, 1280})
    ->Args({16384, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_fp16,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// embedding sizes in LLMs e.g. GPT, LLaMA, PaLM
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_LLMs_fp16,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Half);
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_LLMs_fp16)
    ->Args({8192, 512})
    ->Args({8192, 768})
    ->Args({8192, 1024})
    ->Args({8192, 1280})
    ->Args({8192, 1536})
    ->Args({8192, 1600})
    ->Args({8192, 2048})
    ->Args({8192, 2560})
    ->Args({8192, 4096})
    ->Args({8192, 5140})
    ->Args({8192, 6656})
    ->Args({8192, 8192})
    ->Args({8192, 18432})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// large hidden size
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_LargeHiddenSize_fp16,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Half);
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_LargeHiddenSize_fp16)
    ->Args({8192, 60 * 1024})
    ->Args({8192, 62 * 1024})
    ->Args({8192, 64 * 1024})
    ->Args({8192, 66 * 1024})
    ->Args({8192, 68 * 1024})
    ->Args({8192, 70 * 1024})
    ->Args({8192, 72 * 1024})
    ->Args({8192, 74 * 1024})
    ->Args({8192, 76 * 1024})
    ->Args({8192, 78 * 1024})
    ->Args({8192, 80 * 1024})
    ->Args({8192, 88 * 1024})
    ->Args({8192, 96 * 1024})
    ->Args({8192, 104 * 1024})
    ->Args({8192, 112 * 1024})
    ->Args({8192, 120 * 1024})
    ->Args({8192, 128 * 1024})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_LargeHiddenSize_fp32,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Float);
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_LargeHiddenSize_fp32)
    ->Args({8192, 30 * 1024})
    ->Args({8192, 31 * 1024})
    ->Args({8192, 32 * 1024})
    ->Args({8192, 33 * 1024})
    ->Args({8192, 34 * 1024})
    ->Args({8192, 35 * 1024})
    ->Args({8192, 36 * 1024})
    ->Args({8192, 37 * 1024})
    ->Args({8192, 38 * 1024})
    ->Args({8192, 39 * 1024})
    ->Args({8192, 40 * 1024})
    ->Args({8192, 44 * 1024})
    ->Args({8192, 48 * 1024})
    ->Args({8192, 52 * 1024})
    ->Args({8192, 56 * 1024})
    ->Args({8192, 60 * 1024})
    ->Args({8192, 64 * 1024})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
//------------------------------------------------------------------------------

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 64 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 64 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TIMM, NvFuser
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_LayerNorm_fp16,
    setupLayerNorm,
    NvFuserScheduler_TIMM_LayerNorm,
    DataType::Half);

// hidden_size = 24, 40, 48, 56, 72, 152, 184, 200, 368
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{8, 16, 32, 64, 128, 256},
         {24, 40, 48, 56, 72, 152, 184, 200, 368},
         {7, 14, 28, 56, 112}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// hidden_size = 24, 40, 48, 56, 72, 152
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{128, 256, 512, 1024, 2048},
         {24, 40, 48, 56, 72, 152},
         {7, 14, 28, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TIMM, Baseline
static void Baseline_TIMM_LayerNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_TIMM_LayerNorm(benchmark_state, DataType::Half);
}

BENCHMARK(Baseline_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{8, 16, 32, 64, 128, 256},
         {24, 40, 48, 56, 72, 152, 184, 200, 368},
         {7, 14, 28, 56, 112}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_TIMM_LayerNorm_fp16)
    ->ArgsProduct(
        {{128, 256, 512, 1024, 2048},
         {24, 40, 48, 56, 72, 152},
         {7, 14, 28, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
