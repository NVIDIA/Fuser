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

static void setupLayerNorm_BWD(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  // setup fusion
  auto grad_out = makeContigTensor(2, dtype);
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);

  auto mean = TensorViewBuilder()
                  .contiguity({false, std::nullopt})
                  .shape({-1, 1})
                  .dtype(DataType::Float)
                  .build();
  auto rstd = TensorViewBuilder()
                  .contiguity({false, std::nullopt})
                  .shape({-1, 1})
                  .dtype(DataType::Float)
                  .build();

  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(mean);
  fusion->addInput(rstd);

  if (dtype == DataType::Half) {
    grad_out = castOp(DataType::Float, grad_out);
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto layer_norm_results = layer_norm_backward(
      grad_out, input, {1}, mean, rstd, weight, bias, {true, true, true});

  if (dtype != DataType::Float) {
    layer_norm_results.grad_input =
        castOp(dtype, layer_norm_results.grad_input);
    layer_norm_results.grad_bias = castOp(dtype, layer_norm_results.grad_bias);
    layer_norm_results.grad_weight =
        castOp(dtype, layer_norm_results.grad_weight);
  }

  fusion->addOutput(layer_norm_results.grad_input);
  fusion->addOutput(layer_norm_results.grad_bias);
  fusion->addOutput(layer_norm_results.grad_weight);
}

static void NvFuserScheduler_LayerNorm_BWD(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto maybe_fp16_options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor grad_out = at::randn(input_shape, maybe_fp16_options);
  at::Tensor input = at::randn(input_shape, maybe_fp16_options);
  at::Tensor weight = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor bias = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor mean = at::randn({input_shape[0], 1}, fp32_options);
  at::Tensor rstd = at::randn({input_shape[0], 1}, fp32_options);

  KernelArgumentHolder args = {grad_out, input, weight, bias, mean, rstd};

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (3 * input.numel() + weight.numel() + bias.numel() + mean.numel() +
       rstd.numel()) *
      dataTypeSizeByte(dtype));
}

//------------------------------------------------------------------------------

static void Baseline_LayerNorm_BWD(
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
  auto maybe_fp16_options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor grad_out = at::randn(input_shape, maybe_fp16_options);
  at::Tensor input = at::randn(input_shape, maybe_fp16_options);
  at::Tensor weight = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor bias = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor mean = at::randn({input_shape[0], 1}, fp32_options);
  at::Tensor rstd = at::randn({input_shape[0], 1}, fp32_options);
  std::array<bool, 3> output_mask = {true, true, true};

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    at::native_layer_norm_backward(
        grad_out, input, norm_shape, mean, rstd, weight, bias, output_mask);

    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (3 * input.numel() + weight.numel() + bias.numel() + mean.numel() +
       rstd.numel()) *
      dataTypeSizeByte(dtype));
}

static void Baseline_LayerNorm_BWD_fp32(benchmark::State& benchmark_state) {
  Baseline_LayerNorm_BWD(benchmark_state, DataType::Float);
}

static void Baseline_LayerNorm_BWD_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNorm_BWD(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_fp32,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 16 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 16 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// GPT-2
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    ->Args({8 * 1024, 768})
    ->Args({8 * 1024, 1024})
    ->Args({8 * 1024, 1280})
    ->Args({8 * 1024, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    ->Args({16 * 1024, 768})
    ->Args({16 * 1024, 1024})
    ->Args({16 * 1024, 1280})
    ->Args({16 * 1024, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_fp16,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// GPT-2
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    ->Args({8 * 1024, 768})
    ->Args({8 * 1024, 1024})
    ->Args({8 * 1024, 1280})
    ->Args({8 * 1024, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    ->Args({16 * 1024, 768})
    ->Args({16 * 1024, 1024})
    ->Args({16 * 1024, 1280})
    ->Args({16 * 1024, 1600})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

std::vector<std::vector<int64_t>> getArgs(
    const int64_t batch_size,
    const int64_t hidden_size_factor) {
  std::vector<std::vector<int64_t>> args;
  std::vector<int64_t> prime_factors = {
      7,
      11,
      13,
      17,
      19,
      101,
      103,
      107,
      109,
      113,
      211,
      223,
      227,
      229,
      233,
      239,
      241};
  for (auto p : prime_factors) {
    args.push_back({batch_size, p * hidden_size_factor});
  }
  return args;
}

// Non-divisible batch split
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_nondiv_fp16,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Half);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_nondiv_fp32,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Float);
void addArgsFactor64(benchmark::internal::Benchmark* b) {
  const int64_t batch_size = 16 * 1024;
  const auto& args = getArgs(batch_size, 64l);
  for (const auto& a : args) {
    b->Args(a);
  }
}
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_nondiv_fp16)
    ->Apply(addArgsFactor64)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_nondiv_fp32)
    ->Apply(addArgsFactor64)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Non-vectorized hidden sizes
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_non64_fp16,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Half);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_non64_fp32,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Float);
void addArgsFactor63(benchmark::internal::Benchmark* b) {
  const int64_t batch_size = 16 * 1024;
  const auto& args = getArgs(batch_size, 63l);
  for (const auto& a : args) {
    b->Args(a);
  }
}
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_non64_fp16)
    ->Apply(addArgsFactor63)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_non64_fp32)
    ->Apply(addArgsFactor63)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Multi-reductions per block
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_MultiReductionsPerBlock_fp16,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Half);
NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_LayerNorm_BWD_MultiReductionsPerBlock_fp16)
    ->Args({8 * 1024, 128})
    ->Args({8 * 1024, 256})
    ->Args({8 * 1024, 384})
    ->Args({8 * 1024, 512})
    ->Args({8 * 1024, 640})
    ->Args({8 * 1024, 768})
    ->Args({8 * 1024, 896})
    ->Args({8 * 1024, 1023})
    ->Args({8 * 1024, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
NVFUSER_BENCHMARK_RUN(
    NvFuserScheduler_LayerNorm_BWD_MultiReductionsPerBlock_fp16)
    ->Args({128, 128})
    ->Args({128, 256})
    ->Args({128, 384})
    ->Args({128, 512})
    ->Args({128, 640})
    ->Args({128, 768})
    ->Args({128, 896})
    ->Args({128, 1023})
    ->Args({128, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
//------------------------------------------------------------------------------

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 16 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 16 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
