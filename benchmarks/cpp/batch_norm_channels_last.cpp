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

static void setupBatchNorm_nhwc(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto momentum_ptr = IrBuilder::create<Val>(kMomentum);
  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr,
      true);

  auto output = result.output;

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_BatchNorm_nhwc(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(2),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[3]}, options);
  at::Tensor at_bias = at::zeros({input_shape[3]}, options);
  at::Tensor at_run_mean = at::zeros({input_shape[3]}, fp32_options);
  at::Tensor at_run_var = at::ones({input_shape[3]}, fp32_options);
  KernelArgumentHolder args = {
      at_x, at_weight, at_bias, at_run_mean, at_run_var};

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      ((2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
           dataTypeSizeByte(dtype) +
       (2 * (at_run_mean.numel() + at_run_var.numel()) *
        dataTypeSizeByte(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_nhwc(
    benchmark::State& benchmark_state,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options)
                        .contiguous(c10::MemoryFormat::ChannelsLast);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_run_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_run_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_run_mean = c10::optional<at::Tensor>(at_run_mean);
  auto ato_run_var = c10::optional<at::Tensor>(at_run_var);

  auto output = at::batch_norm(
      at_x,
      ato_weight,
      ato_bias,
      ato_run_mean,
      ato_run_var,
      true,
      kMomentum,
      kEps,
      true);

  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    at::_ops::_batch_norm_impl_index::call(
        at_x,
        ato_weight,
        ato_bias,
        ato_run_mean,
        ato_run_var,
        true,
        kMomentum,
        kEps,
        true);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      ((2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
           dataTypeSizeByte(dtype) +
       (2 * (at_run_mean.numel() + at_run_var.numel()) *
        dataTypeSizeByte(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_nhwc_cuDNN_fp32(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_nhwc(benchmark_state, DataType::Float);
}

static void Baseline_BatchNorm_nhwc_cuDNN_fp16(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_nhwc(benchmark_state, DataType::Half);
}

// Simple aliases just for names in the printed output
static void Baseline_ResNet_BatchNorm_nhwc_cuDNN_fp16(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_nhwc(benchmark_state, DataType::Half);
}

static void Baseline_ResNext_BatchNorm_nhwc_cuDNN_fp16(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_nhwc(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_nhwc_fp32,
    setupBatchNorm_nhwc,
    NvFuserScheduler_BatchNorm_nhwc,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_nhwc_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_nhwc_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_nhwc_fp16,
    setupBatchNorm_nhwc,
    NvFuserScheduler_BatchNorm_nhwc,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_nhwc_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_nhwc_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_BatchNorm_nhwc_cuDNN_fp32)
    // ->RangeMultiplier(2)
    // cuDNN didn't make it to 1024
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_nhwc_cuDNN_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_nhwc_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_nhwc_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
// RESNET and REXNEXT benchmarks

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNet_BatchNorm_nhwc_fp16,
    setupBatchNorm_nhwc,
    NvFuserScheduler_BatchNorm_nhwc,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNet_BatchNorm_nhwc_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNext_BatchNorm_nhwc_fp16,
    setupBatchNorm_nhwc,
    NvFuserScheduler_BatchNorm_nhwc,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNext_BatchNorm_nhwc_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// Permutation of TIMM sizes
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_TIMM_BatchNorm_nhwc_fp16,
    setupBatchNorm_nhwc,
    NvFuserScheduler_BatchNorm_nhwc,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_BatchNorm_nhwc_fp16)
    ->ArgsProduct(
        {{8, 16, 32, 64, 128, 256},
         {24, 40, 48, 56, 72, 152, 184, 200, 368},
         {7, 14, 28, 56, 112}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_TIMM_BatchNorm_nhwc_fp16)
    ->ArgsProduct(
        {{128, 256, 512, 1024, 2048},
         {24, 40, 48, 56, 72, 152},
         {7, 14, 28, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_ResNet_BatchNorm_nhwc_cuDNN_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_ResNext_BatchNorm_nhwc_cuDNN_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
