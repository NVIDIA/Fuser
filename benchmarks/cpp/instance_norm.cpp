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
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

static void setupInstanceNorm(
    Fusion* fusion,
    DataType dtype,
    bool channels_last_3d = false) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  auto input = makeContigTensor(4, dtype);
  if (channels_last_3d) {
    input = makeContigTensor(5, dtype);
  }
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

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  auto momentum_ptr = IrBuilder::create<Val>(kMomentum);
  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto norm = instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr,
      channels_last_3d);

  auto output = unaryOp(UnaryOpType::Relu, norm.output);

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_InstanceNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype,
    bool channels_last_3d = false) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  std::vector<int64_t> input_shape_3d{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x =
      at::randn(channels_last_3d ? input_shape_3d : input_shape, options);
  at::Tensor at_weight = at::ones({benchmark_state.range(2)}, options);
  at::Tensor at_bias = at::zeros({benchmark_state.range(2)}, options);
  at::Tensor at_mean = at::zeros({benchmark_state.range(2)}, fp32_options);
  at::Tensor at_var = at::ones({benchmark_state.range(2)}, fp32_options);

  KernelArgumentHolder args = {at_x, at_weight, at_bias, at_mean, at_var};
  std::vector<at::Tensor> outputs;

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  const size_t kChannels = benchmark_state.range(2);

  // Read: x, weight, bias, running_mean, running_var
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + at_x.numel() * 2) * dataTypeSizeByte(dtype) +
       (kChannels * 2 * 2) * dataTypeSizeByte(DataType::Float)));
}

// ------------------------------------------------------------------------------
// performance of https://github.com/NVIDIA/Fuser/issues/443
static void setupInstanceNormNHWC(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);
  FusionGuard fg(fusion);

  auto input = makeContigTensor(4, dtype);
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

  const float kEps = 1e-5;
  auto s1 = IrBuilder::create<Val>(kEps);
  auto var_mean = variance_mean(input, {1, 2}, 0, true);
  auto tv_mean = var_mean.mean;
  auto tv_var = var_mean.var;
  auto tv_var_s1 = add(tv_var, s1);
  auto tv_sqrt = sqrt(tv_var_s1);
  auto tv_diff = sub(input, tv_mean);
  auto tv_div = div(tv_diff, tv_sqrt);
  auto tv_mul = mul(tv_div, weight);
  auto output = add(tv_mul, bias);
  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }
  fusion->addOutput(output);
}

static void NvFuserScheduler_InstanceNormNHWC(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::randn({benchmark_state.range(2)}, options);
  at::Tensor at_bias = at::randn({benchmark_state.range(2)}, options);

  KernelArgumentHolder args = {at_x, at_weight, at_bias};
  std::vector<at::Tensor> outputs;

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  const size_t kChannels = benchmark_state.range(2);

  // Read: x, weight, bias
  // Write: y
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + at_x.numel() * 2) * dataTypeSizeByte(dtype)));
}

static void Baseline_InstanceNorm(
    benchmark::State& benchmark_state,
    DataType dtype,
    bool channels_last_3d = false) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};
  std::vector<int64_t> input_shape_3d{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(1),
  };

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const auto aten_dtype = data_type_to_aten(dtype);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x = at::randn(input_shape, options);
  if (channels_last_3d) {
    at_x = at::randn(
        input_shape_3d,
        options.memory_format(c10::MemoryFormat::ChannelsLast3d));
  }
  at::Tensor at_weight = at::ones({benchmark_state.range(2)}, options);
  at::Tensor at_bias = at::zeros({benchmark_state.range(2)}, options);
  at::Tensor at_mean = at::zeros({benchmark_state.range(2)}, fp32_options);
  at::Tensor at_var = at::ones({benchmark_state.range(2)}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto norm = at::instance_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);
    auto output = at::relu(norm);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  const size_t kChannels = benchmark_state.range(2);

  // Read: x, weight, bias, running_mean, running_var
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + at_x.numel() * 2) * dataTypeSizeByte(dtype) +
       (kChannels * 2 * 2) * dataTypeSizeByte(DataType::Float)));
}

//------------------------------------------------------------------------------

static void Baseline_InstanceNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_InstanceNorm(benchmark_state, DataType::Float);
}

static void Baseline_InstanceNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_InstanceNorm(benchmark_state, DataType::Half);
}

static void Baseline_InstanceNorm_fp32_channels_last_3d(
    benchmark::State& benchmark_state) {
  Baseline_InstanceNorm(benchmark_state, DataType::Float, true);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InstanceNorm_fp32,
    setupInstanceNorm,
    NvFuserScheduler_InstanceNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InstanceNorm_fp16,
    setupInstanceNorm,
    NvFuserScheduler_InstanceNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InstanceNorm3d_channels_last_fp32,
    setupInstanceNorm,
    NvFuserScheduler_InstanceNorm,
    DataType::Float,
    true);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm3d_channels_last_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{1, 8}, {128, 128}, {32, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm3d_channels_last_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{1, 8}, {64, 64}, {64, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm3d_channels_last_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{1, 8}, {32, 32}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm3d_channels_last_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{1, 8}, {16, 16}, {256, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNorm3d_channels_last_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{1, 8}, {4, 8}, {320, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_InstanceNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp32_channels_last_3d)
    ->RangeMultiplier(2)
    ->Ranges({{2, 8}, {128, 128}, {32, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp32_channels_last_3d)
    ->RangeMultiplier(2)
    ->Ranges({{2, 8}, {64, 64}, {64, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp32_channels_last_3d)
    ->RangeMultiplier(2)
    ->Ranges({{2, 8}, {16, 16}, {256, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp32_channels_last_3d)
    ->RangeMultiplier(2)
    ->Ranges({{2, 8}, {4, 8}, {320, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InstanceNormNHWC_fp16,
    setupInstanceNormNHWC,
    NvFuserScheduler_InstanceNormNHWC,
    DataType::Half);
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNormNHWC_fp16)
    ->ArgsProduct({
        {32, 64, 128, 256}, // N
        {7, 14, 28, 32, 56, 64, 112, 128}, // H,W
        {32, 64, 128, 256}, // C
    }) // HW
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InstanceNormNHWC_fp32,
    setupInstanceNormNHWC,
    NvFuserScheduler_InstanceNormNHWC,
    DataType::Float);
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InstanceNormNHWC_fp32)
    ->ArgsProduct({
        {32, 64, 128, 256}, // N
        {7, 14, 28, 32, 56, 64, 112, 128}, // H,W
        {32, 64, 128, 256}, // C
    }) // HW
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
