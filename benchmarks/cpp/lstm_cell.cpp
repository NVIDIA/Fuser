// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

// TODO: add LSTM function to composite operations
// Function Signature: cy, hy = lstm(x, cx)
static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  TensorView* tvs[16];
  for (size_t i = 0; i < 16; i++) {
    tvs[i] = makeSymbolicTensor(2, DataType::Float);
    tvs[i]->setContiguity({false, true});
    fusion->addInput(tvs[i]);
  }

  const auto cx = makeContigTensor(2, DataType::Float);
  fusion->addInput(cx);

  const auto in_x = add(add(add(tvs[0], tvs[1]), tvs[2]), tvs[3]);
  const auto forget_x = add(add(add(tvs[4], tvs[5]), tvs[6]), tvs[7]);
  const auto cell_x = add(add(add(tvs[8], tvs[9]), tvs[10]), tvs[11]);
  const auto out_x = add(add(add(tvs[12], tvs[13]), tvs[14]), tvs[15]);
  auto lstm_result = lstm(cx, in_x, forget_x, cell_x, out_x);

  fusion->addOutput(lstm_result.cell);
  fusion->addOutput(lstm_result.hidden);
}

static KernelArgumentHolder setupInputs(int hidden_features, int batch_size) {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const at::Tensor large_tensor0 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor1 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor2 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor3 =
      at::randn({batch_size, hidden_features * 4}, options);

  const auto chunked0 = large_tensor0.chunk(4, 1);
  const auto chunked1 = large_tensor1.chunk(4, 1);
  const auto chunked2 = large_tensor2.chunk(4, 1);
  const auto chunked3 = large_tensor3.chunk(4, 1);

  KernelArgumentHolder args;
  args.push(chunked0);
  args.push(chunked1);
  args.push(chunked2);
  args.push(chunked3);

  const auto at_cx = at::randn({batch_size, hidden_features}, options);
  args.push(at_cx);

  return args;
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_SetupFusion(
    benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(NvFuserScheduler_LstmCell_SetupFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_AutoSchedule(
    benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    KernelArgumentHolder args = setupInputs(kHiddenFeatures, kBatchSize);
    benchmark_state.ResumeTiming();

    // Auto-schedule
    SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);
  }
}

BENCHMARK(NvFuserScheduler_LstmCell_AutoSchedule)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_Lower(benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs(kHiddenFeatures, kBatchSize);

  SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  for (auto _ : benchmark_state) {
    GpuLower(&fusion).run();
  }
}

BENCHMARK(NvFuserScheduler_LstmCell_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_Compile(
    benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs(kHiddenFeatures, kBatchSize);

  SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  for (auto _ : benchmark_state) {
    KernelExecutor ke;
    ke.compile(&fusion, args);
  }
}

BENCHMARK(NvFuserScheduler_LstmCell_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_RunFusion(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs(hidden_features, batch_size);

  // outputs
  KernelArgumentHolder outputs;

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.compile(&fusion, args);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : benchmark_state) {
    outputs = ke.run(args, {}, heuristic_params->lparams);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_CAPTURE(NvFuserScheduler_LstmCell_RunFusion, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(NvFuserScheduler_LstmCell_RunFusion, Medium, 1024, 128)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_RunFusion_GpuOnly(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs(hidden_features, batch_size);

  // outputs
  std::vector<at::Tensor> outputs;

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.compile(&fusion, args);

  runBenchmarkIterations(benchmark_state, &ke, args, heuristic_params->lparams);
}

BENCHMARK_CAPTURE(NvFuserScheduler_LstmCell_RunFusion_GpuOnly, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(
    NvFuserScheduler_LstmCell_RunFusion_GpuOnly,
    Medium,
    1024,
    128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void NvFuserScheduler_LstmCell_RunFusion_CpuOnly(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  KernelArgumentHolder args = setupInputs(hidden_features, batch_size);

  // outputs
  KernelArgumentHolder outputs;

  auto heuristic_params =
      SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, args);

  KernelExecutor ke;
  ke.setExecuteKernelFlag(false);
  ke.compile(&fusion, args);

  for (auto _ : benchmark_state) {
    outputs = ke.run(args, {}, heuristic_params->lparams);
  }
}

BENCHMARK_CAPTURE(NvFuserScheduler_LstmCell_RunFusion_CpuOnly, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(
    NvFuserScheduler_LstmCell_RunFusion_CpuOnly,
    Medium,
    1024,
    128)
    ->Unit(benchmark::kMicrosecond);
