// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/lower2device.h>
#include <executor.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmark/utils.h>
#include <test/utils.h>

using namespace nvfuser;

//------------------------------------------------------------------------------

static void setupLayerNormFused(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Half);
  const float kEps = 1e-5;

  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(1, dtype);
  auto tv1 = makeContigTensor(2, dtype);
  auto tv2 = makeContigTensor(1, dtype);
  auto tv3 = makeContigTensor(1, dtype);
  auto tv4 = makeContigTensor(1, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  fusion->addInput(tv4);
  auto tv5 = broadcast(tv0, {true, false});
  auto tv6 = castOp(DataType::Float, tv1);
  auto tv7 = castOp(DataType::Float, tv5);
  auto tv8 = add(tv6, tv7);
  auto tv9 = castOp(DataType::Half, tv8);
  auto tv10 = broadcast(tv2, {true, false});
  auto tv11 = castOp(DataType::Float, tv9);
  auto tv12 = castOp(DataType::Float, tv10);
  auto tv13 = add(tv11, tv12);
  auto tv14 = castOp(DataType::Half, tv13);
  auto tv15 = castOp(DataType::Float, tv14);
  auto tv16 = variance(tv15, {1}, false, false);
  auto tv17 = broadcast(tv16, {false, true});
  auto tv18 = sum(tv15, {1}, false);
  auto tv19 = broadcast(tv18, {false, true});

  nvfuser::Val* num_features =
      IrBuilder::create<Double>(1, dtype = DataType::Double);
  num_features = mul(num_features, tv0->getLeafDomain()[0]->extent());
  auto s20 = num_features;

  auto s21 = reciprocal(s20);
  auto tv22 = mul(tv19, s21);
  auto s23 = IrBuilder::create<Double>(kEps, dtype = DataType::Double);
  auto tv24 = add(tv17, s23);
  auto tv25 = rsqrt(tv24);
  auto tv26 = broadcast(tv22, {false, false});
  auto tv27 = castOp(DataType::Float, tv14);
  auto tv28 = sub(tv27, tv26);
  auto tv29 = broadcast(tv25, {false, false});
  auto tv30 = mul(tv28, tv29);
  auto tv31 = broadcast(tv4, {true, false});
  auto tv32 = castOp(DataType::Float, tv31);
  auto tv33 = mul(tv30, tv32);
  auto tv34 = broadcast(tv3, {true, false});
  auto tv35 = castOp(DataType::Float, tv34);
  auto tv36 = add(tv33, tv35);
  auto tv37 = castOp(DataType::Half, tv36);
  fusion->addOutput(tv37);
}

static void NvFuserScheduler_LayerNormFused(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  const int64_t num_features = benchmark_state.range(1);
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor tv0 = at::randn({num_features}, options);
  at::Tensor tv1 = at::randn(input_shape, options);
  at::Tensor tv2 = at::randn({num_features}, options);
  at::Tensor tv3 = at::randn({num_features}, options);
  at::Tensor tv4 = at::randn({num_features}, options);

  std::vector<c10::IValue> aten_inputs({tv0, tv1, tv2, tv3, tv4});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * tv1.numel() + tv0.numel() + tv2.numel() + tv3.numel() +
       tv4.numel()) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------
static void Baseline_LayerNormFused(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  const int64_t batch_size = input_shape[0];
  const int64_t hidden_size = input_shape[1];
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({hidden_size}, options);
  auto t3 = at::randn({hidden_size}, options);
  auto t4 = at::randn({hidden_size}, options);

  auto eager_implementation = [&]() {
    const float kEps = 1e-5;
    auto t5 = t0.unsqueeze(0).expand({batch_size, hidden_size});
    auto t6 = at::add(t1, t5);
    auto t7 = t2.unsqueeze(0).expand({batch_size, hidden_size});
    auto t8 = at::add(t6, t7);
    auto aten_outputs = at::native_layer_norm(t8, {hidden_size}, t4, t3, kEps);
    return std::get<0>(aten_outputs);
  };

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = eager_implementation();
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * t1.numel() + t0.numel() + t2.numel() + t3.numel() + t4.numel()) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNormFusedOp_fp16,
    setupLayerNormFused,
    NvFuserScheduler_LayerNormFused,
    DataType::Half);

// GPT-2 and 3
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNormFusedOp_fp16)
    ->Args({8192, 768})
    ->Args({8192, 1024})
    ->Args({8192, 1536})
    ->Args({8192, 1280})
    ->Args({8192, 1600})
    ->Args({8192, 2048})
    ->Args({8192, 2560})
    ->Args({8192, 4096})
    ->Args({8192, 5140})
    ->Args({8192, 12288})
    ->Args({8192, 16384})
    ->Args({8192, 20480})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void Baseline_LayerNormFused_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNormFused(benchmark_state, DataType::Half);
}

// GPT-2 and 3
BENCHMARK(Baseline_LayerNormFused_fp16)
    ->Args({8192, 768})
    ->Args({8192, 1024})
    ->Args({8192, 1536})
    ->Args({8192, 1280})
    ->Args({8192, 1600})
    ->Args({8192, 2048})
    ->Args({8192, 2560})
    ->Args({8192, 4096})
    ->Args({8192, 5140})
    ->Args({8192, 12288})
    ->Args({8192, 16384})
    ->Args({8192, 20480})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
