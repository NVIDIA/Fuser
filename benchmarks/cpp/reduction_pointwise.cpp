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
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <sstream>
#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

static void setupReductionNonBcastPointwise(
    Fusion* fusion,
    DataType dtype,
    int red_axis) {
  FusionGuard fg(fusion);

  // input fp16 is converted to fp32 before caluclation and converted back to
  // fp16 after calculation
  bool is_fp16 = dtype == DataType::Half;

  // input tensor
  auto t0 = makeContigTensor(2, dtype);
  auto t1 = makeContigTensor(1, dtype);
  fusion->addInput(t0);
  fusion->addInput(t1);
  if (is_fp16){
    t0 = castOp(DataType::Float, t0);
    t1 = castOp(DataType::Float, t1);
  }
  auto t2 = sum(t0, {red_axis});
  auto t3 = add(t2, t1);
  if (is_fp16){
    t3 = castOp(DataType::Half, t3);
  }
  fusion->addOutput(t3);
}

static void NvFuserScheduler_ReductionNonBcastPointwise(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int reduction_dim) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input_x =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
                     : at::randn({reduction_size, iter_size}, options));

  at::Tensor aten_input_epilogue = at::randn({iter_size}, options);

  std::vector<c10::IValue> aten_inputs = {aten_input_x, aten_input_epilogue};

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // inputs: input tensor [I*R] + epilogue tensor [I]
  // outputs: output_of_reduction [I]
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size * 2) *
      int64_t(dataTypeSize(dtype)));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ReductionNonBcastPointwise_Outer_fp32,
    setupReductionNonBcastPointwise,
    NvFuserScheduler_ReductionNonBcastPointwise,
    DataType::Float,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ReductionNonBcastPointwise_Outer_fp16,
    setupReductionNonBcastPointwise,
    NvFuserScheduler_ReductionNonBcastPointwise,
    DataType::Half,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ReductionNonBcastPointwise_Inner_fp32,
    setupReductionNonBcastPointwise,
    NvFuserScheduler_ReductionNonBcastPointwise,
    DataType::Float,
    1);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ReductionNonBcastPointwise_Inner_fp16,
    setupReductionNonBcastPointwise,
    NvFuserScheduler_ReductionNonBcastPointwise,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ReductionNonBcastPointwise_Outer_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ReductionNonBcastPointwise_Outer_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ReductionNonBcastPointwise_Inner_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ReductionNonBcastPointwise_Inner_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();