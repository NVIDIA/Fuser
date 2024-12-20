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
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/utils.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

// Return reduction tensor view and output of reduction
static void setupDivMaxSoftmaxDropoutForward(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(4)
                        .dtype(dtype)
                        .contiguity({true, std::nullopt, std::nullopt, true})
                        .shape({-1, 1, 1, -1})
                        .build();
  TensorView* tv1 = makeContigTensor(4, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  // TODO: should be input
  auto d16 = IrBuilder::create<Val>(1.0);

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv1 = castOp(DataType::Float, tv1);
  }

  auto tv2 = div(tv1, d16);
  auto tv3 = add(tv2, tv0);

  auto tv10 = softmax(tv3, 3);
  auto dropout_tvs = dropout(tv10, IrBuilder::create<Val>(0.9));
  auto tv12 = dropout_tvs.mask;
  auto tv14 = dropout_tvs.output;

  if (is_fp16) {
    tv14 = castOp(DataType::Half, tv14);
    tv10 = castOp(DataType::Half, tv10);
    tv3 = castOp(DataType::Half, tv3);
  }

  fusion->addOutput(tv14);
  fusion->addOutput(tv12);
  fusion->addOutput(tv10);
  fusion->addOutput(tv3);
}

static void NvFuserScheduler_DivMaxSoftDropFwd(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  auto w = benchmark_state.range(0);
  auto x = benchmark_state.range(1);
  auto y = benchmark_state.range(2);
  auto z = benchmark_state.range(3);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({w, 1, 1, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> at_inputs = {t0, t1};

  auto bytes =
      runBenchmarkIterations(benchmark_state, executor_cache, at_inputs);

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    nick_transformer,
    setupDivMaxSoftmaxDropoutForward,
    NvFuserScheduler_DivMaxSoftDropFwd,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(nick_transformer)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
