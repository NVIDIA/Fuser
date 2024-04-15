// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <benchmark/benchmark.h>
#include <benchmarks/cpp/utils.h>
#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/arith.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

static void setupReductionPointwise(
    Fusion* fusion,
    DataType dtype,
    int red_axis,
    bool is_broadcast) {
  FusionGuard fg(fusion);

  // input fp16 is converted to fp32 before caluclation and converted back to
  // fp16 after calculation
  bool is_fp16 = dtype == DataType::Half;

  // input tensor
  auto t0 = makeContigTensor(2, dtype);
  auto t1 = makeContigTensor(is_broadcast ? 2 : 1, dtype);
  fusion->addInput(t0);
  fusion->addInput(t1);
  if (is_fp16) {
    t0 = castOp(DataType::Float, t0);
    t1 = castOp(DataType::Float, t1);
  }
  auto t2 = sum(t0, {red_axis});
  if (is_broadcast) {
    t2 = broadcast(t2, {0 == red_axis, 1 == red_axis});
  }
  auto t3 = add(t2, t1);
  if (is_fp16) {
    t3 = castOp(DataType::Half, t3);
  }
  fusion->addOutput(t3);
}

static void NvFuserScheduler_ReductionPointwise(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int reduction_dim,
    bool is_broadcast) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  auto input_shape =
      (reduction_dim == 1 ? std::vector<int64_t>{iter_size, reduction_size}
                          : std::vector<int64_t>{reduction_size, iter_size});

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input_x = at::randn(input_shape, options);

  at::Tensor aten_input_epilogue = is_broadcast
      ? at::randn(input_shape, options)
      : at::randn({iter_size}, options);

  std::vector<c10::IValue> aten_inputs = {aten_input_x, aten_input_epilogue};

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // inputs: input tensor [I*R] + epilogue tensor [I]
  // outputs: output_of_reduction [I]
  auto epilogue_size = is_broadcast ? iter_size * reduction_size : iter_size;
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + epilogue_size + iter_size) *
      int64_t(dataTypeSize(dtype)));
}

// define 8 benchmarks for different combinations of data type, reduction dim
// and epilogue.
#define NVFUSER_REDUCTION_POINTWISE(                     \
    NAME_SUFFIX, DATA_TYPE, REDU_DIM, BCAST_ELOG)        \
  NVFUSER_BENCHMARK_DEFINE(                              \
      NvFuserScheduler_ReductionPointwise_##NAME_SUFFIX, \
      setupReductionPointwise,                           \
      NvFuserScheduler_ReductionPointwise,               \
      DATA_TYPE,                                         \
      REDU_DIM,                                          \
      BCAST_ELOG)

NVFUSER_REDUCTION_POINTWISE(Outer_fp32_NonBcastElog, DataType::Float, 0, false);
NVFUSER_REDUCTION_POINTWISE(Outer_fp16_NonBcastElog, DataType::Half, 0, false);
NVFUSER_REDUCTION_POINTWISE(Inner_fp32_NonBcastElog, DataType::Float, 1, false);
NVFUSER_REDUCTION_POINTWISE(Inner_fp16_NonBcastElog, DataType::Half, 1, false);

NVFUSER_REDUCTION_POINTWISE(Outer_fp32_BcastElog, DataType::Float, 0, true);
NVFUSER_REDUCTION_POINTWISE(Outer_fp16_BcastElog, DataType::Half, 0, true);
NVFUSER_REDUCTION_POINTWISE(Inner_fp32_BcastElog, DataType::Float, 1, true);
NVFUSER_REDUCTION_POINTWISE(Inner_fp16_BcastElog, DataType::Half, 1, true);

// run all the benchmarks with the following configurations:
#define NV_RUN(BENCHMARK_NAME)                     \
  NVFUSER_BENCHMARK_RUN(BENCHMARK_NAME)            \
      ->RangeMultiplier(2)                         \
      ->Ranges({{512, 512 * 64}, {512, 512 * 64}}) \
      ->Unit(benchmark::kMicrosecond)              \
      ->UseManualTime();

NV_RUN(NvFuserScheduler_ReductionPointwise_Outer_fp32_NonBcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Outer_fp16_NonBcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Inner_fp32_NonBcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Inner_fp16_NonBcastElog);

NV_RUN(NvFuserScheduler_ReductionPointwise_Outer_fp32_BcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Outer_fp16_BcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Inner_fp32_BcastElog);
NV_RUN(NvFuserScheduler_ReductionPointwise_Inner_fp16_BcastElog);