// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmark/utils.h>
#include <test/utils.h>

using namespace nvfuser;

//------------------------------------------------------------------------------

static void setupSoftmaxFused(
    Fusion* fusion,
    DataType dtype,
    const std::vector<UnaryOpType>& unary_op_types,
    const std::vector<BinaryOpType>& binary_op_types) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);
  // setup fusion
  auto input = makeContigTensor(2, dtype);
  fusion->addInput(input);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
  }
  for (auto uop : unary_op_types) {
    input = unaryOp(uop, input);
  }
  for (auto bop : binary_op_types) {
    input = binaryOp(bop, input, input);
  }
  auto max_val = max(input, {-1});
  auto bcast_max = broadcast(max_val, {false, true});
  auto x_max_sub = sub(input, bcast_max);
  auto exp_val = exp(x_max_sub);
  auto sum_exp = sum(exp_val, {-1});
  auto bcast_sum = broadcast(sum_exp, {false, true});

  if (std::getenv("RECALC")) {
    auto re_exp_val = exp(sub(input, bcast_max));
    auto y = mul(re_exp_val, reciprocal(bcast_sum));
    y = castOp(DataType::Half, y);
    fusion->addOutput(y);
  } else {
    auto y = mul(exp_val, reciprocal(bcast_sum));
    y = castOp(DataType::Half, y);
    fusion->addOutput(y);
  }
}

static void NvFuserScheduler_SoftmaxFused(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const std::vector<UnaryOpType>& unary_op_types,
    const std::vector<BinaryOpType>& binary_op_types) {
  NVF_ERROR(dtype == DataType::Float || dtype == DataType::Half);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto iter_size = benchmark_state.range(0);
  auto reduction_size = benchmark_state.range(1);

  at::Tensor aten_input = at::randn({iter_size, reduction_size}, options);

  std::vector<c10::IValue> aten_inputs({aten_input});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

namespace {
void addCases(benchmark::internal::Benchmark* b) {
  for (auto batch_size : {2048, 32 * 1024}) {
    for (auto hidden_size = 1024; hidden_size <= 20 * 1024;
         hidden_size += 1024) {
      b->Args({batch_size, hidden_size});
    }
  }
}
} // namespace

// will skip this UnaryOpType::IsNan op and fall back to original softmax
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_null_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{},
    std::vector<BinaryOpType>{});

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_null_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_sqrt_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{UnaryOpType::Sqrt},
    std::vector<BinaryOpType>{});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_sqrt_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_sin_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{UnaryOpType::Sin},
    std::vector<BinaryOpType>{});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_sin_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_exp_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{UnaryOpType::Exp},
    std::vector<BinaryOpType>{});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_exp_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_log10_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{UnaryOpType::Log10},
    std::vector<BinaryOpType>{});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_log10_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_add_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{},
    std::vector<BinaryOpType>{BinaryOpType::Add});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_add_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_mul_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{},
    std::vector<BinaryOpType>{BinaryOpType::Mul});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_mul_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SoftmaxFused_sin_log10_mul_add_fp16,
    setupSoftmaxFused,
    NvFuserScheduler_SoftmaxFused,
    DataType::Half,
    std::vector<UnaryOpType>{UnaryOpType::Sin, UnaryOpType::Log10},
    std::vector<BinaryOpType>{BinaryOpType::Mul, BinaryOpType::Add});
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SoftmaxFused_sin_log10_mul_add_fp16)
    ->Apply(addCases)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();