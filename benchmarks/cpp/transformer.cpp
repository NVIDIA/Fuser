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
#include <tests/cpp/multidevice_transformer.h>

using namespace nvfuser;

namespace {
// Note: We test on smaller model and input sizes to avoid high error
// accumulation for validation.
static constexpr int64_t B = 2, E = 768, H = 16, S = 128;
// Note: Dropout probabilities are set to 0. Since the dropout mask is sharded
// it throws off the seed offset between the sharded nvFuser program and the
// unsharded reference.
static constexpr double kDropoutProb = 0.0, kSdpaProb = 0.0, kSdpaScale = 1e-3;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
static constexpr double kParamScale = 0.02;
} // namespace

// Return reduction tensor view and output of reduction
static void setupTransformerForward(Fusion* fusion, DataType dtype) {
  Communicator* communicator_ = &Communicator::getInstance(); // nick TODO call Communicator::getInstance().cleanup() somewhere before program exit

  const int64_t D = communicator_->size(); // number of devices

  std::unique_ptr<DistributedTransformer> model = std::make_unique<DistributedTransformer>(
        D, B, E, H, S, kDropoutProb, kSdpaProb);

  model->setupForward(fusion, dtype, /*sequence_parallel*/false);
}

static void NvFuserScheduler_TransformerFwd(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) { /*
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
      bytes * int64_t(benchmark_state.iterations()));*/
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    TransformerForward,
    setupTransformerForward,
    NvFuserScheduler_TransformerFwd,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(TransformerForward)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
