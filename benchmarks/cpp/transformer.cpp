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
#include <csrc/multidevice/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/multidevice_transformer.h>

using namespace nvfuser;

namespace {
// Note: We test on smaller model and input sizes to avoid high error
// accumulation for validation.
  constexpr int64_t B = 2, E = 768, H = 16, S = 128;
// Note: Dropout probabilities are set to 0. Since the dropout mask is sharded
// it throws off the seed offset between the sharded nvFuser program and the
// unsharded reference.
  constexpr double kDropoutProb = 0.0, kSdpaProb = 0.0, kSdpaScale = 1e-3;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
  constexpr double kParamScale = 0.02;
} // namespace

// Return reduction tensor view and output of reduction
void setupTransformerForward(Fusion* fusion, DataType dtype) {
  Communicator* communicator_ = &Communicator::getInstance();

  const int64_t D = communicator_->size(); // number of devices

  auto model = std::make_unique<DistributedTransformer>(
        D, B, E, H, S, kDropoutProb, kSdpaProb);

  model->setupForward(fusion, dtype, /*sequence_parallel=*/false);
}

at::Tensor transformerShardTensor_Mesh(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh,
    Communicator* communicator_) {
  const auto device_id = communicator_->deviceId();
  return nvfuser::shardTensor(tensor, axis, mesh, device_id);
}

void transformerFwd(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  Communicator* communicator_ = &Communicator::getInstance();
  const int64_t D = communicator_->size(); // number of devices

  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  std::vector<int64_t> norm_shape{E};

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options);
  auto ln0_w_ = at::randn(E, options).to(at::kFloat);
  auto ln0_b_ = at::randn(E, options).to(at::kFloat);
  auto mha_w0_ = at::randn({3 * E, E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;
  auto ln1_w_ = at::randn(E, options).to(at::kFloat);
  auto ln1_b_ = at::randn(E, options).to(at::kFloat);
  auto mlp_w0_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());

  std::vector<c10::IValue> at_inputs = {
      x_,
      ln0_w_,
      ln0_b_,
      transformerShardTensor_Mesh(mha_w0_.view({3, E, E}), 1, mesh, communicator_).view({1, 3 * E / D, E}),
      transformerShardTensor_Mesh(mha_b0_.view({3, E}), 1, mesh, communicator_).view({1, 3 * E / D}),
      transformerShardTensor_Mesh(mha_w1_, 1, mesh, communicator_).unsqueeze(0),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      transformerShardTensor_Mesh(mlp_w0_, 0, mesh, communicator_).unsqueeze(0),
      transformerShardTensor_Mesh(mlp_b0_, 0, mesh, communicator_).unsqueeze(0),
      transformerShardTensor_Mesh(mlp_w1_, 1, mesh, communicator_).unsqueeze(0),
      mlp_b1_};

  auto bytes =
      runBenchmarkIterations(benchmark_state, executor_cache, at_inputs);

  benchmark_state.SetBytesProcessed(
      bytes * int64_t(benchmark_state.iterations()));
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    TransformerForward,
    setupTransformerForward,
    transformerFwd,
    DataType::BFloat16);

NVFUSER_BENCHMARK_RUN(TransformerForward)
    ->Iterations(10)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
