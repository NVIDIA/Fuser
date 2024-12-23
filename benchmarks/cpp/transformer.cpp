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

static std::vector<at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1) {
  auto at_dtype = w0.dtype();
  auto linear0 = at::linear(x, w0, b0);
  auto gelu = at::gelu(linear0.to(at::kFloat), "tanh").to(at_dtype);
  auto linear1 = at::linear(gelu, w1, b1).to(at::kFloat);
  auto [dropout, mask] = at::native_dropout(linear1, kDropoutProb, true);
  return {linear0, gelu, linear1, dropout, mask};
}

static std::vector<at::Tensor> reference_mha(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1) {
  auto linear0 = at::linear(x, w0, b0);
  auto qkv = linear0.view({B, S, 3 * E}).split(E, 2);
  for (auto i = 0; i < 3; i++) {
    qkv[i] = qkv[i].reshape({B, S, H, E / H}).transpose(1, 2);
  }
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv[0], qkv[1], qkv[2], kSdpaProb, true, false, kSdpaScale);
  auto sdpa = std::get<0>(sdpa_out);
  // Reassemble heads (B, H, S, E/H) to (B, S, H, E/H) to (B, S, E)
  auto y = sdpa.transpose(1, 2).reshape({B * S, E});
  auto linear1 = at::linear(y, w1, b1).to(at::kFloat);
  auto [dropout, mask] = at::native_dropout(linear1, kDropoutProb, true);
  return {linear0, sdpa, linear1, dropout, mask};
}

static at::Tensor transformerShardTensor_Mesh(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh,
    Communicator* communicator_) {
  const auto device_id = communicator_->deviceId();
  return nvfuser::shardTensor(tensor, axis, mesh, device_id);
}

static at::Tensor transformerShardTensor(at::Tensor tensor, TensorView* tv, Communicator* communicator_) {
  if (!isSharded(tv)) {
    return tensor;
  }
  NVF_ERROR(tv->hasDeviceMesh(), "`tv` has no DeviceMesh: ", tv);
  return transformerShardTensor_Mesh(
      tensor,
      getShardedLogicalAxis(tv, ParallelType::DIDx),
      tv->getDeviceMesh(), communicator_);
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

  Communicator* communicator_ = &Communicator::getInstance(); // nick TODO call Communicator::getInstance().cleanup() somewhere before program exit
  const int64_t D = communicator_->size(); // number of devices

  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  constexpr float kEps = 1e-5;
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
  auto x_float_ = x_.to(at::kFloat);
  auto ln0_ = at::native_layer_norm(x_float_, norm_shape, ln0_w_, ln0_b_, kEps);
  auto ln0_out_ = std::get<0>(ln0_);

  auto mha_out_ = reference_mha(
      ln0_out_.to(at_dtype), mha_w0_, mha_b0_, mha_w1_, mha_b1_)[3];

  auto resid0_ = mha_out_ + x_float_;
  auto ln1_ = at::native_layer_norm(resid0_, norm_shape, ln1_w_, ln1_b_, kEps);
  auto ln1_out_ = std::get<0>(ln1_);

  auto mlp_out_ = reference_mlp(
      ln1_out_.to(at_dtype), mlp_w0_, mlp_b0_, mlp_w1_, mlp_b1_)[3];
  auto at_out = (resid0_ + mlp_out_).to(at_dtype);

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
    NvFuserScheduler_TransformerFwd,
    DataType::BFloat16);

NVFUSER_BENCHMARK_RUN(TransformerForward)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16, 16}, {128, 128}, {128, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
