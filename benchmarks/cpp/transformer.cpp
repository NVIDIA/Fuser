// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <benchmarks/cpp/utils.h>
#include <fusion.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/multidevice_transformer.h>
#include <tests/cpp/utils.h>
#include <utils.h>

using namespace nvfuser;

constexpr int64_t B = 1, E = 12288, H = 96, S = 2048;
constexpr double kParamScale = 0.02;
constexpr double kDropoutProb = 0.0, kSdpaProb = 0.0;
constexpr int64_t warmup_itrs = 10, num_itrs = 10;

namespace {
at::Tensor shardTensor(
    at::Tensor tensor,
    const int64_t axis,
    const DeviceMesh& mesh,
    Communicator* communicator) {
  const auto device_id = communicator->deviceId();
  return nvfuser::shardTensor(tensor, axis, mesh, device_id);
}
} // namespace

void forward_transformer(
    Communicator* communicator,
    bool profile,
    bool sequence_parallel) {
  int64_t D = communicator->size();
  if (sequence_parallel && D == 1) {
    std::cout << "Sequence parallel requires >1 devices, D=" << D << std::endl;
  }
  auto dtype = DataType::BFloat16;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator->device());

  auto x = at::randn({B * S, E}, options);
  auto ln0_w = at::randn(E, options).to(at::kFloat);
  auto ln0_b = at::randn(E, options).to(at::kFloat);
  auto mha_w0 = at::randn({3 * E, E}, options) * kParamScale;
  auto mha_b0 = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1 = at::randn({E, E}, options) * kParamScale;
  auto mha_b1 = at::randn({E}, options) * kParamScale;
  auto ln1_w = at::randn(E, options).to(at::kFloat);
  auto ln1_b = at::randn(E, options).to(at::kFloat);
  auto mlp_w0 = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0 = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1 = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1 = at::randn({E}, options) * kParamScale;

  KernelArgumentHolder args = {
      sequence_parallel ? shardTensor(x, 0, mesh, communicator).unsqueeze(0)
                        : x,
      ln0_w,
      ln0_b,
      shardTensor(mha_w0.view({3, E, E}), 1, mesh, communicator)
          .view({1, 3 * E / D, E}),
      shardTensor(mha_b0.view({3, E}), 1, mesh, communicator)
          .view({1, 3 * E / D}),
      shardTensor(mha_w1, 1, mesh, communicator).unsqueeze(0),
      mha_b1,
      ln1_w,
      ln1_b,
      shardTensor(mlp_w0, 0, mesh, communicator).unsqueeze(0),
      shardTensor(mlp_b0, 0, mesh, communicator).unsqueeze(0),
      shardTensor(mlp_w1, 1, mesh, communicator).unsqueeze(0),
      mlp_b1};

  DistributedTransformer model(D, B, E, H, S, kDropoutProb, kSdpaProb);
  auto fec = model.forward(dtype, sequence_parallel);

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : arange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      cudaDeviceSynchronize();
      start = std::chrono::high_resolution_clock::now();
      if (profile) {
        cudaProfilerStart();
      }
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("FwdIteration" + std::to_string(i)).c_str());
    }
    auto outputs = fec->runFusionWithInputs(args);

    if (i >= warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  double foward_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      (double)num_itrs;
  std::cout << communicator->deviceId() << ": Average forward time "
            << foward_time << "ms" << std::endl;
}

void backward_transformer(Communicator* communicator, bool profile) {
  auto dtype = DataType::BFloat16;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  int64_t D = communicator->size();
  const auto mesh = DeviceMesh::createForNumDevices(D);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator->device());
  auto x = at::randn({B * S, E}, options);
  auto ln0_w = at::randn(E, options).to(at::kFloat);
  auto ln0_b = at::randn(E, options).to(at::kFloat);
  auto mha_w0 = at::randn({3 * E, E}, options) * kParamScale;
  auto mha_b0 = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1 = at::randn({E, E}, options) * kParamScale;
  auto mha_b1 = at::randn({E}, options) * kParamScale;
  auto ln1_w = at::randn(E, options).to(at::kFloat);
  auto ln1_b = at::randn(E, options).to(at::kFloat);
  auto mlp_w0 = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0 = at::randn({4 * E}, options) * kParamScale;
  auto grad = at::randn({B * S, E}, options) * kParamScale;
  auto mlp_w1 = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1 = at::randn({E}, options) * kParamScale;

  // Recomputed tensors
  auto mlp_dropout_mask = at::rand({B * S, E}, options).lt(1.0 - 0.1);
  auto mha_dropout_mask = at::rand({B * S, E}, options).lt(1.0 - 0.1);
  auto sdpa_output = at::randn({B, H, S, E / H}, options);
  auto sdpa_logsum_exp = at::randn({B, H, S}, options).to(at::kFloat);
  auto [sdpa_seed, sdpa_offset] = createSdpaRngTensors();
  auto ln0_mean = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln0_rstd = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln1_mean = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln1_rstd = at::randn({B * S, 1}, options).to(at::kFloat);
  auto mha_linear1 = at::rand({B * S, E}, options).to(at::kFloat);
  auto mha_linear0 = at::rand({B * S, 3 * E}, options);
  auto mlp_linear1 = at::rand({B * S, 4 * E}, options);

  KernelArgumentHolder args = {
      x,
      grad,
      shardTensor(mha_w0.view({3, E, E}), 1, mesh, communicator)
          .view({1, 3 * E / D, E}),
      shardTensor(mha_w1, 1, mesh, communicator).unsqueeze(0),
      shardTensor(mlp_w0, 0, mesh, communicator).unsqueeze(0),
      shardTensor(mlp_w1, 1, mesh, communicator).unsqueeze(0),
      mlp_dropout_mask,
      mha_dropout_mask,
      shardTensor(sdpa_output, 1, mesh, communicator).unsqueeze(0),
      shardTensor(sdpa_logsum_exp, 1, mesh, communicator).unsqueeze(0),
      sdpa_seed,
      sdpa_offset,
      ln1_w,
      ln1_b,
      ln1_mean,
      ln1_rstd,
      ln0_w,
      ln0_b,
      ln0_mean,
      ln0_rstd,
      shardTensor(mha_linear0, 1, mesh, communicator).unsqueeze(0),
      mha_linear1.to(at::kFloat),
      shardTensor(mlp_linear1, 1, mesh, communicator).unsqueeze(0)};

  DistributedTransformer model(D, B, E, H, S, kDropoutProb, kSdpaProb);
  auto fec = model.backward(dtype);
  KernelArgumentHolder outputs;

  cudaSetDevice(communicator->deviceId());
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : arange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      cudaDeviceSynchronize();
      start = std::chrono::high_resolution_clock::now();
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("BwdIteration" + std::to_string(i)).c_str());
    }
    outputs = fec->runFusionWithInputs(args);

    if (i >= warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  if (profile) {
    cudaProfilerStop();
  }

  double backward_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count() /
      (double)num_itrs;
  std::cout << communicator->deviceId() << ": Average backward time "
            << backward_time << "ms" << std::endl;
}

int main(int argc, char** argv) {
  bool profile = false;
  bool sequence_parallel = false;
  if (argc == 3) {
    profile = (bool)atoi(argv[1]);
    sequence_parallel = (bool)atoi(argv[2]);
  }
  auto communicator = &Communicator::getInstance();
  forward_transformer(communicator, profile, sequence_parallel);
  communicator->barrier();
  backward_transformer(communicator, profile);
}
