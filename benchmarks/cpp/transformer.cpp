// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

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
constexpr int64_t warmup_itrs = 10, num_itrs = 10;

at::Tensor shardTensor(
    at::Tensor tensor,
    int64_t axis,
    const DeviceMesh& mesh,
    Communicator* communicator_) {
  const auto device_id = communicator_->deviceId();
  auto i = mesh.idxOf(device_id);
  auto extent = tensor.size(axis);
  auto nslices = mesh.size();
  NVF_CHECK(
      extent % nslices == 0, "Sharded axis must be evenly divisble by mesh");
  auto stride = extent / nslices;
  i = (i < 0) ? 0 : i;
  auto slice = tensor.slice(axis, i * stride, (i + 1) * stride).contiguous();
  // Temporary until https://github.com/NVIDIA/Fuser/issues/2563. Adds DIDx
  // axis in front representing the sharded extent of the tensor.
  if (stride > 1) {
    slice = slice.unsqueeze(0);
  }
  return slice;
}

void forward_transformer(Communicator* communicator_, bool profile) {
  int64_t D = communicator_->size();
  auto dtype = DataType::BFloat16;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());

  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);
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

  std::vector<c10::IValue> inputs = {
      x_,
      ln0_w_,
      ln0_b_,
      shardTensor(mha_w0_.view({3, E, E}), 1, mesh, communicator_)
          .view({1, 3 * E / D, E}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh, communicator_)
          .view({1, 3 * E / D}),
      shardTensor(mha_w1_, 1, mesh, communicator_),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      shardTensor(mlp_w0_, 0, mesh, communicator_),
      shardTensor(mlp_b0_, 0, mesh, communicator_),
      shardTensor(mlp_w1_, 1, mesh, communicator_),
      mlp_b1_};

  DistributedTransformer model = DistributedTransformer(D, B, E, H, S);
  auto fec = model.forward(dtype);
  cudaSetDevice(communicator_->deviceId());

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : c10::irange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      start = std::chrono::high_resolution_clock::now();
      if (profile) {
        cudaProfilerStart();
      }
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("Iteration" + std::to_string(i)).c_str());
    }
    auto outputs = fec->runFusionWithInputs(inputs);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize is not blocking until kernels are finished on all
    // devices except 0
    // TODO: are we not waiting until all kernels are appended to the stream?
    std::cout << outputs[0][0][0] << std::endl;

    if (i > warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  double foward_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      (double)num_itrs / 1000.0;
  std::cout << communicator_->deviceId() << ": Average forward time "
            << foward_time << "ms" << std::endl;
}

void backward_transformer(Communicator* communicator_, bool profile) {
  auto dtype = DataType::BFloat16;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  int64_t D = communicator_->size();
  const auto mesh = DeviceMesh::createForNumDevices(D);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);
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
  auto grad_ = at::randn({B * S, E}, options).to(at::kFloat) * kParamScale;
  auto mlp_w1_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  // Recomputed tensors
  auto mlp_dropout_mask = at::rand({B * S, E}, options).lt(1.0 - 0.1);
  auto mha_dropout_mask = at::rand({B * S, E}, options).lt(1.0 - 0.1);
  auto sdpa_output = at::randn({B, H, S, E / H}, options);
  auto sdpa_logsum_exp = at::randn({B, H, S}, options).to(at::kFloat);
  auto sdpa_seed = at::scalar_tensor(1, at::kLong);
  auto sdpa_offset = at::scalar_tensor(1, at::kLong);
  auto ln0_mean = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln0_rstd = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln1_mean = at::randn({B * S, 1}, options).to(at::kFloat);
  auto ln1_rstd = at::randn({B * S, 1}, options).to(at::kFloat);
  auto mha_linear1 = at::rand({B * S, E}, options).to(at::kFloat);

  std::vector<c10::IValue> inputs = {
      x_,
      grad_,
      shardTensor(mha_w0_.view({3, E, E}), 1, mesh, communicator_)
          .view({1, 3 * E / D, E}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh, communicator_)
          .view({1, 3 * E / D}),
      shardTensor(mha_w1_, 1, mesh, communicator_),
      shardTensor(mlp_w0_, 0, mesh, communicator_),
      shardTensor(mlp_b0_, 0, mesh, communicator_),
      shardTensor(mlp_w1_, 1, mesh, communicator_),
      mlp_b1_,
      mlp_dropout_mask,
      mha_dropout_mask,
      shardTensor(sdpa_output, 1, mesh, communicator_),
      shardTensor(sdpa_logsum_exp, 1, mesh, communicator_),
      sdpa_seed,
      sdpa_offset,
      ln1_w_,
      ln1_b_,
      ln1_mean,
      ln1_rstd,
      ln0_w_,
      ln0_b_,
      ln0_mean,
      ln0_rstd,
      mha_linear1};

  DistributedTransformer model = DistributedTransformer(D, B, E, H, S);
  auto fec = model.backward(dtype);
  std::vector<at::Tensor> outputs;

  cudaSetDevice(communicator_->deviceId());
  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : c10::irange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      start = std::chrono::high_resolution_clock::now();
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("Iteration" + std::to_string(i)).c_str());
    }
    outputs = fec->runFusionWithInputs(inputs);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize is not blocking until kernels are finished on all
    // devices except 0
    // TODO: are we not waiting until all kernels are appended to the stream?
    std::cout << outputs[0][0][0][0] << std::endl;

    if (i > warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (profile) {
    cudaProfilerStop();
  }

  double backward_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      (double)num_itrs / 1000.0;
  std::cout << communicator_->deviceId() << ": Average backward time "
            << backward_time << "ms" << std::endl;
}

int main(int argc, char** argv) {
  // using this is as a flag for when to profile
  bool profile = argc > 1;
  auto communicator_ = &Communicator::getInstance();
  forward_transformer(communicator_, profile);
  communicator_->barrier();
  backward_transformer(communicator_, profile);
}
