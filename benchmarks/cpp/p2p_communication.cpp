// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <multidevice/ipc_handle.h>
#include <multidevice/cuda_p2p.h>

#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

namespace nvfuser {

void benchmarkP2PCommunication() {
  static constexpr int kTensorSize = 1024;
  static constexpr int kNumRepetitions = 100;
  static constexpr int kWarmupReps = 10;

  // Initialize multidevice environment
  auto communicator = &Communicator::getInstance();
  if (communicator->size() < 2 || torch::cuda::device_count() < 2) {
    std::cout << "Skipping benchmark: need at least 2 GPUs and 2 ranks." << std::endl;
    return;
  }

  const DeviceIdxType my_rank = communicator->deviceId();
  const DeviceIdxType size = communicator->size();
  const DeviceIdxType send_peer = (my_rank + 1) % size;
  const DeviceIdxType recv_peer = (size + my_rank - 1) % size;

  auto container = std::make_unique<hir::HostIrContainer>();
  FusionGuard fg(container.get());

  // Create the P2P communication setup
  auto* send_peer_val = IrBuilder::create<Val>(send_peer, DataType::Int);
  auto* recv_peer_val = IrBuilder::create<Val>(recv_peer, DataType::Int);

  auto* send_tv = makeContigTensor(1);
  auto* recv_tv = makeContigTensor(1);
  container->addInput(send_tv);
  container->addInput(recv_tv);

  auto send = IrBuilder::create<P2PCommunication>(
      P2PCommunicationType::SEND,
      send_tv,
      send_peer_val,
      CommunicatorBackend::kCuda);
  auto recv = IrBuilder::create<P2PCommunication>(
      P2PCommunicationType::RECV,
      recv_tv,
      recv_peer_val,
      CommunicatorBackend::kCuda);
  
  std::vector<P2PCommunication*> grouped_communications = {send, recv};
  auto share_mem_handles = IrBuilder::create<hir::ShareMemHandles>(
      std::move(grouped_communications));
  auto wait_send = IrBuilder::create<hir::Wait>(send);
  auto wait_recv = IrBuilder::create<hir::Wait>(recv);

  container->pushBackTopLevelExprs(share_mem_handles);
  container->pushBackTopLevelExprs(send);
  container->pushBackTopLevelExprs(recv);
  container->pushBackTopLevelExprs(wait_send);
  container->pushBackTopLevelExprs(wait_recv);

  hir::HostIrEvaluator executor(std::move(container), communicator);

  // Create tensors
  at::TensorOptions tensor_options = at::TensorOptions().device(at::kCUDA, my_rank);
  at::Tensor send_tensor = at::empty({kTensorSize}, tensor_options);
  at::Tensor recv_tensor = at::empty({kTensorSize}, tensor_options);

  std::unordered_map<Val*, PolymorphicValue> inputs = {
      {send_tv, send_tensor}, {recv_tv, recv_tensor}};

  // Warmup
  if (my_rank == 0) {
    std::cout << "Running warmup..." << std::endl;
  }
  for (int i = 0; i < kWarmupReps; i++) {
    send_tensor.copy_(at::arange(kTensorSize, tensor_options) + i);
    executor.runWithInput(inputs);
  }

  // Benchmark
  if (my_rank == 0) {
    std::cout << "Starting P2P communication benchmark..." << std::endl;
    std::cout << "Tensor size: " << kTensorSize << " elements" << std::endl;
    std::cout << "Repetitions: " << kNumRepetitions << std::endl;
    std::cout << "Number of devices: " << size << std::endl;
  }

  cudaDeviceSynchronize();
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int rep = 0; rep < kNumRepetitions; rep++) {
    send_tensor.copy_(at::arange(kTensorSize, tensor_options) + rep);
    executor.runWithInput(inputs);
  }

  cudaDeviceSynchronize();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  double avg_time_us = duration.count() / static_cast<double>(kNumRepetitions);
  double bandwidth_gb_s = (kTensorSize * sizeof(float) / (1024.0 * 1024.0 * 1024.0)) / (avg_time_us / 1e6);

  if (my_rank == 0) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average time per communication: " << avg_time_us << " μs" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
  }
}

} // namespace nvfuser

int main(int argc, char* argv[]) {
  nvfuser::benchmarkP2PCommunication();
  return 0;
}