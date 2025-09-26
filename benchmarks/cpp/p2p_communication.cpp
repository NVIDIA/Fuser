// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <ir/all_nodes.h>
#include <multidevice/cuda_p2p.h>
#include <multidevice/ipc_handle.h>
#include <ops/all_ops.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace nvfuser {

void benchmarkP2PCommunication() {
  // Test powers of 2 tensor sizes from 2^10 to 2^26
  std::vector<int> tensor_sizes;
  for (int power = 10; power <= 26; power++) {
    tensor_sizes.push_back(1 << power);
  }

  static constexpr int kNumRepetitions = 100;
  static constexpr int kWarmupReps = 10;

  // Initialize multidevice environment
  auto communicator = &Communicator::getInstance();
  if (communicator->size() < 2 || torch::cuda::device_count() < 2) {
    std::cout << "Skipping benchmark: need at least 2 GPUs and 2 ranks."
              << std::endl;
    return;
  }

  const DeviceIdxType my_rank = communicator->deviceId();
  const DeviceIdxType size = communicator->size();
  const DeviceIdxType send_peer = (my_rank + 1) % size;
  const DeviceIdxType recv_peer = (size + my_rank - 1) % size;

  if (my_rank == 0) {
    std::cout << "Starting P2P communication benchmark..." << std::endl;
    std::cout << "Repetitions per size: " << kNumRepetitions << std::endl;
    std::cout << "Number of devices: " << size << std::endl;
    std::cout << std::endl;

    // Table header
    std::cout << std::left << std::setw(15) << "Message Size" << std::setw(12)
              << "Elements" << std::setw(15) << "Latency (μs)" << std::setw(18)
              << "BiBandwidth (GB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
  }

  // Create fusion definition and executor once, outside the loop
  auto container = std::make_unique<hir::HostIrContainer>();
  FusionGuard fg(container.get());

  // Create the P2P communication setup
  auto* send_peer_val = IrBuilder::create<Val>(send_peer, DataType::Int);
  auto* recv_peer_val = IrBuilder::create<Val>(recv_peer, DataType::Int);

  auto* send_tv = TensorViewBuilder()
                      .ndims(1)
                      .dtype(DataType::Float)
                      .contiguity(true)
                      .build();
  auto* recv_tv = TensorViewBuilder()
                      .ndims(1)
                      .dtype(DataType::Float)
                      .contiguity(true)
                      .build();
  ;
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

  // Test each tensor size
  for (size_t size_idx = 0; size_idx < tensor_sizes.size(); size_idx++) {
    const int current_tensor_size = tensor_sizes[size_idx];

    // Create tensors
    at::TensorOptions tensor_options =
        at::TensorOptions().device(at::kCUDA, my_rank);
    at::Tensor send_tensor = at::rand({current_tensor_size}, tensor_options);
    at::Tensor recv_tensor = at::empty({current_tensor_size}, tensor_options);

    std::unordered_map<Val*, PolymorphicValue> inputs = {
        {send_tv, send_tensor}, {recv_tv, recv_tensor}};

    auto dtype_size = send_tensor.element_size();

    // Calculate data transfer size
    double data_size_mb = (current_tensor_size * dtype_size) / 1e6;

    // Warmup
    for (int i = 0; i < kWarmupReps; i++) {
      executor.runWithInput(inputs);
    }

    // Benchmark using CUDA events for accurate GPU timing
    cudaEvent_t start_event, end_event;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&start_event));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventCreate(&end_event));

    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(start_event));

    for (int rep = 0; rep < kNumRepetitions; rep++) {
      executor.runWithInput(inputs);
    }

    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventRecord(end_event));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventSynchronize(end_event));

    float elapsed_time_ms;
    NVFUSER_CUDA_RT_SAFE_CALL(
        cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event));
    double avg_time_us =
        (elapsed_time_ms * 1000.0) / static_cast<double>(kNumRepetitions);

    // Clean up events
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(start_event));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaEventDestroy(end_event));
    double bandwidth_gb_s =
        (2 * current_tensor_size * dtype_size / 1e9) / (avg_time_us / 1e6);

    if (my_rank == 0) {
      // Format message size with units
      std::string size_str;
      if (data_size_mb >= 1.0) {
        size_str = std::to_string(static_cast<int>(data_size_mb)) + " MB";
      } else {
        size_str = std::to_string(static_cast<int>(data_size_mb * 1e3)) + " KB";
      }

      // Print table row
      std::cout << std::left << std::setw(15) << size_str << std::setw(12)
                << current_tensor_size << std::setw(15) << std::fixed
                << std::setprecision(2) << avg_time_us << std::setw(18)
                << std::fixed << std::setprecision(2) << bandwidth_gb_s
                << std::endl;
    }
  }

  if (my_rank == 0) {
    std::cout << std::string(60, '-') << std::endl;
  }
}

} // namespace nvfuser

int main(int argc, char* argv[]) {
  nvfuser::benchmarkP2PCommunication();
  return 0;
}
