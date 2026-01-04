// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cuda_profiler_api.h>

#include <chrono>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cuda_utils.h>
#include <driver_api.h>
#include <multidevice/execution_utils.h>
#include <multidevice/ipc_utils.h>
#include <ops/all_ops.h>
#include <optimization_pass.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <runtime/communication_executor.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {

enum class CommunicationProtocol { kNccl, kMemcpy, kMultimem, kBatchedMemcpy };

// Helper function to get CommunicatorBackend from CommunicationProtocol
CommunicatorBackend getBackend(CommunicationProtocol protocol) {
  switch (protocol) {
    case CommunicationProtocol::kNccl:
      return CommunicatorBackend::kNccl;
    case CommunicationProtocol::kMemcpy:
    case CommunicationProtocol::kMultimem:
    case CommunicationProtocol::kBatchedMemcpy:
      return CommunicatorBackend::kCuda;
  }
  std::unreachable();
}

// Helper function to get protocol string for MulticastProtocol option
std::string getProtocolString(CommunicationProtocol protocol) {
  switch (protocol) {
    case CommunicationProtocol::kNccl:
      return "nccl";
    case CommunicationProtocol::kMemcpy:
      return "memcpy";
    case CommunicationProtocol::kMultimem:
      return "multimem";
    case CommunicationProtocol::kBatchedMemcpy:
      return "batch_memcpy";
  }
  std::unreachable();
}

} // namespace

class LowerCollectiveCudaAndNcclTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<
          std::tuple<int64_t, CommunicationProtocol>> {
 protected:
  bool isMulticastSupported() {
    const int64_t local_rank = communicator_->local_rank();
    int is_multicast_supported = 0;
    NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
        &is_multicast_supported,
        CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        static_cast<int>(local_rank)));
    return is_multicast_supported != 0;
  }

  // Run complete benchmark: warmup, timing, reduce results, and return output
  at::Tensor runBenchmark(
      MultiDeviceExecutor& executor,
      const std::vector<c10::IValue>& inputs,
      int64_t msg_size_bytes,
      CommunicatorBackend backend_type,
      const std::string& test_name,
      float bandwidth_multiplier = 1.0f,
      int warmup_iters = 50,
      int profiling_iters = 50,
      int timing_iters = 500) {
    // Warm-up iterations
    at::Tensor out_tensor;
    for (int i = 0; i < warmup_iters; ++i) {
      out_tensor = executor.runWithInput(inputs)[0].as<at::Tensor>();
    }

    communicator_->barrier();
    cudaDeviceSynchronize();

    float cpu_elapsed_ms_sum = 0.0f;
    cudaProfilerStart();
    for (int i = 0; i < timing_iters; ++i) {
      communicator_->barrier();
      cudaDeviceSynchronize();
      auto cpu_start = std::chrono::high_resolution_clock::now();
      out_tensor = executor.runWithInput(inputs)[0].as<at::Tensor>();
      cudaDeviceSynchronize();
      auto cpu_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> cpu_duration =
          cpu_end - cpu_start;
      cpu_elapsed_ms_sum += cpu_duration.count();
    }
    cudaProfilerStop();

    float avg_cpu_time_ms = cpu_elapsed_ms_sum / timing_iters;

    // Reduce mean time across all ranks
    at::Tensor time_tensor = at::tensor(
        {avg_cpu_time_ms},
        at::TensorOptions().dtype(at::kFloat).device(communicator_->device()));
    std::vector<at::Tensor> time_tensors = {time_tensor};

    communicator_->getWorld(CommunicatorBackend::kNccl)
        ->allreduce(time_tensors, {c10d::ReduceOp::MAX})
        ->wait();

    // Print results on rank 0
    if (communicator_->deviceId() == 0) {
      float mean_cpu_time_ms = time_tensor.item<float>();
      float cpu_bandwidth_gbps = (msg_size_bytes * bandwidth_multiplier /
                                  (mean_cpu_time_ms / 1000.0)) /
          1e9;
      std::cout << test_name << " - Backend: " << backend_type
                << ", Size: " << (msg_size_bytes / (1024.0 * 1024.0)) << " MB"
                << ", Avg CPU time: " << mean_cpu_time_ms << " ms"
                << ", CPU Bandwidth: " << cpu_bandwidth_gbps << " GB/s"
                << std::endl;
    }

    return out_tensor;
  }

  // Setup protocol options based on CommunicationProtocol enum
  // The guard must be created by the caller and kept alive for the test
  // duration
  void setupProtocolOptions(
      CommunicationProtocol protocol_enum,
      EnableOptionsGuard& guard) {
    // Set MulticastProtocol option only for CUDA backend protocols
    switch (protocol_enum) {
      case CommunicationProtocol::kMultimem: {
        cudaDeviceProp prop;
        NVFUSER_CUDA_RT_SAFE_CALL(
            cudaGetDeviceProperties(&prop, communicator_->device().index()));
        if (prop.major < 9) {
          GTEST_SKIP() << "Multicast protocol 'multimem' requires Compute "
                          "Capability >= 9.0";
        }
        EnableOptionsGuard::getCurOptions().set(
            EnableOption::MulticastProtocol, {"multimem"});
        break;
      }
      case CommunicationProtocol::kBatchedMemcpy:
        EnableOptionsGuard::getCurOptions().set(
            EnableOption::MulticastProtocol, {"batch_memcpy"});
        break;
      case CommunicationProtocol::kMemcpy:
        // Explicitly set for memcpy
        EnableOptionsGuard::getCurOptions().set(
            EnableOption::MulticastProtocol, {"memcpy"});
        break;
      case CommunicationProtocol::kNccl:
        // For nccl backend, MulticastProtocol is irrelevant and should not be
        // set
        break;
    }
  }
};

TEST_P(LowerCollectiveCudaAndNcclTest, Allgather) {
  const auto& [msg_size_bytes, protocol_enum] = GetParam();
  const int64_t kMsgSize = msg_size_bytes / sizeof(float);
  const CommunicatorBackend backend_type = getBackend(protocol_enum);
  const std::string protocol_str = getProtocolString(protocol_enum);

  if (!communicator_->is_available() || communicator_->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks.";
  }

  if (!isMulticastSupported() &&
      (protocol_enum == CommunicationProtocol::kMemcpy ||
       protocol_enum == CommunicationProtocol::kMultimem)) {
    GTEST_SKIP() << "Device does not support Multicast; skipping.";
  }

  // cudaMemcpyBatchAsync requires a non-default stream
  c10::cuda::CUDAStream stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(stream);

  EnableOptionsGuard guard;
  setupProtocolOptions(protocol_enum, guard);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  if (backend_type == CommunicatorBackend::kCuda) {
    out->setMemoryType(MemoryType::Symmetric);
  }

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  in->setDeviceMesh(mesh);
  out->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);

  at::Tensor unsharded_tensor =
      at::randn({num_devices, kMsgSize}, tensor_options_);
  at::Tensor in_tensor = shardTensor(unsharded_tensor, in);

  MultiDeviceExecutorParams params;
  params.lower.communicator_backend = backend_type;
  params.executor.use_allocation_cache = true;
  MultiDeviceExecutor executor(
      std::move(fusion), Communicator::getInstance(), params);

  // Run benchmark and validate correctness
  at::Tensor out_tensor = runBenchmark(
      executor,
      {in_tensor},
      msg_size_bytes,
      backend_type,
      "Allgather/" + protocol_str,
      static_cast<float>(communicator_->size()));

  EXPECT_TRUE(at::allclose(out_tensor, unsharded_tensor));
}

TEST_P(LowerCollectiveCudaAndNcclTest, Broadcast) {
  const auto& [msg_size_bytes, protocol_enum] = GetParam();
  const CommunicatorBackend backend_type = getBackend(protocol_enum);
  const std::string protocol_str = getProtocolString(protocol_enum);
  const int64_t kMsgSize = msg_size_bytes / sizeof(float);

  if (!communicator_->is_available() || communicator_->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks.";
  }

  if (!isMulticastSupported() &&
      (protocol_enum == CommunicationProtocol::kMemcpy ||
       protocol_enum == CommunicationProtocol::kMultimem)) {
    GTEST_SKIP() << "Device does not support Multicast; skipping.";
  }

  // cudaMemcpyBatchAsync requires a non-default stream
  c10::cuda::CUDAStream stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(stream);

  EnableOptionsGuard guard;
  setupProtocolOptions(protocol_enum, guard);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto num_devices = communicator_->size();
  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);
  fusion->addInput(in);
  fusion->addOutput(out);

  if (backend_type == CommunicatorBackend::kCuda) {
    out->setMemoryType(MemoryType::Symmetric);
  }

  auto mesh = DeviceMesh::createForNumDevices(num_devices);
  constexpr DeviceIdxType kRoot = 0;
  in->setDeviceMesh({kRoot});
  out->setDeviceMesh(mesh);

  MultiDeviceExecutorParams params;
  params.lower.communicator_backend = backend_type;
  params.executor.use_allocation_cache = true;
  MultiDeviceExecutor executor(
      std::move(fusion), Communicator::getInstance(), params);

  at::Tensor unsharded_tensor =
      at::randn({num_devices, kMsgSize}, tensor_options_);
  const auto device_id = communicator_->deviceId();
  at::Tensor in_tensor = unsharded_tensor.slice(0, device_id, device_id + 1);

  // Run benchmark and validate correctness
  at::Tensor out_tensor = runBenchmark(
      executor,
      {in_tensor},
      msg_size_bytes,
      backend_type,
      "Broadcast/" + protocol_str,
      1.0f);

  EXPECT_TRUE(
      at::allclose(out_tensor, unsharded_tensor.slice(0, kRoot, kRoot + 1)));
}

namespace {
std::string paramToStringLowerCollectiveCudaAndNcclTest(
    const testing::TestParamInfo<std::tuple<int64_t, CommunicationProtocol>>&
        info) {
  const auto& [msg_size_bytes, protocol_enum] = info.param;
  std::stringstream ss;
  ss << getProtocolString(protocol_enum) << "_";
  int64_t size_mb = msg_size_bytes / (1024 * 1024);
  if (size_mb >= 1024) {
    ss << (size_mb / 1024) << "GB";
  } else {
    ss << size_mb << "MB";
  }
  return ss.str();
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    LowerCollectiveCudaAndNcclTest,
    testing::Combine(
        testing::Values(
            2 * 1024 * 1024LL, // 2 MB
            8 * 1024 * 1024LL, // 8 MB
            32 * 1024 * 1024LL, // 32 MB
            128 * 1024 * 1024LL, // 128 MB
            256 * 1024 * 1024LL // 256 MB
            ),
        testing::Values(
            CommunicationProtocol::kMemcpy,
            CommunicationProtocol::kNccl,
            CommunicationProtocol::kMultimem,
            CommunicationProtocol::kBatchedMemcpy)),
    paramToStringLowerCollectiveCudaAndNcclTest);

} // namespace nvfuser
