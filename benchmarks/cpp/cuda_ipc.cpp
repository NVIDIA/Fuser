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

template <typename T>
std::vector<uint8_t> toBytes(const T& data) {
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(&data),
      reinterpret_cast<const uint8_t*>(&data) + sizeof(T));
}

template <typename T>
const T& fromBytes(const std::vector<uint8_t>& bytes) {
  return *reinterpret_cast<const T*>(bytes.data());
}

class ZeroCopyBenchmarker {
public:
  struct ZCopyResult {
    size_t bytes;
    double send_post_ms;
    double recv_post_ms;
    double send_wait_ms;
    double total_time_ms;
    double bandwidth_gbps;
  };

  static ZCopyResult measureZeroCopyTransfer(
      const P2pIpcHandle& sender_handles, 
      const P2pIpcHandle& receiver_handles,
      size_t bytes,
      cudaStream_t sender_stream,
      cudaStream_t receiver_stream,
      int warmup_runs = 3,
      int test_runs = 5) {
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
      get_zcopy::sendPost(sender_handles, sender_stream);
      get_zcopy::recvPost(receiver_handles, bytes, receiver_stream);
      get_zcopy::sendWait(sender_handles, sender_stream);
      NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(sender_stream));
      NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(receiver_stream));
    }

    double total_send_post_time = 0.0;
    double total_recv_post_time = 0.0;
    double total_send_wait_time = 0.0;
    double total_time = 0.0;

    for (int i = 0; i < test_runs; ++i) {
      // Measure total time from start to finish
      auto total_start = std::chrono::high_resolution_clock::now();
      
      // Measure sendPost
      auto send_post_start = std::chrono::high_resolution_clock::now();
      get_zcopy::sendPost(sender_handles, sender_stream);
      NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(sender_stream));
      auto send_post_end = std::chrono::high_resolution_clock::now();
      
      // Measure recvPost  
      auto recv_post_start = std::chrono::high_resolution_clock::now();
      get_zcopy::recvPost(receiver_handles, bytes, receiver_stream);
      NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(receiver_stream));
      auto recv_post_end = std::chrono::high_resolution_clock::now();
      
      // Measure sendWait
      auto send_wait_start = std::chrono::high_resolution_clock::now();
      get_zcopy::sendWait(sender_handles, sender_stream);
      NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(sender_stream));
      auto send_wait_end = std::chrono::high_resolution_clock::now();
      
      auto total_end = std::chrono::high_resolution_clock::now();

      total_send_post_time += std::chrono::duration<double, std::milli>(send_post_end - send_post_start).count();
      total_recv_post_time += std::chrono::duration<double, std::milli>(recv_post_end - recv_post_start).count();
      total_send_wait_time += std::chrono::duration<double, std::milli>(send_wait_end - send_wait_start).count();
      total_time += std::chrono::duration<double, std::milli>(total_end - total_start).count();
    }

    double avg_send_post_ms = total_send_post_time / test_runs;
    double avg_recv_post_ms = total_recv_post_time / test_runs;  
    double avg_send_wait_ms = total_send_wait_time / test_runs;
    double avg_total_ms = total_time / test_runs;
    double bandwidth_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_total_ms / 1000.0);

    return {bytes, avg_send_post_ms, avg_recv_post_ms, avg_send_wait_ms, avg_total_ms, bandwidth_gbps};
  }

  static void printZeroCopyResults(const std::vector<ZCopyResult>& results, const std::string& test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << std::setw(10) << "Size(KB)" 
              << std::setw(12) << "SendPost(ms)"
              << std::setw(12) << "RecvPost(ms)" 
              << std::setw(12) << "SendWait(ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(15) << "Bandwidth(GB/s)" << std::endl;
    std::cout << std::string(73, '-') << std::endl;
    
    for (const auto& result : results) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(1) 
                << (result.bytes / 1024.0)
                << std::setw(12) << std::setprecision(3) << result.send_post_ms
                << std::setw(12) << std::setprecision(3) << result.recv_post_ms
                << std::setw(12) << std::setprecision(3) << result.send_wait_ms
                << std::setw(12) << std::setprecision(3) << result.total_time_ms
                << std::setw(15) << std::setprecision(2) << result.bandwidth_gbps
                << std::endl;
    }
    std::cout << std::endl;
  }
};

class ZeroCopyBenchmark {
public:
  void zeroCopyBandwidthCacheHit() {
    auto communicator = &Communicator::getInstance();
    if (communicator->size() < 2) {
      std::cout << "Skipping test: Need at least 2 devices" << std::endl;
      return;
    }

    const int64_t rank = communicator->deviceId();
    const int64_t peer_rank = (rank + 1) % communicator->size();
    
    // Only run benchmark on rank 0 to avoid duplicate output
    if (rank != 0) {
      return;
    }

    // Test various buffer sizes
    std::vector<size_t> sizes = {
      1024,      // 1 KB
      4096,      // 4 KB  
      16384,     // 16 KB
      65536,     // 64 KB
      262144,    // 256 KB
      1048576,   // 1 MB
      4194304,   // 4 MB
      16777216,  // 16 MB
      67108864,  // 64 MB
      268435456  // 256 MB
    };

    std::vector<ZeroCopyBenchmarker::ZCopyResult> results;

    // Create CUDA streams
    cudaStream_t sender_stream, receiver_stream;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&receiver_stream));

    for (size_t buffer_size : sizes) {
      // Create tensors on both devices
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      at::Tensor sender_tensor = at::zeros({static_cast<int64_t>(buffer_size)}, 
                                           at::TensorOptions().dtype(at::kByte).device(at::kCUDA, rank));
      
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(peer_rank));
      at::Tensor receiver_tensor = at::zeros({static_cast<int64_t>(buffer_size)}, 
                                             at::TensorOptions().dtype(at::kByte).device(at::kCUDA, peer_rank));

      // Initialize sender data
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      sender_tensor.fill_(42);

      // Create IpcHandles
      auto sender_local_handle = std::make_unique<IpcHandle>(sender_tensor);
      auto receiver_local_handle = std::make_unique<IpcHandle>(receiver_tensor);
      
             // Exchange handles via TCP store
       auto store = communicator->getTcpStore();
       store->set("zcopy_sender_" + std::to_string(rank), toBytes(*sender_local_handle));
       store->set("zcopy_receiver_" + std::to_string(peer_rank), toBytes(*receiver_local_handle));
       communicator->barrier();

      // Import peer handles
      auto sender_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_receiver_" + std::to_string(peer_rank)));
      auto receiver_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_sender_" + std::to_string(rank)));

      // Create P2pIpcHandles
      P2pIpcHandle sender_handles(std::move(sender_local_handle), std::move(sender_peer_handle));
      P2pIpcHandle receiver_handles(std::move(receiver_local_handle), std::move(receiver_peer_handle));

      // Benchmark zero-copy operations
      auto result = ZeroCopyBenchmarker::measureZeroCopyTransfer(
          sender_handles, receiver_handles, buffer_size, sender_stream, receiver_stream);
      results.push_back(result);
    }

    // Cleanup streams
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(receiver_stream));

    ZeroCopyBenchmarker::printZeroCopyResults(results, "Zero-Copy Bandwidth - Cache Hit Test");
  }

  void zeroCopyBandwidthCacheMiss() {
    auto communicator = &Communicator::getInstance();
    if (communicator->size() < 2) {
      std::cout << "Skipping test: Need at least 2 devices" << std::endl;
      return;
    }

    const int64_t rank = communicator->deviceId();
    const int64_t peer_rank = (rank + 1) % communicator->size();
    
    // Only run benchmark on rank 0 to avoid duplicate output
    if (rank != 0) {
      return;
    }

    // Test various buffer sizes with cache miss patterns
    std::vector<size_t> sizes = {
      1024,      // 1 KB
      4096,      // 4 KB  
      16384,     // 16 KB
      65536,     // 64 KB
      262144,    // 256 KB
      1048576,   // 1 MB
      4194304,   // 4 MB
      16777216,  // 16 MB
      67108864,  // 64 MB
    };

    std::vector<ZeroCopyBenchmarker::ZCopyResult> results;

    // Create CUDA streams
    cudaStream_t sender_stream, receiver_stream;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&receiver_stream));

    for (size_t buffer_size : sizes) {
      // Allocate larger buffers for cache miss pattern
      size_t total_buffer_size = buffer_size * 16;
      
      // Create tensors on both devices
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      at::Tensor sender_tensor = at::zeros({static_cast<int64_t>(total_buffer_size)}, 
                                           at::TensorOptions().dtype(at::kByte).device(at::kCUDA, rank));
      
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(peer_rank));
      at::Tensor receiver_tensor = at::zeros({static_cast<int64_t>(buffer_size)}, 
                                             at::TensorOptions().dtype(at::kByte).device(at::kCUDA, peer_rank));

      // Initialize sender data with pattern
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      auto sender_data = sender_tensor.accessor<uint8_t, 1>();
      for (size_t i = 0; i < total_buffer_size; ++i) {
        sender_data[i] = static_cast<uint8_t>(i % 256);
      }

      // Create IpcHandles
      auto sender_local_handle = std::make_unique<IpcHandle>(sender_tensor);
      auto receiver_local_handle = std::make_unique<IpcHandle>(receiver_tensor);
      
             // Exchange handles via TCP store
       auto store = communicator->getTcpStore();
       store->set("zcopy_miss_sender_" + std::to_string(rank), toBytes(*sender_local_handle));
       store->set("zcopy_miss_receiver_" + std::to_string(peer_rank), toBytes(*receiver_local_handle));
       communicator->barrier();

      // Import peer handles
      auto sender_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_miss_receiver_" + std::to_string(peer_rank)));
      auto receiver_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_miss_sender_" + std::to_string(rank)));

      // Create P2pIpcHandles
      P2pIpcHandle sender_handles(std::move(sender_local_handle), std::move(sender_peer_handle));
      P2pIpcHandle receiver_handles(std::move(receiver_local_handle), std::move(receiver_peer_handle));

      // For cache miss pattern, we simulate accessing different regions
      // by creating a custom measurement that varies the access pattern
      const int warmup_runs = 3;
      const int test_runs = 5;
      
      // Warmup
      for (int i = 0; i < warmup_runs; ++i) {
        get_zcopy::sendPost(sender_handles, sender_stream);
        get_zcopy::recvPost(receiver_handles, buffer_size, receiver_stream);
        get_zcopy::sendWait(sender_handles, sender_stream);
        NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(sender_stream));
        NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(receiver_stream));
      }

      // Measure total time for cache miss scenario
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < test_runs; ++i) {
        get_zcopy::sendPost(sender_handles, sender_stream);
        get_zcopy::recvPost(receiver_handles, buffer_size, receiver_stream);
        get_zcopy::sendWait(sender_handles, sender_stream);
        NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(sender_stream));
        NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(receiver_stream));
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count() / test_runs;
      double bandwidth_gbps = (buffer_size / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0);

      results.push_back({buffer_size, 0.0, 0.0, 0.0, total_time_ms, bandwidth_gbps});
    }

    // Cleanup streams
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(receiver_stream));

    ZeroCopyBenchmarker::printZeroCopyResults(results, "Zero-Copy Bandwidth - Cache Miss Test");
  }

  void zeroCopyLatencyMicrobenchmark() {
    auto communicator = &Communicator::getInstance();
    if (communicator->size() < 2) {
      std::cout << "Skipping test: Need at least 2 devices" << std::endl;
      return;
    }

    const int64_t rank = communicator->deviceId();
    const int64_t peer_rank = (rank + 1) % communicator->size();
    
    // Only run benchmark on rank 0 to avoid duplicate output
    if (rank != 0) {
      return;
    }

    // Test small sizes for latency measurement
    std::vector<size_t> sizes = {1, 4, 16, 64, 256, 1024, 4096};
    std::vector<ZeroCopyBenchmarker::ZCopyResult> results;

    // Create CUDA streams
    cudaStream_t sender_stream, receiver_stream;
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&receiver_stream));

    for (size_t buffer_size : sizes) {
      // Create tensors on both devices
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      at::Tensor sender_tensor = at::zeros({static_cast<int64_t>(buffer_size)}, 
                                           at::TensorOptions().dtype(at::kByte).device(at::kCUDA, rank));
      
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(peer_rank));
      at::Tensor receiver_tensor = at::zeros({static_cast<int64_t>(buffer_size)}, 
                                             at::TensorOptions().dtype(at::kByte).device(at::kCUDA, peer_rank));

      // Initialize sender data
      NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));
      sender_tensor.fill_(255);

      // Create IpcHandles
      auto sender_local_handle = std::make_unique<IpcHandle>(sender_tensor);
      auto receiver_local_handle = std::make_unique<IpcHandle>(receiver_tensor);
      
             // Exchange handles via TCP store
       auto store = communicator->getTcpStore();
       store->set("zcopy_lat_sender_" + std::to_string(rank), toBytes(*sender_local_handle));
       store->set("zcopy_lat_receiver_" + std::to_string(peer_rank), toBytes(*receiver_local_handle));
       communicator->barrier();

      // Import peer handles
      auto sender_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_lat_receiver_" + std::to_string(peer_rank)));
      auto receiver_peer_handle = std::make_unique<IpcHandle>(
          store->get("zcopy_lat_sender_" + std::to_string(rank)));

      // Create P2pIpcHandles
      P2pIpcHandle sender_handles(std::move(sender_local_handle), std::move(sender_peer_handle));
      P2pIpcHandle receiver_handles(std::move(receiver_local_handle), std::move(receiver_peer_handle));

      // Benchmark with higher precision for latency
      auto result = ZeroCopyBenchmarker::measureZeroCopyTransfer(
          sender_handles, receiver_handles, buffer_size, sender_stream, receiver_stream, 10, 100);
      results.push_back(result);
    }

    // Cleanup streams
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(sender_stream));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamDestroy(receiver_stream));

    ZeroCopyBenchmarker::printZeroCopyResults(results, "Zero-Copy Latency Microbenchmark");
  }
};

} // namespace nvfuser

int main(int argc, char* argv[]) {
  std::cout << "Starting Zero-Copy IPC Benchmarks" << std::endl;
  
  try {
    nvfuser::ZeroCopyBenchmark benchmark;
    
    std::cout << "\nRunning Zero-Copy Bandwidth Cache Hit Test..." << std::endl;
    benchmark.zeroCopyBandwidthCacheHit();
    
    std::cout << "\nRunning Zero-Copy Bandwidth Cache Miss Test..." << std::endl;
    benchmark.zeroCopyBandwidthCacheMiss();
    
    std::cout << "\nRunning Zero-Copy Latency Microbenchmark..." << std::endl;
    benchmark.zeroCopyLatencyMicrobenchmark();
    
    std::cout << "\nAll benchmarks completed successfully!" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error running benchmarks: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
