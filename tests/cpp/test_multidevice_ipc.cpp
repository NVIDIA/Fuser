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

using IpcTest = MultiDeviceTest;

TEST_F(IpcTest, IpcMemHandle) {
  // Allocate and setup GPU buffers
  constexpr size_t kBufferSize = sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  void* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));
  const int64_t value = rank;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, &value, kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));
  // As a convenience, we use the TCP store to exchange out-of-band the IPC
  // handle as raw data
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string((rank + 1) % num_devices)));
  void* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      &peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(&peer_value, peer_d_ptr, kBufferSize, cudaMemcpyDeviceToHost));
  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

TEST_F(IpcTest, IpcMemHandlePtrArithmeticAtReceiver) {
  // TL;DR: We can do pointer arithmetic on the importer side. IOW, the pointer
  // can be used as a regular pointer on the importer side.

  // Allocate GPU memory. Set up a buffer with two int values.
  constexpr size_t kBufferSize = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;
  void* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));

  // Set up the buffer
  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, values.data(), kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));
  int64_t* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate, by reading the second value in the buffer (c.f. the "+1" offset)
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      &peer_value, peer_d_ptr + 1, kBufferSize / 2, cudaMemcpyDeviceToHost));
  EXPECT_EQ(2 * peer_rank + 1, peer_value);

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

TEST_F(IpcTest, IpcMemHandlePtrArithmeticAtSender) {
  // TL;DR: We CANNOT do pointer arithmetic on the exporter side! The IPC handle
  // points to the beginning of the allocated buffer.

  // Allocate GPU memory. Set up a buffer with two int values.
  constexpr size_t kBufferSize = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;
  int64_t* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));

  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, values.data(), kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr + 1));
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));
  int64_t* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate, noticing that the pointer is not offset by 1, contrarily to the
  // offset used in the exporter side.
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      &peer_value, peer_d_ptr, kBufferSize / 2, cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      2 * peer_rank,
      peer_value); // and not 2 * peer_rank + 1 as could be expected!

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

using StreamOpTest = NVFuserTest;

TEST_F(StreamOpTest, StreamWriteValue32) {
  cudaStream_t stream;
  void* buf;
  int value = 0;
  constexpr int new_value = 42;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(0));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamCreate(&stream));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&buf, sizeof(int)));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      buf, &value, sizeof(int), cudaMemcpyHostToDevice, stream));
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream, (CUdeviceptr)buf, new_value, CU_STREAM_WRITE_VALUE_DEFAULT));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      &value, buf, sizeof(int), cudaMemcpyDeviceToHost, stream));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaStreamSynchronize(stream));
  EXPECT_EQ(value, new_value);
}

} // namespace nvfuser
