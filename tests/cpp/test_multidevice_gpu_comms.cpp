// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda_profiler_api.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/executor.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/multidevice_kernels.h>

namespace nvfuser {

namespace {

#define CUDA_CALL(call) ASSERT_EQ((call), cudaSuccess)

template <typename T>
std::vector<uint8_t> toBytes(T data) {
  return std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(&data),
      reinterpret_cast<uint8_t*>(&data) + sizeof(T));
}

template <typename T>
T fromBytes(std::vector<uint8_t> bytes) {
  return *reinterpret_cast<T*>(bytes.data());
}

} // namespace

class GpuCommTest : public MultiDeviceTest {};

TEST_F(GpuCommTest, IpcMemHandle) {
  // Allocate GPU memory
  constexpr size_t size = sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  void* d_ptr;
  CUDA_CALL(cudaMalloc(&d_ptr, size));

  const int64_t value = rank;
  CUDA_CALL(cudaMemcpy(d_ptr, &value, size, cudaMemcpyHostToDevice));

  cudaIpcMemHandle_t ipc_handle;
  CUDA_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));

  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));
  communicator_->barrier();
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(store->get("ipc_handle_" + std::to_string((rank + 1) % num_devices)));

  void* peer_d_ptr;
  CUDA_CALL(cudaIpcOpenMemHandle(&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  int64_t peer_value;
  CUDA_CALL(cudaMemcpy(&peer_value, peer_d_ptr, size, cudaMemcpyDeviceToHost));

  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up
  CUDA_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  CUDA_CALL(cudaFree(d_ptr));

}

TEST_F(GpuCommTest, DeviceEnablePeerAccess) {
  // Doesn't seem to work when the PID are differents, i.e., when it's one CPU rank per GPU. The line "udaMemcpy(d_ptr, peer_d_ptr, size, cudaMemcpyDeviceToDevice)" throws.
  // https://github.com/NVIDIA/nccl/blob/1672c85781ba6158d5d173d3ecac969f8796af11/src/transport/p2p.cc#L324-328
  // https://github.com/NVIDIA/nccl/blob/1672c85781ba6158d5d173d3ecac969f8796af11/src/transport/p2p.cc#L249
  GTEST_SKIP();

  // Allocate GPU memory
  constexpr size_t size = sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer = (rank + 1) % num_devices;
  // const int64_t accessing_peer = (num_devices + rank - 1) % num_devices;

  int can_access_peer;
  CUDA_CALL(cudaDeviceCanAccessPeer (&can_access_peer, rank, peer));
  if (!can_access_peer) {
    GTEST_SKIP() << "Peer access not enabled between devices " << rank << " and " << peer;
  }

  CUDA_CALL(cudaDeviceEnablePeerAccess(peer, /*flag (reserved)*/0));

  void* d_ptr;
  CUDA_CALL(cudaMalloc(&d_ptr, size));

  const int64_t value = rank;
  CUDA_CALL(cudaMemcpy(d_ptr, &value, size, cudaMemcpyHostToDevice));


  auto store = communicator_->getTcpStore();
  store->set("d_ptr_" + std::to_string(rank), toBytes(d_ptr));
  communicator_->barrier();
  auto peer_d_ptr = fromBytes<void*>(store->get("d_ptr_" + std::to_string(peer)));

  CUDA_CALL(cudaMemcpy(d_ptr, peer_d_ptr, size, cudaMemcpyDeviceToDevice));
  int64_t peer_value;
  CUDA_CALL(cudaMemcpy(&peer_value, d_ptr, size, cudaMemcpyDeviceToHost));

  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up
  CUDA_CALL(cudaDeviceDisablePeerAccess(peer)); // not necessary
  CUDA_CALL(cudaFree(d_ptr));
}

} // namespace nvfuser
