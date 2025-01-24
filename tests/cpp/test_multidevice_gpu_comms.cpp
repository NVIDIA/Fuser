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

  // Write the value 3 to the cuda buffer
  const int64_t value = rank;
  CUDA_CALL(cudaMemcpy(d_ptr, &value, sizeof(int64_t), cudaMemcpyHostToDevice));

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

} // namespace nvfuser
