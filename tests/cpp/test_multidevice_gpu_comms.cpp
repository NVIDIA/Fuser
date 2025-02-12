// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
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
#include <tests/cpp/multidevice_kernels.h>

namespace nvfuser {

#define CUDA_CALL(call) ASSERT_EQ((call), cudaSuccess)

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
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string((rank + 1) % num_devices)));

  void* peer_d_ptr;
  CUDA_CALL(cudaIpcOpenMemHandle(
      &peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  int64_t peer_value;
  CUDA_CALL(cudaMemcpy(&peer_value, peer_d_ptr, size, cudaMemcpyDeviceToHost));

  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up
  CUDA_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  CUDA_CALL(cudaFree(d_ptr));
}

TEST_F(GpuCommTest, IpcMemHandlePtrArithmeticAtReceiver) {
  // TLDR; We can do pointer arithmetic on the receiver side.

  // Allocate GPU memory
  constexpr size_t size = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;
  void* d_ptr;
  CUDA_CALL(cudaMalloc(&d_ptr, size));

  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  CUDA_CALL(cudaMemcpy(d_ptr, values.data(), size, cudaMemcpyHostToDevice));

  cudaIpcMemHandle_t ipc_handle;
  CUDA_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));

  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));
  communicator_->barrier();
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));

  int64_t* peer_d_ptr;
  CUDA_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  int64_t peer_value;
  CUDA_CALL(cudaMemcpy(
      &peer_value, peer_d_ptr + 1, size / 2, cudaMemcpyDeviceToHost));

  EXPECT_EQ(2 * peer_rank + 1, peer_value);

  // Clean up
  CUDA_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  CUDA_CALL(cudaFree(d_ptr));
}

TEST_F(GpuCommTest, IpcMemHandlePtrArithmeticAtSender) {
  // TLDR; We CANNOT do pointer arithmetic on the sender side! The IPC handle
  // points to the beginning of the allocated buffer.

  // Allocate GPU memory
  constexpr size_t size = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;
  int64_t* d_ptr;
  CUDA_CALL(cudaMalloc(&d_ptr, size));

  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  CUDA_CALL(cudaMemcpy(d_ptr, values.data(), size, cudaMemcpyHostToDevice));

  cudaIpcMemHandle_t ipc_handle;
  CUDA_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr + 1));

  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));
  communicator_->barrier();
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));

  int64_t* peer_d_ptr;
  CUDA_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  int64_t peer_value;
  CUDA_CALL(
      cudaMemcpy(&peer_value, peer_d_ptr, size / 2, cudaMemcpyDeviceToHost));

  EXPECT_EQ(
      2 * peer_rank,
      peer_value); // and not 2 * peer_rank + 1 as could be expected!

  // Clean up
  CUDA_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  CUDA_CALL(cudaFree(d_ptr));
}

class StreamOpTest : public NVFuserTest {};

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

TEST_F(GpuCommTest, Allgather) {
  constexpr int64_t kTensorSize = 1024;

  at::Tensor input =
      at::full({kTensorSize}, communicator_->deviceId(), tensor_options);
  auto outputs = std::vector<at::Tensor>(communicator_->size());
  std::generate(outputs.begin(), outputs.end(), [&]() {
    return at::empty({kTensorSize}, tensor_options);
  });

  // AllgatherThroughCudaMemcpyAsync(input, outputs, communicator_);

  torch::cuda::synchronize();
  communicator_->barrier();

  for (int64_t i = 0; i < communicator_->size(); ++i) {
    at::Tensor expected = at::full({kTensorSize}, i, tensor_options);
    EXPECT_TRUE(outputs[i].equal(expected));
  }
}

} // namespace nvfuser
