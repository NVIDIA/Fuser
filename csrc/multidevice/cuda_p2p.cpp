// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/ipc_handle.h>

#include <cuda_utils.h>
#include <multidevice/cuda_p2p.h>
#include <multidevice/symmetric_memory.h>
#include "multidevice/communicator.h"

namespace nvfuser {

namespace get_zcopy {

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream) {
  // wait for sender to be ready
  NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WAIT_VALUE_EQ));
  // RDMA get the data from the sender
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      ipc_handles.local().ptr(),
      ipc_handles.peer().ptr(),
      count,
      cudaMemcpyDeviceToDevice,
      stream));
  // Signals completion to self
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WRITE_VALUE_DEFAULT));
  // Signals completion to sender
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.peer().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WRITE_VALUE_DEFAULT));
}

void sendPost(const P2pIpcHandle& ipc_handles, CUstream stream) {
  // signal to self that transfer is in progress
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WRITE_VALUE_DEFAULT));
  // signal to receiver that the buffer is ready
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.peer().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WRITE_VALUE_DEFAULT)); // passing
                                       // CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
                                       // gives an error
}

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream) {
  NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WAIT_VALUE_EQ));
}

} // namespace get_zcopy

void postBroadcastWithP2pBackend(
    Communication* communication,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    MulticastHandleCache& multicast_handle_cache) {
  NVF_ERROR(
      communication->type() == CommunicationType::Broadcast,
      "Invalid communication type, expected Broadcast, got: ",
      communication->type());

  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();

  const size_t kNumElems = output_tensor.numel();
  const size_t kSizeBytes = kNumElems * output_tensor.element_size();

  const MulticastHandleForBcast& mcast = multicast_handle_cache.get({output_tensor, communication});

  if (my_device_index == communication->root()) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        mcast.ptr(),
        input_tensor.data_ptr(),
        kSizeBytes,
        cudaMemcpyHostToDevice));
  }

  communicator.barrier();
}

} // namespace nvfuser
