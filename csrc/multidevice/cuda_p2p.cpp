// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <multidevice/cuda_p2p.h>

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

} // namespace nvfuser
