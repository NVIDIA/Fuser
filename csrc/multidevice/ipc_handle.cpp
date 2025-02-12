// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include<cuda_utils.h>
#include<multidevice/communicator.h>
#include<multidevice/ipc_handle.h>

namespace nvfuser {

IpcHandle::IpcHandle(at::Tensor tensor)
    : ptr_(tensor.data_ptr()),
      storage_offset_(tensor.storage_offset()),
      element_size_(tensor.element_size()),
      is_imported_(false) {
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcGetMemHandle(&ipc_handle_, tensor.data_ptr()));
  const auto number_of_semaphores = Communicator::getInstance().size();
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(
      (void**)&semaphores_, number_of_semaphores * sizeof(IpcSemaphore)));
  static_assert(
      sizeof(IpcSemaphore) == sizeof(int),
      "IpcSemaphore must be same size as int");
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemset(
      (void*)semaphores_,
      (int)IpcSemaphore::kReady,
      number_of_semaphores * sizeof(IpcSemaphore)));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcGetMemHandle(&semaphores_ipc_handle_, semaphores_));
}

IpcHandle::IpcHandle(std::vector<uint8_t> data)
    : is_imported_(true) {
  const IpcHandle& imported_buffer = fromBytes<IpcHandle>(data);

  storage_offset_ = imported_buffer.storage_offset_;
  element_size_ = imported_buffer.element_size_;
  ipc_handle_ = imported_buffer.ipc_handle_;
  semaphores_ipc_handle_ = imported_buffer.semaphores_ipc_handle_;

  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcOpenMemHandle(&ptr_, ipc_handle_, cudaIpcMemLazyEnablePeerAccess));
  ptr_ = (void*)((uint8_t*)ptr_ + storage_offset_ * element_size_);

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&semaphores_,
      semaphores_ipc_handle_,
      cudaIpcMemLazyEnablePeerAccess));
}

IpcHandle::~IpcHandle() {
  if (is_imported_) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(ptr_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle((void*)semaphores_));
  } else {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaFree((void*)semaphores_));
  }
}

} // nvfuser
