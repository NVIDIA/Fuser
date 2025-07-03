// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <multidevice/communicator.h>
#include <multidevice/ipc_handle.h>

namespace nvfuser {

namespace {

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

} // namespace

IpcHandle::IpcHandle(at::Tensor tensor)
    : ptr_(tensor.data_ptr()),
      rank_(Communicator::getInstance().deviceId()),
      tensor_(tensor) {
  size_t psize = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(
      (CUdeviceptr*)&base_address_, &psize, (CUdeviceptr)ptr_));
  offset_from_base_address_ = static_cast<int64_t>(
      static_cast<uint8_t*>(ptr_) - static_cast<uint8_t*>(base_address_));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcGetMemHandle(&ipc_handle_, tensor.data_ptr()));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMalloc((void**)&semaphore_, sizeof(IpcSemaphore)));
  static_assert(
      sizeof(IpcSemaphore) == sizeof(int),
      "IpcSemaphore must be same size as int");
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemset(
      (void*)semaphore_, (int)IpcSemaphore::kReady, sizeof(IpcSemaphore)));
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcGetMemHandle(&semaphore_ipc_handle_, semaphore_));
}

IpcHandle::IpcHandle(std::vector<uint8_t> data) {
  const IpcHandle& imported_buffer = fromBytes<IpcHandle>(data);

  offset_from_base_address_ = imported_buffer.offset_from_base_address_;
  ipc_handle_ = imported_buffer.ipc_handle_;
  semaphore_ipc_handle_ = imported_buffer.semaphore_ipc_handle_;
  rank_ = imported_buffer.rank_;

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      &base_address_, ipc_handle_, cudaIpcMemLazyEnablePeerAccess));
  ptr_ = (void*)((uint8_t*)base_address_ + offset_from_base_address_);

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&semaphore_,
      semaphore_ipc_handle_,
      cudaIpcMemLazyEnablePeerAccess));
}

IpcHandle::~IpcHandle() {
  if (rank_ == Communicator::getInstance().deviceId()) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaFree((void*)semaphore_));
  } else {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(base_address_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle((void*)semaphore_));
  }
}

// retrieves a key for the TCP store corresponding to a `communication` and the
// exporter `rank`
std::string IpcHandleCache::getTcpStoreKey(
    P2PCommunication* communication,
    int64_t rank) const {
  const int64_t my_rank = Communicator::getInstance().deviceId();
  const int64_t peer =
      expr_evaluator_.evaluate(communication->peer()).as<int64_t>();
  const int64_t src =
      communication->type() == P2PCommunicationType::SEND ? my_rank : peer;
  const int64_t dst =
      communication->type() == P2PCommunicationType::SEND ? peer : my_rank;

  return "nvfuser_ipc_handle_info_P2PComm_dst=" + std::to_string(dst) +
      "_src=" + std::to_string(src) + "_rank=" + std::to_string(rank);
}

void IpcHandleCache::exchangeHandles(
    const std::vector<P2PCommunication*>& communications) {
  Communicator* communicator = &Communicator::getInstance();
  const int64_t my_rank = communicator->deviceId();

  std::vector<P2PCommunication*> non_cached_communications;
  for (auto communication : communications) {
    NVF_ERROR(
        expr_evaluator_.evaluate(communication->peer()).as<int64_t>() !=
            my_rank,
        "send to self not supported");
    if (find(communication) != nullptr) {
      continue;
    }
    non_cached_communications.push_back(communication);
  }

  // put memhandles to TCP store
  std::unordered_map<P2PCommunication*, std::unique_ptr<IpcHandle>>
      local_ipc_handles;
  auto store = communicator->getTcpStore();
  for (P2PCommunication* communication : non_cached_communications) {
    at::Tensor tensor =
        expr_evaluator_.evaluate(communication->buffer()).as<at::Tensor>();
    NVF_ERROR(
        tensor.is_contiguous(), "IpcHandle only supports contiguous tensors");
    auto buffer_handle = std::make_unique<IpcHandle>(tensor);
    auto key = getTcpStoreKey(communication, my_rank);
    // TODO: use multiSet
    store->set(key, toBytes(*buffer_handle));
    local_ipc_handles.emplace(communication, std::move(buffer_handle));
  }

  // get memhandles from TCP store
  for (P2PCommunication* communication : non_cached_communications) {
    const int64_t peer =
        expr_evaluator_.evaluate(communication->peer()).as<int64_t>();
    std::string key = getTcpStoreKey(communication, peer);
    // TCP store get is blocking until a timeout
    // TODO: use multiGet
    auto peer_ipc_handle = std::make_unique<IpcHandle>(store->get(key));
    store->deleteKey(key);
    auto& local_ipc_handle = local_ipc_handles.at(communication);

    auto ipc_handles = std::make_unique<P2pIpcHandle>(
        std::move(local_ipc_handle), std::move(peer_ipc_handle));

    insert(communication, std::move(ipc_handles));
  }

  // a barrier is needed here to ensure all ranks have received the
  // memhandles and the keys are deleted from the store before the next call to
  // exchangeHandles, otherwise there is a correctness issue
  // TODO: precisely select what ranks need to wait on that barrier.
  communicator->barrier();
}

} // namespace nvfuser
