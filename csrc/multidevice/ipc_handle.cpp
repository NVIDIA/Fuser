// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
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
      storage_offset_(tensor.storage_offset()),
      element_size_(tensor.element_size()),
      rank_(Communicator::getInstance().deviceId()) {
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

  storage_offset_ = imported_buffer.storage_offset_;
  element_size_ = imported_buffer.element_size_;
  ipc_handle_ = imported_buffer.ipc_handle_;
  semaphore_ipc_handle_ = imported_buffer.semaphore_ipc_handle_;
  rank_ = imported_buffer.rank_;

  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaIpcOpenMemHandle(&ptr_, ipc_handle_, cudaIpcMemLazyEnablePeerAccess));
  ptr_ = (void*)((uint8_t*)ptr_ + storage_offset_ * element_size_);

  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&semaphore_,
      semaphore_ipc_handle_,
      cudaIpcMemLazyEnablePeerAccess));
}

IpcHandle::~IpcHandle() {
  if (rank_ == Communicator::getInstance().deviceId()) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaFree((void*)semaphore_));
  } else {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(ptr_));
    NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle((void*)semaphore_));
  }
}

void IpcHandleCache::exchangeHandles(
    const std::vector<P2PCommunication*>& communications,
    const ExpressionEvaluator& expr_evaluator) {
  Communicator* communicator = &Communicator::getInstance();
  const int64_t my_rank = communicator->deviceId();
  auto get_tensor =
      [&expr_evaluator](P2PCommunication* communication) -> at::Tensor {
    return expr_evaluator.evaluate(communication->buffer()).as<at::Tensor>();
  };

  std::vector<P2PCommunication*> non_cached_communications;
  for (auto communication : communications) {
    NVF_ERROR(
        expr_evaluator.evaluate(communication->peer()).as<int64_t>() != my_rank,
        "send to self not supported");
    if (find(communication, expr_evaluator) != nullptr) {
      continue;
    }
    non_cached_communications.push_back(communication);
  }

  // put memhandles to TCP store
  auto get_tcp_store_key = [my_rank, &expr_evaluator](
                               P2PCommunication* communication,
                               int64_t rank) -> std::string {
    const int64_t peer =
        expr_evaluator.evaluate(communication->peer()).as<int64_t>();
    const int64_t src =
        communication->type() == P2PCommunicationType::SEND ? my_rank : peer;
    const int64_t dst =
        communication->type() == P2PCommunicationType::SEND ? peer : my_rank;

    return "nvfuser_ipc_handle_info_P2PComm_dst=" + std::to_string(dst) +
        "_src=" + std::to_string(src) + "_rank=" + std::to_string(rank);
  };
  std::unordered_map<P2PCommunication*, std::unique_ptr<IpcHandle>>
      local_ipc_handles;
  auto store = communicator->getTcpStore();
  for (P2PCommunication* communication : non_cached_communications) {
    auto buffer_handle = std::make_unique<IpcHandle>(get_tensor(communication));
    auto key = get_tcp_store_key(communication, my_rank);
    keys_.insert(key);
    // TODO: use multiSet
    store->set(key, toBytes(*buffer_handle));
    local_ipc_handles.emplace(communication, std::move(buffer_handle));
  }

  // barrier to ensure all ranks have pushed their memhandles to the store
  // TODO: precisely select what ranks need to wait on that barrier.
  communicator->barrier();

  // get memhandles from TCP store
  for (P2PCommunication* communication : non_cached_communications) {
    const int64_t peer =
        expr_evaluator.evaluate(communication->peer()).as<int64_t>();
    std::string key = get_tcp_store_key(communication, peer);
    NVF_ERROR(
        store->check({key}),
        "key ",
        key,
        " not found in store at rank ",
        my_rank);
    // TODO: use multiGet
    auto peer_ipc_handle = std::make_unique<IpcHandle>(store->get(key));
    store->deleteKey(key);
    auto& local_ipc_handle = local_ipc_handles.at(communication);

    auto ipc_handles = std::make_unique<P2pIpcHandle>(
        std::move(local_ipc_handle), std::move(peer_ipc_handle));

    insert(communication, expr_evaluator, std::move(ipc_handles));
  }
}

} // namespace nvfuser
