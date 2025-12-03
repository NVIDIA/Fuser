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
#include <multidevice/utils.h>

namespace nvfuser {

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
      (void*)semaphore_, (int)IpcSemaphore::kIdle, sizeof(IpcSemaphore)));
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

// Retrieves a key for the TCP store corresponding to a `communication` and the
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

  // Put memhandles to TCP store
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

  // Get memhandles from TCP store
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

  if (non_cached_communications.empty()) {
    return;
  }
  // a barrier is needed here to ensure all ranks have received the
  // memhandles and the keys are deleted from the store before the next call to
  // exchangeHandles, otherwise there is a correctness issue
  // TODO: precisely select what ranks need to wait on that barrier.
  communicator->barrier();
}

SymMemForBroadcast::SymMemForBroadcast(
    Communication* communication,
    int64_t root,
    at::Tensor buffer)
    : SymMemForBroadcast(
          buffer,
          root,
          "for_Communication" + std::to_string(communication->name())) {}

SymMemForBroadcast::SymMemForBroadcast(
    at::Tensor buffer,
    int64_t root,
    const std::string& name_suffix) {
  std::string store_key_prefix = "nvls_export_mcast_handle_" + name_suffix;

  // Create symmetric tensor for the buffer
  buffer_sym_tensor_ = std::make_unique<SymmetricTensor>(buffer);

  // Setup multicast for the buffer
  buffer_sym_tensor_->setupRemoteHandles(store_key_prefix + "_buffer_unicast");

  // Setup multicast for the buffer
  buffer_sym_tensor_->setupMulticast(root, store_key_prefix + "_buffer_mcast");

  // Create semaphore tensor
  at::Tensor semaphore = SymmetricTensor::allocate(
      /*sizes=*/at::IntArrayRef({1}),
      /*dtype=*/at::ScalarType::Int,
      /*device=*/buffer.device());

  // Initialize the semaphore to kIdle
  IpcSemaphore init_value = IpcSemaphore::kIdle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      semaphore.data_ptr(),
      &init_value,
      sizeof(IpcSemaphore),
      cudaMemcpyHostToDevice));

  // Create symmetric tensor for the semaphore
  semaphore_sym_tensor_ = std::make_unique<SymmetricTensor>(semaphore);

  // Setup (unicast) IPC handles for the semaphore
  semaphore_sym_tensor_->setupRemoteHandles(store_key_prefix + "_semaphore");

  // Setup multicast for the semaphore
  semaphore_sym_tensor_->setupMulticast(
      root, store_key_prefix + "_semaphore_mcast");
}

void* SymMemForBroadcast::bufferMulticastPtr() const {
  return buffer_sym_tensor_->multicastPtr();
}

void* SymMemForBroadcast::bufferUnicastPtr(int64_t rank) const {
  return buffer_sym_tensor_->remoteTensor(rank).data_ptr();
}

void* SymMemForBroadcast::semaphoreMulticastPtr() const {
  return semaphore_sym_tensor_->multicastPtr();
}

void* SymMemForBroadcast::semaphoreUnicastPtr(int64_t rank) const {
  // Use a fixed tag for semaphore remote access
  return semaphore_sym_tensor_->remoteTensor(rank).data_ptr();
}

SymMemForAllgather::SymMemForAllgather(
    Communication* communication,
    at::Tensor buffer) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t world_size = communicator.size();

  // Initialize full buffer symmetric tensor for unicast access
  // We need to setup unicast handles on the full buffer because
  // setupRemoteHandles requires a VMM-aligned allocation, which slices are not.
  full_buffer_sym_tensor_ = std::make_unique<SymmetricTensor>(buffer);
  std::string full_buffer_suffix =
      std::to_string(communication->name()) + "_allgather_full";

  // Setup Unicast
  full_buffer_sym_tensor_->setupRemoteHandles(
      "nvls_export_mcast_handle_" + full_buffer_suffix + "_buffer_unicast");

  int64_t slice_numel = buffer.numel() / world_size;
  slice_size_bytes_ = slice_numel * buffer.element_size();

  // Setup Multicast on full buffer
  full_buffer_sym_tensor_->setupMulticast(
      /*exporter_rank=*/0,
      "nvls_export_mcast_handle_" + full_buffer_suffix + "_buffer_mcast");

  // Allocate semaphores (one per rank) in a single symmetric tensor
  at::Tensor semaphores = SymmetricTensor::allocate(
      at::IntArrayRef({world_size}), at::ScalarType::Int, buffer.device());

  // Init semaphores to kIdle
  std::vector<IpcSemaphore> init_values(world_size, IpcSemaphore::kIdle);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      semaphores.data_ptr(),
      init_values.data(),
      world_size * sizeof(IpcSemaphore),
      cudaMemcpyHostToDevice));

  semaphores_sym_tensor_ = std::make_unique<SymmetricTensor>(semaphores);
  semaphores_sym_tensor_->setupRemoteHandles(
      "nvls_export_mcast_handle_" + full_buffer_suffix + "_semaphores_unicast");
  semaphores_sym_tensor_->setupMulticast(
      /*exporter_rank=*/0,
      "nvls_export_mcast_handle_" + full_buffer_suffix + "_semaphores_mcast");
}

void* SymMemForAllgather::bufferMulticastPtr(int64_t root_rank) const {
  uint8_t* base_ptr = (uint8_t*)full_buffer_sym_tensor_->multicastPtr();
  return base_ptr + (root_rank * slice_size_bytes_);
}

void* SymMemForAllgather::bufferUnicastPtr(int64_t root_rank, int64_t rank)
    const {
  uint8_t* base_ptr =
      (uint8_t*)full_buffer_sym_tensor_->remoteTensor(rank).data_ptr();
  return base_ptr + (root_rank * slice_size_bytes_);
}

void* SymMemForAllgather::semaphoreMulticastPtr(int64_t root_rank) const {
  uint8_t* base_ptr = (uint8_t*)semaphores_sym_tensor_->multicastPtr();
  return base_ptr + (root_rank * sizeof(IpcSemaphore));
}

void* SymMemForAllgather::semaphoreUnicastPtr(int64_t root_rank, int64_t rank)
    const {
  uint8_t* base_ptr =
      (uint8_t*)semaphores_sym_tensor_->remoteTensor(rank).data_ptr();
  return base_ptr + (root_rank * sizeof(IpcSemaphore));
}

SymmetricMemoryHandle* SymmetricMemoryHandleCache::get(KeyType key) {
  auto it = handles_.find(key);
  if (it != handles_.end()) {
    return it->second.get();
  }

  // If not found, create a new handle based on the expr type
  std::unique_ptr<SymmetricMemoryHandle> handle;

  if (auto* dtca = dynamic_cast<hir::SymmetricContiguousView*>(key.expr)) {
    // SymmetricContiguousView
    handle = std::make_unique<SymMemForContiguousView>(key.buffer, dtca);
  } else if (auto* comm = dynamic_cast<Communication*>(key.expr)) {
    // Communication (Broadcast/Allgather)
    if (comm->type() == CommunicationType::Broadcast) {
      handle = std::make_unique<SymMemForBroadcast>(comm, key.root, key.buffer);
    } else if (comm->type() == CommunicationType::Allgather) {
      handle = std::make_unique<SymMemForAllgather>(comm, key.buffer);
    } else {
      NVF_ERROR(
          false,
          "Unsupported communication type for multicast handle: ",
          comm->type());
    }
  } else {
    NVF_ERROR(
        false, "Unsupported expr type for symmetric memory handle: ", key.expr);
  }

  auto inserted = handles_.emplace(key, std::move(handle));
  return inserted.first->second.get();
}

SymMemForContiguousView::SymMemForContiguousView(
    at::Tensor in_tensor,
    hir::SymmetricContiguousView* unshard) {
  std::string tag = "unshard_" + std::to_string(unshard->name());
  sym_tensor_ = std::make_unique<SymmetricTensor>(in_tensor);
  sym_tensor_->setupContiguousView(tag);

  at::Tensor contiguous = sym_tensor_->getContiguousView();

  // Remove the DIDx dimension (outermost) if it has size 1
  if (contiguous.size(0) == 1) {
    contiguous = contiguous.squeeze(0);
  }

  tensor_ = contiguous;
}

} // namespace nvfuser
