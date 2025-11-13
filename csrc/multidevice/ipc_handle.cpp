// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <host_ir/host_ir.h>
#include <multidevice/communicator.h>
#include <multidevice/ipc_handle.h>
#include <multidevice/symmetric_tensor.h>

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
    store->set(key, toBytes(*buffer_handle));
    local_ipc_handles.emplace(communication, std::move(buffer_handle));
  }

  // Get memhandles from TCP store
  for (P2PCommunication* communication : non_cached_communications) {
    const int64_t peer =
        expr_evaluator_.evaluate(communication->peer()).as<int64_t>();
    std::string key = getTcpStoreKey(communication, peer);
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
  // Barrier is needed to ensure all ranks have received the memhandles
  // and keys are deleted from the store before the next call to exchangeHandles
  communicator->barrier();
}

MulticastHandleForBroadcast::MulticastHandleForBroadcast(
    Communication* communication,
    at::Tensor buffer)
    : MulticastHandleForBroadcast(
          buffer,
          communication->root(),
          "for_Communication" + communication->name()) {}

MulticastHandleForBroadcast::MulticastHandleForBroadcast(
    at::Tensor buffer,
    int64_t root,
    const std::string& name_suffix) {
  std::string store_key_prefix = "nvls_export_mcast_handle_" + name_suffix;

  // Create symmetric tensor for the buffer
  buffer_sym_tensor_ = std::make_unique<SymmetricTensor>(
      buffer, store_key_prefix + "_buffer");
  
  // Setup multicast for the buffer
  buffer_sym_tensor_->setupMulticast(root, store_key_prefix + "_buffer_mcast");

  // Create a symmetric memory tensor for the semaphore (single int32 element)
  at::Tensor semaphore = allocateSymmetricTensor(
      /*sizes=*/at::IntArrayRef({1}),
      /*dtype=*/at::ScalarType::Int,
      /*device=*/buffer.device(),
      /*alloc_id=*/std::nullopt);

  // Initialize the semaphore to kReady
  IpcSemaphore init_value = IpcSemaphore::kReady;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      semaphore.data_ptr(),
      &init_value,
      sizeof(IpcSemaphore),
      cudaMemcpyHostToDevice));

  // Create symmetric tensor for the semaphore
  semaphore_sym_tensor_ = std::make_unique<SymmetricTensor>(
      semaphore, store_key_prefix + "_semaphore");

  // Setup (unicast) IPC handles for the semaphore
  semaphore_sym_tensor_->setupIpcHandles();

  // Setup multicast for the semaphore
  semaphore_sym_tensor_->setupMulticast(
      root, store_key_prefix + "_semaphore_mcast");
}

void* MulticastHandleForBroadcast::buffer_multicast_ptr() const {
  return buffer_sym_tensor_->multicastPtr();
}

void* MulticastHandleForBroadcast::semaphore_multicast_ptr() const {
  return semaphore_sym_tensor_->multicastPtr();
}

void* MulticastHandleForBroadcast::semaphore_unicast_ptr(int64_t rank) const {
  return semaphore_sym_tensor_->remoteTensorPtr(rank);
}

MulticastHandleForAllgather::MulticastHandleForAllgather(
    Communication* communication,
    at::Tensor buffer) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t world_size = communicator.size();

  // Allgather is world_size broadcasts, each broadcasting a different slice
  // of the output buffer. Create one MulticastHandleForBroadcast per rank.
  broadcast_handles_.reserve(world_size);

  for (int64_t root_rank = 0; root_rank < world_size; ++root_rank) {
    // Each rank gets a slice of the output buffer
    int64_t slice_size = buffer.numel() / world_size;
    // Flatten the tensor before slicing to ensure it is 1D
    at::Tensor sliced_buffer = buffer.view({-1}).slice(
        /*dim=*/0,
        /*start=*/root_rank * slice_size,
        /*end=*/(root_rank + 1) * slice_size);
    // Create unique name suffix for this broadcast
    std::string name_suffix =
        communication->name() + "_allgather_root" + std::to_string(root_rank);

    // Create MulticastHandleForBroadcast for this slice
    broadcast_handles_.push_back(std::make_unique<MulticastHandleForBroadcast>(
        sliced_buffer, root_rank, name_suffix));
  }
}

void* MulticastHandleForAllgather::buffer_multicast_ptr(int64_t root_rank) const {
  return broadcast_handles_[root_rank]->buffer_multicast_ptr();
}

void* MulticastHandleForAllgather::semaphore_multicast_ptr(int64_t root_rank) const {
  return broadcast_handles_[root_rank]->semaphore_multicast_ptr();
}

void* MulticastHandleForAllgather::semaphore_unicast_ptr(
    int64_t root_rank,
    int64_t rank) const {
  return broadcast_handles_[root_rank]->semaphore_unicast_ptr(rank);
}

SymmetricMemoryHandle* SymmetricMemoryHandleCache::get(KeyType key) {
  auto it = handles_.find(key);
  if (it != handles_.end()) {
    return it->second.get();
  }

  // If not found, create a new handle based on the expr type
  std::unique_ptr<SymmetricMemoryHandle> handle;

  if (auto* dtca =
          dynamic_cast<hir::DistributedTensorContiguousAliasing*>(key.expr)) {
    // DistributedTensorContiguousAliasing
    handle = std::make_unique<ContiguousViewHandle>(key.buffer, dtca);
  } else if (auto* comm = dynamic_cast<Communication*>(key.expr)) {
    // Communication (Broadcast/Allgather)
    if (comm->type() == CommunicationType::Broadcast) {
      handle = std::make_unique<MulticastHandleForBroadcast>(comm, key.buffer);
    } else if (comm->type() == CommunicationType::Allgather) {
      handle = std::make_unique<MulticastHandleForAllgather>(comm, key.buffer);
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

ContiguousViewHandle::ContiguousViewHandle(
    at::Tensor in_tensor,
    hir::DistributedTensorContiguousAliasing* unshard) {
  // Create SymmetricTensor from the input tensor
  // Validation happens automatically in SymmetricTensor constructor
  std::string tag = "unshard_" + std::to_string(unshard->name());
  sym_tensor_ = std::make_unique<SymmetricTensor>(in_tensor, tag);

  // Create contiguous view across all ranks
  at::Tensor contiguous = createContiguousView(*sym_tensor_);

  // Remove the DIDx dimension (outermost) if it has size 1
  if (contiguous.size(0) == 1) {
    contiguous = contiguous.squeeze(0);
  }

  tensor_ = contiguous;
}

} // namespace nvfuser
