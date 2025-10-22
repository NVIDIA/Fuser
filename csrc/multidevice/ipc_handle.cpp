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
#include <multidevice/symmetric_memory.h>

#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

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

  if (non_cached_communications.empty()) {
    return;
  }
  // a barrier is needed here to ensure all ranks have received the
  // memhandles and the keys are deleted from the store before the next call to
  // exchangeHandles, otherwise there is a correctness issue
  // TODO: precisely select what ranks need to wait on that barrier.
  communicator->barrier();
}

UnicastHandle::UnicastHandle(
    at::Tensor tensor,
    int64_t exporter_rank,
    const std::string& store_key_prefix)
    : ptr_(tensor.data_ptr()), tensor_(tensor) {
  NVF_ERROR(
      tensor.is_contiguous(), "UnicastHandle only supports contiguous tensors");

  Communicator& communicator = Communicator::getInstance();
  const int64_t local_rank = communicator.local_rank();

  // Check VMM support
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      local_rank));
  NVF_ERROR(
      is_vmm_supported != 0,
      "Device does not support Virtual Memory Management");

  // Check IPC support
  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  NVF_ERROR(is_ipc_supported != 0, "Device does not support IPC handles");

  // Get base address and size
  CUdeviceptr base_address = 0;
  size_t psize = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(
      &base_address, &psize, reinterpret_cast<CUdeviceptr>(ptr_)));

  // Get the allocation handle for the tensor
  CUmemGenericAllocationHandle alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemRetainAllocationHandle(&alloc_handle, ptr_));

  // Get allocation properties to determine granularity
  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle));

  size_t granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // Export to shareable file descriptor
  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      alloc_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  // Allow peer processes to use pidfd_getfd
  int status = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
  NVF_ERROR(status >= 0, "Failed to set ptrace policy");

  // Export to store
  auto store = communicator.getTcpStore();
  pid_t my_pid = getpid();

  store->set(store_key_prefix + "_fd", toBytes(shared_fd));
  store->set(store_key_prefix + "_pid", toBytes(my_pid));
  store->set(store_key_prefix + "_granularity", toBytes(granularity));
  store->set(store_key_prefix + "_size", toBytes(psize));

  // Release the allocation handle (we don't need to keep it, the tensor keeps
  // the memory alive)
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(alloc_handle));
}

UnicastHandle::UnicastHandle(
    int64_t exporter_rank,
    const std::string& store_key_prefix) {
  // Import from store
  Communicator& communicator = Communicator::getInstance();
  const int64_t local_rank = communicator.local_rank();
  auto store = communicator.getTcpStore();

  int peer_shared_fd = fromBytes<int>(store->get(store_key_prefix + "_fd"));
  pid_t peer_pid = fromBytes<pid_t>(store->get(store_key_prefix + "_pid"));
  size_t granularity =
      fromBytes<size_t>(store->get(store_key_prefix + "_granularity"));
  size_ = fromBytes<size_t>(store->get(store_key_prefix + "_size"));

  // Get the peer's file descriptor using pidfd_open and pidfd_getfd
  pid_fd_ = syscall(SYS_pidfd_open, peer_pid, /*flags=*/0);
  NVF_ERROR(
      pid_fd_ >= 0,
      "Failed to open pidfd for pid ",
      peer_pid,
      " from rank ",
      exporter_rank);

  peer_fd_ = syscall(SYS_pidfd_getfd, pid_fd_, peer_shared_fd, /*flags=*/0);
  NVF_ERROR(peer_fd_ >= 0, "Failed to get peer fd from rank ", exporter_rank);

  // Import the peer's memory handle
  NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
      &mem_handle_,
      (void*)((uint64_t)peer_fd_),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

  // Reserve virtual address space and map the peer's memory
  CUdeviceptr mapped_cu_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &mapped_cu_ptr,
      size_,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(
      mapped_cu_ptr, size_, /*offset=*/0, mem_handle_, /*flags=*/0));

  // Set memory access permissions
  CUmemAccessDesc access_desc{};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = static_cast<int>(local_rank);
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(mapped_cu_ptr, size_, &access_desc, /*count=*/1));

  ptr_ = reinterpret_cast<void*>(mapped_cu_ptr);
}

UnicastHandle::~UnicastHandle() {
  if (tensor_.defined()) {
    // Exporter: nothing to do, tensor_ reference will keep buffer alive
  } else {
    // Importer: clean up VMM resources
    if (ptr_ != nullptr) {
      CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(ptr_);
      NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(cu_ptr, size_));
      NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(cu_ptr, size_));
    }
    if (mem_handle_ != 0) {
      NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mem_handle_));
    }
    if (peer_fd_ >= 0) {
      close(peer_fd_);
    }
    if (pid_fd_ >= 0) {
      close(pid_fd_);
    }
  }
}

MulticastHandle::MulticastHandle(
    at::Tensor tensor,
    int64_t exporter_rank,
    const std::string& store_key_prefix)
    : tensor_(tensor) {
#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t local_rank = communicator.local_rank();
  const int64_t world_size = communicator.size();

  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  NVF_ERROR(is_ipc_supported != 0, "Device does not support IPC handles");

  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  NVF_ERROR(
      is_multicast_supported != 0, "Device does not support Multicast Objects");

  std::string error_message = is_symmetric_memory_valid(tensor);
  NVF_ERROR(error_message.empty(), error_message);
  CUmemGenericAllocationHandle alloc_handle{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&alloc_handle, (void*)tensor.data_ptr()));

  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle));

  const size_t unrounded_size = tensor.numel() * tensor.element_size();
  const int64_t granularity = get_granularity(prop, unrounded_size);
  // Rounds up to the size to the nearest multiple of the granularity
  size_ = ((unrounded_size + granularity - 1) / granularity) * granularity;

  auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = handle_type;
  mcast_prop.numDevices = world_size;
  mcast_prop.size = static_cast<size_t>(size_);

  int shared_handle;
  auto store = communicator.getTcpStore();
  pid_t root_pid;
  if (my_device_index == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle_, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle_, handle_type, /*flags=*/0));

    int status = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    NVF_ERROR(status >= 0, "Failed to set ptrace policy");
    store->set(store_key_prefix + "_fd", toBytes(shared_handle));
    root_pid = getpid();
    store->set(store_key_prefix + "_pid", toBytes(root_pid));
  }

  communicator.barrier();

  if (my_device_index != exporter_rank) {
    shared_handle = fromBytes<int>(store->get(store_key_prefix + "_fd"));
    root_pid = fromBytes<pid_t>(store->get(store_key_prefix + "_pid"));

    int pid_fd, peer_fd;
    pid_fd = syscall(SYS_pidfd_open, root_pid, /*flags=*/0);
    NVF_ERROR(
        pid_fd >= 0,
        "my_device_index ",
        my_device_index,
        " failed to open pidfd for pid ",
        root_pid);

    peer_fd = syscall(SYS_pidfd_getfd, pid_fd, shared_handle, /*flags=*/0);
    NVF_ERROR(
        peer_fd >= 0,
        "my_device_index ",
        my_device_index,
        " failed to get peer fd");
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &mcast_handle_, (void*)((uint64_t)peer_fd), handle_type));
    int status = close(pid_fd);
    NVF_ERROR(status >= 0, "Failed to close pidfd");
  }

  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev_, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle_, cu_dev_));

  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle_,
      /*mcOffset=*/0,
      alloc_handle,
      /*memOffset=*/0,
      size_,
      /*flags=*/0));

  CUdeviceptr mc_cu_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &mc_cu_ptr,
      size_,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemMap(mc_cu_ptr, size_, /*offset=*/0, mcast_handle_, /*flags=*/0));
  CUmemAccessDesc mc_mapping_desc{};
  mc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mc_mapping_desc.location.id = static_cast<int>(local_rank);
  mc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(mc_cu_ptr, size_, &mc_mapping_desc, /*count=*/1));

  mc_ptr_ = reinterpret_cast<void*>(mc_cu_ptr);

  communicator.barrier();

  if (my_device_index == exporter_rank) {
    store->deleteKey(store_key_prefix + "_fd");
    store->deleteKey(store_key_prefix + "_pid");
  }
#else
  NVF_ERROR(false, "Multicast is not supported");
#endif
}

MulticastHandle::~MulticastHandle() {
#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(mc_ptr_);
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(cu_ptr, size_));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(cu_ptr, size_));
  NVFUSER_CUDA_SAFE_CALL(
      cuMulticastUnbind(mcast_handle_, cu_dev_, /*offset=*/0, size_));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mcast_handle_));
#endif
}

MulticastHandleForBroadcast::MulticastHandleForBroadcast(
    Communication* communication,
    at::Tensor buffer) {
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_rank = communicator.deviceId();
  const int64_t world_size = communicator.size();
  const int64_t root = communication->root();
  std::string store_key_prefix =
      "nvls_export_mcast_handle_for_Communication" + communication->name();

  // Create multicast handle for the buffer
  buffer_multicast_handle_ =
      std::make_unique<MulticastHandle>(buffer, root, store_key_prefix);

  // Create a symmetric memory tensor for the semaphore (single int32 element)
  at::Tensor semaphore = empty_strided_cuda_symmetric(
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

  // Create multicast handle for the semaphore
  semaphore_multicast_handle_ = std::make_unique<MulticastHandle>(
      semaphore, root, store_key_prefix + "_semaphore");

  // Create per-rank semaphores: each rank exports its own semaphore using
  // UnicastHandle
  semaphore_handles_.resize(world_size);
  std::string my_store_key =
      store_key_prefix + "_per_rank_semaphore_" + std::to_string(my_rank);
  semaphore_handles_[my_rank] =
      std::make_unique<UnicastHandle>(semaphore, my_rank, my_store_key);

  // Barrier to ensure all ranks have exported before any rank starts importing
  communicator.barrier();

  if (my_rank != root) {
    return;
  }

  // Root imports all other ranks' semaphores
  for (int64_t rank = 0; rank < world_size; ++rank) {
    if (rank != my_rank) {
      std::string rank_store_key =
          store_key_prefix + "_per_rank_semaphore_" + std::to_string(rank);
      semaphore_handles_[rank] =
          std::make_unique<UnicastHandle>(rank, rank_store_key);
    }
  }
}

const MulticastHandleForBroadcast& MulticastHandleCache::get(KeyType key) {
  auto it = handles_.find(key);
  if (it != handles_.end()) {
    return *(it->second);
  }

  // If not found, create a new MulticastHandleForBroadcast, store, and return
  // reference
  auto handle =
      std::make_unique<MulticastHandleForBroadcast>(key.comm, key.buffer);
  auto inserted = handles_.emplace(key, std::move(handle));
  return *(inserted.first->second);
}

} // namespace nvfuser
