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

MulticastHandleForBcast::MulticastHandleForBcast(
    Communication* communication,
    at::Tensor buffer) {
#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  Communicator& communicator = Communicator::getInstance();
  const int64_t my_device_index = communicator.deviceId();
  const int64_t local_rank = communicator.local_rank();
  const int64_t world_size = communicator.size();
  const int64_t exporter_rank = communication->root();

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

  std::string error_message = is_symmetric_memory_valid(buffer);
  NVF_ERROR(error_message.empty(), error_message);
  NVFUSER_CUDA_SAFE_CALL(cuMemRetainAllocationHandle(
      &output_alloc_handle, (void*)buffer.data_ptr()));

  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, output_alloc_handle));

  const size_t kNumElems = buffer.numel();
  const size_t kSizeBytes = kNumElems * buffer.element_size();
  const int64_t granularity = get_granularity(prop, kSizeBytes);
  rounded_size = ((kSizeBytes + granularity - 1) / granularity) * granularity;

  auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = handle_type;
  mcast_prop.numDevices = world_size;
  mcast_prop.size = static_cast<size_t>(rounded_size);

  int shared_handle;
  auto store = communicator.getTcpStore();
  pid_t root_pid;
  if (my_device_index == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle, handle_type, /*flags=*/0));

    int status = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    NVF_ERROR(status >= 0, "Failed to set ptrace policy");
    store->set(
        std::string("nvls_export_fd_mcast_handle"), toBytes(shared_handle));
    root_pid = getpid();
    store->set(std::string("nvls_export_pid_mcast_handle"), toBytes(root_pid));
  }

  communicator.barrier();

  if (my_device_index != exporter_rank) {
    shared_handle =
        fromBytes<int>(store->get(std::string("nvls_export_fd_mcast_handle")));
    root_pid = fromBytes<pid_t>(
        store->get(std::string("nvls_export_pid_mcast_handle")));

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
        &mcast_handle, (void*)((uint64_t)peer_fd), handle_type));
    int status = close(pid_fd);
    NVF_ERROR(status >= 0, "Failed to close pidfd");
  }

  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle, cu_dev));

  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle,
      /*mcOffset=*/0,
      output_alloc_handle,
      /*memOffset=*/0,
      rounded_size,
      /*flags=*/0));

  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &mc_ptr,
      rounded_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(
      mc_ptr,
      rounded_size,
      /*offset=*/0,
      mcast_handle,
      /*flags=*/0));
  CUmemAccessDesc mc_mapping_desc{};
  mc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mc_mapping_desc.location.id = static_cast<int>(local_rank);
  mc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(mc_ptr, rounded_size, &mc_mapping_desc, /*count=*/1));
#else
  NVF_ERROR(false, "Multicast is not supported");
#endif
}

MulticastHandleForBcast::~MulticastHandleForBcast() {
#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(mc_ptr, rounded_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(mc_ptr, rounded_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(output_alloc_handle));
  NVFUSER_CUDA_SAFE_CALL(
      cuMulticastUnbind(mcast_handle, cu_dev, /*offset=*/0, rounded_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mcast_handle));
#endif
}

const MulticastHandleForBcast& MulticastHandleCache::get(
    Communication* communication) {
  auto buffer =
      expr_evaluator_.evaluate(communication->output(0)).as<at::Tensor>();
  auto key = KeyType{buffer, communication};
  auto it = handles_.find(key);
  if (it != handles_.end()) {
    return *(it->second);
  }

  // If not found, create a new MulticastHandleForBcast, store, and return
  // reference
  auto handle =
      std::make_unique<MulticastHandleForBcast>(communication, buffer);
  auto inserted = handles_.emplace(key, std::move(handle));
  return *(inserted.first->second);
}

} // namespace nvfuser
