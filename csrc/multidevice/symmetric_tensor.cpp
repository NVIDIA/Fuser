// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/symmetric_tensor.h>

#include <atomic>
#include <functional>
#include <numeric>
#include <cuda_utils.h>
#include <driver_api.h>
#include <multidevice/communicator.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace nvfuser {

namespace {

// Returns the allocation granularity for symmetric memory.
// Optionally considers multicast granularity if device supports it.
// get_recommended_granularity: if true, uses recommended (larger) multicast granularity
//                               if false (default), uses minimum multicast granularity
int64_t getGranularityForSymmetricMemory(
    const CUmemAllocationProp& prop,
    size_t requested_size_bytes,
    bool get_recommended_granularity = false) {
  size_t alloc_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &alloc_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  // Check if device supports multicast before querying multicast granularity
  int is_multicast_supported = 0;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      prop.location.id));
  
  if (is_multicast_supported == 0) {
    // Device doesn't support multicast, use regular allocation granularity
    return alloc_granularity;
  }

  // Device supports multicast, query multicast granularity
  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mcast_prop.numDevices = Communicator::getInstance().size();
  mcast_prop.size = requested_size_bytes;

  size_t mcast_min_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
      &mcast_min_granularity, &mcast_prop, CU_MULTICAST_GRANULARITY_MINIMUM));

  size_t granularity = mcast_min_granularity;

  // Optionally get recommended granularity (typically larger, better performance)
  if (get_recommended_granularity) {
    size_t mcast_rec_granularity = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
        &mcast_rec_granularity,
        &mcast_prop,
        CU_MULTICAST_GRANULARITY_RECOMMENDED));
    granularity = mcast_rec_granularity;
  }

  return std::max(alloc_granularity, granularity);
#else
  (void)requested_size_bytes;
  (void)get_recommended_granularity;
  return alloc_granularity;
#endif
}

} // namespace

SymmetricTensor::SymmetricTensor(
    const at::Tensor& local_tensor)
    : local_tensor_(local_tensor) {
  NVF_ERROR(local_tensor.is_cuda(), "Expected CUDA tensor, got: ", local_tensor.device());

  std::string error = isSymmetricAllocationValid(local_tensor);
  NVF_CHECK(error.empty(), "Invalid symmetric allocation: ", error);

  Communicator& comm = Communicator::getInstance();
  world_size_ = comm.size();
  my_device_id_ = comm.deviceId();

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(comm.local_rank());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t required_size = local_tensor.numel() * local_tensor.element_size();
  aligned_size_ = ((required_size + granularity_ - 1) / granularity_) * granularity_;

  alloc_handles_.resize(world_size_);
  remote_ptrs_.resize(world_size_);
  
  CUdeviceptr local_ptr = reinterpret_cast<CUdeviceptr>(local_tensor_.data_ptr());
  CUmemGenericAllocationHandle local_handle;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&local_handle, reinterpret_cast<void*>(local_ptr)));

  alloc_handles_[my_device_id_] = local_handle;
  remote_ptrs_[my_device_id_] = local_ptr;
}

SymmetricTensor::~SymmetricTensor() {
#if (CUDA_VERSION >= 13000)
  if (is_multicast_setup_) {
    if (mc_ptr_) {
      cuMemUnmap(reinterpret_cast<CUdeviceptr>(mc_ptr_), aligned_size_);
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(mc_ptr_), aligned_size_);
    }
    if (mcast_handle_) {
      cuMulticastUnbind(mcast_handle_, cu_dev_, 0, aligned_size_);
      cuMemRelease(mcast_handle_);
    }
    if (peer_fd_ >= 0) close(peer_fd_);
    if (pid_fd_ >= 0) close(pid_fd_);
  }
#endif

  if (are_remote_tensors_setup_ == true) {
    for (int64_t rank = 0; rank < world_size_; ++rank) {
      if (rank != my_device_id_ && remote_ptrs_[rank]) {
        cuMemUnmap(remote_ptrs_[rank], aligned_size_);
        cuMemAddressFree(remote_ptrs_[rank], aligned_size_);
      }
      if (rank != my_device_id_ && alloc_handles_[rank]) {
        cuMemRelease(alloc_handles_[rank]);
      }
    }
  }

  if (alloc_handles_.size() > static_cast<size_t>(my_device_id_) && alloc_handles_[my_device_id_]) {
    cuMemRelease(alloc_handles_[my_device_id_]);
  }
}

void SymmetricTensor::setupRemoteHandles(const std::string& tag) const {
  if (are_remote_tensors_setup_ == true) {
    return;
  }
  Communicator& comm = Communicator::getInstance();
  auto store = comm.getTcpStore();
  CUmemGenericAllocationHandle local_handle = alloc_handles_[my_device_id_];

  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      local_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);

  std::string key_prefix = "sym_tensor_" + std::to_string(my_device_id_) + "_" + tag;
  
  std::vector<uint8_t> fd_bytes(
      reinterpret_cast<const uint8_t*>(&shared_fd),
      reinterpret_cast<const uint8_t*>(&shared_fd) + sizeof(int));
  pid_t my_pid = getpid();
  std::vector<uint8_t> pid_bytes(
      reinterpret_cast<const uint8_t*>(&my_pid),
      reinterpret_cast<const uint8_t*>(&my_pid) + sizeof(pid_t));

  store->set(key_prefix + "_fd", fd_bytes);
  store->set(key_prefix + "_pid", pid_bytes);

  comm.barrier();

  for (int64_t peer = 0; peer < world_size_; ++peer) {
    if (peer == my_device_id_) {
      continue;
    }

    std::string peer_key = "sym_tensor_" + std::to_string(peer) + "_" + tag;
    auto fd_bytes = store->get(peer_key + "_fd");
    auto pid_bytes = store->get(peer_key + "_pid");
    int peer_fd = *reinterpret_cast<const int*>(fd_bytes.data());
    pid_t peer_pid = *reinterpret_cast<const pid_t*>(pid_bytes.data());

    int pid_fd = syscall(SYS_pidfd_open, peer_pid, /*flags=*/0);
    NVF_CHECK(pid_fd >= 0, "pidfd_open failed for rank ", peer);

    int local_fd = syscall(SYS_pidfd_getfd, pid_fd, peer_fd, /*flags=*/0);
    NVF_CHECK(local_fd >= 0, "pidfd_getfd failed for rank ", peer);

    CUmemGenericAllocationHandle peer_handle;
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &peer_handle,
        reinterpret_cast<void*>(static_cast<uint64_t>(local_fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    alloc_handles_[peer] = peer_handle;

    CUdeviceptr peer_ptr = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
        &peer_ptr, aligned_size_, granularity_, 0, 0));
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(peer_ptr, aligned_size_, 0, peer_handle, 0));

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = static_cast<int>(comm.local_rank());
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(peer_ptr, aligned_size_, &access, 1));

    remote_ptrs_[peer] = peer_ptr;
    close(local_fd);
    close(pid_fd);
  }

  comm.barrier();
  are_remote_tensors_setup_ = true;
}

at::Tensor SymmetricTensor::remoteTensor(int64_t rank) const {
  NVF_CHECK(rank >= 0 && rank < world_size_, "Rank out of range");

  if (rank == my_device_id_) {
    return local_tensor_;
  }

  NVF_CHECK(are_remote_tensors_setup_ == true, "Remote tensors not setup");
  return at::from_blob(
      reinterpret_cast<void*>(remote_ptrs_[rank]),
      local_tensor_.sizes(),
      local_tensor_.strides(),
      at::TensorOptions().dtype(local_tensor_.scalar_type()).device(at::kCUDA, rank));
}


void* SymmetricTensor::multicastPtr() const {
  NVF_CHECK(is_multicast_setup_, "Multicast not setup");
  return mc_ptr_;
}

void SymmetricTensor::setupMulticast(
    int64_t exporter_rank,
    const std::string& tag) {
#if (CUDA_VERSION >= 13000)
  if (is_multicast_setup_) {
    return;
  }

  Communicator& comm = Communicator::getInstance();
  const int64_t my_rank = comm.deviceId();
  const int64_t local_rank = comm.local_rank();

  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, local_rank));
  NVF_CHECK(is_multicast_supported, "Multicast not supported");

  exporter_rank_ = exporter_rank;

  CUmulticastObjectProp mcast_prop{};
  mcast_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mcast_prop.numDevices = world_size_;
  mcast_prop.size = aligned_size_;

  int shared_handle;
  auto store = comm.getTcpStore();
  pid_t root_pid;

  if (my_rank == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle_, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    root_pid = getpid();

    std::vector<uint8_t> fd_bytes(
        reinterpret_cast<const uint8_t*>(&shared_handle),
        reinterpret_cast<const uint8_t*>(&shared_handle) + sizeof(int));
    std::vector<uint8_t> pid_bytes(
        reinterpret_cast<const uint8_t*>(&root_pid),
        reinterpret_cast<const uint8_t*>(&root_pid) + sizeof(pid_t));

    store->set(tag + "_fd", fd_bytes);
    store->set(tag + "_pid", pid_bytes);
  }

  comm.barrier();

  if (my_rank != exporter_rank) {
    auto fd_bytes = store->get(tag + "_fd");
    shared_handle = *reinterpret_cast<const int*>(fd_bytes.data());
    auto pid_bytes = store->get(tag + "_pid");
    root_pid = *reinterpret_cast<const pid_t*>(pid_bytes.data());

    pid_fd_ = syscall(SYS_pidfd_open, root_pid, 0);
    NVF_CHECK(pid_fd_ >= 0, "pidfd_open failed");
    peer_fd_ = syscall(SYS_pidfd_getfd, pid_fd_, shared_handle, 0);
    NVF_CHECK(peer_fd_ >= 0, "pidfd_getfd failed");

    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &mcast_handle_,
        reinterpret_cast<void*>(static_cast<uint64_t>(peer_fd_)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  }

  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev_, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle_, cu_dev_));

  CUdeviceptr local_ptr = remote_ptrs_[my_device_id_];
  CUdeviceptr base_ptr;
  size_t base_size;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &base_size, local_ptr));
  size_t mem_offset = static_cast<size_t>(local_ptr - base_ptr);
  
  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle_, 0, alloc_handles_[my_device_id_], mem_offset, aligned_size_, 0));

  CUdeviceptr mc_ptr;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(&mc_ptr, aligned_size_, granularity_, 0, 0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(mc_ptr, aligned_size_, 0, mcast_handle_, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(local_rank);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(mc_ptr, aligned_size_, &access, 1));

  mc_ptr_ = reinterpret_cast<void*>(mc_ptr);
  is_multicast_setup_ = true;

  comm.barrier();

  if (my_rank == exporter_rank) {
    store->deleteKey(tag + "_fd");
    store->deleteKey(tag + "_pid");
  }
#else
  (void)exporter_rank;
  (void)tag;
  NVF_ERROR("Multicast requires CUDA 13.0+");
#endif
}

at::Tensor createContiguousView(
    const SymmetricTensor& sym_tensor,
    const std::string& tag) {
  Communicator& comm = Communicator::getInstance();
  const int64_t local_rank = comm.local_rank();
  const int64_t world_size = comm.size();
  const size_t aligned_size = sym_tensor.alignedSize();
  const size_t granularity = sym_tensor.granularity();
  const size_t actual_size = sym_tensor.localTensor().numel() * sym_tensor.localTensor().element_size();

  NVF_CHECK(
      aligned_size == actual_size,
      "Requires aligned_size == actual_size. Got ",
      aligned_size,
      " vs ",
      actual_size);

  size_t total_size = actual_size * world_size;
  CUdeviceptr base;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(&base, total_size, granularity, 0, 0));

  for (int64_t rank = 0; rank < world_size; ++rank) {
    CUdeviceptr region = base + (rank * actual_size);
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(region, actual_size, 0, sym_tensor.getAllocHandle(rank, tag), 0));
    
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = static_cast<int>(local_rank);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(region, actual_size, &access, 1));
  }

  std::vector<int64_t> sizes = {world_size};
  for (int64_t s : sym_tensor.localTensor().sizes()) {
    sizes.push_back(s);
  }

  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int64_t i = sizes.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  return at::from_blob(
      reinterpret_cast<void*>(base),
      sizes,
      strides,
      [=](void* ptr) {
        for (int64_t rank = 0; rank < world_size; ++rank) {
          cuMemUnmap(reinterpret_cast<CUdeviceptr>(ptr) + (rank * actual_size), actual_size);
        }
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), total_size);
      },
      at::TensorOptions().dtype(sym_tensor.localTensor().scalar_type()).device(at::kCUDA, 0));
}

at::Tensor allocateSymmetricTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device) {
  // Query support for Virtual Memory Management
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      device.index()));
  NVF_ERROR(
      is_vmm_supported != 0,
      "Device does not support Virtual Memory Management");

  const int64_t numel = std::accumulate(
      sizes.begin(), sizes.end(), /*init=*/1, std::multiplies<int64_t>());
  const int64_t element_size = c10::elementSize(dtype);
  const int64_t alloc_size = numel * element_size;

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device.index());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = getGranularityForSymmetricMemory(prop, static_cast<size_t>(alloc_size));

  // Round up alloc_size to the nearest multiple of granularity
  int64_t rounded_alloc_size =
      ((alloc_size + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemCreate(&alloc_handle, rounded_alloc_size, &prop, /*flags=*/0));

  CUdeviceptr ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &ptr,
      rounded_alloc_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(
      ptr, rounded_alloc_size, /*offset=*/0, alloc_handle, /*flags=*/0));
  CUmemAccessDesc access_desc{};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = static_cast<int>(device.index());
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(ptr, rounded_alloc_size, &access_desc, /*count=*/1));

  auto options = at::TensorOptions().dtype(dtype).device(device);
  // Compute default (contiguous) strides for the given sizes
  std::vector<int64_t> strides(sizes.size());
  strides.back() = 1;
  for (int64_t i = strides.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return at::from_blob(
      (void*)ptr,
      sizes,
      std::move(strides),
      [=](void* ptr) {
        NVFUSER_CUDA_SAFE_CALL(
            cuMemUnmap((CUdeviceptr)(ptr), rounded_alloc_size));
        NVFUSER_CUDA_SAFE_CALL(
            cuMemAddressFree((CUdeviceptr)(ptr), rounded_alloc_size));
        NVFUSER_CUDA_SAFE_CALL(cuMemRelease(alloc_handle));
      },
      options);
}

std::string isSymmetricAllocationValid(at::Tensor tensor) {
  // Query support for Virtual Memory Management
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      tensor.device().index()));
  if (is_vmm_supported == 0) {
    return "Tensor device " + tensor.device().str() +
        " does not support Virtual Memory Management";
  }

  auto ptr = (CUdeviceptr)tensor.data_ptr();

  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = Communicator::getInstance().local_rank();
  unsigned long long flags = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAccess(&flags, &location, ptr));
  if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
    return "Expected symmetric memory access flags to be "
           "CU_MEM_ACCESS_FLAGS_PROT_READWRITE, but got " +
        std::to_string(flags);
  }

  CUmemGenericAllocationHandle alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&alloc_handle, (void*)ptr));

  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle));
  if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED) {
    return "Expected symmetric allocation to be of type "
           "CU_MEM_ALLOCATION_TYPE_PINNED, got " +
        std::to_string(prop.type);
  }
  if (prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    return "Expected symmetric allocation to be on device memory, got "
           "location.type = " +
        std::to_string(prop.location.type);
  }
  if (prop.location.id != Communicator::getInstance().local_rank()) {
    return "Expected symmetric allocation to be on device " +
        std::to_string(Communicator::getInstance().local_rank()) +
        " got location.id = " + std::to_string(prop.location.id);
  }
  if (prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return "Expected requestedHandleTypes = "
           "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, got " +
        std::to_string(prop.requestedHandleTypes);
  }

  size_t size_bytes = tensor.numel() * tensor.element_size();
  const size_t granularity = getGranularityForSymmetricMemory(prop, size_bytes);

  if ((static_cast<size_t>(ptr) % granularity) != 0) {
    return "Expected symmetric memory address to be aligned to granularity " +
        std::to_string(granularity) + ", got address " +
        std::to_string(static_cast<unsigned long long>(ptr));
  }

  // Check virtual address alignment and mapping size w.r.t. granularity
  CUdeviceptr base_ptr = 0;
  size_t va_size = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &va_size, ptr));

  if ((static_cast<size_t>(base_ptr) % granularity) != 0) {
    return "Expected symmetric memory address to be aligned to granularity " +
        std::to_string(granularity) + ", got address " +
        std::to_string(static_cast<unsigned long long>(base_ptr));
  }

  if ((va_size % granularity) != 0) {
    return "Expected symmetric memory size to be a multiple of granularity " +
        std::to_string(granularity) + ", got size " + std::to_string(va_size);
  }

  return ""; // Memory is valid
}

} // namespace nvfuser

