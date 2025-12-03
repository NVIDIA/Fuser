// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/symmetric_tensor.h>

#include <cuda_utils.h>
#include <driver_api.h>
#include <multidevice/communicator.h>
#include <multidevice/utils.h>

namespace nvfuser {

namespace {

// Returns the allocation granularity for symmetric memory.
// - query_mcast_granularity: if true, considers multicast granularity
// - query_mcast_recommended_granularity: if true, uses recommended (larger)
// multicast granularity
int64_t getGranularityForSymmetricMemory(
    const CUmemAllocationProp& prop,
    size_t requested_size_bytes,
    bool query_mcast_granularity = true,
    bool query_mcast_recommended_granularity = false) {
  size_t alloc_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &alloc_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  if (!query_mcast_granularity) {
    return alloc_granularity;
  }

  // Check if device supports multicast before querying multicast granularity
  int is_multicast_supported = 0;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      prop.location.id));

  if (is_multicast_supported == 0) {
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

  if (query_mcast_recommended_granularity) {
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
  (void)query_mcast_granularity;
  (void)query_mcast_recommended_granularity;
  return alloc_granularity;
#endif
}

} // namespace

at::Tensor SymmetricTensor::allocate(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device) {
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      device.index()));
  NVF_ERROR(is_vmm_supported, "Device does not support VMM");

  const int64_t numel = std::accumulate(
      sizes.begin(), sizes.end(), 1, std::multiplies<int64_t>());
  const int64_t element_size = c10::elementSize(dtype);
  const int64_t alloc_size = numel * element_size;

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device.index());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity =
      getGranularityForSymmetricMemory(prop, static_cast<size_t>(alloc_size));
  int64_t rounded_size =
      ((alloc_size + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemCreate(&handle, rounded_size, &prop, 0));

  CUdeviceptr ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemAddressReserve(&ptr, rounded_size, granularity, 0, 0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(ptr, rounded_size, 0, handle, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(device.index());
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(ptr, rounded_size, &access, 1));

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
        cuMemUnmap((CUdeviceptr)(ptr), rounded_size);
        cuMemAddressFree((CUdeviceptr)(ptr), rounded_size);
        cuMemRelease(handle);
      },
      at::TensorOptions().dtype(dtype).device(device));
}

std::string SymmetricTensor::validate(at::Tensor tensor) {
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      tensor.device().index()));
  if (!is_vmm_supported) {
    return "Device does not support VMM";
  }

  auto ptr = (CUdeviceptr)tensor.data_ptr();

  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = Communicator::getInstance().local_rank();
  unsigned long long flags = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAccess(&flags, &location, ptr));
  if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
    return "Invalid access flags: " + std::to_string(flags);
  }

  CUmemGenericAllocationHandle alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&alloc_handle, (void*)ptr));

  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle));

  if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED) {
    return "Not pinned allocation";
  }
  if (prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    return "Not device memory";
  }
  if (prop.location.id != Communicator::getInstance().local_rank()) {
    return "Wrong device: " + std::to_string(prop.location.id);
  }
  if (prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return "Wrong handle type";
  }

  size_t size_bytes = tensor.numel() * tensor.element_size();
  const size_t granularity = getGranularityForSymmetricMemory(prop, size_bytes);

  CUdeviceptr base_ptr = 0;
  size_t va_size = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &va_size, ptr));

  if ((static_cast<size_t>(ptr) % granularity) != 0 ||
      (static_cast<size_t>(base_ptr) % granularity) != 0 ||
      (va_size % granularity) != 0) {
    return "Misaligned to granularity";
  }

  cuMemRelease(alloc_handle);
  return "";
}

SymmetricTensor::SymmetricTensor(const at::Tensor& local_tensor)
    : local_tensor_(local_tensor) {
  NVF_ERROR(
      local_tensor.is_cuda(),
      "Expected CUDA tensor, got: ",
      local_tensor.device());

  std::string error = SymmetricTensor::validate(local_tensor);
  NVF_CHECK(error.empty(), "Invalid symmetric allocation: ", error);

  Communicator& comm = Communicator::getInstance();
  world_size_ = comm.size();
  my_device_id_ = comm.deviceId();

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(comm.local_rank());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t required_size = local_tensor.numel() * local_tensor.element_size();
  granularity_ = getGranularityForSymmetricMemory(prop, required_size);
  aligned_size_ =
      ((required_size + granularity_ - 1) / granularity_) * granularity_;

  alloc_handles_.resize(world_size_);
  remote_ptrs_.resize(world_size_);

  CUdeviceptr local_ptr =
      reinterpret_cast<CUdeviceptr>(local_tensor_.data_ptr());
  CUmemGenericAllocationHandle local_handle;
  NVFUSER_CUDA_SAFE_CALL(cuMemRetainAllocationHandle(
      &local_handle, reinterpret_cast<void*>(local_ptr)));

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
      // On some driver versions, cuMulticastUnbind is sometimes failing with
      // CUDA_ERROR_INVALID_VALUE, as seen in CI (but not on my system)
      // According to docs, call to cuMulticastUnbind is not required, and
      // destroying the object unbinds it. Therefore, we simply skip the call
      // for now. cuMulticastUnbind(mcast_handle_, cu_dev_, 0, aligned_size_);
      cuMemRelease(mcast_handle_);
    }
    if (peer_fd_ >= 0)
      close(peer_fd_);
    if (pid_fd_ >= 0)
      close(pid_fd_);
  }
#endif

  if (are_remote_tensors_setup_ == true) {
    CUdeviceptr local_ptr =
        reinterpret_cast<CUdeviceptr>(local_tensor_.data_ptr());
    CUdeviceptr base_ptr = 0;
    size_t va_size = 0;
    NVFUSER_CUDA_SAFE_CALL(
        cuMemGetAddressRange(&base_ptr, &va_size, local_ptr));
    size_t offset = local_ptr - base_ptr;

    for (int64_t rank = 0; rank < world_size_; ++rank) {
      if (rank != my_device_id_ && remote_ptrs_[rank]) {
        cuMemUnmap(remote_ptrs_[rank] - offset, va_size);
        cuMemAddressFree(remote_ptrs_[rank] - offset, va_size);
      }
      if (rank != my_device_id_ && alloc_handles_[rank]) {
        cuMemRelease(alloc_handles_[rank]);
      }
    }
  }

  if (alloc_handles_.size() > static_cast<size_t>(my_device_id_) &&
      alloc_handles_[my_device_id_]) {
    cuMemRelease(alloc_handles_[my_device_id_]);
  }
}

void SymmetricTensor::setupRemoteHandles(const std::string& tag) const {
  if (are_remote_tensors_setup_ == true) {
    return;
  }
  Communicator& comm = Communicator::getInstance();
  CUmemGenericAllocationHandle local_handle = alloc_handles_[my_device_id_];
  CUdeviceptr local_ptr = remote_ptrs_[my_device_id_];

  CUdeviceptr base_ptr = 0;
  size_t va_size = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &va_size, local_ptr));
  size_t offset = local_ptr - base_ptr;

  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      local_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  std::string my_socket_path =
      "@nvfuser_sym_p2p_" + std::to_string(my_device_id_) + "_" + tag;
  int listener_fd = createIpcSocket(my_socket_path);

  comm.barrier();

  for (int64_t peer = 0; peer < world_size_; ++peer) {
    if (peer == my_device_id_) {
      continue;
    }
    std::string peer_path =
        "@nvfuser_sym_p2p_" + std::to_string(peer) + "_" + tag;
    int my_rank_data = (int)my_device_id_;
    sendFd(peer_path, shared_fd, &my_rank_data, sizeof(my_rank_data));
  }

  for (int64_t i = 0; i < world_size_ - 1; ++i) {
    int sender_rank = -1;
    int local_fd = recvFd(listener_fd, &sender_rank, sizeof(sender_rank));
    NVF_CHECK(
        sender_rank >= 0 && sender_rank < world_size_, "Invalid sender rank");

    CUmemGenericAllocationHandle peer_handle;
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &peer_handle,
        reinterpret_cast<void*>(static_cast<uint64_t>(local_fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    alloc_handles_[sender_rank] = peer_handle;

    CUdeviceptr peer_ptr = 0;
    NVFUSER_CUDA_SAFE_CALL(
        cuMemAddressReserve(&peer_ptr, va_size, granularity_, 0, 0));
    // cuMemMap does not support for now mapping a subregion of an allocation,
    // so we map the full allocation but store the offseted peer pointer.
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(peer_ptr, va_size, 0, peer_handle, 0));

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = static_cast<int>(comm.local_rank());
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(peer_ptr, va_size, &access, 1));

    remote_ptrs_[sender_rank] = peer_ptr + offset;
    close(local_fd);
  }

  close(listener_fd);
  close(shared_fd);

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
      at::TensorOptions()
          .dtype(local_tensor_.scalar_type())
          .device(at::kCUDA, rank));
}

void* SymmetricTensor::multicastPtr() const {
  NVF_CHECK(is_multicast_setup_, "Multicast not setup");
  return mc_ptr_;
}

void SymmetricTensor::setupContiguousView(const std::string& tag) {
  if (is_contiguous_view_setup_) {
    return;
  }

  Communicator& comm = Communicator::getInstance();
  const int64_t local_rank = comm.local_rank();
  const int64_t world_size = comm.size();
  const size_t actual_size =
      local_tensor_.numel() * local_tensor_.element_size();

  NVF_CHECK(
      aligned_size_ == actual_size, "Requires aligned_size == actual_size");

  size_t total_size = actual_size * world_size;
  CUdeviceptr base;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemAddressReserve(&base, total_size, granularity_, 0, 0));

  for (int64_t rank = 0; rank < world_size; ++rank) {
    CUdeviceptr region = base + (rank * actual_size);
    NVFUSER_CUDA_SAFE_CALL(
        cuMemMap(region, actual_size, 0, getAllocHandle(rank, tag), 0));

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = static_cast<int>(local_rank);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(region, actual_size, &access, 1));
  }

  std::vector<int64_t> sizes = {world_size};
  for (int64_t s : local_tensor_.sizes()) {
    sizes.push_back(s);
  }

  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int64_t i = sizes.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  contiguous_view_ = at::from_blob(
      reinterpret_cast<void*>(base),
      sizes,
      strides,
      [=](void* ptr) {
        for (int64_t rank = 0; rank < world_size; ++rank) {
          cuMemUnmap(
              reinterpret_cast<CUdeviceptr>(ptr) + (rank * actual_size),
              actual_size);
        }
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), total_size);
      },
      at::TensorOptions()
          .dtype(local_tensor_.scalar_type())
          .device(at::kCUDA, 0));

  is_contiguous_view_setup_ = true;
}

at::Tensor SymmetricTensor::getContiguousView() const {
  NVF_CHECK(is_contiguous_view_setup_, "Contiguous view not setup");
  return contiguous_view_;
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
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  NVF_CHECK(is_multicast_supported, "Multicast not supported");

  exporter_rank_ = exporter_rank;

  CUmulticastObjectProp mcast_prop{};
  mcast_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mcast_prop.numDevices = world_size_;
  mcast_prop.size = aligned_size_;

  int shared_handle_fd = -1;
  int listener_fd = -1;

  if (my_rank == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle_, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle_fd,
        mcast_handle_,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        0));
  } else {
    std::string my_path =
        "@nvfuser_sym_mcast_" + std::to_string(my_rank) + "_" + tag;
    listener_fd = createIpcSocket(my_path);
  }

  comm.barrier();

  if (my_rank != exporter_rank) {
    peer_fd_ = recvFd(listener_fd);
    close(listener_fd);

    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &mcast_handle_,
        reinterpret_cast<void*>(static_cast<uint64_t>(peer_fd_)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  } else {
    for (int i = 0; i < world_size_; ++i) {
      if (i == my_rank) {
        continue;
      }
      std::string peer_path =
          "@nvfuser_sym_mcast_" + std::to_string(i) + "_" + tag;
      sendFd(peer_path, shared_handle_fd);
    }
    close(shared_handle_fd);
  }

  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev_, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle_, cu_dev_));

  CUdeviceptr local_ptr = remote_ptrs_[my_device_id_];
  CUdeviceptr base_ptr;
  size_t base_size;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAddressRange(&base_ptr, &base_size, local_ptr));
  size_t mem_offset = static_cast<size_t>(local_ptr - base_ptr);

  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle_,
      0,
      alloc_handles_[my_device_id_],
      mem_offset,
      aligned_size_,
      0));

  CUdeviceptr mc_ptr;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemAddressReserve(&mc_ptr, aligned_size_, granularity_, 0, 0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(mc_ptr, aligned_size_, 0, mcast_handle_, 0));

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(local_rank);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(mc_ptr, aligned_size_, &access, 1));

  mc_ptr_ = reinterpret_cast<void*>(mc_ptr);
  is_multicast_setup_ = true;

  comm.barrier();
#else
  (void)exporter_rank;
  (void)tag;
  NVF_ERROR("Multicast requires CUDA 13.0+");
#endif
}

} // namespace nvfuser
