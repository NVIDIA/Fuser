// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <multidevice/symmetric_memory.h>

#include <cuda_utils.h>
#include <multidevice/communicator.h>

namespace nvfuser {

// Returns the minimum between the allocation granularity and, when available,
// the maximum of multicast minimum and recommended granularities.
int64_t getGranularityForSymmetricMemory(
    const CUmemAllocationProp& prop,
    size_t requested_size_bytes,
    bool get_recommended_granularity) {
  size_t alloc_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &alloc_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)
  size_t mcast_min_granularity = 0;
  size_t mcast_rec_granularity = 0;

  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  mcast_prop.numDevices = Communicator::getInstance().size();
  mcast_prop.size = requested_size_bytes; // is it needed?

  NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
      &mcast_min_granularity, &mcast_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  if (get_recommended_granularity) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
        &mcast_rec_granularity,
        &mcast_prop,
        CU_MULTICAST_GRANULARITY_RECOMMENDED));
  }

  size_t granularity =
      get_recommended_granularity ? mcast_rec_granularity : mcast_min_granularity;

  return std::max(alloc_granularity, granularity);
#else
  return alloc_granularity;
#endif
}

at::Tensor allocateSymmetricTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device,
    std::optional<uint64_t> alloc_id) {
  if (alloc_id.has_value()) {
    NVF_ERROR("Persistent symmetric memory allocation is not yet supported");
  }

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
