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

#include <numeric>

namespace nvfuser {

at::Tensor empty_strided_cuda_symmetric(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    at::ScalarType dtype,
    at::Device device,
    std::optional<uint64_t> alloc_id) {
  if (alloc_id.has_value()) {
    NVF_ERROR("Persistent symmetric memory allocation is not yet supported");
  }
  // Only support contiguous tensors for now
  int64_t expected_stride = 1;
  for (int64_t i = sizes.size() - 1; i >= 0; --i) {
    if (strides[i] != expected_stride) {
      NVF_ERROR(
          false,
          "empty_strided_cuda_symmetric only supports contiguous tensors, but got strides ",
          strides,
          " for size ",
          sizes);
    }
    expected_stride *= sizes[i];
  }

  const int64_t numel = std::accumulate(
      sizes.begin(), sizes.end(), /*init=*/1, std::multiplies<int64_t>());
  const int64_t element_size = c10::elementSize(dtype);
  const int64_t alloc_size = numel * element_size;

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device.index());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

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
  return at::from_blob(
      (void*)ptr,
      sizes,
      strides,
      [=](void* ptr) {
        NVFUSER_CUDA_SAFE_CALL(
            cuMemUnmap((CUdeviceptr)(ptr), rounded_alloc_size));
        NVFUSER_CUDA_SAFE_CALL(
            cuMemAddressFree((CUdeviceptr)(ptr), rounded_alloc_size));
        NVFUSER_CUDA_SAFE_CALL(cuMemRelease(alloc_handle));
      },
      options);
}

std::string is_symmetric_memory_valid(at::Tensor tensor) {
  auto ptr = (CUdeviceptr)tensor.data_ptr();

  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = 0;
  unsigned long long flags = 0;
  CUresult result = cuMemGetAccess(&flags, &location, ptr);
  if (result != CUDA_SUCCESS) {
    return "cuMemGetAccess failed with error code " + std::to_string(result);
  }
  if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
    return "Expected symmetric memory access flags to be CU_MEM_ACCESS_FLAGS_PROT_READWRITE, but got " +
        std::to_string(flags);
  }

  CUmemGenericAllocationHandle alloc_handle = 0;
  result = cuMemRetainAllocationHandle(&alloc_handle, (void*)ptr);
  if (result != CUDA_SUCCESS) {
    return "cuMemRetainAllocationHandle failed with error code " +
        std::to_string(result);
  }

  CUmemAllocationProp prop{};
  result = cuMemGetAllocationPropertiesFromHandle(&prop, alloc_handle);
  if (result != CUDA_SUCCESS) {
    return "cuMemGetAllocationPropertiesFromHandle failed with error code " +
        std::to_string(result);
  }
  if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED) {
    return "Expected symmetric allocation to be of type CU_MEM_ALLOCATION_TYPE_PINNED, got " +
        std::to_string(prop.type);
  }
  if (prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    return "Expected symmetric allocation to be on device memory, got location.type = " +
        std::to_string(prop.location.type);
  }
  if (prop.location.id != Communicator::getInstance().local_rank()) {
    return "Expected symmetric allocation to be on device " +
        std::to_string(Communicator::getInstance().local_rank()) +
        " got location.id = " + std::to_string(prop.location.id);
  }
  if (prop.requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return "Expected requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, got " +
        std::to_string(prop.requestedHandleTypes);
  }

  return ""; // Memory is valid
}

} // namespace nvfuser


