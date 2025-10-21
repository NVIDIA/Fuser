// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>

#include <cuda_utils.h>
#include <multidevice/cuda_p2p.h>
#include <multidevice/symmetric_memory.h>
#include "multidevice/communicator.h"

namespace nvfuser {

namespace get_zcopy {

void recvPost(const P2pIpcHandle& ipc_handles, int64_t count, CUstream stream) {
  // wait for sender to be ready
  NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WAIT_VALUE_EQ));
  // RDMA get the data from the sender
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpyAsync(
      ipc_handles.local().ptr(),
      ipc_handles.peer().ptr(),
      count,
      cudaMemcpyDeviceToDevice,
      stream));
  // Signals completion to self
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WRITE_VALUE_DEFAULT));
  // Signals completion to sender
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.peer().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WRITE_VALUE_DEFAULT));
}

void sendPost(const P2pIpcHandle& ipc_handles, CUstream stream) {
  // signal to self that transfer is in progress
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WRITE_VALUE_DEFAULT));
  // signal to receiver that the buffer is ready
  NVFUSER_CUDA_SAFE_CALL(cuStreamWriteValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.peer().semaphore()),
      (cuuint32_t)(IpcSemaphore::kInUse),
      CU_STREAM_WRITE_VALUE_DEFAULT)); // passing
                                       // CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
                                       // gives an error
}

void sendWait(const P2pIpcHandle& ipc_handles, CUstream stream) {
  NVFUSER_CUDA_SAFE_CALL(cuStreamWaitValue32(
      stream,
      reinterpret_cast<CUdeviceptr>(ipc_handles.local().semaphore()),
      (cuuint32_t)(IpcSemaphore::kReady),
      CU_STREAM_WAIT_VALUE_EQ));
}

} // namespace get_zcopy

#if (CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST)

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

void postBroadcastWithP2pBackend(
    Communication* communication,
    Communicator* communicator,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  NVF_ERROR(
      communication->type() == CommunicationType::Broadcast,
      "Invalid communication type, expected Broadcast, got: ",
      communication->type());

  const int64_t my_device_index = communicator->deviceId();
  const int64_t local_rank = communicator->local_rank();
  const int64_t world_size = communicator->size();

  // Multicast parameters
  const size_t kNumElems = output_tensor.numel();
  const size_t kSizeBytes = kNumElems * output_tensor.element_size();

  const int64_t exporter_rank = communication->root();

  // Query support for IPC handles
  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  NVF_ERROR(is_ipc_supported != 0, "Device does not support IPC handles");

  // Query support for Multicast Objects
  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  NVF_ERROR(
      is_multicast_supported != 0, "Device does not support Multicast Objects");

#define use_file_descriptor \
  true // Set to false to use fabric handles, required for multinode NVLS
#if use_file_descriptor
  using handle_typename = int;
  auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
  using handle_typename = CUmemFabricHandle;
  auto handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
#endif

  // CUmemGenericAllocationHandle input_alloc_handle = 0, output_alloc_handle =
  // 0;
  std::string error_message = is_symmetric_memory_valid(output_tensor);
  NVF_ERROR(error_message.empty(), error_message);
  CUmemGenericAllocationHandle output_alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemRetainAllocationHandle(
      &output_alloc_handle, (void*)output_tensor.data_ptr()));
  // if (my_device_index == communication->root()) {
  //   error_message = is_symmetric_memory_valid(input_tensor);
  //   NVF_ERROR(error_message.empty(), error_message);
  //   NVFUSER_CUDA_SAFE_CALL(cuMemRetainAllocationHandle(
  //       &input_alloc_handle, (void*)input_tensor.data_ptr()));
  // }
  CUmemAllocationProp prop{};
  NVFUSER_CUDA_SAFE_CALL(
      cuMemGetAllocationPropertiesFromHandle(&prop, output_alloc_handle));

  // Create a multicast object at root my_device_index and export it to a
  // shareable mem handled. Put it in the store

  int64_t granularity = get_granularity(prop, kSizeBytes);
  int64_t rounded_size =
      ((kSizeBytes + granularity - 1) / granularity) * granularity;
  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = handle_type;
  mcast_prop.numDevices = world_size;
  mcast_prop.size = rounded_size;

  CUmemGenericAllocationHandle mcast_handle{};
  auto store = communicator->getTcpStore();
  pid_t root_pid;
  handle_typename shared_handle;
  if (my_device_index == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle, handle_type, /*flags=*/0));
    // Allow peer processes to use pidfd_getfd on this process

    int status = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    NVF_ERROR(status >= 0, "Failed to set ptrace policy");
    store->set(
        std::string("nvls_export_fd_mcast_handle"), toBytes(shared_handle));
    root_pid = getpid();
    store->set(std::string("nvls_export_pid_mcast_handle"), toBytes(root_pid));
  }

  communicator->barrier();

  // Import the multicast object at other ranks
  if (my_device_index != exporter_rank) {
    shared_handle = fromBytes<handle_typename>(
        store->get(std::string("nvls_export_fd_mcast_handle")));
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

  // All ranks add their device to multicast group
  CUdevice cu_dev;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle, cu_dev));

  // From the docs
  //
  // https: //
  // docs.nvidia.com/cuda/cuda-c-programming-guide/#add-devices-to-multicast-objects,
  // we need to ensure all devices are added to the multicast group before
  // binding memory, so we need a barrier here. However, it seems that
  // cuMulticastAddDevice already blocks, so the barrier would be
  // redundant.
  // communicator->barrier(); TODO: needed ?

  // // Local memory allocation using Virtual Memory Management
  // // All ranks prepare their per-device allocation handle
  // CUmemAllocationProp prop{};
  // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // prop.location.id = static_cast<int>(local_rank);
  // prop.requestedHandleTypes = handle_type;

  // Bind the local memory to the multicast object
  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle,
      /*mcOffset=*/0,
      output_alloc_handle,
      /*memOffset=*/0,
      rounded_size,
      /*flags=*/0));

  // MC Mapping
  CUdeviceptr mc_ptr = 0;
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

  if (my_device_index == communication->root()) {
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        reinterpret_cast<void*>(mc_ptr),
        input_tensor.data_ptr(),
        kSizeBytes,
        cudaMemcpyHostToDevice));
  }

  communicator->barrier();

  // Cleanup
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(mc_ptr, rounded_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(mc_ptr, rounded_size));
  // if (my_device_index == communication->root()) {
  //   NVFUSER_CUDA_SAFE_CALL(cuMemRelease(input_alloc_handle));
  // }
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(output_alloc_handle));
  NVFUSER_CUDA_SAFE_CALL(
      cuMulticastUnbind(mcast_handle, cu_dev, /*offset=*/0, rounded_size));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRelease(mcast_handle)); // only needed at exporter ?
}

#endif // CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST

} // namespace nvfuser
