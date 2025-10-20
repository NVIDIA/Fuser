// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cuda_utils.h>
#include <multidevice/cuda_p2p.h>
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

void postAllgatherWithP2pBackend(
    Communication* communication,
    Communicator& communicator,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  // const int64_t my_device_index = communicator->deviceId();
  // const int64_t local_rank = communicator.local_rank();
  // const int64_t world_size = communicator->size();

  // // Multicast parameters
  // constexpr size_t kNumElems = input_tensor.numel();
  // constexpr size_t kSizeBytes = kNumElems * input_tensor.element_size();
  // constexpr size_t kTotalSizeBytes = kSizeBytes * world_size;

  // constexpr int64_t exporter_rank = 0;

  // // Query support for IPC handles
  // int is_ipc_supported;
  // NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
  //     &is_ipc_supported,
  //     CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
  //     local_rank));
  // NVF_ERROR(is_ipc_supported != 0, "Device does not support IPC handles");

  // // Query support for Multicast Objects
  // int is_multicast_supported;
  // NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
  //     &is_multicast_supported,
  //     CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
  //     local_rank));
  // NVF_ERROR(
  //     is_multicast_supported != 0, "Device does not support Multicast
  //     Objects");

#define use_file_descriptor \
  true // Set to false to use fabric handles, required for multinode NVLS
#if use_file_descriptor
  // using handle_typename = int;
  // auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
  using handle_typename = CUmemFabricHandle;
  auto handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
#endif

  //   // Query Multicast granularity
  //   CUmulticastObjectProp mcast_prop{};
  //   mcast_prop.flags = 0;
  //   mcast_prop.handleTypes = handle_type;
  //   mcast_prop.numDevices = world_size;
  //   mcast_prop.size = kSizeBytes;

  //   size_t mcast_min_granularity = 0;
  //   NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
  //       &mcast_min_granularity, &mcast_prop,
  //       CU_MULTICAST_GRANULARITY_MINIMUM));
  //   if (mcast_min_granularity > kSizeBytes) {
  //     GTEST_SKIP() << "Device does not support the required multicast "
  //                     "granularity; skipping."
  //                  << "Minimum Granularity: " << mcast_min_granularity
  //                  << ", required: " << kSizeBytes;
  //   }

  //   size_t mcast_granularity = 0;
  //   NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
  //       &mcast_granularity, &mcast_prop,
  //       CU_MULTICAST_GRANULARITY_RECOMMENDED));
  //   if (mcast_granularity > kSizeBytes) {
  //     GTEST_SKIP() << "Device does not recommend the required multicast "
  //                     "granularity; skipping."
  //                  << "Recommended Granularity: " << mcast_granularity
  //                  << ", required: " << kSizeBytes;
  //   }

  //   // Create a multicast object at root my_device_index and export it to a
  //   // shareable mem handled. Put it in the store
  //   CUmemGenericAllocationHandle mcast_handle{};
  //   auto store = communicator_->getTcpStore();
  //   pid_t root_pid;
  //   handle_typename shared_handle;
  //   if (my_device_index == exporter_rank) {
  //     NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle, &mcast_prop));
  //     NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
  //         &shared_handle, mcast_handle, handle_type, /*flags=*/0));
  //     // Allow peer processes to use pidfd_getfd on this process
  //     // A more aggressive solution would be to modify Yama ptrace policy by
  //     // running `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` or to
  //     run
  //     // the docker container with `--cap-add=SYS_PTRACE` and `--sysctl
  //     // kernel.yama.ptrace_scope=0`.
  //     prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
  //     store->set(
  //         std::string("nvls_export_fd_mcast_handle"),
  //         toBytes(shared_handle));
  //     root_pid = getpid();
  //     store->set(std::string("nvls_export_pid_mcast_handle"),
  //     toBytes(root_pid));
  //   }
  //   communicator_->barrier();
  //   // Import the multicast object at other ranks
  //   if (my_device_index != exporter_rank) {
  //     shared_handle = fromBytes<handle_typename>(
  //         store->get(std::string("nvls_export_fd_mcast_handle")));
  //     root_pid = fromBytes<pid_t>(
  //         store->get(std::string("nvls_export_pid_mcast_handle")));

  //     int pid_fd, peer_fd;
  //     pid_fd = syscall(SYS_pidfd_open, root_pid, /*flags=*/0);
  //     ASSERT_GE(pid_fd, 0) << "my_device_index " << my_device_index
  //                          << " failed to open pidfd for pid " << root_pid;

  //     peer_fd = syscall(SYS_pidfd_getfd, pid_fd, shared_handle, /*flags=*/0);
  //     ASSERT_GE(peer_fd, 0) << "my_device_index " << my_device_index
  //                           << " failed to get peer fd";
  //     NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
  //         &mcast_handle, (void*)((uint64_t)peer_fd), handle_type));
  //     close(pid_fd);
  //   }

  //   // All ranks add their device to multicast group
  //   CUdevice cu_dev;
  //   NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev,
  //   static_cast<int>(local_rank)));
  //   NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle, cu_dev));

  //   // From the docs
  //   //
  //   https://docs.nvidia.com/cuda/cuda-c-programming-guide/#add-devices-to-multicast-objects,
  //   // we need to ensure all devices are added to the multicast group before
  //   // binding memory, so we need a barrier here. However, it seems that
  //   // cuMulticastAddDevice already blocks, so the barrier would be
  //   redundant.
  //   // communicator_->barrier(); TODO: needed ?

  //   // Local memory allocation using Virtual Memory Management
  //   // All ranks prepare their per-device allocation handle
  //   CUmemAllocationProp prop{};
  //   prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  //   prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  //   prop.location.id = static_cast<int>(local_rank);
  //   prop.requestedHandleTypes = handle_type; // any shareable type

  //   size_t granularity = 0;
  //   NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
  //       &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  //   if (granularity > kSizeBytes) {
  //     GTEST_SKIP() << "Device does not support the required allocation "
  //                     "granularity; skipping."
  //                  << "Granularity: " << granularity
  //                  << ", required: " << kSizeBytes;
  //   }

  //   CUmemGenericAllocationHandle local_buffer = 0;
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMemCreate(&local_buffer, kSizeBytes, &prop, /*flags=*/0));

  //   // Bind the local memory to the multicast object
  //   NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
  //       mcast_handle,
  //       /*mcOffset=*/0,
  //       local_buffer,
  //       /*memOffset=*/0,
  //       kSizeBytes,
  //       /*flags=*/0));

  //   // MC Mapping
  //   CUdeviceptr mc_ptr = 0;
  //   NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
  //       &mc_ptr,
  //       kSizeBytes,
  //       /*alignment=*/mcast_granularity,
  //       /*baseVA=*/0,
  //       /*flags=*/0));
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMemMap(mc_ptr, kSizeBytes, /*offset=*/0, mcast_handle,
  //       /*flags=*/0));
  //   CUmemAccessDesc mc_mapping_desc{};
  //   mc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  //   mc_mapping_desc.location.id = static_cast<int>(local_rank);
  //   mc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMemSetAccess(mc_ptr, kSizeBytes, &mc_mapping_desc, /*count=*/1));

  //   // UC Mapping
  //   CUdeviceptr uc_ptr = 0;
  //   NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
  //       &uc_ptr,
  //       kSizeBytes,
  //       /*alignment=*/granularity,
  //       /*baseVA=*/0,
  //       /*flags=*/0));
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMemMap(uc_ptr, kSizeBytes, /*offset=*/0, local_buffer,
  //       /*flags=*/0));
  //   CUmemAccessDesc uc_mapping_desc{};
  //   uc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  //   uc_mapping_desc.location.id = static_cast<int>(local_rank);
  //   uc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMemSetAccess(uc_ptr, kSizeBytes, &uc_mapping_desc, /*count=*/1));

  //   // Each my_device_index now has a UC address and a MC address associated
  //   with
  //   // a local buffer. The typical and recommended use case is write to the
  //   mc
  //   // address from the root my_device_index, which will broadcast the data,
  //   after
  //   // which each my_device_index can read from its uc address.

  //   // Root my_device_index writes to mc address
  //   std::vector<uint32_t> host_buffer(kNumElems);
  //   if (my_device_index == root_rank) {
  //     for (size_t i = 0; i < kNumElems; ++i) {
  //       host_buffer[i] = static_cast<uint32_t>(i * 5 + 11);
  //     }
  //     NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
  //         reinterpret_cast<void*>(mc_ptr),
  //         host_buffer.data(),
  //         kSizeBytes,
  //         cudaMemcpyHostToDevice));
  //   }

  //   communicator_->barrier();

  //   // Each my_device_index reads from the UC address
  //   NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
  //       host_buffer.data(),
  //       reinterpret_cast<void*>(uc_ptr),
  //       kSizeBytes,
  //       cudaMemcpyDeviceToHost));

  //   for (size_t i = 0; i < kNumElems; ++i) {
  //     EXPECT_EQ(host_buffer[i], static_cast<uint32_t>(i * 5 + 11));
  //   }

  //   // reading from mc address is an undefined behavior

  //   // Cleanup
  //   NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(mc_ptr, kSizeBytes));
  //   NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(uc_ptr, kSizeBytes));
  //   NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(mc_ptr, kSizeBytes));
  //   NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(uc_ptr, kSizeBytes));
  //   NVFUSER_CUDA_SAFE_CALL(cuMemRelease(local_buffer));
  //   NVFUSER_CUDA_SAFE_CALL(
  //       cuMulticastUnbind(mcast_handle, cu_dev, /*offset=*/0, kSizeBytes));
  //   NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mcast_handle));
}

#endif // CUDA_VERSION >= NVF_MIN_CUDA_FOR_MCAST

} // namespace nvfuser
