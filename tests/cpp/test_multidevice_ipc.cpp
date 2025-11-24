// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <fusion.h>
#include <host_ir/container.h>
#include <host_ir/evaluator.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>

#include <sys/prctl.h>
#include <sys/syscall.h>
#include <cerrno>
#include <cstring>

namespace nvfuser {

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

using IpcTest = MultiDeviceTest;

TEST_F(IpcTest, IpcMemHandle) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  // Allocate and setup GPU buffers
  constexpr size_t kBufferSize = sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));

  void* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));
  const int64_t value = rank;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, &value, kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));
  // As a convenience, we use the TCP store to exchange out-of-band the IPC
  // handle as raw data
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string((rank + 1) % num_devices)));
  void* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      &peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(&peer_value, peer_d_ptr, kBufferSize, cudaMemcpyDeviceToHost));
  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

TEST_F(IpcTest, IpcMemHandlePtrArithmeticAtReceiver) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  // TL;DR: We can do pointer arithmetic on the importer side. IOW, the pointer
  // can be used as a regular pointer on the importer side.

  // Allocate GPU memory. Set up a buffer with two int values.
  constexpr size_t kBufferSize = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));

  void* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));

  // Set up the buffer
  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, values.data(), kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr));
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));
  int64_t* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate, by reading the second value in the buffer (c.f. the "+1" offset)
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      &peer_value, peer_d_ptr + 1, kBufferSize / 2, cudaMemcpyDeviceToHost));
  EXPECT_EQ(2 * peer_rank + 1, peer_value);

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

TEST_F(IpcTest, IpcMemHandlePtrArithmeticAtSender) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  // TL;DR: We CANNOT do pointer arithmetic on the exporter side! The IPC handle
  // points to the beginning of the allocated buffer.

  // Allocate GPU memory. Set up a buffer with two int values.
  constexpr size_t kBufferSize = 2 * sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t peer_rank = (rank + 1) % num_devices;

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(rank));

  int64_t* d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMalloc(&d_ptr, kBufferSize));

  std::vector<int64_t> values;
  values.push_back(2 * rank);
  values.push_back(2 * rank + 1);
  NVFUSER_CUDA_RT_SAFE_CALL(
      cudaMemcpy(d_ptr, values.data(), kBufferSize, cudaMemcpyHostToDevice));

  // Export Ipc Handle
  cudaIpcMemHandle_t ipc_handle;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcGetMemHandle(&ipc_handle, d_ptr + 1));
  auto store = communicator_->getTcpStore();
  store->set("ipc_handle_" + std::to_string(rank), toBytes(ipc_handle));

  // Wait for all ranks to finish exporting the IPC handle
  communicator_->barrier();

  // Import Ipc Handle
  auto peer_ipc_handle = fromBytes<cudaIpcMemHandle_t>(
      store->get("ipc_handle_" + std::to_string(peer_rank)));
  int64_t* peer_d_ptr;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcOpenMemHandle(
      (void**)&peer_d_ptr, peer_ipc_handle, cudaIpcMemLazyEnablePeerAccess));

  // Validate, noticing that the pointer is not offset by 1, contrarily to the
  // offset used in the exporter side.
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      &peer_value, peer_d_ptr, kBufferSize / 2, cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      2 * peer_rank,
      peer_value); // and not 2 * peer_rank + 1 as could be expected!

  // Clean up
  NVFUSER_CUDA_RT_SAFE_CALL(cudaIpcCloseMemHandle(peer_d_ptr));
  NVFUSER_CUDA_RT_SAFE_CALL(cudaFree(d_ptr));
}

TEST_F(IpcTest, IpcP2pWithVmm) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  // Allocate and setup GPU buffers using VMM API
  constexpr size_t kBufferSize = sizeof(int64_t);
  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t local_rank = communicator_->local_rank();

  // Query support for Virtual Memory Management
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      local_rank));
  if (is_vmm_supported == 0) {
    GTEST_SKIP()
        << "Device does not support Virtual Memory Management; skipping.";
  }

  // Query support for IPC handles
  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  if (is_ipc_supported == 0) {
    GTEST_SKIP() << "Device does not support IPC handles; skipping.";
  }

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(local_rank));

  // Set up allocation properties for VMM
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(local_rank);
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // Check allocation granularity
  size_t granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // Align buffer size to granularity
  size_t aligned_size =
      ((kBufferSize + granularity - 1) / granularity) * granularity;

  // Allocate memory using VMM API
  CUmemGenericAllocationHandle mem_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemCreate(&mem_handle, aligned_size, &prop, /*flags=*/0));

  // Reserve virtual address space and map the memory
  CUdeviceptr d_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &d_ptr,
      aligned_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemMap(d_ptr, aligned_size, /*offset=*/0, mem_handle, /*flags=*/0));

  // Set memory access permissions
  CUmemAccessDesc access_desc{};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = static_cast<int>(local_rank);
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(d_ptr, aligned_size, &access_desc, /*count=*/1));

  // Write the rank value to the buffer
  const int64_t value = rank;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      reinterpret_cast<void*>(d_ptr),
      &value,
      kBufferSize,
      cudaMemcpyHostToDevice));

  // Export VMM handle to shareable file descriptor
  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      mem_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  // Allow peer processes to use pidfd_getfd on this process
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);

  // Share the file descriptor and PID via TCP store
  auto store = communicator_->getTcpStore();
  store->set("vmm_ipc_fd_" + std::to_string(rank), toBytes(shared_fd));
  pid_t my_pid = getpid();
  store->set("vmm_ipc_pid_" + std::to_string(rank), toBytes(my_pid));

  // Wait for all ranks to finish exporting
  communicator_->barrier();

  // Import peer's VMM handle
  const int64_t peer_rank = (rank + 1) % num_devices;
  int peer_shared_fd =
      fromBytes<int>(store->get("vmm_ipc_fd_" + std::to_string(peer_rank)));
  pid_t peer_pid =
      fromBytes<pid_t>(store->get("vmm_ipc_pid_" + std::to_string(peer_rank)));

  // Get the peer's file descriptor using pidfd_open and pidfd_getfd
  int pid_fd = syscall(SYS_pidfd_open, peer_pid, /*flags=*/0);
  ASSERT_GE(pid_fd, 0) << "rank " << rank << " failed to open pidfd for pid "
                       << peer_pid;

  int peer_fd = syscall(SYS_pidfd_getfd, pid_fd, peer_shared_fd, /*flags=*/0);
  ASSERT_GE(peer_fd, 0) << "rank " << rank
                        << " failed to get peer fd, errno: " << errno << " ("
                        << strerror(errno) << ")";

  // Import the peer's memory handle
  CUmemGenericAllocationHandle peer_mem_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
      &peer_mem_handle,
      (void*)((uint64_t)peer_fd),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

  // Reserve virtual address space and map the peer's memory
  CUdeviceptr peer_d_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &peer_d_ptr,
      aligned_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(cuMemMap(
      peer_d_ptr, aligned_size, /*offset=*/0, peer_mem_handle, /*flags=*/0));

  // Set memory access permissions for peer memory
  CUmemAccessDesc peer_access_desc{};
  peer_access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  peer_access_desc.location.id = static_cast<int>(local_rank);
  peer_access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(peer_d_ptr, aligned_size, &peer_access_desc, /*count=*/1));

  // Validate by reading peer's value
  int64_t peer_value;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      &peer_value,
      reinterpret_cast<void*>(peer_d_ptr),
      kBufferSize,
      cudaMemcpyDeviceToHost));
  EXPECT_EQ((value + 1) % num_devices, peer_value);

  // Clean up peer memory
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(peer_d_ptr, aligned_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(peer_d_ptr, aligned_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(peer_mem_handle));
  close(peer_fd);
  close(pid_fd);

  // Clean up local memory
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(d_ptr, aligned_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(d_ptr, aligned_size));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mem_handle));
}

#if (CUDA_VERSION >= 13000)

TEST_F(IpcTest, IpcNvlsMulticastBroadcast) {
  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t world_size = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t local_rank = communicator_->local_rank();

  constexpr int64_t exporter_rank = 0;
  constexpr int64_t root_rank = 1;

  // Multicast parameters
  constexpr size_t kNumElems = 524288;
  constexpr size_t kSizeBytes = kNumElems * sizeof(uint32_t);

  // Query support for Virtual Memory Management
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      local_rank));
  if (is_vmm_supported == 0) {
    GTEST_SKIP()
        << "Device does not support Virtual Memory Management; skipping.";
  }

  // Query support for IPC handles
  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  if (is_ipc_supported == 0) {
    GTEST_SKIP() << "Device does not support IPC handles; skipping.";
  }

  // Query support for Multicast Objects
  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  if (is_multicast_supported == 0) {
    GTEST_SKIP() << "Device does not support Multicast Objects; skipping.";
  }

#define use_file_descriptor \
  true // Set to false to use fabric handles, required for multinode NVLS
#if use_file_descriptor
  using handle_typename = int;
  auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
  using handle_typename = CUmemFabricHandle;
  auto handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
#endif

  // Query Multicast granularity
  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = handle_type;
  mcast_prop.numDevices = world_size;
  mcast_prop.size = kSizeBytes;

  size_t mcast_min_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
      &mcast_min_granularity, &mcast_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
  if (mcast_min_granularity > kSizeBytes) {
    GTEST_SKIP() << "Device does not support the required multicast "
                    "granularity; skipping."
                 << "Minimum Granularity: " << mcast_min_granularity
                 << ", required: " << kSizeBytes;
  }

  size_t mcast_granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMulticastGetGranularity(
      &mcast_granularity, &mcast_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  if (mcast_granularity > kSizeBytes) {
    GTEST_SKIP() << "Device does not recommend the required multicast "
                    "granularity; skipping."
                 << "Recommended Granularity: " << mcast_granularity
                 << ", required: " << kSizeBytes;
  }

  // Create a multicast object at root rank and export it to a shareable mem
  // handled. Put it in the store
  CUmemGenericAllocationHandle mcast_handle{};
  auto store = communicator_->getTcpStore();
  pid_t root_pid;
  handle_typename shared_handle;
  if (rank == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle, handle_type, /*flags=*/0));
    // Allow peer processes to use pidfd_getfd on this process
    // A more aggressive solution would be to modify Yama ptrace policy by
    // running `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` or to run
    // the docker container with `--cap-add=SYS_PTRACE` and `--sysctl
    // kernel.yama.ptrace_scope=0`.
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);
    store->set(
        std::string("nvls_export_fd_mcast_handle"), toBytes(shared_handle));
    root_pid = getpid();
    store->set(std::string("nvls_export_pid_mcast_handle"), toBytes(root_pid));
  }
  communicator_->barrier();
  // Import the multicast object at other ranks
  if (rank != exporter_rank) {
    shared_handle = fromBytes<handle_typename>(
        store->get(std::string("nvls_export_fd_mcast_handle")));
    root_pid = fromBytes<pid_t>(
        store->get(std::string("nvls_export_pid_mcast_handle")));

    int pid_fd, peer_fd;
    pid_fd = syscall(SYS_pidfd_open, root_pid, /*flags=*/0);
    ASSERT_GE(pid_fd, 0) << "rank " << rank << " failed to open pidfd for pid "
                         << root_pid;

    peer_fd = syscall(SYS_pidfd_getfd, pid_fd, shared_handle, /*flags=*/0);
    ASSERT_GE(peer_fd, 0) << "rank " << rank
                          << " failed to get peer fd, errno: " << errno << " ("
                          << strerror(errno) << ")";
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &mcast_handle, (void*)((uint64_t)peer_fd), handle_type));
    close(pid_fd);
    close(peer_fd);
  }

  // All ranks add their device to multicast group
  CUdevice cu_dev;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle, cu_dev));

  // From the docs
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#add-devices-to-multicast-objects,
  // we need to ensure all devices are added to the multicast group before
  // binding memory, so we need a barrier here. However, it seems that
  // cuMulticastAddDevice already blocks, so the barrier would be redundant.
  // communicator_->barrier(); TODO: needed ?

  // Local memory allocation using Virtual Memory Management
  // All ranks prepare their per-device allocation handle
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(local_rank);
  prop.requestedHandleTypes = handle_type; // any shareable type

  size_t granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (granularity > kSizeBytes) {
    GTEST_SKIP() << "Device does not support the required allocation "
                    "granularity; skipping."
                 << "Granularity: " << granularity
                 << ", required: " << kSizeBytes;
  }

  CUmemGenericAllocationHandle local_buffer = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemCreate(&local_buffer, kSizeBytes, &prop, /*flags=*/0));

  // Bind the local memory to the multicast object
  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle,
      /*mcOffset=*/0,
      local_buffer,
      /*memOffset=*/0,
      kSizeBytes,
      /*flags=*/0));

  // MC Mapping
  CUdeviceptr mc_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &mc_ptr,
      kSizeBytes,
      /*alignment=*/mcast_granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemMap(mc_ptr, kSizeBytes, /*offset=*/0, mcast_handle, /*flags=*/0));
  CUmemAccessDesc mc_mapping_desc{};
  mc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mc_mapping_desc.location.id = static_cast<int>(local_rank);
  mc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(mc_ptr, kSizeBytes, &mc_mapping_desc, /*count=*/1));

  // UC Mapping
  CUdeviceptr uc_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &uc_ptr,
      kSizeBytes,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemMap(uc_ptr, kSizeBytes, /*offset=*/0, local_buffer, /*flags=*/0));
  CUmemAccessDesc uc_mapping_desc{};
  uc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  uc_mapping_desc.location.id = static_cast<int>(local_rank);
  uc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(uc_ptr, kSizeBytes, &uc_mapping_desc, /*count=*/1));

  // Each rank now has a UC address and a MC address associated with a local
  // buffer. The typical and recommended use case is write to the mc address
  // from the root rank, which will broadcast the data, after which each rank
  // can read from its uc address.

  // Root rank writes to mc address
  std::vector<uint32_t> host_buffer(kNumElems);
  if (rank == root_rank) {
    for (size_t i = 0; i < kNumElems; ++i) {
      host_buffer[i] = static_cast<uint32_t>(i * 5 + 11);
    }
    NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
        reinterpret_cast<void*>(mc_ptr),
        host_buffer.data(),
        kSizeBytes,
        cudaMemcpyHostToDevice));
  }

  communicator_->barrier();

  // Each rank reads from the UC address
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      host_buffer.data(),
      reinterpret_cast<void*>(uc_ptr),
      kSizeBytes,
      cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElems; ++i) {
    EXPECT_EQ(host_buffer[i], static_cast<uint32_t>(i * 5 + 11));
  }

  // reading from mc address is an undefined behavior

  // Cleanup
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(mc_ptr, kSizeBytes));
  NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(uc_ptr, kSizeBytes));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(mc_ptr, kSizeBytes));
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(uc_ptr, kSizeBytes));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(local_buffer));
  NVFUSER_CUDA_SAFE_CALL(
      cuMulticastUnbind(mcast_handle, cu_dev, /*offset=*/0, kSizeBytes));
  NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mcast_handle));

  // Restore ptracer setting to default (clear the PR_SET_PTRACER_ANY exception)
  if (rank == exporter_rank) {
    prctl(PR_SET_PTRACER, 0);
  }
}

#endif

TEST_F(IpcTest, VmmMultiRankContiguousMappingTest) {
  // This test demonstrates distributed Virtual Memory Management where:
  // 1. Each rank allocates a buffer
  // 2. All ranks share their allocation handles via IPC
  // 3. Each rank creates a single contiguous virtual address range that maps
  //    all allocations (local + all remote ranks)

  if (communicator_->size() == 1) {
    GTEST_SKIP() << "Skipping test for single device";
  }

  const int64_t num_devices = communicator_->size();
  const int64_t rank = communicator_->deviceId();
  const int64_t local_rank = communicator_->local_rank();

  // Query support for Virtual Memory Management
  int is_vmm_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_vmm_supported,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      local_rank));
  if (is_vmm_supported == 0) {
    GTEST_SKIP()
        << "Device does not support Virtual Memory Management; skipping.";
  }

  // Query support for IPC handles
  int is_ipc_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_ipc_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      local_rank));
  if (is_ipc_supported == 0) {
    GTEST_SKIP() << "Device does not support IPC handles; skipping.";
  }

  NVFUSER_CUDA_RT_SAFE_CALL(cudaSetDevice(local_rank));

  // Set up allocation properties for VMM
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(local_rank);
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // Check allocation granularity
  size_t granularity = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  // Each rank creates one allocation containing 2048 int64_t elements
  constexpr size_t kElemsPerRank = 2048;
  constexpr size_t kBytesPerRank = kElemsPerRank * sizeof(int64_t);

  // Align allocation size to granularity
  size_t aligned_size =
      ((kBytesPerRank + granularity - 1) / granularity) * granularity;
  ASSERT_EQ(aligned_size % sizeof(int64_t), 0)
      << "aligned_size is not aligned to sizeof(int64_t)";
  size_t aligned_number_of_elements_per_rank = aligned_size / sizeof(int64_t);

  // Create local allocation handle
  CUmemGenericAllocationHandle local_alloc_handle = 0;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemCreate(&local_alloc_handle, aligned_size, &prop, /*flags=*/0));

  // Export local allocation handle to shareable file descriptor
  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      local_alloc_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  // Allow peer processes to use pidfd_getfd on this process
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);

  // Share the file descriptor and PID via TCP store
  auto store = communicator_->getTcpStore();
  store->set("vmm_multi_fd_" + std::to_string(rank), toBytes(shared_fd));
  pid_t my_pid = getpid();
  store->set("vmm_multi_pid_" + std::to_string(rank), toBytes(my_pid));

  // Wait for all ranks to finish exporting
  communicator_->barrier();

  // Import all other ranks' allocation handles
  std::vector<CUmemGenericAllocationHandle> all_alloc_handles(num_devices);
  std::vector<int> peer_fds;
  std::vector<int> pid_fds;

  all_alloc_handles[rank] = local_alloc_handle;

  for (int64_t peer_rank = 0; peer_rank < num_devices; ++peer_rank) {
    if (peer_rank == rank) {
      continue; // Skip self
    }

    int peer_shared_fd =
        fromBytes<int>(store->get("vmm_multi_fd_" + std::to_string(peer_rank)));
    pid_t peer_pid = fromBytes<pid_t>(
        store->get("vmm_multi_pid_" + std::to_string(peer_rank)));

    // Get the peer's file descriptor using pidfd_open and pidfd_getfd
    int pid_fd = syscall(SYS_pidfd_open, peer_pid, /*flags=*/0);
    ASSERT_GE(pid_fd, 0) << "rank " << rank << " failed to open pidfd for pid "
                         << peer_pid;
    pid_fds.push_back(pid_fd);

    int peer_fd = syscall(SYS_pidfd_getfd, pid_fd, peer_shared_fd, /*flags=*/0);
    ASSERT_GE(peer_fd, 0) << "rank " << rank
                          << " failed to get peer fd, errno: " << errno << " ("
                          << strerror(errno) << ")";
    peer_fds.push_back(peer_fd);

    // Import the peer's memory handle
    CUmemGenericAllocationHandle peer_alloc_handle = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &peer_alloc_handle,
        (void*)((uint64_t)peer_fd),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    all_alloc_handles[peer_rank] = peer_alloc_handle;
  }

  // Now create one contiguous virtual address range for all allocations
  size_t total_size = aligned_size * num_devices;
  CUdeviceptr base_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &base_ptr,
      total_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));

  // Map each rank's allocation to consecutive regions in the virtual address
  // space
  for (int64_t i = 0; i < num_devices; ++i) {
    CUdeviceptr offset_ptr = base_ptr + (i * aligned_size);
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(
        offset_ptr,
        aligned_size,
        /*offset=*/0,
        all_alloc_handles[i],
        /*flags=*/0));
  }

  // Set memory access permissions for local allocation (read-write) and remote
  // allocations (read-only)
  for (int64_t peer_rank = 0; peer_rank < num_devices; ++peer_rank) {
    CUdeviceptr peer_offset_ptr = base_ptr + (peer_rank * aligned_size);
    CUmemAccessDesc peer_access_desc{};
    peer_access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    peer_access_desc.location.id = static_cast<int>(local_rank);
    peer_access_desc.flags = (peer_rank == rank)
        ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        : CU_MEM_ACCESS_FLAGS_PROT_READ;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(
        peer_offset_ptr, aligned_size, &peer_access_desc, /*count=*/1));
  }

  // Write data to local allocation region
  std::vector<int64_t> local_host_data(aligned_number_of_elements_per_rank);
  for (size_t i = 0; i < aligned_number_of_elements_per_rank; ++i) {
    local_host_data[i] = static_cast<int64_t>(rank * 100000 + i);
  }

  CUdeviceptr local_device_ptr = base_ptr + (rank * aligned_size);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      reinterpret_cast<void*>(local_device_ptr),
      local_host_data.data(),
      aligned_size,
      cudaMemcpyHostToDevice));

  // Wait for all ranks to finish writing their data
  communicator_->barrier();

  // Read back the entire contiguous buffer containing all ranks' data
  std::vector<int64_t> full_readback_data(
      aligned_number_of_elements_per_rank * num_devices);
  NVFUSER_CUDA_RT_SAFE_CALL(cudaMemcpy(
      full_readback_data.data(),
      reinterpret_cast<void*>(base_ptr),
      aligned_size * num_devices,
      cudaMemcpyDeviceToHost));

  // Verify that we can read all ranks' data from the contiguous buffer
  for (int64_t source_rank = 0; source_rank < num_devices; ++source_rank) {
    for (size_t i = 0; i < aligned_number_of_elements_per_rank; ++i) {
      int64_t expected = static_cast<int64_t>(source_rank * 100000 + i);
      size_t idx = source_rank * aligned_number_of_elements_per_rank + i;
      ASSERT_EQ(full_readback_data[idx], expected)
          << "Rank " << rank << " failed to read from rank " << source_rank
          << " at element " << i;
    }
  }

  // Wait for all ranks to finish reading their data
  communicator_->barrier();

  // Clean up: Unmap all allocations from the virtual address space
  for (int64_t i = 0; i < num_devices; ++i) {
    CUdeviceptr offset_ptr = base_ptr + (i * aligned_size);
    NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(offset_ptr, aligned_size));
  }

  // Free the virtual address space
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(base_ptr, total_size));

  // Release all allocation handles
  for (int64_t i = 0; i < num_devices; ++i) {
    NVFUSER_CUDA_SAFE_CALL(cuMemRelease(all_alloc_handles[i]));
  }

  // Close file descriptors
  for (int fd : peer_fds) {
    close(fd);
  }
  for (int fd : pid_fds) {
    close(fd);
  }
}

} // namespace nvfuser
