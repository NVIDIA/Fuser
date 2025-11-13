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
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    std::optional<uint64_t> alloc_id)
    : world_size_(0), local_rank_(0), granularity_(0), aligned_size_(0) {
  initialize(sizes, dtype, alloc_id);
}

SymmetricTensor::SymmetricTensor(
    const at::Tensor& local_tensor,
    const std::string& tag)
    : local_tensor_(local_tensor),
      world_size_(0),
      local_rank_(0),
      granularity_(0),
      aligned_size_(0),
      tag_(tag) {
  NVF_ERROR(
      local_tensor.is_cuda(),
      "SymmetricTensor requires CUDA tensor, got: ",
      local_tensor.device());

  // Validate that the tensor was allocated with symmetric memory
  std::string error = isSymmetricAllocationValid(local_tensor);
  NVF_CHECK(
      error.empty(),
      "Tensor must be allocated with allocateSymmetricTensor. Error: ",
      error);

  Communicator& comm = Communicator::getInstance();
  world_size_ = comm.size();
  local_rank_ = comm.deviceId();

  // Calculate granularity and aligned size
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(comm.local_rank());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t required_size = local_tensor.numel() * local_tensor.element_size();
  aligned_size_ =
      ((required_size + granularity_ - 1) / granularity_) * granularity_;

  // Initialize local handle and pointer immediately (not lazy)
  alloc_handles_.resize(world_size_);
  remote_ptrs_.resize(world_size_);
  
  CUdeviceptr local_ptr = reinterpret_cast<CUdeviceptr>(local_tensor_.data_ptr());
  
  CUmemGenericAllocationHandle local_handle;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&local_handle, reinterpret_cast<void*>(local_ptr)));

  alloc_handles_[local_rank_] = local_handle;
  remote_ptrs_[local_rank_] = local_ptr;

  // Remote IPC handles setup is lazy - will be initialized on first remote access
}

SymmetricTensor::~SymmetricTensor() {
  if (!moved_from_) {
    cleanup();
  }
}

SymmetricTensor::SymmetricTensor(SymmetricTensor&& other) noexcept
    : local_tensor_(std::move(other.local_tensor_)),
      alloc_handles_(std::move(other.alloc_handles_)),
      remote_ptrs_(std::move(other.remote_ptrs_)),
      world_size_(other.world_size_),
      local_rank_(other.local_rank_),
      granularity_(other.granularity_),
      aligned_size_(other.aligned_size_),
      tag_(std::move(other.tag_)),
      ipc_handles_setup_(other.ipc_handles_setup_),
      multicast_enabled_(other.multicast_enabled_),
      mcast_handle_(other.mcast_handle_),
      cu_dev_(other.cu_dev_),
      mc_ptr_(other.mc_ptr_),
      exporter_rank_(other.exporter_rank_),
      pid_fd_(other.pid_fd_),
      peer_fd_(other.peer_fd_),
      moved_from_(false) {
  other.moved_from_ = true;
  other.ipc_handles_setup_ = false;
  other.multicast_enabled_ = false;
  other.mcast_handle_ = 0;
  other.mc_ptr_ = nullptr;
  other.pid_fd_ = -1;
  other.peer_fd_ = -1;
}

SymmetricTensor& SymmetricTensor::operator=(SymmetricTensor&& other) noexcept {
  if (this != &other) {
    if (!moved_from_) {
      cleanup();
    }
    local_tensor_ = std::move(other.local_tensor_);
    alloc_handles_ = std::move(other.alloc_handles_);
    remote_ptrs_ = std::move(other.remote_ptrs_);
    world_size_ = other.world_size_;
    local_rank_ = other.local_rank_;
    granularity_ = other.granularity_;
    aligned_size_ = other.aligned_size_;
    tag_ = std::move(other.tag_);
    ipc_handles_setup_ = other.ipc_handles_setup_;
    multicast_enabled_ = other.multicast_enabled_;
    mcast_handle_ = other.mcast_handle_;
    cu_dev_ = other.cu_dev_;
    mc_ptr_ = other.mc_ptr_;
    exporter_rank_ = other.exporter_rank_;
    pid_fd_ = other.pid_fd_;
    peer_fd_ = other.peer_fd_;
    moved_from_ = false;
    other.moved_from_ = true;
    other.ipc_handles_setup_ = false;
    other.multicast_enabled_ = false;
    other.mcast_handle_ = 0;
    other.mc_ptr_ = nullptr;
    other.pid_fd_ = -1;
    other.peer_fd_ = -1;
  }
  return *this;
}

void SymmetricTensor::initialize(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    std::optional<uint64_t> alloc_id) {
  Communicator& comm = Communicator::getInstance();
  NVF_CHECK(
      comm.is_available(),
      "SymmetricTensor requires an initialized communicator");

  world_size_ = comm.size();
  local_rank_ = comm.deviceId();

  // Allocate local tensor with symmetric memory
  c10::Device device = comm.device();
  local_tensor_ = allocateSymmetricTensor(sizes, dtype, device, alloc_id);

  // Calculate granularity and aligned size
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(comm.local_rank());
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  NVFUSER_CUDA_SAFE_CALL(cuMemGetAllocationGranularity(
      &granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t required_size = local_tensor_.numel() * local_tensor_.element_size();
  aligned_size_ =
      ((required_size + granularity_ - 1) / granularity_) * granularity_;

  // Initialize local handle and pointer immediately (not lazy)
  alloc_handles_.resize(world_size_);
  remote_ptrs_.resize(world_size_);
  
  CUdeviceptr local_ptr = reinterpret_cast<CUdeviceptr>(local_tensor_.data_ptr());
  
  // Validate VMM allocation properties
  {
    CUdeviceptr base_ptr = 0;
    size_t va_size = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &va_size, local_ptr));
    NVF_CHECK(
        local_ptr == base_ptr,
        "VMM allocation error: expected ptr to be the base of the address range. "
        "Got ptr=", reinterpret_cast<void*>(local_ptr),
        " but base_ptr=", reinterpret_cast<void*>(base_ptr),
        ". cuMemMap does not support suballocation mapping.");
    NVF_CHECK(
        va_size == aligned_size_,
        "VMM allocation error: mapped region is not the full allocation. "
        "Mapped va_size=", va_size,
        ", but expected aligned_size_=", aligned_size_,
        ".");
  }

  CUmemGenericAllocationHandle local_handle;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemRetainAllocationHandle(&local_handle, reinterpret_cast<void*>(local_ptr)));

  alloc_handles_[local_rank_] = local_handle;
  remote_ptrs_[local_rank_] = local_ptr;

  // Remote IPC handles setup is lazy - will be initialized on first remote access
}

void SymmetricTensor::setupIpcHandles() const {
  if (ipc_handles_setup_) {
    return;
  }

  Communicator& comm = Communicator::getInstance();
  auto store = comm.getTcpStore();

  // Local handle and ptr are already set in constructor
  // We only need to setup remote handles here
  CUdeviceptr local_ptr = remote_ptrs_[local_rank_];
  CUmemGenericAllocationHandle local_handle = alloc_handles_[local_rank_];

  // Validate VMM allocation properties
  {
    CUdeviceptr base_ptr = 0;
    size_t va_size = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(&base_ptr, &va_size, local_ptr));
    NVF_CHECK(
        local_ptr == base_ptr,
        "VMM allocation error: expected ptr to be the base of the address range. "
        "Got ptr=", reinterpret_cast<void*>(local_ptr),
        " but base_ptr=", reinterpret_cast<void*>(base_ptr),
        ". cuMemMap does not support suballocation mapping.");
    NVF_CHECK(
        va_size == aligned_size_,
        "VMM allocation error: mapped region is not the full allocation. "
        "Mapped va_size=", va_size,
        ", but expected aligned_size_=", aligned_size_,
        ".");
  }


  // Export local handle to shareable file descriptor
  int shared_fd;
  NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
      &shared_fd,
      local_handle,
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      /*flags=*/0));

  // Allow peer processes to access this handle
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);

  // Coordinate to get a unique instance ID
  uint64_t instance_id;
  
  if (!tag_.empty()) {
    // If tag is provided, use it directly as the instance identifier
    // Callers must ensure tags are unique across different SymmetricTensors
    instance_id = std::hash<std::string>{}(tag_);
  } else {
    // For untagged instances, use a simpler barrier-based protocol
    // Step 1: Barrier to ensure all ranks are ready to coordinate
    comm.barrier();
    
    // Step 2: Rank 0 atomically gets a sequence number and publishes instance ID
    std::string seq_key = "symmetric_tensor_global_seq";
    if (local_rank_ == 0) {
      int64_t seq = store->add(seq_key, 1);
      instance_id = static_cast<uint64_t>(local_ptr);
      
      std::string id_key = "symmetric_tensor_id_" + std::to_string(seq);
      std::vector<uint8_t> id_bytes(
          reinterpret_cast<const uint8_t*>(&instance_id),
          reinterpret_cast<const uint8_t*>(&instance_id) + sizeof(uint64_t));
      store->set(id_key, id_bytes);
      
      std::string seq_key_published = "symmetric_tensor_seq_published";
      std::vector<uint8_t> seq_bytes(
          reinterpret_cast<const uint8_t*>(&seq),
          reinterpret_cast<const uint8_t*>(&seq) + sizeof(int64_t));
      store->set(seq_key_published, seq_bytes);
    }
    
    // Step 3: Barrier to ensure rank 0 has published
    comm.barrier();
    
    // Step 4: All non-zero ranks read the instance ID
    if (local_rank_ != 0) {
      std::string seq_key_published = "symmetric_tensor_seq_published";
      auto seq_bytes = store->get(seq_key_published);
      int64_t seq = *reinterpret_cast<const int64_t*>(seq_bytes.data());
      
      std::string id_key = "symmetric_tensor_id_" + std::to_string(seq);
      auto id_bytes = store->get(id_key);
      instance_id = *reinterpret_cast<const uint64_t*>(id_bytes.data());
    }
    
    // Step 5: Barrier before cleanup
    comm.barrier();
    
    // Step 6: Rank 0 cleans up keys
    if (local_rank_ == 0) {
      std::string seq_key_published = "symmetric_tensor_seq_published";
      auto seq_bytes = store->get(seq_key_published);
      int64_t seq = *reinterpret_cast<const int64_t*>(seq_bytes.data());
      
      std::string id_key = "symmetric_tensor_id_" + std::to_string(seq);
      store->deleteKey(id_key);
      store->deleteKey(seq_key_published);
    }
  }
  
  std::string instance_str = std::to_string(instance_id);

  // Share FD and PID via store
  std::string fd_key =
      "symmetric_tensor_fd_" + std::to_string(local_rank_) + "_" + instance_str;
  std::string pid_key =
      "symmetric_tensor_pid_" + std::to_string(local_rank_) + "_" + instance_str;

  std::vector<uint8_t> fd_bytes(
      reinterpret_cast<const uint8_t*>(&shared_fd),
      reinterpret_cast<const uint8_t*>(&shared_fd) + sizeof(int));
  pid_t my_pid = getpid();
  std::vector<uint8_t> pid_bytes(
      reinterpret_cast<const uint8_t*>(&my_pid),
      reinterpret_cast<const uint8_t*>(&my_pid) + sizeof(pid_t));

  store->set(fd_key, fd_bytes);
  store->set(pid_key, pid_bytes);

  // Barrier to ensure all ranks have published their handles
  comm.barrier();

  // Import peer handles
  for (int64_t peer_rank = 0; peer_rank < world_size_; ++peer_rank) {
    if (peer_rank == local_rank_) {
      continue;
    }

    std::string peer_fd_key =
        "symmetric_tensor_fd_" + std::to_string(peer_rank) + "_" + instance_str;
    std::string peer_pid_key =
        "symmetric_tensor_pid_" + std::to_string(peer_rank) + "_" + instance_str;

    auto peer_fd_bytes = store->get(peer_fd_key);
    auto peer_pid_bytes = store->get(peer_pid_key);

    int peer_fd =
        *reinterpret_cast<const int*>(peer_fd_bytes.data());
    pid_t peer_pid =
        *reinterpret_cast<const pid_t*>(peer_pid_bytes.data());

    // Get peer's FD using pidfd
    int pid_fd = syscall(SYS_pidfd_open, peer_pid, /*flags=*/0);
    NVF_CHECK(
        pid_fd >= 0,
        "Failed to open pidfd for peer rank ",
        peer_rank,
        " pid ",
        peer_pid);

    int local_peer_fd =
        syscall(SYS_pidfd_getfd, pid_fd, peer_fd, /*flags=*/0);
    NVF_CHECK(
        local_peer_fd >= 0,
        "Failed to get peer fd for rank ",
        peer_rank);

    // Import peer's allocation handle
    CUmemGenericAllocationHandle peer_handle;
    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &peer_handle,
        reinterpret_cast<void*>(static_cast<uint64_t>(local_peer_fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    alloc_handles_[peer_rank] = peer_handle;

    // Map peer's memory to local virtual address space
    CUdeviceptr peer_ptr = 0;
    NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
        &peer_ptr,
        aligned_size_,
        /*alignment=*/granularity_,
        /*baseVA=*/0,
        /*flags=*/0));
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(
        peer_ptr, aligned_size_, /*offset=*/0, peer_handle, /*flags=*/0));

    // Set read-only access for peer memory
    CUmemAccessDesc peer_access_desc{};
    peer_access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    peer_access_desc.location.id = static_cast<int>(comm.local_rank());
    peer_access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(
        peer_ptr, aligned_size_, &peer_access_desc, /*count=*/1));

    remote_ptrs_[peer_rank] = peer_ptr;

    close(local_peer_fd);
    close(pid_fd);
  }

  // Barrier to ensure all ranks have completed IPC setup
  comm.barrier();

  // Mark as setup (must be last to ensure atomicity)
  ipc_handles_setup_ = true;
}

void SymmetricTensor::cleanup() {
  if (moved_from_) {
    return;
  }

#if (CUDA_VERSION >= 13000)
  // Clean up multicast if enabled
  if (multicast_enabled_) {
    if (mc_ptr_ != nullptr) {
      CUdeviceptr cu_ptr = reinterpret_cast<CUdeviceptr>(mc_ptr_);
      NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(cu_ptr, aligned_size_));
      NVFUSER_CUDA_SAFE_CALL(cuMemAddressFree(cu_ptr, aligned_size_));
    }
    if (mcast_handle_ != 0) {
      NVFUSER_CUDA_SAFE_CALL(
          cuMulticastUnbind(mcast_handle_, cu_dev_, /*offset=*/0, aligned_size_));
      NVFUSER_CUDA_SAFE_CALL(cuMemRelease(mcast_handle_));
    }
    if (peer_fd_ >= 0) {
      close(peer_fd_);
    }
    if (pid_fd_ >= 0) {
      close(pid_fd_);
    }
  }
#endif

  // Clean up remote IPC handles if they were setup
  if (ipc_handles_setup_) {
    // Unmap remote pointers (but not local, which is managed by the tensor)
    for (int64_t rank = 0; rank < world_size_; ++rank) {
      if (rank == local_rank_) {
        continue;
      }
      if (remote_ptrs_[rank] != 0) {
        NVFUSER_CUDA_SAFE_CALL(cuMemUnmap(remote_ptrs_[rank], aligned_size_));
        NVFUSER_CUDA_SAFE_CALL(
            cuMemAddressFree(remote_ptrs_[rank], aligned_size_));
      }
    }

    // Release remote allocation handles (imported)
    for (int64_t rank = 0; rank < world_size_; ++rank) {
      if (rank != local_rank_ && alloc_handles_[rank] != 0) {
        NVFUSER_CUDA_SAFE_CALL(cuMemRelease(alloc_handles_[rank]));
      }
    }
  }

  // Always release local handle (retained in constructor)
  if (alloc_handles_.size() > static_cast<size_t>(local_rank_) &&
      alloc_handles_[local_rank_] != 0) {
    NVFUSER_CUDA_SAFE_CALL(cuMemRelease(alloc_handles_[local_rank_]));
  }
}

at::Tensor SymmetricTensor::remoteTensor(int64_t rank) const {
  NVF_CHECK(!moved_from_, "Cannot access moved-from SymmetricTensor");
  NVF_CHECK(
      rank >= 0 && rank < world_size_,
      "Rank ",
      rank,
      " out of range [0, ",
      world_size_,
      ")");

  if (rank == local_rank_) {
    return local_tensor_;
  }

  // Ensure IPC handles are setup before accessing remote memory
  setupIpcHandles();

  // Create a tensor view from the remote pointer
  auto options = at::TensorOptions()
                     .dtype(local_tensor_.scalar_type())
                     .device(at::kCUDA, rank); // TODO: use remote's local rank for multinode setups

  return at::from_blob(
      reinterpret_cast<void*>(remote_ptrs_[rank]),
      local_tensor_.sizes(),
      local_tensor_.strides(),
      options);
}

void* SymmetricTensor::remoteTensorPtr(int64_t rank) const {
  NVF_CHECK(!moved_from_, "Cannot access moved-from SymmetricTensor");
  NVF_CHECK(
      rank >= 0 && rank < world_size_,
      "Rank ",
      rank,
      " out of range [0, ",
      world_size_,
      ")");

  if (rank == local_rank_) {
    return local_tensor_.data_ptr();
  }

  // Ensure IPC handles are setup before accessing remote memory
  setupIpcHandles();

  return reinterpret_cast<void*>(remote_ptrs_[rank]);
}

std::vector<at::Tensor> SymmetricTensor::remoteTensors() const {
  NVF_CHECK(!moved_from_, "Cannot access moved-from SymmetricTensor");
  std::vector<at::Tensor> result;
  result.reserve(world_size_ - 1);

  for (int64_t rank = 0; rank < world_size_; ++rank) {
    if (rank != local_rank_) {
      result.push_back(remoteTensor(rank));
    }
  }

  return result;
}

bool SymmetricTensor::isValid() const {
  if (moved_from_) {
    return false;
  }
  if (!local_tensor_.defined()) {
    return false;
  }
  // Don't check alloc_handles_ and remote_ptrs_ since they're lazily initialized
  return true;
}

void* SymmetricTensor::multicastPtr() const {
  NVF_CHECK(
      multicast_enabled_,
      "Multicast not enabled. Call setupMulticast() first.");
  return mc_ptr_;
}

void SymmetricTensor::setupMulticast(
    int64_t exporter_rank,
    const std::string& store_key_prefix) {
#if (CUDA_VERSION >= 13000)
  // Check if multicast is already enabled (init-once)
  if (multicast_enabled_) {
    return;
  }

  NVF_CHECK(isValid(), "SymmetricTensor must be valid before setup multicast");

  // // Ensure IPC handles are setup before setting up multicast
  // setupIpcHandles();

  Communicator& comm = Communicator::getInstance();
  const int64_t my_rank = comm.deviceId();
  const int64_t local_rank = comm.local_rank();

  // Validate device capabilities
  int is_multicast_supported;
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGetAttribute(
      &is_multicast_supported,
      CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
      local_rank));
  NVF_CHECK(
      is_multicast_supported != 0, "Device does not support NVLS Multicast");

  exporter_rank_ = exporter_rank;
  auto handle_type = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // Create multicast properties
  CUmulticastObjectProp mcast_prop{};
  mcast_prop.flags = 0;
  mcast_prop.handleTypes = handle_type;
  mcast_prop.numDevices = world_size_;
  mcast_prop.size = aligned_size_;

  int shared_handle;
  auto store = comm.getTcpStore();
  NVF_CHECK(store != nullptr, "TCP store is null");
  pid_t root_pid;

  // Exporter creates and exports multicast handle
  if (my_rank == exporter_rank) {
    NVFUSER_CUDA_SAFE_CALL(cuMulticastCreate(&mcast_handle_, &mcast_prop));
    NVFUSER_CUDA_SAFE_CALL(cuMemExportToShareableHandle(
        &shared_handle, mcast_handle_, handle_type, /*flags=*/0));

    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY);

    root_pid = getpid();

    std::vector<uint8_t> fd_bytes(
        reinterpret_cast<const uint8_t*>(&shared_handle),
        reinterpret_cast<const uint8_t*>(&shared_handle) + sizeof(int));
    std::vector<uint8_t> pid_bytes(
        reinterpret_cast<const uint8_t*>(&root_pid),
        reinterpret_cast<const uint8_t*>(&root_pid) + sizeof(pid_t));

    store->set(store_key_prefix + "_fd", fd_bytes);
    store->set(store_key_prefix + "_pid", pid_bytes);
  }

  comm.barrier();

  // Importers import the multicast handle
  if (my_rank != exporter_rank) {
    auto fd_bytes = store->get(store_key_prefix + "_fd");
    shared_handle = *reinterpret_cast<const int*>(fd_bytes.data());
    
    auto pid_bytes = store->get(store_key_prefix + "_pid");
    root_pid = *reinterpret_cast<const pid_t*>(pid_bytes.data());

    pid_fd_ = syscall(SYS_pidfd_open, root_pid, /*flags=*/0);
    NVF_CHECK(
        pid_fd_ >= 0,
        "Rank ",
        my_rank,
        " failed to open pidfd for pid ",
        root_pid);

    peer_fd_ = syscall(SYS_pidfd_getfd, pid_fd_, shared_handle, /*flags=*/0);
    NVF_CHECK(peer_fd_ >= 0, "Rank ", my_rank, " failed to get peer fd");

    NVFUSER_CUDA_SAFE_CALL(cuMemImportFromShareableHandle(
        &mcast_handle_,
        reinterpret_cast<void*>(static_cast<uint64_t>(peer_fd_)),
        handle_type));
  }

  // All ranks add their device to multicast group
  NVFUSER_CUDA_SAFE_CALL(cuDeviceGet(&cu_dev_, static_cast<int>(local_rank)));
  NVFUSER_CUDA_SAFE_CALL(cuMulticastAddDevice(mcast_handle_, cu_dev_));

  // Bind local memory to multicast
  // Compute memOffset from the base of the allocation handle
  CUdeviceptr local_ptr = remote_ptrs_[local_rank_];
  CUdeviceptr alloc_base_ptr = 0;
  size_t alloc_size = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemGetAddressRange(
      &alloc_base_ptr, &alloc_size, local_ptr));
  
  size_t mem_offset = static_cast<size_t>(local_ptr - alloc_base_ptr);
  
  NVFUSER_CUDA_SAFE_CALL(cuMulticastBindMem(
      mcast_handle_,
      /*mcOffset=*/0,
      alloc_handles_[local_rank_],
      /*memOffset=*/mem_offset,
      aligned_size_,
      /*flags=*/0));

  // Map multicast address (MC)
  CUdeviceptr mc_cu_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &mc_cu_ptr,
      aligned_size_,
      /*alignment=*/granularity_,
      /*baseVA=*/0,
      /*flags=*/0));
  NVFUSER_CUDA_SAFE_CALL(
      cuMemMap(mc_cu_ptr, aligned_size_, /*offset=*/0, mcast_handle_, /*flags=*/0));

  CUmemAccessDesc mc_mapping_desc{};
  mc_mapping_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  mc_mapping_desc.location.id = static_cast<int>(local_rank);
  mc_mapping_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVFUSER_CUDA_SAFE_CALL(
      cuMemSetAccess(mc_cu_ptr, aligned_size_, &mc_mapping_desc, /*count=*/1));

  mc_ptr_ = reinterpret_cast<void*>(mc_cu_ptr);
  multicast_enabled_ = true;

  comm.barrier();

  // Clean up store keys
  if (my_rank == exporter_rank) {
    store->deleteKey(store_key_prefix + "_fd");
    store->deleteKey(store_key_prefix + "_pid");
  }
#else
  (void)exporter_rank;
  NVF_ERROR(false, "NVLS Multicast requires CUDA 13.0 or higher");
#endif
}

at::Tensor createContiguousView(const SymmetricTensor& sym_tensor) {
  NVF_CHECK(sym_tensor.isValid(), "Invalid SymmetricTensor");

  Communicator& comm = Communicator::getInstance();
  const int64_t local_rank = comm.local_rank();
  const int64_t world_size = sym_tensor.worldSize();
  const size_t aligned_size = sym_tensor.alignedSize();
  const size_t granularity = sym_tensor.granularity();
  const size_t actual_size = sym_tensor.nbytes();

  // For now, we require that aligned_size equals actual_size
  // This simplifies the implementation and avoids dealing with padding
  NVF_CHECK(
      aligned_size == actual_size,
      "createContiguousView currently requires aligned_size to equal actual tensor size. ",
      "aligned_size=",
      aligned_size,
      ", actual_size=",
      actual_size,
      ". This typically happens when tensor size is not a multiple of VMM granularity.");

  // Reserve contiguous virtual address space for all ranks
  size_t total_size = actual_size * world_size;
  CUdeviceptr contiguous_ptr = 0;
  NVFUSER_CUDA_SAFE_CALL(cuMemAddressReserve(
      &contiguous_ptr,
      total_size,
      /*alignment=*/granularity,
      /*baseVA=*/0,
      /*flags=*/0));

  // Map each rank's buffer to consecutive regions
  for (int64_t rank = 0; rank < world_size; ++rank) {
    CUmemGenericAllocationHandle alloc_handle = sym_tensor.getAllocHandle(rank);
    CUdeviceptr rank_region_ptr = contiguous_ptr + (rank * actual_size);
    
    NVFUSER_CUDA_SAFE_CALL(cuMemMap(
        rank_region_ptr,
        actual_size,
        /*offset=*/0,
        alloc_handle,
        /*flags=*/0));
  }

  // Set memory access permissions
  for (int64_t rank = 0; rank < world_size; ++rank) {
    CUdeviceptr rank_region_ptr = contiguous_ptr + (rank * actual_size);
    CUmemAccessDesc access_desc{};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = static_cast<int>(local_rank);
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    NVFUSER_CUDA_SAFE_CALL(cuMemSetAccess(
        rank_region_ptr, actual_size, &access_desc, /*count=*/1));
  }

  // Create tensor view over the contiguous range
  // The first dimension is world_size, followed by the original tensor dimensions
  std::vector<int64_t> sizes = {world_size};
  for (int64_t s : sym_tensor.sizes()) {
    sizes.push_back(s);
  }

  // Calculate standard contiguous strides (no padding between ranks)
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (int64_t i = sizes.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  auto options = at::TensorOptions()
                     .dtype(sym_tensor.dtype())
                     .device(at::kCUDA, 0);

  // Create tensor with custom deleter to clean up VMM resources
  return at::from_blob(
      reinterpret_cast<void*>(contiguous_ptr),
      sizes,
      strides,
      [=](void* ptr) {
        // Cleanup lambda: unmap memory and free virtual address space
        for (int64_t rank = 0; rank < world_size; ++rank) {
          CUdeviceptr rank_region_ptr =
              reinterpret_cast<CUdeviceptr>(ptr) + (rank * actual_size);
          cuMemUnmap(rank_region_ptr, actual_size);
        }
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), total_size);
      },
      options);
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

