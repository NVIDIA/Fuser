// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <cuda.h>
#include <exceptions.h>

#include <optional>
#include <string>
#include <vector>

namespace nvfuser {

// Forward declarations
class Communicator;

// SymmetricTensor represents a distributed tensor with symmetric memory
// allocation across multiple devices. Each rank has local and remote views
// of all device buffers via CUDA Virtual Memory Management (VMM).
//
// Key properties:
// - All ranks can directly access any other rank's buffer via remote pointers
// - Supports efficient P2P communication and zero-copy transfers
// - Compatible with NVLS multicast for broadcast/allgather operations
// - Ensures proper alignment and granularity for VMM allocations
//
// Example usage:
// ```cpp
// // Create a symmetric tensor across all devices
// SymmetricTensor sym_tensor({1024, 1024}, at::ScalarType::Float);
//
// // Access local buffer for this rank
// at::Tensor local = sym_tensor.localTensor();
//
// // Access remote buffer from rank 2
// at::Tensor remote = sym_tensor.remoteTensor(2);
//
// // Get raw pointers for kernel access
// void* peer_ptr = sym_tensor.remoteTensorPtr(2);
// ```
class SymmetricTensor {
 public:
  // Construct a SymmetricTensor with given shape and dtype
  // Allocates symmetric memory across all devices in the communicator
  SymmetricTensor(
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      std::optional<uint64_t> alloc_id = std::nullopt);

  // Construct from a pre-allocated local tensor
  // The tensor must be allocated with allocateSymmetricTensor
  // param local_tensor: Pre-allocated symmetric tensor
  // param tag: Unique tag for coordination (must be same on all ranks)
  //            If empty, uses a default coordination key
  explicit SymmetricTensor(
      const at::Tensor& local_tensor,
      const std::string& tag = "");

  // Destructor cleans up VMM mappings and handles
  ~SymmetricTensor();

  // Disable copy to prevent handle management issues
  SymmetricTensor(const SymmetricTensor&) = delete;
  SymmetricTensor& operator=(const SymmetricTensor&) = delete;

  // Enable move semantics
  SymmetricTensor(SymmetricTensor&&) noexcept;
  SymmetricTensor& operator=(SymmetricTensor&&) noexcept;

  // Get the local tensor for the current rank
  const at::Tensor& localTensor() const {
    return local_tensor_;
  }

  // Get a remote tensor view for the specified rank
  // param rank: The remote rank to access (must be in range [0, world_size))
  // returns: Tensor view pointing to the remote rank's buffer
  at::Tensor remoteTensor(int64_t rank) const;

  // Get raw pointer to remote rank's buffer (useful for passing to CUDA kernels)
  void* remoteTensorPtr(int64_t rank) const;

  // Get raw pointer to local buffer
  void* localTensorPtr() const {
    return local_tensor_.data_ptr();
  }

  // Get the world size (number of ranks)
  int64_t worldSize() const {
    return world_size_;
  }

  // Get the local rank
  int64_t localRank() const {
    return local_rank_;
  }

  // Get all remote tensor views (excluding local)
  std::vector<at::Tensor> remoteTensors() const;

  // Get tensor shape
  at::IntArrayRef sizes() const {
    return local_tensor_.sizes();
  }

  // Get tensor dtype
  at::ScalarType dtype() const {
    return local_tensor_.scalar_type();
  }

  // Get total number of elements
  int64_t numel() const {
    return local_tensor_.numel();
  }

  // Get element size in bytes
  int64_t element_size() const {
    return local_tensor_.element_size();
  }

  // Get total size in bytes
  int64_t nbytes() const {
    return local_tensor_.numel() * local_tensor_.element_size();
  }

  // Check if symmetric tensor is valid and properly initialized
  bool isValid() const;

  // Get the VMM allocation granularity used
  size_t granularity() const {
    return granularity_;
  }

  // Get the aligned allocation size (may be larger than requested)
  size_t alignedSize() const {
    return aligned_size_;
  }

  // Setup IPC handles and remote mappings (lazy, called once on first remote access)
  void setupIpcHandles() const;

  // Setup NVLS multicast for efficient broadcast operations
  // Must be called after construction. Requires CUDA 13.0+
  // param exporter_rank: Rank that exports the multicast handle
  // param store_key_prefix: Unique prefix for store keys (must be same on all ranks)
  void setupMulticast(int64_t exporter_rank, const std::string& store_key_prefix);

  // Check if multicast is enabled
  bool hasMulticast() const {
    return multicast_enabled_;
  }

  // Get multicast pointer (write-only for root, undefined for others)
  // Only valid if setupMulticast() was called
  void* multicastPtr() const;

  // Get unicast pointer (read-only, all ranks)
  // Returns the local tensor's data pointer
  void* unicastPtr() const {
    return local_tensor_.data_ptr();
  }

  // Get allocation handle for a specific rank (for internal use)
  CUmemGenericAllocationHandle getAllocHandle(int64_t rank) const {
    setupIpcHandles();
    NVF_CHECK(
        rank >= 0 && rank < world_size_,
        "Rank out of range");
    return alloc_handles_[rank];
  }

 private:
  // Initialize symmetric memory allocation and IPC sharing
  void initialize(
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      std::optional<uint64_t> alloc_id);


  // Clean up IPC handles and remote mappings
  void cleanup();

  // Local tensor (owned by this rank)
  at::Tensor local_tensor_;

  // VMM allocation handles for each rank (lazily initialized)
  mutable std::vector<CUmemGenericAllocationHandle> alloc_handles_;

  // Remote virtual addresses for each rank's buffer (lazily initialized)
  mutable std::vector<CUdeviceptr> remote_ptrs_;

  // World size (total number of ranks)
  int64_t world_size_;

  // Local rank index
  int64_t local_rank_;

  // VMM allocation granularity
  size_t granularity_;

  // Aligned allocation size
  size_t aligned_size_;

  // Unique tag for this instance
  std::string tag_;

  // Lazy initialization flags
  mutable bool ipc_handles_setup_ = false;

  // Multicast support (CUDA 13.0+)
  bool multicast_enabled_ = false;
  CUmemGenericAllocationHandle mcast_handle_{};
  CUdevice cu_dev_{};
  void* mc_ptr_{nullptr};
  int exporter_rank_{-1};
  int pid_fd_{-1};
  int peer_fd_{-1};

  // Flag indicating if this object has been moved from
  bool moved_from_ = false;
};

// Factory function to create a contiguous view across all ranks
// This creates a single contiguous virtual address space containing
// all ranks' buffers in order: [rank0, rank1, ..., rankN]
//
// This is useful for operations that need to access the global view,
// such as allgather results or broadcasting to all ranks.
//
// \param sym_tensor The SymmetricTensor to create a contiguous view from
// \return Tensor with contiguous view of all ranks' buffers
at::Tensor createContiguousView(const SymmetricTensor& sym_tensor);

} // namespace nvfuser

