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
// - Supports NVLS multicast
class SymmetricTensor {
 public:
  // Allocate new symmetric tensor across all devices
  SymmetricTensor(
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      std::optional<uint64_t> alloc_id = std::nullopt);

  // Wrap pre-allocated symmetric tensor (must use allocateSymmetricTensor)
  // tag: unique coordination key (auto-generated if empty)
  explicit SymmetricTensor(
      const at::Tensor& local_tensor,
      const std::string& tag = "");

  ~SymmetricTensor();

  SymmetricTensor(const SymmetricTensor&) = delete;
  SymmetricTensor& operator=(const SymmetricTensor&) = delete;

  const at::Tensor& localTensor() const {
    return local_tensor_;
  }

  at::Tensor remoteTensor(int64_t rank) const;

  void setupMulticast(int64_t exporter_rank, const std::string& store_key_prefix);
  
  void* multicastPtr() const;

  size_t granularity() const {
    return granularity_;
  }

  size_t alignedSize() const {
    return aligned_size_;
  }
  
  CUmemGenericAllocationHandle getAllocHandle(int64_t rank) const {
    setupIpcHandles();
    return alloc_handles_[rank];
  }

  void setupIpcHandles() const;

 private:
  void initialize(at::IntArrayRef sizes, at::ScalarType dtype, std::optional<uint64_t> alloc_id);
  void cleanup();

  at::Tensor local_tensor_;
  mutable std::vector<CUmemGenericAllocationHandle> alloc_handles_;
  mutable std::vector<CUdeviceptr> remote_ptrs_;
  int64_t world_size_;
  int64_t local_rank_;
  size_t granularity_;
  size_t aligned_size_;
  std::string tag_;
  mutable bool ipc_handles_setup_ = false;
  bool multicast_enabled_ = false;
  CUmemGenericAllocationHandle mcast_handle_{};
  CUdevice cu_dev_{};
  void* mc_ptr_{nullptr};
  int exporter_rank_{-1};
  int pid_fd_{-1};
  int peer_fd_{-1};
};

// Factory function to create a contiguous view across all ranks
// This creates a single contiguous virtual address space containing
// all ranks' buffers in order: [rank0, rank1, ..., rankN]
at::Tensor createContiguousView(const SymmetricTensor& sym_tensor);

// Locally allocate a symmetric CUDA tensor using VMM
NVF_API at::Tensor allocateSymmetricTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device,
    std::optional<uint64_t> alloc_id);

// Validate that the local allocation is compatible with Symmetric memory backend 
// Returns empty string if valid, error string otherwise
NVF_API std::string isSymmetricAllocationValid(at::Tensor tensor);

} // namespace nvfuser

