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
  // Wrap pre-allocated symmetric tensor (must use allocateSymmetricTensor)
  explicit SymmetricTensor(const at::Tensor& local_tensor);

  ~SymmetricTensor();

  SymmetricTensor(const SymmetricTensor&) = delete;
  SymmetricTensor& operator=(const SymmetricTensor&) = delete;

  const at::Tensor& localTensor() const {
    return local_tensor_;
  }

  // Setup remote IPC handles (lazy, init-once per tag)
  // tag: unique coordination key (must be same on all ranks)
  void setupRemoteHandles(const std::string& tag = "") const;
  
  at::Tensor remoteTensor(int64_t rank) const;

  // Setup NVLS multicast (CUDA 13.0+, init-once)
  // tag: unique coordination key (must be same on all ranks)
  void setupMulticast(int64_t exporter_rank, const std::string& tag = "");
  
  void* multicastPtr() const;

  size_t granularity() const {
    return granularity_;
  }

  size_t alignedSize() const {
    return aligned_size_;
  }
  
  CUmemGenericAllocationHandle getAllocHandle(int64_t rank, const std::string& tag) const {
    setupRemoteHandles(tag);
    return alloc_handles_[rank];
  }

 private:
  at::Tensor local_tensor_;
  mutable std::vector<CUmemGenericAllocationHandle> alloc_handles_;
  mutable std::vector<CUdeviceptr> remote_ptrs_;
  int64_t world_size_;
  int64_t my_device_id_;
  size_t granularity_;
  size_t aligned_size_;
  mutable bool are_remote_tensors_setup_ = false;
  bool is_multicast_setup_ = false;
  CUmemGenericAllocationHandle mcast_handle_{};
  CUdevice cu_dev_{};
  void* mc_ptr_{nullptr};
  int exporter_rank_{-1};
  int pid_fd_{-1};
  int peer_fd_{-1};
};

// Create contiguous view of all ranks: [rank0, rank1, ..., rankN]
// tag: unique coordination key for IPC setup (must be same on all ranks)
at::Tensor createContiguousView(
    const SymmetricTensor& sym_tensor,
    const std::string& tag);

// Locally allocate a symmetric CUDA tensor using VMM
NVF_API at::Tensor allocateSymmetricTensor(
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    at::Device device);

// Validate that the local allocation is compatible with Symmetric memory backend 
// Returns empty string if valid, error string otherwise
NVF_API std::string isSymmetricAllocationValid(at::Tensor tensor);

// Helper functions for serializing data to bytes for TCP store
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

} // namespace nvfuser

