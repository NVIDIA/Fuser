// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/Tensor.h>
#include <cuda.h>

namespace nvfuser {

// SymmetricTensor wraps a local symmetric memory allocation and enables:
// - Remote access to other ranks' buffers via CUDA VMM
// - NVLS multicast for efficient broadcasts
// - Contiguous view creation across all ranks
//
// Design: Decouples local allocation from IPC handle exchange for better
// interoperability and support for pre-allocated user buffers
//
// TODO: Long term plan is to integrate pytorch's native symmetric memory as a
// possible backend. One important reason to use pytorch's allocator is to use
// pytorch's memory pool to let the framework own the memory stack and not
// further fragment the memory. On the other hand, having our own implementation
// allows us to experiment more advanced features like contigous view creation.
class SymmetricTensor {
 public:
  // Wrap pre-allocated symmetric tensor (must use allocate())
  explicit SymmetricTensor(const at::Tensor& local_tensor);

  ~SymmetricTensor();

  SymmetricTensor(const SymmetricTensor&) = delete;
  SymmetricTensor& operator=(const SymmetricTensor&) = delete;

  // Local allocation (decoupled from IPC setup for flexibility)
  static at::Tensor allocate(
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      at::Device device);

  // Validate local tensor is symmetric memory compatible
  static std::string validate(at::Tensor tensor);

  const at::Tensor& localTensor() const {
    return local_tensor_;
  }

  // Setup remote access (lazy, init-once)
  void setupRemoteHandles(const std::string& tag = "");
  at::Tensor remoteTensor(int64_t rank) const;

  // Setup multicast (CUDA 13.0+, init-once)
  void setupMulticast(int64_t exporter_rank, const std::string& tag = "");
  void* multicastPtr() const;

  // Setup contiguous view (lazy, init-once)
  void setupContiguousView(const std::string& tag = "");
  at::Tensor getContiguousView() const;

  size_t granularity() const {
    return granularity_;
  }

  size_t alignedSize() const {
    return aligned_size_;
  }

  CUmemGenericAllocationHandle getAllocHandle(
      int64_t rank,
      const std::string& tag) {
    setupRemoteHandles(tag);
    return alloc_handles_[rank];
  }

 private:
  at::Tensor local_tensor_;
  std::vector<CUmemGenericAllocationHandle> alloc_handles_;
  std::vector<CUdeviceptr> remote_ptrs_;
  int64_t world_size_;
  int64_t my_device_id_;
  size_t granularity_;
  size_t aligned_size_;
  bool are_remote_tensors_setup_ = false;
  bool is_multicast_setup_ = false;
  CUmemGenericAllocationHandle mcast_handle_{};
  CUdevice cu_dev_{};
  void* mc_ptr_{nullptr};
  int exporter_rank_{-1};
  int peer_fd_{-1};
  bool is_contiguous_view_setup_ = false;
  at::Tensor contiguous_view_;
};

} // namespace nvfuser
