// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <ATen/core/TensorBody.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <expr_evaluator.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace nvfuser {

enum class IpcSemaphore : cuuint32_t { kReady, kInUse };

// The class IpcHandle represents a cuda buffer that can be exported/imported to
// remote devices. It comes with a semaphore that is allocated on the buffer's
// device.
class IpcHandle {
 public:
  NVF_API IpcHandle(at::Tensor tensor);
  NVF_API ~IpcHandle();

  // This constructor is used when importing a remote Ipc Handle
  NVF_API IpcHandle(std::vector<uint8_t> data);

  void* ptr() const {
    return ptr_;
  }

  auto semaphore() const {
    return semaphore_;
  }

 private:
  void* ptr_;
  // a cudaIpcMemHandle always points to the base address of the allocated
  // buffer. Therefore we need to store the offset separately
  void* base_address_ = nullptr;
  int64_t offset_from_base_address_ = 0;
  cudaIpcMemHandle_t ipc_handle_ = {};
  cudaIpcMemHandle_t semaphore_ipc_handle_ = {};
  IpcSemaphore* semaphore_ = nullptr;
  int64_t rank_;
  // we keep a reference of the tensor to prevent the cuda buffer to be freed
  // before the IpcHandle gets destroyed
  at::Tensor tensor_;
};

// This class wraps two IpcHandles involved in a P2P Communication
class P2pIpcHandle {
 public:
  P2pIpcHandle(
      std::unique_ptr<IpcHandle> local,
      std::unique_ptr<IpcHandle> peer)
      : local_(std::move(local)), peer_(std::move(peer)) {}

  const auto& local() const {
    return *local_;
  }

  const auto& peer() const {
    return *peer_;
  }

 private:
  std::unique_ptr<IpcHandle> local_;
  std::unique_ptr<IpcHandle> peer_;
};

namespace {

struct TensorHash {
  std::size_t operator()(const at::Tensor& tensor) const {
    auto ptr = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    auto offset = tensor.storage_offset();
    auto element_size = tensor.element_size();
    auto numel = tensor.numel();
    return std::hash<std::uintptr_t>()(ptr) ^ std::hash<int64_t>()(offset) ^
        std::hash<int64_t>()(element_size) ^ std::hash<int64_t>()(numel);
  }
};

struct TensorEqual {
  bool operator()(const at::Tensor& lhs, const at::Tensor& rhs) const {
    return lhs.equal(rhs);
  }
};

} // namespace

// IpcHandleCache manages and cache the IpcHandles.
// Caching is done on the runtime values of (peer, tensor) and the
// P2PCommunication* pointer.
class IpcHandleCache {
 public:
  IpcHandleCache(const ExpressionEvaluator& expr_evaluator)
      : expr_evaluator_(expr_evaluator) {}
  ~IpcHandleCache() = default;

  // Create IpcHandles, import and export them, and populate the cache. This
  // method must be called priori to calling get. In many case, the handles need
  // to be exported by batch (thus the function taking a vector of
  // P2PCommunication*) to improve performance and to avoid creating deadlocks
  // when imports and exports order differ accross ranks.
  void exchangeHandles(const std::vector<P2PCommunication*>& communications);

  // Retrieves a cached item and throws if not present
  const P2pIpcHandle& get(P2PCommunication* communication) const {
    auto it = find(communication);
    NVF_ERROR(
        it != nullptr,
        "No remote buffer found for ",
        communication->toString());
    return *it;
  }

 private:
  struct KeyType {
    int64_t peer;
    at::Tensor buffer;
    P2PCommunication* comm;

    bool operator==(const KeyType& other) const {
      return peer == other.peer && TensorEqual{}(buffer, other.buffer) &&
          comm == other.comm;
    }

    struct Hash {
      std::size_t operator()(const KeyType& key) const {
        return (std::hash<int64_t>()(key.peer)) ^ (TensorHash{}(key.buffer)) ^
            (std::hash<P2PCommunication*>()(key.comm));
      }
    };
  };

  void insert(P2PCommunication* comm, std::unique_ptr<P2pIpcHandle> handle) {
    handles_[getKey(comm)] = std::move(handle);
  }

  P2pIpcHandle* find(P2PCommunication* comm) const {
    auto it = handles_.find(getKey(comm));
    if (it == handles_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  KeyType getKey(P2PCommunication* comm) const {
    auto peer = expr_evaluator_.evaluate(comm->peer()).as<int64_t>();
    auto buffer = expr_evaluator_.evaluate(comm->buffer()).as<at::Tensor>();
    return KeyType{peer, buffer, comm};
  }

  std::string getTcpStoreKey(P2PCommunication* communication, int64_t rank)
      const;

  const ExpressionEvaluator& expr_evaluator_;
  std::unordered_map<KeyType, std::unique_ptr<P2pIpcHandle>, KeyType::Hash>
      handles_;
};

// UnicastHandle allows a single rank (exporter) to share its buffer with
// multiple other ranks (importers) using IPC handles.
class UnicastHandle {
 public:
  // Exporter constructor: creates an IPC handle for the tensor and exports it
  // to the store
  NVF_API UnicastHandle(
      at::Tensor tensor,
      int64_t exporter_rank,
      const std::string& store_key_prefix);

  // Importer constructor: imports an IPC handle from the store
  NVF_API UnicastHandle(
      int64_t exporter_rank,
      const std::string& store_key_prefix);

  NVF_API ~UnicastHandle();

  void* ptr() const {
    return ptr_;
  }

 private:
  void* ptr_ = nullptr;
  // VMM-related members (importer only)
  size_t size_ = 0;
  CUmemGenericAllocationHandle mem_handle_ = 0;
  // Keep a reference to the tensor to prevent the cuda buffer from being freed
  // before the UnicastHandle gets destroyed. Only set for exporter.
  at::Tensor tensor_;
  // File descriptors for cleanup (importer only)
  int pid_fd_ = -1;
  int peer_fd_ = -1;
};

// MulticastHandle creates and shares a multicast object across all ranks,
// and maps an address to that multicast object.
class MulticastHandle {
 public:
  // Creates a multicast object for the given tensor and shares it across all
  // ranks. Bind the multicast object to the tensor. The tensor must be
  // allocated with symmetric memory.
  MulticastHandle(
      at::Tensor tensor,
      int64_t exporter_rank,
      const std::string& store_key_prefix,
      int64_t offset = 0);

  ~MulticastHandle();

  void* multicast_ptr() const {
    return mc_ptr_;
  }

 private:
  CUmemGenericAllocationHandle mcast_handle_{};
  CUdevice cu_dev_{};
  void* mc_ptr_{nullptr};
  int64_t size_{0};
  at::Tensor tensor_;
};

class MulticastHandleForBroadcast {
 public:
  MulticastHandleForBroadcast(Communication* communication, at::Tensor buffer);
  
  // Constructor for use when creating multiple broadcasts (e.g., for allgather)
  MulticastHandleForBroadcast(
      at::Tensor buffer,
      int64_t root,
      const std::string& name_suffix,
      int64_t offset = 0);

  ~MulticastHandleForBroadcast() = default;

  void* buffer_multicast_ptr() const {
    return buffer_multicast_handle_->multicast_ptr();
  }

  void* semaphore_multicast_ptr() const {
    return semaphore_multicast_handle_->multicast_ptr();
  }

  void* semaphore_unicast_ptr(int64_t rank) const {
    return semaphore_handles_[rank]->ptr();
  }

 private:
  std::unique_ptr<MulticastHandle> buffer_multicast_handle_;
  std::unique_ptr<MulticastHandle> semaphore_multicast_handle_;
  // Per-rank semaphores: each rank exports its own semaphore using
  // UnicastHandle
  std::vector<std::unique_ptr<UnicastHandle>> semaphore_handles_;
};

class MulticastHandleForAllgather {
 public:
  MulticastHandleForAllgather(Communication* communication, at::Tensor buffer);

  ~MulticastHandleForAllgather() = default;

  // Accessors for a specific root rank's handles
  void* buffer_multicast_ptr(int64_t root_rank) const {
    return broadcast_handles_[root_rank]->buffer_multicast_ptr();
  }

  void* semaphore_multicast_ptr(int64_t root_rank) const {
    return broadcast_handles_[root_rank]->semaphore_multicast_ptr();
  }

  void* semaphore_unicast_ptr(int64_t root_rank, int64_t rank) const {
    return broadcast_handles_[root_rank]->semaphore_unicast_ptr(rank);
  }

 private:
  // Allgather is world_size broadcasts, each broadcasting a different slice
  // One MulticastHandleForBroadcast per rank (each rank acts as root once)
  std::vector<std::unique_ptr<MulticastHandleForBroadcast>> broadcast_handles_;
};

// Variant type that can hold either type of multicast handle
using SymmetricMemoryHandle = std::variant<
    std::unique_ptr<MulticastHandleForBroadcast>,
    std::unique_ptr<MulticastHandleForAllgather>>;

class MulticastHandleCache {
 public:
  MulticastHandleCache() = default;
  ~MulticastHandleCache() = default;

  struct KeyType {
    at::Tensor buffer;
    Communication* comm;

    bool operator==(const KeyType& other) const {
      return TensorEqual{}(buffer, other.buffer) && comm == other.comm;
    }

    struct Hash {
      std::size_t operator()(const KeyType& key) const {
        return (TensorHash{}(key.buffer)) ^
            (std::hash<Communication*>()(key.comm));
      }
    };
  };

  const SymmetricMemoryHandle& get(KeyType key);

 private:
  std::unordered_map<KeyType, SymmetricMemoryHandle, KeyType::Hash>
      handles_;
};

} // namespace nvfuser
