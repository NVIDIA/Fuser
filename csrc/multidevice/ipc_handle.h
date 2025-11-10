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
#include <multidevice/symmetric_tensor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvfuser {

namespace hir {
class DistributedTensorContiguousAliasing;
} // namespace hir

// Semaphore values for P2P communication synchronization
enum class IpcSemaphore : cuuint32_t { kReady, kInUse };

// Basic IPC handle for legacy P2P communication using cudaIpc* APIs
// This class is kept for backward compatibility with non-VMM setups
class IpcHandle {
 public:
  NVF_API IpcHandle(at::Tensor tensor);
  NVF_API ~IpcHandle();

  // Constructor for importing a remote IPC handle
  NVF_API IpcHandle(std::vector<uint8_t> data);

  void* ptr() const {
    return ptr_;
  }

  auto semaphore() const {
    return semaphore_;
  }

 private:
  void* ptr_;
  // cudaIpcMemHandle always points to base address of the allocated buffer
  // Therefore we need to store the offset separately
  void* base_address_ = nullptr;
  int64_t offset_from_base_address_ = 0;
  cudaIpcMemHandle_t ipc_handle_ = {};
  cudaIpcMemHandle_t semaphore_ipc_handle_ = {};
  IpcSemaphore* semaphore_ = nullptr;
  int64_t rank_;
  // Keep a reference to prevent buffer from being freed
  at::Tensor tensor_;
};

// Wraps two IpcHandles involved in a P2P communication
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

// Helper structs for tensor hashing and equality
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

// Manages and caches IpcHandles for P2P communications
// Caching is based on (peer, tensor) runtime values and P2PCommunication* pointer
class IpcHandleCache {
 public:
  IpcHandleCache(const ExpressionEvaluator& expr_evaluator)
      : expr_evaluator_(expr_evaluator) {}
  ~IpcHandleCache() = default;

  // Create IpcHandles, import and export them, and populate the cache
  // Must be called before calling get(). Handles are exchanged in batch
  // to improve performance and avoid deadlocks when import/export orders differ
  void exchangeHandles(const std::vector<P2PCommunication*>& communications);

  // Retrieves a cached item (throws if not present)
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

// Base class for symmetric memory handles used in collective communications
// Symmetric memory handles enable efficient multi-device operations using
// CUDA VMM and NVLS multicast primitives
class SymmetricMemoryHandle {
 public:
  virtual ~SymmetricMemoryHandle() = default;
};

// SymmetricMemoryHandle for broadcast operations using NVLS multicast
// Provides efficient one-to-many communication with hardware acceleration
class MulticastHandleForBroadcast : public SymmetricMemoryHandle {
 public:
  MulticastHandleForBroadcast(Communication* communication, at::Tensor buffer);

  // Constructor for creating multiple broadcasts (e.g., for allgather)
  MulticastHandleForBroadcast(
      at::Tensor buffer,
      int64_t root,
      const std::string& name_suffix);

  ~MulticastHandleForBroadcast() = default;

  void* buffer_multicast_ptr() const;

  void* semaphore_multicast_ptr() const;

  void* semaphore_unicast_ptr(int64_t rank) const;

 private:
  // Buffer symmetric tensor with multicast support
  std::unique_ptr<SymmetricTensor> buffer_sym_tensor_;
  // Semaphore symmetric tensor with multicast support
  std::unique_ptr<SymmetricTensor> semaphore_sym_tensor_;
};

// SymmetricMemoryHandle for allgather operations using NVLS multicast
// Allgather is implemented as world_size broadcasts, each rank acting as root once
class MulticastHandleForAllgather : public SymmetricMemoryHandle {
 public:
  MulticastHandleForAllgather(Communication* communication, at::Tensor buffer);

  ~MulticastHandleForAllgather() override = default;

  // Accessors for a specific root rank's handles
  void* buffer_multicast_ptr(int64_t root_rank) const;

  void* semaphore_multicast_ptr(int64_t root_rank) const;

  void* semaphore_unicast_ptr(int64_t root_rank, int64_t rank) const;

 private:
  // One MulticastHandleForBroadcast per rank (each rank acts as root once)
  std::vector<std::unique_ptr<MulticastHandleForBroadcast>> broadcast_handles_;
};

// ContiguousAliasingHandle creates a contiguous virtual address mapping
// across all ranks for a sharded symmetric memory tensor
// This enables all ranks to access each other's data in a contiguous address space
class ContiguousAliasingHandle : public SymmetricMemoryHandle {
 public:
  ContiguousAliasingHandle(
      at::Tensor buffer,
      hir::DistributedTensorContiguousAliasing* expr);

  ~ContiguousAliasingHandle() override = default;

  // Returns the contiguous tensor that provides access to all ranks' data
  at::Tensor tensor() const {
    return tensor_;
  }

 private:
  at::Tensor tensor_;
};

// Cache for symmetric memory handles keyed by (buffer tensor, expr)
// Avoids recreating expensive VMM mappings and multicast handles
class SymmetricMemoryHandleCache {
 public:
  SymmetricMemoryHandleCache() = default;
  ~SymmetricMemoryHandleCache() = default;

  struct KeyType {
    at::Tensor buffer;
    Expr* expr;

    bool operator==(const KeyType& other) const {
      return TensorEqual{}(buffer, other.buffer) && expr == other.expr;
    }

    struct Hash {
      std::size_t operator()(const KeyType& key) const {
        return (TensorHash{}(key.buffer)) ^ (std::hash<Expr*>()(key.expr));
      }
    };
  };

  // Get or create a symmetric memory handle for the given key
  // Creates the handle on first access and caches it for future use
  SymmetricMemoryHandle* get(KeyType key);

 private:
  std::unordered_map<
      KeyType,
      std::unique_ptr<SymmetricMemoryHandle>,
      KeyType::Hash>
      handles_;
};

} // namespace nvfuser
