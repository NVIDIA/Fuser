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

namespace nvfuser {

enum class IpcSemaphore : cuuint32_t { kReady, kInUse };

// The class IpcHandle represents a cuda buffer that can be exported/imported to
// remote devices. It comes with a semaphore that is allocated on the buffer's
// device.
class IpcHandle {
 public:
  IpcHandle(at::Tensor tensor);
  ~IpcHandle();

  // This constructor is used when importing a remote Ipc Handle
  IpcHandle(std::vector<uint8_t> data);

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

} // namespace nvfuser
