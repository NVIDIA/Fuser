// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
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
  int64_t storage_offset_;
  int64_t element_size_;
  cudaIpcMemHandle_t ipc_handle_ = {};
  cudaIpcMemHandle_t semaphore_ipc_handle_ = {};
  IpcSemaphore* semaphore_ = nullptr;
  int64_t rank_;
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
// Caching is done on the runtime values of (dst, src, tensor) and the
// P2PCommunication* pointer.
class IpcHandleCache {
 public:
  IpcHandleCache() = default;
  ~IpcHandleCache() = default;

  // Create IpcHandles, import and export them, and populate the cache. This
  // method must be called priori to calling get. In many case, the handles need
  // to be exported by batch (thus the function taking a vector of
  // P2PCommunication*) to improve performance and to avoid creating deadlocks
  // when imports and exports order differ accross ranks.
  void exchangeHandles(
      const std::vector<P2PCommunication*>& communications,
      const ExpressionEvaluator& expr_evaluator);

  // Retrieves a cached item and throws if not present
  const P2pIpcHandle& get(
      P2PCommunication* communication,
      ExpressionEvaluator& expr_evaluator) const {
    auto it = find(communication, expr_evaluator);
    NVF_ERROR(
        it != nullptr,
        "No remote buffer found for ",
        communication->toString());
    return *it;
  }

 private:
  using KeyType = std::tuple<int64_t, int64_t, at::Tensor, P2PCommunication*>;

  KeyType getKey(
      P2PCommunication* comm,
      const ExpressionEvaluator& expr_evaluator) const {
    int64_t dst = expr_evaluator.evaluate(comm->dst()).as<int64_t>();
    int64_t src = expr_evaluator.evaluate(comm->src()).as<int64_t>();
    at::Tensor buffer =
        expr_evaluator.evaluate(comm->buffer()).as<at::Tensor>();
    return std::make_tuple(dst, src, buffer, comm);
  }

  void insert(
      P2PCommunication* comm,
      const ExpressionEvaluator& expr_evaluator,
      std::unique_ptr<P2pIpcHandle> handle) {
    handles_[getKey(comm, expr_evaluator)] = std::move(handle);
  }

  P2pIpcHandle* find(
      P2PCommunication* comm,
      const ExpressionEvaluator& expr_evaluator) const {
    auto it = handles_.find(getKey(comm, expr_evaluator));
    if (it == handles_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  struct TensorHash {
    std::size_t operator()(const at::Tensor& tensor) const {
      auto ptr = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
      auto offset = tensor.storage_offset();
      auto element_size = tensor.element_size();
      return std::hash<std::uintptr_t>()(ptr) ^ std::hash<int64_t>()(offset) ^
          std::hash<int64_t>()(element_size);
    }
  };

  struct TensorEqual {
    bool operator()(const at::Tensor& lhs, const at::Tensor& rhs) const {
      return lhs.equal(rhs);
    }
  };

  struct KeyHash {
    std::size_t operator()(const KeyType& key) const {
      return (std::hash<int64_t>()(std::get<0>(key))) ^
          (std::hash<int64_t>()(std::get<1>(key))) ^
          (TensorHash{}(std::get<2>(key))) ^
          (std::hash<P2PCommunication*>()(std::get<3>(key)));
    }
  };

  struct KeyEqual {
    bool operator()(const KeyType& lhs, const KeyType& rhs) const {
      return std::get<0>(lhs) == std::get<0>(rhs) &&
          std::get<1>(lhs) == std::get<1>(rhs) &&
          TensorEqual{}(std::get<2>(lhs), std::get<2>(rhs)) &&
          std::get<3>(lhs) == std::get<3>(rhs);
    }
  };

  std::unordered_map<KeyType, std::unique_ptr<P2pIpcHandle>, KeyHash, KeyEqual>
      handles_;
  std::unordered_set<std::string> keys_;
};

} // namespace nvfuser
