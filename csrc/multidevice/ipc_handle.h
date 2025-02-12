// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <cuda.h>
#include <expr_evaluator.h>
#include <ATen/core/TensorBody.h>

namespace nvfuser {

enum class IpcSemaphore : cuuint32_t { kReady, kInUse };

class IpcHandle {
 public:
  IpcHandle(at::Tensor tensor);
  IpcHandle(std::vector<uint8_t> data); // means it is imported
  ~IpcHandle();

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
  cudaIpcMemHandle_t ipc_handle_;
  cudaIpcMemHandle_t semaphore_ipc_handle_;
  IpcSemaphore* semaphore_;
  int64_t rank_;
};

class P2pIpcHandle {
 public:

  P2pIpcHandle(std::unique_ptr<IpcHandle> local, std::unique_ptr<IpcHandle> peer) : local_(std::move(local)), peer_(std::move(peer)) {}

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

// The cache key must be match on (dst, src, tensor, Id of SendComm, Id of RecvComm) or (int64_t dst, int64_t src, tensor, P2PCommunication*)
// we need a counter on Tensor+P2PCommunication* for each given dst, src
// In the store, we need the key to be computed on (dst, src, counter), also bc it cannot depend nor on tensor neither on P2PCommunication* (not even its ID)
// We could store separately the local and remote handles, or by first mapping with the IpcHandle's rank. Btw, we need to add rank to IpcHandle.
class IpcHandleCache {
 public:
  IpcHandleCache() = default;
  ~IpcHandleCache() = default;

  const P2pIpcHandle& get(P2PCommunication* communication, ExpressionEvaluator& expr_evaluator) const {
    auto it = find(communication, expr_evaluator);
    NVF_ERROR(
      it != nullptr,
      "No remote buffer found for ",
      communication->toString());
    return *it;
  }

  void exchangeHandles(const std::vector<P2PCommunication*>& communications, const ExpressionEvaluator& expr_evaluator);

 private:
  using KeyType = std::tuple<int64_t, int64_t, at::Tensor, P2PCommunication*>;

  KeyType getKey(P2PCommunication* comm, const ExpressionEvaluator& expr_evaluator) const  {
    int64_t dst = expr_evaluator.evaluate(comm->dst()).as<int64_t>();
    int64_t src = expr_evaluator.evaluate(comm->src()).as<int64_t>();
    at::Tensor buffer = expr_evaluator.evaluate(comm->buffer()).as<at::Tensor>();
    return std::make_tuple(dst, src, buffer, comm);
  }

  void insert(P2PCommunication* comm, const ExpressionEvaluator& expr_evaluator, std::unique_ptr<P2pIpcHandle> handle)  {
    handles_[getKey(comm, expr_evaluator)] = std::move(handle);
  }

  P2pIpcHandle* find(P2PCommunication* comm, const ExpressionEvaluator& expr_evaluator) const  {
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
      return std::hash<std::uintptr_t>()(ptr) ^ std::hash<int64_t>()(offset) << 32 ^
          std::hash<int>()(element_size);
    }
  };

  struct TensorEqual {
    bool operator()(const at::Tensor& lhs, const at::Tensor& rhs) const {
      return lhs.equal(rhs);
    }
  };

  struct KeyHash {
    std::size_t operator()(const KeyType& key) const {
      return (std::hash<int64_t>()(std::get<0>(key)) << 13) ^
         (std::hash<int64_t>()(std::get<1>(key)) << 7) ^
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

  std::unordered_map<
    KeyType,
    std::unique_ptr<P2pIpcHandle>,
    KeyHash,
    KeyEqual>
    handles_;
};

} // nvfuser
