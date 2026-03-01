// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/TensorBody.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>

#include "exceptions.h"
#include "multidevice/communicator.h"
#include "visibility.h"

namespace nvfuser {

// Transfer direction. NIXL uses a one-sided model:
//   Read  = pull remote data into local buffers
//   Write = push local data into remote buffers
enum class NixlXferOp {
  kRead,
  kWrite,
};

enum class NixlXferStatus {
  kDone,
  kInProgress,
  kError,
};

// ------------------------------------------------------------------
// Todo - those functions should be moved to a more global file
// Helper functions for serializing and deserializing tensors descriptors for TCP store
struct TensorDesc {
  uintptr_t addr;
  size_t size;
  uint32_t dev;
};
static_assert(std::is_trivially_copyable_v<TensorDesc>,
  "TensorDesc must be trivially copyable for serialization");

inline TensorDesc toTensorDesc(const at::Tensor& tensor) {
  return {
    .addr = reinterpret_cast<uintptr_t>(tensor.data_ptr()),
    .size = static_cast<size_t>(tensor.numel()) * tensor.element_size(),
    .dev = static_cast<uint32_t>(tensor.device().index())
  };
}

inline at::Tensor fromTensorDesc(const TensorDesc& desc) {
  /*
  Tensors must be valid on this device
  */
  return at::from_blob(
    reinterpret_cast<void*>(desc.addr),
    {static_cast<int64_t>(desc.size)},
    at::TensorOptions().device(at::Device(at::kCUDA, desc.dev)).dtype(at::kByte)
  );
}

inline std::vector<uint8_t> serializeTensorsDescs(
    const std::vector<TensorDesc>& descs) {
  size_t count = descs.size();
  std::vector<uint8_t> buf(sizeof(count) + count * sizeof(TensorDesc));
  std::memcpy(buf.data(), &count, sizeof(count));
  if (count == 0)
    return buf;
  
  std::memcpy(
      buf.data() + sizeof(count),
      descs.data(),
      descs.size() * sizeof(TensorDesc));
  return buf;
}

inline std::vector<TensorDesc> deserializeTensorsDescs(
    const std::vector<uint8_t>& buf) {
  NVF_ERROR(buf.size() >= sizeof(size_t), "Invalid serialized descriptor data");
  size_t count;
  std::memcpy(&count, buf.data(), sizeof(count));
  NVF_ERROR(
      buf.size() == sizeof(count) + count * sizeof(TensorDesc),
      "Corrupted serialized descriptor data");

  std::vector<TensorDesc> descs(count);
  if (count > 0) {
    std::memcpy(
        descs.data(),
        buf.data() + sizeof(count),
        count * sizeof(TensorDesc));
  }
  return descs;
}

inline void storeTensorDescs(Communicator& communicator, const std::string& key, const std::vector<TensorDesc>& descs) {
  NVF_CHECK(communicator.is_available(), "Communicator is not available");
  communicator.getTcpStore()->set(key, serializeTensorsDescs(descs));
}

inline void storeTensorDescs(Communicator& communicator, const std::string& key, const std::vector<at::Tensor>& tensors) {
  std::vector<TensorDesc> descs;
  descs.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    descs.push_back(toTensorDesc(tensor));
  }
  storeTensorDescs(communicator, key, descs);
}

inline std::vector<TensorDesc> fetchTensorDescs(Communicator& communicator, const std::string& key) {
  NVF_CHECK(communicator.is_available(), "Communicator is not available");
  auto bytes = communicator.getTcpStore()->get(key);
  return deserializeTensorsDescs(bytes);
}

// End of Todo - those functions should be moved to a more global file
// ------------------------------------------------------------------

// -------------------------------------------------------------------
// NixlTransferHandle: opaque handle for a prepared transfer
// -------------------------------------------------------------------
// Returned by NixlBackend::prepareTransfer(). Callers hold this handle
// and pass it to postTransfer() / waitTransfer(). The actual NIXL
// transfer handle lives inside the impl; this is just an owning wrapper.
class NixlTransferHandleImpl;

class NVF_API NixlTransferHandle {
 public:
  NixlTransferHandle();
  ~NixlTransferHandle();
  NixlTransferHandle(NixlTransferHandle&&) noexcept;
  NixlTransferHandle& operator=(NixlTransferHandle&&) noexcept;

  NixlTransferHandle(const NixlTransferHandle&) = delete;
  NixlTransferHandle& operator=(const NixlTransferHandle&) = delete;

  [[nodiscard]] bool isValid() const;

 private:
  friend class NixlBackend;
  std::unique_ptr<NixlTransferHandleImpl> impl_;
};

// -------------------------------------------------------------------
// NixlBackend: singleton NIXL backend over UCX for GPU tensors
// -------------------------------------------------------------------
// Singleton - Wraps a nixlAgent with the UCX backend and provides a tensor-level
// API for registering GPU memory and performing RDMA transfers.
//
// Lifecycle:
//   1. getInstance()      - creates agent, loads UCX backend
//   2. registerTensors()  - register GPU tensors for RDMA access
//   3. exchangeMetadata() - all ranks share their registration info
//   4. prepareTransfer()  - expensive one-time setup per transfer pattern
//   5. postTransfer()     - cheap, non-blocking data movement
//   6. waitTransfer()     - block until complete
//
// Thread safety: methods are NOT thread-safe. The caller must
// synchronize if the same NixlBackend is used from multiple threads.
class NixlBackend {
 public:
  static NixlBackend& getInstance();

  NixlBackend(const NixlBackend&) = delete;
  NixlBackend& operator=(const NixlBackend&) = delete;
  ~NixlBackend() = delete;

  // Explicitly tear down the singleton. Must be called before program
  // exit (same pattern as Communicator::cleanup).
  void cleanup();

  [[nodiscard]] bool isAvailable() const;

  // ------------------------------------------------------------------
  // Memory registration
  // ------------------------------------------------------------------

  // Register CUDA tensors with the NIXL agent so they can participate
  // in RDMA transfers. Tensors must be contiguous and remain alive
  // until deregisterTensors() is called.
  void registerTensors(const std::vector<at::Tensor>& tensors);

  void deregisterTensors(const std::vector<at::Tensor>& tensors);

  // ------------------------------------------------------------------
  // Metadata exchange
  // ------------------------------------------------------------------
  // Exchange local agent metadata with all peers through the TCPStore.
  // Must be called after registerTensors() and before prepareTransfer()
  // whenever the set of registered tensors changes.
  void exchangeMetadata();

  // ------------------------------------------------------------------
  // Transfer lifecycle
  // ------------------------------------------------------------------

  // Prepare a transfer between pairs of tensors.
  // local_tensors[i] and remote_tensors[i] must have the same byte size.
  // All tensors must be contiguous CUDA tensors and previously registered.
  // The returned handle can be posted multiple times (preparation is
  // amortized).
  [[nodiscard]] NixlTransferHandle prepareTransfer(
      const std::vector<TensorDesc>& local_descs,
      const std::vector<TensorDesc>& remote_descs,
      int64_t remote_rank,
      NixlXferOp op);

  // Post a previously prepared transfer for execution (non-blocking).
  void postTransfer(NixlTransferHandle& handle);

  // Poll the status of a posted transfer without blocking.
  [[nodiscard]] NixlXferStatus getTransferStatus(const NixlTransferHandle& handle) const;

  // Block until the transfer completes (or errors out).
  void waitTransfer(NixlTransferHandle& handle);

 private:
  NixlBackend();
  bool cleaned_up_ = false;

  class Impl;
  std::unique_ptr<Impl> impl_;
};




} // namespace nvfuser
