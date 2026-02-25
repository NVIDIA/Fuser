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

  bool isValid() const;

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

  bool isAvailable() const;

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
  NixlTransferHandle prepareTransfer(
      const std::vector<at::Tensor>& local_tensors,
      const std::vector<at::Tensor>& remote_tensors,
      int64_t remote_rank,
      NixlXferOp op);

  // Post a previously prepared transfer for execution (non-blocking).
  void postTransfer(NixlTransferHandle& handle);

  // Poll the status of a posted transfer without blocking.
  NixlXferStatus getTransferStatus(const NixlTransferHandle& handle) const;

  // Block until the transfer completes (or errors out).
  void waitTransfer(NixlTransferHandle& handle);

 private:
  NixlBackend();

  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nvfuser
