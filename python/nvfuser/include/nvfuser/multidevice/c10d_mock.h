// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// This file provides a mock implementation of c10d that builds but doesn't
// function.
//
// nvFuser is sometimes built on a pytorch without c10d. When that
// happens, c10d isn't linked, NVFUSER_DISTRIBUTED is undefined and the
// multi-GPU component of nvFuser is expected to be disabled.
//
// Instead of adding `#ifdef NVFUSER_DISTRIBUTED` in too many places, this file
// provides a buildable mock implementation of c10d to keep nvFuser code less
// divergent. This implementation won't run because tests and user code are
// guarded by Communicator::is_available.

#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <c10/util/intrusive_ptr.h>

namespace c10d {

inline void setDebugLevelFromEnvironment() {}

class Work : public torch::CustomClassHolder {
 public:
  void wait() {}
};

struct ReduceOp : torch::CustomClassHolder {
  enum RedOpType {
    SUM,
    AVG,
    PRODUCT,
    MIN,
    MAX,
    BAND,
    BOR,
    BXOR,
    UNUSED,
  };

  ReduceOp() = default;
  ReduceOp(RedOpType op) : op_(op) {}

  RedOpType op_ = UNUSED;
};

struct ReduceScatterOptions {
  ReduceOp reduceOp = ReduceOp::UNUSED;
};

struct ScatterOptions {
  int64_t rootRank = 0;
};

struct AllgatherOptions {};

struct GatherOptions {
  int64_t rootRank = 0;
};

struct BroadcastOptions {
  int64_t rootRank = 0;
};

struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::UNUSED;
};

struct ReduceOptions {
  ReduceOp reduceOp = ReduceOp::UNUSED;
  int64_t rootRank = 0;
};

struct BarrierOptions {
  std::vector<int64_t> device_ids;
};

class Backend : public torch::CustomClassHolder {
 public:
  void startCoalescing() {}

  c10::intrusive_ptr<Work> endCoalescing() {
    return c10::make_intrusive<Work>();
  }

  const std::string getBackendName() const {
    return "";
  };

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) {
    return c10::make_intrusive<Work>();
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) {
    return c10::make_intrusive<Work>();
  }

  int getSize() const {
    return 0;
  }
};

struct TCPStoreOptions {
  static constexpr uint16_t kDefaultPort = 0;
};

class TCPStore : public torch::CustomClassHolder {
 public:
  std::vector<uint8_t> get(const std::string&) {
    return {};
  }

  void set(const std::string&, const std::vector<uint8_t>&) {}

  bool check(const std::vector<std::string>&) {
    return false;
  }

  bool deleteKey(const std::string&) {
    return false;
  }
};

} // namespace c10d
