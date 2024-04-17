#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <c10/util/intrusive_ptr.h>

namespace c10d {
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

class Backend : public torch::CustomClassHolder {
 public:
  c10::intrusive_ptr<Work> barrier() {
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
};

class TCPStore : public torch::CustomClassHolder {};

} // namespace c10d
