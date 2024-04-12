#pragma once

#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <c10/util/intrusive_ptr.h>

namespace c10d {
class Work : public torch::CustomClassHolder {
 public:
  void wait() {}
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
};

class TCPStore : public torch::CustomClassHolder {};

} // namespace c10d
