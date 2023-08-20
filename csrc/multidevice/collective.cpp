// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED

#include <multidevice/collective.h>

namespace nvfuser {

static inline void doLocalCopy(at::Tensor dst, at::Tensor src) {
  // TODO: can we make it non-blocking?
  at::copy(dst, src, /*non_blocking*/ true);
}

static inline std::vector<std::vector<at::Tensor>> setBufList(
    std::vector<at::Tensor>& bufs) {
  return {std::move(bufs)};
}

static inline void assertBufferCount(
    std::vector<at::Tensor> bufs,
    size_t count) {
  TORCH_INTERNAL_ASSERT(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

static inline void assertBuffersHaveSameSize(std::vector<at::Tensor> bufs) {
  if (bufs.size() < 2) {
    return;
  }
  auto sizes = bufs.at(0).sizes();
  for (auto& buf : bufs) {
    TORCH_INTERNAL_ASSERT(
        buf.sizes() == sizes, "all buffers must have the same size");
  }
}

void Collective::post() {
  if (!is_init_) {
    init();
    is_init_ = true;
  }

  TORCH_INTERNAL_ASSERT(
      !work_, "the collective must complete before being posted again");
  post_specialized();
};

bool Collective::test() {
  if (!work_)
    return true;
  bool ret = work_->isCompleted();
  if (ret) {
    work_ = nullptr;
  }
  return ret;
}

void Collective::wait() {
  if (!work_)
    return;
  TORCH_INTERNAL_ASSERT(work_->wait(), "the collective has been aborted");
  work_ = nullptr;
}

void Collective::init() {
  assertBuffersHaveSameSize(src_bufs_);
  assertBuffersHaveSameSize(dst_bufs_);

  TORCH_INTERNAL_ASSERT(
      std::unique(team_.begin(), team_.end()) == team_.end(),
      "the collective must not involve the same device more than once");
  TORCH_INTERNAL_ASSERT(
      comm_,
      "the communicator of the collective must be set before using the method setCommunicator");
  TORCH_INTERNAL_ASSERT(
      std::find(team_.begin(), team_.end(), comm_->deviceId()) != team_.end(),
      "current device index ",
      comm_->deviceId(),
      " must be present in the collective's team");
  TORCH_INTERNAL_ASSERT(!team_.empty(), "the team is empty");
  backend_ = comm_->getTeam(team_);
  if (has_root_) {
    auto it = std::find(team_.begin(), team_.end(), root_);
    TORCH_INTERNAL_ASSERT(
        it != team_.end(),
        "root (device ",
        root_,
        ") must be present in the collective's team");
    root_rank_ = std::distance(team_.begin(), it);
  }

  init_specialized();
}

std::string Collective::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Collective " << collective_type_ << ": {\n";

  if (has_root_) {
    ss << indent1 << "root: " << root_ << ",\n";
  }
  ss << indent1 << "team: {";
  for (auto r : team_) {
    ss << r << ", ";
  }
  ss << indent1 << "}\n";
  ss << indent1 << "src_bufs: {";
  for (auto& t : src_bufs_) {
    ss << "\n" << t;
  }
  ss << "\n" << indent1 << "}\n";
  ss << ext_indent << "}";

  return ss.str();
}

void Broadcast::post_specialized() {
  if (comm_->deviceId() == root_ && !dst_bufs_.empty()) {
    doLocalCopy(dst_bufs_.at(0), src_bufs_.at(0));
    if (team_.size() == 1)
      return;
  }
  work_ = backend_->broadcast(
      comm_->deviceId() == root_ ? src_bufs_ : dst_bufs_,
      {.rootRank = root_rank_});
}

void Broadcast::init_specialized() {
  if (comm_->deviceId() == root_) {
    assertBufferCount(src_bufs_, 1);
    assertBufferCount(dst_bufs_, 0);
  } else {
    assertBufferCount(src_bufs_, 0);
    assertBufferCount(dst_bufs_, 1);
  }
}

void Gather::post_specialized() {
  work_ = backend_->gather(buf_list_, src_bufs_, {.rootRank = root_rank_});
}

void Gather::init_specialized() {
  assertBufferCount(src_bufs_, 1);
  if (comm_->deviceId() == root_) {
    assertBufferCount(dst_bufs_, team_.size());
    buf_list_ = setBufList(dst_bufs_);
  } else {
    assertBufferCount(dst_bufs_, 0);
  }
}

void Allgather::post_specialized() {
  work_ = backend_->allgather(buf_list_, src_bufs_, {});
}

void Allgather::init_specialized() {
  assertBufferCount(src_bufs_, 1);
  assertBufferCount(dst_bufs_, team_.size());
  buf_list_ = setBufList(dst_bufs_);
}

void Scatter::post_specialized() {
  work_ = backend_->scatter(dst_bufs_, buf_list_, {.rootRank = root_rank_});
}

void Scatter::init_specialized() {
  assertBufferCount(dst_bufs_, 1);
  if (comm_->deviceId() == root_) {
    assertBufferCount(src_bufs_, team_.size());
    buf_list_ = setBufList(src_bufs_);
  } else {
    assertBufferCount(src_bufs_, 0);
  }
}

} // namespace nvfuser

#endif
