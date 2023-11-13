// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#include <multidevice/communication.h>

namespace nvfuser {
namespace {

inline void assertBufferCount(
    const std::vector<at::Tensor>& bufs,
    size_t count) {
  NVF_ERROR(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

inline void assertBuffersHaveSameSize(
    const std::vector<at::Tensor>& bufs1,
    const std::vector<at::Tensor>& bufs2) {
  if (bufs1.empty() && bufs2.empty()) {
    return;
  }
  auto sizes = (bufs1.empty() ? bufs2 : bufs1).at(0).sizes();
  for (auto& bufs : {bufs1, bufs2}) {
    for (auto& buf : bufs) {
      NVF_ERROR(buf.sizes() == sizes, "all buffers must have the same size");
    }
  }
}

inline void post_common(Communication& self, Communicator& comm) {
  NVF_ERROR(
      std::find(
          self.params().team.begin(),
          self.params().team.end(),
          comm.deviceId()) != self.params().team.end(),
      "current device index ",
      comm.deviceId(),
      " must be present in the communication's team");
}

inline void doLocalCopy(const at::Tensor& dst, const at::Tensor& src) {
  dst.copy_(src, /* non-blocking */ true);
}

} // namespace

Communication::Communication(CommParams params, std::string name, bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      std::unique(params_.team.begin(), params_.team.end()) ==
          params_.team.end(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(!params_.team.empty(), "the team size must be greater than 0");
  if (has_root_) {
    auto it = std::find(params_.team.begin(), params_.team.end(), params_.root);
    NVF_ERROR(
        it != params_.team.end(),
        "root (device ",
        params_.root,
        ") must be present in the communication's team");
    // pytorch's process group expects the root to be specified
    // as an integer between 0 and world_size-1. We choose it to be
    // the device's relative index within the team
    root_relative_index_ = std::distance(params_.team.begin(), it);
  }
}

std::string Communication::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Communication " << collective_type_ << ": {\n";

  if (has_root_) {
    ss << indent1 << "root: " << params_.root << ",\n";
  }
  ss << indent1 << "team: {";
  for (auto r : params_.team) {
    ss << r << ", ";
  }
  ss << indent1 << "}\n";
  ss << indent1 << "src_bufs: {";
  for (auto& t : params_.src_bufs) {
    ss << "\n" << t;
  }
  ss << "\n" << indent1 << "}\n";
  ss << ext_indent << "}";

  return ss.str();
}

Broadcast::Broadcast(CommParams params) : Communication(params, "broadcast") {}

c10::intrusive_ptr<c10d::Work> Broadcast::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    if (params_.dst_bufs.size() == 1) {
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
    } else {
      assertBufferCount(params_.dst_bufs, 0);
    }
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  if (params_.team.size() == 1) {
    return nullptr;
  }

  return comm.getBackendForTeam(params_.team, backend)
      ->broadcast(
          comm.deviceId() == params_.root ? params_.src_bufs : params_.dst_bufs,
          {.rootRank = root_relative_index_});
}

Gather::Gather(CommParams params) : Communication(params, "gather") {
  assertBufferCount(params_.src_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Gather::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.dst_bufs, params_.team.size());
    buf_list = {std::move(params_.dst_bufs)};
  } else {
    assertBufferCount(params_.dst_bufs, 0);
  }
  auto work =
      comm.getBackendForTeam(params_.team, backend)
          ->gather(
              buf_list, params_.src_bufs, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.dst_bufs = std::move(buf_list.back());
  }
  return work;
}

Allgather::Allgather(CommParams params)
    : Communication(params, "allgather", false) {
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, params_.team.size());
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Allgather::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  buf_list = {std::move(params_.dst_bufs)};
  auto work = comm.getBackendForTeam(params_.team, backend)
                  ->allgather(buf_list, params_.src_bufs, {});
  params_.dst_bufs = std::move(buf_list.back());
  return work;
}

Scatter::Scatter(CommParams params) : Communication(params, "scatter") {
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Scatter::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, params_.team.size());
    buf_list = {std::move(params_.src_bufs)};
  } else {
    assertBufferCount(params_.src_bufs, 0);
  }
  auto work =
      comm.getBackendForTeam(params_.team, backend)
          ->scatter(
              params_.dst_bufs, buf_list, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.src_bufs = std::move(buf_list.back());
  }
  return work;
}

Reduce::Reduce(CommParams params) : Communication(params, "reduce") {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  assertBufferCount(params_.src_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Reduce::post(Communicator& comm) {
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.dst_bufs, 1);
  } else {
    assertBufferCount(params_.dst_bufs, 0);
  }
  post_common(*this, comm);
  auto backend = comm.getBackendForTeam(params_.team);
#ifdef USE_C10D_NCCL
  auto nccl_backend = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
#else
  constexpr bool nccl_backend = false;
#endif
  auto& buf =
      (comm.deviceId() == params_.root) ? params_.dst_bufs : params_.src_bufs;
  c10d::ReduceOptions options = {
      .reduceOp = params_.redOp, .rootRank = root_relative_index_};
  if (nccl_backend) {
#ifdef USE_C10D_NCCL
    return nccl_backend->_reduce_oop(buf, params_.src_bufs, options);
#endif
  } else {
    if (comm.deviceId() == params_.root) {
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
    }
    return backend->reduce(buf, options);
  }
}

Allreduce::Allreduce(CommParams params)
    : Communication(params, "allreduce", false) {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Allreduce::post(Communicator& comm) {
  post_common(*this, comm);
  doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
  return comm.getBackendForTeam(params_.team)
      ->allreduce(params_.dst_bufs, {.reduceOp = params_.redOp});
}

ReduceScatter::ReduceScatter(CommParams params)
    : Communication(params, "reduce_scatter", false) {
  assertBufferCount(params_.src_bufs, params_.team.size());
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> ReduceScatter::post(Communicator& comm) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list = {std::move(params_.src_bufs)};
  auto work = comm.getBackendForTeam(params_.team)
                  ->reduce_scatter(
                      params_.dst_bufs, buf_list, {.reduceOp = params_.redOp});
  params_.src_bufs = std::move(buf_list.back());
  return work;
}

SendRecv::SendRecv(CommParams params) : Communication(params, "send/recv") {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      params_.team.size() == 1 || params_.team.size() == 2,
      "the team size should be 1 or 2");
}

c10::intrusive_ptr<c10d::Work> SendRecv::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    if (params_.team.size() == 1) {
      assertBufferCount(params_.dst_bufs, 1);
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
      return nullptr;
    } else {
      assertBufferCount(params_.dst_bufs, 0);
    }
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  return comm.sendRecv(
      (params_.team.at(0) == params_.root) ? params_.team.at(1)
                                           : params_.team.at(0),
      params_.root,
      params_.dst_bufs.empty() ? params_.src_bufs : params_.dst_bufs,
      backend);
}

} // namespace nvfuser

#endif
