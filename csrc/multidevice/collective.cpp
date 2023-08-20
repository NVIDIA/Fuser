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

static inline std::vector<std::vector<at::Tensor>> setBufList(
    std::vector<at::Tensor> bufs) {
  return {std::move(bufs)};
}

static inline void assertBufferCount(
    const std::vector<at::Tensor>& bufs,
    size_t count) {
  TORCH_INTERNAL_ASSERT(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

static inline void assertBuffersHaveSameSize(
    const std::vector<at::Tensor>& bufs1,
    const std::vector<at::Tensor>& bufs2) {
  if (bufs1.empty() && bufs2.empty()) {
    return;
  }
  auto sizes = (bufs1.empty() ? bufs2 : bufs1).at(0).sizes();
  for (auto& bufs : {bufs1, bufs2}) {
    for (auto& buf : bufs) {
      TORCH_INTERNAL_ASSERT(
          buf.sizes() == sizes, "all buffers must have the same size");
    }
  }
}

Collective::Collective(
    CollectiveParams params,
    std::string name,
    bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  TORCH_INTERNAL_ASSERT(
      std::unique(params_.team.begin(), params_.team.end()) ==
          params_.team.end(),
      "the collective must not involve the same device more than once");
  TORCH_INTERNAL_ASSERT(
      params_.team.size() > 1, "the team size must be greater than 1");
  if (has_root_) {
    auto it = std::find(params_.team.begin(), params_.team.end(), params_.root);
    TORCH_INTERNAL_ASSERT(
        it != params_.team.end(),
        "root (device ", params_.root,
        ") must be present in the collective's team");
    root_rank_ = std::distance(params_.team.begin(), it);
  }
}

void Collective::post_common(Communicator& comm) {
  TORCH_INTERNAL_ASSERT(
      std::find(params_.team.begin(), params_.team.end(), comm.deviceId()) !=
          params_.team.end(),
      "current device index ",
      comm.deviceId(),
      " must be present in the collective's team");
}

std::string Collective::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Collective " << collective_type_ << ": {\n";

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

Broadcast::Broadcast(CollectiveParams params)
    : Collective(params, "broadcast") {}

c10::intrusive_ptr<c10d::Work> Broadcast::post(Communicator& comm) {
  post_common(comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    assertBufferCount(params_.dst_bufs, 0);
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  return comm.getTeam(params_.team)
                       ->broadcast(
                           comm.deviceId() == params_.root ? params_.src_bufs
                                                           : params_.dst_bufs,
                           {.rootRank = root_rank_});
}

Gather::Gather(CollectiveParams params) : Collective(params, "gather") {
  assertBufferCount(params_.src_bufs, 1);
}

c10::intrusive_ptr<c10d::Work> Gather::post(Communicator& comm) {
  post_common(comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.dst_bufs, params_.team.size());
    buf_list_ = setBufList(params_.dst_bufs);
  } else {
    assertBufferCount(params_.dst_bufs, 0);
  }

  return comm.getTeam(params_.team)
          ->gather(buf_list_, params_.src_bufs, {.rootRank = root_rank_});
}

Allgather::Allgather(CollectiveParams params)
    : Collective(params, "allgather", false) {
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, params_.team.size());
  buf_list_ = setBufList(params_.dst_bufs);
}

c10::intrusive_ptr<c10d::Work> Allgather::post(Communicator& comm) {
  post_common(comm);
  return comm.getTeam(params_.team)->allgather(buf_list_, params_.src_bufs, {});
}

Scatter::Scatter(CollectiveParams params) : Collective(params, "scatter") {
  assertBufferCount(params_.dst_bufs, 1);
}

c10::intrusive_ptr<c10d::Work> Scatter::post(Communicator& comm) {
  post_common(comm);
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, params_.team.size());
    buf_list_ = setBufList(params_.src_bufs);
  } else {
    assertBufferCount(params_.src_bufs, 0);
  }
  return comm.getTeam(params_.team)
          ->scatter(params_.dst_bufs, buf_list_, {.rootRank = root_rank_});
}

SendRecv::SendRecv(CollectiveParams params) : Collective(params, "send/recv") {
  TORCH_INTERNAL_ASSERT(
      params_.team.size() == 2, "the team size should be 2");
}

c10::intrusive_ptr<c10d::Work> SendRecv::post(Communicator& comm) {
  post_common(comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    assertBufferCount(params_.dst_bufs, 0);
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  return comm.sendRecv(
      (params_.team.at(0) == params_.root) ? params_.team.at(1)
                                          : params_.team.at(0),
      params_.root,
      params_.dst_bufs.empty() ? params_.src_bufs : params_.dst_bufs);
}

} // namespace nvfuser

#endif
