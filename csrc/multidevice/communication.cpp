// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/communication.h>
#if defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#include <utils.h>

namespace nvfuser {
namespace {

// FIXME: rename this.
void post_common(Communication& self, Communicator& comm) {
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
  dst.view_as(src).copy_(src, /*non_blocking=*/true);
}

template <typename T>
T getInitialValue(c10d::ReduceOp::RedOpType op) {
  // TODO: add other ops
  switch (op) {
    case c10d::ReduceOp::RedOpType::SUM:
      return 0;
    case c10d::ReduceOp::RedOpType::PRODUCT:
      return 1;
    case c10d::ReduceOp::RedOpType::MAX:
      return std::numeric_limits<T>::min();
    case c10d::ReduceOp::RedOpType::MIN:
      return std::numeric_limits<T>::max();
    default:
      NVF_ERROR(false, "unsupported reduction op type");
      return 0;
  }
}

} // namespace

Communication::Communication(CommParams params, std::string name, bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {
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
  ss << ext_indent << "}";

  return ss.str();
}

Broadcast::Broadcast(CommParams params) : Communication(params, "broadcast") {}

c10::intrusive_ptr<c10d::Work> Broadcast::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root) {
    doLocalCopy(output_tensor, input_tensor);
  }

  if (params_.team.size() == 1) {
    return nullptr;
  }

  std::vector<at::Tensor> tensors(
      {comm.deviceId() == params_.root ? input_tensor : output_tensor});
  return comm.getBackendForTeam(params_.team, backend)
      ->broadcast(tensors, {.rootRank = root_relative_index_});
}

Gather::Gather(CommParams params) : Communication(params, "gather") {}

c10::intrusive_ptr<c10d::Work> Gather::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root && !params_.is_root_in_mesh) {
    input_tensor = at::empty_like(output_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> input_tensors({input_tensor});

  std::vector<std::vector<at::Tensor>> output_tensors;
  if (comm.deviceId() == params_.root) {
    output_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(params_.team.size())) {
      if (root_relative_index_ == static_cast<DeviceIdxType>(i) &&
          !params_.is_root_in_mesh) {
        output_tensors.front().push_back(input_tensor);
        continue;
      }
      output_tensors.front().push_back(output_tensor.slice(0, j, j + 1));
      j++;
    }
  }

  return comm.getBackendForTeam(params_.team, backend)
      ->gather(
          output_tensors, input_tensors, {.rootRank = root_relative_index_});
}

Allgather::Allgather(CommParams params)
    : Communication(params, "allgather", false) {}

c10::intrusive_ptr<c10d::Work> Allgather::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  std::vector<at::Tensor> input_tensors({input_tensor});

  // FIXME: use split
  std::vector<std::vector<at::Tensor>> output_tensors(1);
  for (auto i : c10::irange(params_.team.size())) {
    output_tensors[0].push_back(output_tensor.slice(0, i, i + 1));
  }
  return comm.getBackendForTeam(params_.team, backend)
      ->allgather(output_tensors, input_tensors, {});
}

Scatter::Scatter(CommParams params) : Communication(params, "scatter") {}

c10::intrusive_ptr<c10d::Work> Scatter::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root && !params_.is_root_in_mesh) {
    output_tensor = at::empty_like(input_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> output_tensors({output_tensor});

  std::vector<std::vector<at::Tensor>> input_tensors;
  if (comm.deviceId() == params_.root) {
    input_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(params_.team.size())) {
      if (root_relative_index_ == static_cast<DeviceIdxType>(i) &&
          !params_.is_root_in_mesh) {
        input_tensors.front().push_back(output_tensor);
        continue;
      }
      input_tensors.front().push_back(input_tensor.slice(0, j, j + 1));
      j++;
    }
  }

  return comm.getBackendForTeam(params_.team, backend)
      ->scatter(
          output_tensors, input_tensors, {.rootRank = root_relative_index_});
}

Reduce::Reduce(CommParams params) : Communication(params, "reduce") {}

c10::intrusive_ptr<c10d::Work> Reduce::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  at::Tensor tensor;
  if (comm.deviceId() == params_.root) {
    if (params_.is_root_in_mesh) {
      doLocalCopy(output_tensor, input_tensor);
      tensor = output_tensor;
    } else {
      NVF_ERROR(
          output_tensor.scalar_type() == at::kFloat,
          "only float tensors are supported");
      output_tensor.fill_(getInitialValue<float>(params_.redOp));
      tensor = output_tensor;
    }
  } else {
    tensor = input_tensor;
  }
  std::vector<at::Tensor> tensors({tensor});

  c10d::ReduceOptions options = {
      .reduceOp = params_.redOp, .rootRank = root_relative_index_};
  // TODO: avoid local copy by using out-of-place reduction.
  return comm.getBackendForTeam(params_.team, backend)
      ->reduce(tensors, options);
}

Allreduce::Allreduce(CommParams params)
    : Communication(params, "allreduce", false) {}

c10::intrusive_ptr<c10d::Work> Allreduce::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  doLocalCopy(output_tensor, input_tensor);
  std::vector<at::Tensor> output_tensors({output_tensor});

  return comm.getBackendForTeam(params_.team, backend)
      ->allreduce(output_tensors, {.reduceOp = params_.redOp});
}

ReduceScatter::ReduceScatter(CommParams params)
    : Communication(params, "reduce_scatter", false) {}

c10::intrusive_ptr<c10d::Work> ReduceScatter::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  std::vector<std::vector<at::Tensor>> input_tensors(1);
  // FIXME: use split
  NVF_ERROR(
      params_.scattered_axis >= 0,
      "scattered_axis is expected to be non-negative: ",
      params_.scattered_axis)
  const auto scattered_dim_size = input_tensor.size(params_.scattered_axis);
  const auto team_size = static_cast<int64_t>(params_.team.size());
  NVF_ERROR(
      scattered_dim_size == team_size, scattered_dim_size, " vs ", team_size);
  for (auto i : c10::irange(team_size)) {
    input_tensors[0].push_back(
        input_tensor.slice(params_.scattered_axis, i, i + 1));
  }

  std::vector<at::Tensor> output_tensors({output_tensor});
  return comm.getBackendForTeam(params_.team, backend)
      ->reduce_scatter(
          output_tensors, input_tensors, {.reduceOp = params_.redOp});
}

SendRecv::SendRecv(CommParams params) : Communication(params, "send/recv") {
  NVF_ERROR(
      params_.team.size() == 1 || params_.team.size() == 2,
      "the team size should be 1 or 2");
}

c10::intrusive_ptr<c10d::Work> SendRecv::post(
    Communicator& comm,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (params_.team.size() == 1) {
    doLocalCopy(output_tensor, input_tensor);
    return nullptr;
  }

  std::vector<at::Tensor> tensors(
      {comm.deviceId() == params_.root ? input_tensor : output_tensor});
  return comm.sendRecv(
      /*receiver=*/(params_.team.at(0) == params_.root) ? params_.team.at(1)
                                                        : params_.team.at(0),
      /*sender=*/params_.root,
      tensors,
      backend);
}

} // namespace nvfuser
