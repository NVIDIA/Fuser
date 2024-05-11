// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <multidevice/communication.h>
#if defined(NVFUSER_DISTRIBUTED) && defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#include <utils.h>

namespace nvfuser {

std::ostream& operator<<(std::ostream& os, const CommunicationType& type) {
  switch (type) {
    case CommunicationType::Gather:
      os << "Gather";
      break;
    case CommunicationType::Allgather:
      os << "Allgather";
      break;
    case CommunicationType::Scatter:
      os << "Scatter";
      break;
    case CommunicationType::Reduce:
      os << "Reduce";
      break;
    case CommunicationType::Allreduce:
      os << "Allreduce";
      break;
    case CommunicationType::ReduceScatter:
      os << "ReduceScatter";
      break;
    case CommunicationType::Broadcast:
      os << "Broadcast";
      break;
    case CommunicationType::SendRecv:
      os << "SendRecv";
      break;
  }
  return os;
}

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
  const auto numel = (bufs1.empty() ? bufs2 : bufs1).at(0).numel();
  for (const auto& bufs : {bufs1, bufs2}) {
    for (const auto& buf : bufs) {
      NVF_ERROR(
          buf.numel() == numel,
          "all buffers must have the same number of elements");
    }
  }
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

bool hasRoot(CommunicationType type) {
  return type == CommunicationType::Gather ||
      type == CommunicationType::Scatter ||
      type == CommunicationType::Broadcast ||
      type == CommunicationType::SendRecv;
}

bool isReduction(CommunicationType type) {
  return type == CommunicationType::Reduce ||
      type == CommunicationType::Allreduce ||
      type == CommunicationType::ReduceScatter;
}

void assertValid(const CommParams& params) {
  std::unordered_set<DeviceIdxType> team_without_duplicates(
      params.team.begin(), params.team.end());
  NVF_ERROR(
      team_without_duplicates.size() == params.team.size(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(!params.team.empty(), "the team size must be greater than 0");
  if (hasRoot(params.type)) {
    auto it = std::find(params.team.begin(), params.team.end(), params.root);
    NVF_ERROR(
        it != params.team.end(),
        "root (device ",
        params.root,
        ") must be present in the communication's team");
  }
}

// pytorch's process group expects the root to be specified
// as an integer between 0 and world_size-1. We choose it to be
// the device's relative index within the team
int64_t getRootRelativeIndex(const CommParams& params) {
  auto it = std::find(params.team.begin(), params.team.end(), params.root);
  return std::distance(params.team.begin(), it);
}

} // namespace

Communication::Communication(IrBuilderPasskey passkey, CommParams params)
    : Expr(passkey), params_(std::move(params)) {
  assertValid(params_);
}

Communication::Communication(const Communication* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), params_(src->params()) {}

NVFUSER_DEFINE_CLONE_AND_CREATE(Communication)

std::string Communication::toString(const int indent_size) const {
  std::stringstream ss;

  indent(ss, indent_size) << "Communication " << params_.type << ": {"
                          << std::endl;

  if (hasRoot(params_.type)) {
    indent(ss, indent_size + 1) << "root: " << params_.root << "," << std::endl;
  }
  indent(ss, indent_size + 1) << "team: " << params_.team << "," << std::endl;
  indent(ss, indent_size) << "}";

  return ss.str();
}

std::string Communication::toInlineString(int indent_size) const {
  return toString(indent_size);
}

// TODO add checking symbolic representation of src and dst buffers
bool Communication::sameAs(const Statement* other) const {
  if (other == this) {
    return true;
  }
  if (!other->isA<Communication>()) {
    return false;
  }
  const auto& p1 = this->params();
  const auto& p2 = other->as<Communication>()->params();

  return (
      p1.type == p2.type && (!hasRoot(p1.type) || p1.root == p2.root) &&
      p1.team == p2.team && (!isReduction(p1.type) || p1.redOp == p2.redOp));
}

namespace {
c10::intrusive_ptr<c10d::Work> postBroadcast(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();
  if (my_device_index == params.root) {
    if (params.is_root_in_mesh) {
      // Do a local copy and the subsequent broadcast will be in place. Consider
      // ProcessGroupNCCL::_broadcast_oop so ncclBroadcast doesn't wait for the
      // local copy to complete.
      doLocalCopy(output_tensor, input_tensor);
    } else {
      // `output_tensor` isn't allocated for this device.
      output_tensor = input_tensor;
    }
  }

  if (params.team.size() == 1) {
    return nullptr;
  }

  std::vector<at::Tensor> tensors({output_tensor});
  return backend->broadcast(
      tensors, {.rootRank = getRootRelativeIndex(params)});
}

c10::intrusive_ptr<c10d::Work> postGather(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();
  if (my_device_index == params.root && !params.is_root_in_mesh) {
    // This is likely a suboptimal way to allocate tensors for nccl. To benefit
    // from zero copy
    // (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html),
    // tensors for nccl should be `ncclMemAlloc`ed and be `ncclCommRegister`ed.
    // https://github.com/pytorch/pytorch/issues/124807 is one proposal trying
    // to partially address this problem.
    input_tensor = at::empty_like(output_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> input_tensors({input_tensor});

  auto root_relative_index = getRootRelativeIndex(params);
  std::vector<std::vector<at::Tensor>> output_tensors;
  if (my_device_index == params.root) {
    output_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(params.team.size())) {
      if (root_relative_index == static_cast<DeviceIdxType>(i) &&
          !params.is_root_in_mesh) {
        output_tensors[0].push_back(input_tensor);
        continue;
      }
      output_tensors[0].push_back(output_tensor.slice(0, j, j + 1));
      j++;
    }

    assertBufferCount(output_tensors[0], params.team.size());
    assertBuffersHaveSameSize(input_tensors, output_tensors[0]);
  }

  return backend->gather(
      output_tensors, input_tensors, {.rootRank = root_relative_index});
}

c10::intrusive_ptr<c10d::Work> postAllgather(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  std::vector<at::Tensor> input_tensors({input_tensor});
  std::vector<std::vector<at::Tensor>> output_tensors(1);
  output_tensors[0] = at::split(output_tensor, /*split_size=*/1, /*dim=*/0);

  assertBufferCount(output_tensors[0], params.team.size());
  assertBuffersHaveSameSize(input_tensors, output_tensors[0]);
  return backend->allgather(output_tensors, input_tensors, {});
}

c10::intrusive_ptr<c10d::Work> postScatter(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  if (my_device_index == params.root && !params.is_root_in_mesh) {
    output_tensor = at::empty_like(input_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> output_tensors({output_tensor});

  auto root_relative_index = getRootRelativeIndex(params);
  std::vector<std::vector<at::Tensor>> input_tensors;
  if (my_device_index == params.root) {
    input_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(params.team.size())) {
      if (root_relative_index == static_cast<DeviceIdxType>(i) &&
          !params.is_root_in_mesh) {
        input_tensors.front().push_back(output_tensor);
        continue;
      }
      input_tensors.front().push_back(input_tensor.slice(0, j, j + 1));
      j++;
    }

    assertBufferCount(input_tensors[0], params.team.size());
    assertBuffersHaveSameSize(input_tensors[0], output_tensors);
  }

  return backend->scatter(
      output_tensors, input_tensors, {.rootRank = root_relative_index});
}

c10::intrusive_ptr<c10d::Work> postReduce(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  at::Tensor tensor;
  if (my_device_index == params.root) {
    if (params.is_root_in_mesh) {
      doLocalCopy(output_tensor, input_tensor);
      tensor = output_tensor;
    } else {
      NVF_ERROR(
          output_tensor.scalar_type() == at::kFloat,
          "only float tensors are supported");
      output_tensor.fill_(getInitialValue<float>(params.redOp));
      tensor = output_tensor;
    }
  } else {
    tensor = input_tensor;
  }
  std::vector<at::Tensor> tensors({tensor});

  c10d::ReduceOptions options = {
      .reduceOp = params.redOp, .rootRank = getRootRelativeIndex(params)};
  // TODO: avoid local copy by using out-of-place reduction.
  return backend->reduce(tensors, options);
}

c10::intrusive_ptr<c10d::Work> postAllreduce(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  doLocalCopy(output_tensor, input_tensor);
  std::vector<at::Tensor> output_tensors({output_tensor});

  return backend->allreduce(output_tensors, {.reduceOp = params.redOp});
}

c10::intrusive_ptr<c10d::Work> postReduceScatter(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  std::vector<std::vector<at::Tensor>> input_tensors(1);
  NVF_ERROR(
      params.scattered_axis >= 0,
      "scattered_axis is expected to be non-negative: ",
      params.scattered_axis)
  input_tensors[0] =
      at::split(input_tensor, /*split_size=*/1, params.scattered_axis);

  std::vector<at::Tensor> output_tensors({output_tensor});

  assertBufferCount(input_tensors[0], params.team.size());
  return backend->reduce_scatter(
      output_tensors, input_tensors, {.reduceOp = params.redOp});
}

c10::intrusive_ptr<c10d::Work> postSendRecv(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();

  NVF_ERROR(
      params.team.size() == 1 || params.team.size() == 2,
      "the team size should be 1 or 2");

  if (params.team.size() == 1) {
    doLocalCopy(output_tensor, input_tensor);
    return nullptr;
  }

  const DeviceIdxType sender = params.root;
  const DeviceIdxType receiver =
      params.team.at(0) == sender ? params.team.at(1) : params.team.at(0);

  std::vector<at::Tensor> tensors;
  if (my_device_index == sender) {
    tensors = {input_tensor};
    return backend->send(tensors, static_cast<int>(receiver), /*tag=*/0);
  } else {
    NVF_ERROR(my_device_index == receiver);
    tensors = {output_tensor};
    return backend->recv(tensors, static_cast<int>(sender), /*tag=*/0);
  }
}
} // namespace

c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const CommParams& params = communication->params();
  NVF_ERROR(
      std::find(params.team.begin(), params.team.end(), my_device_index) !=
          params.team.end(),
      "current device index ",
      my_device_index,
      " must be present in the communication's team");

  switch (communication->params().type) {
    case CommunicationType::Gather:
      return postGather(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::Allgather:
      return postAllgather(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::Scatter:
      return postScatter(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::Reduce:
      return postReduce(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::Allreduce:
      return postAllreduce(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::ReduceScatter:
      return postReduceScatter(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::Broadcast:
      return postBroadcast(
          communication, my_device_index, backend, input_tensor, output_tensor);
    case CommunicationType::SendRecv:
      return postSendRecv(
          communication, my_device_index, backend, input_tensor, output_tensor);
    default:
      NVF_ERROR(false, "Wrong communication type: ", params.type);
      return nullptr;
  }
}

} // namespace nvfuser
