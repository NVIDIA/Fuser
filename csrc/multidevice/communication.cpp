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

void assertBufferCount(const std::vector<at::Tensor>& bufs, size_t count) {
  NVF_ERROR(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

void assertBuffersHaveSameSize(
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

void doLocalCopy(const at::Tensor& dst, const at::Tensor& src) {
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
      type == CommunicationType::Scatter || type == CommunicationType::Reduce ||
      type == CommunicationType::Broadcast ||
      type == CommunicationType::SendRecv;
}

bool isReduction(CommunicationType type) {
  return type == CommunicationType::Reduce ||
      type == CommunicationType::Allreduce ||
      type == CommunicationType::ReduceScatter;
}

} // namespace

Communication::Communication(
    IrBuilderPasskey passkey,
    CommunicationType type,
    DeviceMesh mesh,
    Team team,
    DeviceIdxType root,
    RedOpType red_op,
    int64_t scattered_axis,
    TensorView* input_tv,
    TensorView* output_tv)
    : Expr(passkey) {
  NVF_ERROR(mesh.size() > 0, "The mesh size must be greater than 0.");
  NVF_ERROR(
      hasRoot(type) == (root >= 0),
      "Root ",
      root,
      " is not expected by CommunicationType ",
      type);
  NVF_ERROR(isReduction(type) == (red_op != RedOpType::UNUSED))
  NVF_ERROR(
      (type == CommunicationType::ReduceScatter) == (scattered_axis >= 0));

  if (input_tv != nullptr) {
    addInput(input_tv);
  }
  if (output_tv != nullptr) {
    addOutput(output_tv);
  }
  addDataAttribute(type);
  addDataAttribute(mesh);
  addDataAttribute(team);
  addDataAttribute(root);
  addDataAttribute(red_op);
  addDataAttribute(scattered_axis);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Communication)

int64_t Communication::getRootRelativeIndex() {
  auto i = std::find(team().begin(), team().end(), root());
  NVF_ERROR(
      i != team().end(), "Unable to find root ", root(), " in team ", team());
  return std::distance(team().begin(), i);
}

std::string Communication::toString(const int indent_size) const {
  std::stringstream ss;

  indent(ss, indent_size) << "Communication " << type() << ": {" << std::endl;
  if (hasRoot(type())) {
    indent(ss, indent_size + 1) << "root: " << root() << "," << std::endl;
  }
  indent(ss, indent_size + 1) << "mesh: " << mesh() << "," << std::endl;
  indent(ss, indent_size + 1) << "team: " << team() << "," << std::endl;
  indent(ss, indent_size) << "}";

  return ss.str();
}

std::string Communication::toInlineString(int indent_size) const {
  return toString(indent_size);
}

namespace {
c10::intrusive_ptr<c10d::Work> postBroadcast(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  if (my_device_index == communication->root()) {
    if (communication->isRootInMesh()) {
      // Do a local copy and the subsequent broadcast will be in place. Consider
      // ProcessGroupNCCL::_broadcast_oop so ncclBroadcast doesn't wait for the
      // local copy to complete.
      doLocalCopy(output_tensor, input_tensor);
    } else {
      // `output_tensor` isn't allocated for this device.
      output_tensor = input_tensor;
    }
  }

  if (communication->team().size() == 1) {
    return nullptr;
  }

  std::vector<at::Tensor> tensors({output_tensor});
  return backend->broadcast(
      tensors, {.rootRank = communication->getRootRelativeIndex()});
}

c10::intrusive_ptr<c10d::Work> postGather(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  if (my_device_index == communication->root() &&
      !communication->isRootInMesh()) {
    // This is likely a suboptimal way to allocate tensors for nccl. To benefit
    // from zero copy
    // (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html),
    // tensors for nccl should be `ncclMemAlloc`ed and be `ncclCommRegister`ed.
    // https://github.com/pytorch/pytorch/issues/124807 is one proposal trying
    // to partially address this problem.
    input_tensor = at::empty_like(output_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> input_tensors({input_tensor});

  auto root_relative_index = communication->getRootRelativeIndex();
  std::vector<std::vector<at::Tensor>> output_tensors;
  if (my_device_index == communication->root()) {
    output_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(communication->team().size())) {
      if (root_relative_index == static_cast<DeviceIdxType>(i) &&
          !communication->isRootInMesh()) {
        output_tensors[0].push_back(input_tensor);
        continue;
      }
      output_tensors[0].push_back(output_tensor.slice(0, j, j + 1));
      j++;
    }

    assertBufferCount(output_tensors[0], communication->team().size());
    assertBuffersHaveSameSize(input_tensors, output_tensors[0]);
  }

  return backend->gather(
      output_tensors, input_tensors, {.rootRank = root_relative_index});
}

c10::intrusive_ptr<c10d::Work> postAllgather(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  auto splits = at::split(output_tensor, /*split_size=*/1, /*dim=*/0);
  assertBufferCount(splits, communication->team().size());
  assertBuffersHaveSameSize({input_tensor}, splits);

  // allgather primitive in c10d induces extra buffering time to copy out the
  // received tensors into user buffer. It is therefore always preferable to use
  // _allgather_base, which does not perform any extra copy at the cost of
  // assuming that the receive buffers are placed contiguously. See #2384 for an
  // illustration.
  return backend->_allgather_base(output_tensor, input_tensor, {});
}

c10::intrusive_ptr<c10d::Work> postScatter(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  if (my_device_index == communication->root() &&
      !communication->isRootInMesh()) {
    output_tensor = at::empty_like(input_tensor.slice(0, 0, 1));
  }
  std::vector<at::Tensor> output_tensors({output_tensor});

  auto root_relative_index = communication->getRootRelativeIndex();
  std::vector<std::vector<at::Tensor>> input_tensors;
  if (my_device_index == communication->root()) {
    input_tensors.resize(1);
    int64_t j = 0;
    for (auto i : c10::irange(communication->team().size())) {
      if (root_relative_index == static_cast<DeviceIdxType>(i) &&
          !communication->isRootInMesh()) {
        input_tensors.front().push_back(output_tensor);
        continue;
      }
      input_tensors.front().push_back(input_tensor.slice(0, j, j + 1));
      j++;
    }

    assertBufferCount(input_tensors[0], communication->team().size());
    assertBuffersHaveSameSize(input_tensors[0], output_tensors);
  }

  return backend->scatter(
      output_tensors, input_tensors, {.rootRank = root_relative_index});
}

c10::intrusive_ptr<c10d::Work> postReduce(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  at::Tensor tensor;
  if (my_device_index == communication->root()) {
    if (communication->isRootInMesh()) {
      doLocalCopy(output_tensor, input_tensor);
      tensor = output_tensor;
    } else {
      NVF_ERROR(
          output_tensor.scalar_type() == at::kFloat,
          "only float tensors are supported");
      output_tensor.fill_(getInitialValue<float>(communication->reduceOp()));
      tensor = output_tensor;
    }
  } else {
    tensor = input_tensor;
  }
  std::vector<at::Tensor> tensors({tensor});

  c10d::ReduceOptions options = {
      .reduceOp = communication->reduceOp(),
      .rootRank = communication->getRootRelativeIndex()};
  // TODO: avoid local copy by using out-of-place reduction.
  return backend->reduce(tensors, options);
}

c10::intrusive_ptr<c10d::Work> postAllreduce(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  doLocalCopy(output_tensor, input_tensor);
  std::vector<at::Tensor> output_tensors({output_tensor});

  return backend->allreduce(
      output_tensors, {.reduceOp = communication->reduceOp()});
}

c10::intrusive_ptr<c10d::Work> postReduceScatter(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  std::vector<std::vector<at::Tensor>> input_tensors(1);
  const auto scattered_axis = communication->scatteredAxis();
  NVF_ERROR(
      scattered_axis >= 0,
      "scattered_axis is expected to be non-negative: ",
      scattered_axis)
  input_tensors[0] = at::split(input_tensor, /*split_size=*/1, scattered_axis);

  std::vector<at::Tensor> output_tensors({output_tensor});

  assertBufferCount(input_tensors[0], communication->team().size());
  return backend->reduce_scatter(
      output_tensors, input_tensors, {.reduceOp = communication->reduceOp()});
}

c10::intrusive_ptr<c10d::Work> postSendRecv(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  NVF_ERROR(communication->mesh().size() == 1, "The mesh size should be 1.");

  if (communication->isRootInMesh()) {
    doLocalCopy(output_tensor, input_tensor);
    return nullptr;
  }

  const DeviceIdxType sender = communication->root();
  const DeviceIdxType receiver = communication->mesh().at(0);

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
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  const Team& team = communication->team();
  NVF_ERROR(
      std::find(team.begin(), team.end(), my_device_index) != team.end(),
      "current device index ",
      my_device_index,
      " must be present in the communication's team");

  switch (communication->type()) {
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
      NVF_ERROR(false, "Wrong communication type: ", communication->type());
      return nullptr;
  }
}

} // namespace nvfuser
