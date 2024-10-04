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
    default:
      NVF_THROW("unrecognized CommunicationType: ", type);
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
      NVF_THROW("unsupported reduction op type");
      return 0;
  }
}

bool hasRoot(CommunicationType type) {
  switch (type) {
    case CommunicationType::Gather:
    case CommunicationType::Scatter:
    case CommunicationType::Reduce:
    case CommunicationType::Broadcast:
    case CommunicationType::SendRecv:
      return true;
    case CommunicationType::Allgather:
    case CommunicationType::Allreduce:
    case CommunicationType::ReduceScatter:
      return false;
    default:
      NVF_THROW("unrecognized CommunicationType: ", type);
  }
}

bool isReduction(CommunicationType type) {
  switch (type) {
    case CommunicationType::Reduce:
    case CommunicationType::Allreduce:
    case CommunicationType::ReduceScatter:
      return true;
    case CommunicationType::Gather:
    case CommunicationType::Allgather:
    case CommunicationType::Scatter:
    case CommunicationType::Broadcast:
    case CommunicationType::SendRecv:
      return false;
    default:
      NVF_THROW("unrecognized CommunicationType: ", type);
  }
}

} // namespace

Communication::Communication(
    IrBuilderPasskey passkey,
    CommunicationType type,
    TensorView* out,
    TensorView* in,
    Team team,
    DeviceIdxType root,
    RedOpType red_op,
    int64_t scattered_axis)
    : Expr(passkey) {
  NVF_ERROR(
      in->getDeviceMesh().size() > 0,
      "The input mesh size must be greater than 0.");
  NVF_ERROR(
      out->getDeviceMesh().size() > 0,
      "The output mesh size must be greater than 0.");

  addInput(in);
  addOutput(out);
  addDataAttribute(type);
  addDataAttribute(team);
  addDataAttribute(root);
  addDataAttribute(red_op);
  addDataAttribute(scattered_axis);

  validate();
}

void Communication::validate() {
  NVF_ERROR(
      hasRoot(type()) == (root() >= 0),
      "Root ",
      root(),
      " is not expected by CommunicationType ",
      type());
  NVF_ERROR(isReduction(type()) == (reduceOp() != RedOpType::UNUSED))
  NVF_ERROR(
      (type() == CommunicationType::ReduceScatter) == (scatteredAxis() >= 0));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Communication)

namespace {
int64_t getRelativeIndex(const Team& team, const DeviceIdxType rank) {
  auto i = std::find(team.begin(), team.end(), rank);
  NVF_ERROR(i != team.end(), "Unable to find rank ", rank, " in team ", team);
  return std::distance(team.begin(), i);
}
} // namespace

int64_t Communication::getRootRelativeIndex() {
  return getRelativeIndex(team(), root());
}

std::string Communication::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Communication " << name() << " ("
                          << "type=" << type() << ", "
                          << "team=(" << team() << ")";
  if (hasRoot(type())) {
    ss << ", root=" << root();
  }
  if (!inputs().empty()) {
    ss << ", input=" << in();
  }
  if (!outputs().empty()) {
    ss << ", output=" << out();
  }
  ss << ")\n";
  return ss.str();
}

std::string Communication::toInlineString(int indent_size) const {
  return toString(indent_size);
}

std::ostream& operator<<(std::ostream& os, const P2PCommunicationType& type) {
  switch (type) {
    case P2PCommunicationType::SEND:
      os << "send";
      break;
    case P2PCommunicationType::RECV:
      os << "recv";
      break;
    default:
      NVF_THROW("unrecognized P2PCommunicationType: ", type);
  }
  return os;
}

P2PCommunication::P2PCommunication(
    IrBuilderPasskey passkey,
    P2PCommunicationType type,
    TensorView* buffer,
    Val* peer)
    : Expr(passkey) {
  addInput(buffer);
  addDataAttribute(type);
  addAttribute(peer);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(P2PCommunication)

std::string P2PCommunication::toString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "P2PCommunication " << name() << " ("
                          << "type=" << type() << ", "
                          << "buffer=" << buffer() << ", "
                          << "peer=" << peer() << ")\n";
  return ss.str();
}

std::string P2PCommunication::toInlineString(int indent_size) const {
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
    if (communication->out()->getDeviceMesh().has(communication->root())) {
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
      !communication->in()->getDeviceMesh().has(communication->root())) {
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
          !communication->in()->getDeviceMesh().has(communication->root())) {
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
      !communication->out()->getDeviceMesh().has(communication->root())) {
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
          !communication->out()->getDeviceMesh().has(communication->root())) {
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
    if (communication->in()->getDeviceMesh().has(communication->root())) {
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
  const auto scattered_axis = communication->scatteredAxis();
  NVF_ERROR(
      scattered_axis >= 0,
      "scattered_axis is expected to be non-negative: ",
      scattered_axis);
// reduce_scatter primitive in c10d induces extra buffering time to copy the
// user's input tensors to an internal source buffer. It is therefore always
// preferable to use _reduce_scatter_base (which does not perform any extra
// copy) when the tensors are stored contiguously (i.e., when
// scattered_axis==0). Note however than only nccl supports
// _reduce_scatter_base, not ucc.
#if defined(NVFUSER_DISTRIBUTED) && defined(USE_C10D_NCCL)
  if (scattered_axis == 0 &&
      backend->getBackendName() == c10d::NCCL_BACKEND_NAME) {
    return backend->_reduce_scatter_base(
        output_tensor, input_tensor, {.reduceOp = communication->reduceOp()});
  }
#endif
  std::vector<std::vector<at::Tensor>> input_tensors(1);
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
  const Team& team = communication->team();
  const DeviceIdxType sender = communication->root();
  DeviceIdxType receiver = -1;
  if (team.size() == 1) {
    receiver = sender;
  } else {
    NVF_ERROR(
        team.size() == 2,
        "SendRecv's team size is expected to be 1 or 2, however found ",
        team.size());
    receiver = (team[0] == sender ? team[1] : team[0]);
  }

  if (sender == receiver) {
    doLocalCopy(output_tensor, input_tensor);
    return nullptr;
  }

  std::vector<at::Tensor> tensors;
  if (my_device_index == sender) {
    tensors = {input_tensor};
    return backend->send(
        tensors,
        static_cast<int>(getRelativeIndex(communication->team(), receiver)),
        /*tag=*/0);
  } else {
    NVF_ERROR(my_device_index == receiver);
    tensors = {output_tensor};
    return backend->recv(
        tensors,
        static_cast<int>(getRelativeIndex(communication->team(), sender)),
        /*tag=*/0);
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
  if (std::find(team.begin(), team.end(), my_device_index) == team.end()) {
    return nullptr;
  }
  NVF_ERROR(backend != nullptr);

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
      NVF_THROW("Wrong communication type: ", communication->type());
      return nullptr;
  }
}

namespace {

c10::intrusive_ptr<c10d::Work> postSend(
    P2PCommunication* communication,
    DeviceIdxType my_device_index,
    DeviceIdxType peer,
    c10d::Backend* backend,
    at::Tensor buffer) {
  NVF_ERROR(peer < backend->getSize(), "invalid peer: ", peer);

  // Needed to match ProcessGroup API
  std::vector<at::Tensor> packed_buffer = {buffer};
  return backend->send(packed_buffer, static_cast<int>(peer), /*tag=*/0);
}

c10::intrusive_ptr<c10d::Work> postRecv(
    P2PCommunication* communication,
    DeviceIdxType my_device_index,
    DeviceIdxType peer,
    c10d::Backend* backend,
    at::Tensor buffer) {
  NVF_ERROR(
      peer < backend->getSize(),
      "invalid peer: ",
      peer,
      ", which should be strictly smaller than the world size ",
      backend->getSize());

  // Needed to match ProcessGroup API
  std::vector<at::Tensor> packed_buffer = {buffer};
  return backend->recv(packed_buffer, static_cast<int>(peer), /*tag=*/0);
}

} // namespace

c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    P2PCommunication* communication,
    DeviceIdxType my_device_index,
    DeviceIdxType peer,
    c10d::Backend* backend,
    at::Tensor buffer) {
  NVF_ERROR(backend != nullptr);

  switch (communication->type()) {
    case P2PCommunicationType::SEND:
      return postSend(communication, my_device_index, peer, backend, buffer);
    case P2PCommunicationType::RECV:
      return postRecv(communication, my_device_index, peer, backend, buffer);
    default:
      NVF_THROW("Wrong communication type: ", communication->type());
      return nullptr;
  }
}

} // namespace nvfuser
