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
#include <multidevice/utils.h>
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
  const auto shape = (bufs1.empty() ? bufs2 : bufs1).at(0).sizes();
  for (const auto& bufs : {bufs1, bufs2}) {
    for (const auto& buf : bufs) {
      NVF_ERROR(
          buf.sizes() == shape,
          "all buffers must have the same shape, but got: ",
          buf.sizes(),
          " vs ",
          shape);
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
    CommunicatorBackend backend)
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
  addDataAttribute(backend);

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

std::string Communication::toInlineString(const int indent_size) const {
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
  ss << ", backend=" << backend();
  ss << ")";
  return ss.str();
}

std::string Communication::toString(int indent_size) const {
  return toInlineString(indent_size) + "\n";
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
    Val* peer,
    CommunicatorBackend backend)
    : Expr(passkey) {
  addInput(buffer);
  addDataAttribute(type);
  addAttribute(peer);
  addDataAttribute(backend);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(P2PCommunication)

std::string P2PCommunication::toInlineString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "P2PCommunication " << name() << " ("
                          << "type=" << type() << ", "
                          << "buffer=" << buffer() << ", "
                          << "peer=" << peer() << ", "
                          << "backend=" << backend() << ")";
  return ss.str();
}

std::string P2PCommunication::toString(int indent_size) const {
  return toInlineString(indent_size) + "\n";
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
    for (auto i : arange(communication->team().size())) {
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
  // input and output tensors maybe strided (tensor with shape [m, n, k] and
  // strides [1, k*m, m]), so we flatten them to match the ProcessGroupNCCL
  // contiguity requirements. Presegmentation pass `makeReshardingContiguous`
  // ensures that the tvs are contiguous. CommunicationExecutor and
  // HostIrEvaluator validate the tensor against the tv allocation domain.

  NVF_ERROR(
      isTvContiguous(communication->in()), "Input tensor is not contiguous");
  NVF_ERROR(
      isTvContiguous(communication->out()), "Output tensor is not contiguous");

  auto flattened_output_tensor =
      output_tensor.as_strided({output_tensor.numel()}, {1});
  auto flattened_input_tensor =
      input_tensor.as_strided({input_tensor.numel()}, {1});
  auto splits = at::tensor_split(
      flattened_output_tensor, communication->team_size(), /*dim=*/0);
  assertBuffersHaveSameSize({flattened_input_tensor}, splits);

  // allgather primitive in c10d induces extra buffering time to copy out the
  // received tensors into user buffer. It is therefore always preferable to use
  // _allgather_base, which does not perform any extra copy at the cost of
  // assuming that the receive buffers are placed contiguously. See #2384 for an
  // illustration.
  return backend->_allgather_base(
      flattened_output_tensor, flattened_input_tensor, {});
}

c10::intrusive_ptr<c10d::Work> postScatter(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  NVF_ERROR(
      isTvContiguous(communication->in()), "Input tensor is not contiguous");
  NVF_ERROR(
      isTvContiguous(communication->out()), "Output tensor is not contiguous");

  auto output_device_mesh = communication->out()->getDeviceMesh();
  NVF_ERROR(
      output_device_mesh.has(communication->root()),
      "communication->root() ",
      communication->root(),
      " is not in the output device mesh ",
      output_device_mesh,
      ".");

  std::vector<std::vector<at::Tensor>> input_tensors;

  output_tensor = output_tensor.as_strided({output_tensor.numel()}, {1});
  std::vector<at::Tensor> output_tensors({output_tensor});

  if (my_device_index == communication->root()) {
    auto splits = at::tensor_split(
        input_tensor.as_strided({input_tensor.numel()}, {1}),
        output_device_mesh.size(),
        /*dim=*/0);

    input_tensors.resize(1);
    for (const auto& split : splits) {
      input_tensors.front().push_back(split);
    }

    assertBufferCount(input_tensors[0], output_device_mesh.size());
    assertBuffersHaveSameSize(input_tensors[0], output_tensors);
  }

  return backend->scatter(
      output_tensors,
      input_tensors,
      {.rootRank = communication->getRootRelativeIndex()});
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
  NVF_ERROR(
      isTvContiguous(communication->in()),
      "Input tensor is not contiguous: ",
      communication->in(),
      " contiguity: ",
      communication->in()->domain()->getContiguityString());
  NVF_ERROR(
      isTvContiguous(communication->out()),
      "Output tensor is not contiguous: ",
      communication->out(),
      " contiguity: ",
      communication->out()->domain()->getContiguityString());

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
  NVF_ERROR(
      isTvContiguous(communication->in()),
      "Input tensor is not contiguous: ",
      communication->in(),
      " contiguity: ",
      communication->in()->domain()->getContiguityString());
  NVF_ERROR(
      isTvContiguous(communication->out()),
      "Output tensor is not contiguous: ",
      communication->out(),
      " contiguity: ",
      communication->out()->domain()->getContiguityString());

  auto flattened_input_tensor =
      input_tensor.as_strided({input_tensor.numel()}, {1});
  auto splits = at::tensor_split(
      flattened_input_tensor, communication->team_size(), /*dim=*/0);
  auto flattened_output_tensor =
      output_tensor.as_strided({output_tensor.numel()}, {1});
  assertBuffersHaveSameSize(splits, {flattened_output_tensor});

  // reduce_scatter primitive in c10d induces extra buffering time to copy the
  // user's input tensors to an internal source buffer. It is therefore always
  // preferable to use _reduce_scatter_base (which does not perform any extra
  // copy) when the tensors are stored contiguously
  return backend->_reduce_scatter_base(
      flattened_output_tensor,
      flattened_input_tensor,
      {.reduceOp = communication->reduceOp()});
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

  if (isDebugDumpEnabled(DebugDumpOption::Communication)) {
    debug() << "Posting " << communication->toInlineString()
            << " with input_tensor " << input_tensor.sizes()
            << " and output_tensor " << output_tensor.sizes() << std::endl;
  }

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
