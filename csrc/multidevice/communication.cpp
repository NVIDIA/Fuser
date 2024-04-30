// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/printer.h>
#include <ir/cloner.h>
#include <multidevice/communication.h>
#if defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#include <utils.h>

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

inline void doLocalCopy(const at::Tensor& dst, const at::Tensor& src) {
  dst.copy_(src, /* non-blocking */ true);
}

inline bool hasRoot(CommunicationType type) {
  return type == CommunicationType::Gather || type == CommunicationType::Scatter || type == CommunicationType::Broadcast || type == CommunicationType::SendRecv;
}

inline bool typeToString(CommunicationType type) {
  switch(type) {
  case CommunicationType::Gather:
    return "Gather";
  case CommunicationType::Allgather:
    return "Allgather";
  case CommunicationType::Scatter:
    return "Scatter";
  case CommunicationType::Reduce:
    return "Reduce";
  case CommunicationType::Allreduce:
    return "Allreduce";
  case CommunicationType::ReduceScatter:
    return "ReduceScatter";
  case CommunicationType::Broadcast:
    return "Broadcast";
  case CommunicationType::SendRecv:
    return "SendRecv";
  default:
    NVF_ERROR(false);
    return "";
  }
}

inline void assertValid(const CommParams& params,
          const DeviceIdxType my_device_index) {
  assertBuffersHaveSameSize(params.src_bufs, params.dst_bufs);
  NVF_ERROR(
      std::adjacent_find(params.team.cbegin(), params.team.cend()) ==
          params.team.cend(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(!params.team.empty(), "the team size must be greater than 0");
  NVF_ERROR(
    std::find(
        params.team.begin(),
        params.team.end(),
        my_device_index) != params.team.end(),
    "current device index ",
    my_device_index,
    " must be present in the communication's team");
  if (hasRoot(params.type)) {
    auto it = std::find(params.team.begin(), params.team.end(), params.root);
    NVF_ERROR(
        it != params.team.end(),
        "root (device ",
        params.root,
        ") must be present in the communication's team");
  }
  bool is_root = (my_device_index == params.root);
  switch(params.type) {
  case CommunicationType::Gather:
    assertBufferCount(params.src_bufs, 1);
    assertBufferCount(params.dst_bufs, is_root? params.team.size() : 0);
    break;
  case CommunicationType::Allgather:
    assertBufferCount(params.src_bufs, 1);
    assertBufferCount(params.dst_bufs, params.team.size());
    break;
  case CommunicationType::Scatter:
    assertBufferCount(params.dst_bufs, 1);
    assertBufferCount(params.src_bufs, is_root? params.team.size():0);
    break;
  case CommunicationType::Reduce:
    assertBufferCount(params.src_bufs, 1);
    assertBufferCount(params.dst_bufs, is_root? 1:0);
    break;
  case CommunicationType::Allreduce:
    assertBufferCount(params.dst_bufs, 1);
    assertBufferCount(params.src_bufs, 1);
    break;
  case CommunicationType::ReduceScatter:
    assertBufferCount(params.dst_bufs, 1);
    assertBufferCount(params.src_bufs, params.team.size());
    break;
  case CommunicationType::Broadcast:
    if (is_root) {
      assertBufferCount(params.src_bufs, 1);
      NVF_ERROR(
        params.dst_bufs.size() < 2,
        "there must be at most 2 buffer(s)");
    } else {
      assertBufferCount(params.src_bufs, 0);
      assertBufferCount(params.dst_bufs, 1);
    }
    break;
  case CommunicationType::SendRecv:
    NVF_ERROR(
      params.team.size() == 1 || params.team.size() == 2,
      "the team size should be 1 or 2");
    if (is_root) {
      assertBufferCount(params.src_bufs, 1);
      assertBufferCount(params.dst_bufs, (params.team.size() == 1)? 1:0);
    } else {
      assertBufferCount(params.src_bufs, 0);
      assertBufferCount(params.dst_bufs, 1);
    }
    break;
  }
}

// pytorch's process group expects the root to be specified
// as an integer between 0 and world_size-1. We choose it to be
// the device's relative index within the team
DeviceIdxType getRootRelativeIndex(const CommParams& params) {
    auto it = std::find(params.team.begin(), params.team.end(), params.root);
    return std::distance(params.team.begin(), it);
}

} // namespace

Communication::Communication(IrBuilderPasskey passkey, CommParams params)
    : Expr(passkey), params_(std::move(params)) {}

Communication::Communication(const Communication* src, IrCloner* ir_cloner) : Expr(src, ir_cloner), params_(src->params()) {}

NVFUSER_DEFINE_CLONE_AND_CREATE(Communication)

std::string Communication::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Communication " << typeToString(params_.type) << ": {\n";

  if (hasRoot(params_.type)) {
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

std::string Communication::toInlineString(int indent_size) const {
    return toString(indent_size);
}

// TODO: implement
bool Communication::sameAs(const Statement* other) const {
    return false;
}

c10::intrusive_ptr<c10d::Work> postCommunication(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend) {

  auto params = communication->params();
  assertValid(params, my_device_index);
  bool is_root = (my_device_index == params.root);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  switch(params.type) {
  case CommunicationType::Gather:
{
  if (is_root) {
      buf_list = {params.dst_bufs};
    }
    auto work =
        backend->gather(
                buf_list, params.src_bufs, {.rootRank = getRootRelativeIndex(params)});
    return work;
}
  case CommunicationType::Allgather:
{
    // This is used to change the representation of the buffers to match c10d
    // ProcessGroup API
    buf_list = {params.dst_bufs};
    auto work = backend->allgather(buf_list, params.src_bufs, {});
    return work;
}
  case CommunicationType::Scatter:
{
    // This is used to change the representation of the buffers to match c10d
    // ProcessGroup API
    if (is_root) {
      buf_list = {params.src_bufs};
    }
    auto work =
        backend->scatter(
                params.dst_bufs, buf_list, {.rootRank = getRootRelativeIndex(params)});
    return work;
}
  case CommunicationType::Reduce:
{
    auto& buf =
        (is_root) ? params.dst_bufs : params.src_bufs;
    c10d::ReduceOptions options = {
        .reduceOp = params.redOp, .rootRank = getRootRelativeIndex(params)};
  #if defined(NVFUSER_DISTRIBUTED) && defined(USE_C10D_NCCL)
    auto nccl_backend = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
    if (nccl_backend) {
  #if NVF_TORCH_VERSION_NO_LESS(2, 3, 0)
      // API change https://github.com/pytorch/pytorch/pull/119421
      return nccl_backend->_reduce_oop(
          buf.at(0), params.src_bufs.at(0), options);
  #else
      return nccl_backend->_reduce_oop(buf, params.src_bufs, options);
  #endif
    }
  #endif
    if (is_root) {
      doLocalCopy(params.dst_bufs.at(0), params.src_bufs.at(0));
    }
    return backend->reduce(buf, options);
}
  case CommunicationType::Allreduce:
{
    doLocalCopy(params.dst_bufs.at(0), params.src_bufs.at(0));
    return backend->allreduce(params.dst_bufs, {.reduceOp = params.redOp});
}
  case CommunicationType::ReduceScatter:
{
    // This is used to change the representation of the buffers to match c10d
    // ProcessGroup API
    buf_list = {params.src_bufs};
    auto work = backend->reduce_scatter(
                        params.dst_bufs, buf_list, {.reduceOp = params.redOp});
    return work;
}
  case CommunicationType::Broadcast:
{
    if (is_root && params.dst_bufs.size() == 1) {
      doLocalCopy(params.dst_bufs.at(0), params.src_bufs.at(0));
    }
    if (params.team.size() == 1) {
      return nullptr;
    }
    return backend->broadcast(
            is_root ? params.src_bufs : params.dst_bufs,
            {.rootRank = getRootRelativeIndex(params)});
}
  case CommunicationType::SendRecv:
{
    // if (is_root && params.team.size() == 1) {
    //   doLocalCopy(params.dst_bufs.at(0), params.src_bufs.at(0));
    //   return nullptr;
    // }
    // return comm.sendRecv(
    //     (params.team.at(0) == params.root) ? params.team.at(1)
    //                                         : params.team.at(0),
    //     params.root,
    //     params.dst_bufs.empty() ? params.src_bufs : params.dst_bufs,
    //     backend);
}
  default:
    NVF_ERROR(false);
    return nullptr;
  }
}

} // namespace nvfuser
