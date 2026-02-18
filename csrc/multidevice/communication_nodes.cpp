// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "multidevice/communication_nodes.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>

#include "ir/cloner.h"
#include "ir/iostream.h"

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
    case CommunicationType::AllToAll:
      os << "AllToAll";
      break;
  }
  return os;
}

namespace {

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
    case CommunicationType::AllToAll:
      return false;
  }
  std::unreachable();
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
    case CommunicationType::AllToAll:
      return false;
    default:
      NVF_THROW("unrecognized CommunicationType: ", type);
  }
}

int64_t getRelativeIndex(const Team& team, const DeviceIdxType rank) {
  auto i = std::find(team.begin(), team.end(), rank);
  NVF_ERROR(i != team.end(), "Unable to find rank ", rank, " in team ", team);
  return std::distance(team.begin(), i);
}

} // namespace

Communication::Communication(
    IrBuilderPasskey passkey,
    CommunicationType type,
    TensorView* out,
    TensorView* in,
    Team team,
    Val* root,
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
  addInput(root);
  addOutput(out);
  addDataAttribute(type);
  addDataAttribute(team);
  addDataAttribute(red_op);
  addDataAttribute(backend);

  validate();
}

Communication::Communication(
    IrBuilderPasskey passkey,
    CommunicationType type,
    TensorView* out,
    TensorView* in,
    Team team,
    DeviceIdxType root,
    RedOpType red_op,
    CommunicatorBackend backend)
    : Communication(
          passkey,
          type,
          out,
          in,
          team,
          IrBuilder::createInContainer<Val>(
              passkey.ir_container_,
              root,
              DataType::Index),
          red_op,
          backend) {}

void Communication::validate() {
  if (root()->isConstScalar() && root()->isIntegralScalar()) {
    auto root_val = root()->evaluate().as<int64_t>();
    NVF_ERROR(
        hasRoot(type()) == (root_val >= 0),
        "Root ",
        root_val,
        " is not expected by CommunicationType ",
        type());
  }
  NVF_ERROR(isReduction(type()) == (reduceOp() != RedOpType::UNUSED));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(Communication)

int64_t Communication::getRootRelativeIndex(DeviceIdxType root_val) {
  return getRelativeIndex(team(), root_val);
}

std::string Communication::toInlineString(const int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Communication " << name() << " ("
                          << "type=" << type() << ", " << "team=(" << team()
                          << ")";
  if (hasRoot(type())) {
    ss << ", root=" << root()->toInlineString();
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
                          << "type=" << type() << ", " << "buffer=" << buffer()
                          << ", " << "peer=" << peer() << ", "
                          << "backend=" << backend() << ")";
  return ss.str();
}

std::string P2PCommunication::toString(int indent_size) const {
  return toInlineString(indent_size) + "\n";
}

MoeDispatch::MoeDispatch(
    IrBuilderPasskey passkey,
    TensorView* out_x,
    TensorView* out_topk_idx,
    TensorView* out_topk_weights,
    TensorView* out_src_idx,
    TensorView* out_n_tokens_to_rank,
    TensorView* out_n_tokens_from_rank,
    TensorView* in_x,
    TensorView* in_topk_idx,
    TensorView* in_topk_weights,
    int64_t num_experts,
    CommunicatorBackend backend)
    : Expr(passkey) {
  addInput(in_x);
  addInput(in_topk_idx);
  addInput(in_topk_weights);
  addOutput(out_x);
  addOutput(out_topk_idx);
  addOutput(out_topk_weights);
  addOutput(out_src_idx);
  addOutput(out_n_tokens_to_rank);
  addOutput(out_n_tokens_from_rank);
  addDataAttribute(num_experts);
  addDataAttribute(backend);
  validate();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MoeDispatch)

std::string MoeDispatch::toInlineString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Dispatch " << name() << " ("
                          << "num_experts=" << numExperts() << ", "
                          << "backend=" << backend() << ", "
                          << "in=" << inX() << ", "
                          << "topk_idx=" << inTopkIdx() << ", "
                          << "topk_weights=" << inTopkWeights() << ", "
                          << "out=" << outX() << ")";
  return ss.str();
}

std::string MoeDispatch::toString(int indent_size) const {
  return toInlineString(indent_size) + "\n";
}

void MoeDispatch::validate() {
  NVF_CHECK(numExperts() > 0, "num_experts must be positive.");
  NVF_CHECK(inX()->isA<TensorView>(), "in_x must be a TensorView.");
  NVF_CHECK(inTopkIdx()->isA<TensorView>(), "topk_idx must be a TensorView.");
  NVF_CHECK(
      inTopkIdx()->getDataType().has_value() &&
          isIntegralType(*inTopkIdx()->getDataType()),
      "topk_idx must be integral.");
  NVF_CHECK(
      inTopkWeights()->getDataType().has_value() &&
          isFloatingPointType(*inTopkWeights()->getDataType()),
      "topk_weights must be floating point.");
  NVF_CHECK(
      outTopkIdx()->getDataType().has_value() &&
          isIntegralType(*outTopkIdx()->getDataType()),
      "out_topk_idx must be integral.");
  NVF_CHECK(
      outTopkWeights()->getDataType().has_value() &&
          isFloatingPointType(*outTopkWeights()->getDataType()),
      "out_topk_weights must be floating point.");
  NVF_CHECK(
      outSrcIdx()->getDataType().has_value() &&
          isIntegralType(*outSrcIdx()->getDataType()),
      "out_src_idx must be integral.");
  NVF_CHECK(
      outTokensToRank()->getDataType().has_value() &&
          isIntegralType(*outTokensToRank()->getDataType()),
      "out_n_tokens_to_rank must be integral.");
  NVF_CHECK(
      outTokensFromRank()->getDataType().has_value() &&
          isIntegralType(*outTokensFromRank()->getDataType()),
      "out_n_tokens_from_rank must be integral.");
}

MoeCombine::MoeCombine(
    IrBuilderPasskey passkey,
    TensorView* out_x,
    TensorView* in_x,
    TensorView* in_topk_weights,
    TensorView* in_src_idx,
    TensorView* in_n_tokens_to_rank,
    TensorView* in_n_tokens_from_rank,
    CommunicatorBackend backend)
    : Expr(passkey) {
  addInput(in_x);
  addInput(in_topk_weights);
  addInput(in_src_idx);
  addInput(in_n_tokens_to_rank);
  addInput(in_n_tokens_from_rank);
  addOutput(out_x);
  addDataAttribute(backend);
  validate();
}

NVFUSER_DEFINE_CLONE_AND_CREATE(MoeCombine)

std::string MoeCombine::toInlineString(int indent_size) const {
  std::stringstream ss;
  indent(ss, indent_size) << "Combine " << name() << " ("
                          << "backend=" << backend() << ", "
                          << "in=" << inX() << ", "
                          << "topk_weights=" << inTopkWeights() << ", "
                          << "src_idx=" << inSrcIdx() << ", "
                          << "out=" << outX() << ")";
  return ss.str();
}

std::string MoeCombine::toString(int indent_size) const {
  return toInlineString(indent_size) + "\n";
}

void MoeCombine::validate() {
  NVF_CHECK(inX()->isA<TensorView>(), "in_x must be a TensorView.");
  NVF_CHECK(
      inTopkWeights()->getDataType().has_value() &&
          isFloatingPointType(*inTopkWeights()->getDataType()),
      "in_topk_weights must be floating point.");
  NVF_CHECK(
      inSrcIdx()->getDataType().has_value() &&
          isIntegralType(*inSrcIdx()->getDataType()),
      "in_src_idx must be integral.");
  NVF_CHECK(
      inTokensToRank()->getDataType().has_value() &&
          isIntegralType(*inTokensToRank()->getDataType()),
      "in_n_tokens_to_rank must be integral.");
  NVF_CHECK(
      inTokensFromRank()->getDataType().has_value() &&
          isIntegralType(*inTokensFromRank()->getDataType()),
      "in_n_tokens_from_rank must be integral.");
}

} // namespace nvfuser
