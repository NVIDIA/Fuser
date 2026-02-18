// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#ifdef NVFUSER_DISTRIBUTED
#include <torch/csrc/distributed/c10d/Types.hpp>
#else
#include "multidevice/c10d_mock.h"
#endif

#include "ir/base_nodes.h"
#include "ir/builder.h"
#include "ir/interface_nodes.h"
#include "multidevice/communicator.h"
#include "multidevice/device_mesh.h"
#include "multidevice/multidevice.h"

namespace nvfuser {

enum class CommunicationType {
  Gather,
  Allgather,
  Scatter,
  Reduce,
  Allreduce,
  ReduceScatter,
  Broadcast,
  SendRecv,
  AllToAll
};

std::ostream& operator<<(std::ostream& os, const CommunicationType& type);

using RedOpType = c10d::ReduceOp::RedOpType;

// The class "Communication" represents a MPI-style communication
// communication operation to be executed on the network. The base class
// Communication should not be used directly but through its derived classes:
// Broadcast, Gather, Scatter, Allgather, and SendRecv. Other collectives will
// be added later.
class Communication : public Expr {
 public:
  using Expr::Expr;
  // Only specify `root` for types that have root.
  // Only specify `red_op` for reduction types.
  Communication(
      IrBuilderPasskey passkey,
      CommunicationType type,
      TensorView* out,
      TensorView* in,
      Team team, // All devices involved in this communication. It must include
                 // `root`. It can be a subset of `root`+`mesh` in case of 2D
                 // sharding.
      Val* root,
      RedOpType red_op = RedOpType::UNUSED,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  Communication(
      IrBuilderPasskey passkey,
      CommunicationType type,
      TensorView* out,
      TensorView* in,
      Team team, // All devices involved in this communication. It must include
                 // `root`. It can be a subset of `root`+`mesh` in case of 2D
                 // sharding.
      DeviceIdxType root = -1,
      RedOpType red_op = RedOpType::UNUSED,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  Communication(const Communication& other) = delete;
  Communication& operator=(const Communication& other) = delete;
  Communication(Communication&& other) = delete;
  Communication& operator=(Communication&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "Communication";
  }

  CommunicationType type() const {
    return attribute<CommunicationType>(0);
  }

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  const Team& team() const {
    return attribute<Team>(1);
  }

  // A convenience helper so the user doesn't need to convert size_t to int64_t.
  int64_t team_size() const {
    return static_cast<int64_t>(team().size());
  }

  Val* root() const {
    return input(1);
  }

  RedOpType reduceOp() const {
    return attribute<RedOpType>(2);
  }

  CommunicatorBackend backend() const {
    return attribute<CommunicatorBackend>(3);
  }

  // PyTorch's process group expects the root to be specified
  // as an integer between 0 and world_size-1. We choose it to be
  // the device's relative index within the team
  int64_t getRootRelativeIndex(DeviceIdxType root_val);

 private:
  void validate();
};

enum class P2PCommunicationType { SEND, RECV };

std::ostream& operator<<(std::ostream& os, const P2PCommunicationType& type);

class P2PCommunication : public Expr {
 public:
  using Expr::Expr;

  P2PCommunication(
      IrBuilderPasskey passkey,
      P2PCommunicationType type,
      TensorView* buffer,
      Val* peer,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  P2PCommunication(const P2PCommunication& other) = delete;
  P2PCommunication& operator=(const P2PCommunication& other) = delete;
  P2PCommunication(P2PCommunication&& other) = delete;
  P2PCommunication& operator=(P2PCommunication&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "P2PCommunication";
  }

  P2PCommunicationType type() const {
    return attribute<P2PCommunicationType>(0);
  }

  TensorView* buffer() const {
    return input(0)->as<TensorView>();
  }

  Val* peer() const {
    return attributeVal(1);
  }

  auto backend() const {
    return attribute<CommunicatorBackend>(2);
  }
};

// Dispatch represents intra-node MoE token dispatch. It shuffles tokens from
// the local rank to destination ranks based on explicit routing.
//
// Example shapes (topk=1):
//   in_x: [T, H], in_topk_idx: [T, 1],
//   in_topk_weights: [T, 1], num_experts = R * experts_per_rank.
//   For topk>1, use [T, K] for both topk inputs.
//   Experts are assumed to be placed contiguously by rank.
//   out_src_idx is returned for the combine step to restore the original token
//   order.
//   Outputs are recv-aligned tensors: out_x/out_topk_idx/out_topk_weights/
//   out_src_* with [T_recv, ...] and
//   out_n_tokens_to_rank/out_n_tokens_from_rank with shape [R].
class MoeDispatch : public Expr {
 public:
  using Expr::Expr;

  MoeDispatch(
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
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  MoeDispatch(const MoeDispatch& other) = delete;
  MoeDispatch& operator=(const MoeDispatch& other) = delete;
  MoeDispatch(MoeDispatch&& other) = delete;
  MoeDispatch& operator=(MoeDispatch&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "MoeDispatch";
  }

  TensorView* outX() const {
    return output(0)->as<TensorView>();
  }

  TensorView* outTopkIdx() const {
    return output(1)->as<TensorView>();
  }

  TensorView* outTopkWeights() const {
    return output(2)->as<TensorView>();
  }

  TensorView* outSrcIdx() const {
    return output(3)->as<TensorView>();
  }

  TensorView* outTokensToRank() const {
    return output(4)->as<TensorView>();
  }

  TensorView* outTokensFromRank() const {
    return output(5)->as<TensorView>();
  }

  TensorView* inX() const {
    return input(0)->as<TensorView>();
  }

  TensorView* inTopkIdx() const {
    return input(1)->as<TensorView>();
  }

  TensorView* inTopkWeights() const {
    return input(2)->as<TensorView>();
  }

  int64_t numExperts() const {
    return attribute<int64_t>(0);
  }

  CommunicatorBackend backend() const {
    return attribute<CommunicatorBackend>(1);
  }

 private:
  void validate();
};

// Combine represents intra-node MoE token combine. It shuffles tokens back to
// their source ranks using `in_src_idx`.
//
// Example shapes (topk=1):
//   in_x: [T_recv, H], in_topk_weights: [T_recv, 1], in_src_idx: [T_recv],
//   in_n_tokens_to_rank: [R], in_n_tokens_from_rank:
//   [R]. Output out_x is source-aligned with shape [T_src, ...].
class MoeCombine : public Expr {
 public:
  using Expr::Expr;

  MoeCombine(
      IrBuilderPasskey passkey,
      TensorView* out_x,
      TensorView* in_x,
      TensorView* in_topk_weights,
      TensorView* in_src_idx,
      TensorView* in_n_tokens_to_rank,
      TensorView* in_n_tokens_from_rank,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  MoeCombine(const MoeCombine& other) = delete;
  MoeCombine& operator=(const MoeCombine& other) = delete;
  MoeCombine(MoeCombine&& other) = delete;
  MoeCombine& operator=(MoeCombine&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "MoeCombine";
  }

  TensorView* outX() const {
    return output(0)->as<TensorView>();
  }

  TensorView* inX() const {
    return input(0)->as<TensorView>();
  }

  TensorView* inTopkWeights() const {
    return input(1)->as<TensorView>();
  }

  TensorView* inSrcIdx() const {
    return input(2)->as<TensorView>();
  }

  TensorView* inTokensToRank() const {
    return input(3)->as<TensorView>();
  }

  TensorView* inTokensFromRank() const {
    return input(4)->as<TensorView>();
  }

  CommunicatorBackend backend() const {
    return attribute<CommunicatorBackend>(0);
  }

 private:
  void validate();
};

} // namespace nvfuser
