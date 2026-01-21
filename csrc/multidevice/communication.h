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
#include "type.h"
#include "visibility.h"

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
// the local rank to destination ranks based on `in_is_token_in_rank`.
//
// Example shapes (topk=1):
//   in_x: [T, H], in_topk_idx: [T] or [T, 1], in_topk_weights: [T] or [T, 1],
//   in_is_token_in_rank: [T, R] (one-hot), num_experts = R * experts_per_rank.
//   Outputs are recv-aligned tensors: out_x/out_topk_*/out_src_* with [T_recv,
//   ...] and out_n_tokens_to_rank/out_n_tokens_from_rank with shape [R].
class MoEDispatch : public Expr {
 public:
  using Expr::Expr;

  MoEDispatch(
      IrBuilderPasskey passkey,
      TensorView* out_x,
      TensorView* out_topk_idx,
      TensorView* out_topk_weights,
      TensorView* out_src_idx,
      TensorView* out_src_rank,
      TensorView* out_n_tokens_to_rank,
      TensorView* out_n_tokens_from_rank,
      TensorView* in_x,
      TensorView* in_topk_idx,
      TensorView* in_topk_weights,
      TensorView* in_is_token_in_rank,
      int64_t num_experts,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  MoEDispatch(const MoEDispatch& other) = delete;
  MoEDispatch& operator=(const MoEDispatch& other) = delete;
  MoEDispatch(MoEDispatch&& other) = delete;
  MoEDispatch& operator=(MoEDispatch&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "MoEDispatch";
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

  TensorView* outSrcRank() const {
    return output(4)->as<TensorView>();
  }

  TensorView* outTokensToRank() const {
    return output(5)->as<TensorView>();
  }

  TensorView* outTokensFromRank() const {
    return output(6)->as<TensorView>();
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

  TensorView* inIsTokenInRank() const {
    return input(3)->as<TensorView>();
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
// their source ranks using `in_src_rank` and `in_src_idx`.
//
// Example shapes (topk=1):
//   in_x: [T_recv, H], in_topk_weights: [T_recv], in_src_idx: [T_recv],
//   in_src_rank: [T_recv], in_n_tokens_to_rank: [R], in_n_tokens_from_rank:
//   [R]. Outputs are source-aligned: out_x/out_topk_weights with shape [T_src,
//   ...].
class MoECombine : public Expr {
 public:
  using Expr::Expr;

  MoECombine(
      IrBuilderPasskey passkey,
      TensorView* out_x,
      TensorView* out_topk_weights,
      TensorView* in_x,
      TensorView* in_topk_weights,
      TensorView* in_src_idx,
      TensorView* in_src_rank,
      TensorView* in_n_tokens_to_rank,
      TensorView* in_n_tokens_from_rank,
      CommunicatorBackend backend = CommunicatorBackend::kNccl);

  MoECombine(const MoECombine& other) = delete;
  MoECombine& operator=(const MoECombine& other) = delete;
  MoECombine(MoECombine&& other) = delete;
  MoECombine& operator=(MoECombine&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "MoECombine";
  }

  TensorView* outX() const {
    return output(0)->as<TensorView>();
  }

  TensorView* outTopkWeights() const {
    return output(1)->as<TensorView>();
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

  TensorView* inSrcRank() const {
    return input(3)->as<TensorView>();
  }

  TensorView* inTokensToRank() const {
    return input(4)->as<TensorView>();
  }

  TensorView* inTokensFromRank() const {
    return input(5)->as<TensorView>();
  }

  CommunicatorBackend backend() const {
    return attribute<CommunicatorBackend>(0);
  }

 private:
  void validate();
};

// The method "post" triggers the execution of the communication. This call is
// non-blocking. The communication can be posted multiple times.
// It is assumed that the current device_index (given by
// communicator.deviceId()) belongs to the team of the communication,
// otherwise an error is thrown.
//
// NOTE: pytorch's NCCL process group API needs <team_size> buffers on root for
// scatter/gather operation.
// (*) Broadcast
// Copies the root's src buffer to each device's dst buffer
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffer, and no or one dst buffer
//   - non-roots have no src buffer and one dst buffer
//   - all buffers have the same size
// (*) Gather
// Copies each device's source buffer to the root's respective src
// buffer. The order of the sender devices matches the order of the
// root's buffers.
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffer and <team_size> dst buffers
//   - non-roots have one src buffer and no dst buffer
//   - all buffers have the same size
// (*) Allgather
// Copies each device's src buffer to each device's respective src
// buffer. The order of the devices matches the order of the
// buffers
// Requirements:
//   - all device have one src buffer and <team_size> dst buffers
//   - all buffers have the same size
// (*) Scatter
// Copies each root's src buffer to each device's dst buffer.
// The order of the buffers matches the order of the receiver devices
// Requirements:
//   - the root is set and belongs to the team
//   - the root has <team_size> src buffers and one dst buffer
//   - non-roots have no src buffer and one dst buffer
//   - all buffers have the same size
// (*) Reduce
// Reduce the src buffers to the root's dst buffer.
// Requirements:
//   - the root is set and belongs to the team
//   - the root has one src buffers and one dst buffer
//   - non-roots have one src buffer and no dst buffer
//   - all buffers have the same size
// (*) Allreduce
// Reduce the src buffers to the dst buffer.
// Requirements:
//   - all devices have one src buffer and one dst buffer
//   - all buffers have the same size
// (*) ReduceScatter
// Reduce all the src buffers and shard the result to the dst buffers.
// Requirements:
//   - all devices have <team_size> src buffer and one dst buffer
//   - all buffers have the same size
// (*) SendRecv
// Copies the sender's src buffers to the receiver's dst buffer
// It is equivalent to a Broadcast with a team of size == 2
c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10d::Backend* backend,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    DeviceIdxType root_index = -1);

c10::intrusive_ptr<c10d::Work> postSingleCommunication(
    P2PCommunication* communication,
    DeviceIdxType my_device_index,
    DeviceIdxType peer,
    c10d::Backend* backend,
    at::Tensor buffer);

} // namespace nvfuser
