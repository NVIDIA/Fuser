// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <utility>

#include "fusion.h"
#include "host_ir/container.h"
#include "host_ir/evaluator.h"
#include "ir/all_nodes.h"
#include "multidevice/communication.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {
namespace hir {

class DispatchCombineTest : public MultiDeviceTest {};

TEST_F(DispatchCombineTest, DispatchCombineTop1) {
  if (!communicator_->is_available() || communicator_->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks.";
  }

  const int64_t world_size = communicator_->size();
  const int64_t my_rank = communicator_->deviceId();
  constexpr int64_t kNumExpertsPerRank = 2;
  const int64_t num_experts = world_size * kNumExpertsPerRank;
  constexpr int64_t kNumTokens = 4;
  constexpr int64_t kHidden = 4;

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* in_x = makeSymbolicTensor(2);
  auto* in_topk_idx = makeSymbolicTensor(1, DataType::Int);
  auto* in_topk_weights = makeSymbolicTensor(1);
  auto* in_is_token_in_rank = makeSymbolicTensor(2, DataType::Bool);

  auto* recv_x = makeSymbolicTensor(2);
  auto* recv_topk_idx = makeSymbolicTensor(1, DataType::Int);
  auto* recv_topk_weights = makeSymbolicTensor(1);
  auto* recv_src_idx = makeSymbolicTensor(1, DataType::Int);
  auto* recv_src_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_to_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_from_rank = makeSymbolicTensor(1, DataType::Int);

  auto* dispatch = IrBuilder::create<MoEDispatch>(
      recv_x,
      recv_topk_idx,
      recv_topk_weights,
      recv_src_idx,
      recv_src_rank,
      n_tokens_to_rank,
      n_tokens_from_rank,
      in_x,
      in_topk_idx,
      in_topk_weights,
      in_is_token_in_rank,
      num_experts,
      CommunicatorBackend::kNccl);

  auto* combined_x = makeSymbolicTensor(2);
  auto* combined_topk_weights = makeSymbolicTensor(1);
  auto* combine = IrBuilder::create<MoECombine>(
      combined_x,
      combined_topk_weights,
      recv_x,
      recv_topk_weights,
      recv_src_idx,
      recv_src_rank,
      n_tokens_to_rank,
      n_tokens_from_rank,
      CommunicatorBackend::kNccl);

  hic->pushBackTopLevelExprs(dispatch);
  hic->pushBackTopLevelExprs(combine);

  hic->addInput(in_x);
  hic->addInput(in_topk_idx);
  hic->addInput(in_topk_weights);
  hic->addInput(in_is_token_in_rank);
  hic->addOutput(combined_x);

  HostIrEvaluator hie(std::move(hic), communicator_);

  auto float_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  auto int_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kLong);

  auto x = at::arange(kNumTokens * kHidden, float_options)
               .reshape({kNumTokens, kHidden}) +
      static_cast<double>(my_rank) * 1000.0;
  auto topk_idx = at::zeros({kNumTokens}, int_options);
  auto topk_weights = at::ones({kNumTokens}, float_options);

  // Asymmetric example:
  // token->rank: [0, 1, 1, 1] so rank0 gets 1 token, rank1 gets 3 tokens.
  auto rank_ids = at::arange(world_size, int_options);
  auto token_rank = at::tensor({0, 1, 1, 1}, int_options);
  auto is_token_in_rank = token_rank.unsqueeze(1).eq(rank_ids);

  // Experts are partitioned by rank. Use rank0 expert0, rank1 experts0/1.
  topk_idx.index_put_({0}, 0);
  topk_idx.index_put_({1}, kNumExpertsPerRank);
  topk_idx.index_put_({2}, kNumExpertsPerRank + 1);
  topk_idx.index_put_({3}, kNumExpertsPerRank);

  auto outputs = hie.runWithInput(
      {{in_x, x},
       {in_topk_idx, topk_idx},
       {in_topk_weights, topk_weights},
       {in_is_token_in_rank, is_token_in_rank}});
  auto combined = outputs.back().as<at::Tensor>();

  EXPECT_TRUE(at::allclose(combined, x))
      << "Dispatch/Combine mismatch on rank " << my_rank;
}

} // namespace hir
} // namespace nvfuser
