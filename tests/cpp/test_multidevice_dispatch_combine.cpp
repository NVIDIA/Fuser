// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <memory>
#include <utility>

#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "fusion.h"
#include "host_ir/container.h"
#include "host_ir/evaluator.h"
#include "multidevice/communication.h"
#include "multidevice/dispatch_combine.h"
#include "tests/cpp/multidevice.h"

namespace nvfuser {
namespace hir {

using DispatchCombineTest = MultiDeviceTest;

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
  auto* in_topk_idx = makeSymbolicTensor(2, DataType::Int);
  auto* in_topk_weights = makeSymbolicTensor(2);

  auto* recv_x = makeSymbolicTensor(2);
  auto* recv_topk_idx = makeSymbolicTensor(2, DataType::Int);
  auto* recv_topk_weights = makeSymbolicTensor(2);
  auto* recv_src_idx = makeSymbolicTensor(1, DataType::Int);
  auto* recv_src_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_to_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_from_rank = makeSymbolicTensor(1, DataType::Int);

  auto* dispatch = IrBuilder::create<MoeDispatch>(
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
      num_experts,
      CommunicatorBackend::kNccl);

  auto* combined_x = makeSymbolicTensor(2);
  auto* combined_topk_weights = makeSymbolicTensor(2);
  auto* combine = IrBuilder::create<MoeCombine>(
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
  hic->addOutput(combined_x);
  hic->addOutput(combined_topk_weights);

  HostIrEvaluator hie(std::move(hic), communicator_);

  auto float_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  auto int_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kLong);

  auto x = at::arange(kNumTokens * kHidden, float_options)
               .reshape({kNumTokens, kHidden}) +
      static_cast<double>(my_rank) * 1000.0;
  auto topk_idx = at::zeros({kNumTokens, 1}, int_options);
  auto topk_weights =
      at::arange(kNumTokens, float_options)
          .reshape({kNumTokens, 1}) +
      static_cast<double>(my_rank);

  // Asymmetric example:
  // token->rank: [0, 1, 1, 1] so rank0 gets 1 token, rank1 gets 3 tokens.
  // Experts are partitioned by rank. Use rank0 expert0, rank1 experts0/1.
  topk_idx.index_put_({0, 0}, 0);
  topk_idx.index_put_({1, 0}, kNumExpertsPerRank);
  topk_idx.index_put_({2, 0}, kNumExpertsPerRank + 1);
  topk_idx.index_put_({3, 0}, kNumExpertsPerRank);

  auto outputs = hie.runWithInput(
      {{in_x, x},
       {in_topk_idx, topk_idx},
       {in_topk_weights, topk_weights}});
  auto combined = outputs[0].as<at::Tensor>();
  auto combined_weights = outputs[1].as<at::Tensor>();

  EXPECT_TRUE(at::allclose(combined, x))
      << "Dispatch/Combine mismatch on rank " << my_rank;
  EXPECT_TRUE(at::allclose(combined_weights, topk_weights))
      << "Dispatch/Combine topk_weights mismatch on rank " << my_rank;
}

TEST_F(DispatchCombineTest, DispatchOnlyTop1) {
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
  auto* in_topk_idx = makeSymbolicTensor(2, DataType::Int);
  auto* in_topk_weights = makeSymbolicTensor(2);

  auto* recv_x = makeSymbolicTensor(2);
  auto* recv_topk_idx = makeSymbolicTensor(2, DataType::Int);
  auto* recv_topk_weights = makeSymbolicTensor(2);
  auto* recv_src_idx = makeSymbolicTensor(1, DataType::Int);
  auto* recv_src_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_to_rank = makeSymbolicTensor(1, DataType::Int);
  auto* n_tokens_from_rank = makeSymbolicTensor(1, DataType::Int);

  auto* dispatch = IrBuilder::create<MoeDispatch>(
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
      num_experts,
      CommunicatorBackend::kNccl);

  hic->pushBackTopLevelExprs(dispatch);

  hic->addInput(in_x);
  hic->addInput(in_topk_idx);
  hic->addInput(in_topk_weights);
  hic->addOutput(recv_x);
  hic->addOutput(recv_topk_idx);
  hic->addOutput(recv_topk_weights);
  hic->addOutput(recv_src_idx);
  hic->addOutput(recv_src_rank);
  hic->addOutput(n_tokens_to_rank);
  hic->addOutput(n_tokens_from_rank);

  HostIrEvaluator hie(std::move(hic), communicator_);

  auto float_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  auto int_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kLong);

  auto x = at::arange(kNumTokens * kHidden, float_options)
               .reshape({kNumTokens, kHidden}) +
      static_cast<double>(my_rank) * 1000.0;
  auto topk_idx = at::zeros({kNumTokens, 1}, int_options);
  auto topk_weights =
      at::arange(kNumTokens, float_options)
          .reshape({kNumTokens, 1}) +
      static_cast<double>(my_rank);

  // Asymmetric example:
  // token->rank: [0, 1, 1, 1] so rank0 gets 1 token, rank1 gets 3 tokens.
  // Experts are partitioned by rank. Use rank0 expert0, rank1 experts0/1.
  topk_idx.index_put_({0, 0}, 0);
  topk_idx.index_put_({1, 0}, kNumExpertsPerRank);
  topk_idx.index_put_({2, 0}, kNumExpertsPerRank + 1);
  topk_idx.index_put_({3, 0}, kNumExpertsPerRank);

  auto outputs = hie.runWithInput(
      {{in_x, x},
       {in_topk_idx, topk_idx},
       {in_topk_weights, topk_weights}});

  auto expected = doMoeDispatch(
      x,
      topk_idx,
      topk_weights,
      num_experts,
      communicator_,
      CommunicatorBackend::kNccl);

  EXPECT_TRUE(at::allclose(outputs[0].as<at::Tensor>(), expected.recv_x))
      << "Dispatch recv_x mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[1].as<at::Tensor>(), expected.recv_topk_idx))
      << "Dispatch recv_topk_idx mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[2].as<at::Tensor>(), expected.recv_topk_weights))
      << "Dispatch recv_topk_weights mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[3].as<at::Tensor>(), expected.recv_src_idx))
      << "Dispatch recv_src_idx mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[4].as<at::Tensor>(), expected.recv_src_rank))
      << "Dispatch recv_src_rank mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[5].as<at::Tensor>(), expected.n_tokens_to_rank))
      << "Dispatch n_tokens_to_rank mismatch on rank " << my_rank;
  EXPECT_TRUE(
      at::allclose(outputs[6].as<at::Tensor>(), expected.n_tokens_from_rank))
      << "Dispatch n_tokens_from_rank mismatch on rank " << my_rank;
}

TEST_F(DispatchCombineTest, CombineOnlyTop1) {
  if (!communicator_->is_available() || communicator_->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks.";
  }

  const int64_t world_size = communicator_->size();
  const int64_t my_rank = communicator_->deviceId();
  constexpr int64_t kNumExpertsPerRank = 2;
  const int64_t num_experts = world_size * kNumExpertsPerRank;
  constexpr int64_t kNumTokens = 4;
  constexpr int64_t kHidden = 4;

  auto float_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kFloat);
  auto int_options =
      at::TensorOptions().device(communicator_->device()).dtype(at::kLong);

  auto x = at::arange(kNumTokens * kHidden, float_options)
               .reshape({kNumTokens, kHidden}) +
      static_cast<double>(my_rank) * 1000.0;
  auto topk_idx = at::zeros({kNumTokens, 1}, int_options);
  auto topk_weights =
      at::arange(kNumTokens, float_options)
          .reshape({kNumTokens, 1}) +
      static_cast<double>(my_rank);

  // Asymmetric example:
  // token->rank: [0, 1, 1, 1] so rank0 gets 1 token, rank1 gets 3 tokens.
  // Experts are partitioned by rank. Use rank0 expert0, rank1 experts0/1.
  topk_idx.index_put_({0, 0}, 0);
  topk_idx.index_put_({1, 0}, kNumExpertsPerRank);
  topk_idx.index_put_({2, 0}, kNumExpertsPerRank + 1);
  topk_idx.index_put_({3, 0}, kNumExpertsPerRank);

  auto dispatch_result = doMoeDispatch(
      x,
      topk_idx,
      topk_weights,
      num_experts,
      communicator_,
      CommunicatorBackend::kNccl);

  auto hic = std::make_unique<HostIrContainer>();
  FusionGuard fg(hic.get());

  auto* in_x = makeSymbolicTensor(2);
  auto* in_topk_weights = makeSymbolicTensor(2);
  auto* in_src_idx = makeSymbolicTensor(1, DataType::Int);
  auto* in_src_rank = makeSymbolicTensor(1, DataType::Int);
  auto* in_n_tokens_to_rank = makeSymbolicTensor(1, DataType::Int);
  auto* in_n_tokens_from_rank = makeSymbolicTensor(1, DataType::Int);

  auto* combined_x = makeSymbolicTensor(2);
  auto* combined_topk_weights = makeSymbolicTensor(2);
  auto* combine = IrBuilder::create<MoeCombine>(
      combined_x,
      combined_topk_weights,
      in_x,
      in_topk_weights,
      in_src_idx,
      in_src_rank,
      in_n_tokens_to_rank,
      in_n_tokens_from_rank,
      CommunicatorBackend::kNccl);

  hic->pushBackTopLevelExprs(combine);

  hic->addInput(in_x);
  hic->addInput(in_topk_weights);
  hic->addInput(in_src_idx);
  hic->addInput(in_src_rank);
  hic->addInput(in_n_tokens_to_rank);
  hic->addInput(in_n_tokens_from_rank);
  hic->addOutput(combined_x);
  hic->addOutput(combined_topk_weights);

  HostIrEvaluator hie(std::move(hic), communicator_);

  auto outputs = hie.runWithInput(
      {{in_x, dispatch_result.recv_x},
       {in_topk_weights, dispatch_result.recv_topk_weights},
       {in_src_idx, dispatch_result.recv_src_idx},
       {in_src_rank, dispatch_result.recv_src_rank},
       {in_n_tokens_to_rank, dispatch_result.n_tokens_to_rank},
       {in_n_tokens_from_rank, dispatch_result.n_tokens_from_rank}});

  auto combined = outputs[0].as<at::Tensor>();
  auto combined_weights = outputs[1].as<at::Tensor>();

  EXPECT_TRUE(at::allclose(combined, x))
      << "Combine mismatch on rank " << my_rank;
  EXPECT_TRUE(at::allclose(combined_weights, topk_weights))
      << "Combine topk_weights mismatch on rank " << my_rank;
}

} // namespace hir
} // namespace nvfuser
