// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ATen/core/Tensor.h>

#include "multidevice/communicator.h"
#include "visibility.h"

namespace nvfuser {

struct DispatchResult {
  at::Tensor recv_x; // Dispatched tokens received on this rank.
  at::Tensor recv_topk_idx; // Expert ids aligned with recv_x.
  at::Tensor recv_topk_weights; // Gating weights aligned with recv_x.
  at::Tensor recv_src_idx; // Source token indices for combine.
  at::Tensor recv_src_rank; // Source ranks for combine.
  at::Tensor n_tokens_to_rank; // Tokens sent to each rank (this rank's view).
  at::Tensor n_tokens_from_rank; // Tokens received from each rank.
};

struct CombineResult {
  at::Tensor combined_x; // Combined tokens back in original order.
  at::Tensor combined_topk_weights; // Combined gating weights per token.
};

// Dispatch MoE tokens to the owning ranks. Only k=1 is supported for now.
//
// Args:
//   x: Token embeddings on this rank, shape [T, H].
//   topk_idx: Global expert ids per token, shape [T, K] (K=1 supported).
//   topk_weights: Gating weights per token, shape [T, K] (K=1 supported).
//   Experts are assumed to be placed contiguously by rank so
//   rank = topk_idx / experts_per_rank.
//   num_experts: Total experts across all ranks (must be divisible by R).
//   communicator: Communicator for alltoall exchange.
//   backend: Communication backend (only NCCL is supported for now).
//
// Returns:
//   DispatchResult with recv_* tensors on this rank.
//
// Example:
//   world_size=2, num_experts=4, T=4, H=2, topk=1
//   Experts are partitioned by rank:
//     rank0 owns experts {0, 1}, rank1 owns experts {2, 3}
//   Rank0 holds tokens 0,1 and rank1 holds tokens 2,3 in x:
//     rank0 x = [x0, x1], rank1 x = [x2, x3]
//   token->rank: [0, 1, 1, 1]  (via expert ids below)
//   topk_idx = [[0], [2], [3], [2]]  (global expert ids, contiguous per rank)
//   After dispatch on rank0:
//     recv_x has token {0}
//     recv_topk_idx aligned with recv_x (e.g., [0])
//     recv_topk_weights aligned with recv_x (e.g., [1.0])
//     recv_src_idx tells original token positions (e.g., [0])
//   After dispatch on rank1:
//     recv_x has tokens {1, 2, 3}
//     recv_topk_idx aligned with recv_x (e.g., [2, 2, 3]). Tokens are grouped
//     by expert id for local expert processing.
//     recv_src_idx tells original token positions (e.g., [1, 2, 3])
//   auto out = doMoeDispatch(
//       x,
//       topk_idx,
//       topk_weights,
//       4,
//       comm,
//       CommunicatorBackend::kNccl);
NVF_API DispatchResult doMoeDispatch(
    const at::Tensor& x, // [T, H]
    const at::Tensor& topk_idx, // [T] or [T, 1]
    const at::Tensor& topk_weights, // [T] or [T, 1]
    int64_t num_experts,
    Communicator* communicator,
    CommunicatorBackend backend);

// Combine dispatched MoE results back to original token order.
//
// Args:
//   x: Token embeddings after expert compute, shape [T_recv, H].
//   topk_weights: Gating weights aligned with x, shape [T_recv, K] (K=1).
//   src_idx: Original token indices for each row of x, shape [T_recv].
//   src_rank: Original source rank per token, shape [T_recv].
//   n_tokens_to_rank: Tokens sent to each rank (from dispatch), shape [R].
//   n_tokens_from_rank: Tokens received from each rank (from dispatch), shape
//   [R].
//   communicator: Communicator for alltoall exchange.
//   backend: Communication backend (only NCCL is supported for now).
//
// Returns:
//   CombineResult with tokens restored to original order on this rank.
//
// Example:
//   // Continuing the dispatch example (experts partitioned by rank):
//   // rank0 owns experts {0, 1}, rank1 owns experts {2, 3}
//   // After expert compute:
//   //   rank0 recv_x has token {0} with src_idx = [0], src_rank = [0]
//   //   rank1 recv_x has tokens {1, 2, 3} with src_idx = [1, 2, 3],
//   //   src_rank = [0, 1, 1]
//   // n_tokens_to_rank and n_tokens_from_rank are [R] counts per rank.
//   // Combine scatters results back to original token order per rank.
//   auto combined = doMoeCombine(
//       x, topk_weights, src_idx, src_rank, n_tokens_to_rank,
//       n_tokens_from_rank, comm, CommunicatorBackend::kNccl);
NVF_API CombineResult doMoeCombine(
    const at::Tensor& x,
    const at::Tensor& topk_weights,
    const at::Tensor& src_idx,
    const at::Tensor& src_rank,
    const at::Tensor& n_tokens_to_rank,
    const at::Tensor& n_tokens_from_rank,
    Communicator* communicator,
    CommunicatorBackend backend);

} // namespace nvfuser
