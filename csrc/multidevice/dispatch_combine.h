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

// Notation used throughout dispatch/combine:
//   T = num_tokens    local token count on this rank before dispatch
//   H = hidden        hidden dimension
//   K = topk          only K=1 supported
//   R = world_size    number of ranks
//   C = T * R         capacity (upper-bound recv allocation, CUDA backend)
//   V = Σ n_tokens_from_rank   actual tokens received  (V ≤ C)

struct DispatchResult {
  // NCCL: recv_* are [V, ...] (tight).
  // CUDA: recv_* are [C, ...] (over-allocated for graph capture).
  //   Rows [0,V) are valid in sender-rank order; [V,C) are padding.
  at::Tensor recv_x; // [C|V, H]
  at::Tensor recv_topk_idx; // [C|V, K]
  at::Tensor recv_topk_weights; // [C|V, K]
  at::Tensor recv_src_idx; // [C|V]
  at::Tensor n_tokens_to_rank; // [R]
  at::Tensor n_tokens_from_rank; // [R]
};

struct CombineResult {
  at::Tensor combined_x; // [T, H]
};

//! Dispatch MoE tokens to expert-owning ranks via alltoall (topk=1 only).
//!
//! CUDA backend: fully graph-capturable. Recv buffers are over-allocated to
//! C = T*R so all shapes are CPU-deterministic (see DispatchResult).
//! Buffer allocation and IPC setup happen once (rendezvous).
//!
//! NCCL backend: exact sizes, not graph-capturable.
NVF_API DispatchResult doMoeDispatch(
    const at::Tensor& x, // [T, H]
    const at::Tensor& topk_idx, // [T, K]
    const at::Tensor& topk_weights, // [T, K]
    int64_t num_experts,
    Communicator* communicator,
    CommunicatorBackend backend);

//! Combine dispatched MoE results back to original token order via alltoall.
//!
//! CUDA backend: fully graph-capturable. x may be [C, H] (padded from
//! dispatch); tokens are already in rank-order so no sort is needed.
//! Recv buffers are [T, H] (exact). Output combined_x is [T, H].
//!
//! \p num_tokens T — the original local count (= x.size(0) at dispatch
//! time). CPU-known; used for recv sizing and output allocation so the
//! data path needs no GPU-to-CPU sync.
//!
//! NCCL backend: exact sizes, not graph-capturable.
NVF_API CombineResult doMoeCombine(
    const at::Tensor& x, // [C|V, H]
    const at::Tensor& topk_weights, // [C|V, K]
    const at::Tensor& src_idx, // [C|V]
    const at::Tensor& n_tokens_to_rank, // [R]
    const at::Tensor& n_tokens_from_rank, // [R]
    int64_t num_tokens, // T
    Communicator* communicator,
    CommunicatorBackend backend);

} // namespace nvfuser
