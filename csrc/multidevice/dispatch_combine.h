// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
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
  at::Tensor recv_x;
  at::Tensor recv_topk_idx;
  at::Tensor recv_topk_weights;
  at::Tensor recv_src_idx;
  at::Tensor recv_src_rank;
  at::Tensor n_tokens_to_rank;
  at::Tensor n_tokens_from_rank;
};

struct CombineResult {
  at::Tensor combined_x;
  at::Tensor combined_topk_weights;
};

NVF_API DispatchResult dispatchWithCudaBackend(
    const at::Tensor& x,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    const at::Tensor& is_token_in_rank,
    int64_t num_experts,
    Communicator* communicator,
    CommunicatorBackend backend);

NVF_API CombineResult combineWithCudaBackend(
    const at::Tensor& x,
    const at::Tensor& topk_weights,
    const at::Tensor& src_idx,
    const at::Tensor& src_rank,
    const at::Tensor& n_tokens_to_rank,
    const at::Tensor& n_tokens_from_rank,
    Communicator* communicator,
    CommunicatorBackend backend);

} // namespace nvfuser
