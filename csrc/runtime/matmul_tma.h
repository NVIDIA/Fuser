// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/torch.h>

namespace nvfuser {

//! Run an SM90 TMA-based matmul (A[M,K] x B[K,N]) on the current CUDA stream.
//! Returns a new output tensor with the same dtype as the inputs.
at::Tensor matmulTma(const at::Tensor& a, const at::Tensor& b);

} // namespace nvfuser
