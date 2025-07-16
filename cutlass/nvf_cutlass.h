// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/torch.h>

namespace nvfuser::cutlass_kernels {

// Performs scaled matrix multiplication using NVFP4 format
//
// This function implements a scaled matrix multiplication C = alpha * (A @ B)
// where A and B are matrices in NVFP4 format with per-block scaling factors.
// The function uses CUTLASS kernels optimized for NVIDIA GPUs with SM100+
// architecture.
//
// Parameters:
//   a: Input matrix A in Float4_e2m1fn_x2 format (M x K/2)
//   b: Input matrix B in Float4_e2m1fn_x2 format (N x K/2)
//   scales_a: Per-block scaling factors for matrix A in FP8_E4M3 format
//   scales_b: Per-block scaling factors for matrix B in FP8_E4M3 format
//   alpha: Global scaling factor in FP32 format
//   out_dtype: Output data type (Half, BFloat16, or Float)
//
// Returns: Matrix C = alpha * (A @ B) in the specified output dtype
torch::Tensor nvfp4_scaled_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    at::ScalarType out_dtype);

void nvfp4_scaled_grouped_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets);

} // namespace nvfuser::cutlass_kernels
