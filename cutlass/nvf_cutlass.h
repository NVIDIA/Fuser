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

bool nvfp4_scaled_mm_check(
    at::ScalarType output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha);

void nvfp4_scaled_mm(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha);

void nvfp4_quantize(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    const torch::Tensor& input,
    const torch::Tensor& input_scale);

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
