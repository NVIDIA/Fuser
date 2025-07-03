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

} // namespace nvfuser::cutlass_kernels
