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

// Validates all input parameters and tensor properties for NVFP4 scaled matrix
// multiplication
//
// This function performs comprehensive validation of input tensors including:
// - CUDA device and contiguity checks
// - Data type validation for all inputs
// - Matrix dimension and shape compatibility
// - Alignment requirements for optimal performance
// - Scale matrix shape validation
//
// Parameters:
//   a, b: Input matrices to validate
//   scales_a, scales_b: Scale matrices to validate
//   alpha: Alpha scaling factor to validate
//
// Returns: Tuple of (m, n, k) dimensions for the GEMM operation
//
// Throws: NVF_CHECK exceptions for any validation failures
std::tuple<int64_t, int64_t, int64_t> validateInputsNvfp4ScaledMm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    bool skip_checks = false);

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
    at::ScalarType out_dtype,
    bool skip_checks = false);

} // namespace nvfuser::cutlass_kernels
