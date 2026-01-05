// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <torch/torch.h>
#include <visibility.h>
#include <utility>

namespace nvfuser::cutlass_kernels {

// Helper function to compute ceil(x / y)
inline int64_t ceilDiv(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Helper function to round up to the nearest multiple of y
inline int64_t roundUp(int64_t x, int64_t y) {
  return ceilDiv(x, y) * y;
}

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
NVF_API std::tuple<int64_t, int64_t, int64_t> validateInputsNvfp4ScaledMm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    bool skip_checks = false);

// Performs scaled matrix multiplication using NVFP4 format.
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
//   alpha: Combined global scaling factor for operands A and B in FP32 format
//   out_dtype: Output data type (Half, BFloat16, or Float)
//
// Returns: Matrix C = alpha * (A @ B) in the specified output dtype
NVF_API torch::Tensor nvfp4_scaled_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    at::ScalarType out_dtype,
    bool skip_checks = false);

// Validates all input parameters and tensor properties for MXFP8 scaled matrix
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
NVF_API std::tuple<int64_t, int64_t, int64_t> validateInputsMxFp8ScaledMm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    bool skip_checks = false);

// Performs scaled matrix multiplication using MXFP8 format.
//
// This function implements a scaled matrix multiplication C = alpha * (A @ B)
// where A and B are matrices in MXFP8 format with per-block scaling factors.
// The function uses CUTLASS kernels optimized for NVIDIA GPUs with SM100+
// architecture.
//
// Parameters:
//   a: Input matrix A in Float8_e4m3fn format
//   b: Input matrix B in Float8_e4m3fn format
//   scales_a: Per-block scaling factors for matrix A in FP8_E8M0fnu format
//   scales_b: Per-block scaling factors for matrix B in FP8_E8M0fnu format
//   alpha: Combined global scaling factor for operands A and B in FP32 format
//   out_dtype: Output data type (Half, BFloat16, or Float)
//
// Returns: Matrix C = alpha * (A @ B) in the specified output dtype
NVF_API torch::Tensor mxfp8_scaled_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    at::ScalarType out_dtype,
    bool skip_checks = false);

// Performs scaled matrix multiplication using NVFP4 format with fused epilogue
// blockscale quantization.
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
//   alpha: Combined global scaling factor for operands A and B in FP32 format
//   global_normconst: Global scaling factor for output matrix C in FP32 format
//
// Returns: A tuple of nvfp4 output matrix and blockscale factor.
//  * Matrix C = alpha * (A @ B) in Float4_e2m1fn_x2 format (M x N/2)
//  * Blockscale factor for Matrix C in Float8_e4m3fn format
NVF_API std::pair<torch::Tensor, torch::Tensor> nvfp4_scaled_mm_blockscale(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    const torch::Tensor& global_normconst,
    bool skip_checks = false);

// Performs grouped scaled matrix multiplication using NVFP4 format.
//
// This function implements multiple scaled matrix multiplications in a grouped
// format, where each group represents a separate GEMM operation with its own
// parameters. The function is optimized for scenarios like mixture-of-experts
// models where multiple independent matrix multiplications need to be performed
// efficiently. The function uses CUTLASS kernels optimized for NVIDIA GPUs
// with SM100+ architecture.
//
// Parameters:
//   a: Input matrix A in Float4_e2m1fn_x2 format (M x K/2)
//   b: Input matrix B in Float4_e2m1fn_x2 format (G x N, K/2)
//   a_blockscale: Per-block scaling factors for matrix A in FP8_E4M3 format
//   b_blockscales: Per-block scaling factors for matrix B in FP8_E4M3 format
//   alphas: Global scaling factors for each group in FP32 format
//   ab_strides: Stride information for matrices A and B across groups
//   c_strides: Stride information for output matrix C across groups
//   problem_sizes: Matrix dimensions (M, N, K) for each group
//   expert_offsets: Offset indices for expert selection in grouped format
//   sf_offsets: Scale factor offsets for each group
//   out_dtype: Output data type (Half or BFloat16)
//
// Returns: Grouped matrix C = alpha * (A @ B) for all groups in the specified
// output dtype
NVF_API torch::Tensor nvfp4_scaled_grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    const at::ScalarType out_dtype);

NVF_API void mxfp8_scaled_grouped_mm(
    torch::Tensor& output,
    torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs,
    torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs,
    torch::Tensor& b_scales_ptrs,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& stride_a,
    const torch::Tensor& stride_b,
    const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa,
    const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& workspace);

// Performs grouped matrix multiplication.
//
// This function implements multiple matrix multiplications in a grouped format
// where each group represents a separate GEMM operation with its own
// parameters. The function is optimized for scenarios like mixture-of-experts
// models where multiple independent matrix multiplications need to be performed
// efficiently.
//
// Parameters:
//   a: Input matrix A in BF16 or FP16 format (M x K)
//   b: Input matrix B in BF16 or FP16 format (G x N, K)
//   ab_strides: Stride information for matrices A and B across groups
//   c_strides: Stride information for output matrix C across groups
//   problem_sizes: Matrix dimensions (M, N, K) for each group
//   expert_offsets: Offset indices for expert selection in grouped format
//
// Returns: Grouped matrix C = A @ B for all groups in the specified
// output dtype
NVF_API torch::Tensor grouped_mm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& ab_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets);

} // namespace nvfuser::cutlass_kernels
