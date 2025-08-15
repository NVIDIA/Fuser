// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exceptions.h>
#include <nvf_cutlass.h>

namespace nvfuser::cutlass_kernels {

std::tuple<int64_t, int64_t, int64_t> validateInputsNvfp4ScaledMm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Tensor& alpha,
    bool skip_checks) {
  // Validate matrix dimensions
  NVF_CHECK(a.dim() == 2, "Operand A must be a matrix.");
  NVF_CHECK(b.dim() == 2, "Operand B must be a matrix.");
  NVF_CHECK(
      a.sizes()[1] == b.sizes()[1],
      "A and B shapes cannot be multiplied (",
      a.sizes()[0],
      ",",
      a.sizes()[1],
      " and ",
      b.sizes()[0],
      ",",
      b.sizes()[1],
      ")");

  const int64_t m = a.sizes()[0];
  const int64_t n = b.sizes()[0];
  const int64_t k = a.sizes()[1] * 2;

  std::tuple<int64_t, int64_t, int64_t> ret = {m, n, k};

  if (skip_checks) {
    return ret;
  }

  // Check CUDA device and contiguity for all input tensors
  for (const torch::Tensor& t : {a, b, scales_a, scales_b, alpha}) {
    NVF_CHECK(
        t.is_cuda() && t.is_contiguous(),
        "Input argument must be a CUDA tensor and contiguous.")
  }

  // Validate data types
  NVF_CHECK(
      a.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand A.")
  NVF_CHECK(
      b.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "Expected Float4_e2m1fn_x2 for Operand B.")
  NVF_CHECK(
      scales_a.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_a.")
  NVF_CHECK(
      scales_b.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "Expected FP8_E4M3 for Blockscale scale_b.")
  NVF_CHECK(
      alpha.scalar_type() == at::ScalarType::Float,
      "Expected FP32 for alpha scalar.")

  // Check alignment requirements
  constexpr int64_t alignment = 32;
  NVF_CHECK(
      k % alignment == 0,
      "The K dimension",
      k,
      "is not divisible by ",
      alignment)
  NVF_CHECK(
      n % alignment == 0,
      "The N dimension",
      n,
      "is not divisible by ",
      alignment)

  // Calculate rounded dimensions for scale matrix validation
  int64_t rounded_m = roundUp(m, 128);
  int64_t rounded_n = roundUp(n, 128);
  int64_t rounded_k = roundUp(k / 16, 4);

  // Validate scale matrix properties
  NVF_CHECK(scales_a.dim() == 2, "Blockscale scale_a must be a matrix.");
  NVF_CHECK(scales_b.dim() == 2, "Blockscale scale_b must be a matrix.");
  NVF_CHECK(
      scales_a.sizes()[1] == scales_b.sizes()[1],
      "scale_a and scale_b shapes cannot be multiplied because the inner-most "
      "dimensions are not equal.")
  NVF_CHECK(
      scales_a.sizes()[0] == rounded_m && scales_a.sizes()[1] == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      ",",
      rounded_k,
      "), but got a shape (",
      scales_a.sizes()[0],
      ",",
      scales_a.sizes()[1],
      ")");
  NVF_CHECK(
      scales_b.sizes()[0] == rounded_n && scales_b.sizes()[1] == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      ",",
      rounded_k,
      "), but got a shape (",
      scales_b.sizes()[0],
      ",",
      scales_b.sizes()[1],
      ")");

  return ret;
}

} // namespace nvfuser::cutlass_kernels
