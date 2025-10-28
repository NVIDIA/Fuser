// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <cutlass/block_scaling.h>
#include <exceptions.h>
#include <fusion.h>
#include <ir/interface_nodes.h>
#include <ir/internal_nodes.h>
#include <ir/utils.h>
#include <type.h>

namespace nvfuser {

namespace {

bool isBlockScaledDtype(const DataType& dtype) {
  return dtype == DataType::Float4_e2m1fn || dtype == DataType::Float8_e4m3fn ||
      dtype == DataType::Float8_e5m2;
}

} // namespace

namespace cutlass_codegen {

// This matches a standard block scaling pattern expressed in Fusion IR like
// follows:
//
// Pattern 1 (without global scale):
//   tv_data_scaled = div(tv_data_hp_reshaped, tv_block_scale_fp32_unsqueeze)
//   tv_data_scaled_clamp = clamp(tv_data_scaled, ...)
//   tv_data_lp = castOp(low_precision_dtype, tv_data_scaled_clamp)
//   outputs: tv_block_scale_fp8 (Float8_e4m3fn), tv_data_lp
//
// Pattern 2 (with global scale):
//   tv_total_scale = mul(tv_global_scale, tv_scaled_block_scales_fp32)
//   tv_total_scale_unsqueeze = unsqueeze(tv_total_scale, -1)
//   tv_data_scaled = div(tv_data_hp_reshaped, tv_total_scale_unsqueeze)
//   tv_data_scaled_clamp = clamp(tv_data_scaled, ...)
//   tv_data_lp = castOp(low_precision_dtype, tv_data_scaled_clamp)
//   outputs: tv_scaled_block_scales_fp8 (Float8_e4m3fn), tv_data_lp
//
// Supports low-precision types: Float4_e2m1fn, Float8_e4m3fn, Float8_e5m2
std::vector<BlockScaledOutputPattern> findBlockScaledOutputs(Fusion* fusion) {
  std::vector<BlockScaledOutputPattern> patterns;

  // Look for low-precision outputs (the quantized data)
  for (Val* out_val : fusion->outputs()) {
    auto* out_tv = dynamic_cast<TensorView*>(out_val);
    if (out_tv == nullptr) {
      continue;
    }

    // Check if this output might use block scaling
    if (!isBlockScaledDtype(out_tv->dtype())) {
      continue;
    }

    // The low-precision output might have a reshape after the cast
    // Trace back through any reshapes first
    TensorView* low_precision_tv = out_tv;
    while (auto* reshape_op =
               dynamic_cast<ReshapeOp*>(low_precision_tv->definition())) {
      low_precision_tv = reshape_op->in();
      // Verify it's still the same dtype after unwrapping reshape
      if (!isBlockScaledDtype(low_precision_tv->dtype())) {
        break;
      }
    }

    // The low-precision output should be produced by a cast operation
    auto* cast_op = dynamic_cast<UnaryOp*>(low_precision_tv->definition());
    if (cast_op == nullptr || cast_op->getUnaryOpType() != UnaryOpType::Cast) {
      continue;
    }

    // The input to the cast is the prescaled output (after clamping)
    TensorView* prescaled_output = cast_op->in()->as<TensorView>();

    // Look backwards to find the division operation that scales the data
    // The prescaled_output is typically the result of a clamp operation
    TensorView* data_scaled = prescaled_output;
    if (auto* clamp_op = dynamic_cast<TernaryOp*>(data_scaled->definition());
        clamp_op != nullptr &&
        clamp_op->getTernaryOpType() == TernaryOpType::Clamp) {
      data_scaled = clamp_op->in1()->as<TensorView>();
    }

    // The data_scaled should be the result of a division
    auto* div_op = dynamic_cast<BinaryOp*>(data_scaled->definition());
    if (div_op == nullptr || div_op->getBinaryOpType() != BinaryOpType::Div ||
        !div_op->lhs()->isA<TensorView>() ||
        !div_op->rhs()->isA<TensorView>()) {
      continue;
    }

    // The LHS of the division is the data, which should come from a reshape
    TensorView* data_reshaped = div_op->lhs()->as<TensorView>();

    // The RHS of the division should be the unsqueezed/broadcasted scale factor
    TensorView* scale_unsqueezed = div_op->rhs()->as<TensorView>();

    // Trace back through unsqueeze/broadcast operation
    // Note: unsqueeze() is implemented as broadcast() in nvFuser
    TensorView* total_scale = scale_unsqueezed;
    if (auto* broadcast_op =
            dynamic_cast<BroadcastOp*>(total_scale->definition())) {
      total_scale = broadcast_op->in()->as<TensorView>();
    }

    // Now we need to find the corresponding FP8 scale output
    // It could be either:
    // 1. Direct FP8 cast (without global scale)
    // 2. Result of global_scale * scaled_block_scales_fp32 (with global scale)

    TensorView* block_scale_factors = nullptr;
    TensorView* global_scale_factor = nullptr;

    // Helper to unwrap broadcasts to find the original tensor
    auto unwrap_broadcasts = [](TensorView* tv) -> TensorView* {
      while (auto* bcast_op = dynamic_cast<BroadcastOp*>(tv->definition())) {
        tv = bcast_op->in()->as<TensorView>();
      }
      return tv;
    };

    // Check if total_scale is the result of a multiplication (pattern 2)
    if (auto* mul_op = dynamic_cast<BinaryOp*>(total_scale->definition())) {
      if (mul_op->getBinaryOpType() == BinaryOpType::Mul) {
        // One operand should be a 0-dim tensor (global scale),
        // the other should be scaled_block_scales_fp32
        // Note: the 0-dim tensor might be wrapped in broadcasts
        TensorView* lhs = mul_op->lhs()->as<TensorView>();
        TensorView* rhs = mul_op->rhs()->as<TensorView>();

        // Unwrap broadcasts to find the original tensors
        TensorView* lhs_unwrapped = unwrap_broadcasts(lhs);
        TensorView* rhs_unwrapped = unwrap_broadcasts(rhs);

        // The global scale factor should be a 0-dim tensor
        // It can be either a fusion input or computed (e.g., from amax)
        if (lhs_unwrapped->nDims() == 0) {
          global_scale_factor = lhs_unwrapped;
          total_scale = rhs;
        } else if (rhs_unwrapped->nDims() == 0) {
          global_scale_factor = rhs_unwrapped;
          total_scale = lhs;
        }
      }
    }

    // Now total_scale should be the FP32 version of the FP8 block scales
    // Trace back through the cast from FP8 to FP32
    if (auto* fp32_cast_op =
            dynamic_cast<UnaryOp*>(total_scale->definition())) {
      if (fp32_cast_op->getUnaryOpType() == UnaryOpType::Cast) {
        TensorView* fp8_scale = fp32_cast_op->in()->as<TensorView>();

        // Verify this is an FP8 tensor (typically e4m3fn for scale factors)
        if (fp8_scale->dtype() == DataType::Float8_e4m3fn ||
            fp8_scale->dtype() == DataType::Float8_e5m2 ||
            fp8_scale->dtype() == DataType::Float8_e8m0fnu) {
          // Check if this FP8 scale is a fusion output
          if (std::find(
                  fusion->outputs().begin(),
                  fusion->outputs().end(),
                  fp8_scale) != fusion->outputs().end()) {
            block_scale_factors = fp8_scale;
          }
        }
      }
    }

    // If we found the block scale factors, we have a valid pattern
    if (block_scale_factors != nullptr) {
      // Infer block size from the reshape operation that created data_reshaped
      int64_t block_size = -1;

      // Trace back through the reshape to find the split
      if (data_reshaped->definition()->isA<ReshapeOp>()) {
        // Look for the innermost split in the rfactor domain
        // The reshape typically splits the last dimension by block_size
        IterDomain* id = data_reshaped->getLogicalDomain().back();
        auto* split_expr = dynamic_cast<Split*>(id->definition());
        if (split_expr == nullptr) {
          continue;
        }
        Val* factor = split_expr->factor();
        // If the split factor is a constant scalar, use it as block_size
        if (factor->isConstScalar()) {
          block_size = factor->value().as<int64_t>();
        }
      }

      NVF_ERROR(block_size > 1, "Could not infer block size");

      BlockScaledOutputPattern pattern;
      pattern.prescaled_output = prescaled_output;
      pattern.output = out_tv;
      pattern.block_scale_factors = block_scale_factors;
      pattern.global_scale_factor = global_scale_factor;
      pattern.block_size = block_size;

      patterns.push_back(pattern);
    }
  }

  return patterns;
}

} // namespace cutlass_codegen

} // namespace nvfuser
