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

// These are the quantized types that make use of block scaling for which we
// have established Fusion IR recipes.
bool isBlockScaledDtype(const DataType& dtype) {
  return dtype == DataType::Float4_e2m1fn || dtype == DataType::Float8_e4m3fn ||
      dtype == DataType::Float8_e5m2;
}

} // namespace

namespace cutlass_codegen {

std::string BlockScaledOutputPattern::toString() const {
  std::stringstream ss;
  ss << "BlockScaledOutputPattern: {\n";
  ss << "  unquantized_output: " << unquantized_output->toString() << "\n";
  ss << "  output: " << quantized_output->toString() << "\n";
  ss << "  block_scale_factors: " << block_scale_factors->toString() << "\n";
  ss << "  block_size: " << block_size << "\n";
  ss << "}\n";
  return ss.str();
}

// This matches a standard block scaling pattern expressed in Fusion IR.
//
// Block scaling quantizes a tensor by dividing it by per-block scale factors
// that are computed from the absolute maximum values within each block.
//
// Pattern diagram (without global scale):
//
//          unquantized_output (high precision)
//                   |
//       Reshape (split by block_size)
//                   |
//           data_hp_reshaped
//           /            \
//          /              \
//         /                \.
//        /                Abs
//       |                   |
//       |             Max (reduction)
//       |                   |
//       |            Div (by constant)
//       |                   |
//       |             [Optional: Clamp]
//       |                   |
//       |             Cast (to FP8)
//       |                   |
//       |           block_scale_fp8 --> OUTPUT (FP8 scales)
//       |                   |
//       |             Cast (to FP32)
//       |                   |
//       |           Broadcast/Unsqueeze
//       |                   |
//       +--------+----------+
//                |
//               Div
//                |
//           data_scaled
//                |
//         [Optional: Clamp]
//                |
//        Cast (to low precision)
//                |
//         [Optional: Reshape]
//                |
//         quantized_output --> OUTPUT (quantized)
//
// The pattern with global scale accepts a fusion input that divides the data
// before the first abs/max
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

    // We will fill out this pattern as we go
    BlockScaledOutputPattern pattern;

    // Check if this output might use block scaling
    if (!isBlockScaledDtype(out_tv->dtype())) {
      continue;
    }
    pattern.quantized_output = out_tv;

    // The quantized output might have a reshape after the cast
    // Trace back through any reshapes first
    TensorView* quantized_data = out_tv;
    if (auto* reshape_op =
            dynamic_cast<ReshapeOp*>(quantized_data->definition())) {
      quantized_data = reshape_op->in();
    }

    // The quantized output should be produced by a cast operation
    auto* cast_to_low_precision =
        dynamic_cast<UnaryOp*>(quantized_data->definition());
    if (cast_to_low_precision == nullptr ||
        cast_to_low_precision->getUnaryOpType() != UnaryOpType::Cast) {
      continue;
    }

    // The input to the cast is the scaled output (not yet quantized)
    TensorView* scaled_output = cast_to_low_precision->in()->as<TensorView>();

    // Look backwards to find the division operation that scales the data. Note
    // that the scaled output might have been clamped before casting, so we
    // unwrap the clamp if needed.
    if (auto* clamp_op = dynamic_cast<TernaryOp*>(scaled_output->definition());
        clamp_op != nullptr &&
        clamp_op->getTernaryOpType() == TernaryOpType::Clamp) {
      scaled_output = clamp_op->in1()->as<TensorView>();
    }

    // The scaled_output should be the result of a division
    auto* data_div_op = dynamic_cast<BinaryOp*>(scaled_output->definition());
    if (data_div_op == nullptr ||
        data_div_op->getBinaryOpType() != BinaryOpType::Div ||
        !data_div_op->lhs()->isA<TensorView>() ||
        !data_div_op->rhs()->isA<TensorView>()) {
      continue;
    }

    // The LHS of the division is the data, which should come from a reshape
    TensorView* data_reshaped = data_div_op->lhs()->as<TensorView>();

    auto* reshape = dynamic_cast<ReshapeOp*>(data_reshaped->definition());
    if (reshape == nullptr) {
      continue;
    }
    pattern.unquantized_output = reshape->in();

    // The RHS of the division is the broadcasted block scales
    TensorView* block_scales_broadcasted = data_div_op->rhs()->as<TensorView>();

    // Trace back through broadcast operation
    TensorView* block_scales_fp32 = block_scales_broadcasted;
    if (auto* broadcast_op =
            dynamic_cast<BroadcastOp*>(block_scales_fp32->definition())) {
      block_scales_fp32 = broadcast_op->in()->as<TensorView>();
    }

    // Now we need to find the corresponding block scale output
    // It could be either:
    // 1. Direct cast to FP8 (without global scale)
    // 2. Result of global_scale * block_scales_fp32 (with global scale)

    // Helper to unwrap broadcasts to find the original tensor
    auto unwrap_broadcast = [](TensorView* tv) -> TensorView* {
      if (auto* bcast_op = dynamic_cast<BroadcastOp*>(tv->definition())) {
        tv = bcast_op->in()->as<TensorView>();
      }
      return tv;
    };

    // Check if block_scales_fp32 is the result of a multiplication (pattern 2)
    if (auto* mul_op =
            dynamic_cast<BinaryOp*>(block_scales_fp32->definition())) {
      if (mul_op->getBinaryOpType() == BinaryOpType::Mul) {
        // One operand should be a 0-dim tensor (global scale),
        // the other should be the block scales from FP8
        // Note: the 0-dim tensor will be broadcast to the full dimension
        TensorView* lhs = mul_op->lhs()->as<TensorView>();
        TensorView* rhs = mul_op->rhs()->as<TensorView>();

        // Unwrap broadcasts to find the original tensors
        TensorView* lhs_unwrapped = unwrap_broadcast(lhs);
        TensorView* rhs_unwrapped = unwrap_broadcast(rhs);

        // The global scale factor should be a 0-dim tensor
        // It can be either a fusion input or computed (e.g., from amax)
        if (lhs_unwrapped->nDims() == 0) {
          pattern.global_scale_factor = lhs_unwrapped;
          block_scales_fp32 = rhs;
        } else if (rhs_unwrapped->nDims() == 0) {
          pattern.global_scale_factor = rhs_unwrapped;
          block_scales_fp32 = lhs;
        }
      }
    }

    // block_scales_fp32 should be a cast from the quantized block scales
    auto* cast_from_quantized =
        dynamic_cast<UnaryOp*>(block_scales_fp32->definition());
    if (cast_from_quantized == nullptr ||
        cast_from_quantized->getUnaryOpType() != UnaryOpType::Cast) {
      continue;
    }
    TensorView* block_scales_quantized =
        cast_from_quantized->in()->as<TensorView>();

    // Verify this is a quantized tensor (typically e4m3fn for scale
    // factors)
    if (!block_scales_quantized->isFusionOutput() ||
        !(block_scales_quantized->dtype() == DataType::Float8_e4m3fn ||
          block_scales_quantized->dtype() == DataType::Float8_e5m2 ||
          block_scales_quantized->dtype() == DataType::Float8_e8m0fnu)) {
      continue;
    }
    pattern.block_scale_factors = block_scales_quantized;

    // If we found the block scale factors, verify the full pattern
    if (pattern.block_scale_factors != nullptr) {
      // Now verify that block_scale_factors was computed from data_reshaped
      // Expected pattern:
      //   abs(data_reshaped) -> max reduction -> div by constant ->
      //   clamp -> cast to FP8

      // block_scale_factors is the quantized output, trace back to find what
      // was cast TO it
      auto* cast_to_quantized =
          dynamic_cast<UnaryOp*>(pattern.block_scale_factors->definition());
      if (cast_to_quantized == nullptr ||
          cast_to_quantized->getUnaryOpType() != UnaryOpType::Cast) {
        continue;
      }
      TensorView* block_scales_unquantized =
          cast_to_quantized->in()->as<TensorView>();

      // Skip optional clamp before quantizing the scales
      TensorView* block_scales_unclamped = block_scales_unquantized;
      if (auto* clamp_op =
              dynamic_cast<TernaryOp*>(block_scales_unclamped->definition());
          clamp_op != nullptr &&
          clamp_op->getTernaryOpType() == TernaryOpType::Clamp) {
        block_scales_unclamped = clamp_op->in1()->as<TensorView>();
      }

      // If there's a global scale, there's an extra division:
      //   block_scales_raw / global_scale -> block_scales_scaled
      // Otherwise it's just: amax / constant -> block_scales_raw
      TensorView* block_scales_raw = block_scales_unclamped;
      if (pattern.global_scale_factor != nullptr) {
        // Expect: block_scales_scaled = block_scales_raw / global_scale
        auto* global_div_op =
            dynamic_cast<BinaryOp*>(block_scales_raw->definition());
        if (global_div_op == nullptr ||
            global_div_op->getBinaryOpType() != BinaryOpType::Div ||
            !global_div_op->lhs()->isA<TensorView>()) {
          continue;
        }
        // Verify the RHS is related to the global_scale_factor
        // (might be wrapped in broadcasts)
        TensorView* divisor = global_div_op->rhs()->as<TensorView>();
        TensorView* divisor_unwrapped = unwrap_broadcast(divisor);
        if (divisor_unwrapped != pattern.global_scale_factor) {
          continue;
        }
        block_scales_raw = global_div_op->lhs()->as<TensorView>();
      }

      // block_scales_raw should be: amax / constant
      auto* amax_div_op =
          dynamic_cast<BinaryOp*>(block_scales_raw->definition());
      if (amax_div_op == nullptr ||
          amax_div_op->getBinaryOpType() != BinaryOpType::Div ||
          !amax_div_op->lhs()->isA<TensorView>()) {
        continue;
      }

      TensorView* data_hp_amax = amax_div_op->lhs()->as<TensorView>();

      // data_hp_amax should be a max reduction
      auto* max_reduction =
          dynamic_cast<ReductionOp*>(data_hp_amax->definition());
      if (max_reduction == nullptr ||
          max_reduction->getReductionOpType() != BinaryOpType::Max) {
        continue;
      }

      // The input to the max reduction should be abs(data_reshaped)
      TensorView* data_hp_abs = max_reduction->in()->as<TensorView>();
      auto* abs_op = dynamic_cast<UnaryOp*>(data_hp_abs->definition());
      if (abs_op == nullptr || abs_op->getUnaryOpType() != UnaryOpType::Abs) {
        continue;
      }

      // The input to abs should be the same data_reshaped we're dividing
      TensorView* data_hp_reshaped = abs_op->in()->as<TensorView>();
      if (data_hp_reshaped != data_reshaped) {
        continue;
      }

      // Now we've verified the full pattern! Infer block size
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

      pattern.block_size = block_size;

      patterns.push_back(pattern);
    }
  }

  return patterns;
}

} // namespace cutlass_codegen

} // namespace nvfuser
