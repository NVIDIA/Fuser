// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir/utils.h>
#include <optimization/consecutive_cast_pass.h>

namespace nvfuser::optimization {

namespace {

void castOptimizationPass(Fusion* fusion) {
  auto is_cast_op = [](Expr* expr) {
    if (expr->isA<UnaryOp>()) {
      auto op = expr->as<UnaryOp>();
      if (op->getUnaryOpType() == UnaryOpType::Cast) {
        return true;
      }
    }
    return false;
  };

  // NOTE: not the most efficient pass
  std::unordered_map<Val*, Val*> replacement_map;
  for (auto expr : fusion->exprs()) {
    if (is_cast_op(expr)) {
      bool mutated = false;
      while (true) {
        // in the loop, we just repetitively skip consecutive casts.
        auto intermediate_cast = expr->input(0);
        auto prev_expr = intermediate_cast->definition();
        if (prev_expr != nullptr && is_cast_op(prev_expr)) {
          auto original_dtype = prev_expr->input(0)->getDataType().value();
          auto intermediate_dtype = intermediate_cast->getDataType().value();
          auto out_dtype = expr->output(0)->getDataType().value();
          // cases where skipping the intermediate cast is relatively safe,
          // two conditions:
          //   1. original_dtype is the same as out_dtype; or
          //   2. we support direct cast from original_dtype to out_dtype.
          // and
          //   1. intermediate_dtype is the same type category as with
          //   out_dtype; or
          //   2. intermediate_dtype is more relaxed than out_dtype. e.g. a
          //   floating point vs. integral;
          if ((original_dtype == out_dtype ||
               cast_func_str({original_dtype, out_dtype}).has_value()) &&
              ((isIntegralType(intermediate_dtype) &&
                isIntegralType(out_dtype)) ||
               (isFloatingPointType(intermediate_dtype) &&
                isFloatingPointType(out_dtype)) ||
               (isComplexType(intermediate_dtype) &&
                isComplexType(out_dtype)) ||
               (isFloatingPointType(intermediate_dtype) &&
                isIntegralType(out_dtype)))) {
            expr = nvfuser::ir_utils::replaceValInExpr(
                expr, intermediate_cast, prev_expr->input(0));
            mutated = true;
          } else {
            break;
          }
        } else {
          break;
        }
      }

      if (mutated) {
        // quick short-wire to skip current cast node if it's trivially
        // casting to the same type
        if (expr->input(0)->getDataType().value() ==
            expr->output(0)->getDataType().value()) {
          replacement_map[expr->output(0)] = expr->input(0);
          // NOTE: if current output is a fusion output, DCE won't kick in and
          // we'll ended up with an illegal cast.
          if (expr->output(0)->isFusionOutput()) {
            fusion->replaceOutput(expr->output(0), expr->input(0));
          }
        }
      }
    }
  }
  if (!replacement_map.empty()) {
    nvfuser::ir_utils::replaceValue(fusion, replacement_map);
  }
}

} // namespace

void ConsecutiveCastPass::run(Fusion* fusion) {
  castOptimizationPass(fusion);
}

std::string ConsecutiveCastPass::name() {
  return "ConsecutiveCastOptimization";
}

} // namespace nvfuser::optimization
