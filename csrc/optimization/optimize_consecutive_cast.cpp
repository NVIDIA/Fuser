// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ir_utils.h>
#include <optimization/opt_pass.h>

namespace nvfuser::optimization {

namespace {

class ConsecutiveCastPass : OptimizationPass {
 public:
  static void runPass(Fusion* fusion) {
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
	    // cases where skipping the intermediate cast is relatively safe, two conditions:
	    //   1. original_dtype is the same as out_dtype; or
	    //   2. we support direct cast from original_dtype to out_dtype.
	    // and
	    //   1. intermediate_dtype is the same type category as with out_dtype; or
	    //   2. intermediate_dtype is a floating point while output is integral;
            if (cast_func_str({original_dtype, out_dtype}).has_value() &&
	        ((isIntegralType(intermediate_dtype) && isIntegralType(out_dtype)) ||
	        (isFloatingPointType(intermediate_dtype) && isFloatingPointType(out_dtype)) ||
	        (isComplexType(intermediate_dtype) && isComplexType(out_dtype)) ||
	        (isFloatingPointType(intermediate_dtype) && isIntegralType(out_dtype)))) {
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
	  // quick short-wire to skip current cast node if it's trivially casting to the same type
          if (expr->input(0)->getDataType().value() == expr->output(0)->getDataType().value()) {
            replacement_map[expr->output(0)] = expr->input(0);
	    // NOTE: if current output is a fusion output, DCE won't kick in and we'll ended up with an illegal cast.
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

  std::string name() override {
    return "ConsecutiveCastOptimization";
  }

  FusionPass func() override {
    return runPass;
  }

  ConsecutiveCastPass() {
    registerOptimizationPass(OptimizationPassCategory::PreSegmenter, this);
  }
};

static ConsecutiveCastPass register_;

} // namespace

} // namespace nvfuser::optimization
