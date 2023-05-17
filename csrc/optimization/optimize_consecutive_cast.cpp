// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/opt_pass.h>
#include <ir_utils.h>

namespace nvfuser::optimization {

namespace {

class ConsecutiveCastPass : OptimizationPass {
 public:
  static void runPass(Fusion* fusion) {
    auto is_cast_op = [] (Expr* expr) {
      if (expr->isA<UnaryOp>()) {
        auto op = expr->as<UnaryOp>();
        if (op->getUnaryOpType() == UnaryOpType::Cast) {
          return true;
        }
      }
      return false;
    };

    std::cout << "original fusion:" << std::endl;
    fusion->printMath();

    // NOTE: not the most efficient pass
    for (auto expr : fusion->exprs()) {
      if (is_cast_op(expr)) {
        while (true) {
          // in the loop, we just repetitively skip consecutive casts.
          auto intermediate_cast = expr->input(0);
          auto prev_expr = intermediate_cast->definition();
          if (prev_expr!=nullptr && is_cast_op(prev_expr)) {
            expr = nvfuser::ir_utils::replaceValInExpr(expr, intermediate_cast, prev_expr->input(0));
          } else {
            break;
          }
        }
      }
    }
    std::cout << "after mutation fusion:" << std::endl;
    fusion->printMath();
  }
  std::string name() override { return "ConsecutiveCastOptimization"; }
  FusionPass func() override { return runPass; }

  ConsecutiveCastPass() {
   registerOptimizationPass(OptimizationPassCategory::PreSegmenter, this);
  }
};

static ConsecutiveCastPass register_;

}

} // namespace nvfuser::optimization
