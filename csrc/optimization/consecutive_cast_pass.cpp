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

bool isSameDtypeCategory(const DataType& input_t, const DataType& output_t) {
  if ((isIntegralType(input_t) && isIntegralType(output_t)) ||
      (isFloatingPointType(input_t) && isFloatingPointType(output_t)) ||
      (isComplexType(input_t) && isComplexType(output_t))) {
    return true;
  }
  return false;
}

// check if type is a wider type than ref
// Which indicates a cast from ref -> type -> ref should be bit-wise identical
bool isWiderType(const DataType& ref, const DataType& type) {
  if (ref == type) {
    return true;
  } else if (type == DataType::Double && (ref == DataType::Float || ref == DataType::Half || ref == DataType::BFloat16)) {
    return true;
  } else if (type == DataType::Float && (ref == DataType::Half || ref == DataType::BFloat16)) {
    return true;
  } else if (type == DataType::Int && ref == DataType::Int32) {
    return true;
  } else if (type == DataType::ComplexDouble && ref == DataType::ComplexFloat) {
    return true;
  }
  return false;
}

// note: returns
//  - -1 : v0 contains strictly more information than v1;
//  - 0  : a complex case, where each v0 and v1 isn't a super set of the other;
//  - 1  : v0 and v1 has the same dtype;
//  - 2  : v0 contains strictly less information than v1;
int checkInformationLoss(Val* v0, Val* v1) {
  auto dtype0 = v0->getDataType().value();
  auto dtype1 = v1->getDataType().value();
  if (dtype0 == dtype1) {
    return 1;
  }
  if ((dtype0 == DataType::BFloat16 && dtype1 == DataType::Half) ||
      (dtype1 == DataType::BFloat16 && dtype0 == DataType::Half)) {
    return 0;
  } 
  if (isWiderType(dtype0, dtype1)) {
    return 2;
  }
  TORCH_INTERNAL_ASSERT(isWiderType(dtype1, dtype0), "unrecognized cast category is encountered");
  return -1;
}

void castOptimizationPass(Fusion* fusion) {
  auto is_foldable_cast_op = [](Expr* expr) {
    if (expr != nullptr && expr->isA<UnaryOp>()) {
      auto op = expr->as<UnaryOp>();
      if (op->getUnaryOpType() == UnaryOpType::Cast && 
	isSameDtypeCategory(expr->input(0)->getDataType().value(), expr->output(0)->getDataType().value())) {
        return true;
      }
    }
    return false;
  };

  // TODO: Traveral implies topological order on returns exprs, we can leverage that to improve the effieciency of the pass. In the case of a straight line casts, we are doing a lot of meaningless work here on mutating intermediate casts that would have been done again at the end of the chain.
  for (auto expr : fusion->exprs()) {
    if (is_foldable_cast_op(expr)) {
      std::vector<Val*> chain_casts;
      auto prev_expr = expr->input(0)->definition();
      while (prev_expr != nullptr && is_foldable_cast_op(prev_expr)) {
	auto intermediate_cast = prev_expr->output(0);
        // Note, if the output f prev_expr
        //   is used by other operation(s); or
        //   is a direct output from fusion
        // we skip the casting chaining
        if (intermediate_cast->isFusionOutput() ||
            intermediate_cast->uses().size() > 1) {
          break;
        }

        // in the loop, we just repetitively chaining consecutive casts.
	chain_casts.push_back(intermediate_cast);
        prev_expr = prev_expr->input(0)->definition();
      }

      // Note, chain_casts has a straight-line use without branches
      if (!chain_casts.empty()) {
        auto lo_anchor = chain_casts[0]->definition()->input(0);
	auto starting_anchor = lo_anchor;
	for (auto val : chain_casts) {
	  auto info = checkInformationLoss(lo_anchor, val);
	  // if information on new val drops below the anchor, we want to update the anchor
          if (info <= 0) {
	    // we run into a complex case where we are casting between two types that can't be folded away. i.e. bf16 & fp16. We need to update the starting_anchor for the final fold to be past this current cast.
            if (info == 0) {
	      auto tmp_expr = val->definition();
	      if (lo_anchor != tmp_expr->input(0)) {
                tmp_expr = nvfuser::ir_utils::replaceValInExpr(tmp_expr, tmp_expr->input(0), lo_anchor);
	      }
	      // move starting_anchor past the ambiguous case
	      starting_anchor = val;
	    }
	    // updating lo_anchor
	    lo_anchor = val;
	  }
	}

	auto info = checkInformationLoss(lo_anchor, expr->output(0));
	if (info == 1) {
          // replacing output with lo_anchor in the fusion
          ir_utils::replaceValue(fusion, {{expr->output(0), lo_anchor}});
          if (expr->output(0)->isFusionOutput()) {
            fusion->replaceOutput(expr->output(0), lo_anchor);
          }
	} else if (info == 2 || info == 0) {
          // expr output has either:
	  //   higher precision than lo_anchor; or
	  //   incompatible precision
          // in either case, we can't fold away lo_anchor, we'll just re-wire the input to expr to lo_anchor
          expr = nvfuser::ir_utils::replaceValInExpr(
              expr, expr->input(0), lo_anchor);
	} else if (info == -1) {
          // if expr has lower precision than lo_anchor, we'll just fold away to the starting_anchor instead
          expr = nvfuser::ir_utils::replaceValInExpr(
              expr, expr->input(0), starting_anchor);
        } else {
          TORCH_INTERNAL_ASSERT(false, "checkInformationLoss returns a flag that's not recognized");
	}
      }
    }
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
