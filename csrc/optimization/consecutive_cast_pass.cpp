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

bool isCast(Expr* expr) {
  if (expr != nullptr && expr->isA<UnaryOp>()) {
    auto op = expr->as<UnaryOp>();
    return op->getUnaryOpType() == UnaryOpType::Cast;
  }
  return false;
}

// replaces input to the cast op that produes cast_output, return the new
// cast_output
Val* replaceInputInCast(Val* cast_output, Val* new_input) {
  auto tmp_expr = cast_output->definition();
  // short-cut for cases when no substitution is needed;
  if (cast_output == new_input || new_input == tmp_expr->input(0)) {
    return cast_output;
  }
  auto new_expr = nvfuser::ir_utils::replaceValInExpr(
      tmp_expr, tmp_expr->input(0), new_input);
  return new_expr->output(0);
}

// castOptimizationPass folds away consecutive cast operations. e.g. a chain of
// cast like fp16 -> fp32 -> fp16 can be simplified away without impacting the
// output of the fusion. For a section of consecutive casts, chained as:
//     first_tv -> cast_tv0 -> cast_tv1 -> cast_tv2 -> output_tv
// Supposed we have a TensorView `lo_anchor` in the chain, that the dtype of
// every other tv is either the same as or wider than
// `lo_anchor->getDataType()`, we can then assume that all the other cast ops in
// the chain is a no-op (except the last one, which defines the output dtype).
// so the above chain can be re-wired with only two casts as:
//     first_tv -> lo_anchor -> output_tv
//
// A complexity that could happen is, we might not necessarily have a narrowest
// dtype in the chain. i.e. think about pairs like fp16/bfloat16, or fp32/int32,
// where one can't represent the other. In order to handle this scenario, we can
// just keep track of a `starting_anchor` that indicates the starting point of
// the section, that has a valid `lo_anchor`. When we encounter a new cast op
// that breaks the assumption, we'll optimize what we have seen in the existing
// section and start a new section with the next cast op.
//
// The algorithm:
// 1. iterating through all expr in the fusion:
//    1.1 we skip all exprs other than cast;
//    1.2 for each end cast-op 'expr', we trace back its producers iteratively
//    and push the value(s) on top of `chain_cast_tvs`, until:
//
//        a. the producer is not a cast op; or
//
//        b. the producer is used by other ops, or is a fusion output.
//
//    1.3 at this point, each `chain_cast_tvs` has an ordered cast outputs with
//    a straight line dependency:
//        1.3.1 we point starting_anchor at the beginning op, indicating the
//        starting point of our folding optimization, meanwhile, we point
//        lo_anchor at the first op, indicating the narrowest dtype we have seen
//        in the segment;
//        1.3.2 we enter the loop to iterate through items
//        inside `chain_cast_tvs`, for item `val`:
//
//              a. if `val_dtype` is the same as, or wider than `anchor_dtype`
//              of `lo_anchor`, current cast is a no-op and can be ignored;
//
//              b. if `anchor_dtype` is narrower than `val_dtype`, previous cast
//              to `lo_anchor` is a no-op and can be folded away. We update
//              `lo_anchor` to point to `val`;
//
//              c. otherwise, `val` and `lo_anchor` are incompatible casts and
//              both needs to be preserved. We'll rewire it as:
//              `starting_anchor`->`lo_anchor`->`val`. Afterwards, we'll update
//              `starting_anchor` and `lo_anchor` to both point at `val`.
//
//    1.4 At this point we look at `anchor_dtype` of `lo_anchor` and
//    `output_dtype` of `expr->output(0)`:
//
//        a. if `anchor_dtype` is the same as `output_dtype`, we skip the last
//        cast op and replace all its uses with `lo_anchor`;
//
//        b. if `anchor_dtype` is wider than `output_dtype`, all previous cast
//        after `starting_anchor` is no-op, we re-wire `starting_anchor`
//        directly to `expr`;
//
//        c. otherwise, we can't bypass `lo_anchor` cast, we rewire this
//        section as `starting_anchor`->`lo_anchor`->`expr->output(0)`
void castOptimizationPass(Fusion* fusion) {
  // TODO: Traveral implies topological order on returned exprs, we can leverage
  // that to improve the effieciency of the pass. In the case of a straight line
  // casts, we are doing a lot of meaningless work here on mutating intermediate
  // casts that would have been done again at the end of the chain.
  // We should really use the reverse topological order and filters out exprs
  // that has been rendered as dead code during the pass.
  for (auto expr : fusion->exprs()) {
    // skip current expr if it's not a foldable cast
    if (!isCast(expr)) {
      continue;
    }
    std::list<Val*> chain_cast_tvs;
    auto prev_expr = expr->input(0)->definition();
    while (prev_expr != nullptr && isCast(prev_expr)) {
      auto intermediate_cast = prev_expr->output(0);
      // 1.2 Note, if the output of prev_expr
      //   is used by other operation(s); or
      //   is a direct output from fusion
      // we skip the casting chaining
      if (intermediate_cast->isFusionOutput() ||
          intermediate_cast->uses().size() > 1) {
        break;
      }

      // in the loop, we just repetitively chaining consecutive casts.
      chain_cast_tvs.push_front(intermediate_cast);
      prev_expr = prev_expr->input(0)->definition();
    }

    // skip current expr if there's no chain_cast_tvs
    if (chain_cast_tvs.empty()) {
      continue;
    }

    // 1.3.1 Note, chain_cast_tvs has a straight-line use without branches
    auto lo_anchor = chain_cast_tvs.front()->definition()->input(0);
    auto anchor_dtype = lo_anchor->getDataType().value();
    auto starting_anchor = lo_anchor;
    for (auto val : chain_cast_tvs) {
      auto val_dtype = val->getDataType().value();

      // 1.3.2.a short-cut when we are not losing precision, either:
      //   1. casting to the same type as the previously seen lowest precision;
      //   or
      //   2. casting to a wider type.
      if (val_dtype == anchor_dtype || isWiderType(anchor_dtype, val_dtype)) {
        continue;
      }

      // 1.3.2.c NOTE: To enter here, we have
      //   !isWiderType(anchor_dtype, val_dtype) && isWiderType(val_dtype,
      //   anchor_dtype)
      //
      // Which means the dtype between lo_anchor and val isn't compatible and
      // can't be fold away without losing information. So we update the
      // starting_anchor to current val, which ensures that we preserve the
      // incompatible casts. e.g. for cases where no one type is strictly wider
      // than the other: i.e. bf16 & fp16, int32 & float32 e.t.c.
      if (!isWiderType(val_dtype, anchor_dtype)) {
        lo_anchor = replaceInputInCast(lo_anchor, starting_anchor);
        val = replaceInputInCast(val, lo_anchor);
        // We need to update the starting_anchor for the fold to be past this
        // current cast.
        starting_anchor = val;
      }
      // 1.3.2.b/c updating new lo_anchor to current val
      lo_anchor = val;
      anchor_dtype = lo_anchor->getDataType().value();
    }

    auto output_dtype = expr->output(0)->getDataType().value();
    if (anchor_dtype == output_dtype) {
      // 1.4.a final cast is the same dtype as with previous lo_anchor,
      // replacing output with lo_anchor in the fusion
      ir_utils::replaceValue(fusion, {{expr->output(0), lo_anchor}});
      if (expr->output(0)->isFusionOutput()) {
        fusion->replaceOutput(expr->output(0), lo_anchor);
      }
    } else if (isWiderType(output_dtype, anchor_dtype)) {
      // 1.4.b: if lo_anchor is wider than output_dtype, casting to lo_anchor
      // isn't doing anything, we'll just fold away to the starting_anchor
      // instead
      replaceInputInCast(expr->output(0), starting_anchor);
    } else {
      // 1.4.c: This is the case where we cannot fold away the cast of
      // lo_anchor; we'll just re-wire input to expr with lo_anchor
      lo_anchor = replaceInputInCast(lo_anchor, starting_anchor);
      replaceInputInCast(expr->output(0), lo_anchor);
    }
  }
}

} // namespace

void ConsecutiveCastPass::runPass(Fusion* fusion) {
  castOptimizationPass(fusion);
}

} // namespace nvfuser::optimization
