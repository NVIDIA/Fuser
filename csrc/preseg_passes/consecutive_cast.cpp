// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/consecutive_cast.h>

#include <ir/utils.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <transform_iter.h>
#include <transform_replay.h>
#include <type.h>

namespace nvfuser::preseg_passes {

namespace {

bool isCast(Expr* expr) {
  if (auto op = dynamic_cast<UnaryOp*>(expr)) {
    return op->getUnaryOpType() == UnaryOpType::Cast;
  }
  return false;
}

// for pattern `meta -> cast`, this function returns whether to replace it with
// `cast -> meta`
bool shouldSwapMetaCast(Expr* cast) {
  if (!cast->output(0)->isA<TensorView>()) {
    return false;
  }
  // If cast is promoting dtype size, stop pushing cast along inputs to avoid
  // increase in intermediate buffer size.
  // NOTE since we don't have dtype size for index at compile time, we'll skip
  // cast on index types.
  if (cast->input(0)->getDataType().value() == DataType::Index ||
      cast->output(0)->getDataType().value() == DataType::Index ||
      (dataTypeSizeBit(cast->input(0)->getDataType().value()) <
       dataTypeSizeBit(cast->output(0)->getDataType().value()))) {
    return false;
  }

  Expr* meta = cast->input(0)->definition();
  // don't move meta operation when the its output is a fusion output, or has
  // multiple uses. In which case, we will have to duplicate the meta operation,
  // which might not be optimal. See
  // PresegTest.FusionTestCastOptimizationMetaOp2
  // TODO BroadcastOp is a moveable meta operation. We should enable it after
  // matmul scheduler is updated to support the new pattern with matmul. See
  // issue: https://github.com/NVIDIA/Fuser/issues/3665.
  return (meta != nullptr && !meta->output(0)->isFusionOutput() &&
          meta->output(0)->uses().size() == 1) &&
      (meta->isOneOf<SqueezeOp, ReshapeOp>() || ir_utils::isSimpleTVSet(meta));
}

// replays meta operation on `new_in`. return the new output from replayed meta
// operation
// TODO merge this into replayExprWithNewInput. There are two missing features
// in replayExprWithNewInput:
//   1. It expects new input to be of the same DataType as the old one, we
//   should be able to update that;
//   2. It doesn't support replay allocation domain transformations from the old
//   outputs to the new outputs.
Val* replayMetaOnNewInput(
    Expr* meta,
    Val* new_in,
    const std::vector<int64_t>& allocation_permutation) {
  // preparing new meta output.
  Val* replayed_meta_out = nullptr;
  ops::newValLike(meta->output(0), new_in->getDataType().value());

  if (meta->isA<SqueezeOp>()) {
    replayed_meta_out =
        ops::newValLike(meta->output(0), new_in->getDataType().value());
    IrBuilder::create<SqueezeOp>(
        replayed_meta_out, new_in, meta->as<SqueezeOp>()->getSqueezeDimFlags());
  } else if (meta->isA<BroadcastOp>()) {
    replayed_meta_out =
        ops::newValLike(meta->output(0), new_in->getDataType().value());
    IrBuilder::create<BroadcastOp>(
        replayed_meta_out,
        new_in,
        meta->as<BroadcastOp>()->getBroadcastDimFlags());
  } else if (meta->isA<ReshapeOp>()) {
    // replay transformation for ReshapeOp
    NVF_ERROR(meta->output(0)->isA<TensorView>());
    TensorView* meta_tv_out = meta->output(0)->as<TensorView>();

    const std::vector<IterDomain*>& meta_tv_out_root_domain =
        meta_tv_out->getMaybeRootDomain();

    // clone root domain from the original meta output for replay
    std::vector<IterDomain*> replayed_root_domain =
        IterDomain::clone(meta_tv_out_root_domain);

    // creating map from original to replayed ID
    std::unordered_map<IterDomain*, IterDomain*> id_map;
    for (const auto i : arange(meta_tv_out_root_domain.size())) {
      id_map[meta_tv_out_root_domain[i]] = replayed_root_domain[i];
    }

    // replay from root to logical.
    ReplayTransformations replay(meta_tv_out->getLogicalDomain(), id_map);
    std::vector<IterDomain*> replayed_logical_domain;
    for (auto id : meta_tv_out->getLogicalDomain()) {
      NVF_ERROR(replay.getReplay().count(id), "logical domain replay failed");
      replayed_logical_domain.push_back(replay.getReplay().at(id));
    }

    // preserving alloc_dom permutation on logical domain
    std::vector<IterDomain*> replayed_allocation_domain;
    if (allocation_permutation.empty()) {
      replayed_allocation_domain = replayed_logical_domain;
    } else {
      replayed_allocation_domain = ir_utils::applyPermutation(
          replayed_logical_domain, allocation_permutation);
    }

    // update the logical domain with replayed transformed.
    replayed_meta_out = IrBuilder::create<TensorView>(
        IrBuilder::create<TensorDomain>(
            replayed_root_domain,
            replayed_logical_domain,
            replayed_allocation_domain,
            replayed_logical_domain,
            meta_tv_out->getContiguity()),
        new_in->getDataType().value());

    // create the view op.
    IrBuilder::create<ReshapeOp>(replayed_meta_out, new_in);
  } else {
    NVF_ERROR(
        ir_utils::isSimpleTVSet(meta), "Unidentified operation for replay");
    replayed_meta_out =
        ops::newValLike(meta->output(0), new_in->getDataType().value());
    IrBuilder::create<LoadStoreOp>(
        LoadStoreOpType::Set, replayed_meta_out, new_in);
  }

  return replayed_meta_out;
}

// replaces input to the cast op that produes cast_output, return the new
// cast_output
Val* replaceInputInCast(Val* cast_output, Val* new_input) {
  auto tmp_expr = cast_output->definition();
  // short-cut for cases when no substitution is needed;
  if (cast_output == new_input || new_input == tmp_expr->input(0)) {
    return cast_output;
  }
  auto new_expr = nvfuser::ir_utils::replaceValInExprInputs(
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
//    and push the value(s) on top of `chain_cast_vals`, until:
//
//        a. the producer is not a cast op; or
//
//        b. the producer is used by other ops, or is a fusion output.
//
//    1.3 at this point, each `chain_cast_vals` has an ordered cast outputs with
//    a straight line dependency:
//        1.3.1 we point starting_anchor at the beginning op, indicating the
//        starting point of our folding optimization, meanwhile, we point
//        lo_anchor at the first op, indicating the narrowest dtype we have seen
//        in the segment;
//        1.3.2 we enter the loop to iterate through items
//        inside `chain_cast_vals`, for item `val`:
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
//        a. if `anchor_dtype` is no narrower than `output_dtype`, all previous
//        cast after `starting_anchor` is no-op, we re-wire `starting_anchor`
//        directly to `expr`;
//
//        b. otherwise, we can't bypass `lo_anchor` cast, we rewire this
//        section as `starting_anchor`->`lo_anchor`->`expr->output(0)`
Expr* removeChainedCasts(Expr* expr, std::unordered_set<Expr*>& folded) {
  std::list<Val*> chain_cast_vals;
  auto prev_expr = expr->input(0)->definition();
  while (isCast(prev_expr)) {
    auto intermediate_cast = prev_expr->output(0);
    // 1.2 Note, if the output of prev_expr
    //   is used by other operation(s); or
    //   is a direct output from fusion
    // we skip the casting chaining
    if (intermediate_cast->isFusionOutput() ||
        intermediate_cast->uses().size() > 1) {
      break;
    }

    // adding prev_expr to folded so we'll short-cut it.
    folded.insert(prev_expr);
    // in the loop, we just repetitively chaining consecutive casts.
    chain_cast_vals.push_front(intermediate_cast);
    prev_expr = prev_expr->input(0)->definition();
  }

  // skip current expr if there's no chain_cast_vals
  if (chain_cast_vals.empty()) {
    return expr;
  }

  // 1.3.1 Note, chain_cast_vals has a straight-line use without branches
  auto lo_anchor = chain_cast_vals.front()->definition()->input(0);
  auto anchor_dtype = lo_anchor->getDataType().value();
  auto starting_anchor = lo_anchor;
  for (auto val : chain_cast_vals) {
    auto val_dtype = val->getDataType().value();

    // 1.3.2.a short-cut when we are not losing precision
    if (isInclusiveType(anchor_dtype, val_dtype)) {
      continue;
    }

    // 1.3.2.c NOTE: To enter here, we have
    //   !isInclusiveType(anchor_dtype, val_dtype) &&
    //   !isInclusiveType(val_dtype, anchor_dtype)
    //
    // Which means the dtype between lo_anchor and val isn't compatible and
    // can't be fold away without losing information. So we update the
    // starting_anchor to current val, which ensures that we preserve the
    // incompatible casts. e.g. for cases where no one type is strictly wider
    // than the other: i.e. bf16 & fp16, int32 & float32 e.t.c.
    if (!isInclusiveType(val_dtype, anchor_dtype)) {
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

  if (isInclusiveType(output_dtype, anchor_dtype)) {
    // 1.4.a: if lo_anchor is no narrower than output_dtype, everything is an
    // no-op

    if (starting_anchor->getDataType().value() == output_dtype) {
      // if output dtype is identical to starting_anchor dtype, we can't keep
      // the last cast op and will need to re-write all uses here
      ir_utils::replaceValue(
          expr->fusion(), {{expr->output(0), starting_anchor}});
    } else {
      Val* new_expr_val = replaceInputInCast(expr->output(0), starting_anchor);
      expr = new_expr_val->definition();
    }
  } else {
    // 1.4.b: This is the case where we cannot fold away the cast of
    // lo_anchor; we'll just re-wire input to expr with lo_anchor
    lo_anchor = replaceInputInCast(lo_anchor, starting_anchor);
    Val* new_expr_val = replaceInputInCast(expr->output(0), lo_anchor);
    expr = new_expr_val->definition();
  }
  return expr;
}

void castOptimizationPass(Fusion* fusion) {
  FusionGuard fusion_guard(fusion);
  auto exprs = fusion->exprs();
  std::unordered_set<Expr*> folded;
  for (auto iter = exprs.rbegin(); iter != exprs.rend(); ++iter) {
    auto expr = *iter;
    // skip current expr if it's not a foldable cast or it has already been
    // removed in removeChainedCasts and is now a dangling pointer.
    if (folded.count(expr) != 0 || !isCast(expr)) {
      continue;
    }

    // initialize changed to true so we'll enter the loop in initial iteration.
    bool changed = true;
    while (changed) {
      changed = false;
      // when down cast follows a meta operation that's safe to be swapped, we
      // do so for two reasons:
      // 1. lifting a down cast to inputs would reduce intermediate buffer size
      // 2. it might place the cast op next to another cast op that can be
      // optimized away. e.g. for a trivial reduction on reduced precision, the
      // pattern will be
      //    T1 = castOp(T0, fp32)
      //    T2 = squeeze(T1)
      //    T3 = castOp(T2, fp16) // downCast
      // by swapping the last two op, we get
      //    T1 = castOp(T0, fp32)
      //    T2 = castOp(T1, fp16)
      //    T3 = squeeze(T2)      // operation in reduced precision
      // and we can further cancel out the two cast ops.
      if (shouldSwapMetaCast(expr)) {
        // replay [meta -> expr] with
        //        [replayed_expr -> replayed_meta]
        Val* expr_out = expr->output(0);

        // initializing alloc_domain permutation as empty
        std::optional<std::vector<int64_t>> expr_out_allocation_permutation =
            {};

        // compute logical_dom to alloc_dom permutation
        if (expr_out->isA<TensorView>()) {
          TensorView* expr_out_tv = expr_out->as<TensorView>();
          expr_out_allocation_permutation = ir_utils::computePermutation(
              expr_out_tv->getLogicalDomain(),
              expr_out_tv->getMaybeAllocationDomain());
        }

        // We do not support the replay if expr out has non-trivial transforms
        // between its logical_dom to alloc_dom.
        if (expr_out_allocation_permutation.has_value()) {
          Expr* meta = expr->input(0)->definition();

          // replayed expr(cast).
          Val* replayed_expr_out = castOp(expr_out->dtype(), meta->input(0));

          // replay meta operation on replayed expr output.
          Val* replayed_meta_out = replayMetaOnNewInput(
              meta, replayed_expr_out, expr_out_allocation_permutation.value());

          // replace uses of expr output with output of replayed_meta.
          ir_utils::replaceValInAllExprInputsAndFusionOutputs(
              expr_out, replayed_meta_out);

          // update expr to point to the replayed_expr
          expr = replayed_expr_out->definition();
        }
        changed = true;
      }

      // optimize chained cast operations ending at expr
      if (Expr* new_expr = removeChainedCasts(expr, folded); new_expr != expr) {
        expr = new_expr;
        changed = true;
      }
    }
  }
}

} // namespace

void ConsecutiveCastPass::runPass(Fusion* fusion) {
  castOptimizationPass(fusion);
}

} // namespace nvfuser::preseg_passes
