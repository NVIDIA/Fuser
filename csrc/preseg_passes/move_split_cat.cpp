// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <preseg_passes/move_split_cat.h>

#include <vector>

#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>
#include <ops/alias.h>

namespace nvfuser::preseg_passes {

namespace {

// Returns true when Exprs in the frontier can be horizontally merged and
// applied on the unsplit tensor.
bool horizontallyMergeable(
    const std::vector<Expr*>& frontier,
    int64_t& split_axis) {
  NVF_ERROR(!frontier.empty());

  // Check all Exprs in `frontier`
  // 1. have the same op type and attributes,
  // 2. transform IDs in the same way, and
  // 3. don't resize the split axis.

  if (std::adjacent_find(
          frontier.begin(), frontier.end(), [](Expr* lhs, Expr* rhs) {
            return !lhs->sameOp(rhs);
          }) != frontier.end()) {
    return false;
  }

  if (auto* set = dynamic_cast<LoadStoreOp*>(frontier[0])) {
    if (set->opType() == LoadStoreOpType::Set) {
      auto* set_out = set->out()->as<TensorView>();
      std::optional<std::vector<int64_t>> permutation =
          ir_utils::computePermutation(
              set_out->getRootDomain(), set_out->getMaybeRFactorDomain());
      if (!permutation.has_value()) {
        return false;
      }

      for (size_t i = 1; i < frontier.size(); i++) {
        auto* other_set_out =
            frontier[i]->as<LoadStoreOp>()->out()->as<TensorView>();
        std::optional<std::vector<int64_t>> other_permutation =
            ir_utils::computePermutation(
                other_set_out->getRootDomain(),
                other_set_out->getMaybeRFactorDomain());
        if (!other_permutation.has_value()) {
          return false;
        }
        if (*permutation != *other_permutation) {
          return false;
        }
      }

      split_axis = (*permutation)[split_axis];
      return true;
    }
  }

  return false;
}

// Returns the inputs of `cat` and the dimension over which the inputs are
// concatenated.
std::pair<std::vector<PadOp*>, int64_t> getCatInputsAndAxis(CatOp* cat) {
  std::vector<PadOp*> pads;
  pads.reserve(cat->inputs().size());
  int64_t cat_axis = -1;

  for (Val* in : cat->inputs()) {
    // nvFuser implements cat as PadOp->CatOp.
    PadOp* pad = in->definition()->as<PadOp>();
    pads.push_back(pad);

    std::vector<int> padded_axes = pad->getPaddedAxes();
    NVF_ERROR(
        padded_axes.size() == 1,
        "One of `cat`'s consumers pads 0 or multiple dimensions: ",
        pad);
    if (cat_axis == -1) {
      cat_axis = padded_axes[0];
    } else {
      NVF_ERROR(
          cat_axis == padded_axes[0],
          "Pads before a Cat should have the same padded axis, but found ",
          cat_axis,
          " vs ",
          padded_axes[0]);
    }
  }
  NVF_ERROR(cat_axis != -1);
  return {pads, cat_axis};
}

// Finds the canceling split of `cat` and returns the input TensorView of the
// split. A split (implemented as multiple `slice`s) and a cat cancel when they
// work on the same dimension. For example, when
//
//   s0 = in[:, :5]
//   s1 = in[:, 5:]
//   out = cat([s0, s1], dim=-1)
//
// findCancelingSplit(out) returns `in`.
//
// `cat` doesn't have to immediately follow the split. For example, when
//
//   s0 = in[:, :5]
//   s1 = in[:, 5:]
//   t0 = permute(s0)
//   t1 = permute(s1)
//   out = cat([t0, t1], dim=0)
//
// In addition to returning `in`, findCancelingSplit(out) puts `t0`'s defining
// `permute` into `use_def_chain` so the caller can reconstruct `out` by
// replaying `use_def_chain` (in reverse order) on `in`.
TensorView* findCancelingSplit(CatOp* cat, std::vector<Expr*>& use_def_chain) {
  NVF_CHECK(!cat->inputs().empty(), "`cat` has zero inputs: ", cat);

  auto [pads, cat_axis] = getCatInputsAndAxis(cat);

  // `frontier` initially contains the preceding Exprs of `pads`. Then, we
  // repeatedly try to move the frontier up in lockstep as long as Exprs in the
  // frontier can be horizontally merged and applied on the unsplit tensor.
  std::vector<Expr*> frontier;
  frontier.reserve(pads.size());
  for (PadOp* pad : pads) {
    frontier.push_back(pad->in()->definition());
  }

  // Exit the loop when any Expr in `frontier` is a slice or a null.
  int64_t split_axis = cat_axis;
  while (std::none_of(frontier.begin(), frontier.end(), [](Expr* e) {
    return e == nullptr || e->isA<SliceOp>();
  })) {
    if (!horizontallyMergeable(frontier, std::ref(split_axis))) {
      return nullptr;
    }
    use_def_chain.push_back(frontier[0]);

    // Advance the frontier in lockstep.
    for (Expr*& e : frontier) {
      NVF_ERROR(
          e->inputs().size() == 1,
          "All mergeable Exprs should be unary at this moment, but found: ",
          e);
      e = e->input(0)->definition();
    }
  }

  // Check that `frontier` has only slices, and that all slices are based on the
  // same tensor. Otherwise, they don't form a split.
  TensorView* split_in = nullptr;
  for (Expr* e : frontier) {
    auto* slice = dynamic_cast<SliceOp*>(e);
    if (slice == nullptr) {
      return nullptr;
    }

    if (split_in == nullptr) {
      split_in = slice->in();
    } else if (split_in != slice->in()) {
      return nullptr;
    }
  }
  NVF_ERROR(split_in != nullptr);

  for (auto i : c10::irange(pads.size())) {
    // For each branch, check the sliced amount is the same as the padded
    // amount. Otherwise, the slices don't form a split.
    auto* slice = frontier[i]->as<SliceOp>();
    PadOp* pad = pads[i];

    auto [left_padding, right_padding] =
        pad->getPadWidths(static_cast<int>(cat_axis));

    for (Slice slice_range : slice->getRanges()) {
      if (!slice_range.step->isOne()) {
        return nullptr;
      }
    }

    // Get the left and right expand of the slice, which are zero or negative.
    Val* left_expand = nullptr;
    Val* right_expand = nullptr;
    auto* slice_out = slice->out()->as<TensorView>();
    std::vector<Expr*> transforms = StmtSort::getExprsBetween(
        {slice_out->getRootDomain().begin(), slice_out->getRootDomain().end()},
        {slice_out->getRFactorDomain().begin(),
         slice_out->getRFactorDomain().end()});
    for (auto* transform : transforms) {
      auto* resize = dynamic_cast<Resize*>(transform);
      if (resize == nullptr) {
        return nullptr;
      }
      if (resize->out() != slice_out->getRFactorDomain()[split_axis]) {
        return nullptr;
      }
      left_expand = resize->leftExpand();
      right_expand = resize->rightExpand();
    }
    if (left_expand == nullptr || right_expand == nullptr) {
      return nullptr;
    }

    if (!simplifyExpr(IrBuilder::addExpr(left_padding, left_expand))
             ->isZeroInt()) {
      return nullptr;
    }
    if (!simplifyExpr(IrBuilder::addExpr(right_padding, right_expand))
             ->isZeroInt()) {
      return nullptr;
    }
  }

  return split_in;
}

} // namespace

void MoveSplitCatPass::runPass(Fusion* fusion) {
  std::vector<Expr*> exprs = fusion->exprs();
  for (auto* cat : ir_utils::filterByType<CatOp>(exprs)) {
    std::vector<Expr*> use_def_chain;
    TensorView* split_in = findCancelingSplit(cat, std::ref(use_def_chain));
    if (split_in == nullptr) {
      continue;
    }

    TensorView* merged_out = split_in;
    for (auto i = use_def_chain.rbegin(), end = use_def_chain.rend(); i != end;
         i++) {
      Expr* to_replay = *i;
      // TODO(wujingyue): instead of an op-type dispatch, try a more general
      // approach suggested by @jacobhinkle:
      // https://github.com/NVIDIA/Fuser/pull/1782#discussion_r1496123087.
      if (to_replay->isA<LoadStoreOp>()) {
        auto* set_out = to_replay->output(0)->as<TensorView>();
        std::vector<int64_t> permutation = *ir_utils::computePermutation(
            set_out->getRootDomain(), set_out->getMaybeRFactorDomain());
        merged_out = permute(merged_out, permutation);
        continue;
      }
      NVF_ERROR("Not implemented");
    }

    ir_utils::replaceValInAllExprInputsAndFusionOutputs(
        cat->output(0), merged_out);
  }
}

} // namespace nvfuser::preseg_passes
