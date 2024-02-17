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

bool mergeable(const std::vector<Expr*>& frontier, int& split_axis) {
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
// Currently, the cat has to immediately follow the split. But we will soon
// handle patterns like `split ->
// a_chain_of_unary_ops_that_do_not_resize_the_split_dimension -> cat`.
TensorView* findCancelingSplit(CatOp* cat, std::vector<Expr*>& use_def_chain) {
  NVF_CHECK(!cat->inputs().empty(), "`cat` has zero inputs: ", cat);

  // nvFuser implements cat as PadOp->CatOp. We first locate these `PadOp`s.
  std::vector<PadOp*> pads;
  pads.reserve(cat->inputs().size());
  for (Val* in : cat->inputs()) {
    pads.push_back(in->definition()->as<PadOp>());
  }

  int cat_axis = -1;
  for (PadOp* pad : pads) {
    std::vector<int> padded_axes = pad->getPaddedAxes();
    if (padded_axes.size() != 1) {
      return nullptr;
    }
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

  std::vector<Expr*> frontier;
  frontier.reserve(cat->inputs().size());
  for (PadOp* pad : pads) {
    frontier.push_back(pad->in()->definition());
  }

  // Break the loop when any Expr in `frontier` is a slice or a null.
  int split_axis = cat_axis;
  while (std::none_of(frontier.begin(), frontier.end(), [](Expr* e) {
    return e == nullptr || e->isA<SliceOp>();
  })) {
    if (!mergeable(frontier, std::ref(split_axis))) {
      return nullptr;
    }

    use_def_chain.push_back(frontier[0]);

    // Advance the frontier.
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

    auto [left_padding, right_padding] = pad->getPadWidths(cat_axis);

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
