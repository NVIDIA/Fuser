// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <optimization/move_split_cat.h>

#include <vector>

#include <expr_simplifier.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ir/internal_base_nodes.h>
#include <ir/utils.h>

namespace nvfuser::optimization {

namespace {

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
TensorView* findCancelingSplit(CatOp* cat) {
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
  NVF_CHECK(cat_axis != -1, "`cat` has zero inputs: ", cat);

  // We then locate the `SliceOp`s right before these `PadOp`s.
  std::vector<SliceOp*> slices;
  slices.reserve(pads.size());
  for (PadOp* pad : pads) {
    auto* slice = dynamic_cast<SliceOp*>(pad->in()->definition());
    if (slice == nullptr) {
      return nullptr;
    }
    slices.push_back(slice);
  }

  // Check the slices are based on the same tensor. Otherwise, they don't form a
  // split.
  TensorView* split_in = nullptr;
  for (SliceOp* slice : slices) {
    if (split_in == nullptr) {
      split_in = slice->in();
    } else if (split_in != slice->in()) {
      return nullptr;
    }
  }
  NVF_CHECK(split_in != nullptr, "`cat` has zero inputs: ", cat);

  for (auto i : c10::irange(pads.size())) {
    // For each branch, check the sliced amount is the same as the padded
    // amount. Otherwise, the slices don't form a split.
    SliceOp* slice = slices[i];
    PadOp* pad = pads[i];

    std::vector<int> padded_axes = pad->getPaddedAxes();
    if (padded_axes.size() != 1) {
      return nullptr;
    }
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
      if (resize->out() != slice_out->getRFactorDomain()[cat_axis]) {
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
  const auto& exprs = fusion->exprs();
  for (CatOp* cat : ir_utils::filterByType<CatOp>(exprs)) {
    TensorView* split_in = findCancelingSplit(cat);
    if (split_in == nullptr) {
      continue;
    }
    ir_utils::replaceValInAllExprInputsAndFusionOutputs(
        cat->output(0), split_in);
  }
}

} // namespace nvfuser::optimization
