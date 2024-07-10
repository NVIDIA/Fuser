// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <id_model/id_model_index_compute.h>
#include <id_model/utils.h>
#include <swizzle.h>

namespace nvfuser {

void IdGraphIndexCompute::handle(Split* split) {
  const bool is_forward = isForward(split);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << split->toString();

  auto inner_extent = split->inner()->extent();

  if (is_forward) {
    auto in_idx = getIndex(split->in());
    auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, inner_extent);
    Val* inner_idx = nullptr;
    if (isInMaxPath(split->in())) {
      inner_idx = SimplifyingIrBuilder::subExpr(
          inner_extent, in_idx->fusion()->oneVal());
    } else {
      inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
    }
    setIndex(split->outer(), outer_idx);
    setIndex(split->inner(), inner_idx);
  } else {
    auto outer_idx = getIndex(split->outer());
    auto inner_idx = getIndex(split->inner());
    auto in_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_extent), inner_idx);
    setIndex(split->in(), in_idx);
  }
}

void IdGraphIndexCompute::handle(Merge* merge) {
  const bool is_forward = isForward(merge);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << merge->toString();

  // TODO: use getMaybeExpandedExtent?
  auto inner_ext = merge->inner()->extent();

  if (is_forward) {
    auto outer_idx = getIndex(merge->outer());
    auto inner_idx = getIndex(merge->inner());
    auto out_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(inner_ext, outer_idx), inner_idx);
    setIndex(merge->out(), out_idx);
  } else {
    auto out_idx = getIndex(merge->out());
    auto outer_idx = SimplifyingIrBuilder::divExpr(out_idx, inner_ext);
    setIndex(merge->outer(), outer_idx);
    Val* inner_idx = nullptr;
    // TODO: This is a safe but conservative workaround. See the old
    // IndexCompute for optimization
    // Leave it for now. Revisit after contig indexing. See if how
    // it's impacting. Maybe just fine to look at if all subsequent
    // depedent splits are divisible. That shoud be the case of the
    // transpose.
    if (isInMaxPath(merge->out())) {
      VERBOSE() << "Taking max path: " << merge->toString();
      inner_idx = SimplifyingIrBuilder::subExpr(
          inner_ext, inner_ext->fusion()->oneVal());
    } else {
      inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
    }
    setIndex(merge->inner(), inner_idx);
  }
}

void IdGraphIndexCompute::handle(Swizzle* swizzle) {
  const bool is_forward = isForward(swizzle);

  auto x_ext = swizzle->inX()->extent();
  auto y_ext = swizzle->inY()->extent();

  if (is_forward) {
    auto x_idx = getIndex(swizzle->inX());
    auto y_idx = getIndex(swizzle->inY());
    auto [result_x, result_y] =
        dispatchUnSwizzle(swizzle->swizzleType(), x_idx, y_idx, x_ext, y_ext);
    setIndex(swizzle->outX(), result_x);
    setIndex(swizzle->outY(), result_y);
  } else {
    auto x_idx = getIndex(swizzle->outX());
    auto y_idx = getIndex(swizzle->outY());
    auto [result_x, result_y] =
        dispatchSwizzle(swizzle->swizzleType(), x_idx, y_idx, x_ext, y_ext);
    setIndex(swizzle->inX(), result_x);
    setIndex(swizzle->inY(), result_y);
  }
}

void IdGraphIndexCompute::handle(Resize* resize) {
  const bool is_forward = isForward(resize);

  VERBOSE() << "IdGraphIndexCompute handle (" << (is_forward ? "fwd" : "bwd")
            << "): " << resize->toString();

  auto left_expand = resize->leftExpand();

  auto in_id = is_forward ? resize->in() : resize->out();
  auto out_id = is_forward ? resize->out() : resize->in();

  if (left_expand->isZeroInt()) {
    // Just forward as is
    setIndex(out_id, getIndex(in_id));
    return;
  }

  auto in_idx = getIndex(in_id);
  Val* out_idx = nullptr;

  if (is_forward) {
    out_idx = SimplifyingIrBuilder::addExpr(in_idx, left_expand);
  } else {
    out_idx = SimplifyingIrBuilder::subExpr(in_idx, left_expand);
  }

  setIndex(out_id, out_idx);
}

} // namespace nvfuser
