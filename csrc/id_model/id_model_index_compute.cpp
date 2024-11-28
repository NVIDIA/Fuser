// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <id_model/id_model_index_compute.h>
#include <swizzle.h>

namespace nvfuser {

void IdGraphIndexCompute::handle(Split* split) {
  const bool is_forward = isForward(split);

  auto inner_extent = split->inner()->extent();

  if (is_forward) {
    // When propagating Split forward, if one of the outputs is mapped
    // with the input (because of the almost-exact mapping), don't
    // update the index and just set 0 as the index of the other
    // output. This is necessary when the other output is a broadcast
    // ID, which is ignored for predication. See
    // IndexingTest.AlmostExactIndexingUpdate for a concrete example.
    if (traversal_graph_.disjointValSets().strictAreMapped(
            split->in(), split->inner())) {
      setIndex(split->outer(), split->fusion()->zeroVal());
    } else if (traversal_graph_.disjointValSets().strictAreMapped(
                   split->in(), split->outer())) {
      setIndex(split->inner(), split->fusion()->zeroVal());
    } else {
      auto in_idx = getIndex(split->in());
      auto outer_idx = SimplifyingIrBuilder::divExpr(in_idx, inner_extent);
      Val* inner_idx = SimplifyingIrBuilder::modExpr(in_idx, inner_extent);
      setIndex(split->outer(), outer_idx);
      setIndex(split->inner(), inner_idx);
    }
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

  auto inner_ext = merge->inner()->extent();

  if (is_forward) {
    auto outer_idx = getIndex(merge->outer());
    auto inner_idx = getIndex(merge->inner());
    auto out_idx = SimplifyingIrBuilder::addExpr(
        SimplifyingIrBuilder::mulExpr(outer_idx, inner_ext), inner_idx);
    setIndex(merge->out(), out_idx);
  } else {
    // Similar to the forward propagation of Split, when propagating Merge
    // backward, if one of the inputs is mapped with the output, don't update
    // the index and just set 0 as the index of the other input.
    if (traversal_graph_.disjointValSets().strictAreMapped(
            merge->out(), merge->inner())) {
      setIndex(merge->outer(), merge->fusion()->zeroVal());
    } else if (traversal_graph_.disjointValSets().strictAreMapped(
                   merge->out(), merge->outer())) {
      setIndex(merge->inner(), merge->fusion()->zeroVal());
    } else {
      auto out_idx = getIndex(merge->out());
      auto outer_idx = SimplifyingIrBuilder::divExpr(out_idx, inner_ext);
      setIndex(merge->outer(), outer_idx);
      Val* inner_idx = SimplifyingIrBuilder::modExpr(out_idx, inner_ext);
      setIndex(merge->inner(), inner_idx);
    }
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
