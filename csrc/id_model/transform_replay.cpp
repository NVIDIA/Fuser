// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <id_model/transform_replay.h>

#include <ir/builder.h>

namespace nvfuser {

Expr* ReplayTransform::replayAs(
    const std::vector<IterDomain*>& ordered_inputs,
    const Expr* expression_to_match) {
  ReplayTransform replay(ordered_inputs, expression_to_match);
  return replay.replayed_expr_;
}

ReplayTransform::ReplayTransform(
    const std::vector<IterDomain*>& ordered_inputs,
    const Expr* expression_to_match)
    : input_ids_(ordered_inputs) {
  OptOutConstDispatch::dispatch(expression_to_match);
}

// We're going to replay this split operation on the corresponding ID
void ReplayTransform::handle(const Split* split) {
  NVF_ERROR(
      input_ids_.size() == 1,
      "Expected one input to match split: ",
      split->toString());
  replayed_expr_ =
      IterDomain::split(input_ids_[0], split->factor(), split->innerSplit())
          .first->definition();
}

// We're going to replay this merge operation on the corresponding IDs
void ReplayTransform::handle(const Merge* merge) {
  NVF_ERROR(
      input_ids_.size() == 2,
      "Expected two inputs to match merge: ",
      merge->toString());
  replayed_expr_ =
      IterDomain::merge(input_ids_[0], input_ids_[1])->definition();
}

// We're going to replay this swizzle operation on the corresponding IDs
//  if replaying swizzle is enabled.
void ReplayTransform::handle(const Swizzle2D* swizzle_2d) {
  NVF_ERROR(
      input_ids_.size() == 2,
      "Expected two inputs to match swizzle: ",
      swizzle_2d->toString());
  replayed_expr_ = IterDomain::swizzle(
                       swizzle_2d->swizzleType(),
                       input_ids_[0],
                       input_ids_[1],
                       swizzle_2d->swizzleMode())
                       .first->definition();
}

void ReplayTransform::handle(const Swizzle* swizzle) {
  NVF_ERROR(
      input_ids_.size() == 2,
      "Expected two inputs to match swizzle: ",
      swizzle->toString());
  replayed_expr_ =
      IterDomain::swizzle(swizzle->swizzleType(), input_ids_[0], input_ids_[1])
          .first->definition();
}

void ReplayTransform::handle(const Resize* resize) {
  NVF_ERROR(
      input_ids_.size() == 1,
      "Expected one input to match resize: ",
      resize->toString());
  replayed_expr_ =
      IterDomain::resize(
          input_ids_[0], resize->leftExpand(), resize->rightExpand())
          ->definition();
}

} // namespace nvfuser
