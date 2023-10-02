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
  replayed_expr_ = IterDomain::split(
                       input_ids_[0],
                       split->factor(),
                       split->innerSplit(),
                       split->startOffset(),
                       split->stopOffset())
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

Expr* ReplacementTransformCloner::clone(
    const std::unordered_map<IterDomain*, IterDomain*>&
        provided_expr_val_2_replacement_val,
    const Expr* expression_to_match) {
  ReplacementTransformCloner replay(
      provided_expr_val_2_replacement_val, expression_to_match);
  return replay.new_expr_;
}

ReplacementTransformCloner::ReplacementTransformCloner(
    const std::unordered_map<IterDomain*, IterDomain*>&
        provided_expr_val_2_replacement_val,
    const Expr* expression_to_match)
    : provided_expr_val_2_replacement_val_(
          provided_expr_val_2_replacement_val) {
  OptOutConstDispatch::dispatch(expression_to_match);
}

IterDomain* ReplacementTransformCloner::replaceOrClone(IterDomain* id) {
  if (provided_expr_val_2_replacement_val_.find(id) !=
      provided_expr_val_2_replacement_val_.end()) {
    return provided_expr_val_2_replacement_val_.at(id);
  }
  return id->cloneWithoutRFactor();
}

// We're going to replay this split operation on the corresponding ID
void ReplacementTransformCloner::handle(const Split* split) {
  // Replace or clone

  auto split_in = replaceOrClone(split->in());
  auto split_outer = replaceOrClone(split->outer());
  auto split_inner = replaceOrClone(split->inner());

  // TODO: Should we check inner/outer matches the factor if
  // innerSplit()/!innerSplit()?

  new_expr_ = IrBuilder::create<Split>(
      split_outer,
      split_inner,
      split_in,
      split->factor(),
      split->innerSplit(),
      split->startOffset(),
      split->stopOffset());
}

// We're going to replay this merge operation on the corresponding IDs
void ReplacementTransformCloner::handle(const Merge* merge) {
  // Replace or clone
  auto merge_outer = replaceOrClone(merge->outer());
  auto merge_inner = replaceOrClone(merge->inner());
  auto merge_out = replaceOrClone(merge->out());
  new_expr_ = IrBuilder::create<Merge>(merge_out, merge_outer, merge_inner);
}

// We're going to replay this swizzle operation on the corresponding IDs
//  if replaying swizzle is enabled.
void ReplacementTransformCloner::handle(const Swizzle2D* swizzle_2d) {
  // Replace or clone
  auto swizzle_inx = replaceOrClone(swizzle_2d->inX());
  auto swizzle_iny = replaceOrClone(swizzle_2d->inY());
  auto swizzle_outx = replaceOrClone(swizzle_2d->outX());
  auto swizzle_outy = replaceOrClone(swizzle_2d->outY());

  new_expr_ = IrBuilder::create<Swizzle2D>(
      swizzle_outx,
      swizzle_outy,
      swizzle_inx,
      swizzle_iny,
      swizzle_2d->swizzleType(),
      swizzle_2d->swizzleMode());
}

void ReplacementTransformCloner::handle(const Resize* resize) {
  auto resize_in = resize->in();
  resize_in = provided_expr_val_2_replacement_val_.find(resize_in) !=
          provided_expr_val_2_replacement_val_.end()
      ? provided_expr_val_2_replacement_val_.at(resize_in)
      : resize_in->cloneWithoutRFactor();

  auto resize_out = resize->out();
  resize_out = provided_expr_val_2_replacement_val_.find(resize_out) !=
          provided_expr_val_2_replacement_val_.end()
      ? provided_expr_val_2_replacement_val_.at(resize_out)
      : resize_out->cloneWithoutRFactor();

  new_expr_ = IrBuilder::create<Resize>(
      resize_out, resize_in, resize->leftExpand(), resize->rightExpand());
}
} // namespace nvfuser
