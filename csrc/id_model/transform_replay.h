// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>

#include <ir/all_nodes.h>

#include <unordered_map>
#include <vector>

namespace nvfuser {

class ReplayTransform : OptInConstDispatch {
 public:
  // Replays expression_to_match with the provided ordered_inputs. Inputs should
  // be ordered as they would be used in provided expression. Returns new
  // replayed expression.
  static Expr* replayAs(
      const std::vector<IterDomain*>& ordered_inputs,
      const Expr* expression_to_match);

 private:
  ReplayTransform(
      const std::vector<IterDomain*>& ordered_inputs,
      const Expr* expression_to_match);

  using OptInConstDispatch::handle;

  // We're going to replay this split operation on the corresponding ID
  void handle(const Split* split) final;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(const Merge* merge) final;

  // We're going to replay this swizzle operation on the corresponding IDs
  //  if replaying swizzle is enabled.
  void handle(const Swizzle2D* swizzle_2d) final;

  // We're going to replay this resize operation on the corresponding IDs
  //  if replaying resize is enabled.
  void handle(const Resize* resize) final;

  Expr* replayed_expr_ = nullptr;
  const std::vector<IterDomain*>& input_ids_;
};

class ReplacementTransformCloner : OptInConstDispatch {
 public:
  // Generates a copy of expression_to_match with inputs and/or outputs replaced
  // by entries provided in the map. Inputs and outputs are expected to be
  // "clones". Not literally, but it's up to the envoking code to make the
  // input/output replacements are safe to use in the cloned expression. No
  // validation is done on provided inputs/outputs.
  //
  // In other words a split i0{I0}->i1{I0//2}, i2{2} with a map:
  // i2{2} -> i3{48} wouldn't throw an error, but would not be valid.
  static Expr* clone(
      const std::unordered_map<IterDomain*, IterDomain*>&
          provided_expr_val_2_replacement_val,
      const Expr* expression_to_match);

 private:
  ReplacementTransformCloner(
      const std::unordered_map<IterDomain*, IterDomain*>&
          expr_to_match_2_replacement,
      const Expr* expression_to_match);

  using OptInConstDispatch::handle;

  // Returns entry in provided_expr_val_2_replacement_val_ if exists otherwise
  // returns a clone of the provided iter domain.
  IterDomain* replaceOrClone(IterDomain* id);

  // We're going to replay this split operation on the corresponding ID
  void handle(const Split* split) override;

  // We're going to replay this merge operation on the corresponding IDs
  void handle(const Merge* merge) override;

  // We're going to replay this swizzle operation on the corresponding IDs
  //  if replaying swizzle is enabled.
  void handle(const Swizzle2D* swizzle_2d) override;

  // We're going to replay this resize operation on the corresponding IDs
  //  if replaying resize is enabled.
  void handle(const Resize* resize) override;

  Expr* new_expr_ = nullptr;
  const std::unordered_map<IterDomain*, IterDomain*>&
      provided_expr_val_2_replacement_val_;
};

} // namespace nvfuser
