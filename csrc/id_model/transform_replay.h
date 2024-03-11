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

// TODO: Consider merging this class with the existing replay
// classes. The use cases are not exactly the same, so it isn't
// immediately clear if they could be trivially merge.
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

  void handle(const Swizzle* swizzle) final;

  // We're going to replay this resize operation on the corresponding IDs
  //  if replaying resize is enabled.
  void handle(const Resize* resize) final;

  Expr* replayed_expr_ = nullptr;
  const std::vector<IterDomain*>& input_ids_;
};

} // namespace nvfuser
