// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/dependencies.h>

#include <kernel_ir_dispatch.h>
#include <algorithm>

namespace nvfuser {

DependencyMapper::DependencyMapper(const std::vector<Expr*>& top_level_exprs)
    : top_level_exprs_(top_level_exprs) {
  current_pos_ = 0;
  current_coords_ = {-1};
  nontrivial_expr_stack_ = {nontrivial_expr_tree_.getBaseNode()};

  handle(top_level_exprs_);
}

Expr* DependencyMapper::exprFromCoord(const Coords& coords) const {
  const NonTrivialExprTree::Node* node =
      nontrivial_expr_tree_.nodeFromCoords(coords);
  NVF_ERROR(
      node != nullptr,
      "Coordinates ",
      coords,
      " do not point to a NonTrivialExprTree::Node");
  NVF_ERROR(node->expr != nullptr, "Node does not point to an expression");
  return node->expr;
}

NonTrivialExprTree::Node* NonTrivialExprTree::nodeFromCoords(
    const Coords& coords) const {
  NonTrivialExprTree::Node* node = base_node_;
  for (int64_t c : coords) {
    NVF_ERROR(node != nullptr);
    node = node->members.at((size_t)c);
  }
  NVF_ERROR(node != nullptr);
  return node;
}

void DependencyMapper::dispatch(Expr* expr) {
  if (auto* fl = dynamic_cast<ForLoop*>(expr); fl && fl->isTrivial()) {
    // Flatten away non-trivial loops
    handle(fl->body().exprs());
    return;
  }
  current_pos_++;
  current_coords_.back()++;
  // Record expr position
  NVF_ERROR(
      expr_pos_int_.count(expr) == 0,
      "We do not expect to see expressions repeated in multiple places in a "
      "Kernel");
  ExprPosition& expr_pos = getExprPosition(expr);
  expr_pos.pos = current_pos_;
  expr_pos.coords = current_coords_;
  NonTrivialExprTree::Node* node = nontrivial_expr_tree_.insertNode(expr);
  nontrivial_expr_stack_.back()->members.push_back(node);

  auto recurse_to_scope = [&](const Scope& scope) {
    current_coords_.push_back(-1);
    handle(scope.exprs());
    current_pos_++; // increment for close of scope
    exprs_.push_back(nullptr);
    expr_position_up_.emplace_back(std::make_unique<ExprPosition>());
    current_coords_.pop_back();
  };

  if (auto* ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    NonTrivialExprTree::Node* if_node =
        nontrivial_expr_tree_.insertNode(expr, /*is_else_branch=*/false);
    nontrivial_expr_stack_.back()->members.push_back(if_node);
    nontrivial_expr_stack_.push_back(if_node);
    recurse_to_scope(ite->thenBody());
    nontrivial_expr_stack_.pop_back();

    NonTrivialExprTree::Node* else_node =
        nontrivial_expr_tree_.insertNode(expr, /*is_else_branch=*/true);
    nontrivial_expr_stack_.back()->members.push_back(else_node);
    nontrivial_expr_stack_.push_back(else_node);
    recurse_to_scope(ite->elseBody());
    nontrivial_expr_stack_.pop_back();

    return;
  } else if (auto* fl = dynamic_cast<ForLoop*>(expr)) {
    // We don't create another level of coords for ForLoops
    nontrivial_expr_stack_.push_back(node);
    recurse_to_scope(fl->body());
    nontrivial_expr_stack_.pop_back();

    return;
  }

  // Record reads
  for (Val* v : expr->inputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(v)) {
      TensorAccesses& accesses = getTensorAccesses(tv);
      accesses.reads.push_back(&expr_pos);
    }
  }

  // Record writes
  for (Val* v : expr->outputs()) {
    if (auto* tv = dynamic_cast<TensorView*>(v)) {
      TensorAccesses& accesses = getTensorAccesses(tv);
      accesses.writes.push_back(&expr_pos);
    }
  }
}

DependencyMapper::Coords centerCoord(
    const DependencyMapper::Coords& coord1,
    const DependencyMapper::Coords& coord2) {
  auto ancestor = nearestCommonAncestor(coord1, coord2);

  // Handle case where one coordinate is empty
  if (coord1.empty() && !coord2.empty()) {
    // Return {1} when coord1 is empty and coord2 is not
    return {1};
  } else if (!coord1.empty() && coord2.empty()) {
    // Return {1} when coord2 is empty and coord1 is not
    return {1};
  }

  // If one coord is a prefix of the other, extend the ancestor
  if (coord1.size() > ancestor.size() && coord2.size() > ancestor.size()) {
    // Both coords extend beyond the ancestor, compute midpoint
    int64_t mid_val = (coord1[ancestor.size()] + coord2[ancestor.size()]) / 2;
    ancestor.push_back(mid_val);
  } else if (coord1.size() > ancestor.size()) {
    // coord1 extends beyond ancestor, coord2 doesn't
    // Consider coord2 as having implicit -1 values, but ensure result is
    // non-negative
    int64_t coord2_implicit = -1; // implicit value for coord2
    int64_t mid_val = (coord1[ancestor.size()] + coord2_implicit) / 2;
    // Ensure the result is non-negative
    mid_val = std::max(0L, mid_val);
    ancestor.push_back(mid_val);
  } else if (coord2.size() > ancestor.size()) {
    // coord2 extends beyond ancestor, coord1 doesn't
    // Consider coord1 as having implicit -1 values, but ensure result is
    // non-negative
    int64_t coord1_implicit = -1; // implicit value for coord1
    int64_t mid_val = (coord1_implicit + coord2[ancestor.size()]) / 2;
    // Ensure the result is non-negative
    mid_val = std::max(0L, mid_val);
    ancestor.push_back(mid_val);
  }

  return ancestor;
}

} // namespace nvfuser
