// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <device_lower/dependencies.h>

#include <kernel_ir_dispatch.h>

namespace nvfuser {

DependencyMapper::DependencyMapper(const std::vector<Expr*>& top_level_exprs)
    : top_level_exprs_(top_level_exprs) {
  current_pos_ = 0;
  current_coords_ = {-1};
  nontrivial_expr_stack_ = {&nontrivial_exprs_};

  handle(top_level_exprs_);
}

Expr* DependencyMapper::exprFromCoord(const Coords& coords) const {
  NVF_ERROR(!coords.empty());
  const std::vector<Expr*>* scope_exprs = &top_level_exprs_;
  Expr* expr = nullptr;
  for (int64_t c : coords) {
    if (auto* ite = dynamic_cast<kir::IfThenElse*>(expr)) {
      if (c == 0) {
        const std::vector<Expr*>& exprs_ref = ite->thenBody().exprs();
        scope_exprs = &exprs_ref;
      } else if (c == 1) {
        const std::vector<Expr*>& exprs_ref = ite->elseBody().exprs();
        scope_exprs = &exprs_ref;
      } else {
        NVF_THROW("Encountered ITE with coord not equal to 0 or 1");
      }
      // Reset expr as this location points to somewhere within a branch
      expr = nullptr;
      continue;
    } else if (auto* fl = dynamic_cast<ForLoop*>(expr)) {
      const std::vector<Expr*>& exprs_ref = fl->body().exprs();
      scope_exprs = &exprs_ref;
      // Reset expr as this location points to somewhere within the loop scope
      expr = nullptr;
    }

    NVF_ERROR(expr == nullptr, "Expected expr to be null before assigning");

    expr = scope_exprs->at((size_t)c);
  }
  NVF_ERROR(
      expr != nullptr,
      "Coordinates ",
      coords,
      " do not point to an Expr, but might point to an ITE branch");
  return expr;
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
  auto& this_nontriv_expr = nontrivial_expr_stack_.back().members.emplace_back(
      std::make_shared<NonTrivialExprOrScope>(expr));

  auto recurse_to_scope = [&](const Scope& scope, bool then_branch = false) {
    nontrivial_expr_stack_.push_back(nontrivial_expr_stack_.back());
    current_coords_.push_back(-1);
    handle(scope.exprs());
    current_pos_++; // increment for close of scope
    exprs_.push_back(nullptr);
    expr_position_up_.emplace_back(std::make_unique<ExprPosition>());
    current_coords_.pop_back();
    nontrivial_expr_stack_.pop_back()
  };

  if (auto* ite = dynamic_cast<kir::IfThenElse*>(expr)) {
    nontrivial_expr_stack_.back().members.push_back(
        std::make_shared<NonTrivialExprOrScope>(
            expr, /*is_else_branch=*/false));
    recurse_to_scope(ite->thenBody(), /*else_branch=*/false);
    nontrivial_expr_stack_.back().members.push_back(
        std::make_shared<NonTrivialExprOrScope>(expr, /*is_else_branch=*/true));
    recurse_to_scope(ite->elseBody(), /*else_branch=*/true);

    return;
  } else if (auto* fl = dynamic_cast<ForLoop*>(expr)) {
    recurse_to_scope(fl->body());

    return;
  }

  std::cout << expr_pos.pos << " (" << expr_pos.coords
            << "): " << expr->toString() << std::endl;

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

} // namespace nvfuser
