// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <kernel.h>
#include <kernel_ir_dispatch.h>

#include <unordered_map>
#include <vector>

namespace nvfuser {

//!
class DependencyMapper : public kir::IrVisitor {
 public:
  DependencyMapper(const std::vector<Expr*>& top_level_exprs);

  using Coords = std::vector<int64_t>;

  //! This describes the position of a particular expression in the kernel
  struct ExprPosition {
    //! This gives the tree coordinates of
    Coords coords;

    //! This position is the order in which we would see the expressions if they
    //! were converted to a CUDA kernel as-is. Note that a position is given to
    //! the end of each scope.
    int64_t pos = -1;

    Scope* scope = nullptr;
  };

  //! Given coordinates expressed in terms of _non-trivial_ for loops only,
  //! return the non-null expr at a given location. Note that this method is
  //! slow as it must reconstruc
  Expr* exprFromCoord(const Coords& coords) const;

  //! Given coordinates expressed in terms of _non-trivial_ for loops only,
  //! return the non-null expr at a given location. Note that this method is
  //! slow as it must reconstruc
  Scope& scopeFromCoord(const Coords& coords) const;

  const ExprPosition& getExprPosition(Expr* expr) const {
    auto pos_int_it = expr_pos_int_.find(expr);
    NVF_ERROR(pos_int_it != expr_pos_int_.end());
    size_t pos_int = pos_int_it->second;
    const auto& pos_ptr = expr_position_up_.at(pos_int);
    NVF_ERROR(pos_ptr != nullptr);
    return *pos_ptr;
  }

  ExprPosition& getExprPosition(Expr* expr) {
    size_t pos_int;
    auto pos_int_it = expr_pos_int_.find(expr);
    if (pos_int_it == expr_pos_int_.end()) {
      pos_int = exprs_.size();
      expr_pos_int_[expr] = pos_int;
      exprs_.push_back(expr);
      expr_position_up_.emplace_back(std::make_unique<ExprPosition>());
    } else {
      pos_int = pos_int_it->second;
    }
    const auto& pos_ptr = expr_position_up_.at(pos_int);
    NVF_ERROR(pos_ptr != nullptr);
    return *pos_ptr;
  }

  const std::vector<Expr*>& trackedExprs() const {
    return exprs_;
  }

  //! This struct is used to record all accesses to a particular tensor
  struct TensorAccesses {
    //! All expressions that write to this tensor, in program order
    std::vector<ExprPosition*> writes;
    //! All expressions that read from this tensor, in program order
    std::vector<ExprPosition*> reads;
  };

  const TensorAccesses& getTensorAccesses(TensorView* tv) const {
    auto pos_int_it = tv_pos_int_.find(tv);
    NVF_ERROR(pos_int_it != tv_pos_int_.end());
    size_t pos_int = pos_int_it->second;
    const auto& access_ptr = tv_access_up_.at(pos_int);
    NVF_ERROR(access_ptr != nullptr);
    return *access_ptr;
  }

  TensorAccesses& getTensorAccesses(TensorView* tv) {
    size_t pos_int;
    const auto pos_int_it = tv_pos_int_.find(tv);
    if (pos_int_it == tv_pos_int_.end()) {
      pos_int = tvs_.size();
      tv_pos_int_[tv] = pos_int;
      tvs_.push_back(tv);
      tv_access_up_.emplace_back(std::make_unique<TensorAccesses>());
    } else {
      pos_int = pos_int_it->second;
    }
    const auto& access_ptr = tv_access_up_.at(pos_int);
    NVF_ERROR(access_ptr != nullptr);
    return *access_ptr;
  }

  const std::vector<TensorView*>& trackedTensors() const {
    return tvs_;
  }

 private:
  using IrVisitor::dispatch;

  void dispatch(Expr* expr) override;

 private:
  const std::vector<Expr*> top_level_exprs_;
  std::vector<Expr*> exprs_;
  std::vector<std::unique_ptr<ExprPosition>> expr_position_up_;
  std::unordered_map<Expr*, size_t> expr_pos_int_;

  std::vector<TensorView*> tvs_;
  std::vector<std::unique_ptr<TensorAccesses>> tv_access_up_;
  std::unordered_map<TensorView*, size_t> tv_pos_int_;

  //! This holds either a single Expr or a collection of these nodes. In cases
  //! where we hold a collection of nodes, it corresponds to a collection of
  //! nested scopes and we record the a reference to the ForLoop with the
  //! smallest enclosing scope as expr. In case of an ITE we record whether this
  //! is the then or else branch.
  //!
  //! for iS0
  //!   for iS1
  //!     expr0
  //!     expr1
  //!   endfor
  //!   expr2
  //!   for iS2
  //!     for iS3
  //!       expr3
  //!     endfor
  //!   endfor
  //!   expr3
  //! endfor
  //!
  // TODO: finish examples
  class NonTrivialExprOrScope {
   public:
    explicit NonTrivialExprOrScope(Expr* expr, bool is_else_branch = false)
        : expr_(expr), is_else_branch_(is_else_branch) {}

    Expr* expr() const {
      return expr_;
    }
    bool isElseBranch() const {
      return is_else_branch_;
    }

   private:
    Expr* expr_;
    bool is_else_branch_;
  };

  class NonTrivialExprTree {
   public:
    NonTrivialExprTree() {
      base_node_ = insertNode(nullptr);
    }

    struct Node {
      Expr* expr;
      bool is_else_branch = false;
      std::vector<std::shared_ptr<NonTrivialExprOrScope>> members;
    };

    Node* insertNode(Expr* expr, bool is_else_branch = false) {
      nodes_up_.emplace_back(std::make_unique<Node>(expr, is_else_branch));
      return nodes_up_.back().get();
    }

    Node* getBaseNode() const { return base_node_; }

   private:
    std::vector<std::unique_ptr<Node>> nodes_up_;
    Node* base_node_;
  } nontrivial_expr_tree_;

  std::vector<NonTrivialExprTree::Node*> nontrivial_expr_stack_;

  int64_t current_pos_;
  Coords current_coords_;
};

//! Compute the nearest common ancestor between two DependencyMapper::Coords
//! The nearest common ancestor is the innermost scope that is common between two Coords
//! For example, the nearest common ancestor between {2, 3, 1, 5} and {2, 3, 3} is {2, 3}
inline DependencyMapper::Coords nearestCommonAncestor(
    const DependencyMapper::Coords& coord1,
    const DependencyMapper::Coords& coord2) {
  DependencyMapper::Coords result;
  
  for (auto [val1, val2] : views::zip(coord1, coord2)) {
    if (val1 == val2) {
      result.push_back(val1);
    } else {
      break;
    }
  }
  
  return result;
}

//! Compute the center point between two DependencyMapper::Coords using their nearest common ancestor
//! For example, for {2, 3, 1, 5} and {2, 3, 3}, we return {2, 3, 2}
DependencyMapper::Coords centerCoord(
    const DependencyMapper::Coords& coord1,
    const DependencyMapper::Coords& coord2);

} // namespace nvfuser
