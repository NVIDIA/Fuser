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

  //! This describes the position of a particular expression in the kernel
  struct ExprPosition {
    //! This gives the tree coordinates of
    std::vector<int64_t> coords;

    //! This position is the order in which we would see the expressions if they
    //! were converted to a CUDA kernel as-is. Note that a position is given to
    //! the end of each scope.
    int64_t pos = -1;

    Scope* scope = nullptr;
  };

  Expr* exprFromCoord(const std::vector<int64_t>& coords) const;

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

  int64_t current_pos_;
  std::vector<int64_t> current_coords_;
};

} // namespace nvfuser
