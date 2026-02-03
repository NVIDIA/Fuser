// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "ir/container.h"

#include "fusion.h"
#include "instrumentation.h"
#include "ir/base_nodes.h"
#include "ir/builder.h"
#include "ir/cloner.h"
#include "ir/internal_nodes.h"

namespace nvfuser {

//! Return values in insertion order
const std::deque<Val*> IrContainer::deterministic_vals() const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Val*> vals_deque;
  std::transform(
      vals_up_.begin(),
      vals_up_.end(),
      std::back_inserter(vals_deque),
      [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
  return vals_deque;
}

//! Return expression in insertion order
const std::deque<Expr*> IrContainer::deterministic_exprs() const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Expr*> exprs_deque;
  std::transform(
      exprs_up_.begin(),
      exprs_up_.end(),
      std::back_inserter(exprs_deque),
      [](const std::unique_ptr<Expr>& expr_up) { return expr_up.get(); });
  return exprs_deque;
}

//! Return mapping from value to integer id
const std::unordered_map<Val*, int64_t> IrContainer::deterministic_vals_map()
    const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Val*, int64_t> vals_map;
  int64_t count = 0;
  std::transform(
      vals_up_.begin(),
      vals_up_.end(),
      std::inserter(vals_map, vals_map.end()),
      [&count](const std::unique_ptr<Val>& val_up) {
        return std::make_pair(val_up.get(), count++);
      });
  return vals_map;
}

//! Return mapping from expression to integer id
const std::unordered_map<Expr*, int64_t> IrContainer::deterministic_exprs_map()
    const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Expr*, int64_t> exprs_map;
  int64_t count = 0;
  std::transform(
      exprs_up_.begin(),
      exprs_up_.end(),
      std::inserter(exprs_map, exprs_map.end()),
      [&count](const std::unique_ptr<Expr>& expr_up) {
        return std::make_pair(expr_up.get(), count++);
      });
  return exprs_map;
}

const std::unordered_set<Expr*>& IrContainer::unordered_exprs() const noexcept {
  // Note: Returns reference - caller responsible for not holding across
  // concurrent modifications. Lock provides snapshot consistency during call.
  std::shared_lock lock(mutex_);
  return exprs_;
}

const std::unordered_set<Val*>& IrContainer::vals() const noexcept {
  // Note: Returns reference - caller responsible for not holding across
  // concurrent modifications. Lock provides snapshot consistency during call.
  std::shared_lock lock(mutex_);
  return vals_;
}

int64_t IrContainer::numExprs() const noexcept {
  std::shared_lock lock(mutex_);
  return std::ssize(exprs_);
}

int64_t IrContainer::numVals(bool include_shortcuts) const noexcept {
  std::shared_lock lock(mutex_);
  return include_shortcuts ? std::ssize(vals_) : std::ssize(vals_up_);
}

void IrContainer::swap(IrContainer& a, IrContainer& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  // Lock both containers in consistent order to avoid deadlock
  // Use std::lock to lock both mutexes atomically
  std::unique_lock lock_a(a.mutex_, std::defer_lock);
  std::unique_lock lock_b(b.mutex_, std::defer_lock);
  std::lock(lock_a, lock_b);

  // Swap the content
  std::swap(a.vals_up_, b.vals_up_);
  std::swap(a.vals_, b.vals_);

  std::swap(a.exprs_up_, b.exprs_up_);
  std::swap(a.exprs_, b.exprs_);

  std::swap(a.val_type_name_map_, b.val_type_name_map_);
  std::swap(a.expr_name_counter_, b.expr_name_counter_);

  std::swap(a.parent_, b.parent_);

  // Note: Special values, axioms, and metadata are now per-Fusion,
  // not per-IrContainer. They are handled by Fusion::swap.
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  // Lock both containers: shared for reading from, unique for writing to
  std::shared_lock lock_from(from->mutex_);
  std::unique_lock lock_to(to->mutex_);

  // Clear without calling clear() which would try to re-acquire the lock
  to->vals_.clear();
  to->vals_up_.clear();
  to->exprs_.clear();
  to->exprs_up_.clear();
  to->axioms_.reset();
  to->val_type_name_map_.clear();
  to->metadata_.clear();
  to->expr_name_counter_ = 0;

  IrCloner ir_cloner(to->parent());

  // Copy values in deterministic order
  // deterministic_vals can contain special values like one_val_, zero_val_, etc
  // that are not registered in the container.
  std::deque<Val*> from_vals;
  std::transform(
      from->vals_up_.begin(),
      from->vals_up_.end(),
      std::back_inserter(from_vals),
      [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
  for (auto val : from_vals) {
    if (from->vals_.count(val) > 0) {
      to->vals_.insert(ir_cloner.clone(val));
    }
  }

  // Copy expressions in deterministic order
  std::deque<Expr*> from_exprs;
  std::transform(
      from->exprs_up_.begin(),
      from->exprs_up_.end(),
      std::back_inserter(from_exprs),
      [](const std::unique_ptr<Expr>& expr_up) { return expr_up.get(); });
  for (auto expr : from_exprs) {
    if (from->exprs_.count(expr) > 0) {
      to->exprs_.insert(ir_cloner.clone(expr));
    }
  }

  to->val_type_name_map_ = from->val_type_name_map_;
  to->expr_name_counter_ = from->expr_name_counter_;

  // Note: axioms and metadata are now per-Fusion, handled by Fusion::copy

  return ir_cloner;
}

IrContainer::IrContainer() = default;

IrContainer::~IrContainer() {
  clear();
}

void IrContainer::removeExpr(Expr* expr) {
  NVF_ERROR(
      exprs_.find(expr) != exprs_.end(),
      "Wanted to remove an expression but it doesn't exist in this container.");
  auto expr_in_deque = std::find_if(
      exprs_up_.begin(),
      exprs_up_.end(),
      [expr](std::unique_ptr<Expr>& expr_up) { return expr_up.get() == expr; });

  NVF_ERROR(
      expr_in_deque != exprs_up_.end(),
      "Wanted to remove an expression but its unique ptr is missing.");

  exprs_.erase(expr);
  exprs_up_.erase(expr_in_deque);
}

//! Completely remove val from the fusion, break all dependencies associated
//! with it
void IrContainer::removeVal(Val* val) {
  // Note: Special values (zero_val_, one_val_, etc.) are now per-Fusion,
  // stored in Fusion class. They are registered as normal vals and can
  // be removed like any other val.

  NVF_ERROR(
      vals_.find(val) != vals_.end(),
      "Wanted to remove a value but it doesn't exist in this container.");
  auto val_in_deque = std::find_if(
      vals_up_.begin(), vals_up_.end(), [val](std::unique_ptr<Val>& val_up) {
        return val_up.get() == val;
      });

  NVF_ERROR(
      val_in_deque != vals_up_.end(),
      "Wanted to remove a value but its unique ptr is missing.");

  vals_.erase(val);
  vals_up_.erase(val_in_deque);
}

//! Register the Val with this container
void IrContainer::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }

  // Otherwise handle registration locally
  vals_up_.emplace_back(val);
  vals_.insert(val);
  val->setName(IrContainerPasskey(), getValName(val->vtype()));
}

//! Register expr with this container.
void IrContainer::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  // Otherwise handle registration locally
  exprs_up_.emplace_back(expr);
  exprs_.insert(expr);
  expr->setName(IrContainerPasskey(), getExprName());
}

void IrContainer::clear() noexcept {
  FUSER_PERF_SCOPE("IrContainer clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  val_type_name_map_.clear();
  expr_name_counter_ = 0;
}

bool IrContainer::inContainer(const Statement* const_stmt) const {
  // We don't use dynamic_cast here because `const_stmt` may be an invalid
  // pointer. Specifically a pointer to a Statement owned by another container
  // that has been freed.

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  void* raw_ptr = const_cast<void*>(reinterpret_cast<const void*>(const_stmt));
  if (exprs_.count(reinterpret_cast<Expr*>(raw_ptr)) == 0 &&
      vals_.count(reinterpret_cast<Val*>(raw_ptr)) == 0) {
    return false;
  }

  NVF_ERROR(
      const_stmt->container() == this->parent(),
      "Container claims to own stmt, but stmt disagrees.");

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* stmt = const_cast<Statement*>(const_stmt);
  if (stmt->isExpr()) {
    NVF_ERROR(
        exprs_.find(stmt->as<Expr>()) != exprs_.end(),
        "Somehow container claims to and not to own an Expr.");
  }
  if (stmt->isVal()) {
    NVF_ERROR(
        vals_.find(stmt->as<Val>()) != vals_.end(),
        "Somehow container claims to and not to own an Val.");
  }

  return true;
}

// Note: Shortcut values (zeroVal, oneVal, trueVal, falseVal, magicZeroVal),
// metadata, and axioms are now per-Fusion. Use Fusion::zeroVal(),
// Fusion::metadataOf(), Fusion::axioms(), etc. instead.
// This avoids ownership conflicts when multiple Fusions share an IrContainer.

void IrContainer::removeStatementsCreatedAfter(
    int64_t prev_num_exprs,
    int64_t prev_num_vals) {
  NVF_ERROR(
      exprs_up_.size() == exprs_.size(),
      "exprs_up_ (size ",
      exprs_up_.size(),
      ") and exprs_ (size ",
      exprs_.size(),
      ") are out of sync.");
  NVF_ERROR(
      std::ssize(exprs_up_) >= prev_num_exprs,
      "exprs_up_ size (",
      std::ssize(exprs_up_),
      ") is less than prev_num_exprs (",
      prev_num_exprs,
      ").");

  // Remove expressions before values because we need to change Val::uses_.
  while (std::ssize(exprs_up_) > prev_num_exprs) {
    Expr* e = exprs_up_.back().get();
    for (Val* in : e->inputs()) {
      in->removeUse(e);
    }
    exprs_.erase(e);
    exprs_up_.pop_back();
  }

  while (std::ssize(vals_up_) > prev_num_vals) {
    vals_.erase(vals_up_.back().get());
    vals_up_.pop_back();
  }
}

} // namespace nvfuser
