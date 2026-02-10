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

// =========================================================================
// Per-Fusion Deterministic Accessors (Phase 2)
// =========================================================================

std::deque<Val*> IrContainer::deterministicValsOwnedBy(
    Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Val*> result;

  // Get the set of vals owned by this Fusion for O(1) lookup
  auto it = per_fusion_vals_.find(fusion);
  if (it == per_fusion_vals_.end()) {
    return result; // Empty - no vals owned by this Fusion
  }
  const auto& owned_vals = it->second;

  // Iterate in insertion order, filtering to only owned vals
  for (const auto& val_up : vals_up_) {
    Val* val = val_up.get();
    if (owned_vals.count(val) > 0) {
      result.push_back(val);
    }
  }
  return result;
}

std::deque<Expr*> IrContainer::deterministicExprsOwnedBy(
    Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Expr*> result;

  // Get the set of exprs owned by this Fusion for O(1) lookup
  auto it = per_fusion_exprs_.find(fusion);
  if (it == per_fusion_exprs_.end()) {
    return result; // Empty - no exprs owned by this Fusion
  }
  const auto& owned_exprs = it->second;

  // Iterate in insertion order, filtering to only owned exprs
  for (const auto& expr_up : exprs_up_) {
    Expr* expr = expr_up.get();
    if (owned_exprs.count(expr) > 0) {
      result.push_back(expr);
    }
  }
  return result;
}

std::unordered_map<Val*, int64_t> IrContainer::deterministicValsMapOwnedBy(
    Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Val*, int64_t> result;

  // Get the set of vals owned by this Fusion for O(1) lookup
  auto it = per_fusion_vals_.find(fusion);
  if (it == per_fusion_vals_.end()) {
    return result; // Empty - no vals owned by this Fusion
  }
  const auto& owned_vals = it->second;

  // Iterate in insertion order, assigning sequential ids to owned vals
  int64_t count = 0;
  for (const auto& val_up : vals_up_) {
    Val* val = val_up.get();
    if (owned_vals.count(val) > 0) {
      result[val] = count++;
    }
  }
  return result;
}

std::unordered_map<Expr*, int64_t> IrContainer::deterministicExprsMapOwnedBy(
    Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Expr*, int64_t> result;

  // Get the set of exprs owned by this Fusion for O(1) lookup
  auto it = per_fusion_exprs_.find(fusion);
  if (it == per_fusion_exprs_.end()) {
    return result; // Empty - no exprs owned by this Fusion
  }
  const auto& owned_exprs = it->second;

  // Iterate in insertion order, assigning sequential ids to owned exprs
  int64_t count = 0;
  for (const auto& expr_up : exprs_up_) {
    Expr* expr = expr_up.get();
    if (owned_exprs.count(expr) > 0) {
      result[expr] = count++;
    }
  }
  return result;
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

// =========================================================================
// Fusion tracking for shared container support (Phase 2)
// =========================================================================

void IrContainer::addFusion(Fusion* fusion) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.insert(fusion);
}

void IrContainer::removeFusion(Fusion* fusion) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.erase(fusion);
  removeStatementsOwnedByUnlocked(fusion);
}

void IrContainer::transferFusion(Fusion* from, Fusion* to) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.erase(from);
  sharing_fusions_.insert(to);
  // Note: Statements retain their container() pointer - they don't need
  // to be updated because container() returns Fusion* which points to
  // the owning Fusion, and that ownership is what we're transferring.
}

size_t IrContainer::sharingCount() const {
  std::shared_lock lock(mutex_);
  return sharing_fusions_.size();
}

bool IrContainer::hasMultipleFusions() const {
  std::shared_lock lock(mutex_);
  return sharing_fusions_.size() > 1;
}

const std::unordered_set<Fusion*>& IrContainer::sharingFusions() const {
  std::shared_lock lock(mutex_);
  return sharing_fusions_;
}

// =========================================================================
// Per-Fusion Statement Tracking (Phase 2 Task 4)
// =========================================================================

const std::unordered_set<Val*>& IrContainer::valsOwnedBy(Fusion* fusion) const {
  std::shared_lock lock(mutex_);
  static const std::unordered_set<Val*> empty;
  auto it = per_fusion_vals_.find(fusion);
  return it != per_fusion_vals_.end() ? it->second : empty;
}

const std::unordered_set<Expr*>& IrContainer::exprsOwnedBy(
    Fusion* fusion) const {
  std::shared_lock lock(mutex_);
  static const std::unordered_set<Expr*> empty;
  auto it = per_fusion_exprs_.find(fusion);
  return it != per_fusion_exprs_.end() ? it->second : empty;
}

void IrContainer::transferStatementOwnership(Fusion* from, Fusion* to) {
  std::unique_lock lock(mutex_);

  // Transfer vals ownership tracking
  auto vals_it = per_fusion_vals_.find(from);
  if (vals_it != per_fusion_vals_.end()) {
    // Move the set to 'to', merging if 'to' already has entries
    auto& to_vals = per_fusion_vals_[to];
    to_vals.insert(vals_it->second.begin(), vals_it->second.end());
    per_fusion_vals_.erase(vals_it);
  }

  // Transfer exprs ownership tracking
  auto exprs_it = per_fusion_exprs_.find(from);
  if (exprs_it != per_fusion_exprs_.end()) {
    // Move the set to 'to', merging if 'to' already has entries
    auto& to_exprs = per_fusion_exprs_[to];
    to_exprs.insert(exprs_it->second.begin(), exprs_it->second.end());
    per_fusion_exprs_.erase(exprs_it);
  }

  // Transfer per-Fusion name counters (Phase 2 Task 10)
  auto val_names_it = per_fusion_val_name_map_.find(from);
  if (val_names_it != per_fusion_val_name_map_.end()) {
    // Merge counter maps: take max of each ValType counter
    auto& to_map = per_fusion_val_name_map_[to];
    for (auto& [vtype, counter] : val_names_it->second) {
      to_map[vtype] = std::max(to_map[vtype], counter);
    }
    per_fusion_val_name_map_.erase(val_names_it);
  }

  auto expr_names_it = per_fusion_expr_name_counter_.find(from);
  if (expr_names_it != per_fusion_expr_name_counter_.end()) {
    auto& to_counter = per_fusion_expr_name_counter_[to];
    to_counter = std::max(to_counter, expr_names_it->second);
    per_fusion_expr_name_counter_.erase(expr_names_it);
  }
}

void IrContainer::removeStatementsOwnedBy(Fusion* fusion) {
  std::unique_lock lock(mutex_);
  removeStatementsOwnedByUnlocked(fusion);
}

void IrContainer::removeStatementsOwnedByUnlocked(Fusion* fusion) {
  // Remove all Vals owned by this Fusion
  for (auto it = vals_up_.begin(); it != vals_up_.end();) {
    Val* val = it->get();
    // Check if this Val's container points to the Fusion being removed
    if (val->container() == fusion) {
      vals_.erase(val);
      it = vals_up_.erase(it);
    } else {
      ++it;
    }
  }

  // Remove all Exprs owned by this Fusion
  for (auto it = exprs_up_.begin(); it != exprs_up_.end();) {
    Expr* expr = it->get();
    // Check if this Expr's container points to the Fusion being removed
    if (expr->container() == fusion) {
      exprs_.erase(expr);
      it = exprs_up_.erase(it);
    } else {
      ++it;
    }
  }

  // Clean up per-Fusion tracking (Phase 2 Task 4)
  per_fusion_vals_.erase(fusion);
  per_fusion_exprs_.erase(fusion);

  // Clean up per-Fusion name counters (Phase 2 Task 10)
  per_fusion_val_name_map_.erase(fusion);
  per_fusion_expr_name_counter_.erase(fusion);
}

void IrContainer::swap(IrContainer& a, IrContainer& b) noexcept {
  FUSER_PERF_SCOPE("IrContainer swap");

  // NOTE: This method is deprecated in Phase 2. Fusion::swap handles
  // pointer-based swapping of shared containers. This is kept for
  // backward compatibility but should not be called directly.

  // Lock both containers in consistent order to avoid deadlock
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
  std::swap(a.sharing_fusions_, b.sharing_fusions_);
  std::swap(a.per_fusion_vals_, b.per_fusion_vals_);
  std::swap(a.per_fusion_exprs_, b.per_fusion_exprs_);

  // Swap per-Fusion name counters (Phase 2 Task 10)
  std::swap(a.per_fusion_val_name_map_, b.per_fusion_val_name_map_);
  std::swap(a.per_fusion_expr_name_counter_, b.per_fusion_expr_name_counter_);
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  // NOTE: This method is deprecated in Phase 2. Fusion::copy handles
  // copying with shared containers. This is kept for backward compatibility
  // but should not be called directly.

  // Lock both containers: shared for reading from, unique for writing to
  std::shared_lock lock_from(from->mutex_);
  std::unique_lock lock_to(to->mutex_);

  // Clear without calling clear() which would try to re-acquire the lock
  to->vals_.clear();
  to->vals_up_.clear();
  to->exprs_.clear();
  to->exprs_up_.clear();
  to->val_type_name_map_.clear();
  to->expr_name_counter_ = 0;
  to->per_fusion_vals_.clear();
  to->per_fusion_exprs_.clear();
  to->per_fusion_val_name_map_.clear();
  to->per_fusion_expr_name_counter_.clear();

  // NOTE: In Phase 2, we can't use to->parent() here because parent_ might
  // not be set correctly for shared containers. Fusion::copy handles this.
  NVF_ERROR(
      to->parent_ != nullptr,
      "IrContainer::copy requires parent_ to be set. Use Fusion::copy "
      "instead.");
  IrCloner ir_cloner(to->parent());

  // Copy values in deterministic order
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

  // Remove from per-Fusion tracking (Phase 2 Task 4)
  if (expr->container() != nullptr) {
    auto it = per_fusion_exprs_.find(expr->container());
    if (it != per_fusion_exprs_.end()) {
      it->second.erase(expr);
    }
  }

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

  // Remove from per-Fusion tracking (Phase 2 Task 4)
  if (val->container() != nullptr) {
    auto it = per_fusion_vals_.find(val->container());
    if (it != per_fusion_vals_.end()) {
      it->second.erase(val);
    }
  }

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

  // Phase 2 Task 10: Use per-Fusion counter if val has an owning Fusion.
  // This ensures cloned Fusions get matching names (T0=T0, T1=T1)
  // instead of incrementing global names (T0=T10, T1=T11).
  Fusion* owning_fusion = val->container();
  val->setName(IrContainerPasskey(), getValName(owning_fusion, val->vtype()));

  // Track per-Fusion ownership (Phase 2 Task 4)
  if (owning_fusion != nullptr) {
    per_fusion_vals_[owning_fusion].insert(val);
  }
}

//! Register expr with this container.
void IrContainer::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  // Otherwise handle registration locally
  exprs_up_.emplace_back(expr);
  exprs_.insert(expr);

  // Phase 2 Task 10: Use per-Fusion counter if expr has an owning Fusion.
  Fusion* owning_fusion = expr->container();
  expr->setName(IrContainerPasskey(), getExprName(owning_fusion));

  // Track per-Fusion ownership (Phase 2 Task 4)
  if (owning_fusion != nullptr) {
    per_fusion_exprs_[owning_fusion].insert(expr);
  }
}

void IrContainer::clear() noexcept {
  FUSER_PERF_SCOPE("IrContainer clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  val_type_name_map_.clear();
  expr_name_counter_ = 0;

  // Clear per-Fusion tracking (Phase 2 Task 4)
  per_fusion_vals_.clear();
  per_fusion_exprs_.clear();

  // Clear per-Fusion name counters (Phase 2 Task 10)
  per_fusion_val_name_map_.clear();
  per_fusion_expr_name_counter_.clear();
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

  // Phase 2: With shared containers, multiple Fusions can share this container.
  // The statement's container() returns its owning Fusion, which should be
  // one of the Fusions sharing this container.
  // Phase 1 (single Fusion): sharing_fusions_ == {parent_}
  // Phase 2 (shared container): sharing_fusions_ contains multiple Fusions
  NVF_ERROR(
      sharing_fusions_.count(const_stmt->container()) > 0,
      "Container claims to own stmt, but stmt's owning Fusion is not "
      "registered with this container.");

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
    Fusion* fusion,
    int64_t prev_num_exprs,
    int64_t prev_num_vals) {
  std::unique_lock lock(mutex_);

  // Phase 2: Remove only statements owned by the specified Fusion.
  // This correctly handles shared containers where multiple Fusions
  // have statements interleaved in the container's deques.

  // Get current per-Fusion counts
  auto vals_it = per_fusion_vals_.find(fusion);
  auto exprs_it = per_fusion_exprs_.find(fusion);

  int64_t current_fusion_exprs =
      (exprs_it != per_fusion_exprs_.end()) ? exprs_it->second.size() : 0;
  int64_t current_fusion_vals =
      (vals_it != per_fusion_vals_.end()) ? vals_it->second.size() : 0;

  // Calculate how many statements to remove from this Fusion
  int64_t exprs_to_remove = current_fusion_exprs - prev_num_exprs;
  int64_t vals_to_remove = current_fusion_vals - prev_num_vals;

  if (exprs_to_remove <= 0 && vals_to_remove <= 0) {
    return; // Nothing to remove
  }

  // Remove expressions owned by this Fusion (from back of deque)
  // We iterate backwards and remove only those owned by this Fusion
  int64_t exprs_removed = 0;
  // Use index-based iteration to avoid iterator invalidation issues
  for (int64_t i = static_cast<int64_t>(exprs_up_.size()) - 1;
       i >= 0 && exprs_removed < exprs_to_remove;
       --i) {
    Expr* e = exprs_up_[i].get();
    if (e->container() == fusion) {
      // Clean up use-def chains
      for (Val* in : e->inputs()) {
        in->removeUse(e);
      }
      // Remove from tracking sets
      exprs_.erase(e);
      if (exprs_it != per_fusion_exprs_.end()) {
        exprs_it->second.erase(e);
      }
      // Erase from deque
      exprs_up_.erase(exprs_up_.begin() + i);
      exprs_removed++;
    }
  }

  // Remove vals owned by this Fusion (from back of deque)
  int64_t vals_removed = 0;
  for (int64_t i = static_cast<int64_t>(vals_up_.size()) - 1;
       i >= 0 && vals_removed < vals_to_remove;
       --i) {
    Val* v = vals_up_[i].get();
    if (v->container() == fusion) {
      // Remove from tracking sets
      vals_.erase(v);
      if (vals_it != per_fusion_vals_.end()) {
        vals_it->second.erase(v);
      }
      // Erase from deque
      vals_up_.erase(vals_up_.begin() + i);
      vals_removed++;
    }
  }
}

} // namespace nvfuser
