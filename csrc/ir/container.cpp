// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "ir/container.h"

#include "instrumentation.h"
#include "ir/base_nodes.h"

namespace nvfuser {

//! Return values in insertion order
const std::deque<Val*> IrContainer::deterministic_vals() const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Val*> vals_deque;
  std::ranges::transform(
      vals_up_,
      std::back_inserter(vals_deque),
      [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
  return vals_deque;
}

//! Return expression in insertion order
const std::deque<Expr*> IrContainer::deterministic_exprs() const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Expr*> exprs_deque;
  std::ranges::transform(
      exprs_up_,
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
  std::ranges::transform(
      vals_up_,
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
  std::ranges::transform(
      exprs_up_,
      std::inserter(exprs_map, exprs_map.end()),
      [&count](const std::unique_ptr<Expr>& expr_up) {
        return std::make_pair(expr_up.get(), count++);
      });
  return exprs_map;
}

IrContainer::IrContainer() = default;

IrContainer::~IrContainer() {
  clear();
}

// Note: clear() does not acquire mutex_. It is only called from the
// destructor and Fusion::copy(), both of which guarantee exclusive access.
// This assumption must be revisited in Phase 3 when containers may be shared
// across threads.
void IrContainer::clear() noexcept {
  FUSER_PERF_SCOPE("IrContainer clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  val_type_name_map_.clear();
  expr_name_counter_ = 0;
  per_fusion_vals_.clear();
  per_fusion_exprs_.clear();
}

bool IrContainer::inContainer(const Statement* const_stmt) const {
  std::shared_lock lock(mutex_);
  return inContainerImpl(const_stmt);
}

bool IrContainer::inContainerImpl(const Statement* const_stmt) const {
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
      sharing_fusions_.count(const_stmt->container()) > 0,
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

void IrContainer::assertInContainerImpl(
    const Statement* stmt,
    const std::string& msg) const {
  NVF_CHECK(
      inContainerImpl(stmt), msg, " it was not found in the active container.");
}

const std::unordered_set<Expr*>& IrContainer::unordered_exprs() const noexcept {
  std::shared_lock lock(mutex_);
  return exprs_;
}

const std::unordered_set<Val*>& IrContainer::vals() const noexcept {
  std::shared_lock lock(mutex_);
  return vals_;
}

int64_t IrContainer::numExprs() const noexcept {
  std::shared_lock lock(mutex_);
  return std::ssize(exprs_);
}

int64_t IrContainer::numVals() const noexcept {
  std::shared_lock lock(mutex_);
  return std::ssize(vals_up_);
}

void IrContainer::addFusion(Fusion* fusion) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.insert(fusion);
  per_fusion_vals_[fusion]; // Pre-insert key so no outer-map rehash occurs
                            // during concurrent val/expr registration
  per_fusion_exprs_[fusion];
}

void IrContainer::removeFusion(Fusion* fusion) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.erase(fusion);
}

void IrContainer::transferFusion(Fusion* from, Fusion* to) {
  std::unique_lock lock(mutex_);
  sharing_fusions_.erase(from);
  sharing_fusions_.insert(to);
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

const std::unordered_set<Val*>& IrContainer::valsOwnedBy(
    const Fusion* fusion) const {
  std::shared_lock lock(mutex_);
  static const std::unordered_set<Val*> empty;
  auto it = per_fusion_vals_.find(fusion);
  return it != per_fusion_vals_.end() ? it->second : empty;
}

const std::unordered_set<Expr*>& IrContainer::exprsOwnedBy(
    const Fusion* fusion) const {
  std::shared_lock lock(mutex_);
  static const std::unordered_set<Expr*> empty;
  auto it = per_fusion_exprs_.find(fusion);
  return it != per_fusion_exprs_.end() ? it->second : empty;
}

void IrContainer::transferStatementOwnership(
    const Fusion* from,
    const Fusion* to) {
  std::unique_lock lock(mutex_);
  auto vals_it = per_fusion_vals_.find(from);
  if (vals_it != per_fusion_vals_.end()) {
    auto& to_vals = per_fusion_vals_[to];
    to_vals.insert(vals_it->second.begin(), vals_it->second.end());
    per_fusion_vals_.erase(vals_it);
  }

  auto exprs_it = per_fusion_exprs_.find(from);
  if (exprs_it != per_fusion_exprs_.end()) {
    auto& to_exprs = per_fusion_exprs_[to];
    to_exprs.insert(exprs_it->second.begin(), exprs_it->second.end());
    per_fusion_exprs_.erase(exprs_it);
  }
}

void IrContainer::removeStatementsOwnedBy(const Fusion* fusion) {
  std::unique_lock lock(mutex_);

  // Process Exprs FIRST — clean up uses_/definition_ pointers on Vals
  // before freeing Exprs. This prevents dangling pointers in shared
  // scalars' uses_ vectors (shared scalars survive Val cleanup via the
  // multi-owner guard but their uses_ would reference freed Exprs).
  auto exprs_it = per_fusion_exprs_.find(fusion);
  if (exprs_it != per_fusion_exprs_.end()) {
    const auto& owned = exprs_it->second;
    std::erase_if(exprs_up_, [&](const std::unique_ptr<Expr>& e) {
      if (owned.count(e.get()) > 0) {
        for (Val* out : e->outputs()) {
          out->setDefinition(nullptr);
        }
        for (Val* inp : e->inputs()) {
          inp->removeUse(e.get());
        }
        exprs_.erase(e.get());
        return true;
      }
      return false;
    });
    per_fusion_exprs_.erase(exprs_it);
  }

  // Then Vals — shared vals survive via multi-owner guard, now with
  // clean uses_ (dangling Expr pointers already removed above).
  auto vals_it = per_fusion_vals_.find(fusion);
  if (vals_it != per_fusion_vals_.end()) {
    const auto& owned = vals_it->second;
    std::erase_if(vals_up_, [&](const std::unique_ptr<Val>& v) {
      if (owned.count(v.get()) > 0) {
        // Multi-owner guard: only free if this is the last owning Fusion.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        if (!v->removeOwningFusion(const_cast<Fusion*>(fusion))) {
          return false; // other Fusions still own this Val — keep alive
        }
        vals_.erase(v.get());
        return true; // last owner gone → Val freed
      }
      return false;
    });
    per_fusion_vals_.erase(vals_it);
  }
}

std::deque<Val*> IrContainer::deterministicValsOwnedBy(
    const Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Val*> result;
  auto it = per_fusion_vals_.find(fusion);
  if (it == per_fusion_vals_.end()) {
    return result;
  }
  const auto& owned = it->second;
  for (const auto& val_up : vals_up_) {
    if (owned.count(val_up.get()) > 0) {
      result.push_back(val_up.get());
    }
  }
  return result;
}

std::deque<Expr*> IrContainer::deterministicExprsOwnedBy(
    const Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::deque<Expr*> result;
  auto it = per_fusion_exprs_.find(fusion);
  if (it == per_fusion_exprs_.end()) {
    return result;
  }
  const auto& owned = it->second;
  for (const auto& expr_up : exprs_up_) {
    if (owned.count(expr_up.get()) > 0) {
      result.push_back(expr_up.get());
    }
  }
  return result;
}

std::unordered_map<Val*, int64_t> IrContainer::deterministicValsMapOwnedBy(
    const Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Val*, int64_t> result;
  auto it = per_fusion_vals_.find(fusion);
  if (it == per_fusion_vals_.end()) {
    return result;
  }
  const auto& owned = it->second;
  int64_t count = 0;
  for (const auto& val_up : vals_up_) {
    if (owned.count(val_up.get()) > 0) {
      result[val_up.get()] = count++;
    }
  }
  return result;
}

std::unordered_map<Expr*, int64_t> IrContainer::deterministicExprsMapOwnedBy(
    const Fusion* fusion) const noexcept {
  std::shared_lock lock(mutex_);
  std::unordered_map<Expr*, int64_t> result;
  auto it = per_fusion_exprs_.find(fusion);
  if (it == per_fusion_exprs_.end()) {
    return result;
  }
  const auto& owned = it->second;
  int64_t count = 0;
  for (const auto& expr_up : exprs_up_) {
    if (owned.count(expr_up.get()) > 0) {
      result[expr_up.get()] = count++;
    }
  }
  return result;
}

} // namespace nvfuser
