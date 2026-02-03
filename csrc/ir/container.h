// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <deque>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

#include "base.h"
#include "exceptions.h"
#include "ir/base_nodes.h"
#include "visibility.h"

namespace nvfuser {

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;

 private:
  explicit IrContainerPasskey() = default;
};

class NamedScalar;

class IrContainer {
 public:
  NVF_API IrContainer();

  // Copy/Move Constructors and Operators are deleted. IrContainer is managed
  // through a smart pointer in IrContainer. Semantic operations for Fusion
  // types are handled directly through copy and swap functions.
  IrContainer(const IrContainer& other) = delete;
  IrContainer(IrContainer&& other) noexcept = delete;

  IrContainer& operator=(const IrContainer& other) = delete;
  IrContainer& operator=(IrContainer&& other) noexcept = delete;

  ~IrContainer();

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    NVF_CHECK(
        inContainer(stmt), msg, " it was not found in the active container.");
  }

  //! Return values in insertion order
  const std::deque<Val*> deterministic_vals() const noexcept;

  //! Return expression in insertion order
  const std::deque<Expr*> deterministic_exprs() const noexcept;

  //! Return mapping from value to integer id
  const std::unordered_map<Val*, int64_t> deterministic_vals_map()
      const noexcept;

  //! Return mapping from expression to integer id
  const std::unordered_map<Expr*, int64_t> deterministic_exprs_map()
      const noexcept;

  //! Return the set of Exprs registered with this fusion. Warning: This will
  //! return exprs outside inputs/outputs, so can be unsafe for use with
  //! segmented fusions.
  //! Note: Returns reference - caller must not hold across concurrent mods
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept;

  //! Return the set of Vals registered with this fusion
  //! Note: Returns reference - caller must not hold across concurrent mods
  const std::unordered_set<Val*>& vals() const noexcept;

  int64_t numExprs() const noexcept;

  // Note: The include_shortcuts parameter is now deprecated.
  // With Phase 2 per-Fusion special values, all vals (including special values)
  // are stored in vals_up_, so both vals_ and vals_up_ have the same size.
  // This parameter is kept for API compatibility but has no effect.
  int64_t numVals(bool include_shortcuts) const noexcept {
    return include_shortcuts ? std::ssize(vals_) : std::ssize(vals_up_);
  }

  // Note: Shortcut values (zeroVal, oneVal, trueVal, falseVal, magicZeroVal),
  // metadata, and axioms are now per-Fusion. Use Fusion::zeroVal(),
  // Fusion::metadataOf(), Fusion::axioms(), etc. instead.
  // This avoids ownership conflicts when multiple Fusions share an IrContainer.

 protected:
  // Mutex for thread-safe access when container is shared between Fusions
  // mutable because we need to lock in const methods
  mutable std::shared_mutex mutex_;

  static IrCloner copy(const IrContainer* from, IrContainer* to);

  static void swap(IrContainer& a, IrContainer& b) noexcept;

  // Let Fusion access IrContainer::clear()
  friend class Fusion;

  void removeExpr(Expr* expr);

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  void removeVal(Val* val);

  //! Register the Val with this container
  NVF_API void registerVal(Val* val);

  //! Register expr with this container.
  NVF_API void registerExpr(Expr* expr);

  StmtNameType getValName(ValType vtype) {
    if (val_type_name_map_.find(vtype) == val_type_name_map_.end()) {
      val_type_name_map_[vtype] = 0;
    }
    return val_type_name_map_[vtype]++;
  }

  StmtNameType getExprName() {
    return expr_name_counter_++;
  }

  void clear() noexcept;

  friend class StatementGuard;

  // A simple garbage collection mechanism to remove all Exprs and Vals that
  // were created after a certain point. This is useful for analysis that
  // creates new Exprs and Vals in the container and wants to clean up after
  // itself.
  //
  // Used by StatementGuard only.
  void removeStatementsCreatedAfter(
      int64_t prev_num_exprs,
      int64_t prev_num_vals);

  // Deque of unique pointer is the memory owning data structure
  std::deque<std::unique_ptr<Val>> vals_up_;

  // A convenient set to return when we just need an unordered set to do
  // something like check if a Val is in this container
  std::unordered_set<Val*> vals_;

  // Deque of unique pointer is the memory owning data structure
  std::deque<std::unique_ptr<Expr>> exprs_up_;

  // A convenient set to return when we just need an unordered set to do
  // something like check if an Expr is in this container
  std::unordered_set<Expr*> exprs_;

  // Values names counters
  std::unordered_map<ValType, StmtNameType> val_type_name_map_;

  // Expression names counter
  StmtNameType expr_name_counter_ = 0;

  // Note: Special values (zero_val_, one_val_, true_val_, false_val_,
  // magic_zero_val_) are now per-Fusion, stored in Fusion class.
  // This avoids ownership conflicts when multiple Fusions share an IrContainer.
  // See Fusion::zeroVal(), Fusion::axioms(), Fusion::metadataOf(), etc.
  // for the per-Fusion implementations.

 public:
  Fusion* parent() const {
    NVF_ERROR(
        parent_ != nullptr, "Call to IrContainer::parent() holds nullptr.")
    return parent_;
  }

 private:
  // Parent Fusion that owns this container (for pure composition pattern)
  // Used by Statement::fusion() to navigate back to owning Fusion
  Fusion* parent_ = nullptr;
};

} // namespace nvfuser
