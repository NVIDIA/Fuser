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
  friend class Fusion;

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
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept;

  //! Return the set of Vals registered with this fusion
  const std::unordered_set<Val*>& vals() const noexcept;

  int64_t numExprs() const noexcept;

  int64_t numVals() const noexcept;

 protected:
  // Let Fusion access IrContainer internals (mutex_, fields, Impl helpers)
  friend class Fusion;

  mutable std::shared_mutex mutex_;

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

 public:
  void addFusion(Fusion* fusion);
  void removeFusion(Fusion* fusion);
  void transferFusion(Fusion* from, Fusion* to);
  size_t sharingCount() const;
  bool hasMultipleFusions() const;
  const std::unordered_set<Fusion*>& sharingFusions() const;

  NVF_API const std::unordered_set<Val*>& valsOwnedBy(
      const Fusion* fusion) const;
  const std::unordered_set<Expr*>& exprsOwnedBy(const Fusion* fusion) const;
  void transferStatementOwnership(const Fusion* from, const Fusion* to);
  void removeStatementsOwnedBy(const Fusion* fusion);

  std::deque<Val*> deterministicValsOwnedBy(
      const Fusion* fusion) const noexcept;
  std::deque<Expr*> deterministicExprsOwnedBy(
      const Fusion* fusion) const noexcept;
  std::unordered_map<Val*, int64_t> deterministicValsMapOwnedBy(
      const Fusion* fusion) const noexcept;
  std::unordered_map<Expr*, int64_t> deterministicExprsMapOwnedBy(
      const Fusion* fusion) const noexcept;

 private:
  // Lock-free implementations for use by Fusion (which holds mutex_ directly)
  bool inContainerImpl(const Statement* stmt) const;
  void assertInContainerImpl(
      const Statement* stmt,
      const std::string& msg) const;

  std::unordered_set<Fusion*> sharing_fusions_;
  std::unordered_map<const Fusion*, std::unordered_set<Val*>> per_fusion_vals_;
  std::unordered_map<const Fusion*, std::unordered_set<Expr*>>
      per_fusion_exprs_;
};

} // namespace nvfuser
