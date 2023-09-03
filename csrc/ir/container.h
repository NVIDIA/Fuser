// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <exceptions.h>

#include <ir/base_nodes.h>
#include <union_find.h>
#include <utils.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

class IrBuilderPasskey;
class ExprPasskey;
class OptOutMutator;

class IterDomain;
class NamedScalar;

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;

 private:
  explicit IrContainerPasskey() = default;
};

class TORCH_CUDA_CU_API IrContainer : public PolymorphicBase {
 public:
  IrContainer();

  IrContainer(const IrContainer& other);
  IrContainer(IrContainer&& other) noexcept;

  IrContainer& operator=(const IrContainer& other);
  IrContainer& operator=(IrContainer&& other) noexcept;

  ~IrContainer() override;

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    NVF_CHECK(
        inContainer(stmt), msg, " it was not found in the active container.");
  }

  //! Return in insertion order
  const std::deque<Val*> deterministic_vals() const noexcept {
    std::deque<Val*> vals_deque;
    std::transform(
        vals_up_.begin(),
        vals_up_.end(),
        std::back_inserter(vals_deque),
        [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
    return vals_deque;
  }

  //! Register the Statement with this container
  virtual void registerStmt(IrBuilderPasskey, Statement* stmt);

  //! Register the Val with this container
  virtual void registerVal(IrBuilderPasskey, Val* val);

  //! Register expr with this container.
  virtual void registerExpr(IrBuilderPasskey, Expr* expr);

  //! Return the set of Exprs registered with this fusion. Warning: This will
  //! return exprs outside inputs/outputs, so can be unsafe for use with
  //! segmented fusions.
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
    return exprs_;
  }

  //! Return the set of Vals registered with this fusion
  const std::unordered_set<Val*>& vals() const noexcept {
    return vals_;
  }

  // Shortcuts for frequently used vals
  Val* zeroVal();
  Val* oneVal();
  Val* falseVal();
  Val* trueVal();
  NamedScalar* magicZeroVal();
  Val* zeroVal(DataType dtype);
  Val* oneVal(DataType dtype);
  Val* metadataOf(Val*);

  // Axioms about CUDA programming, for example: threadIdx.x < blockDim.x
  const std::vector<Val*>& axioms() {
    lazyInitAxioms();
    return *axioms_;
  }

  void assumePositive(Val* val);
  void assumeNonNegative(Val* val);

  //! Assume two scalar vals are equal. This merges the equivalence classes of a
  //! and b in the scalar_equality_ UnionFind.
  void assumeEqual(const Val* a, const Val* b);

  //! Return true only if we have previously called assumeEqual(a, b)
  //! Note this method is non-const since it will do path compression on the
  //! underlying UnionFind.
  bool areEqual(const Val* a, const Val* b);

  //! Invert the mapping v -> v->name(). If name is invalid, nullptr is
  //! returned.
  Val* getValFromName(ValType vtype, StmtNameType name) {
    const auto& vfn = val_from_name_[(size_t)vtype];
    if (name >= vfn.size()) {
      return nullptr;
    }
    return vfn[name];
  }

  //! Print equivalence classes of scalars that were defined using
  //! assumeEqual
  void printScalarEquivalences();

  //! Set two IterDomains to be part of the same equivalence class in
  //! exact_mapping_. This also calls assumeEqual on their extents.
  void setExactMapped(const IterDomain* a, const IterDomain* b);

  //! Determine whether two IterDomains are marked as being exact mapped
  //! Note this method is non-const since it will do path compression on the
  //! underlying UnionFind.
  bool areExactMapped(const IterDomain* a, const IterDomain* b);

 protected:
  static IrCloner copy(const IrContainer* from, IrContainer* to);

  friend void swap(IrContainer& a, IrContainer& b) noexcept;

  // Let mutator remove Exprs.
  friend OptOutMutator;

  virtual void removeExpr(Expr* expr);

  //! Completely remove val from the fusion, break all dependencies associated
  //! with it
  virtual void removeVal(Val* val);

  //! Register the Val with this container
  virtual void registerVal(Val* val);

  //! Register expr with this container.
  virtual void registerExpr(Expr* expr);

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

  void lazyInitAxioms();

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

  // Used to implement a generic "inContainer" that can be passed an invalid
  // pointer. Specifically a pointer to a Statement owned by another container
  // that has been freed. We can't check normally with the unordered_sets we
  // already have because it would require a const_cast from a constant
  // expr/val, or a dynamic cast from a Statement.
  std::unordered_set<void*> raw_ptrs_;

  // Values names counters
  std::unordered_map<ValType, StmtNameType> val_type_name_map_;

  // We keep a mapping from name() to index in vals_up_ for each ValType. The
  // following should hold for any Val v:
  //   v == val_from_name_[(size_t)v->valType()][v->name()]]
  // This allows us to work with integers only and still be able to retrieve a
  // Val* for a given ValType. Note that since Vals can be removed, these
  // vectors might contain nullptr.
  std::vector<std::vector<Val*>> val_from_name_;

  // UnionFinds represent equivalence relations. Exact mapped IterDomains are
  // tracked with a UnionFind, and their extents are marked as equal using the
  // more general scalar_equality_ UnionFind. The scalar_equality_ UnionFind
  // should also be updated whenever Vals are proven to be equal.
  using UnionFindIndexType = uint16_t; // Change this if >64k Vals are expected
  UnionFind<UnionFindIndexType> scalar_equality_;
  UnionFind<UnionFindIndexType> exact_mapping_;

  // Expression names counter
  StmtNameType expr_name_counter_ = 0;

  // Manually store some persistent, frequently used nodes. It's very
  // challenging to do this anything but manually as detecting when a container
  // may or may not have one of these vals is tricky. Specifically because if
  // the container doesn't own it, it's hard to understand from the outside if
  // the node may have been removed then re-registered. It could also be tricky
  // to know when we're using a different container as in FusionCopy_test
  // demonstrates deleting then creating containers can result in the same
  // pointer for the container.
  std::unique_ptr<Val> true_val_;
  std::unique_ptr<Val> false_val_;
  std::unique_ptr<Val> one_val_;
  std::unique_ptr<Val> zero_val_;
  std::unique_ptr<NamedScalar> magic_zero_val_;
  std::unique_ptr<std::vector<Val*>> axioms_;
  std::unordered_map<Val*, std::pair<Val*, Expr*>> metadata_;
};

} // namespace nvfuser
