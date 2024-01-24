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
#include <utils.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace nvfuser {

class IrBuilderPasskey;
class ExprPasskey;
class OptOutMutator;

class NamedScalar;

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;

 private:
  explicit IrContainerPasskey() = default;
};

class IrContainer : public PolymorphicBase {
 public:
  IrContainer();

  IrContainer(const IrContainer& other);
  IrContainer(IrContainer&& other) noexcept;

  IrContainer& operator=(const IrContainer& other);
  IrContainer& operator=(IrContainer&& other) noexcept;

  ~IrContainer() override;

  flatbuffers::Offset<serde::IrContainer> serialize(
      const IrSerde& container,
      flatbuffers::FlatBufferBuilder& builder) const;

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    NVF_CHECK(
        inContainer(stmt), msg, " it was not found in the active container.");
  }

  Expr* getExpr(int64_t index) {
    NVF_CHECK(
        index < (int64_t)exprs_up_.size(), "Out of bounds expression index.");
    if (index < 0) {
      return nullptr;
    }
    return exprs_up_.at(index).get();
  }

  template <typename NvfuserValType>
  NvfuserValType* getVal(int64_t index) {
    NVF_CHECK(index < (int64_t)vals_up_.size(), "Out of bounds value index.");
    if (index < 0) {
      return nullptr;
    }
    Val* v = vals_up_.at(index).get();
    if constexpr (std::is_same_v<Val, NvfuserValType>) {
      return v;
    }
    NVF_CHECK(
        v->isA<NvfuserValType>(), "nvf::Val* does not have desired type.");
    return v->as<NvfuserValType>();
  }

  template <typename NvfuserValType>
  std::vector<NvfuserValType*> getValues(
      const flatbuffers::Vector<int64_t>* buffer) {
    NVF_CHECK(buffer != nullptr, "Values buffer is nullptr");
    std::vector<NvfuserValType*> result;
    result.reserve(buffer->size());
    std::transform(
        buffer->begin(),
        buffer->end(),
        std::back_inserter(result),
        [&](int64_t index) { return getVal<NvfuserValType>(index); });
    return result;
  }

  std::vector<Expr*> getExpressions(const flatbuffers::Vector<int64_t>* buffer);

  std::vector<Statement*> getStatements(
      const flatbuffers::Vector<flatbuffers::Offset<serde::Statement>>* buffer);

  //! Return values in insertion order
  const std::deque<Val*> deterministic_vals() const noexcept {
    std::deque<Val*> vals_deque;
    std::transform(
        vals_up_.begin(),
        vals_up_.end(),
        std::back_inserter(vals_deque),
        [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
    return vals_deque;
  }

  //! Return expression in insertion order
  const std::deque<Expr*> deterministic_exprs() const noexcept {
    std::deque<Expr*> exprs_deque;
    std::transform(
        exprs_up_.begin(),
        exprs_up_.end(),
        std::back_inserter(exprs_deque),
        [](const std::unique_ptr<Expr>& expr_up) { return expr_up.get(); });
    return exprs_deque;
  }

  //! Return mapping from value to integer id
  const std::unordered_map<Val*, int64_t> deterministic_vals_map()
      const noexcept {
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
  const std::unordered_map<Expr*, int64_t> deterministic_exprs_map()
      const noexcept {
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
