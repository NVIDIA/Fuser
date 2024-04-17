// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>

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
  NVF_API IrContainer();

  IrContainer(const IrContainer& other);
  IrContainer(IrContainer&& other) noexcept;

  IrContainer& operator=(const IrContainer& other);
  IrContainer& operator=(IrContainer&& other) noexcept;

  ~IrContainer() override;

  flatbuffers::Offset<serde::IrContainer> serialize(
      const IrSerde& container,
      flatbuffers::FlatBufferBuilder& builder) const;

  void deserialize(const serde::IrContainer* buffer);

  bool inContainer(const Statement* stmt) const;

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    NVF_CHECK(
        inContainer(stmt), msg, " it was not found in the active container.");
  }

  bool validSerializationState() const {
    return valid_serialize_state_;
  }

  template <typename NvfuserExprType>
  NvfuserExprType* getExpr(int64_t index) {
    NVF_CHECK(
        index < (int64_t)exprs_up_.size(), "Out of bounds expression index.");
    if (index < 0) {
      return nullptr;
    }
    Expr* e = exprs_up_.at(index).get();
    if constexpr (std::is_same_v<Expr, NvfuserExprType>) {
      return e;
    }
    NVF_CHECK(
        e->isA<NvfuserExprType>(), "nvf::Expr* does not have desired type.");
    return e->as<NvfuserExprType>();
  }

  template <typename NvfuserValType>
  NvfuserValType* getVal(int64_t index) {
    NVF_CHECK(
        index < (int64_t)vals_up_.size(),
        "Out of bounds value index. Desired index ",
        index,
        " but there are only ",
        vals_up_.size(),
        " values.");

    if constexpr (!std::is_same_v<Val, NvfuserValType>) {
      if (index < 0) {
        return nullptr;
      }
    }

    if constexpr (std::is_same_v<Val, NvfuserValType>) {
      if (index == -1) {
        return nullptr;
      } else if (index == -2) {
        return zeroVal();
      } else if (index == -3) {
        return oneVal();
      } else if (index == -4) {
        return falseVal();
      } else if (index == -5) {
        return trueVal();
      }
    }

    if constexpr (std::is_same_v<NamedScalar, NvfuserValType>) {
      if (index == -6) {
        return magicZeroVal();
      }
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
      const flatbuffers::Vector<flatbuffers::Offset<serde::StatementIndex>>*
          buffer);

  //! Return values in insertion order
  std::deque<Val*> deterministic_vals() const noexcept;

  //! Return expression in insertion order
  std::deque<Expr*> deterministic_exprs() const noexcept;

  //! Return statements in insertion order
  std::vector<Statement*> deterministic_stmts() const noexcept {
    return stmts_;
  }

  //! Return mapping from value to integer id in deterministic order
  std::unordered_map<Val*, int64_t> deterministic_vals_map() const noexcept;

  //! Return mapping from expression to integer id
  std::unordered_map<Expr*, int64_t> deterministic_exprs_map() const noexcept;

  //! Register the Statement with this container
  NVF_API virtual void registerStmt(IrBuilderPasskey, Statement* stmt);

  //! Register the Val with this container
  NVF_API virtual void registerVal(IrBuilderPasskey, Val* val);

  //! Register expr with this container.
  NVF_API virtual void registerExpr(IrBuilderPasskey, Expr* expr);

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
  NVF_API Val* zeroVal();
  NVF_API Val* oneVal();
  Val* falseVal();
  Val* trueVal();
  NamedScalar* magicZeroVal();
  NVF_API Val* zeroVal(DataType dtype);
  NVF_API Val* oneVal(DataType dtype);
  Val* metadataOf(Val*);

  // These const methods will return a nullptr if the special values do not
  // exist.
  Val* getZeroVal() const;
  Val* getOneVal() const;
  Val* getFalseVal() const;
  Val* getTrueVal() const;
  NamedScalar* getMagicZeroVal() const;

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

  // All statements in container inserted in deterministic order
  std::vector<Statement*> stmts_;

  // Determine if the IrContainer is in a serialized state where its statements
  // are in toposort order. If this container is created from a serde buffer,
  // we do not need to run topological sort again during serialization. If we
  // modify the container, then the value and expression order is not valid.
  bool valid_serialize_state_ = false;

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
