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
#include "ir/builder.h"
#include "ir/cloner.h"
#include "ir/internal_nodes.h"

namespace nvfuser {

//! Return values in insertion order
const std::deque<Val*> IrStorage::deterministic_vals() const noexcept {
  std::deque<Val*> vals_deque;
  std::transform(
      vals_up_.begin(),
      vals_up_.end(),
      std::back_inserter(vals_deque),
      [](const std::unique_ptr<Val>& val_up) { return val_up.get(); });
  return vals_deque;
}

//! Return expression in insertion order
const std::deque<Expr*> IrStorage::deterministic_exprs() const noexcept {
  std::deque<Expr*> exprs_deque;
  std::transform(
      exprs_up_.begin(),
      exprs_up_.end(),
      std::back_inserter(exprs_deque),
      [](const std::unique_ptr<Expr>& expr_up) { return expr_up.get(); });
  return exprs_deque;
}

//! Return mapping from value to integer id
const std::unordered_map<Val*, int64_t> IrStorage::deterministic_vals_map()
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
const std::unordered_map<Expr*, int64_t> IrStorage::deterministic_exprs_map()
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

void IrStorage::swap(IrStorage& a, IrStorage& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  // Swap the content
  std::swap(a.vals_up_, b.vals_up_);
  std::swap(a.vals_, b.vals_);

  std::swap(a.exprs_up_, b.exprs_up_);
  std::swap(a.exprs_, b.exprs_);

  std::swap(a.val_type_name_map_, b.val_type_name_map_);
  std::swap(a.expr_name_counter_, b.expr_name_counter_);

  std::swap(a.metadata_, b.metadata_);

  std::swap(a.parent_, b.parent_);

  std::swap(a.zero_val_, b.zero_val_);
  std::swap(a.one_val_, b.one_val_);
  std::swap(a.true_val_, b.true_val_);
  std::swap(a.false_val_, b.false_val_);
  std::swap(a.magic_zero_val_, b.magic_zero_val_);
  std::swap(a.axioms_, b.axioms_);
}

IrCloner IrStorage::copy(const IrStorage* from, IrStorage* to) {
  to->clear();
  IrCloner ir_cloner(to->parent());

  // Copy values in deterministic order
  // deterministic_vals can contain special values like one_val_, zero_val_, etc
  // that are not registered in the container.
  for (auto val : from->deterministic_vals()) {
    if (from->vals().count(val) > 0) {
      to->vals_.insert(ir_cloner.clone(val));
    }
  }

  // Copy expressions in deterministic order
  for (auto expr : from->deterministic_exprs()) {
    if (from->unordered_exprs().count(expr) > 0) {
      to->exprs_.insert(ir_cloner.clone(expr));
    }
  }

  to->val_type_name_map_ = from->val_type_name_map_;
  to->expr_name_counter_ = from->expr_name_counter_;

  if (from->axioms_ != nullptr) {
    to->axioms_ = std::make_unique<std::vector<Val*>>();
    for (auto pred : *from->axioms_) {
      to->axioms_->push_back(ir_cloner.clone(pred));
    }
  }

  to->metadata_ = ir_cloner.clone(from->metadata_);

  return ir_cloner;
}

IrStorage::IrStorage() = default;

IrStorage::~IrStorage() {
  clear();
}

void IrStorage::removeExpr(Expr* expr) {
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
void IrStorage::removeVal(Val* val) {
  // Don't remove shortcuts
  if (val == true_val_.get() || val == false_val_.get() ||
      val == one_val_.get() || val == zero_val_.get() ||
      val == magic_zero_val_.get()) {
    return;
  }

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
void IrStorage::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }

  // Otherwise handle registration locally
  vals_up_.emplace_back(val);
  vals_.insert(val);
  val->setName(IrContainerPasskey(), getValName(val->vtype()));
}

//! Register expr with this container.
void IrStorage::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }

  // Otherwise handle registration locally
  exprs_up_.emplace_back(expr);
  exprs_.insert(expr);
  expr->setName(IrContainerPasskey(), getExprName());
}

void IrStorage::clear() noexcept {
  FUSER_PERF_SCOPE("IrStorage clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  axioms_.reset();
  val_type_name_map_.clear();
  metadata_.clear();
  expr_name_counter_ = 0;
}

bool IrStorage::inContainer(const Statement* const_stmt) const {
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

// Shortcuts for frequently used vals
Val* IrStorage::zeroVal() {
  if (!zero_val_) {
    auto zero_val =
        IrBuilder::createInContainer<Val>(this->parent(), 0L, DataType::Index);
    NVF_ERROR(vals_up_.back().get() == zero_val);
    zero_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return zero_val_.get();
}

Val* IrStorage::zeroVal(DataType dtype) {
  if (dtype == DataType::Index) {
    return zeroVal();
  } else if (isBooleanType(dtype)) {
    return falseVal();
  } else {
    // NOTE: this does not cache values
    return IrBuilder::createInContainer<Val>(this->parent(), 0L, dtype);
  }
}

Val* IrStorage::oneVal() {
  if (!one_val_) {
    auto one_val =
        IrBuilder::createInContainer<Val>(this->parent(), 1L, DataType::Index);
    NVF_ERROR(vals_up_.back().get() == one_val);
    one_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return one_val_.get();
}

Val* IrStorage::oneVal(DataType dtype) {
  if (dtype == DataType::Index) {
    return oneVal();
  } else if (isBooleanType(dtype)) {
    return trueVal();
  } else {
    // NOTE: this does not cache values
    return IrBuilder::createInContainer<Val>(this->parent(), 1L, dtype);
  }
}

Val* IrStorage::falseVal() {
  if (!false_val_) {
    auto false_val = IrBuilder::createInContainer<Val>(
        this->parent(), false, DataType::Bool);
    NVF_ERROR(vals_up_.back().get() == false_val);
    false_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return false_val_.get();
}

Val* IrStorage::trueVal() {
  if (!true_val_) {
    auto true_val =
        IrBuilder::createInContainer<Val>(this->parent(), true, DataType::Bool);
    NVF_ERROR(vals_up_.back().get() == true_val);
    true_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return true_val_.get();
}

NamedScalar* IrStorage::magicZeroVal() {
  if (!magic_zero_val_) {
    auto magic_zero =
        IrBuilder::create<NamedScalar>(kMagicZeroName, DataType::Index);
    NVF_ERROR(vals_up_.back().get() == magic_zero);
    magic_zero_val_ = std::unique_ptr<NamedScalar>(
        vals_up_.back().release()->as<NamedScalar>());
    vals_up_.pop_back();
  }
  return magic_zero_val_.get();
}

Val* IrStorage::metadataOf(Val* v) {
  if (metadata_.count(v) == 0) {
    auto metadata_val =
        IrBuilder::createInContainer<Val>(this->parent(), metaDataTypeOf(v));
    auto metadata_expr = IrBuilder::createInContainer<GetMetaData>(
        this->parent(), metadata_val, v);
    metadata_[v] = std::make_pair(metadata_val, metadata_expr);
  }
  return metadata_.at(v).first;
}

void IrStorage::lazyInitAxioms() {
  if (!axioms_) {
    axioms_ = std::make_unique<std::vector<Val*>>();
    axioms_->reserve(kParallelTypeThreads.size() * 3);
    auto zero = zeroVal();
    for (auto p : kParallelTypeThreads) {
      auto pidx = NamedScalar::getParallelIndex(p);
      auto pdim = NamedScalar::getParallelDim(p);
      axioms_->push_back(SimplifyingIrBuilder::geExpr(pidx, zero));
      axioms_->push_back(SimplifyingIrBuilder::gtExpr(pdim, zero));
      axioms_->push_back(SimplifyingIrBuilder::ltExpr(pidx, pdim));
    }
  }
}

void IrStorage::assumePositive(Val* val) {
  NVF_ERROR(val->container() == this->parent());
  lazyInitAxioms();
  axioms_->emplace_back(IrBuilder::gtExpr(val, zeroVal()));
}

void IrStorage::assumeNonNegative(Val* val) {
  NVF_ERROR(val->container() == this->parent());
  lazyInitAxioms();
  axioms_->emplace_back(IrBuilder::geExpr(val, zeroVal()));
}

void IrStorage::removeStatementsCreatedAfter(
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
