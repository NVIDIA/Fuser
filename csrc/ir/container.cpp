// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <instrumentation.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/container.h>

namespace nvfuser {

void swap(IrContainer& a, IrContainer& b) noexcept {
  FUSER_PERF_SCOPE("Fusion swap");

  using std::swap;

  // Swap the content
  swap(a.vals_up_, b.vals_up_);
  swap(a.vals_, b.vals_);

  swap(a.exprs_up_, b.exprs_up_);
  swap(a.exprs_, b.exprs_);

  swap(a.raw_ptrs_, b.raw_ptrs_);

  swap(a.val_type_name_map_, b.val_type_name_map_);
  swap(a.expr_name_counter_, b.expr_name_counter_);

  swap(a.metadata_, b.metadata_);

  // Fixup the Statement::fusion_ links for a
  for (auto val : a.vals_) {
    val->ir_container_ = &a;
  }
  for (auto expr : a.exprs_) {
    expr->ir_container_ = &a;
  }

  // Fixup the Statement::fusion_ links for b
  for (auto val : b.vals_) {
    val->ir_container_ = &a;
  }
  for (auto expr : b.exprs_) {
    expr->ir_container_ = &a;
  }
}

IrCloner IrContainer::copy(const IrContainer* from, IrContainer* to) {
  to->clear();
  IrCloner ir_cloner(to);

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
      to->axioms_->emplace_back(ir_cloner.clone(pred));
    }
  }

  to->metadata_ = ir_cloner.clone(from->metadata_);

  return ir_cloner;
}

IrContainer::IrContainer() = default;

IrContainer::IrContainer(const IrContainer& other) {
  FUSER_PERF_SCOPE("IrContainer copy");
  IrContainer::copy(&other, this);
}

IrContainer::IrContainer(IrContainer&& other) noexcept {
  FUSER_PERF_SCOPE("IrContainer move");
  swap(*this, other);
}

IrContainer& IrContainer::operator=(const IrContainer& other) {
  FUSER_PERF_SCOPE("IrContainer copy assign");
  IrContainer copy(other);
  clear();
  swap(*this, copy);
  return *this;
}

IrContainer& IrContainer::operator=(IrContainer&& other) noexcept {
  FUSER_PERF_SCOPE("IrContainer move assign");
  clear();
  swap(*this, other);
  return *this;
}

IrContainer::~IrContainer() {
  clear();
}

flatbuffers::Offset<serde::IrContainer> IrContainer::serialize(
    const IrSerde& container,
    flatbuffers::FlatBufferBuilder& builder) const {
  // Copy values in deterministic order
  // deterministic_vals can contain special values like one_val_, zero_val_, etc
  // that are not registered in the container.
  std::vector<flatbuffers::Offset<serde::Value>> fb_vals;
  fb_vals.reserve(vals().size());
  for (auto val : deterministic_vals()) {
    if (vals().count(val) > 0) {
      fb_vals.push_back(val->serialize(container, builder));
    }
  }

  // Copy expressions in deterministic order
  std::vector<flatbuffers::Offset<serde::Expression>> fb_exprs;
  fb_exprs.reserve(unordered_exprs().size());
  for (auto expr : deterministic_exprs()) {
    if (unordered_exprs().count(expr) > 0) {
      fb_exprs.push_back(expr->serialize(container, builder));
    }
  }

  std::vector<int64_t> fb_val_type_name_map_keys;
  std::vector<uint64_t> fb_val_type_name_map_values;
  fb_val_type_name_map_keys.reserve(val_type_name_map_.size());
  fb_val_type_name_map_values.reserve(val_type_name_map_.size());
  for (const auto& item : val_type_name_map_) {
    fb_val_type_name_map_keys.push_back(toUnderlying(item.first));
    fb_val_type_name_map_values.push_back(item.second);
  }

  std::vector<int64_t> fb_axioms;
  if (axioms_ != nullptr) {
    fb_axioms.reserve(axioms_->size());
    for (auto pred : *axioms_) {
      fb_axioms.push_back(container.map(pred));
    }
  }

  std::vector<int64_t> fb_metadata_keys;
  std::vector<int64_t> fb_metadata_values_lhs;
  std::vector<int64_t> fb_metadata_values_rhs;
  fb_metadata_keys.reserve(metadata_.size());
  fb_metadata_values_lhs.reserve(metadata_.size());
  fb_metadata_values_rhs.reserve(metadata_.size());
  for (const auto& item : metadata_) {
    fb_metadata_keys.push_back(container.map(item.first));
    auto&& [val, expr] = item.second;
    fb_metadata_values_lhs.push_back(container.map(val));
    fb_metadata_values_rhs.push_back(container.map(expr));
  }

  return serde::CreateIrContainerDirect(
      builder,
      &fb_vals,
      &fb_exprs,
      &fb_val_type_name_map_keys,
      &fb_val_type_name_map_values,
      expr_name_counter_,
      (fb_axioms.empty()) ? 0 : &fb_axioms,
      &fb_metadata_keys,
      &fb_metadata_values_lhs,
      &fb_metadata_values_rhs);
}

//! Register the Statement with this container
void IrContainer::registerStmt(IrBuilderPasskey, Statement* stmt) {
  if (stmt->isVal()) {
    registerVal(stmt->asVal());
  } else {
    registerExpr(stmt->asExpr());
  }
}

//! Register the Val with this container
void IrContainer::registerVal(IrBuilderPasskey, Val* val) {
  registerVal(val);
}

//! Register expr with this container.
void IrContainer::registerExpr(IrBuilderPasskey, Expr* expr) {
  registerExpr(expr);
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
  raw_ptrs_.erase((void*)expr);
}

//! Completely remove val from the fusion, break all dependencies associated
//! with it
void IrContainer::removeVal(Val* val) {
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
  raw_ptrs_.erase((void*)val);
}

//! Register the Val with this container
void IrContainer::registerVal(Val* val) {
  if (inContainer(val)) {
    return;
  }

  vals_up_.emplace_back(std::unique_ptr<Val>(val));
  vals_.emplace(vals_up_.back().get());
  val->setName(IrContainerPasskey(), getValName(vals_up_.back()->vtype()));
  raw_ptrs_.emplace((void*)vals_up_.back().get());
}

//! Register expr with this container.
void IrContainer::registerExpr(Expr* expr) {
  if (inContainer(expr)) {
    return;
  }
  exprs_up_.emplace_back(std::unique_ptr<Expr>(expr));
  exprs_.emplace(exprs_up_.back().get());
  expr->setName(IrContainerPasskey(), getExprName());
  raw_ptrs_.emplace((void*)exprs_up_.back().get());
}

void IrContainer::clear() noexcept {
  FUSER_PERF_SCOPE("IrContainer clear");
  vals_.clear();
  vals_up_.clear();
  exprs_.clear();
  exprs_up_.clear();
  raw_ptrs_.clear();
  axioms_.reset();
  val_type_name_map_.clear();
  metadata_.clear();
  expr_name_counter_ = 0;
}

bool IrContainer::inContainer(const Statement* stmt) const {
  const void* const_void = (const void*)(stmt);
  void* nonconst_void = const_cast<void*>(const_void); // NOLINT
  if (raw_ptrs_.find(nonconst_void) == raw_ptrs_.end()) {
    return false;
  }

  NVF_ERROR(
      stmt->container() == this,
      "Container claims to own stmt, but stmt disagrees.");

  Statement* nonconst_stmt = const_cast<Statement*>(stmt); // NOLINT
  if (stmt->isExpr()) {
    NVF_ERROR(
        exprs_.find(nonconst_stmt->as<Expr>()) != exprs_.end(),
        "Somehow container claims to and not to own an Expr.");
  }
  if (stmt->isVal()) {
    NVF_ERROR(
        vals_.find(nonconst_stmt->as<Val>()) != vals_.end(),
        "Somehow container claims to and not to own an Val.");
  }

  return true;
}

// Shortcuts for frequently used vals
Val* IrContainer::zeroVal() {
  if (!zero_val_) {
    auto zero_val = IrBuilder::create<Val>(this, 0L, DataType::Index);
    NVF_ERROR(vals_up_.back().get() == zero_val);
    zero_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return zero_val_.get();
}

Val* IrContainer::zeroVal(DataType dtype) {
  if (dtype == DataType::Index) {
    return zeroVal();
  } else if (isBooleanType(dtype)) {
    return falseVal();
  } else {
    // NOTE: this does not cache values
    return IrBuilder::create<Val>(this, 0L, dtype);
  }
}

Val* IrContainer::oneVal() {
  if (!one_val_) {
    auto one_val = IrBuilder::create<Val>(this, 1L, DataType::Index);
    NVF_ERROR(vals_up_.back().get() == one_val);
    one_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return one_val_.get();
}

Val* IrContainer::oneVal(DataType dtype) {
  if (dtype == DataType::Index) {
    return oneVal();
  } else if (isBooleanType(dtype)) {
    return trueVal();
  } else {
    // NOTE: this does not cache values
    return IrBuilder::create<Val>(this, 1L, dtype);
  }
}

Val* IrContainer::falseVal() {
  if (!false_val_) {
    auto false_val = IrBuilder::create<Val>(this, false, DataType::Bool);
    NVF_ERROR(vals_up_.back().get() == false_val);
    false_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return false_val_.get();
}

Val* IrContainer::trueVal() {
  if (!true_val_) {
    auto true_val = IrBuilder::create<Val>(this, true, DataType::Bool);
    NVF_ERROR(vals_up_.back().get() == true_val);
    true_val_ = std::unique_ptr<Val>(vals_up_.back().release());
    vals_up_.pop_back();
  }
  return true_val_.get();
}

NamedScalar* IrContainer::magicZeroVal() {
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

Val* IrContainer::metadataOf(Val* v) {
  if (metadata_.count(v) == 0) {
    auto metadata_val = IrBuilder::create<Val>(this, metaDataTypeOf(v));
    auto metadata_expr = IrBuilder::create<GetMetaData>(this, metadata_val, v);
    metadata_[v] = std::make_pair(metadata_val, metadata_expr);
  }
  return metadata_.at(v).first;
}

void IrContainer::lazyInitAxioms() {
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

void IrContainer::assumePositive(Val* val) {
  NVF_ERROR(val->container() == this);
  lazyInitAxioms();
  axioms_->emplace_back(IrBuilder::gtExpr(val, zeroVal()));
}

void IrContainer::assumeNonNegative(Val* val) {
  NVF_ERROR(val->container() == this);
  lazyInitAxioms();
  axioms_->emplace_back(IrBuilder::geExpr(val, zeroVal()));
}

} // namespace nvfuser
