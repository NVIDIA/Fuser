// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <memory>

#include <ir/base_nodes.h>
#include <ir/storage.h>
#include <visibility.h>

namespace nvfuser {

class IrBuilderPasskey;
class ExprPasskey;
class OptOutMutator;

// Passkey for container to register names with statements
class IrContainerPasskey {
  friend class IrContainer;
  friend class IrStorage;

 private:
  explicit IrContainerPasskey() = default;
};

// IrContainer: Base class for types that provide IrContainer API via
// composition
//
// This class handles the composition infrastructure and forwarding boilerplate
// for accessing IrStorage functionality. Derived classes (like Fusion) can
// focus on their specific logic while inheriting the full IrContainer API.
//
// Key Features:
// - Owns IrStorage via unique_ptr (can be shared_ptr in Phase 2)
// - Forwards all IrStorage public methods
// - Allows derived classes to override protected IrContainer methods
class NVF_API IrContainer : public PolymorphicBase {
 protected:
  // Constructors
  explicit IrContainer();

  // TODO: The semantics of IrContainers are largely driven through copy/swap
  // function behavior. It might be better if this behaviour was properly
  // defined through class semantics directly.
  //
  // Copy/Move are deleted. IrContainer is a forwarding interface class. We
  // rely on copy/swap function behavior to handle the semantics of IrStorage.
  IrContainer(const IrContainer& other) = delete;
  IrContainer(IrContainer&& other) noexcept = delete;
  IrContainer& operator=(const IrContainer& other) = delete;
  IrContainer& operator=(IrContainer&& other) noexcept = delete;

  ~IrContainer() override;

  // Let mutator remove Exprs.
  friend OptOutMutator;

 public:
  //===================================================================
  // IrStorage API Forwarding (Public Methods)
  //===================================================================

  // Container queries
  bool inContainer(const Statement* stmt) const {
    return ir_storage()->inContainer(stmt);
  }

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    ir_storage()->assertInContainer(stmt, msg);
  }

  // Collections access (return values in insertion order)
  const std::deque<Val*> deterministic_vals() const noexcept {
    return ir_storage()->deterministic_vals();
  }

  const std::deque<Expr*> deterministic_exprs() const noexcept {
    return ir_storage()->deterministic_exprs();
  }

  const std::unordered_map<Val*, int64_t> deterministic_vals_map()
      const noexcept {
    return ir_storage()->deterministic_vals_map();
  }

  const std::unordered_map<Expr*, int64_t> deterministic_exprs_map()
      const noexcept {
    return ir_storage()->deterministic_exprs_map();
  }

  // Collections access (unordered sets)
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
    return ir_storage()->unordered_exprs();
  }

  const std::unordered_set<Val*>& vals() const noexcept {
    return ir_storage()->vals();
  }

  // Count queries
  int64_t numExprs() const noexcept {
    return ir_storage()->numExprs();
  }

  int64_t numVals(bool include_shortcuts) const noexcept {
    return ir_storage()->numVals(include_shortcuts);
  }

  // Shortcut values (frequently used constants)
  Val* zeroVal() {
    return ir_storage()->zeroVal();
  }

  Val* oneVal() {
    return ir_storage()->oneVal();
  }

  Val* falseVal() {
    return ir_storage()->falseVal();
  }

  Val* trueVal() {
    return ir_storage()->trueVal();
  }

  NamedScalar* magicZeroVal() {
    return ir_storage()->magicZeroVal();
  }

  Val* zeroVal(DataType dtype) {
    return ir_storage()->zeroVal(dtype);
  }

  Val* oneVal(DataType dtype) {
    return ir_storage()->oneVal(dtype);
  }

  Val* metadataOf(Val* val) {
    return ir_storage()->metadataOf(val);
  }

  // Axioms (CUDA programming assumptions)
  const std::vector<Val*>& axioms() {
    return ir_storage()->axioms();
  }

  void assumePositive(Val* val) {
    ir_storage()->assumePositive(val);
  }

  void assumeNonNegative(Val* val) {
    ir_storage()->assumeNonNegative(val);
  }

  // Statement removal
  void removeStatementsCreatedAfter(
      int64_t num_exprs_before,
      int64_t num_vals_before) {
    ir_storage()->removeStatementsCreatedAfter(
        num_exprs_before, num_vals_before);
  }

  // Registration (public API with passkey)
  virtual void registerStmt(IrBuilderPasskey passkey, Statement* stmt) {
    // Dispatch to Val or Expr registration, which calls the virtual protected
    // methods that subclasses (like Fusion) override
    if (stmt->isVal()) {
      registerVal(passkey, stmt->asVal());
    } else {
      registerExpr(passkey, stmt->asExpr());
    }
  }

  virtual void registerVal(IrBuilderPasskey passkey, Val* val) {
    // Call the protected virtual method that subclasses override
    registerVal(val);
  }

  virtual void registerExpr(IrBuilderPasskey passkey, Expr* expr) {
    // Call the protected virtual method that subclasses override
    registerExpr(expr);
  }

  //===================================================================
  // Container Access
  //===================================================================

  // Direct access to underlying container
  IrStorage* ir_storage() {
    NVF_ERROR(
        ir_storage_.get() != nullptr, "Accessing a uninitialized IrContainer!.")
    return ir_storage_.get();
  }

  const IrStorage* ir_storage() const {
    NVF_ERROR(
        ir_storage_.get() != nullptr, "Accessing a uninitialized IrContainer!.")
    return ir_storage_.get();
  }

 protected:
  //===================================================================
  // Protected Registration API (for derived class overrides)
  //===================================================================

  static IrCloner copy(const IrContainer* from, IrContainer* to);
  static void swap(IrContainer& a, IrContainer& b) noexcept;

  // Derived classes (like Fusion) override these to add custom logic
  virtual void registerVal(Val* val) {
    ir_storage()->registerVal(val);
  }

  virtual void registerExpr(Expr* expr) {
    ir_storage()->registerExpr(expr);
  }

  virtual void removeExpr(Expr* expr) {
    ir_storage()->removeExpr(expr);
  }

  virtual void removeVal(Val* val) {
    ir_storage()->removeVal(val);
  }

 private:
  //===================================================================
  // Data Members
  //===================================================================

  std::unique_ptr<IrStorage> ir_storage_;
};

} // namespace nvfuser
