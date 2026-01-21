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
#include <ir/container.h>
#include <visibility.h>

namespace nvfuser {

// IrInterface: Base class for types that provide IrContainer API via
// composition
//
// This class handles the composition infrastructure and forwarding boilerplate
// for accessing IrContainer functionality. Derived classes (like Fusion) can
// focus on their specific logic while inheriting the full IrContainer API.
//
// Key Features:
// - Owns IrContainer via unique_ptr (can be shared_ptr in Phase 2)
// - Forwards all IrContainer public methods
// - Allows derived classes to override protected IrContainer methods
//
// This eliminates ~20 forwarding methods from Fusion and provides a reusable
// pattern for other classes that need IrContainer composition.
//
// Note: Uses virtual inheritance to avoid diamond inheritance ambiguity during
// Stage 2 (when Fusion inherits from both IrInterface and IrContainer).
class NVF_API IrInterface : public virtual PolymorphicBase {
 public:
  // Constructors
  IrInterface();

  // Copy/Move
  IrInterface(const IrInterface& other);
  IrInterface(IrInterface&& other) noexcept;
  IrInterface& operator=(const IrInterface& other);
  IrInterface& operator=(IrInterface&& other) noexcept;

  ~IrInterface() override;

  //===================================================================
  // IrContainer API Forwarding (Public Methods)
  //===================================================================

  // Container queries
  bool inContainer(const Statement* stmt) const {
    return container()->inContainer(stmt);
  }

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    container()->assertInContainer(stmt, msg);
  }

  // Collections access (return values in insertion order)
  const std::deque<Val*> deterministic_vals() const noexcept {
    return container()->deterministic_vals();
  }

  const std::deque<Expr*> deterministic_exprs() const noexcept {
    return container()->deterministic_exprs();
  }

  const std::unordered_map<Val*, int64_t> deterministic_vals_map()
      const noexcept {
    return container()->deterministic_vals_map();
  }

  const std::unordered_map<Expr*, int64_t> deterministic_exprs_map()
      const noexcept {
    return container()->deterministic_exprs_map();
  }

  // Collections access (unordered sets)
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
    return container()->unordered_exprs();
  }

  const std::unordered_set<Val*>& vals() const noexcept {
    return container()->vals();
  }

  // Count queries
  int64_t numExprs() const noexcept {
    return container()->numExprs();
  }

  int64_t numVals(bool include_shortcuts) const noexcept {
    return container()->numVals(include_shortcuts);
  }

  // Shortcut values (frequently used constants)
  Val* zeroVal() {
    return container()->zeroVal();
  }

  Val* oneVal() {
    return container()->oneVal();
  }

  Val* falseVal() {
    return container()->falseVal();
  }

  Val* trueVal() {
    return container()->trueVal();
  }

  NamedScalar* magicZeroVal() {
    return container()->magicZeroVal();
  }

  Val* zeroVal(DataType dtype) {
    return container()->zeroVal(dtype);
  }

  Val* oneVal(DataType dtype) {
    return container()->oneVal(dtype);
  }

  Val* metadataOf(Val* val) {
    return container()->metadataOf(val);
  }

  // Axioms (CUDA programming assumptions)
  const std::vector<Val*>& axioms() {
    return container()->axioms();
  }

  void assumePositive(Val* val) {
    container()->assumePositive(val);
  }

  void assumeNonNegative(Val* val) {
    container()->assumeNonNegative(val);
  }

  // Statement removal
  void removeStatementsCreatedAfter(
      int64_t num_exprs_before,
      int64_t num_vals_before) {
    container()->removeStatementsCreatedAfter(
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
  IrContainer* container() {
    NVF_ERROR(
        container_.get() != nullptr, "Accessing a uninitialized IrContainer!.")
    return container_.get();
  }

  const IrContainer* container() const {
    NVF_ERROR(
        container_.get() != nullptr, "Accessing a uninitialized IrContainer!.")
    return container_.get();
  }

 protected:
  //===================================================================
  // Protected Registration API (for derived class overrides)
  //===================================================================

  // Derived classes (like Fusion) override these to add custom logic
  virtual void registerVal(Val* val) {
    container()->registerVal(val);
  }

  virtual void registerExpr(Expr* expr) {
    container()->registerExpr(expr);
  }

  virtual void removeExpr(Expr* expr) {
    container()->removeExpr(expr);
  }

  virtual void removeVal(Val* val) {
    container()->removeVal(val);
  }

  // Note: getValName, getExprName, and clear are protected in IrContainer
  // and cannot be directly forwarded. Derived classes that need these
  // should access them through their own container_ member or implement
  // their own public wrappers.

  friend void swap(IrInterface& a, IrInterface& b) noexcept;

 private:
  //===================================================================
  // Data Members
  //===================================================================

  std::unique_ptr<IrContainer> container_;
};

// Swap support
void swap(IrInterface& a, IrInterface& b) noexcept;

} // namespace nvfuser
