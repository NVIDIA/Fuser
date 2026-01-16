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

class Fusion;

// IrInterface: Base class for types that provide IrContainer API via composition
//
// This class handles the composition infrastructure and forwarding boilerplate
// for accessing IrContainer functionality. Derived classes (like Fusion) can
// focus on their specific logic while inheriting the full IrContainer API.
//
// Key Features:
// - Owns IrContainer via unique_ptr (can be shared_ptr in Phase 2)
// - Forwards all IrContainer public methods
// - Provides virtual owningFusion() for polymorphic casting compatibility
// - Allows derived classes to override protected IrContainer methods
//
// This eliminates ~20 forwarding methods from Fusion and provides a reusable
// pattern for other classes that need IrContainer composition.
class NVF_API IrInterface : public PolymorphicBase {
 public:
  // Constructors
  IrInterface();
  explicit IrInterface(std::unique_ptr<IrContainer> container);

  // Copy/Move
  IrInterface(const IrInterface& other);
  IrInterface(IrInterface&& other) noexcept;
  IrInterface& operator=(const IrInterface& other);
  IrInterface& operator=(IrInterface&& other) noexcept;

  ~IrInterface() override = default;

  //===================================================================
  // IrContainer API Forwarding (Public Methods)
  //===================================================================

  // Container queries
  bool inContainer(const Statement* stmt) const {
    return container_->inContainer(stmt);
  }

  void assertInContainer(const Statement* stmt, const std::string& msg) const {
    container_->assertInContainer(stmt, msg);
  }

  // Collections access (return values in insertion order)
  const std::deque<Val*> deterministic_vals() const noexcept {
    return container_->deterministic_vals();
  }

  const std::deque<Expr*> deterministic_exprs() const noexcept {
    return container_->deterministic_exprs();
  }

  const std::unordered_map<Val*, int64_t> deterministic_vals_map()
      const noexcept {
    return container_->deterministic_vals_map();
  }

  const std::unordered_map<Expr*, int64_t> deterministic_exprs_map()
      const noexcept {
    return container_->deterministic_exprs_map();
  }

  // Collections access (unordered sets)
  const std::unordered_set<Expr*>& unordered_exprs() const noexcept {
    return container_->unordered_exprs();
  }

  const std::unordered_set<Val*>& vals() const noexcept {
    return container_->vals();
  }

  // Count queries
  int64_t numExprs() const noexcept {
    return container_->numExprs();
  }

  int64_t numVals(bool include_shortcuts) const noexcept {
    return container_->numVals(include_shortcuts);
  }

  // Shortcut values (frequently used constants)
  Val* zeroVal() {
    return container_->zeroVal();
  }

  Val* oneVal() {
    return container_->oneVal();
  }

  Val* falseVal() {
    return container_->falseVal();
  }

  Val* trueVal() {
    return container_->trueVal();
  }

  NamedScalar* magicZeroVal() {
    return container_->magicZeroVal();
  }

  Val* zeroVal(DataType dtype) {
    return container_->zeroVal(dtype);
  }

  Val* oneVal(DataType dtype) {
    return container_->oneVal(dtype);
  }

  Val* metadataOf(Val* val) {
    return container_->metadataOf(val);
  }

  // Axioms (CUDA programming assumptions)
  const std::vector<Val*>& axioms() {
    return container_->axioms();
  }

  void assumePositive(Val* val) {
    container_->assumePositive(val);
  }

  void assumeNonNegative(Val* val) {
    container_->assumeNonNegative(val);
  }

  // Registration (public API with passkey)
  virtual void registerStmt(IrBuilderPasskey passkey, Statement* stmt) {
    container_->registerStmt(passkey, stmt);
  }

  virtual void registerVal(IrBuilderPasskey passkey, Val* val) {
    container_->registerVal(passkey, val);
  }

  virtual void registerExpr(IrBuilderPasskey passkey, Expr* expr) {
    container_->registerExpr(passkey, expr);
  }

  //===================================================================
  // Container Access
  //===================================================================

  // Direct access to underlying container
  IrContainer* container() {
    return container_.get();
  }

  const IrContainer* container() const {
    return container_.get();
  }

  //===================================================================
  // Virtual Methods for Derived Classes
  //===================================================================

  // Returns the owning Fusion if this IrInterface is a Fusion, nullptr otherwise
  // This enables polymorphic casting patterns like: container()->as<Fusion>()
  virtual Fusion* owningFusion() {
    return nullptr;
  }

  virtual const Fusion* owningFusion() const {
    return nullptr;
  }

 protected:
  //===================================================================
  // Protected Registration API (for derived class overrides)
  //===================================================================

  // Derived classes (like Fusion) override these to add custom logic
  virtual void registerVal(Val* val) {
    container_->registerVal(val);
  }

  virtual void registerExpr(Expr* expr) {
    container_->registerExpr(expr);
  }

  virtual void removeExpr(Expr* expr) {
    container_->removeExpr(expr);
  }

  virtual void removeVal(Val* val) {
    container_->removeVal(val);
  }

  // Naming infrastructure (protected, used by derived classes)
  StmtNameType getValName(ValType vtype) {
    return container_->getValName(vtype);
  }

  StmtNameType getExprName() {
    return container_->getExprName();
  }

  // Clear all contents
  void clear() noexcept {
    container_->clear();
  }

  //===================================================================
  // Data Members
  //===================================================================

  std::unique_ptr<IrContainer> container_;

  friend void swap(IrInterface& a, IrInterface& b) noexcept;
};

// Swap support
void swap(IrInterface& a, IrInterface& b) noexcept;

} // namespace nvfuser
