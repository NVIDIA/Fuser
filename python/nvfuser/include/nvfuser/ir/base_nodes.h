// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/core/ScalarType.h>
#include <exceptions.h>

#include <ir/builder_passkey.h>
#include <polymorphic_value.h>
#include <type.h>
#include <utils.h>
#include <visibility.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// TODO: Add more types (int32, int64)
// TODO: sameAs should have better logic to check against any type and return
// gracefully

/*
 * This file defines the base IR structure. Any IR node in this system will
 * inherit from one of the following classes: Statement, Expr, Val,
 * IrInputOutput IR is any information that the code generation stack may need
 * for analysis. By analysis we're refering to anything done in response to a
 * user facing call of this stack. This could be careful tracking of user calls,
 * and any transformation including optimizing transformations, user declared
 * transformations, and lowering the IR.
 */

//! IR header hierarchy
//! 1. utils.h - PolymorphicBase and NonCopyable
//! 2. ** ir/base_nodes.h ** - Statement, Expr, and Val
//! 3. ir/internal_base_nodes.h - IterDomain and TensorDomain
//! 4. ir/interface_nodes.h - TensorView and Scalar
//! 5. ir/internal_nodes.h - Any internal-only IR nodes

namespace nvfuser {

using ValueId = int32_t;

using StmtNameType = unsigned int;

constexpr StmtNameType kInvalidStmName =
    std::numeric_limits<unsigned int>::max();

class Fusion;
class Expr;
class Val;
class IrCloner;
class IrContainer;
class IrBuilderPasskey;
class IrContainerPasskey;
class ExpressionEvaluator;

namespace kir {
class Kernel;
class Predicate;
} // namespace kir

// Passkey for container to register names with statements
class ExprPasskey {
  friend class Expr;

 private:
  explicit ExprPasskey() = default;
};

#define NVFUSER_DECLARE_CLONE \
  virtual Statement* clone(IrCloner* ir_cloner) const override;

#define NVFUSER_DEFINE_CLONE(ClassName)                    \
  Statement* ClassName::clone(IrCloner* ir_cloner) const { \
    return IrBuilder::clone(this, ir_cloner);              \
  }

//! Statement is the highest level node representation. Everything that is
//! considered "IR" will be derived from this class at some point. Both Values
//! and Expr's are a Statement. If there will ever be any more fundamental
//! types, they will also derive from Statement.
//!
//! We use Statements to pass around nodes of unknown compile type. Therefore it
//! is also important for the design to have a dispatch system for a Statment.
//! Basically beinng able to succienctly traverse down the inhereitance stack of
//! a Statment at runtime. This is currently implemented in dispatch.h
class NVF_API Statement : public NonCopyable, public PolymorphicBase {
  friend void swap(Fusion&, Fusion&) noexcept;
  friend void swap(IrContainer& a, IrContainer& b) noexcept;

 public:
  Statement() = delete;

  // Cloning constructor
  Statement(const Statement* src, IrCloner* ir_cloner);

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Statement*);

  template <typename T>
  static void constDispatch(T handler, const Statement* const);

  template <typename T>
  static void mutatorDispatch(T mutator, Statement*);

  // Accessor functions to types. Vals always have a DataType, Exprs never do
  virtual std::optional<ValType> getValType() const {
    return std::nullopt;
  }
  virtual std::optional<DataType> getDataType() const {
    return std::nullopt;
  }

  // Short cut to figure out if it is a value/expression
  bool isVal() const {
    return getValType() != std::nullopt;
  }
  bool isExpr() const {
    return isA<Expr>();
  }

  // Make sure this is a Val and return it as a Val*
  Val* asVal();

  // Make sure this is an Expr and return it as an Expr*
  Expr* asExpr();

  // Return the fusion this statement belongs to
  Fusion* fusion() const;

  // Return the kernel this statement belongs to
  kir::Kernel* kernel() const;

  // Return the container this statement belongs to
  IrContainer* container() const {
    return ir_container_;
  }

  // Return the int that represents its name
  StmtNameType name() const {
    return name_;
  }

  // Set the statements' name. Typically the container will set the name,
  // however if we're dealing with cloning, IrBuilder will set the name, this
  // maybe should be from IrCloner, however I didn't want to add another
  // passkey.
  void setName(IrContainerPasskey, StmtNameType name);
  void setName(IrBuilderPasskey, StmtNameType name);

  virtual bool sameType(const Statement* const other) {
    return typeid(*this) == typeid(*other);
  }

  // Return if this statement is the same as another statement
  // TODO: should this run through dispatch on this and other?
  virtual bool sameAs(const Statement* other) const {
    return this == other;
  }

  static bool lessThan(const Statement* stmt1, const Statement* stmt2);

  virtual std::string toString(int indent_size = 0) const;

  virtual std::string toInlineString(int indent_size = 0) const;

  virtual Statement* clone(IrCloner* ir_cloner) const;

 protected:
  Statement(IrBuilderPasskey);

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  StmtNameType name_ = kInvalidStmName;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  IrContainer* ir_container_ = nullptr;
};

inline std::string toString(Statement* stmt) {
  return stmt->toString();
}

//! A Val represents a "value." These are objects, like tensors, scalars, and
//! memory locations, that are inputs and outputs of computations (represented
//! by Exprs, below)
//!
//! Vals are constant and unique and should always be passed
//! around as a pointer. Val can generally be thought of as representing any
//! type of data. Some examples:
//!   * a constant size like convolution filter width
//!   * a runtime constant like batch normalizations momentum
//!   * a "symbolic" tensor like one passed down from the JIT
//!   * a memory buffer used in device code
//!
//! Adding a Val:
//! Right now adding a Val is quite involved. Val's can be defined in ir.h or in
//! their own header file. The following is what is currently needed to add a
//! new Val:
//!
//! 1) Definition inheriting from Val
//!     - Members must be private or protected
//!     - Accessor functions for members
//!     - Must call Val constructor, Val constructor registers with fusion
//!     - Implementation of bool sameAs(...)
//!     - Must implement a "cloning" constructor, ex.
//!        Scalar::Scalar(const Val* src, IrCloner* ir_cloner)
//! 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//! 3) Default mutator function should be added to mutator.cpp
//! 4a) Printing functions should be added to ir/iostream.h/.cpp
//! 4b) Graphviz generation must be added to ir/graphviz.h/.cpp
//! 5) An enum value must be added to ValType in type.h
//! 6) A string entry must be added in val_type_string_map
//!
class NVF_API Val : public Statement {
 public:
  // When we create a Val we immediately register them with the active fusion.
  explicit Val(
      IrBuilderPasskey passkey,
      ValType _vtype,
      DataType _dtype = DataType::Null,
      PolymorphicValue _value = std::monostate{})
      : Statement(passkey),
        vtype_(_vtype),
        dtype_(std::move(_dtype)),
        value_(std::move(_value)) {
    if (value_.hasValue()) {
      NVF_CHECK(
          hasCompatibleDataType(value_, dtype_),
          "Scalar value is not compatible with the given data type ",
          dtype_,
          " for value ",
          PolymorphicValue_functions::toString(value_));
    }
  }
  explicit Val(IrBuilderPasskey passkey, DataType dtype)
      : Val(passkey, ValType::Others, std::move(dtype)) {}
  explicit Val(IrBuilderPasskey passkey, PrimDataType dtype)
      : Val(passkey, ValType::Others, DataType(dtype)) {}
  explicit Val(IrBuilderPasskey passkey, PolymorphicValue value)
      : Val(passkey, ValType::Others, nvfuser::getDataType(value), value) {}
  explicit Val(IrBuilderPasskey passkey, PolymorphicValue value, DataType dtype)
      : Val(passkey,
            ValType::Others,
            dtype,
            castToDtype(std::move(value), dtype)) {}

  // NOTE: we don't clone the definition_ and uses_ here
  //  since they may introduce cloning cycles. Instead, we copy
  //  the original pointers and we'll fix them up later part of the
  //  Fusion copy. Neither definition_ nor uses_ are copied through
  //  this constructor now leaving them to be resolved by later stages
  //
  Val(const Val* src, IrCloner* ir_cloner)
      : Statement(src, ir_cloner),
        vtype_(src->vtype_),
        dtype_(src->dtype_),
        value_(src->value_) {}

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Val*);

  template <typename T>
  static void constDispatch(T handler, const Val* const);

  template <typename T>
  static void mutatorDispatch(T mutator, Val*);

  std::optional<ValType> getValType() const override {
    return vtype_;
  }

  ValType vtype() const {
    return vtype_;
  }

  DataType dtype() const {
    return dtype_;
  }

  const PolymorphicValue& value() const {
    return value_;
  }

  PolymorphicValue& value() {
    return value_;
  }

  bool isSymbolic() const {
    return !value_.hasValue();
  }

  // Throws if no DataType is found. Vals must have a DataType
  std::optional<DataType> getDataType() const override;

  bool isScalar() const {
    return vtype_ == ValType::Others || vtype_ == ValType::NamedScalar;
  }

  // Returns if all dependencies are constant scalars
  bool isConstScalar() const;

  // Returns if all dependencies are constant integers
  bool isConstInt() const;

  bool isIntegralScalar() const {
    return isScalar() && isIntegralType(dtype_);
  }

  bool isFloatingPointScalar() const {
    return isScalar() && isFloatingPointType(dtype_);
  }

  bool isABool() const {
    return isScalar() && dtype_ == DataType::Bool;
  }

  // If this Val's history is comprised only of constant values, will return a
  // PolymorphicValue. Cannot make constant as expression evaluator takes
  // non-constant Vals.
  PolymorphicValue evaluate();

  // Returns if no dependencies and is a constant scalar.
  virtual bool isConst() const {
    return value_.hasValue() && definition() == nullptr;
  }

  bool isZero() const;
  bool isZeroInt() const;
  bool isOne() const;
  bool isOneInt() const;
  bool isTrue() const;
  bool isFalse() const;

  // Returns the Expr that this value is an output of, returns nullptr if none
  // was found
  Expr* definition() const {
    if (is_fusion_input_) {
      return nullptr;
    }
    return definition_;
  }

  // Determine if value definition matches given expression type
  template <typename T>
  inline bool isDefinitionType() const;

  //! Returns the Exprs for which this is an input.
  //! Note that uses() will occasionally trigger a deferred call to
  //! resetTvUses() which can be expensive as it requires traversing the graph
  //! using Val definitions.
  const std::vector<Expr*>& uses() const;

  bool isFusionInput() const {
    return is_fusion_input_;
  }

  bool isFusionOutput() const {
    return is_fusion_output_;
  }

  bool sameType(const Statement* other) override {
    return Statement::sameType(other) &&
        getDataType() == other->as<Val>()->getDataType();
  }

  bool sameAs(const Statement* other) const override;

  void setEvaluatorIndex(int to) {
    // Only allow resetting evaluator_index to -1 OR
    // setting evaluator_index if it isn't in-use
    NVF_ERROR(evaluator_index_ == -1 || to == -1);
    evaluator_index_ = to;
  }

  int evaluatorIndex() const {
    return evaluator_index_;
  }

  // Following is managed by Fusion (or kirIrBuilder) and can change.
  // TODO: Protect with a passkey.
  void setDefinition(Expr* expr) {
    definition_ = expr;
  }

  NVFUSER_DECLARE_CLONE

 protected:
  friend class Fusion;
  friend class IrContainer;

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const ValType vtype_;

  // TODO: Add fusion passkey for this
  void setIsFusionInput(bool is_fusion_input) {
    is_fusion_input_ = is_fusion_input;
  }

  // TODO: Add fusion passkey for this
  void setIsFusionOutput(bool is_fusion_output) {
    is_fusion_output_ = is_fusion_output;
  }

  // TODO: Add fusion or container passkey for this
  void setUses(const std::vector<Expr*>& uses) {
    uses_ = uses;
  }

  //! Insert a new expression into uses() if it is not already present and
  //! return whether an insertion occurred.
  bool addUse(Expr*);

  //! Remove an expression from uses() if it is already present and return
  //! whether a removal occurred.
  bool removeUse(Expr*);

 private:
  // There's only one instance where dtype can change, and that's through
  // resolving the index data type from nvfuser to either Int or Int32 for
  // welford operations.
  DataType dtype_;

  // Following is managed by Fusion and can change.
  bool is_fusion_input_ = false;
  bool is_fusion_output_ = false;

  Expr* definition_ = nullptr;
  std::vector<Expr*> uses_;

  // Expr evaluator idx;
  int evaluator_index_ = -1;

  // The concrete value of this Val. This is only used for constant Vals.
  // Depending on the actual type of the Val, the allowed types of the
  // value_ can be different. For example, for a TensorView, the value_ must be
  // a at::Tensor, while for IterDomain, the value_ must be std::monostate{}.
  PolymorphicValue value_;
};

using newObjectFuncType = Expr*(
    IrContainer*,
    std::vector<Val*>,
    std::vector<Val*>,
    std::vector<Statement*>);

//!  A Expr represents a "computation." These are functions that takes inputs
//!  and produce outputs, inputs and outputs all being Vals. There are
//!  specializations of BinaryOp which takes 2 inputs and produces 1 output, and
//!  UnaryOp which takes 1 input and produces 1 output. Exprs are unique and
//!  immutable. Conceptually, Exprs could always be manipulated using unique
//!  pointers, and we could add this later. However, for now Exprs can be
//!  replaced in a fusion, but they cannot be modified in place.
//!
//!  The IR is static single assignment (SSA). Values can only be defined as an
//!  output of an Expr once. If they are re-defined the original definition is
//!  deleted from the program, as opposed to an ordered redefinition of the
//!  value in the program.
//!
//!  Note: Registering an Expr with a Fusion is actually 2 parts, one part is
//!  done in the Expr constructor, so that should be called on anything that
//!  inherits Expr. The issue with having registration in Expr's constructor, is
//!  that the constructor of an Expr will set ouputs and inputs. This
//!  information is important for registration with Fuser, so it can track the
//!  dependency chain.
//!
//!  Adding an Expr:
//!  Right now adding an Expr is quite involved. Expr's can be defined in ir.h
//!  or in their own header file. The following is what is currently needed for
//!  Expr definitions:
//!
//! 1) Definition inheriting from Expr.
//!      - Members must be private or protected
//!      - Accessor functions for members
//!      - Constructors need to register with the Fusion after inputs/outputs
//!         are defined
//!      - Implementation of bool sameAs(...)
//!  2) dispatch.h/.cpp must be updated to include dispatch of the new Expr
//!  3) Default mutator function should be added to mutator.h/.cpp
//!  4) Printing functions should be added to ir/iostream.h/.cpp
//!  5) Lower case convenience functions should be added to arith.h/.cpp (If
//!     user facing)
//!  7) A string entry must be added in expr_type_string_map
//!  8) Entry added to ir_graphviz .cpp/.h
//!
class NVF_API Expr : public Statement {
 public:
  explicit Expr(IrBuilderPasskey);

  Expr(const Expr* src, IrCloner* ir_cloner);

  Expr(
      IrBuilderPasskey,
      std::vector<Val*> inputs,
      std::vector<Val*> outputs,
      std::vector<Statement*> attributes);

  virtual newObjectFuncType* newObjectFunc() const = 0;

  // Creates a new instance of the expression with all its field copied.
  // Note that unlike IrCloner, this function only do a shallow copy
  Expr* shallowCopy() const;

  // Check that if this and other are the same operator. This main difference
  // from sameAs is that sameOp does not check the inputs.
  virtual bool sameOp(const Expr* other) const;

  bool sameAs(const Statement* other) const override;

  virtual std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const;

  // This version allows evaluation of multiple ops together instead of one op
  // at a time by overriding and skipping computation of intermediate inputs
  // that are not required. For example:
  // 1. CatOp is internally preceded by PadOp but the ATen evaluation uses only
  // the unpadded inputs and the evaluation of padded inputs can be skipped.
  // 2. Evaluating patterns in matmul fallback such as MmaOp + Cast/ MmaOp +
  // Bias + Cast
  virtual std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      std::unordered_map<const Val*, PolymorphicValue>& known_values) const;

  // Input/output accessors
  const auto& inputs() const {
    return inputs_;
  }

  const auto& outputs() const {
    return outputs_;
  }

  const auto& attributes() const {
    return attributes_;
  }

  auto input(size_t index) const {
    return inputs_.at(index);
  }

  auto output(size_t index) const {
    return outputs_.at(index);
  }

  auto attribute(size_t index) const {
    return attributes_.at(index);
  }

  auto attributeVal(size_t index) const {
    return dynamic_cast<Val*>(attributes_.at(index));
  }

  template <typename T>
  T& attribute(size_t index) const;

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Expr*);

  template <typename T>
  static void constDispatch(T handler, const Expr* const);

  // TODO: Protect based on being in kernel container
  kir::Predicate* predicate() const;

  // Creates a shallow copy the expression with the given predicate attached.
  // TODO: Protect based on being in kernel container
  Expr* withPredicate(kir::Predicate* predicate);

  // TODO: Protect based on being in kernel container
  kir::Predicate* writePredicate() const;

  // Creates a shallow copy the expression with the given write-predicate
  // attached.
  // TODO: Protect based on being in kernel container
  Expr* withWritePredicate(kir::Predicate* write_predicate);

  // Get the name of an expression
  virtual const char* getOpString() const = 0;

  // Get the label for Graphviz
  virtual std::string getGraphvizLabel() const;

  //! Perform assertions on new_val to ensure that it is valid for this
  //! particular expression. This ensures that invalid values are not propagated
  //! through the graph during concretization.
  virtual void checkConcretization(Val* old_val, Val* new_val) const;

 protected:
  // TODO: Protect based on being in kernel container
  void setPredicate(kir::Predicate* predicate);

  // TODO: Protect based on being in kernel container
  void setWritePredicate(kir::Predicate* write_predicate);

  // TODO: Add Fusion passkey
  void addInput(Val* input) {
    NVF_ERROR(input != nullptr);
    inputs_.push_back(input);
  }

  // TODO: Add Fusion passkey
  void addOutput(Val* output) {
    NVF_ERROR(output != nullptr);
    outputs_.push_back(output);
  }

  // TODO: Add Fusion passkey
  void addAttribute(Statement* attr) {
    attributes_.push_back(attr);
  }

  // TODO: Add Fusion passkey
  void addDataAttribute(PolymorphicValue attr);

  // TODO: Add Fusion passkey
  template <typename T>
  void addDataAttribute(T attr) {
    if constexpr (PolymorphicValue::is_candidate_type<T>) {
      addDataAttribute(PolymorphicValue(std::move(attr)));
    } else {
      addDataAttribute(Opaque(std::move(attr)));
    }
  }

  ExprPasskey exprPasskey() {
    return ExprPasskey();
  }

  std::vector<Statement*> attributes_;

 private:
  std::vector<Val*> inputs_;
  std::vector<Val*> outputs_;
  kir::Predicate* predicate_ = nullptr;

  // Only used for reduction-related expressions
  kir::Predicate* write_predicate_ = nullptr;
};

template <typename T>
bool Val::isDefinitionType() const {
  if (definition() != nullptr) {
    return definition()->isA<T>();
  }
  return false;
}

#define NVFUSER_DECLARE_CLONE_AND_CREATE                        \
  virtual Statement* clone(IrCloner* ir_cloner) const override; \
  static Expr* newObject(                                       \
      IrContainer* container,                                   \
      std::vector<Val*> inputs,                                 \
      std::vector<Val*> outputs,                                \
      std::vector<Statement*> attributes);                      \
  virtual newObjectFuncType* newObjectFunc() const override {   \
    return newObject;                                           \
  }

#define NVFUSER_DEFINE_CLONE_AND_CREATE(ClassName)         \
  Statement* ClassName::clone(IrCloner* ir_cloner) const { \
    return IrBuilder::clone(this, ir_cloner);              \
  }                                                        \
  Expr* ClassName::newObject(                              \
      IrContainer* container,                              \
      std::vector<Val*> inputs,                            \
      std::vector<Val*> outputs,                           \
      std::vector<Statement*> attributes) {                \
    return IrBuilder::createInContainer<ClassName>(        \
        container, inputs, outputs, attributes);           \
  }

} // namespace nvfuser
