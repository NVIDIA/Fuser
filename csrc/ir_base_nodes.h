// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <dynamic_type.h>
#include <ir_builder_passkey.h>
#include <type.h>
#include <utils.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
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

namespace nvfuser {

using ValueId = int32_t;

using StmtNameType = unsigned int;

constexpr StmtNameType kInvalidStmName =
    std::numeric_limits<unsigned int>::max();

class NonCopyable;
class PolymorphicBase;
class Fusion;
class FusionGuard;
class Expr;
class Val;
class UnaryOp;
class BinaryOp;
class RNGOp;
class IterDomain;
class IrCloner;
class IrContainer;
class IrBuilderPasskey;
class IrContainerPasskey;

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

TORCH_CUDA_CU_API void swap(Fusion& a, Fusion& b) noexcept;

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
class TORCH_CUDA_CU_API Statement : public NonCopyable, public PolymorphicBase {
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
  virtual c10::optional<ValType> getValType() const {
    return c10::nullopt;
  }
  virtual c10::optional<DataType> getDataType() const {
    return c10::nullopt;
  }

  // Short cut to figure out if it is a value/expression
  bool isVal() const {
    return getValType() != c10::nullopt;
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

//! A Val represents a "value." These are objects, like tensors, scalars, and
//! memory locations, that are inputs and outputs of computations (represented
//! by Exprs, below)
//!
//! Vals are constant and unique and should always be passed
//! around as a pointer. Val can generally be thought of as representing any
//! type of data. Some examples: a constant size like convolution filter width a
//! runtime constant like batch normalizations momentum a "symbolic" tensor like
//! one passed down from the JIT a memory buffer used in device code
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
//!        Int::Int(const Int* src, IrCloner* ir_cloner)
//! 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//! 3) Default mutator function should be added to mutator.cpp
//! 4a) Printing functions should be added to ir_iostream.h/.cpp
//! 4b) Graphviz generation must be added to ir_graphviz.h/.cpp
//! 5) An enum value must be added to ValType in type.h
//! 6) A string entry must be added in val_type_string_map
//!
class TORCH_CUDA_CU_API Val : public Statement {
 public:
  explicit Val(
      IrBuilderPasskey,
      ValType _vtype,
      DataType _dtype = DataType::Null);

  Val(const Val* src, IrCloner* ir_cloner);

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Val*);

  template <typename T>
  static void constDispatch(T handler, const Val* const);

  template <typename T>
  static void mutatorDispatch(T mutator, Val*);

  c10::optional<ValType> getValType() const override {
    return vtype_;
  }

  ValType vtype() const {
    return vtype_;
  }

  DataType dtype() const {
    return dtype_;
  }

  // Throws if no DataType is found. Vals must have a DataType
  c10::optional<DataType> getDataType() const override;

  bool isScalar() const {
    return vtype_ == ValType::Scalar || vtype_ == ValType::NamedScalar;
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

  // If this Val is an integer with a direct constant value associated with it,
  // will return the value of that constant integer. If this integer has
  // defining expressions it will return a c10::nullopt. Those values should be
  // infered using evaluateInt.
  c10::optional<int64_t> getInt() const;

  // If this Val is a double with a direct constant value associated with it,
  // will return the value of that constant double. If this double has
  // defining expressions it will return a c10::nullopt. Those values should be
  // infered using evaluateDouble.
  c10::optional<double> getDouble() const;

  // If this Val is a bool with a direct constant value associated with it,
  // will return the value of that constant bool. If this bool has defining
  // expressions it will return a c10::nullopt. Those values should be infered
  // using evaluateBool.
  c10::optional<bool> getBool() const;

  // If this Val is a constant integer, and its history is comprised only of
  // constant values, will return the value of that constant integer. Cannot
  // make constant as expression evaluator takes non-constant Vals.
  int64_t evaluateInt();

  // If this Val is a constant double, and its history is comprised only of
  // constant values, will return the value of that constant double. Cannot
  // make constant as expression evaluator takes non-constant Vals.
  double evaluateDouble();

  // If this Val is a constant bool, and its history is comprised only of
  // constant values, will return the value of that constant bool. Cannot
  // make constant as expression evaluator takes non-constant Vals.
  bool evaluateBool();

  // Returns if no dependencies and is a constant scalar.
  virtual bool isConst() const {
    return false;
  }

  bool isZero() const;
  bool isZeroInt() const;
  bool isOne() const;
  bool isOneInt() const;

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

  //! Returns true when other is a producer of this
  bool isProducerOf(const Val* other) const;

  //! Returns true when other is a consumer of this
  bool isConsumerOf(const Val* other) const;

  bool sameType(const Statement* other) override {
    return Statement::sameType(other) &&
        getDataType() == other->as<Val>()->getDataType();
  }

  bool sameAs(const Statement* other) const override;

  void setEvaluatorIndex(int to) {
    TORCH_INTERNAL_ASSERT(evaluator_index_ == -1);
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

  void resolveIndexDtype();

  NVFUSER_DECLARE_CLONE

 protected:
  friend Fusion;

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
};

//! A Val object that stores a plain data. Note that this class is only intended
//! to hold non-IR data, such as DataType, std::vector<int>, etc. Please don't
//! use this class to hold IR nodes or their pointers.
template <typename T>
class TORCH_CUDA_CU_API Attribute : public Val {
 public:
  T value;
  Attribute(IrBuilderPasskey passkey, const T& value)
      : Val(passkey, ValType::Attribute), value(value) {}
  Attribute(const Attribute* src, IrCloner* ir_cloner)
      : Val(src, ir_cloner), value(src->value) {}
  template <typename... Args>
  Attribute(IrBuilderPasskey passkey, Args... args)
      : Val(passkey, ValType::Attribute), value(std::forward<Args>(args)...) {}

  NVFUSER_DECLARE_CLONE

  bool sameAs(const Statement* other) const override {
    if (auto pv = dynamic_cast<const Attribute*>(other)) {
      return pv->value == value;
    }
    return false;
  }

  std::string toString(int) const override {
    return Printer<T>::toString(value);
  }

  std::string toInlineString(int) const override {
    return Printer<T>::toString(value);
  }
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
//!  2) dispatch.h/.cpp must be updated to include dispatch of the new Val
//!  3) Default mutator function should be added to mutator.h/.cpp
//!  4) Printing functions should be added to ir_iostream.h/.cpp
//!  5) Lower case convenience functions should be added to arith.h/.cpp (If
//!     user facing)
//!  7) A string entry must be added in expr_type_string_map
//!  8) Entry added to ir_graphviz .cpp/.h
//!
class TORCH_CUDA_CU_API Expr : public Statement {
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

  bool sameAs(const Statement* other) const override;

  virtual std::vector<EvaluatorValue> evaluate(
      const std::vector<EvaluatorValue>& inputs) const;

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

 protected:
  // TODO: Protect based on being in kernel container
  void setPredicate(kir::Predicate* predicate);

  // TODO: Protect based on being in kernel container
  void setWritePredicate(kir::Predicate* write_predicate);

  // TODO: Add Fusion passkey
  void addInput(Val* input) {
    TORCH_INTERNAL_ASSERT(input != nullptr);
    inputs_.push_back(input);
  }

  // TODO: Add Fusion passkey
  void addOutput(Val* output) {
    TORCH_INTERNAL_ASSERT(output != nullptr);
    outputs_.push_back(output);
  }

  // TODO: Add Fusion passkey
  void addAttribute(Statement* attr) {
    attributes_.push_back(attr);
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
    return IrBuilder::create<ClassName>(                   \
        container, inputs, outputs, attributes);           \
  }

// Friends for direct access to split
class TensorDomain;
class ReplayTransformations;
class IndexReferenceReplay;
class ViewTransform;
class Scope;
class IrCloner;
struct AnalyzeViewResult;

// Convenience utility to initialize IterDomain's without having to sort through
// all the default values. Intended to be used with
// IterDomain::IterDomain(IrBuilderPasskey IterDomainBuildArgs)
class TORCH_CUDA_CU_API IterDomainBuilder {
 public:
  // Match legacy constructor
  IterDomainBuilder(Val* _start, Val* _extent);

  // Grab all the parameters from id to set the IterDomainBuilder
  IterDomainBuilder(const IterDomain* id);

  // Resets defaults for rfactor, is padded dim, padded to size, and is mma
  // swizzle which should only be set during scheduling.
  IterDomainBuilder& resetSchedulingParams();

  // Resets is_rfactor_domain
  IterDomainBuilder& resetRfactor();

  IterDomainBuilder& start(Val* _start);
  IterDomainBuilder& extent(Val* _extent);
  IterDomainBuilder& expanded_extent(Val* _expanded_extent);
  IterDomainBuilder& stop_offset(Val* _stop_offset);
  IterDomainBuilder& parallel_type(ParallelType _parallel_type);
  IterDomainBuilder& iter_type(IterType _iter_type);
  IterDomainBuilder& is_rfactor_domain(bool _is_rfactor_domain);
  IterDomainBuilder& is_padded_dimension(bool _is_padded_dimension);
  IterDomainBuilder& padded_to_size(c10::optional<int64_t> _padded_to_size);
  IterDomainBuilder& is_mma_swizzled(bool _is_mma_swizzled);

  IterDomain* build() const;

  // Must have start and extent at least
  IterDomainBuilder() = delete;

  Val* start_ = nullptr;
  Val* extent_ = nullptr;
  Val* expanded_extent_ = nullptr;
  Val* stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;

  // Only relevant at scheduling time or compile time.
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  c10::optional<int64_t> padded_to_size_ = c10::nullopt;
  bool is_mma_swizzled_ = false;
};

//! Simply a representation of an annotated 1D iterable from start to extent.
//! TensorDomains which represent how to iterate over a tensor is made up of
//! IterDomains to form an ND iterable. We directly set parallization strategies
//! on IterDomains.
class TORCH_CUDA_CU_API IterDomain : public Val {
 public:
  IterDomain(IrBuilderPasskey, const IterDomainBuilder& args);

  // Legacy constructor, TODO: should start moving to use IterDomainBuildArgs
  // constructor Same as the above but can set the offset of the stop point
  IterDomain(
      IrBuilderPasskey,
      Val* start,
      Val* extent,
      Val* expanded_extent,
      Val* stop_offset,
      ParallelType parallel_type,
      IterType iter_type,
      bool is_rfactor_domain,
      bool is_padded_dimension,
      c10::optional<int64_t> padded_to_size_,
      bool is_mma_swizzled);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  //! Returns a new IterDomain matching properties of this
  //!
  //! This does NOT copy the is_rfactor_domain flag.
  IterDomain* cloneWithoutRFactor() const;

  //! Clone a vector domains
  static std::vector<IterDomain*> clone(
      const std::vector<IterDomain*>& domains);

  static IterDomain* merge(IterDomain* outer, IterDomain* inner);

  //! start_offset and stop_offset defines partial split. Only root
  //! domains are allowed to have non-zero start and stop offsets.
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      Val* start_offset = nullptr,
      Val* stop_offset = nullptr);

  //! trim_out_of_bounds controls how the values outside start and stop
  //! positions are treated. The option is only valid with root
  //! domains as non-root domains do not have valid start and stop
  //! positions.
  //!
  //! \param trim_out_of_bounds Trims [0, start_] and [-stop_offset_, extent_]
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds);

  //! Resize an IterDomain by expanding both the left and right sides
  //! by given widths. The resulting IterDomain has an extent of
  //! (left_expansion + in->extent() + right_expansion). Note that the
  //! expansion factors can be negative, meaning the input IterDomain
  //! is shrunk. This is the case when resize is used to represent
  //! slice.
  //!
  //! When mark_as_rfactor is true, the output IterDomain
  //! is marked as an rfactor domain. For example, expressions such as
  //! PadOp and SliceOp resize IterDomains and generate rfactor
  //! resized domains.
  static IterDomain* resize(
      IterDomain* in,
      Val* left_expansion,
      Val* right_expansion,
      bool mark_as_rfactor = false);

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isIteration() const {
    return getIterType() == IterType::Iteration;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::Broadcast;
  }

  bool isGatherScatter() const {
    return getIterType() == IterType::GatherScatter;
  }

  bool isGather() const {
    return getIterType() == IterType::Gather;
  }

  bool isStride() const {
    return getIterType() == IterType::Stride;
  }

  bool isVectorComponent() const {
    return getIterType() == IterType::VectorComponent;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  //! Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return isParallelTypeBlockDim(getParallelType());
  }

  //! Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return isParallelTypeThreadDim(getParallelType());
  }

  //! Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  void parallelize(ParallelType t);

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* stop() const;

  Val* stopOffset() const;

  Val* extent() const {
    TORCH_INTERNAL_ASSERT(extent_ != nullptr);
    return extent_;
  }

  bool hasExpandedExtent() const {
    return expanded_extent_ != nullptr;
  }

  // Returns the expanded extent of a strided broadcast entry.
  Val* expandedExtent() const {
    TORCH_INTERNAL_ASSERT(
        hasExpandedExtent(),
        "Requested expanded extent, but none found on this dimension.");
    return expanded_extent_;
  }

  Val* getMaybeExpandedExtent() const {
    if (hasExpandedExtent()) {
      return expandedExtent();
    }
    return extent();
  }

  //! Dimension padding interface:
  //!  2 modes are currently supported:
  //!
  //!   - mode 1: if to_size is given as a positive number,
  //!      the dimension will be padded to the size so that
  //!      this iterdomain will be compile-time constant
  //!      size and it is the scheduler's responsibility
  //!      to ensure no input larger than the padded size
  //!      will be observed
  //!
  //!   - mode 2: if no to_size is given, this dimension
  //!      is "dynamically" padded to next smallest multiple
  //!      of a warp size, i.e. 17 padded to 32, 33 padded to 64
  //!      based on the given input.
  void padToMultipleOfWarp(c10::optional<int64_t> maybe_to_size = {}) {
    // Currently only restricted to TIDx to generate warp reduce
    TORCH_CHECK(
        parallel_type_ == ParallelType::TIDx,
        "padToMultipleOfWarp : warp padding only supported on TIDx parallel dimension");
    is_padded_dimension_ = true;
    if (maybe_to_size.has_value()) {
      if (maybe_to_size.value() > 0) {
        padded_to_size_ = maybe_to_size.value();
      }
    }
  }

  //! Indicates if this iterdomain had padding
  //!  dynamical or statical
  bool hasPaddingToMultipleOfWarp() const {
    return is_padded_dimension_;
  }

  //! Returns a concrete value if this iterdomain
  //!  has been padded to a statical size.
  c10::optional<int64_t> getMaybeSizeAfterPadding() const {
    return padded_to_size_;
  }

  //! True if range of iteration domain isn't across the full extent
  bool maybePartial() const;

  //! Check if IterDomain is a broadcast axis with compile-time
  //! known extent. This is the case with all size-1 IterDomains on
  //! a TensorView's root domain when the TensorView is created.
  bool isImplicitBroadcast() const {
    return isBroadcast() && extent()->isOneInt();
  }

  //! Split for stride by a given factor. It effectively does an inner
  //! split by the factor and sets the inner domain as a Stride
  //! domain.
  std::pair<IterDomain*, IterDomain*> stridedSplit(int factor);

  //! Marks that this id represents a
  //!  instruction loop, mma use only.
  //!
  //! An instruction loop can be considered a generalization of
  //!  vectorization. It also represents a loop that's implemented
  //!  by an instruction and should not be realized by codegen and
  //!  cannot be inlined with.
  //! As an example, if a mma macro, call it mma_eg implements:
  //!  for m in M
  //!    for n in N
  //!      for k in K
  //!         C[m,n] += A[m,k]*B[k,n],
  //! But the generated code should simply be:
  //!  mma_eg(C,A,B)
  //! without the 3 level loopnest, i.e. they're instruction loops.
  //!
  //! In the actual mma macros, the loopnests it implements is a
  //!  transformed version of above to match the mma swizzle.
  //!  So it's different implicit loopnest for different macros.
  //!  WarpMmaSwizzler will label the instruction loops case-by-case.
  bool isMma() const {
    return parallel_type_ == ParallelType::Mma;
  }

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains.
  static std::pair<IterDomain*, IterDomain*> swizzle(
      Swizzle2DType swizzle_type,
      IterDomain* in_x,
      IterDomain* in_y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  bool isMmaSwizzled() const {
    return is_mma_swizzled_;
  }

  //! Used by WarpMmaSwizzler, this is an utility for WarpMmaSwizzler
  //!  to lock the thread swizzled iterdomains.
  //! Only true for the iterdomains produced by WarpMmaSwizzler.
  //! Mma ops require specific swizzle patterns
  //!  and this label utility is to prevent any further transform on the
  //!  iterdomains involved in the swizzle so that the pattern remain correct in
  //!  generated code.
  //!
  //! Note:
  //!    Used only through WarpMmaSwizzler only and mma validation relies on
  //!    this
  //!  flag being set on the correct iterdomains.
  void toMmaSwizzled() {
    is_mma_swizzled_ = true;
  }

 protected:
  friend TensorDomain;
  friend ReplayTransformations;
  friend IndexReferenceReplay;

 private:
  //! Valid range is defined as [start:-stop_offset]
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;

  // Broadcast dimensions are assumed to be size 1 for the sake of code
  // generation. If a user though calls `expand` on a tensor that dimension is
  // still considered a broadcast dimension. However if we ever output that
  // dimension it should be a size dictated by the `expand` operation, and have
  // a stride of zero. Since this extent is important to track, but not
  // necessarily generate code for (still want loops on broadcast to be of size
  // 0), we simply store it separately from extent_. Having an expanded_extent_
  // is only allowed with broadcasted dimsneions. Only in this instance does it
  // make sense to have an expanded_extent_, because it's used when users are
  // expecting return tensors to have a physical domain. If a user simply
  // "broadcasts" an operation
  Val* const expanded_extent_ = nullptr;

  //! Distance of stop from the end
  Val* const stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  c10::optional<int64_t> padded_to_size_ = c10::nullopt;

  //! Tracks if this id represents a thread swizzled loop or
  //!   models an implicit loop within instructions. Should not make
  //!   any changes once an id is warp mapped.
  bool is_mma_swizzled_ = false;
};

//! TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
//! logical axis in its associated tensor. TensorDomain does not directly hold
//! the Tensor it is associated with, and in theory could be associated with
//! multiple tensors. TensorDomain's primary responsibility is to provide a
//! mechanism to access history of transformations that were used to generate
//! it. This is done through the normal interaction of Expr/Val in Fusion. i.e.
//! if we want to know the previous operation generating a particular
//! TensorDomain we can simply call:
//!
//!     FusionGuard::getCurFusion()->definition(a_tensor_domain)
//!
//! which should give us an operation in the list [split, merge] or similar
//! operations that take in a TensorDomain, applies a transformation and outputs
//! a tensor domain.
class TORCH_CUDA_CU_API TensorDomain : public Val {
 public:
  explicit TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> rfactor_domain,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> rfactor_domain,
      std::vector<IterDomain*> allocation,
      std::vector<IterDomain*> leaf_domain,
      std::vector<std::optional<bool>> contiguity = {});

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  NVFUSER_DECLARE_CLONE

  bool operator==(const TensorDomain& other) const;
  bool operator!=(const TensorDomain& other) const {
    return !(*this == other);
  }

  std::vector<IterDomain*>::size_type nDims() const {
    return leaf_domain_.size();
  }

  bool sameAs(const Statement* other) const override;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  // Note: [Contiguity]
  // Contiguity is a vector of optional<bool> which has the same number of
  // elements as rfactor_domain_. The contiguity of a broadcast dimension is
  // meaningless, so it has to be nullopt. The contiguity of a non-broadcasting
  // dimension is true if and only if it is memory dense with the next
  // non-broadcasting dimension.
  // For example, if I have a tensor torch.zeros(4, 1, 3).expand(-1, 10, -1),
  // the contiguity will be (true, nullopt, true), which means 4 is memory dense
  // with 3.
  const std::vector<std::optional<bool>>& contiguity() const {
    return contiguity_;
  }

  void setContiguity(const std::vector<std::optional<bool>>& contig);

  std::string getContiguityString() const {
    std::stringstream ss;
    bool first = true;
    for (auto b : contiguity()) {
      if (!first) {
        ss << " ";
      }
      first = false;
      ss << (b.has_value() ? (*b ? "t" : "f") : "n");
    }
    return ss.str();
  }

  bool hasReduction() const {
    return has_reduction_;
  }

  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasGridBroadcast() const;

  bool hasBroadcast() const {
    return no_bcast_domain_.size() != leaf_domain_.size();
  }

  bool hasRFactor() const {
    return !rfactor_domain_.empty();
  }

  bool hasAllocation() const {
    return !allocation_domain_.empty();
  }

  // Returns if rfactor domain only consists of id's of iter type.
  bool hasViewLikeRFactor() const;

  bool hasVectorize() const;

  bool hasSymbolicAxis() const;

  c10::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& root() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& rfactor() const {
    return rfactor_domain_;
  };

  const std::vector<IterDomain*>& allocation() const {
    return allocation_domain_;
  }

  const std::vector<IterDomain*>& leaf() const {
    return leaf_domain_;
  }

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& maybeRFactor() const {
    return hasRFactor() ? rfactor() : root();
  }

  const std::vector<IterDomain*>& maybeAllocation() const {
    return hasAllocation() ? allocation_domain_ : maybeRFactor();
  };

  // Set the allocation domain of this TensorDomain. The new allocation domain
  // must satisfy root <= allocation <= leaf, that is, it must be within the
  // history between root and leaf domain. Because contiguity is always defined
  // w.r.t. the allocation domain, the contiguity must be updated accordingly.
  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      std::vector<std::optional<bool>> new_contiguity);

  // Similar to the previous one, but with new contiguity filled with all true
  // or all false.
  void setAllocationDomain(
      std::vector<IterDomain*> new_allocation_domain,
      bool new_contiguity) {
    auto contiguity_flags =
        getContiguityFilledWith(new_allocation_domain, new_contiguity);
    setAllocationDomain(
        std::move(new_allocation_domain), std::move(contiguity_flags));
  }

  void resetDomains() {
    no_reduction_domain_ = noReductions(leaf_domain_);
    no_bcast_domain_ = noBroadcasts(leaf_domain_);
    has_reduction_ = hasReduction(leaf_domain_);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  int64_t posOf(IterDomain* id) const;

  //! Returns a position of a root domain
  int64_t rootPosOf(IterDomain* id) const;

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  void split(
      int axis_,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds = false);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int axis_o, int axis_i);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int, int>& old2new);

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains contained in this domain.
  void swizzle(
      Swizzle2DType swizzle_type,
      int x,
      int y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // Transform TensorView according to merge and split transformations
  TensorDomain* view(const AnalyzeViewResult& view_analysis);

  TensorDomain* flatten(int64_t start_dim, int64_t end_dim);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // Get a vector whose size is the number of IDs in the given rfactor_domain
  // filled with fill_value or nullopt depending on whether its corresponding ID
  // is broadcast.
  static std::vector<std::optional<bool>> getContiguityFilledWith(
      const std::vector<IterDomain*>& rfactor_domain,
      bool fill_value);

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(const std::vector<int>& axes);

 private:
  const std::vector<IterDomain*> root_domain_;
  const std::vector<IterDomain*> rfactor_domain_;
  std::vector<IterDomain*> allocation_domain_;
  std::vector<IterDomain*> leaf_domain_;

  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  std::vector<std::optional<bool>> contiguity_;
  bool has_reduction_;
};

} // namespace nvfuser
