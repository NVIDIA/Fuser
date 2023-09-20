// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <device_lower/lower2device.h>
#include <exceptions.h>
#include <executor_params.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <polymorphic_value.h>
#include <utils.h>

#include <c10/core/DeviceType.h>

namespace nvfuser {

class PrecomputedValues;
class KernelArgumentHolder;
struct TensorArgAbstract;

//! NaiveValueMachine:
//!  This is an un-optimized runtime for evaluating a
//!   set of values in one run. The runtime contains
//!   a vector of instructions inferred from IR at compile-time
//!   and it currently must be associated with an instance of
//!   PrecomputedValues that will provide the workspace
//!   containing the concrete values for the values.
class NaiveValueMachine {
  //! The generic types of instructions supported for this machine.
  enum class InstructionType { UNARY_OP, BINARY_OP, TERNARY_OP, SET_OP };

 public:
  //! Constructor lowers all the expr IR nodes stored in precomputed_values
  //!  and stores them in the private state.
  NaiveValueMachine(PrecomputedValues& precomputed_values);

  //! Copy all values other than `precomputed_values_` from other
  //! This would be better implemented as a copy constructor, except that would
  //! also presumably bind precomputed_values_ which we could not then rebind,
  //! as we need to during cloning.
  void copyFrom(const NaiveValueMachine& other);

  //! Runs all the instructions and write results to the associated
  //!  precomputed_values.
  void run();

 private:
  //! Convert an unary IR expr to an instruction
  void makeUnaryOp(UnaryOp* uop);

  //! Convert an binary IR expr to an instruction
  void makeBinaryOp(BinaryOp* bop);

  //! Convert an ternary IR expr to an instruction
  void makeTernaryOp(TernaryOp* bop);

  //! Create an empty instruction with all default values
  //!  and place it at the end of the instruction buffer.
  int makeInstructionEntry();

  //! Run a single instruction at the given index of
  //!  the instruction buffer. Decodes and dispatches
  //!  to the corresponding instruction handle functions.
  void runInstruction(int index);

  //! Runs a unary operation at given index of instruction buffer
  void runUnaryOp(int index);

  //! Runs a binary operation at given index of instruction buffer
  void runBinaryOp(int index);

  //! Runs a ternary operation at given index of instruction buffer
  void runTernaryOp(int index);

 private:
  friend PrecomputedValues;

  //! Reference to the PrecomputedValues workspace associated with
  //!   this runtime. All the instructions will read and write the
  //!   values in this workspace.
  PrecomputedValues& precomputed_values_;

  //! Instruction buffer. All states are in separate vectors and
  //!  the entry of each vector at the same index correspond to
  //!  the same instruction.

  //! Total number of instructions
  int num_of_instructions_ = 0;

  //! Machine instruction type for each instruction i.e.
  //!  unary or binary
  std::vector<InstructionType> inst_type_;

  //! Unary operator type if applicable, contains a default
  //!  value at each index corresponding to a binary op.
  std::vector<UnaryOpType> uop_type_;

  //! Data type for unary op of type UnaryOpType::Cast, contains a default
  //!  value at each index corresponding other ops.
  std::vector<DataType> data_type_;

  //! Binary operator type if applicable, contains a default
  //!  value at each index corresponding to a binary op.
  std::vector<BinaryOpType> bop_type_;

  //! Ternary operator type if applicable, contains a default
  //!  value at each index corresponding to a ternary op.
  std::vector<TernaryOpType> top_type_;

  //! Indexes of operands and destination of each instruction.
  //!  The indexes corresponds to positions in the workspace
  //!  where concrete values are hosted.

  //! Operand 0 of each instruction.
  std::vector<int> src0_;

  //! Operand 1 of each instruction, a default value at
  //!  each index corresponding to a unary op.
  std::vector<int> src1_;

  //! Operand 2 of each instruction, a default value at
  //!  each index corresponding to a unary or binary op.
  std::vector<int> src2_;

  //! Destination of each instruction.
  std::vector<int> dest_;
};

//! PrecomputedValues:
//!  A class to support optimized evaluation of values
//!  at runtime.
//!    At compile time all necessary values are collected
//!  from given IR nodes and a runtime and a workspace containing
//!  the concrete values is created and pre-allocated.
//!    At runtime the value vm is used to evaluate all the
//!  values and store them in the workspace ahead of time.
class PrecomputedValues {
 public:
  PrecomputedValues() = delete;

  explicit PrecomputedValues(Fusion* fusion);

  //! Bind concrete values from fusion runtime inputs
  void bindInputs(const KernelArgumentHolder& args);

  using ParallelExtentMap =
      std::unordered_map<ParallelType, std::vector<const Val*>>;

  //! Bind concrete values from launch constraints
  void bindParallelExtents(
      const ParallelExtentMap& parallel_extents,
      const LaunchParams& launch_constraint);

  //! Bind the NamedScalars corresponding to the
  //!  concrete parallel dimension sizes after the
  //!  actual value has been resolved.
  void bindConcreteParallelTypeValue(ParallelType pt, PolymorphicValue value);

  //! Returns if the workspace contains evaluated results.
  bool ready() {
    return has_valid_values_;
  }

  //! Runs the internal value machine that will compute
  //!  the values allocated in the workspace.
  void evaluate();

  //! Returns value for the given IR node if it's stored
  //!  in the workspace and has been evaluated.
  const PolymorphicValue& getMaybeValueFor(const Val* val) const;

  //! Debugging helper, prints all the currently known values
  void print() const;

  PrecomputedValues clone(IrCloner& ir_cloner) const;

 protected:
  // Fusion IR associated with the precomputed values. Can be kir::Kernel or
  // Fusion.
  Fusion* fusion_ = nullptr;

  //! Contains all the named scalars correspond
  //!  to thread size of each parallel type.
  std::unordered_map<ParallelType, std::unique_ptr<std::vector<int>>>
      thread_dim_value_indices_;

  //! Initialize the workspace before first use.
  //!  Assume the given value list IR nodes have
  //!  been topologically sorted.
  void initializeValueList(const std::vector<Val*>& sorted_value_list);

  //! Bind concrete value to the given index
  //!  if the index is valid.
  void bindValue_(int index, const PolymorphicValue& value) {
    if (index < 0 || is_constant_[index]) {
      return;
    }
    defined_[index] = true;
    values_[index] = value;
    binding_log_.emplace_back(index, value);
  }
  template <typename T>
  void bindValue(int index, const T& value) {
    bindValue_(index, PolymorphicValue(value));
  }

  //! Invalidate all computed values in the workspace.
  void invalidate();

  //! Interface for subclasses to access symbols_
  void loadSymbols(std::vector<Val*> symbols) {
    symbols_ = std::move(symbols);
  }

  //! Interface for subclasses to access symbols_
  std::vector<Val*>& symbols() {
    return symbols_;
  }

  //! Initialize the value runtime that will
  //!  infer instructions from the workspace.
  void initializeIntegerMachine() {
    value_machine_ = std::make_unique<NaiveValueMachine>(*this);
  }

  bool hasValidValues() {
    return has_valid_values_;
  }

  //! Iterate through all the named scalars corresponding
  //!  to thread sizes and pre-group them by their parallel
  //!  types.
  void initializeNamedScalars();

 private:
  //! Post evaluation check, throws if any computed value
  //!  is inconsistent with its bound value
  void validate();

  //! Returns true if workspace has a computed or constant
  //!  value for given index.
  bool hasValue(int index) {
    NVF_ERROR(index > 0);
    return defined_[index] || is_constant_[index];
  }

  void bindTensorMetaData(TensorView* tv, const at::Tensor& tensor);

 private:
  friend NaiveValueMachine;

  //! Marks if an evaluation has finished
  bool has_valid_values_ = false;

  //! The size of workspace
  int num_of_values_ = -1;

  //! Marks if a value has been bound or
  //!  computed at each index.
  std::vector<bool> defined_;

  //! Marks if a value is compile-time constant
  //!  at each index.
  std::vector<bool> is_constant_;

  //! Stores the concrete values at each index.
  std::vector<PolymorphicValue> values_;

  //! Use a single monostate to represent null, instead of creating a new
  //! PolymorphicValue for each null.
  PolymorphicValue null_ = std::monostate{};

  //! Stores the IR nodes corresponding to each index.
  std::vector<Val*> symbols_;

  //! An internal log to keep track of all the bindings
  //!  used in each evaluation cycle. To be used for
  //!  consistency check.
  std::vector<std::pair<int, PolymorphicValue>> binding_log_;

  //! Integer runtime for realizing the values computations.
  std::unique_ptr<NaiveValueMachine> value_machine_;
};

} // namespace nvfuser
