// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/all_nodes.h>
#include <ir/builder_passkey.h>
#include <utils.h>

namespace nvfuser {

namespace kir {
class Kernel;
}

class IrCloner;

//! IR builder interface
class TORCH_CUDA_CU_API IrBuilder {
 public:
  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(Args&&... args) {
    auto container = FusionGuard::getCurFusion();
    // return create<T>(container, std::forward<Args>(args)...);
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(IrContainer* container, Args&&... args) {
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Clone an IR node, forwarding the arguments to the IrCloner constructor.
  //! Register clones with IrCloner's target container.
  template <class T>
  static T* clone(const T* src, IrCloner* ir_cloner);

  // Unary operations
  static Scalar* negExpr(Val* val);
  static Scalar* notExpr(Val* val);
  static Scalar* absExpr(Val* val);
  static Scalar* setExpr(Val* val);
  static NamedScalar* setExprNamedScalar(const std::string& name, Val* val);
  static NamedScalar* addressExprNamedScalar(const std::string& name, Val* val);

  // Binary operations
  static Scalar* andExpr(Val* lhs, Val* rhs);
  static Scalar* orExpr(Val* lhs, Val* rhs);
  static Scalar* eqExpr(Val* lhs, Val* rhs);
  static Scalar* neExpr(Val* lhs, Val* rhs);
  static Scalar* gtExpr(Val* lhs, Val* rhs);
  static Scalar* ltExpr(Val* lhs, Val* rhs);
  static Scalar* leExpr(Val* lhs, Val* rhs);
  static Scalar* geExpr(Val* lhs, Val* rhs);
  static Scalar* addExpr(Val* lhs, Val* rhs);
  static Scalar* subExpr(Val* lhs, Val* rhs);
  static Scalar* mulExpr(Val* lhs, Val* rhs);
  static Scalar* divExpr(Val* lhs, Val* rhs);
  static Scalar* ceilDivExpr(Val* lhs, Val* rhs);
  static Scalar* modExpr(Val* lhs, Val* rhs);
  static Scalar* maxExpr(Val* lhs, Val* rhs);
  static Scalar* minExpr(Val* lhs, Val* rhs);
  static Scalar* gcdExpr(Val* lhs, Val* rhs);

  // Ternary operations
  static Scalar* whereExpr(Val* pred, Val* lhs, Val* rhs);

  // Array and struct access
  static Scalar* getItemExpr(Val* array, Val* index);
  static Scalar* getAttrExpr(Val* struct_, std::string attr);

  // Get tensor metadata
  static Scalar* metadataExpr(TensorView* tv);

  // Construct an array of values, or nested arrays of values.
  template <typename T>
  static Scalar* arrayExpr(std::vector<T> members) {
    if constexpr (std::is_same_v<T, Val*>) {
      TORCH_INTERNAL_ASSERT(
          !members.empty(), "Cannot create an array with no members.");
      auto in_dtype = members.at(0)->dtype();
      auto out_dtype =
          ArrayOf{std::make_shared<DataType>(in_dtype), members.size()};
      auto out = newScalar(out_dtype);
      create<ArrayConstruct>(out, members);
      return out;
    } else {
      static_assert(
          is_std_vector_v<T>,
          "Argument for function array must be vector of value or nested vector");
      std::vector<Val*> array_members;
      std::transform(
          members.begin(),
          members.end(),
          std::back_inserter(array_members),
          [](const T& member) { return arrayExpr(member); });
      return arrayExpr(array_members);
    }
  }

  static Scalar* newScalar(DataType dtype);

  static Scalar* newConstant(PolymorphicValue value, DataType dtype) {
    return IrBuilder::create<Scalar>(value, dtype);
  }

 private:
  static Scalar* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  static Scalar* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
};

//! A wrapper builder with static expression simplification
//!
//! Example:
//! - addExpr(new Scalar(DataType::Int, 1), new Scalar(DataType::Int, 2)) ->
//! Scalar(DataType::Int, 3)
//! - addExpr(new Scalar(DataType::Int, 0), new NamedScalar("foo")) ->
//! NamedScalar("foo")
//!
//! Designed to be used to simplify predicate and index expressions in
//! generated code. Also, the shift validation may fail without
//! this simplification.
class TORCH_CUDA_CU_API SimplifyingIrBuilder : public IrBuilder {
 public:
  static Scalar* negExpr(Val* val);
  static Scalar* notExpr(Val* val);

  static Scalar* addExpr(Scalar* lhs, PolymorphicValue rhs);
  static Val* addExpr(Val* lhs, PolymorphicValue rhs);
  static Scalar* addExpr(Scalar* lhs, Scalar* rhs);
  static Val* addExpr(Val* lhs, Val* rhs);

  static Val* subExpr(Val* lhs, Val* rhs);

  static Scalar* mulExpr(Scalar* lhs, PolymorphicValue rhs);
  static Val* mulExpr(Val* lhs, PolymorphicValue rhs);
  static Scalar* mulExpr(Scalar* lhs, Scalar* rhs);
  static Val* mulExpr(Val* lhs, Val* rhs);

  static Val* divExpr(Val* lhs, Val* rhs);

  static Scalar* ceilDivExpr(Scalar* lhs, Scalar* rhs);
  static Scalar* ceilDivExpr(Val* lhs, Val* rhs);

  static Val* modExpr(Val* lhs, Val* rhs);
  static Scalar* andExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);
  static Val* gcdExpr(Val* lhs, Val* rhs);

  static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);
};

} // namespace nvfuser
