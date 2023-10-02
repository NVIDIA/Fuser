// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/builder_passkey.h>
#include <utils.h>

namespace nvfuser {

namespace kir {
class Kernel;
}

class IrCloner;

//! IR builder interface
class IrBuilder {
 public:
  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(Args&&... args) {
    auto container = FusionGuard::getCurFusion();
    // return create<T>(container, std::forward<Args>(args)...);
    NVF_ERROR(container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(IrContainer* container, Args&&... args) {
    NVF_ERROR(container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Clone an IR node, forwarding the arguments to the IrCloner constructor.
  //! Register clones with IrCloner's target container.
  template <class T>
  static T* clone(const T* src, IrCloner* ir_cloner);

  // Unary operations
  static Val* derefExpr(Val* val);
  static Val* negExpr(Val* val);
  static Val* logicalNotExpr(Val* val);
  static Val* bitwiseNotExpr(Val* val);
  static Val* absExpr(Val* val);
  static Val* setExpr(Val* val);
  static Val* maybeCastExpr(DataType dtype, Val* val);
  static Val* addressExpr(Val* val);
  static NamedScalar* setExprNamedScalar(const std::string& name, Val* val);
  static NamedScalar* addressExprNamedScalar(const std::string& name, Val* val);

  // Binary operations
  static Val* logicalAndExpr(Val* lhs, Val* rhs);
  static Val* logicalOrExpr(Val* lhs, Val* rhs);
  static Val* bitwiseAndExpr(Val* lhs, Val* rhs);
  static Val* bitwiseOrExpr(Val* lhs, Val* rhs);
  static Val* eqExpr(Val* lhs, Val* rhs);
  static Val* neExpr(Val* lhs, Val* rhs);
  static Val* gtExpr(Val* lhs, Val* rhs);
  static Val* ltExpr(Val* lhs, Val* rhs);
  static Val* leExpr(Val* lhs, Val* rhs);
  static Val* geExpr(Val* lhs, Val* rhs);
  static Val* addExpr(Val* lhs, Val* rhs);
  static Val* subExpr(Val* lhs, Val* rhs);
  static Val* mulExpr(Val* lhs, Val* rhs);
  static Val* divExpr(Val* lhs, Val* rhs);
  static Val* ceilDivExpr(Val* lhs, Val* rhs);
  static Val* modExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);
  static Val* gcdExpr(Val* lhs, Val* rhs);

  // Ternary operations
  static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);

  // Array and struct access
  static Val* getItemExpr(Val* array, Val* index);
  static Val* getItemExpr(Val* array, PolymorphicValue index);
  static Val* getAttrExpr(Val* struct_, std::string attr);
  static Val* reverseArrayExpr(Val* array);

  // Get tensor metadata
  static Val* metadataExpr(TensorView* tv);

  // Construct an array of values, or nested arrays of values.
  template <typename T>
  static Val* arrayExpr(std::vector<T> members) {
    if constexpr (std::is_same_v<T, Val*>) {
      NVF_ERROR(!members.empty(), "Cannot create an array with no members.");
      auto in_dtype = members.at(0)->dtype();
      auto out_dtype =
          ArrayType{std::make_shared<DataType>(in_dtype), members.size()};
      auto out = create<Val>(out_dtype);
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

  template <typename T = NotImplementedStruct>
  static Val* structExpr(
      const std::vector<std::pair<std::string, Val*>>& fields,
      std::string name = "") {
    std::vector<StructType::FieldInfo> field_infos;
    field_infos.reserve(fields.size());
    for (auto& field : fields) {
      field_infos.emplace_back(StructType::FieldInfo{
          field.first,
          std::make_shared<DataType>(field.second->dtype()),
          true});
    }
    DataType dtype =
        StructType::make<T>(std::move(field_infos), std::move(name));
    auto out = create<Val>(dtype);
    create<StructConstruct>(out, fields);
    return out;
  }

 private:
  static Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  static Val* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
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
class SimplifyingIrBuilder : public IrBuilder {
 public:
  static Val* negExpr(Val* val);
  static Val* logicalNotExpr(Val* val);
  static Val* bitwiseNotExpr(Val* val);
  static Val* maybeCastExpr(DataType dtype, Val* val);

  static Val* addExpr(
      Val* lhs,
      PolymorphicValue rhs,
      DataType rhs_dtype = DataType::Null);
  static Val* addExpr(Val* lhs, Val* rhs);

  static Val* subExpr(Val* lhs, Val* rhs);

  static Val* mulExpr(
      Val* lhs,
      PolymorphicValue rhs,
      DataType rhs_dtype = DataType::Null);
  static Val* mulExpr(Val* lhs, Val* rhs);
  static Val* divExpr(Val* lhs, Val* rhs);

  static Val* ceilDivExpr(Val* lhs, Val* rhs);

  static Val* modExpr(Val* lhs, Val* rhs);
  static Val* logicalAndExpr(Val* lhs, Val* rhs);
  static Val* logicalOrExpr(Val* lhs, Val* rhs);
  static Val* bitwiseAndExpr(Val* lhs, Val* rhs);
  static Val* bitwiseOrExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);
  static Val* gcdExpr(Val* lhs, Val* rhs);

  static Val* ltExpr(Val* lhs, Val* rhs);
  static Val* leExpr(Val* lhs, Val* rhs);
  static Val* eqExpr(Val* lhs, Val* rhs);
  static Val* neExpr(Val* lhs, Val* rhs);
  static Val* geExpr(Val* lhs, Val* rhs);
  static Val* gtExpr(Val* lhs, Val* rhs);

  static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);
};

} // namespace nvfuser
