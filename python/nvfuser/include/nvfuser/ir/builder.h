// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <fusion_guard.h>
#include <ir/builder_passkey.h>
#include <ir/container.h>
#include <utils.h>
#include <visibility.h>

namespace nvfuser {

namespace kir {
class Kernel;
}

class ArrayConstruct;
class IrCloner;
class NamedScalar;
class StructConstruct;
class TensorView;
class Val;

//! IR builder interface
class IrBuilder {
 public:
  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(Args&&... args) {
    Fusion* fusion = FusionGuard::getCurFusion();
    return createInContainer<T>(fusion, std::forward<Args>(args)...);
  }

  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* createInContainer(IrContainer* container, Args&&... args) {
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
  NVF_API static Val* derefExpr(Val* val);
  NVF_API static Val* negExpr(Val* val);
  NVF_API static Val* logicalNotExpr(Val* val);
  static Val* bitwiseNotExpr(Val* val);
  NVF_API static Val* bitCeilExpr(Val* val);
  NVF_API static Val* absExpr(Val* val);
  static Val* setExpr(Val* val);
  static Val* maybeCastExpr(DataType dtype, Val* val);
  static Val* maybeRefCastExpr(DataType dtype, Val* val);
  static Val* addressExpr(Val* val);
  static NamedScalar* setExprNamedScalar(const std::string& name, Val* val);
  static NamedScalar* addressExprNamedScalar(const std::string& name, Val* val);

  // Binary operations
  NVF_API static Val* logicalAndExpr(Val* lhs, Val* rhs);
  NVF_API static Val* logicalOrExpr(Val* lhs, Val* rhs);
  NVF_API static Val* bitwiseAndExpr(Val* lhs, Val* rhs);
  NVF_API static Val* bitwiseOrExpr(Val* lhs, Val* rhs);
  NVF_API static Val* bitwiseXorExpr(Val* lhs, Val* rhs);
  NVF_API static Val* lShiftExpr(Val* lhs, Val* rhs);
  NVF_API static Val* rShiftExpr(Val* lhs, Val* rhs);
  NVF_API static Val* eqExpr(Val* lhs, Val* rhs);
  NVF_API static Val* neExpr(Val* lhs, Val* rhs);
  NVF_API static Val* gtExpr(Val* lhs, Val* rhs);
  NVF_API static Val* ltExpr(Val* lhs, Val* rhs);
  NVF_API static Val* leExpr(Val* lhs, Val* rhs);
  NVF_API static Val* geExpr(Val* lhs, Val* rhs);
  NVF_API static Val* addExpr(Val* lhs, Val* rhs);
  NVF_API static Val* subExpr(Val* lhs, Val* rhs);
  NVF_API static Val* mulExpr(Val* lhs, Val* rhs);
  NVF_API static Val* divExpr(Val* lhs, Val* rhs);
  NVF_API static Val* ceilDivExpr(Val* lhs, Val* rhs);
  NVF_API static Val* modExpr(Val* lhs, Val* rhs);
  NVF_API static Val* maxExpr(Val* lhs, Val* rhs);
  NVF_API static Val* minExpr(Val* lhs, Val* rhs);
  NVF_API static Val* gcdExpr(Val* lhs, Val* rhs);
  NVF_API static Val* isDivisibleExpr(Val* dividend, Val* divisor);

  // Ternary operations
  NVF_API static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);

  // Array and struct access
  NVF_API static Val* getItemExpr(Val* array, Val* index);
  NVF_API static Val* getItemExpr(Val* array, PolymorphicValue index);
  NVF_API static Val* getAttrExpr(Val* struct_, std::string attr);
  NVF_API static Val* reverseArrayExpr(Val* array);

  // Get tensor metadata
  NVF_API static Val* metadataExpr(TensorView* tv);

  // Get tensor base address, for gmem tensor, it is something like
  // `T1.data`. For smem tensor, it is something like `toSmem(T1)`.
  static Val* baseAddressExpr(TensorView* tv);

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

  NVF_API static Val* mulExpr(
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
