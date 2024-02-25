// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <type_promotion.h>

#include <ir/interface_nodes.h>
#include <ops/arith.h>

namespace nvfuser {

namespace {

enum ValueType { Tensor, Scalar, None };

struct OperandType {
  ValueType value_type = ValueType::Tensor;
  DataType scalar_type = DataType::Null;
  size_t dim = 0;
};

struct ResultTypeState {
  DataType dimResult = DataType::Null;
  DataType wrappedResult = DataType::Null;
  DataType zeroResult = DataType::Null;
};

DataType promoteTypesSkipUndefined(DataType a, DataType b) {
  if (a == DataType::Null) {
    return b;
  }
  if (b == DataType::Null) {
    return a;
  }
  return promoteType(a, b);
}

ResultTypeState updateResultTypeState(
    OperandType tensor,
    const ResultTypeState& in_state) {
  ResultTypeState new_state = in_state;
  DataType current = tensor.scalar_type;

  if (tensor.dim > 0) {
    new_state.dimResult =
        promoteTypesSkipUndefined(in_state.dimResult, current);
  } else {
    new_state.zeroResult =
        promoteTypesSkipUndefined(in_state.zeroResult, current);
  }
  return new_state;
}

ResultTypeState updateResultTypeState(
    const DataType scalar,
    const ResultTypeState& in_state) {
  ResultTypeState new_state = in_state;
  DataType current = scalar;
  if (scalar == DataType::Half || scalar == DataType::BFloat16) {
    current = DataType::Float;
  }
  new_state.wrappedResult =
      promoteTypesSkipUndefined(in_state.wrappedResult, current);
  return new_state;
}

inline DataType combineCategories(DataType higher, DataType lower) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (isComplexType(higher)) {
    return higher;
  } else if (isComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (isFloatingPointType(higher)) {
      return getComplexTypeFromType(higher);
    }
    // in case of integral input
    // lower complex takes precedence.
    return lower;
  } else if (isFloatingPointType(higher)) {
    return higher;
  }
  if (higher == DataType::Bool || isFloatingPointType(lower)) {
    return promoteTypesSkipUndefined(higher, lower);
  }
  if (higher != DataType::Null) {
    return higher;
  }
  return lower;
}

DataType resultType(const ResultTypeState& in_state) {
  return combineCategories(
      in_state.dimResult,
      combineCategories(in_state.zeroResult, in_state.wrappedResult));
}

// Computes a common dtype using type promotion
DataType computeCommonDtype(const std::vector<OperandType>& operands) {
  ResultTypeState state = {};
  for (const auto& op : operands) {
    if (op.value_type == ValueType::Tensor) {
      state = updateResultTypeState(op, state);
    } else {
      state = updateResultTypeState(op.scalar_type, state);
    }
  }
  auto common_dtype = resultType(state);
  NVF_ERROR(common_dtype != DataType::Null);
  return common_dtype;
}

DataType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<OperandType>& operands) {
  DataType common_dtype = DataType::Null;

  bool has_different_input_dtypes = false;
  for (auto& op : operands) {
    if (op.scalar_type != common_dtype) {
      if (common_dtype == DataType::Null) {
        common_dtype = op.scalar_type;
      } else {
        has_different_input_dtypes = true;
      }
    }
  }

  // Computes a common dtype, if needed
  if (has_different_input_dtypes) {
    common_dtype = computeCommonDtype(operands);
  }

  // Promotes common dtype to the default float scalar type, if needed
  if (config.promote_integer_inputs_to_float &&
      (isIntegralType(common_dtype) || isBooleanType(common_dtype))) {
    common_dtype = DataType::Float;
  }

  // Some ops like nextafter are not implemented for non-float types
  if (config.require_full_precision_promoted) {
    NVF_CHECK(
        common_dtype == DataType::Float || common_dtype == DataType::Double,
        "Promoted type must be single or double precision float but found ",
        common_dtype);
  }

  return common_dtype;
}

OperandType getValueType(at::TypePtr type) {
  if (auto tensor_type = type->cast<at::TensorType>()) {
    NVF_ERROR(
        tensor_type->scalarType().has_value(),
        "Missing Scalar Type information");
    // TODO: Type Inference does not propagate Shape Information
    return {
        ValueType::Tensor,
        aten_to_data_type(tensor_type->scalarType().value()),
        tensor_type->dim().has_value() ? tensor_type->dim().value() : 1};
  } else if (auto scalar_type = tryScalarTypeFromJitType(*type)) {
    return {ValueType::Scalar, aten_to_data_type(scalar_type.value())};
  } else {
    return {ValueType::None, DataType::Null};
  }
}

OperandType getValueType(Val* type) {
  NVF_ERROR(type->getDataType().has_value());

  if (type->isA<TensorView>()) {
    auto tensor_view = type->as<TensorView>();
    return {
        ValueType::Tensor,
        tensor_view->getDataType().value(),
        tensor_view->getMaybeRFactorDomain().size()};
  } else if (type->getDataType().has_value()) {
    return {ValueType::Scalar, type->getDataType().value()};
  } else {
    return {ValueType::None, DataType::Null};
  }
}

} // namespace

DataType computeTypes(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands,
    const bool cast_half_to_float) {
  std::vector<OperandType> vt_operands;
  vt_operands.reserve(operands.size());
  for (const auto& op : operands) {
    vt_operands.push_back(getValueType(op));
  }

  auto common_type = computeTypes(config, vt_operands);
  // Cast FP16 / BFloat16 to Float
  if (cast_half_to_float &&
      (common_type == DataType::Half || common_type == DataType::BFloat16)) {
    common_type = DataType::Float;
  }

  return common_type;
}

std::vector<Val*> promoteValues(
    const std::vector<Val*>& operands,
    DataType common_type) {
  std::vector<Val*> promoted_operands;
  promoted_operands.reserve(operands.size());
  for (auto op : operands) {
    promoted_operands.push_back(optionalCast(common_type, op));
  }

  NVF_ERROR(operands.size() == promoted_operands.size());
  return promoted_operands;
}

std::vector<Val*> promoteValues(
    const TypePromotionConfig& config,
    const std::vector<Val*>& operands) {
  return promoteValues(operands, computeTypes(config, operands));
}

Val* optionalCast(DataType dtype, Val* v) {
  NVF_ERROR(v->getDataType().has_value());
  // Avoid casting Float/Int/ComplexDouble scalar to any corresponding
  // FloatingPoint/Integral/Double type in fusion. Instead, we cast them
  // directly. The exception is Bool, which is always cast to the desired
  // type.
  const bool kSameDtype = v->getDataType().value() == dtype;
  const bool kIsScalarFloat =
      !v->isA<TensorView>() && isFloatingPointType(dtype);
  const bool kIsScalarInt = !v->isA<TensorView>() && isIntegralType(dtype);
  const bool kIsScalarComplex = !v->isA<TensorView>() && isComplexType(dtype);
  if (kSameDtype ||
      (kIsScalarFloat && isFloatingPointType(v->getDataType().value())) ||
      (kIsScalarInt && isIntegralType(v->getDataType().value())) ||
      (kIsScalarComplex && isComplexType(v->getDataType().value()))) {
    return v;
  } else {
    return castOp(dtype, v);
  }
}

Val* optionalCastStrict(DataType dtype, Val* v) {
  NVF_ERROR(v->getDataType().has_value());
  const bool kSameDtype = v->getDataType().value() == dtype;
  return (kSameDtype) ? v : castOp(dtype, v);
}

} // namespace nvfuser
