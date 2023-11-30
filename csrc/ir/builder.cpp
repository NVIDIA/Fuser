// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <kernel.h>
#include <C++20/compare>

#include <ir/all_nodes.h>
#include <ir/container.h>

#include <complex>
#include <cstdint>

namespace nvfuser {

Val* IrBuilder::newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  NVF_CHECK(
      lhs != nullptr && rhs != nullptr,
      "Either lhs or rhs is a nullptr in newArithmeticExpr.");

  auto dtype = lhs->dtype();

  // In principle, we should keep these IrBuilder functions as
  // simple as possible since they are just used by the lowering for
  // scalar computations. We should enforce strict typing with no
  // implicit type promotion unless required. However, for
  // int and int64_t, our usages are pretty loose in many places. Originally we
  // only had int64_t, then we added nvfuser_index_t and replaced the types of
  // some of the values from int64_t to int just at the beginning of lowering.
  // This resulted in inconsistent usages of integer types in many places, and
  // fixing all of them to make everything consistent would be a lot of work
  // than just allowing the integer type promotion for the two inputs as below.
  // Note that this is only needed for integer types. See also PR #2228.
  if (lhs->dtype() != rhs->dtype()) {
    dtype = promoteType(lhs->dtype(), rhs->dtype());
    if (isPointerType(lhs->dtype()) || isPointerType(rhs->dtype())) {
      NVF_ERROR(op_type == BinaryOpType::Add || op_type == BinaryOpType::Sub);
    }
  }
  auto result = create<Val>(dtype);
  IrBuilder::create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Val* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  NVF_CHECK(
      lhs != nullptr && rhs != nullptr,
      "Either lhs or rhs is a nullptr in newLogicExpr.");
  auto result = create<Val>(DataType::Bool);
  IrBuilder::create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Val* IrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  NVF_CHECK(
      pred != nullptr && lhs != nullptr && rhs != nullptr,
      "Either pred, lhs, or rhs is a nullptr in whereExpr.");
  NVF_CHECK(lhs->dtype() == rhs->dtype(), "Incompatible operand types");
  auto result = create<Val>(lhs->dtype());
  IrBuilder::create<TernaryOp>(TernaryOpType::Where, result, pred, lhs, rhs);
  return result;
}

Val* IrBuilder::negExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in negExpr.");
  auto result = create<Val>(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Neg, result, val);
  return result;
}

Val* IrBuilder::logicalNotExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in logicalNotExpr.");
  auto result = create<Val>(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::LogicalNot, result, val);
  return result;
}

Val* IrBuilder::bitwiseNotExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in bitwiseNotExpr.");
  auto result = create<Val>(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::BitwiseNot, result, val);
  return result;
}

Val* IrBuilder::derefExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in derefExpr.");
  auto result = create<Val>(*(std::get<PointerType>(val->dtype().type).type));
  IrBuilder::create<UnaryOp>(UnaryOpType::Dereference, result, val);
  return result;
}

Val* IrBuilder::absExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in absExpr.");
  auto result = create<Val>(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Abs, result, val);
  return result;
}

Val* IrBuilder::setExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in setExpr.");
  auto result = create<Val>(val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

Val* IrBuilder::maybeCastExpr(DataType dtype, Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in castExpr.");
  if (val->dtype() == dtype) {
    return val;
  }
  auto result = create<Val>(dtype);
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, result, val);
  return result;
}

Val* IrBuilder::maybeRefCastExpr(DataType dtype, Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in bitCastExpr.");
  if (val->dtype() == dtype) {
    return val;
  }
  auto result = create<Val>(dtype);
  IrBuilder::create<UnaryOp>(UnaryOpType::RefCast, result, val);
  return result;
}

Val* IrBuilder::addressExpr(Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in addressExpr.");
  auto result = create<Val>(
      DataType(PointerType{std::make_shared<DataType>(val->dtype())}));
  IrBuilder::create<UnaryOp>(UnaryOpType::Address, result, val);
  return result;
}

NamedScalar* IrBuilder::setExprNamedScalar(const std::string& name, Val* val) {
  NVF_CHECK(val != nullptr, "val is a nullptr in setExprNamedScalar.");
  auto result = IrBuilder::create<NamedScalar>(name, val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

NamedScalar* IrBuilder::addressExprNamedScalar(
    const std::string& name,
    Val* val) {
  auto ptr = addressExpr(val);
  auto result = IrBuilder::create<NamedScalar>(name, DataType::Int);
  IrBuilder::create<UnaryOp>(UnaryOpType::BitCast, result, ptr);
  return result;
}

Val* IrBuilder::logicalAndExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LogicalAnd, lhs, rhs);
}

Val* IrBuilder::logicalOrExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LogicalOr, lhs, rhs);
}

Val* IrBuilder::bitwiseAndExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::BitwiseAnd, lhs, rhs);
}

Val* IrBuilder::bitwiseOrExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::BitwiseOr, lhs, rhs);
}

Val* IrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Eq, lhs, rhs);
}

Val* IrBuilder::neExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::NE, lhs, rhs);
}

Val* IrBuilder::gtExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GT, lhs, rhs);
}

Val* IrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LT, lhs, rhs);
}

Val* IrBuilder::leExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LE, lhs, rhs);
}

Val* IrBuilder::geExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GE, lhs, rhs);
}

Val* IrBuilder::addExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Add, lhs, rhs);
}

Val* IrBuilder::subExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Sub, lhs, rhs);
}

Val* IrBuilder::mulExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mul, lhs, rhs);
}

Val* IrBuilder::divExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Div, lhs, rhs);
}

Val* IrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::CeilDiv, lhs, rhs);
}

Val* IrBuilder::modExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mod, lhs, rhs);
}

Val* IrBuilder::maxExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Max, lhs, rhs);
}

Val* IrBuilder::minExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Min, lhs, rhs);
}

Val* IrBuilder::gcdExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Gcd, lhs, rhs);
}

Val* IrBuilder::getItemExpr(Val* array, Val* index) {
  auto item_dtype = std::get<ArrayType>(array->dtype().type).type;
  auto out = create<Val>(*item_dtype);
  create<GetItem>(array->container(), out, array, index);
  return out;
}

Val* IrBuilder::getItemExpr(Val* array, PolymorphicValue index) {
  auto item_dtype = std::get<ArrayType>(array->dtype().type).type;
  auto out = create<Val>(*item_dtype);
  create<GetItem>(
      array->container(), out, array, create<Val>(index, DataType::Int));
  return out;
}

Val* IrBuilder::getAttrExpr(Val* struct_, std::string attr) {
  auto struct_type = std::get<StructType>(struct_->dtype().type);
  const auto& item_type = struct_type.fieldDataType(attr);
  auto out = create<Val>(item_type);
  create<GetAttr>(struct_->container(), out, struct_, std::move(attr));
  return out;
}

Val* IrBuilder::reverseArrayExpr(Val* array) {
  auto out = create<Val>(array->dtype());
  create<ReverseArray>(out, array);
  return out;
}

Val* IrBuilder::metadataExpr(TensorView* tv) {
  return tv->fusion()->metadataOf(tv);
}

Val* IrBuilder::baseAddressExpr(TensorView* tv) {
  auto metadata = metadataExpr(tv);
  switch (auto memtype = tv->getMemoryType()) {
    case MemoryType::Global:
      return getAttrExpr(metadata, "data");
    case MemoryType::Shared: {
      auto output = create<Val>(DataType::SMemAddress);
      create<UnaryOp>(UnaryOpType::ToUnsignedSmemAddr, output, metadata);
      return output;
    }
    default:
      NVF_CHECK(false, "Unsupported memory type ", memtype);
  }
}

Val* SimplifyingIrBuilder::negExpr(Val* val) {
  if (val->isZeroInt()) {
    return val->container()->zeroVal(val->dtype());
  } else if (val->isConst()) {
    return IrBuilder::create<Val>(-val->value(), val->dtype());
  }
  return IrBuilder::negExpr(val);
}

Val* SimplifyingIrBuilder::logicalNotExpr(Val* val) {
  if (val->isConst()) {
    if (val->value()) {
      return FusionGuard::getCurFusion()->falseVal();
    } else {
      return FusionGuard::getCurFusion()->trueVal();
    }
  }
  return IrBuilder::logicalNotExpr(val);
}

Val* SimplifyingIrBuilder::bitwiseNotExpr(Val* val) {
  if (val->isConst()) {
    return IrBuilder::create<Val>(~(val->value()), val->dtype());
  }
  return IrBuilder::bitwiseNotExpr(val);
}

Val* SimplifyingIrBuilder::maybeCastExpr(DataType dtype, Val* val) {
  if (val->isConst()) {
    return IrBuilder::create<Val>(val->value(), dtype);
  }
  return IrBuilder::maybeCastExpr(dtype, val);
}

Val* SimplifyingIrBuilder::addExpr(
    Val* lhs,
    PolymorphicValue rhs,
    DataType rhs_dtype) {
  if (rhs_dtype == DataType::Null) {
    rhs_dtype = getDataType(rhs);
  }
  if (lhs == nullptr) {
    return IrBuilder::IrBuilder::create<Val>(rhs, rhs_dtype);
  }
  auto target_dtype = promoteType(lhs->dtype(), rhs_dtype);
  if (rhs == 0) {
    return maybeCastExpr(target_dtype, lhs);
  } else if (lhs->isConst()) {
    return IrBuilder::IrBuilder::create<Val>(lhs->value() + rhs, target_dtype);
  } else if (rhs > 0) {
    return IrBuilder::addExpr(
        lhs, IrBuilder::IrBuilder::create<Val>(rhs, rhs_dtype));
  } else {
    return IrBuilder::subExpr(
        lhs, IrBuilder::IrBuilder::create<Val>(-rhs, rhs_dtype));
  }
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, Val* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return addExpr(rhs, lhs->value());
  } else if (rhs->isConst()) {
    return addExpr(lhs, rhs->value(), rhs->dtype());
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  if (lhs->isIntegralScalar() && lhs->sameAs(rhs)) {
    return lhs->fusion()->zeroVal(lhs->dtype());
  }
  return addExpr(lhs, negExpr(rhs));
}

Val* SimplifyingIrBuilder::mulExpr(
    Val* lhs,
    PolymorphicValue rhs,
    DataType rhs_dtype) {
  if (rhs_dtype == DataType::Null) {
    rhs_dtype = getDataType(rhs);
  }
  if (lhs == nullptr) {
    return IrBuilder::create<Val>(rhs, rhs_dtype);
  }
  auto target_dtype = promoteType(lhs->dtype(), rhs_dtype);
  if (rhs == 0) {
    return lhs->container()->zeroVal(target_dtype);
  } else if (rhs == 1) {
    return maybeCastExpr(target_dtype, lhs);
  } else if (lhs->isConst()) {
    return IrBuilder::create<Val>(lhs->value() * rhs, target_dtype);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Val>(rhs, rhs_dtype));
  }
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, Val* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return mulExpr(rhs, lhs->value());
  } else if (rhs->isConst()) {
    return mulExpr(lhs, rhs->value());
  } else {
    return IrBuilder::mulExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::divExpr(Val* lhs, Val* rhs) {
  if (rhs->isOneInt()) {
    return lhs;
  }
  return IrBuilder::divExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  if (rhs->isOneInt()) {
    return lhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    auto l = lhs->value();
    auto r = rhs->value();
    using namespace PolymorphicValue_functions;
    return IrBuilder::IrBuilder::create<Val>(
        ceildiv(l, r), promoteType(lhs->dtype(), rhs->dtype()));
  } else {
    return IrBuilder::ceilDivExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::modExpr(Val* lhs, Val* rhs) {
  NVF_ERROR(isIntegralType(lhs->dtype()));
  NVF_ERROR(isIntegralType(rhs->dtype()));
  if (rhs->isOneInt() || lhs->isZeroInt() || lhs->sameAs(rhs)) {
    return FusionGuard::getCurFusion()->zeroVal(
        promoteType(lhs->dtype(), rhs->dtype()));
  }
  return IrBuilder::modExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::logicalAndExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
  NVF_ERROR(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

  if (lhs == nullptr) {
    return rhs_scalar;
  } else if (rhs == nullptr) {
    return lhs_scalar;
  }

  bool lhs_definitely_true = false;
  bool lhs_definitely_false = false;
  if (lhs_scalar && lhs_scalar->isConst()) {
    lhs_definitely_true = lhs_scalar->value().as<bool>();
    lhs_definitely_false = !lhs_scalar->value().as<bool>();
  }
  bool rhs_definitely_true = false;
  bool rhs_definitely_false = false;
  if (rhs_scalar && rhs_scalar->isConst()) {
    rhs_definitely_true = rhs_scalar->value().as<bool>();
    rhs_definitely_false = !rhs_scalar->value().as<bool>();
  }

  if (lhs_definitely_true && rhs_definitely_true) {
    return FusionGuard::getCurFusion()->trueVal();
  } else if (lhs_definitely_false || rhs_definitely_false) {
    return FusionGuard::getCurFusion()->falseVal();
  } else if (lhs_definitely_true) {
    return rhs_scalar;
  } else if (rhs_definitely_true) {
    return lhs_scalar;
  }

  return IrBuilder::logicalAndExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::logicalOrExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
  NVF_ERROR(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

  if (lhs == nullptr) {
    return rhs_scalar;
  } else if (rhs == nullptr) {
    return lhs_scalar;
  }

  bool lhs_definitely_true = false;
  bool lhs_definitely_false = false;
  if (lhs_scalar && lhs_scalar->isConst()) {
    lhs_definitely_true = lhs_scalar->value().as<bool>();
    lhs_definitely_false = !lhs_scalar->value().as<bool>();
  }
  bool rhs_definitely_true = false;
  bool rhs_definitely_false = false;
  if (rhs_scalar && rhs_scalar->isConst()) {
    rhs_definitely_true = rhs_scalar->value().as<bool>();
    rhs_definitely_false = !rhs_scalar->value().as<bool>();
  }

  if (lhs_definitely_true || rhs_definitely_true) {
    return FusionGuard::getCurFusion()->trueVal();
  } else if (lhs_definitely_false && rhs_definitely_false) {
    return FusionGuard::getCurFusion()->falseVal();
  } else if (lhs_definitely_false) {
    return rhs_scalar;
  } else if (rhs_definitely_false) {
    return lhs_scalar;
  }

  return IrBuilder::logicalOrExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::bitwiseAndExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
  NVF_ERROR(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

  if (lhs == nullptr) {
    return rhs_scalar;
  } else if (rhs == nullptr) {
    return lhs_scalar;
  }

  bool lhs_zero = false;
  bool lhs_all_ones = false;
  if (lhs_scalar && lhs_scalar->isConst()) {
    if (rhs_scalar && rhs_scalar->isConst()) {
      return IrBuilder::create<Val>(lhs_scalar->value() & rhs_scalar->value());
    }
    lhs_zero = lhs_scalar->value().as<int64_t>() == 0;
    lhs_all_ones = lhs_scalar->value().as<int64_t>() == -1;
  }
  bool rhs_zero = false;
  bool rhs_all_ones = false;
  if (rhs_scalar && rhs_scalar->isConst()) {
    rhs_zero = rhs_scalar->value().as<int64_t>() == 0;
    rhs_all_ones = rhs_scalar->value().as<int64_t>() == -1;
  }

  if (lhs_zero || rhs_zero) {
    return FusionGuard::getCurFusion()->zeroVal(
        promoteType(lhs->dtype(), rhs->dtype()));
  } else if (lhs_all_ones && rhs_all_ones) {
    return IrBuilder::IrBuilder::create<Val>((int64_t)-1, lhs->dtype());
  } else if (lhs_all_ones) {
    return rhs_scalar;
  } else if (rhs_all_ones) {
    return lhs_scalar;
  }

  return IrBuilder::bitwiseAndExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::bitwiseOrExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
  NVF_ERROR(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

  if (lhs == nullptr) {
    return rhs_scalar;
  } else if (rhs == nullptr) {
    return lhs_scalar;
  }

  bool lhs_zero = false;
  bool lhs_all_ones = false;
  if (lhs_scalar && lhs_scalar->isConst()) {
    if (rhs_scalar && rhs_scalar->isConst()) {
      return IrBuilder::create<Val>(lhs_scalar->value() | rhs_scalar->value());
    }
    lhs_zero = lhs_scalar->value().as<int64_t>() == 0;
    lhs_all_ones = lhs_scalar->value().as<int64_t>() == -1;
  }
  bool rhs_zero = false;
  bool rhs_all_ones = false;
  if (rhs_scalar && rhs_scalar->isConst()) {
    rhs_zero = rhs_scalar->value().as<int64_t>() == 0;
    rhs_all_ones = rhs_scalar->value().as<int64_t>() == -1;
  }

  if (lhs_all_ones || rhs_all_ones) {
    return IrBuilder::IrBuilder::create<Val>((int64_t)-1, lhs->dtype());
  } else if (lhs_zero && rhs_zero) {
    return FusionGuard::getCurFusion()->zeroVal(
        promoteType(lhs->dtype(), rhs->dtype()));
  } else if (lhs_zero) {
    return rhs_scalar;
  } else if (rhs_zero) {
    return lhs_scalar;
  }

  return IrBuilder::bitwiseOrExpr(lhs, rhs);
}

namespace {

template <typename IrBuilderFunc, typename Fimc>
Val* minOrMaxExpr(
    Val* lhs,
    Val* rhs,
    IrBuilderFunc ir_builder_func,
    Fimc func) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr || lhs->sameAs(rhs)) {
    return rhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    return IrBuilder::create<Val>(func(lhs->value(), rhs->value()));
  } else {
    return ir_builder_func(lhs, rhs);
  }
}

} // namespace

Val* SimplifyingIrBuilder::maxExpr(Val* lhs, Val* rhs) {
  using namespace PolymorphicValue_functions;
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::maxExpr(lhs, rhs); },
      [](auto lhs, auto rhs) { return max(lhs, rhs); });
}

Val* SimplifyingIrBuilder::minExpr(Val* lhs, Val* rhs) {
  using namespace PolymorphicValue_functions;
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::minExpr(lhs, rhs); },
      [](auto lhs, auto rhs) { return min(lhs, rhs); });
}

Val* SimplifyingIrBuilder::gcdExpr(Val* lhs, Val* rhs) {
  NVF_ERROR(isIntegralType(lhs->dtype()));
  NVF_ERROR(isIntegralType(rhs->dtype()));
  if (lhs->isZeroInt()) {
    return rhs;
  }
  if (rhs->isZeroInt()) {
    return lhs;
  }
  if (lhs->sameAs(rhs)) {
    return lhs;
  }
  if (lhs->isOneInt() || rhs->isOneInt()) {
    return lhs->container()->oneVal(promoteType(lhs->dtype(), rhs->dtype()));
  }
  return IrBuilder::gcdExpr(lhs, rhs);
}

namespace {

//! Compares a to b if they are both const scalars convertible to double
std::partial_ordering compareScalars(Val* a, Val* b) {
  ExpressionEvaluator ee;
  auto a_val = ee.evaluate(a);
  if (!a_val.hasValue()) {
    return std::partial_ordering::unordered;
  }
  auto b_val = ee.evaluate(b);
  if (!b_val.hasValue()) {
    return std::partial_ordering::unordered;
  }
  if (a_val < b_val) {
    return std::partial_ordering::less;
  } else if (a_val == b_val) {
    return std::partial_ordering::equivalent;
  } else {
    return std::partial_ordering::greater;
  }
}
} // namespace

Val* SimplifyingIrBuilder::ltExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::ltExpr(lhs, rhs);
  } else if (c == std::partial_ordering::less) {
    return lhs->fusion()->trueVal();
  } else {
    return lhs->fusion()->falseVal();
  }
}

Val* SimplifyingIrBuilder::leExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::leExpr(lhs, rhs);
  } else if (c == std::partial_ordering::greater) {
    return lhs->fusion()->falseVal();
  } else {
    return lhs->fusion()->trueVal();
  }
}

Val* SimplifyingIrBuilder::eqExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::eqExpr(lhs, rhs);
  } else if (c == std::partial_ordering::equivalent) {
    return lhs->fusion()->trueVal();
  } else {
    return lhs->fusion()->falseVal();
  }
}

Val* SimplifyingIrBuilder::neExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::neExpr(lhs, rhs);
  } else if (c == std::partial_ordering::equivalent) {
    return lhs->fusion()->falseVal();
  } else {
    return lhs->fusion()->trueVal();
  }
}

Val* SimplifyingIrBuilder::geExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::geExpr(lhs, rhs);
  } else if (c == std::partial_ordering::less) {
    return lhs->fusion()->falseVal();
  } else {
    return lhs->fusion()->trueVal();
  }
}

Val* SimplifyingIrBuilder::gtExpr(Val* lhs, Val* rhs) {
  auto c = compareScalars(lhs, rhs);
  if (c == std::partial_ordering::unordered) {
    return IrBuilder::gtExpr(lhs, rhs);
  } else if (c == std::partial_ordering::greater) {
    return lhs->fusion()->trueVal();
  } else {
    return lhs->fusion()->falseVal();
  }
}

Val* SimplifyingIrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  NVF_ERROR(
      pred->dtype() == DataType::Bool,
      "Where requires a predicate as an input, but received");
  if (lhs->sameAs(rhs)) {
    return lhs; // return value is independent of predicate
  }
  if (pred->isConstScalar() && pred->isABool()) {
    if (pred->evaluate()) {
      return lhs;
    } else {
      return rhs;
    }
  }
  return IrBuilder::whereExpr(pred, lhs, rhs);
}

} // namespace nvfuser
