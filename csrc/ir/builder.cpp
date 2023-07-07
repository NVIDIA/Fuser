// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <kernel.h>

#include <ir/all_nodes.h>
#include <ir/container.h>

#include <complex>

namespace nvfuser {

Val* IrBuilder::newScalar(DataType dtype) {
  if (isPointerType(dtype)) {
    return IrBuilder::create<Int>(dtype);
  }
  if (!std::holds_alternative<PrimDataType>(dtype.type)) {
    return IrBuilder::create<Val>(ValType::Scalar, dtype);
  }
  switch (std::get<PrimDataType>(dtype.type)) {
    case DataType::Bool:
      return IrBuilder::create<Bool>();
    case DataType::Float:
    case DataType::Double:
      return IrBuilder::create<Double>(dtype);
    case DataType::Int:
    case DataType::Int32:
    case DataType::Index:
      return IrBuilder::create<Int>(dtype);
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return IrBuilder::create<ComplexDouble>(dtype);
    default:
      TORCH_CHECK(false, "Unexpected data type: ", dtype);
  }
}

Val* IrBuilder::newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  TORCH_CHECK(
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
      TORCH_INTERNAL_ASSERT(
          op_type == BinaryOpType::Add || op_type == BinaryOpType::Sub);
    }
  }
  auto result = newScalar(dtype);
  IrBuilder::create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Bool* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  TORCH_CHECK(
      lhs != nullptr && rhs != nullptr,
      "Either lhs or rhs is a nullptr in newLogicExpr.");
  auto result = IrBuilder::create<Bool>(std::nullopt);
  IrBuilder::create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Val* IrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  TORCH_CHECK(
      pred != nullptr && lhs != nullptr && rhs != nullptr,
      "Either pred, lhs, or rhs is a nullptr in whereExpr.");
  TORCH_CHECK(lhs->dtype() == rhs->dtype(), "Incompatible operand types");
  auto result = newScalar(lhs->dtype());
  IrBuilder::create<TernaryOp>(TernaryOpType::Where, result, pred, lhs, rhs);
  return result;
}

Val* IrBuilder::negExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in negExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Neg, result, val);
  return result;
}

Val* IrBuilder::notExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in notExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Not, result, val);
  return result;
}

Val* IrBuilder::absExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in notExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Abs, result, val);
  return result;
}

Val* IrBuilder::setExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in setExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

Val* IrBuilder::setExprNamedScalar(const std::string& name, Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in setExprNamedScalar.");
  auto result = IrBuilder::create<NamedScalar>(name, val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

Val* IrBuilder::addressExprNamedScalar(const std::string& name, Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in addressExprNamedScalar.");
  auto result = IrBuilder::create<NamedScalar>(name, DataType::Int);
  IrBuilder::create<UnaryOp>(UnaryOpType::Address, result, val);
  return result;
}

Bool* IrBuilder::andExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::And, lhs, rhs);
}

Bool* IrBuilder::orExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Or, lhs, rhs);
}

Bool* IrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Eq, lhs, rhs);
}

Bool* IrBuilder::neExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::NE, lhs, rhs);
}

Bool* IrBuilder::gtExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GT, lhs, rhs);
}

Bool* IrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LT, lhs, rhs);
}

Bool* IrBuilder::leExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LE, lhs, rhs);
}

Bool* IrBuilder::geExpr(Val* lhs, Val* rhs) {
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
  auto item_dtype = std::get<ArrayOf>(array->dtype().type).type;
  auto out = newScalar(*item_dtype);
  create<GetItem>(array->container(), out, array, index);
  return out;
}

Val* IrBuilder::getAttrExpr(Val* struct_, std::string attr) {
  auto item_dtype =
      NVFUSER_MAYBE_STAR std::get<StructOf>(getMaybeMetaDataType(struct_).type)
          .types.at(attr);
  auto out = newScalar(item_dtype);
  create<GetAttr>(struct_->container(), out, struct_, std::move(attr));
  return out;
}

Val* SimplifyingIrBuilder::negExpr(Val* val) {
  if (val->isZeroInt()) {
    return val->container()->zeroVal();
  } else if (auto int_val = dynamic_cast<Int*>(val)) {
    if (int_val->isConst()) {
      return IrBuilder::create<Int>(-int_val->value().value());
    }
  }
  return IrBuilder::negExpr(val);
}

Val* SimplifyingIrBuilder::notExpr(Val* val) {
  if (auto bool_val = dynamic_cast<Bool*>(val)) {
    if (bool_val->isConst()) {
      if (bool_val->value().value()) {
        return FusionGuard::getCurFusion()->falseVal();
      } else {
        return FusionGuard::getCurFusion()->trueVal();
      }
    }
  }
  return IrBuilder::notExpr(val);
}

Val* SimplifyingIrBuilder::addExpr(Int* lhs, Int::ScalarType rhs) {
  if (rhs == 0) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::IrBuilder::create<Int>(rhs);
  } else if (lhs->isConst()) {
    return IrBuilder::IrBuilder::create<Int>(lhs->value().value() + rhs);
  } else if (rhs > 0) {
    return IrBuilder::addExpr(lhs, IrBuilder::IrBuilder::create<Int>(rhs));
  } else {
    return IrBuilder::subExpr(lhs, IrBuilder::IrBuilder::create<Int>(-rhs));
  }
}

Val* SimplifyingIrBuilder::addExpr(Int* lhs, Int* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return addExpr(rhs, lhs->value().value());
  } else if (rhs->isConst()) {
    return addExpr(lhs, rhs->value().value());
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr || lhs->isZeroInt()) {
    return rhs;
  } else if (rhs == nullptr || rhs->isZeroInt()) {
    return lhs;
  }
  auto lhs_int = dynamic_cast<Int*>(lhs);
  auto rhs_int = dynamic_cast<Int*>(rhs);
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return addExpr(lhs_int, rhs_int);
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, Int::ScalarType rhs) {
  auto lhs_int = dynamic_cast<Int*>(lhs);
  if (lhs_int != nullptr) {
    return addExpr(lhs_int, rhs);
  } else {
    return addExpr(lhs, IrBuilder::create<Int>(rhs));
  }
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  return addExpr(lhs, negExpr(rhs));
}

Val* SimplifyingIrBuilder::mulExpr(Int* lhs, Int::ScalarType rhs) {
  if (rhs == 0) {
    return lhs->container()->zeroVal();
  } else if (rhs == 1) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::create<Int>(rhs);
  } else if (lhs->isConst()) {
    return IrBuilder::create<Int>(lhs->value().value() * rhs);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Int>(rhs));
  }
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, Int::ScalarType rhs) {
  auto lhs_int = dynamic_cast<Int*>(lhs);
  if (lhs_int != nullptr) {
    return mulExpr(lhs_int, rhs);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Int>(rhs));
  }
}

Val* SimplifyingIrBuilder::mulExpr(Int* lhs, Int* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return mulExpr(rhs, lhs->value().value());
  } else if (rhs->isConst()) {
    return mulExpr(lhs, rhs->value().value());
  } else {
    return IrBuilder::mulExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr || lhs->isOneInt()) {
    return rhs;
  } else if (rhs == nullptr || rhs->isOneInt()) {
    return lhs;
  } else if (lhs->isZeroInt() || rhs->isZeroInt()) {
    return lhs->container()->zeroVal();
  }
  auto lhs_int = dynamic_cast<Int*>(lhs);
  auto rhs_int = dynamic_cast<Int*>(rhs);
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return mulExpr(lhs_int, rhs_int);
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

Val* SimplifyingIrBuilder::ceilDivExpr(Int* lhs, Int* rhs) {
  if (rhs->isOneInt()) {
    return lhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    auto l = lhs->value().value();
    auto r = rhs->value().value();
    if (r > 0) {
      return IrBuilder::IrBuilder::create<Int>((l + r - 1) / r);
    } else {
      return IrBuilder::IrBuilder::create<Int>((l + r + 1) / r);
    }
  } else {
    return IrBuilder::ceilDivExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr && rhs != nullptr);
  auto lhs_int = dynamic_cast<Int*>(lhs);
  auto rhs_int = dynamic_cast<Int*>(rhs);
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return ceilDivExpr(lhs_int, rhs_int);
  } else {
    return IrBuilder::ceilDivExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::modExpr(Val* lhs, Val* rhs) {
  if (rhs->isOneInt()) {
    return FusionGuard::getCurFusion()->zeroVal();
  }
  return IrBuilder::modExpr(lhs, rhs);
}

Bool* SimplifyingIrBuilder::andExpr(Val* lhs, Val* rhs) {
  auto lhs_bool = dynamic_cast<Bool*>(lhs);
  auto rhs_bool = dynamic_cast<Bool*>(rhs);
  TORCH_INTERNAL_ASSERT(!(lhs_bool == nullptr && rhs_bool == nullptr));

  if (lhs == nullptr) {
    return rhs_bool;
  } else if (rhs == nullptr) {
    return lhs_bool;
  }

  bool lhs_definitely_true = false;
  bool lhs_definitely_false = false;
  if (lhs_bool && lhs_bool->isConst()) {
    lhs_definitely_true = lhs_bool->value().value();
    lhs_definitely_false = !lhs_bool->value().value();
  }
  bool rhs_definitely_true = false;
  bool rhs_definitely_false = false;
  if (rhs_bool && rhs_bool->isConst()) {
    rhs_definitely_true = rhs_bool->value().value();
    rhs_definitely_false = !rhs_bool->value().value();
  }

  if (lhs_definitely_true && rhs_definitely_true) {
    return FusionGuard::getCurFusion()->trueVal();
  } else if (lhs_definitely_false || rhs_definitely_false) {
    return FusionGuard::getCurFusion()->falseVal();
  } else if (lhs_definitely_true) {
    return rhs_bool;
  } else if (rhs_definitely_true) {
    return lhs_bool;
  }

  return IrBuilder::andExpr(lhs, rhs);
}

namespace {

template <typename IrBuilderFunc, typename IntFunc>
Val* minOrMaxExpr(
    Int* lhs,
    Int* rhs,
    IrBuilderFunc ir_builder_func,
    IntFunc int_func) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    return IrBuilder::create<Int>(
        int_func(lhs->value().value(), rhs->value().value()));
  } else {
    return ir_builder_func(lhs, rhs);
  }
}

template <typename IrBuilderFunc, typename IntFunc>
Val* minOrMaxExpr(
    Val* lhs,
    Val* rhs,
    IrBuilderFunc ir_builder_func,
    IntFunc int_func) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr) {
    return rhs;
  } else if (rhs == nullptr || lhs == rhs) {
    return lhs;
  }
  auto lhs_int = dynamic_cast<Int*>(lhs);
  auto rhs_int = dynamic_cast<Int*>(rhs);
  if (lhs_int != nullptr && rhs_int != nullptr) {
    return minOrMaxExpr(lhs_int, rhs_int, ir_builder_func, int_func);
  } else {
    return ir_builder_func(lhs, rhs);
  }
}

} // namespace

Val* SimplifyingIrBuilder::maxExpr(Val* lhs, Val* rhs) {
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::maxExpr(lhs, rhs); },
      [](int64_t lhs, int64_t rhs) { return std::max(lhs, rhs); });
}

Val* SimplifyingIrBuilder::minExpr(Val* lhs, Val* rhs) {
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::minExpr(lhs, rhs); },
      [](int64_t lhs, int64_t rhs) { return std::min(lhs, rhs); });
}

Val* SimplifyingIrBuilder::gcdExpr(Val* lhs, Val* rhs) {
  if (lhs->isZeroInt()) {
    return rhs;
  }
  if (rhs->isZeroInt()) {
    return lhs;
  }
  if (lhs->isOneInt() || rhs->isOneInt()) {
    return lhs->container()->oneVal();
  }
  return IrBuilder::gcdExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(
      pred->dtype() == DataType::Bool,
      "Where requires a predicate as an input, but received");

  if (pred->isConstScalar() && pred->isABool() && pred->isA<Bool>()) {
    if (pred->evaluateBool()) {
      return lhs;
    } else {
      return rhs;
    }
  }

  return IrBuilder::whereExpr(pred, lhs, rhs);
}

} // namespace nvfuser
