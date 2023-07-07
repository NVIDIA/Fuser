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
#include <cstdint>

namespace nvfuser {

Scalar* IrBuilder::newScalar(DataType dtype) {
  return IrBuilder::create<Scalar>(dtype);
}

Scalar* IrBuilder::newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
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

Scalar* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  TORCH_CHECK(
      lhs != nullptr && rhs != nullptr,
      "Either lhs or rhs is a nullptr in newLogicExpr.");
  auto result = newScalar(DataType::Bool);
  IrBuilder::create<BinaryOp>(op_type, result, lhs, rhs);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return result;
}

Scalar* IrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  TORCH_CHECK(
      pred != nullptr && lhs != nullptr && rhs != nullptr,
      "Either pred, lhs, or rhs is a nullptr in whereExpr.");
  TORCH_CHECK(lhs->dtype() == rhs->dtype(), "Incompatible operand types");
  auto result = newScalar(lhs->dtype());
  IrBuilder::create<TernaryOp>(TernaryOpType::Where, result, pred, lhs, rhs);
  return result;
}

Scalar* IrBuilder::negExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in negExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Neg, result, val);
  return result;
}

Scalar* IrBuilder::notExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in notExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Not, result, val);
  return result;
}

Scalar* IrBuilder::absExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in notExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::Abs, result, val);
  return result;
}

Scalar* IrBuilder::setExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in setExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

NamedScalar* IrBuilder::setExprNamedScalar(const std::string& name, Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in setExprNamedScalar.");
  auto result = IrBuilder::create<NamedScalar>(name, val->dtype());
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, result, val);
  return result;
}

NamedScalar* IrBuilder::addressExprNamedScalar(
    const std::string& name,
    Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in addressExprNamedScalar.");
  auto result = IrBuilder::create<NamedScalar>(name, DataType::Int);
  IrBuilder::create<UnaryOp>(UnaryOpType::Address, result, val);
  return result;
}

Scalar* IrBuilder::andExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::And, lhs, rhs);
}

Scalar* IrBuilder::orExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Or, lhs, rhs);
}

Scalar* IrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::Eq, lhs, rhs);
}

Scalar* IrBuilder::neExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::NE, lhs, rhs);
}

Scalar* IrBuilder::gtExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GT, lhs, rhs);
}

Scalar* IrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LT, lhs, rhs);
}

Scalar* IrBuilder::leExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::LE, lhs, rhs);
}

Scalar* IrBuilder::geExpr(Val* lhs, Val* rhs) {
  return newLogicExpr(BinaryOpType::GE, lhs, rhs);
}

Scalar* IrBuilder::addExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Add, lhs, rhs);
}

Scalar* IrBuilder::subExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Sub, lhs, rhs);
}

Scalar* IrBuilder::mulExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mul, lhs, rhs);
}

Scalar* IrBuilder::divExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Div, lhs, rhs);
}

Scalar* IrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::CeilDiv, lhs, rhs);
}

Scalar* IrBuilder::modExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Mod, lhs, rhs);
}

Scalar* IrBuilder::maxExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Max, lhs, rhs);
}

Scalar* IrBuilder::minExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Min, lhs, rhs);
}

Scalar* IrBuilder::gcdExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Gcd, lhs, rhs);
}

Scalar* IrBuilder::getItemExpr(Val* array, Val* index) {
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
  } else if (auto scalar = dynamic_cast<Scalar*>(val)) {
    if (scalar->isConst()) {
      return IrBuilder::create<Scalar>(-scalar->value(), scalar->dtype());
    }
  }
  return IrBuilder::negExpr(val);
}

Scalar* SimplifyingIrBuilder::notExpr(Val* val) {
  if (auto scalar = dynamic_cast<Scalar*>(val)) {
    if (scalar->isConst()) {
      if (scalar->value()) {
        return FusionGuard::getCurFusion()->falseVal();
      } else {
        return FusionGuard::getCurFusion()->trueVal();
      }
    }
  }
  return IrBuilder::notExpr(val);
}

Scalar* SimplifyingIrBuilder::addExpr(Scalar* lhs, ScalarValue rhs) {
  if (rhs == 0) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::IrBuilder::create<Scalar>(rhs, DataType::Int);
  }
  auto target_dtype = promoteType(lhs->dtype(), getDataType(rhs));
  if (lhs->isConst()) {
    return IrBuilder::IrBuilder::create<Scalar>(
        lhs->value() + rhs, target_dtype);
  } else if (rhs > 0) {
    return IrBuilder::addExpr(lhs, IrBuilder::IrBuilder::create<Scalar>(rhs));
  } else {
    return IrBuilder::subExpr(lhs, IrBuilder::IrBuilder::create<Scalar>(-rhs));
  }
}

Scalar* SimplifyingIrBuilder::addExpr(Scalar* lhs, Scalar* rhs) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst()) {
    return addExpr(rhs, lhs->value());
  } else if (rhs->isConst()) {
    return addExpr(lhs, rhs->value());
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
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  auto rhs_scalar = dynamic_cast<Scalar*>(rhs);
  if (lhs_scalar != nullptr && rhs_scalar != nullptr) {
    return addExpr(lhs_scalar, rhs_scalar);
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, ScalarValue rhs) {
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  if (lhs_scalar != nullptr) {
    return addExpr(lhs_scalar, rhs);
  } else {
    return addExpr(lhs, IrBuilder::create<Scalar>(rhs));
  }
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  return addExpr(lhs, negExpr(rhs));
}

Scalar* SimplifyingIrBuilder::mulExpr(Scalar* lhs, ScalarValue rhs) {
  if (rhs == 0) {
    return lhs->container()->zeroVal();
  } else if (rhs == 1) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::create<Scalar>(rhs);
  } else if (lhs->isConst()) {
    return IrBuilder::create<Scalar>(lhs->value() * rhs);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Scalar>(rhs));
  }
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, ScalarValue rhs) {
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  if (lhs_scalar != nullptr) {
    return mulExpr(lhs_scalar, rhs);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Scalar>(rhs));
  }
}

Scalar* SimplifyingIrBuilder::mulExpr(Scalar* lhs, Scalar* rhs) {
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

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr || lhs->isOneInt()) {
    return rhs;
  } else if (rhs == nullptr || rhs->isOneInt()) {
    return lhs;
  } else if (lhs->isZeroInt() || rhs->isZeroInt()) {
    return lhs->container()->zeroVal();
  }
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  auto rhs_scalar = dynamic_cast<Scalar*>(rhs);
  if (lhs_scalar != nullptr && rhs_scalar != nullptr) {
    return mulExpr(lhs_scalar, rhs_scalar);
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

Scalar* SimplifyingIrBuilder::ceilDivExpr(Scalar* lhs, Scalar* rhs) {
  if (rhs->isOneInt()) {
    return lhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    auto l = lhs->value();
    auto r = rhs->value();
    if (r > 0) {
      return IrBuilder::IrBuilder::create<Scalar>((l + r - 1) / r);
    } else {
      return IrBuilder::IrBuilder::create<Scalar>((l + r + 1) / r);
    }
  } else {
    return IrBuilder::ceilDivExpr(lhs, rhs);
  }
}

Scalar* SimplifyingIrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr && rhs != nullptr);
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  auto rhs_scalar = dynamic_cast<Scalar*>(rhs);
  if (lhs_scalar != nullptr && rhs_scalar != nullptr) {
    return ceilDivExpr(lhs_scalar, rhs_scalar);
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

Scalar* SimplifyingIrBuilder::andExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  auto rhs_scalar = dynamic_cast<Scalar*>(rhs);
  TORCH_INTERNAL_ASSERT(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

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

  return IrBuilder::andExpr(lhs, rhs);
}

namespace {

template <typename IrBuilderFunc, typename ScalarFunc>
Val* minOrMaxExpr(
    Scalar* lhs,
    Scalar* rhs,
    IrBuilderFunc ir_builder_func,
    ScalarFunc scalar_func) {
  if (rhs == nullptr) {
    return lhs;
  } else if (lhs == nullptr) {
    return rhs;
  } else if (lhs->isConst() && rhs->isConst()) {
    return IrBuilder::create<Scalar>(scalar_func(lhs->value(), rhs->value()));
  } else {
    return ir_builder_func(lhs, rhs);
  }
}

template <typename IrBuilderFunc, typename ScalarFunc>
Val* minOrMaxExpr(
    Val* lhs,
    Val* rhs,
    IrBuilderFunc ir_builder_func,
    ScalarFunc scalar_func) {
  TORCH_INTERNAL_ASSERT(lhs != nullptr || rhs != nullptr);
  if (lhs == nullptr) {
    return rhs;
  } else if (rhs == nullptr || lhs == rhs) {
    return lhs;
  }
  auto lhs_scalar = dynamic_cast<Scalar*>(lhs);
  auto rhs_scalar = dynamic_cast<Scalar*>(rhs);
  if (lhs_scalar != nullptr && rhs_scalar != nullptr) {
    return minOrMaxExpr(lhs_scalar, rhs_scalar, ir_builder_func, scalar_func);
  } else {
    return ir_builder_func(lhs, rhs);
  }
}

} // namespace

Val* SimplifyingIrBuilder::maxExpr(Val* lhs, Val* rhs) {
  using namespace ScalarValue_functions;
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::maxExpr(lhs, rhs); },
      [](auto lhs, auto rhs) { return max(lhs, rhs); });
}

Val* SimplifyingIrBuilder::minExpr(Val* lhs, Val* rhs) {
  using namespace ScalarValue_functions;
  return minOrMaxExpr(
      lhs,
      rhs,
      [](Val* lhs, Val* rhs) { return IrBuilder::minExpr(lhs, rhs); },
      [](auto lhs, auto rhs) { return min(lhs, rhs); });
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

  if (pred->isConstScalar() && pred->isABool() && pred->isA<Scalar>()) {
    if (pred->evaluateBool()) {
      return lhs;
    } else {
      return rhs;
    }
  }

  return IrBuilder::whereExpr(pred, lhs, rhs);
}

} // namespace nvfuser
