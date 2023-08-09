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

Val* IrBuilder::newScalar(DataType dtype) {
  return IrBuilder::create<Val>(dtype);
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

Val* IrBuilder::newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs) {
  TORCH_CHECK(
      lhs != nullptr && rhs != nullptr,
      "Either lhs or rhs is a nullptr in newLogicExpr.");
  auto result = newScalar(DataType::Bool);
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

Val* IrBuilder::logicalNotExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in logicalNotExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::LogicalNot, result, val);
  return result;
}

Val* IrBuilder::bitwiseNotExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in bitwiseNotExpr.");
  auto result = newScalar(val->dtype());
  IrBuilder::create<UnaryOp>(UnaryOpType::BitwiseNot, result, val);
  return result;
}

Val* IrBuilder::derefExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in derefExpr.");
  auto result = newScalar(*(std::get<PointerOf>(val->dtype().type).type));
  IrBuilder::create<UnaryOp>(UnaryOpType::Dereference, result, val);
  return result;
}

Val* IrBuilder::absExpr(Val* val) {
  TORCH_CHECK(val != nullptr, "val is a nullptr in absExpr.");
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
  auto item_dtype = std::get<ArrayOf>(array->dtype().type).type;
  auto out = newScalar(*item_dtype);
  create<GetItem>(array->container(), out, array, index);
  return out;
}

Val* IrBuilder::getItemExpr(Val* array, PolymorphicValue index) {
  auto item_dtype = std::get<ArrayOf>(array->dtype().type).type;
  auto out = newScalar(*item_dtype);
  create<GetItem>(
      array->container(), out, array, newConstant(index, DataType::Int));
  return out;
}

Val* IrBuilder::getAttrExpr(Val* struct_, std::string attr) {
  auto item_dtype = NVFUSER_MAYBE_STAR std::get<StructOf>(struct_->dtype().type)
                        .types.at(attr);
  auto out = newScalar(item_dtype);
  create<GetAttr>(struct_->container(), out, struct_, std::move(attr));
  return out;
}

Val* IrBuilder::metadataExpr(TensorView* tv) {
  return tv->fusion()->metadataOf(tv);
}

Val* SimplifyingIrBuilder::negExpr(Val* val) {
  if (val->isZeroInt()) {
    return val->container()->zeroVal();
  } else if (auto scalar = dynamic_cast<Val*>(val)) {
    if (scalar->isConst()) {
      return IrBuilder::create<Val>(-scalar->value(), scalar->dtype());
    }
  }
  return IrBuilder::negExpr(val);
}

Val* SimplifyingIrBuilder::logicalNotExpr(Val* val) {
  if (auto scalar = dynamic_cast<Val*>(val)) {
    if (scalar->isConst()) {
      if (scalar->value()) {
        return FusionGuard::getCurFusion()->falseVal();
      } else {
        return FusionGuard::getCurFusion()->trueVal();
      }
    }
  }
  return IrBuilder::logicalNotExpr(val);
}

Val* SimplifyingIrBuilder::bitwiseNotExpr(Val* val) {
  if (auto scalar = dynamic_cast<Val*>(val); scalar->isConst()) {
    return IrBuilder::create<Val>(~(scalar->value()), scalar->dtype());
  }
  return IrBuilder::bitwiseNotExpr(val);
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, PolymorphicValue rhs) {
  if (rhs == 0) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::IrBuilder::create<Val>(rhs);
  }
  auto target_dtype = promoteType(lhs->dtype(), getDataType(rhs));
  if (lhs->isConst()) {
    return IrBuilder::IrBuilder::create<Val>(lhs->value() + rhs, target_dtype);
  } else if (rhs > 0) {
    return IrBuilder::addExpr(lhs, IrBuilder::IrBuilder::create<Val>(rhs));
  } else {
    return IrBuilder::subExpr(lhs, IrBuilder::IrBuilder::create<Val>(-rhs));
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
    return addExpr(lhs, rhs->value());
  } else {
    return IrBuilder::addExpr(lhs, rhs);
  }
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  return addExpr(lhs, negExpr(rhs));
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, PolymorphicValue rhs) {
  if (rhs == 0) {
    return lhs->container()->zeroVal();
  } else if (rhs == 1) {
    return lhs;
  } else if (lhs == nullptr) {
    return IrBuilder::create<Val>(rhs);
  } else if (lhs->isConst()) {
    return IrBuilder::create<Val>(lhs->value() * rhs);
  } else {
    return IrBuilder::mulExpr(lhs, IrBuilder::create<Val>(rhs));
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
    return IrBuilder::IrBuilder::create<Val>(ceildiv(l, r));
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

Val* SimplifyingIrBuilder::logicalAndExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
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

  return IrBuilder::logicalAndExpr(lhs, rhs);
}

Val* SimplifyingIrBuilder::logicalOrExpr(Val* lhs, Val* rhs) {
  auto lhs_scalar = dynamic_cast<Val*>(lhs);
  auto rhs_scalar = dynamic_cast<Val*>(rhs);
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
  TORCH_INTERNAL_ASSERT(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

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
    return FusionGuard::getCurFusion()->zeroVal();
  } else if (lhs_all_ones && rhs_all_ones) {
    return IrBuilder::IrBuilder::create<Val>(-1, lhs->dtype());
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
  TORCH_INTERNAL_ASSERT(!(lhs_scalar == nullptr && rhs_scalar == nullptr));

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
    return IrBuilder::IrBuilder::create<Val>(-1, lhs->dtype());
  } else if (lhs_zero && rhs_zero) {
    return FusionGuard::getCurFusion()->zeroVal();
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
  } else if (lhs == nullptr) {
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

  if (pred->isConstScalar() && pred->isABool()) {
    if (pred->evaluateBool()) {
      return lhs;
    } else {
      return rhs;
    }
  }

  return IrBuilder::whereExpr(pred, lhs, rhs);
}

} // namespace nvfuser
