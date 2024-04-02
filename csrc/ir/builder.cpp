// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_evaluator.h>
#include <expr_simplifier.h>
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

Val* IrBuilder::lShiftExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Lshift, lhs, rhs);
}

Val* IrBuilder::rShiftExpr(Val* lhs, Val* rhs) {
  return newArithmeticExpr(BinaryOpType::Rshift, lhs, rhs);
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
  return simplifyExpr(IrBuilder::negExpr(val), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::logicalNotExpr(Val* val) {
  return simplifyExpr(IrBuilder::logicalNotExpr(val), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::bitwiseNotExpr(Val* val) {
  return simplifyExpr(IrBuilder::bitwiseNotExpr(val), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::maybeCastExpr(DataType dtype, Val* val) {
  return simplifyExpr(IrBuilder::maybeCastExpr(dtype, val), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::addExpr(
    Val* lhs,
    PolymorphicValue rhs,
    DataType rhs_dtype) {
  if (rhs_dtype == DataType::Null) {
    rhs_dtype = getDataType(rhs);
  }
  return simplifyExpr(
      IrBuilder::addExpr(
          lhs, IrBuilder::IrBuilder::create<Val>(rhs, rhs_dtype)),
      {},
      {},
      false,
      10);
}

Val* SimplifyingIrBuilder::addExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::addExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::subExpr(Val* lhs, Val* rhs) {
  return addExpr(lhs, negExpr(rhs));
}

Val* SimplifyingIrBuilder::mulExpr(
    Val* lhs,
    PolymorphicValue rhs,
    DataType rhs_dtype) {
  if (rhs_dtype == DataType::Null) {
    rhs_dtype = getDataType(rhs);
  }
  return simplifyExpr(
      IrBuilder::mulExpr(lhs, IrBuilder::create<Val>(rhs, rhs_dtype)),
      {},
      {},
      false,
      10);
}

Val* SimplifyingIrBuilder::mulExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::mulExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::divExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::divExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::ceilDivExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::ceilDivExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::modExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::modExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::logicalAndExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::logicalAndExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::logicalOrExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::logicalOrExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::bitwiseAndExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::bitwiseAndExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::bitwiseOrExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::bitwiseOrExpr(lhs, rhs), {}, {}, false, 10);
}

namespace {

template <typename IrBuilderFunc, typename Fimc>
Val* minOrMaxExpr(
    Val* lhs,
    Val* rhs,
    IrBuilderFunc ir_builder_func,
    Fimc func) {
  return simplifyExpr(ir_builder_func(lhs, rhs), {}, {}, false, 10);
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
  return simplifyExpr(IrBuilder::gcdExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::ltExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::ltExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::leExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::leExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::eqExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::eqExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::neExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::neExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::geExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::geExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::gtExpr(Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::gtExpr(lhs, rhs), {}, {}, false, 10);
}

Val* SimplifyingIrBuilder::whereExpr(Val* pred, Val* lhs, Val* rhs) {
  return simplifyExpr(IrBuilder::whereExpr(pred, lhs, rhs), {}, {}, false, 10);
}

} // namespace nvfuser
