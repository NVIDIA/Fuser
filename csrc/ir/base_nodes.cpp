// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <dispatch.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <kernel.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>

#include <torch/csrc/jit/ir/ir.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace nvfuser {

Statement::Statement(IrBuilderPasskey passkey)
    : ir_container_{passkey.ir_container_} {}

Statement::Statement(const Statement* src, IrCloner* ir_cloner)
    : ir_container_{ir_cloner->container()} {}

NVFUSER_DEFINE_CLONE(Statement)

void Statement::setName(IrContainerPasskey, StmtNameType name) {
  name_ = name;
}

void Statement::setName(IrBuilderPasskey, StmtNameType name) {
  name_ = name;
}

Val* Statement::asVal() {
  TORCH_INTERNAL_ASSERT(isVal(), "Cannot cast to Val as this is not a Val.");
  return this->as<Val>();
}

Expr* Statement::asExpr() {
  TORCH_INTERNAL_ASSERT(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return this->as<Expr>();
}

bool Statement::lessThan(const Statement* stmt1, const Statement* stmt2) {
  TORCH_INTERNAL_ASSERT(stmt1 != nullptr);
  TORCH_INTERNAL_ASSERT(stmt2 != nullptr);
  return stmt1->name() < stmt2->name();
}

std::string Statement::toString(int indent_size) const {
  TORCH_INTERNAL_ASSERT(
      false, "toString for IR node ", typeid(*this).name(), " is not defined");
}

std::string Statement::toInlineString(int indent_size) const {
  TORCH_INTERNAL_ASSERT(
      false,
      "toInlineString for IR node ",
      typeid(*this).name(),
      " is not defined");
}

Fusion* Statement::fusion() const {
  TORCH_INTERNAL_ASSERT(
      ir_container_->isA<Fusion>(), "Statement does not belong to a fusion.");
  return ir_container_->as<Fusion>();
}

kir::Kernel* Statement::kernel() const {
  TORCH_INTERNAL_ASSERT(
      ir_container_->isA<kir::Kernel>(),
      "Statement does not belong to a kernel.");
  return ir_container_->as<kir::Kernel>();
}

NVFUSER_DEFINE_CLONE(Val)

const std::vector<Expr*>& Val::uses() const {
  if (vtype_ == ValType::TensorView) {
    if (!fusion()->isTVUseInfoValid() && !fusion()->isUpdatingTVUseInfo()) {
      fusion()->resetTvUses();
    }
  }
  return uses_;
}

bool Val::addUse(Expr* expr) {
  if (std::find(uses_.begin(), uses_.end(), expr) == uses_.end()) {
    uses_.push_back(expr);
    return true;
  }
  return false;
}

bool Val::removeUse(Expr* expr) {
  auto it = std::find(uses_.begin(), uses_.end(), expr);
  if (it != uses_.end()) {
    uses_.erase(it);
    if (this->isA<TensorView>()) {
      // Call for a rebuild of uses_ vector
      fusion()->invalidateTvUses();
    }
    return true;
  }
  return false;
}

// Converts the data type of TensorView or Scalar representing index
// values. The data type of the original input should be
// DataType::Index, but DataType::Int is also allowed as it is used
// for index expressions.
// TODO: remove this function. I think we are fine removing this now, but I need
// to double check the benchmarks.
void Val::resolveIndexDtype() {
  TORCH_INTERNAL_ASSERT(
      vtype_ == ValType::TensorView || vtype_ == ValType::Others ||
          vtype_ == ValType::NamedScalar,
      "Resolving index type is currently only supported on tensor view or scalar values. "
      "Value type: ",
      vtype_);
  TORCH_INTERNAL_ASSERT(
      isIntegralType(dtype_),
      "Can only resolve index type if a Val has an Index or Int DataType. ",
      "Data type: ",
      dtype_);
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(),
      "Index type can only be resolved at compile time.");
  auto index_dtype = container()->as<kir::Kernel>()->indexType();
  TORCH_INTERNAL_ASSERT(
      index_dtype == DataType::Int || index_dtype == DataType::Int32,
      "Invalid index data type: ",
      index_dtype);
  dtype_ = DataType::Index;
}

bool Val::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (auto other_val = dynamic_cast<const Val*>(other)) {
    if (typeid(*this) != typeid(*other_val)) {
      return false;
    }
    if ((definition_ == nullptr) != (other_val->definition_ == nullptr)) {
      return false;
    }
    if (vtype_ != other_val->vtype_) {
      return false;
    }
    if (dtype_ != other_val->dtype_) {
      return false;
    }
    if (value_.hasValue() != other_val->value_.hasValue()) {
      return false;
    }
    if (definition_ == nullptr) {
      if (value_.hasValue()) {
        return value_ == other_val->value_;
      } else {
        return false;
      }
    }
    if (!definition_->sameAs(other_val->definition_)) {
      return false;
    }
    if (definition_->outputs().size() !=
        other_val->definition_->outputs().size()) {
      return false;
    }
    // For definition with multiple outputs, only outputs at the same position
    // could be the same
    for (auto i : c10::irange(definition_->outputs().size())) {
      if ((definition_->output(i) == this) !=
          (other_val->definition_->output(i) == other_val)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string Val::toString(int indent_size) const {
  std::stringstream ss;
  if (isSymbolic()) {
    ss << ir_utils::varName(this);
    return ss.str();
  }
  auto dtype = getDataType().value();
  if (dtype == DataType::Bool) {
    ss << (value() ? "true" : "false");
  } else if (isIntegralType(dtype)) {
    ss << value();
  } else if (isFloatingPointType(dtype) || isComplexType(dtype)) {
    ss << dtype << "(" << std::setprecision(max_digits10(dtype)) << value()
       << ")";
  } else if (dtype == DataType::Opaque) {
    ss << "<opaque value>";
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unknown scalar type: ", dtype);
  }
  return ss.str();
}

std::string Val::toInlineString(int indent_size) const {
  if (definition() != nullptr) {
    std::stringstream ss;
    ss << "( " << definition()->toInlineString(indent_size) << " )";
    return ss.str();
  } else {
    return toString(indent_size);
  }
}

bool Val::isConstScalar() const {
  if (!isScalar()) {
    return false;
  }
  return ir_utils::dependenciesSatisfied({this});
}

bool Val::isConstInt() const {
  return ir_utils::dependenciesSatisfied({this}) && isIntegralScalar();
}

int64_t Val::evaluateInt() {
  TORCH_INTERNAL_ASSERT(
      ir_utils::dependenciesSatisfied(std::vector<const Val*>{this}),
      "Cannot get Int of not const values through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->value().hasValue()) {
    return this->value().as<int64_t>();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.hasValue(),
      "Detected a const integer but failed to infer its value: ",
      toInlineString());
  return evaluated_val.as<int64_t>();
}

double Val::evaluateDouble() {
  TORCH_INTERNAL_ASSERT(
      ir_utils::dependenciesSatisfied(std::vector<const Val*>{this}),
      "Cannot get Double of not const doubles through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->value().hasValue()) {
    return this->value().as<double>();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.hasValue(),
      "Detected a const integer but failed to infer its value.");
  return evaluated_val.as<double>();
}

bool Val::evaluateBool() {
  TORCH_INTERNAL_ASSERT(
      ir_utils::dependenciesSatisfied(std::vector<const Val*>{this}),
      "Cannot get Bool of not const bools through IR nodes, must use runtime ExpressionEvaluator.");

  if (this->value().hasValue()) {
    return this->value().as<bool>();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  TORCH_INTERNAL_ASSERT(
      evaluated_val.hasValue(),
      "Detected a const integer but failed to infer its value.");
  return evaluated_val.as<bool>();
}

std::optional<int64_t> Val::getInt() const {
  if (isConstScalar() && isIntegralScalar()) {
    auto val = this->value();
    if (val.is<int64_t>()) {
      return val.as<int64_t>();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<double> Val::getDouble() const {
  if (isConstScalar() && isFloatingPointScalar()) {
    auto val = this->value();
    if (val.is<double>()) {
      return val.as<double>();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<bool> Val::getBool() const {
  if (isConstScalar() && isABool()) {
    auto val = this->value();
    if (val.is<bool>()) {
      return val.as<bool>();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool Val::isZero() const {
  return getInt() == 0 || getDouble() == 0.0;
}

bool Val::isZeroInt() const {
  auto int_val = getInt();
  return int_val.has_value() && int_val.value() == 0;
}

bool Val::isOne() const {
  return getInt() == 1 || getDouble() == 1.0;
}

bool Val::isOneInt() const {
  auto int_val = getInt();
  return int_val.has_value() && int_val.value() == 1;
}

bool Val::isTrue() const {
  return getBool() == true;
}

bool Val::isFalse() const {
  return getBool() == false;
}

std::optional<DataType> Val::getDataType() const {
  TORCH_INTERNAL_ASSERT(
      dtype_ != DataType::Null, "Value does not have a data type.");
  return dtype_;
}

bool Val::isProducerOf(const Val* other) const {
  TORCH_INTERNAL_ASSERT(other != nullptr);
  TORCH_INTERNAL_ASSERT(container() == other->container());

  if (definition() == nullptr) {
    return false;
  }
  return std::any_of(
      definition()->inputs().begin(),
      definition()->inputs().end(),
      [other](const Val* input) { return input == other; });
}

bool Val::isConsumerOf(const Val* other) const {
  return other->isProducerOf(this);
}

// We don't register with the active fusion in Expr as this needs to be done
// after inputs and outputs are registered with the Expr
Expr::Expr(IrBuilderPasskey passkey) : Statement(passkey) {}

Expr::Expr(const Expr* src, IrCloner* ir_cloner)
    : Statement(src, ir_cloner),
      attributes_(ir_cloner->clone(src->attributes_)),
      inputs_(ir_cloner->clone(src->inputs_)),
      outputs_(ir_cloner->clone(src->outputs_)) {}

Expr::Expr(
    IrBuilderPasskey passkey,
    std::vector<Val*> inputs,
    std::vector<Val*> outputs,
    std::vector<Statement*> attributes)
    : Statement(passkey),
      attributes_(std::move(attributes)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)) {}

Expr* Expr::shallowCopy() const {
  auto result =
      newObjectFunc()(ir_container_, inputs(), outputs(), attributes());
  if (container()->isA<kir::Kernel>()) {
    result->predicate_ = predicate_;
    result->write_predicate_ = write_predicate_;
  }
  return result;
}

std::string Expr::getGraphvizLabel() const {
  if (attributes().empty()) {
    return getOpString();
  }
  std::stringstream ss;
  const char* separator = "";
  ss << "{" << getOpString() << "|{";
  for (auto attr : attributes()) {
    ss << separator << attr->toString();
    separator = "|";
  }
  ss << "}}";
  return ss.str();
}

void Expr::checkConcretization(Val* old_val, Val* new_val) const {
  TORCH_CHECK(old_val, "Pre-concretized value was null");
  TORCH_CHECK(new_val, "Concretized value is null");
  TORCH_CHECK(
      old_val->vtype() == new_val->vtype(),
      "Concretization must not change ValType");
}

bool Expr::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<Expr>()) {
    return false;
  }
  const Expr* other_expr = other->as<Expr>();
  if (typeid(*this) != typeid(*other_expr)) {
    return false;
  }
  if (inputs().size() != other_expr->inputs().size() ||
      outputs().size() != other_expr->outputs().size() ||
      attributes().size() != other_expr->attributes().size()) {
    return false;
  }
  for (const auto i : c10::irange(inputs().size())) {
    if (!input(i)->sameAs(other_expr->input(i))) {
      return false;
    }
  }
  for (const auto i : c10::irange(attributes().size())) {
    if (!attribute(i)->sameAs(other_expr->attribute(i))) {
      return false;
    }
  }
  return true;
}

kir::Predicate* Expr::predicate() const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return predicate_;
}

void Expr::setPredicate(kir::Predicate* predicate) {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  predicate_ = predicate;
}

Expr* Expr::withPredicate(kir::Predicate* predicate) {
  auto result = shallowCopy();
  result->setPredicate(predicate);
  return result;
}

kir::Predicate* Expr::writePredicate() const {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return write_predicate_;
}

void Expr::setWritePredicate(kir::Predicate* write_predicate) {
  TORCH_INTERNAL_ASSERT(
      container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  write_predicate_ = write_predicate;
}

Expr* Expr::withWritePredicate(kir::Predicate* predicate) {
  auto result = shallowCopy();
  result->setWritePredicate(predicate);
  return result;
}

std::vector<PolymorphicValue> Expr::evaluate(
    const ExpressionEvaluator& ee,
    const std::vector<PolymorphicValue>& inputs) const {
  TORCH_INTERNAL_ASSERT(
      false,
      "`evaluate` method for expression ",
      getOpString(),
      " is not defined. ",
      "Please override the evaluate method");
}

void Expr::addScalarAttribute(PolymorphicValue attr) {
  addAttribute(IrBuilder::create<Val>(container(), std::move(attr)));
}

} // namespace nvfuser
