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
  NVF_ERROR(isVal(), "Cannot cast to Val as this is not a Val.");
  return this->as<Val>();
}

Expr* Statement::asExpr() {
  NVF_ERROR(isExpr(), "Cannot cast to Expr as this is not a Expr.");
  return this->as<Expr>();
}

bool Statement::lessThan(const Statement* stmt1, const Statement* stmt2) {
  NVF_ERROR(stmt1 != nullptr);
  NVF_ERROR(stmt2 != nullptr);
  return stmt1->name() < stmt2->name();
}

std::string Statement::toString(int indent_size) const {
  NVF_ERROR(
      false, "toString for IR node ", typeid(*this).name(), " is not defined");
}

std::string Statement::toInlineString(int indent_size) const {
  NVF_ERROR(
      false,
      "toInlineString for IR node ",
      typeid(*this).name(),
      " is not defined");
}

Fusion* Statement::fusion() const {
  NVF_ERROR(
      ir_container_->isA<Fusion>(), "Statement does not belong to a fusion.");
  return ir_container_->as<Fusion>();
}

kir::Kernel* Statement::kernel() const {
  NVF_ERROR(
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
  } else if (isFloatingPointType(dtype) || isComplexType(dtype)) {
    ss << dtype << "(" << std::setprecision(max_digits10(dtype)) << value()
       << ")";
  } else {
    ss << value();
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
  return ir_utils::dependenciesSatisfied(this);
}

bool Val::isConstInt() const {
  return ir_utils::dependenciesSatisfied(this) && isIntegralScalar();
}

PolymorphicValue Val::evaluate() {
  if (this->value().hasValue()) {
    return this->value();
  }

  ExpressionEvaluator ee;
  auto evaluated_val = ee.evaluate(this);
  NVF_ERROR(
      evaluated_val.hasValue(),
      "Detected a const value but failed to infer its value: ",
      toInlineString());
  return evaluated_val;
}

bool Val::isZero() const {
  return value().hasValue() && value() == 0;
}

bool Val::isZeroInt() const {
  return value().hasValue() && value().is<int64_t>() && value() == 0;
}

bool Val::isOne() const {
  return value().hasValue() && value() == 1;
}

bool Val::isOneInt() const {
  return value().hasValue() && value().is<int64_t>() && value() == 1;
}

bool Val::isTrue() const {
  return value().hasValue() && value().is<bool>() && value().as<bool>();
}

bool Val::isFalse() const {
  return value().hasValue() && value().is<bool>() && !value().as<bool>();
}

std::optional<DataType> Val::getDataType() const {
  NVF_ERROR(dtype_ != DataType::Null, "Value does not have a data type.");
  return dtype_;
}

bool Val::isProducerOf(const Val* other) const {
  NVF_ERROR(other != nullptr);
  NVF_ERROR(container() == other->container());

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
  NVF_CHECK(old_val, "Pre-concretized value was null");
  NVF_CHECK(new_val, "Concretized value is null");
  NVF_CHECK(
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
  NVF_ERROR(container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return predicate_;
}

void Expr::setPredicate(kir::Predicate* predicate) {
  NVF_ERROR(container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  predicate_ = predicate;
}

Expr* Expr::withPredicate(kir::Predicate* predicate) {
  auto result = shallowCopy();
  result->setPredicate(predicate);
  return result;
}

kir::Predicate* Expr::writePredicate() const {
  NVF_ERROR(container()->isA<kir::Kernel>(), "Function invalid for fusion.");
  return write_predicate_;
}

void Expr::setWritePredicate(kir::Predicate* write_predicate) {
  NVF_ERROR(container()->isA<kir::Kernel>(), "Function invalid for fusion.");
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
  NVF_ERROR(
      false,
      "`evaluate` method for expression ",
      getOpString(),
      " is not defined. ",
      "Please override the evaluate method");
}

void Expr::addDataAttribute(PolymorphicValue attr) {
  addAttribute(IrBuilder::create<Val>(container(), std::move(attr)));
}

} // namespace nvfuser
