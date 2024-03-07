// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <simplification/ast.h>

namespace nvfuser {

namespace simplification {

/*static*/ thread_local Program* ProgramGuard::active_program_ = nullptr;

ProgramGuard::ProgramGuard(Program* program) : prev_program_(active_program_) {
  active_program_ = program;
}

ProgramGuard::~ProgramGuard() {
  active_program_ = prev_program_;
}

/*static*/ Program* ProgramGuard::getCurProgram() {
  return active_program_;
}

/*static*/ void ProgramGuard::setCurProgram(Program* program) {
  active_program_ = program;
}

FunctionSymbol exprToFunctionSymbol(Expr* expr) {
  if (expr == nullptr) {
    return std::monostate{};
  } else if (auto* uop = dynamic_cast<UnaryOp*>(expr)) {
    return uop->getUnaryOpType();
  } else if (auto bop = dynamic_cast<BinaryOp*>(expr)) {
    return bop->getBinaryOpType();
  } else if (auto top = dynamic_cast<TernaryOp*>(expr)) {
    return top->getTernaryOpType();
  }
  NVF_ERROR(false, "Unsupported expression in AST: ", expr->toString());
}

size_t symbolId(FunctionSymbol symbol) {
  size_t id = 0;
  if (isUnaryOpType(symbol)) {
    id = (1LL << 32) + toUnderlying(std::get<UnaryOpType>(symbol));
  } else if (isBinaryOpType(symbol)) {
    id = (2LL << 32) + toUnderlying(std::get<BinaryOpType>(symbol));
  } else if (isTernaryOpType(symbol)) {
    id = (3LL << 32) + toUnderlying(std::get<TernaryOpType>(symbol));
  }
  return id;
}

Term::operator bool() const {
  if (constant.is<bool>()) {
    return constant.as<bool>();
  }
  Program* program = ProgramGuard::getCurProgram();
  return program->isProvenTrue(*this);
}

// Convenience function for creating new terms using the current Program
static const Term& term(
    FunctionSymbol symbol,
    PolymorphicValue constant,
    const std::vector<const Term*>& producer_terms) {
  Program* program = ProgramGuard::getCurProgram();
  return *(program->makeTerm(symbol, constant, producer_terms));
}

const Term& Term::operator==(const Term& other) {
  if (constant.hasValue() && other.constant.hasValue()) {
    // Fold constants
    return term(BinaryOpType::Eq, constant == other.constant, {});
  }
  return term(BinaryOpType::Eq, std::monostate{}, {this, &other});
}

const Term& operator+(const Term& a, const Term& b) {
  return (a.constant.hasValue() && b.constant.hasValue())
      ? term(std::monostate{}, a.constant + b.constant, {})
      : term(BinaryOpType::Add, std::monostate{}, {&a, &b});
}

} // namespace simplification

} // namespace nvfuser
