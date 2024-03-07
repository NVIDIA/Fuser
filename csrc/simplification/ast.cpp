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

const Term& Term::operator==(const Term& other) {
  Program* program = ProgramGuard::getCurProgram();
  return *(program->makeTerm(BinaryOpType::Eq, std::monostate{}, {this, &other}));
}
/*#define TERM_BINARY_OP(cppname, optypename) \
  const Term& cppname(const Term& b) { \
    Program* = ProgramGuard::getCurProgram(); \
    return *(program->makeTerm(BinaryOpType::optypename, std::monostate, {this, &b})); \
  }
  TERM_BINARY_OP(operator==, Eq)
#undef TERM_BINARY_OP
*/

} // namespace simplification

} // namespace nvfuser

