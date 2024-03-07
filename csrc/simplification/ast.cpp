// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/builder.h>
#include <ops/arith.h>
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

Term* Program::makeTerm(
    FunctionSymbol symbol,
    const ConstantValue& constant,
    const std::vector<const Term*>& producers) {
  // If this is a free variable, we will not re-use any existing Term
  bool is_free_var = true;
  TermMapKey key;
  // Look up term or create a new one
  key.push_back(symbolId(symbol));
  is_free_var &= std::holds_alternative<std::monostate>(symbol);

  if (constant.hasValue()) {
    // When there is a constant value, the symbol should be std::monostate, so
    // the key should be [0] at this point. The next largest key is [1 << 16],
    // so there is room for us to fill the first int with the datatype then push
    // another int holding the actual value.
    NVF_ERROR(
        key.size() == 1 && key.back() == 0,
        "Constant terms should have key [0]. ",
        "This could indicate a non-constant Val with non-null definition.");
    if (constant.is<bool>()) {
      key.back() = 1;
      key.push_back((size_t)constant.as<bool>());
    } else if (constant.is<int64_t>()) {
      key.back() = 2;
      key.push_back(*reinterpret_cast<const size_t*>(&constant.as<int64_t>()));
    } else if (constant.is<double>()) {
      key.back() = 3;
      key.push_back(*reinterpret_cast<const size_t*>(&constant.as<double>()));
    } else {
      NVF_ERROR(
          false,
          "Could not create Term key: unrecognized type for constant ",
          constant);
    }
    is_free_var = false;
  }

  if (is_free_var) {
    std::cout << "Creating unique Term for free variable" << std::endl;
    Term* term = new Term{symbol, constant, producers};
    std::cout << "term = " << (void*)term << std::endl;
    terms_up_.emplace_back(std::unique_ptr<Term>(term));
    return term;
  }

  for (auto* producer : producers) {
    // We cast the pointers directly to size_t since these pointer equality
    // is equivalent to structural equality if all terms are deduplicated.
    key.push_back((size_t)producer);
  }

  auto term_it = term_map_.find(key);
  if (term_it == term_map_.end()) {
    std::cout << "Creating term with key " << key << std::endl;
    Term* term = new Term{symbol, constant, producers};
    std::cout << "term = " << (void*)term << std::endl;
    terms_up_.emplace_back(std::unique_ptr<Term>(term));
    term_map_.emplace(key, term);
    return term;
  }
  std::cout << "Reusing existing Term " << (void*)(term_it->second)
            << " with key " << key << std::endl;
  return term_it->second;
}

Term::operator bool() const {
  if (constant.hasValue() && constant.is<bool>()) {
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

Val* Program::termToVal(const Term* term) {
  // TODO: Non-recursive version
  if (term->representing_val != nullptr) {
    return term->representing_val;
  }
  // We need to construct a Val to represent this Term
  Val* val = nullptr;
  if (std::holds_alternative<std::monostate>(term->symbol)) {
    // Constant or free variable
    // TODO: this is where we might want to track PrimDataType. We currently
    // just need to guess at the types we create for constants. That is
    // likely not a problem as we can just assume Bool, Double, or Int.
    NVF_ERROR(
        !(term->constant.hasValue()), "Refusing to create free variable Val");
    if (term->constant.is<bool>()) {
      val = IrBuilder::create<Val>(term->constant.as<bool>(), DataType::Bool);
    } else if (term->constant.is<int64_t>()) {
      val = IrBuilder::create<Val>(term->constant.as<int64_t>(), DataType::Int);
    } else if (term->constant.is<double>()) {
      val =
          IrBuilder::create<Val>(term->constant.as<double>(), DataType::Double);
    }
  } else if (isUnaryOpType(term->symbol)) {
    NVF_ERROR(term->producers.size() == 1);
    val = unaryOp(
        std::get<UnaryOpType>(term->symbol), termToVal(term->producers[0]));
  } else if (isBinaryOpType(term->symbol)) {
    NVF_ERROR(term->producers.size() == 2);
    val = binaryOp(
        std::get<BinaryOpType>(term->symbol),
        termToVal(term->producers[0]),
        termToVal(term->producers[1]));
  } else if (isTernaryOpType(term->symbol)) {
    NVF_ERROR(term->producers.size() == 3);
    // There is no "ternaryOp" helper, so we special case on the op instead
    auto op_type = std::get<TernaryOpType>(term->symbol);
    switch (op_type) {
      case TernaryOpType::Where:
        val = where(
            termToVal(term->producers[0]),
            termToVal(term->producers[1]),
            termToVal(term->producers[2]));
        break;
      default:
        NVF_ERROR(false, "Unhandled ternary op type ", op_type);
    }
  }
  return val;
}

} // namespace simplification

} // namespace nvfuser
