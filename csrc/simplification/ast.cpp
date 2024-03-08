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

ConstantValue AbstractTerm::evaluate(
    FunctionSymbol symbol,
    PrimDataType dtype,
    const std::vector<ConstantValue>& producer_values) {
  return std::monostate{};
}

Term* Program::makeTerm(
    FunctionSymbol symbol,
    PrimDataType dtype,
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
    key.back() = (size_t)toUnderlying(dtype);
    if (constant.is<bool>()) {
      key.push_back((size_t)constant.as<bool>());
    } else if (constant.is<int64_t>()) {
      key.push_back(*reinterpret_cast<const size_t*>(&constant.as<int64_t>()));
    } else if (constant.is<double>()) {
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
    Term* term = new Term{symbol, dtype, constant, producers};
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
    // Try to fold constants
    std::vector<ConstantValue> producer_values;
    producer_values.reserve(producers.size());
    for (const Term* producer : producers) {
      producer_values.push_back(producer->constant);
    }
    ConstantValue c = AbstractTerm::evaluate(symbol, dtype, producer_values);
    if (c.hasValue()) {
      // Successful evaluation means constant folding is possible. Create a new
      // constant term with the evaluated value and the given dtype.
      std::cout << "Folded constant" << std::endl;
      Term* term = makeTerm(std::monostate{}, dtype, c, {});
      // makeTerm will record an entry in term_map_ using the key corresponding
      // to the constant c. We additionally map the original key from the
      // unfolded expression here, so that we short-cut to the same value if we
      // recognize a constant subexpression later.
      term_map_.emplace(key, term);
      return term;
    }
    Term* term = new Term{symbol, dtype, constant, producers};
    std::cout << "term = " << (void*)term << std::endl;
    terms_up_.emplace_back(std::unique_ptr<Term>(term));
    term_map_.emplace(key, term);
    return term;
  }
  std::cout << "Reusing existing Term " << (void*)(term_it->second)
            << " with key " << key << std::endl;
  return term_it->second;
}

// Convenience function for creating new terms using the current Program
static const Term& term(
    FunctionSymbol symbol,
    PrimDataType dtype,
    PolymorphicValue constant,
    const std::vector<const Term*>& producer_terms) {
  Program* program = ProgramGuard::getCurProgram();
  return *(program->makeTerm(symbol, dtype, constant, producer_terms));
}

const Term& Term::equal(const Term& other) const {
  return term(
      BinaryOpType::Eq, PrimDataType::Bool, std::monostate{}, {this, &other});
}
const Term& Term::notEqual(const Term& other) const {
  return term(
      BinaryOpType::NE, PrimDataType::Bool, std::monostate{}, {this, &other});
}

#define UNARY_TERM_OP(cppop, optype)                                   \
  const Term& cppop(const Term& a) {                                   \
    return term(UnaryOpType::optype, a.dtype, std::monostate{}, {&a}); \
  }
UNARY_TERM_OP(abs, Abs)
UNARY_TERM_OP(operator-, Neg)
UNARY_TERM_OP(operator!, LogicalNot)
UNARY_TERM_OP(operator~, BitwiseNot)
#undef UNARY_TERM_OP

// TODO: Use real promotion rules from type_promotion.h here
#define BINARY_TERM_OP_CONST_HELPER(cppop, optype, consttype)               \
  const Term& cppop(const Term& a, const consttype b_const) {               \
    const Term& b = term(std::monostate{}, a.dtype, b_const, {});           \
    return term(BinaryOpType::optype, a.dtype, std::monostate{}, {&a, &b}); \
  }                                                                         \
  const Term& cppop(const consttype a_const, const Term& b) {               \
    const Term& a = term(std::monostate{}, b.dtype, a_const, {});           \
    return term(BinaryOpType::optype, b.dtype, std::monostate{}, {&a, &b}); \
  }
#define BINARY_TERM_OP(cppop, optype)                                       \
  const Term& cppop(const Term& a, const Term& b) {                         \
    return term(BinaryOpType::optype, a.dtype, std::monostate{}, {&a, &b}); \
  }                                                                         \
  BINARY_TERM_OP_CONST_HELPER(cppop, optype, bool);                         \
  BINARY_TERM_OP_CONST_HELPER(cppop, optype, int64_t);                      \
  BINARY_TERM_OP_CONST_HELPER(cppop, optype, double);
BINARY_TERM_OP(operator+, Add)
BINARY_TERM_OP(operator-, Sub)
BINARY_TERM_OP(operator*, Mul)
BINARY_TERM_OP(operator/, Div)
BINARY_TERM_OP(operator%, Mod)
BINARY_TERM_OP(operator&&, LogicalAnd)
BINARY_TERM_OP(operator||, LogicalOr)
BINARY_TERM_OP(operator&, BitwiseAnd)
BINARY_TERM_OP(operator|, BitwiseOr)
BINARY_TERM_OP(operator^, BitwiseXor)
BINARY_TERM_OP(operator<<, Lshift)
BINARY_TERM_OP(operator>>, Rshift)
BINARY_TERM_OP(ceilDiv, CeilDiv)
BINARY_TERM_OP(gcd, Gcd)
#undef BINARY_TERM_OP
#undef BINARY_TERM_OP_CONST_HELPER

#define BINARY_LOGIC_TERM_OP_CONST_HELPER(cppop, optype, consttype)        \
  const Term& cppop(const Term& a, const consttype b_const) {              \
    const Term& b = term(std::monostate{}, a.dtype, b_const, {});          \
    return term(                                                           \
        BinaryOpType::optype, DataType::Bool, std::monostate{}, {&a, &b}); \
  }                                                                        \
  const Term& cppop(const consttype a_const, const Term& b) {              \
    const Term& a = term(std::monostate{}, b.dtype, a_const, {});          \
    return term(                                                           \
        BinaryOpType::optype, DataType::Bool, std::monostate{}, {&a, &b}); \
  }
#define BINARY_LOGIC_TERM_OP(cppop, optype)                                \
  const Term& cppop(const Term& a, const Term& b) {                        \
    return term(                                                           \
        BinaryOpType::optype, DataType::Bool, std::monostate{}, {&a, &b}); \
  }                                                                        \
  BINARY_LOGIC_TERM_OP_CONST_HELPER(cppop, optype, bool);                  \
  BINARY_LOGIC_TERM_OP_CONST_HELPER(cppop, optype, int64_t);               \
  BINARY_LOGIC_TERM_OP_CONST_HELPER(cppop, optype, double);
BINARY_LOGIC_TERM_OP(operator<, LT)
BINARY_LOGIC_TERM_OP(operator<=, LE)
BINARY_LOGIC_TERM_OP(operator>, GT)
BINARY_LOGIC_TERM_OP(operator>=, GE)
BINARY_LOGIC_TERM_OP(operator==, Eq)
BINARY_LOGIC_TERM_OP(operator!=, NE)
#undef BINARY_LOGIC_TERM_OP
#undef BINARY_LOGIC_TERM_OP_CONST_HELPER

#define TERNARY_TERM_OP(cppop, optype)                                   \
  const Term& cppop(const Term& a, const Term& b, const Term& c) {       \
    return term(                                                         \
        TernaryOpType::optype, b.dtype, std::monostate{}, {&a, &b, &c}); \
  }
TERNARY_TERM_OP(where, Where)
#undef TERNARY_TERM_OP

void Program::assume(const Term& term) {
  NVF_ERROR(
      term.dtype == PrimDataType::Bool,
      "Program::assume accepts Bool-valued Terms only")
  proven_true_terms_.insert(&term).second;
  if (isBinaryOpType(term.symbol)) {
    BinaryOpType op_type = std::get<BinaryOpType>(term.symbol);
    const Term& a = *term.producers[0];
    const Term& b = *term.producers[1];
    if (op_type == BinaryOpType::Eq) {
      NVF_ERROR(term.producers.size() == 2);
      proven_true_terms_.insert(&(b.equal(a)));
      /*
      proven_true_terms_.insert(*term.producers[0] <= *term.producers[1]);
      proven_true_terms_.insert(*term.producers[0] >= *term.producers[1]);
      proven_true_terms_.insert(*term.producers[1] <= *term.producers[0]);
      proven_true_terms_.insert(*term.producers[1] >= *term.producers[0]);
    } else if (term.symbol == BinaryOpType::LT) {
      proven_true_terms_.insert(*term.producers[0] <= *term.producers[1]);
      proven_true_terms_.insert(*term.producers[1] > *term.producers[0]);
      proven_true_terms_.insert(*term.producers[1] >= *term.producers[0]);
    } else if (term.symbol == BinaryOpType::LE) {
      proven_true_terms_.insert(*term.producers[1] >= *term.producers[0]);
      */
    }
  }
}

bool Program::isProven(const Term& term) const {
  if (proven_true_terms_.count(&term)) {
    return true;
  }
  if (std::holds_alternative<BinaryOpType>(term.symbol)) {
    auto op_type = std::get<BinaryOpType>(term.symbol);
    switch (op_type) {
      case BinaryOpType::Eq: {
        NVF_ERROR(term.producers.size() >= 2);
        bool allmatch = true;
        const Term* ptr = term.producers[0];
        for (auto producer : term.producers) {
          if (producer != ptr) {
            allmatch = false;
            break;
          }
        }
        if (allmatch) {
          // All arguments are structurally identical
          return true;
        }
        break;
      }
      default:
        break;
    }
  }
  return false;
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
    NVF_ERROR(
        !(term->constant.hasValue()), "Refusing to create free variable Val");
    PolymorphicValue pv;
    if (term->constant.is<bool>()) {
      pv = term->constant.as<bool>();
    } else if (term->constant.is<int64_t>()) {
      pv = term->constant.as<int64_t>();
    } else if (term->constant.is<double>()) {
      pv = term->constant.as<double>();
    } else {
      NVF_ERROR(false, "Unrecognized constant value: ", term->constant);
    }
    val = IrBuilder::create<Val>(pv, term->dtype);
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
  auto* mut_term = const_cast<Term*>(term);
  mut_term->representing_val = val;
  val_term_map_[val] = mut_term;
  return val;
}

} // namespace simplification

} // namespace nvfuser
