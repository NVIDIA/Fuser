// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <expr_simplifier.h>
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
  } else if (
      auto facop = dynamic_cast<assoc_comm::FlattenedAssocCommOp*>(expr)) {
    return facop->getOpType();
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
    Term* term = new Term{symbol, dtype, constant, producers};
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
      Term* term = makeTerm(std::monostate{}, dtype, c, {});
      // makeTerm will record an entry in term_map_ using the key corresponding
      // to the constant c. We additionally map the original key from the
      // unfolded expression here, so that we short-cut to the same value if we
      // recognize a constant subexpression later.
      term_map_.emplace(key, term);
      return term;
    }
    Term* term = new Term{symbol, dtype, constant, producers};
    terms_up_.emplace_back(std::unique_ptr<Term>(term));
    term_map_.emplace(key, term);
    return term;
  }
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
    switch (op_type) {
      case BinaryOpType::Eq: {
        NVF_ERROR(term.producers.size() == 2);
        proven_true_terms_.insert(&(b.equal(a)));
      }
      case BinaryOpType::LT:
      case BinaryOpType::LE:
      case BinaryOpType::GT:
      case BinaryOpType::GE: {
        NVF_ERROR(term.producers.size() == 2);
        const Term* a = term.producers[0];
        const Term* b = term.producers[1];
        bool inclusive =
            op_type == BinaryOpType::LE || op_type == BinaryOpType::GE;
        if (op_type == BinaryOpType::GT || op_type == BinaryOpType::GE) {
          std::swap(a, b);
        }
        // Test whether a and b are already involved in inequalities. If not,
        // then grow the matrices.
        auto pos_a_it = terms_in_orderings_.find(a);
        int num_inserted = 0;
        int pos_a, pos_b;
        if (pos_a_it == terms_in_orderings_.end()) {
          pos_a = terms_in_orderings_.size();
          terms_in_orderings_.emplace(a, pos_a);
          num_inserted++;
        } else {
          pos_a = pos_a_it->second;
        }
        auto pos_b_it = terms_in_orderings_.find(b);
        if (pos_b_it == terms_in_orderings_.end()) {
          pos_b = terms_in_orderings_.size();
          terms_in_orderings_.emplace(b, pos_b);
          num_inserted++;
        } else {
          pos_b = pos_b_it->second;
        }
        if (num_inserted > 0) {
          less_than_ = at::pad(less_than_, {0, num_inserted, 0, num_inserted});
          less_equal_ =
              at::pad(less_equal_, {0, num_inserted, 0, num_inserted});
          // Initialize diagonal
          less_equal_[pos_a][pos_a] = 1;
          less_equal_[pos_b][pos_b] = 1;
        }

        // Check the relevant entry for this comparison. If it needs to be
        // updated, then mark proofs as unsaturated.
        auto& matrix = inclusive ? less_equal_ : less_than_;
        proofs_saturated_ &= matrix[pos_a][pos_b].item<int>() > 0;
        matrix[pos_a][pos_b] = 1;
      }
      default:
        break;
    }
  }
}

bool Program::prove(const Term& term) {
  if (proven_true_terms_.count(&term)) {
    return true;
  }
  if (std::holds_alternative<BinaryOpType>(term.symbol)) {
    auto op_type = std::get<BinaryOpType>(term.symbol);

    auto proveComparison =
        [&](BinaryOpType op_type, const Term* a, const Term* b) -> bool {
      if (a == b) {
        // No need to consult the tables for trivial comparisons
        if (op_type == BinaryOpType::LE || op_type == BinaryOpType::GE) {
          return true;
        } else if (op_type == BinaryOpType::LT || op_type == BinaryOpType::GT) {
          return false;
        }
      }
      auto pos_a_it = terms_in_orderings_.find(a);
      if (pos_a_it == terms_in_orderings_.end()) {
        return false;
      }
      auto pos_b_it = terms_in_orderings_.find(b);
      if (pos_b_it == terms_in_orderings_.end()) {
        return false;
      }
      // Positions of a and b in adjacency matrices
      int pos_a = pos_a_it->second;
      int pos_b = pos_b_it->second;
      auto lt_acc = less_than_.accessor<int, 2>();
      // Check position without saturating so that we avoid saturating unless
      // necessary.
      if (lt_acc[pos_a][pos_b]) {
        return true;
      }
      if (!proofs_saturated_) {
        proveOrderings();
      }
      return lt_acc[pos_a][pos_b];
    };
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
      case BinaryOpType::LT: {
        NVF_ERROR(term.producers.size() == 2);
        const Term* a = term.producers[0];
        const Term* b = term.producers[1];
        return proveComparison(BinaryOpType::LT, a, b);
      }
      case BinaryOpType::GT: {
        NVF_ERROR(term.producers.size() == 2);
        const Term* a = term.producers[0];
        const Term* b = term.producers[1];
        return proveComparison(BinaryOpType::LT, b, a);
      }
      case BinaryOpType::LE: {
        NVF_ERROR(term.producers.size() == 2);
        const Term* a = term.producers[0];
        const Term* b = term.producers[1];
        return proveComparison(BinaryOpType::LE, a, b);
      }
      case BinaryOpType::GE: {
        NVF_ERROR(term.producers.size() == 2);
        const Term* a = term.producers[0];
        const Term* b = term.producers[1];
        return proveComparison(BinaryOpType::LE, b, a);
      }
      default:
        break;
    }
  }
  return false;
}

void Program::proveOrderings(int max_steps) {
  // The CPU matrices less_than_ and less_equal_ represent the < and <= ordering
  // relations. A given Term* is mapped to a row/column position in these
  // matrices by the map terms_in_orderings_. Given two Terms a and b with
  // integer positions na and nb, less_than_[na][nb] gives the number of ways
  // we have found to prove that a < b using the transitivity of < and <=, as
  // well as the implication "x < y implies x <= y".
  //
  // Matrix multiplication allows us to efficiently expand the transitivity
  // closure. Specifically, the (i, k) entry of less_than_ @ less_than_ sums
  // over all Terms j, the number of ways to prove i < j times the number of
  // ways to prove j < k, encoding the transitivity implication "x < y && y < z
  // implies x < z". Likewise we can compute less_than_ @ less_equal_ to
  // propagate "x < y && y <= z implies x < z" and we can multiply them in the
  // other order to prove a similar result. Note however that less_equal_ @
  // less_equal_ encodes the weaker implication "x <= y && y <= z implies x <=
  // z".
  //
  // We compute at each iteration:
  //   A = (less_than_ + less_equal_) @ (less_than_ + less_equal_)
  //   B = less_equal_ @  less_equal_
  // A - B then propagates the "x < z" implications above, so we increment
  // less_than_ by A. Meanwhile, B itself encodes the "x <= z" implication so we
  // increment less_equal_ by B. We then count the non-zero entries in both
  // less_than_ and less_equal_ to determine if any new proofs were generated;
  // if not, or if we exceed the iteration limit, then we stop.

  // Count nonzero entries
  int lt_nnz = less_than_.count_nonzero().item<int>();
  int le_nnz = less_equal_.count_nonzero().item<int>();

  for ([[maybe_unused]] int step : c10::irange(max_steps)) {
    at::Tensor ltplusle = less_than_ + less_equal_;
    at::Tensor A = at::matmul(ltplusle, ltplusle);
    at::Tensor B = at::matmul(less_equal_, less_equal_);

    less_than_ += A - B;
    less_equal_ += B;

    // x < y implies x <= y
    // TODO: This could potentially be combined with the incremental updates,
    // assuming we start in a canonical state. It's left this way for clarity
    // for now.
    less_equal_ += less_than_;

    less_than_.clamp_(c10::nullopt, 1);
    less_equal_.clamp_(c10::nullopt, 1);

    // TODO: Test consistency of less_than_ here, i.e. optionally test for
    // cycles

    int lt_nnz_n = less_than_.count_nonzero().item<int>();
    int le_nnz_n = less_equal_.count_nonzero().item<int>();
    if (lt_nnz_n == lt_nnz && le_nnz_n == le_nnz) {
      proofs_saturated_ = true;
      return;
    }
  }
}

Term* Program::valToTermHelper(Val* val) {
  // First check whether we've seen this Val before so we can return early
  auto val_it = val_term_map_.find(val);
  if (val_it != val_term_map_.end()) {
    return val_it->second;
  }
  if (auto* ns = dynamic_cast<NamedScalar*>(val)) {
    // We may encounter multiple NamedScalars with the same name. They should
    // all map to the same Term.
    auto ns_it = named_scalar_term_map_.find(ns->name());
    if (ns_it != named_scalar_term_map_.end()) {
      return ns_it->second;
    }
  }
  // Create a new Term
  Expr* def = val->definition();
  FunctionSymbol symbol = exprToFunctionSymbol(def);
  std::vector<const Term*> producer_terms;
  if (def != nullptr) {
    symbol = exprToFunctionSymbol(def);
    producer_terms.reserve(def->inputs().size());
    for (auto inp : def->inputs()) {
      producer_terms.push_back(valToTermHelper(inp));
    }
  }
  Term* term = makeTerm(
      symbol,
      dataTypeToPrim(val->dtype()),
      polymorphicValueToConstant(val->value()),
      producer_terms);
  term->representing_val = val;
  val_term_map_.emplace(val, term);
  if (auto* ns = dynamic_cast<NamedScalar*>(val)) {
    named_scalar_term_map_.emplace(ns->name(), term);
  }
  return term;
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
