// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <disjoint_set.h>
#include <exceptions.h>
#include <ir/base_nodes.h>
#include <ir/internal_nodes.h>
#include <polymorphic_value.h>
#include <type.h>
#include <utils.h>

#include <iostream>
#include <variant>
#include <vector>

namespace nvfuser {

namespace simplification {

class Program;

class ProgramGuard {
 public:
  //! Set the active fusion so it can be manipulated.
  NVF_API explicit ProgramGuard(Program* program);

  NVF_API ~ProgramGuard();

  NVF_API static Program* getCurProgram();
  static void setCurProgram(Program* program);

 private:
  Program* prev_program_;

  static thread_local Program* active_program_;
};

using FunctionSymbol =
    std::variant<std::monostate, UnaryOpType, BinaryOpType, TernaryOpType>;

std::string toString(const FunctionSymbol& symbol);
std::ostream& operator<<(std::ostream& os, const FunctionSymbol& symbol);

//! This returns std::nullopt for unsupported Exprs. Note that this is different
//! from std::monostate, which indicates a constant or free variable.
FunctionSymbol exprToFunctionSymbol(Expr* expr);

inline bool isUnaryOpType(const FunctionSymbol symb) {
  return std::holds_alternative<UnaryOpType>(symb);
}
inline bool isBinaryOpType(const FunctionSymbol symb) {
  return std::holds_alternative<BinaryOpType>(symb);
}
inline bool isTernaryOpType(const FunctionSymbol symb) {
  return std::holds_alternative<TernaryOpType>(symb);
}

//! Convert the function symbol to a unique 2-byte integer. Collisions should
//! be avoided at all cost here.
size_t symbolId(FunctionSymbol symbol);

//! Constant values must be bool, long int, or double
using ConstantValue = dynamic_type::
    DynamicType<dynamic_type::NoContainers, double, int64_t, bool>;

inline ConstantValue polymorphicValueToConstant(const PolymorphicValue& pv) {
  if (!pv.hasValue()) {
    return std::monostate{};
  }
  if (pv.is<bool>()) {
    return pv.as<bool>();
  } else if (pv.is<int64_t>()) {
    return pv.as<int64_t>();
  } else if (pv.is<double>()) {
    return pv.as<double>();
  }
  NVF_ERROR(
      false,
      "Cannot convert ",
      pv,
      " from PolymorphicValue to simplification::ConstantValue");
  return std::monostate{};
}

//! We use PrimDataType to represent scalar types. DataType is a much larger
//! type that also includes structs, arrays, pointers, and opaque types. The
//! difference in size is dramatic. On gcc 11.4.0, sizeof(PrimDataType) == 4,
//! while sizeof(DataType) == 96. This function strips a DataType object to
//! obtain the underlying PrimDataType.
// TODO: This belongs in type.h
inline PrimDataType dataTypeToPrim(const DataType& dtype) {
  NVF_ERROR(
      std::holds_alternative<PrimDataType>(dtype.type),
      "Expected DataType holding PrimDataType but found ",
      dtype);
  return std::get<PrimDataType>(dtype.type);
}

//! An AbstractTerm models a generic function symbol without describing its
//! producers. This allows us to generalize functions across producer types,
//! depending on whether we are modelling terms or implementing an e-graph.
//!
//! Note that root variables are AbstractTerms whose symbol is std::monostate.
struct AbstractTerm {
  FunctionSymbol symbol;

  // [DataTypes in the AST]
  // Notice that we do not represent datatypes here. We could place a
  // PrimDataType attribute in this class, but then we would need to shadow the
  // type promotion rules here. This is not impossible, but it is complexity
  // that might not be needed, so we will avoid doing so until it is needed.
  PrimDataType dtype;

 public:
  static ConstantValue evaluate(
      FunctionSymbol symbol,
      PrimDataType dtype,
      const std::vector<ConstantValue>& producer_values);

  ConstantValue evaluate(
      const std::vector<ConstantValue>& producer_values) const {
    return evaluate(symbol, dtype, producer_values);
  }
};

//! [AST model]
//! A Term is either a constant, a free variable, or an AbstractTerm of other
//! Terms.
struct Term : AbstractTerm {
  //! This is std::monostate for free variables and whenever symbol !=
  //! std::monostate. Otherwise it indicates a constant value.
  ConstantValue constant;

  std::vector<const Term*> producers;

  //! At most a single Val* can represent a Term. This will typically be the
  //! latest "seen" Val of this form. In this way, we can re-use proofs even
  //! when the visibility of terms changes.
  Val* representing_val = nullptr;

 public:
  // This returns a new Term representing the condition a == b. Note that this
  // does not directly return bool. However, since terms are implicitly
  // convertible to bool, using if(a.equals(b)) will return true if it is proven
  // that a == b.
  const Term& equal(const Term& other) const;
  const Term& notEqual(const Term& other) const;

  std::string toInlineString() const;
  std::string toString() const {
    return toInlineString();
  }
};

const Term& abs(const Term& a);
const Term& operator-(const Term& a);
const Term& operator!(const Term& a);
const Term& operator~(const Term& a);

// TODO: restrict these
#define DECLARE_BINARY_TERM_OP_CONST_HELPER(cppop, consttype) \
  const Term& cppop(const Term& a, const consttype b_const);  \
  const Term& cppop(const consttype a_const, const Term& b);
#define DECLARE_BINARY_TERM_OP(cppop)                  \
  const Term& cppop(const Term& a, const Term& b);     \
  DECLARE_BINARY_TERM_OP_CONST_HELPER(cppop, bool);    \
  DECLARE_BINARY_TERM_OP_CONST_HELPER(cppop, int32_t); \
  DECLARE_BINARY_TERM_OP_CONST_HELPER(cppop, int64_t); \
  DECLARE_BINARY_TERM_OP_CONST_HELPER(cppop, double);
DECLARE_BINARY_TERM_OP(operator+)
DECLARE_BINARY_TERM_OP(operator-)
DECLARE_BINARY_TERM_OP(operator*)
DECLARE_BINARY_TERM_OP(operator/)
DECLARE_BINARY_TERM_OP(operator%)
DECLARE_BINARY_TERM_OP(operator&&)
DECLARE_BINARY_TERM_OP(operator||)
DECLARE_BINARY_TERM_OP(operator&)
DECLARE_BINARY_TERM_OP(operator|)
DECLARE_BINARY_TERM_OP(operator^)
DECLARE_BINARY_TERM_OP(operator<<)
DECLARE_BINARY_TERM_OP(operator>>)
DECLARE_BINARY_TERM_OP(operator<)
DECLARE_BINARY_TERM_OP(operator<=)
DECLARE_BINARY_TERM_OP(operator>)
DECLARE_BINARY_TERM_OP(operator>=)
DECLARE_BINARY_TERM_OP(operator==)
DECLARE_BINARY_TERM_OP(operator!=)
DECLARE_BINARY_TERM_OP(ceilDiv)
DECLARE_BINARY_TERM_OP(gcd)
#undef DECLARE_BINARY_TERM_OP
#undef DECLARE_BINARY_TERM_OP_CONST_HELPER

const Term& where(const Term& a, const Term& b, const Term& c);

// [Uniqueness of Terms]
// In the Fusion IR, we can have multiple Vals representing the exact same
// computation. For example,
//   x = IrBuilder::create<Val>(3);
//   y = IrBuilder::create<Val>(3);
// This results in two separate Val*s (x != y) both representing a constant
// value of 3. This is fine for Vals, but for simplification it is useful to
// deduplicate Terms so that we don't need to reprove and resimplify
// expressions.
class Program {
 public:
  Program()
      : less_than_(at::zeros({0, 0}, at::kInt)),
        less_equal_(at::ones({0, 0}, at::kInt)) {}

  //! This is non-const since we register newly-created Vals in order to quickly
  //! recognize them if they appear as producers in later seen Vals.
  Val* termToVal(const Term* term);
  Val* termToVal(const Term& term) {
    return termToVal(&term);
  }

  //! Map a Val into Program and return a const reference to the corresponding
  //! Term. If we have previously seen this Val or any of its producers, we will
  //! re-use their Terms.
  const Term& valToTerm(Val* val) {
    return *valToTermHelper(val);
  }

  //! Terms are owned by Program. When we need to make a new Term, we first
  //! check whether a Term with that form already exists. If so we return it and
  //! if not we create a new one.
  //!
  //! If force_creation=true, then we will not try and re-use existing terms.
  //! This is necessary for inserting distinct Terms related to multiple free
  //! variables. Each free variable has no symbol, no constant, and no producers
  //! so they will all have the same key. If we did not force creation of a new
  //! Term, then all free variables would be represented by the same Term.
  Term* makeTerm(
      FunctionSymbol symbol,
      PrimDataType dtype,
      const ConstantValue& constant,
      const std::vector<const Term*>& producers);

  //! Assume a Bool-valued Term is true
  void assume(const Term& term);

  //! Test whether a Bool-valued term can be proven true
  bool prove(const Term& term);

 private:
  //! Find a given Val and return its Term*. If we haven't yet seen this Val*,
  //! return nullptr.
  Term* findTerm(Val* val) {
    auto it = val_term_map_.find(val);
    if (it == val_term_map_.end()) {
      return nullptr;
    }
    return it->second;
  };

  //! This is like findTerm, but if we haven't seen Val, then recursively make
  //! terms to represent val's producers and its definition.
  Term* valToTermHelper(Val* val);

  //! Try to prove new orderings implied by current assumptions and proofs. This
  //! computes the transitive closure of the ordering assumptions by matrix
  //! multiplication of the adjacency graphs less_than_ and less_equal_.
  void proveOrderings(int max_steps = 10);

 private:
  using TermMapKey = std::vector<size_t>;
  class TermMapKeyHasher {
   public:
    size_t operator()(const TermMapKey& key) const {
      size_t h = 0;
      for (auto ki : key) {
        nvfuser::hashCombine(h, ki);
      }
      return h;
    }
  };

  // Owns all Terms in this Program
  std::vector<std::unique_ptr<Term>> terms_up_;

  // Maps keys to Term*. This is used as a check that no structurally identical
  // Term already exists in terms_up_ before adding it. This guarantees that
  // for any two terms in terms_up_, pointer equality implies structural
  // equality.
  std::unordered_map<TermMapKey, Term*, TermMapKeyHasher> term_map_;

  // Map Vals to Terms. This allows us to quickly retrieve a Term, without
  // needing to recursively inspect the structure of all the producers of that
  // Val. This assumes that the structure of Vals does not change. If a Val* is
  // redefined after it is seen, then the Term pointed to by this map will still
  // refer to the previous definition.
  std::unordered_map<Val*, Term*> val_term_map_;

  // Named scalars are treated as ordinary Terms. We map their name to the Term
  // that represents them to avoid reproving relations when possible.
  std::unordered_map<std::string, Term*> named_scalar_term_map_;

  // A collection of Terms that have been proven true
  // This lets us easily test whether a term is true or not. For example,
  //
  //   const Term& a = program.valToTerm(val_a);
  //   const Term& b = program.valToTerm(val_b);
  //   if (a < b) {
  //     foo();
  //   }
  //
  // Here we used Term::operator<() to find or create the new term a < b, then
  // we used the implicit conversion operator that casts Terms to true only if
  // they appear in this set.
  std::unordered_set<const Term*> proven_true_terms_;

  // This holds a mapping from each ConstantValue to a vector of Terms with that
  // value. Note that there might be multiple Terms with a single constant since
  // those terms can have distinct dtypes. When we detect that an element was
  // inserted, we iterate through this collection, adding relations to
  // less_than_ and less_equal_, then we rerun proveOrderings(). In this way we
  // automatically prove implications like  a < 3 && 5 <= b => a < b.
  // Also note that double is able to represent ints and int arithmetic exactly
  // on the interval [-2^52, 2^52].
  std::unordered_map<double, std::vector<const Term*>> real_constants_;

  // These aten tensors contain int-valued matrices that encode proven ordering
  // relations. These get completed via proveOrderings(). Whenever a
  // new relation is proven involving two terms, that method is called.
  //
  // For example, suppose we have made the following assumptions:
  //
  //    c < d
  //    b <= a
  //    c < b
  //    f < e
  //    e <= c
  //
  // Then originally we have the following:
  //
  //   terms_in_orderings_ = {a:0, b:1, c:2, d:3, e:4, f:5}
  //
  //   less_than_ =        less_equal_ =
  //     0 0 0 0 0           0 0 0 0 0
  //     0 0 0 0 0           1 0 0 0 0
  //     0 1 0 1 0           0 0 0 0 0
  //     0 0 0 0 0           0 0 0 0 0
  //     0 0 0 0 0           0 0 1 0 0
  //     0 0 0 1 0           0 0 0 0 0
  //
  // proveOrderings() will develop this like so
  //
  //   less_than_ =        less_equal_ =
  //     0 0 0 0 0           0 0 0 0 0
  //     0 0 0 0 0           1 0 0 0 0
  //     1 1 0 1 0           1 1 0 1 0
  //     0 0 0 0 0           0 0 0 0 0
  //     1 1 0 0 0           1 1 1 0 0
  //     1 1 1 1 0           1 1 1 1 0
  //
  // NOTE: constant terms are ordered so that we can manually prove relations
  // between constants whenever they appear in queries.
  std::unordered_map<const Term*, int> terms_in_orderings_;
  at::Tensor less_than_;
  at::Tensor less_equal_;
  // This is set to true when we insert a new relation or grow the size of
  // terms_in_orderings_, less_than_,  and less_equal_
  bool proofs_saturated_ = false;
};

} // namespace simplification

} // namespace nvfuser
