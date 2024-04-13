// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <visibility.h>

#include <vector>

// Note: [The Mathematics of Integer Arithmetic]
//
// We learnt arithmetic from as early as elementary school, and have been used
// to simplify expressions using rules like (a+b)/c = a/c+b/c. However, when we
// are dealing with integer arithmetic, which is the case for index and
// predicate simplification, lots of rules we learnt in elementary school no
// longer hold. For example, (1+1)/2 != 1/2+1/2 because the left hand side is 1
// and the right hand side is 0 + 0 = 0. So when considering adding a new
// simplification rule, we need to be very careful to make sure the rule is
// mathematically correct.
//
// Suggested reading matherials:
// - doc/math/abstract-algebra.md reviews abstract algebra, a theory that tells
//   us which rule we are used to is still valid, and which is not.
// - doc/math/integer-division.md reviews the definitions and properties of div
//   and mod in textbooks, it also describes some theorems that we proved
//   ourselves that is useful for simplifying integer expressions.
//
// We can use the following rules to simplify integer expressions:
//
// A) Associativity of +: a + (b + c) = (a + b) + c
// B) Associativity of *: a * (b * c) = (a * b) * c
// C) Commutativity of +: a + b = b + a
// D) Commutativity of *: a * b = b * a
// E) Distributivity of * over +: a * (b + c) = (a * b) + (a * c)
// F) Distributivity of * over +: (a + b) * c = (a * c) + (b * c)
// G) (-a) / b = -(a / b) = a / (-b)
// H) (-a) % b = -(a % b) = a % (-b)
// I) If -|a| < r < |a|, then r % a = r, r / a = 0
// J) Distributivity of % over +:
//    If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// J.1) If compatible_sign(a, b) and a % c = 0, then (a + b) % c = b % c
// J.2) Let g = gcd(a, c). If compatible_sign(a, b), and -|g| < b < |g|
//      then (a + b) % c = a % c + b
// K) Distributivity of % over *:
//    If compatible_sign(a, b), then (a * b) % c = (a % c * b % c) % c
// L) If a is a multiple of b, then a % b = 0
// M) If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
// N) a / (b * c) = (a / b) / c
// O) If d divides a and b, then a % b = ((a / d) % (b / d)) * d
// P) If b is a multiple of c, then a/(b/c) = (a*c)/b
// Q) If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
//    (a+b)/c = a/c + b/c
// Q.1) If compatible_sign(a, b) and a % c = 0, then
//    (a+b)/c = a/c + b/c
// Q.2) Let g = gcd(a, c). If compatible_sign(a, b), and -|g| < b < |g|
//      then (a + b) / c = a/c
//
// See doc/math/integer-division.md for proofs of these rules.
//
// Some examples on applying the above rules to simplify expressions:
//
// Example 7.1: Given that a >= 0 and b >= 0, simplify (a*4 + b) % 4
// Answer: (a*4 + b) % 4 = ((a*4)%4 + b%4) % 4 (Rule J)
// = (0 + b%4) % 4 (Rule L)
// = b % 4 % 4 (Basic math)
// = b % 4 (Rule I)
//
// Example 7.2: Given that 0 <= a < 3, simplify a % 4
// Answer: a % 4 = a (Rule I)
//
// Example 7.3: Simplify (a * 256) / 4
// Answer: (a * 256) / 4 = a * (256 / 4) (Rule M)
// = a * 64 (Basic math)
//
// Example 7.4: Simplify (a / 4) / 64
// Answer: (a / 4) / 64 = a / (4 * 64) (Rule N)
// = a / 256 (Basic math)
//
// Example 7.5: Simplify (a * 64) % 256 / 4
// Answer: (a * 64) % 256 / 4 = ((a % 4) * 64) / 4 (Rule O)
// = (a % 4) * (64 / 4) (Rule M)
// = (a % 4) * 16 (Basic math)
//
// Example 7.6: Simplify (a * 4) / 256
// Answer: (a * 4) / 256 = a / (256 / 4) (Rule P)
// = a / 64 (Basic math)
//
// Example 7.7: Given that a >= 0 and b >= 0, simplify (a * 256 + b) / 4
// Answer: because (a * 256) % 4 = 0, we have
// (a * 256 + b) / 4 = a * 256 / 4 + b / 4 (Rule Q)
// = a * (256 / 4) + b / 4 (Rule M)
// = a * 64 + b / 4 (Basic math)
//
// Example 7.8: Given that a >= 0 and 0 <= b < 4, simplify (a * 4 + b) / 4
// Answer: Similar to above, we have
// (a * 4 + b) / 4 = a + b / 4
// = a + 0 (Rule I)
// = a

// Note: [Simplification of boolean predicates]
//
// TODO: move this note to a markdown file.
//
// This note lists some rules we use to simplify boolean predicates, and provide
// the proofs of these rules.
//
// The following definitions and lemmas are helpful:
// (Reference: https://en.wikipedia.org/wiki/Monotonic_function)
//
// Definition 1: A function is called strictly increasing if for all x < y,
//   we have f(x) < f(y)
// Definition 2: A function is called weakly increasing if for all x <= y,
//   we have f(x) <= f(y)
//
// Lemma 1: if f is strictly increasing, then
// 1. x < y <=> f(x) < f(y)
// 2. x <= y <=> f(x) <= f(y)
// Proof:
// 1. x < y => f(x) < f(y) by definition.
// 2. x <= y => f(x) <= f(y)
//    x <= y means either x < y or x = y. For the first case, f(x) < f(y).
//    For the second case, f(x) = f(y), in combination, f(x) <= f(y).
// 1. f(x) < f(y) => x < y
//    Assume x < y was not true, then y <= x, then f(y) <= f(x), which conflict
//    with f(x) < f(y)
// 2. f(x) <= f(y) => x <= y
//    Assume x <= y was not true, then y < x, then f(y) < f(x), which conflict
//    with f(x) <= f(y)
//
// Lemma 2: if f is weakly increasing, then
// 1. x <= y => f(x) <= f(y)
// 2. x < y => f(x) <= f(y)
// 3. f(x) <= f(y) does not provide much information, especially, it is not
//    guaranteed that x <= y
// 4. f(x) < f(y) => x < y
// Proof:
// 1. x <= y => f(x) <= f(y) by definition
// 2. x < y => f(x) <= f(y) because x < y => x <= y
// 3. Consider f(x) = 1 (constant function), then f is weakly increasing. For
//    this function, f(x) <= f(y) is trivially true, and x and y can be
//    arbitrary number.
// 4. f(x) < f(y) => x < y
//    Assume that x < y was not true, that is, y <= x, then f(y) <= f(x), which
//    conflict with f(x) < f(y)
//
// With these lemmas, we can study possible simplification rules for boolean
// predicates. One interesting application is on division.
//
// Different type of divisions are studied in the following paper:
//   Boute, Raymond T. "The Euclidean definition of the functions div and mod."
//   ACM Transactions on Programming Languages and Systems (TOPLAS) 14.2 (1992):
//   127-144.
// Referring to that paper, we can conclude that:
// - For all types of division (Euclidean/trunc/floor div) f(x) = x / d, f is
//   weakly increasing if d > 0, and weakly decreasing if d < 0.
//
// So, we have the following rule for trunc division:
// Rule 1: For i >= 0, d > 0, i / d < D <=> i < D * d
// Proof:
// 1. i / d < D => i < D * d
//    Consider the function f(x) = x / d, it is weakly increasing. Also note
//    that D = D * d / d.
//    According to Lemma 2 (4), we have f(i) < f(D*d) => i < D*d.
// 2. i < D * d => i / d < D
//    According to the fundamental division-with-remainder equation, i < D * d
//    can be written as i / d * d + i % d < D * d. Because i >= 0, d > 0, we
//    have i % d >= 0. So i / d * d <= i / d * d + i % d < D * d. That is,
//    i / d * d < D * d. Consider the function g(x) = x * d, which is strictly
//    increasing. According to Lemma 1 (1), g(i / d) < g(D) indicates i/d < D.

namespace nvfuser {

// Information for a single variable. Possible values that this variable can
// take is: start, start + step, start + 2 * step, ... (< stop), which is
// similar to the loop variable of for loop:
//   for variable in range(start, stop, step)
struct VarInfo {
  Val* variable = nullptr;
  // If this variable is an unrolled loop index. It is important to know this
  // because unrolled loop index is compile constant to nvRTC. Note that a
  // constant to nvRTC might not be a constant to nvFuser. For example, if I
  // have loop
  //   #pragma unroll
  //   FOR i1 in ...:
  //     ...
  // Then `i1` is a compile constant to nvRTC, but not a compile time constant
  // to nvFuser.
  bool is_unrolled_loop_index = false;
};

// Analyze expression register usage
enum class RegisterType { GeneralPurpose, Uniform, Immediate, Unknown };
RegisterType getRegisterType(Val* value);

// Simplify expressions with the given information of variables.
//
// The argument `variables` specifies which scalar are considered variable and
// some information about these variables. Any scalar not contained in
// `variables` are considered constants. Tensors are always considered as
// variables, regardless of if it is specified in `variables`.
//
// Note that in `variables`, the order matters. This order specifies how we
// should organize associative and commutative expressions. For example, if the
// `variables` is {a, b, c, d}, then we will simplify (a + d) + (c + b) as
// ((a + b) + c) + d. Tensors are always considered as at the right of all
// scalars, regardless of if it is inside `variables` or not.
// See note [Reordering associative and commutative operators] for detailed
// information about this reordering.
//
// Some simplifications like a*b/b -> a is always correct in valid case, but
// when there is an error (e.g. division-by-zero), these simplifications could
// potentially hide the error. The argument `preserve_error` specifies whether
// we should disable these optimization, unless we can prove there won't be an
// error.
NVF_API Val* simplifyExpr(
    Val* value,
    const std::list<VarInfo>& variables = {},
    std::vector<Val*> assumptions = {},
    bool preserve_error = false);

class Context;
namespace assoc_comm {
// The expression type that represents the flattened ops. For example, if I have
// out = a + b + 3 + c + 5, then I will have:
//   FlattenedAssocCommOp {
//     inputs: [a, b, 3, c, 5]
//     outputs: [out]
//   }
class FlattenedAssocCommOp : public Expr {
 public:
  using Expr::Expr;

  FlattenedAssocCommOp(
      IrBuilderPasskey passkey,
      BinaryOpType op,
      Val* out,
      std::vector<Val*> terms);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override;

  // FlattenedAssocCommOp is unordered, so we should have
  // FlattenedAdd(a, b)->sameAs(FlattenedAdd(b, a))
  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

  DataType dtype() const {
    return *output(0)->getDataType();
  }

  BinaryOpType getOpType() const {
    return attribute<BinaryOpType>(0);
  }

  // Get a vector of inputs, sorted as the order given by `variables`. Note that
  // the sorting key is the rightmost variable that an input depends on. For
  // example, if I have two inputs.
  // v1 = a * c
  // v2 = b
  // and variables is [a, b, c], then v2 < v1 because the rightmost depending
  // variable of v2 is b, and the rightmost depending variable of v1 is c,
  // and b < c. So in this example, this function will return [v2, v1].
  // Tensors are always considered as variables and they are always considered
  // as the rightmost.
  std::vector<Val*> sortedInputs(const Context& context);

  bool isTrivial() const {
    return inputs().size() == 1;
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override;
};

} // namespace assoc_comm

} // namespace nvfuser
