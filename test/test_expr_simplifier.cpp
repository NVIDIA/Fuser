// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <assume.h>
#include <expr_simplifier.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

#include <cctype>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace nvfuser {
namespace {

// check if x and y are equivalent expressions by checking that x == y
// simplifies to true. We don't use x->sameAs(y), because we want to consider
// a(b+c), ab+ac, (b+c)a, ba+ca as equivalent, but `sameAs` can not do this job.
bool isEquivalent(Val* x, Val* y) {
  return simplifyExpr(eq(x, y))->getBool() == true;
}

// assert that x/y -> z
void assertSimplifiedDiv(
    Val* x,
    Val* y,
    Val* z,
    std::vector<Bool*> assumptions = {}) {
  auto simplified = simplifyExpr(div(x, y), {}, assumptions);
  TORCH_CHECK(
      isEquivalent(simplified, z),
      "Expect ",
      x->toInlineString(),
      " / ",
      y->toInlineString(),
      " to be simplified to ",
      z->toInlineString(),
      ", but get ",
      simplified->toInlineString());
}

// assert that x % y -> z
void assertSimplifiedMod(
    Val* x,
    Val* y,
    Val* z,
    std::vector<Bool*> assumptions = {}) {
  auto simplified = simplifyExpr(mod(x, y), {}, assumptions);
  TORCH_CHECK(
      isEquivalent(simplified, z),
      "Expect ",
      x->toInlineString(),
      " % ",
      y->toInlineString(),
      " to be simplified to ",
      z->toInlineString(),
      ", but get ",
      simplified->toInlineString());
}

} // namespace

// A stupid and simple compiler that compiles a string into fusion IR. It is
// stupid because of the following limitations:
// - only support named scalars as variables
// - tokens must be separated by one and only one space, for example, i1+i2 and
//   i1  + i2 are all illegal, you have to write i1 + i2. Also note -5 is a
//   single negative integer constant, but - 5 is an expression neg(5)
// - poor error message
namespace stupid_simple_compiler {

using fun1_t = Val* (*)(Val*);
using fun2_t = Val* (*)(Val*, Val*);
struct LeftParenthesis {
  int64_t prev_lparen_pos;
};
struct FunctionCall {
  int64_t prev_lparen_pos;
  std::string_view name;
};
struct Comma {};
struct LowestPrecedence {};

using token_t = std::variant<
    Val*, // variable or constant
    fun1_t, // unary op
    fun2_t, // binary op
    LeftParenthesis,
    FunctionCall,
    Comma,
    LowestPrecedence>;

Val* parseIdentifier(std::string_view token_str) {
  if (token_str == "true") {
    return IrBuilder::newConstant(true, DataType::Bool);
  } else if (token_str == "false") {
    return IrBuilder::newConstant(false, DataType::Bool);
  } else if (
      token_str.at(0) == 'i' || token_str.at(0) == 'T' ||
      token_str == "threadIdx.x" || token_str == "threadIdx.y" ||
      token_str == "threadIdx.z" || token_str == "blockIdx.x" ||
      token_str == "blockIdx.y" || token_str == "blockIdx.z" ||
      token_str == "blockDim.x" || token_str == "blockDim.y" ||
      token_str == "blockDim.z" || token_str == "gridDim.x" ||
      token_str == "gridDim.y" || token_str == "gridDim.z") {
    return IrBuilder::create<NamedScalar>(
        std::string(token_str), DataType::Int);
  } else if (token_str.at(0) == 'b') {
    return IrBuilder::create<NamedScalar>(
        std::string(token_str), DataType::Bool);
  } else if (token_str.at(0) == 'd') {
    return IrBuilder::create<NamedScalar>(
        std::string(token_str), DataType::Double);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Identifier with unknown type: ", token_str);
  }
}

Val* parseNumber(std::string_view token_str) {
  auto s = token_str;
  bool neg = (s.at(0) == '-');
  if (neg) {
    s = s.substr(1);
  }
  TORCH_CHECK(!s.empty(), "Invalid number: ", token_str);
  int64_t i = 0;
  while (!s.empty()) {
    auto ch = s.at(0);
    if (ch == '.') {
      break;
    }
    TORCH_CHECK(std::isdigit(ch), "Invalid number: ", token_str)
    i = i * 10 + (ch - '0');
    s = s.substr(1);
  }
  if (s.empty()) {
    if (neg) {
      i = -i;
    }
    return IrBuilder::newConstant(i, DataType::Int);
  } else {
    s = s.substr(1);
    double d = i;
    double factor = 0.1;
    while (!s.empty()) {
      auto ch = s.at(0);
      TORCH_CHECK(std::isdigit(ch), "Invalid number: ", token_str)
      d += factor * (ch - '0');
      factor /= 10;
      s = s.substr(1);
    }
    if (neg) {
      d = -d;
    }
    return IrBuilder::newConstant(d, DataType::Double);
  }
}

Val* functionCall(std::string_view name, std::deque<Val*> args) {
  if (name == "max") {
    TORCH_CHECK(
        args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::maxExpr(args.at(0), args.at(1));
  } else if (name == "min") {
    TORCH_CHECK(
        args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::minExpr(args.at(0), args.at(1));
  } else if (name == "ceilDiv") {
    TORCH_CHECK(
        args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::ceilDivExpr(args.at(0), args.at(1));
  }
  TORCH_CHECK(false, "Unknown function: ", name);
}

token_t parseToken(std::string_view token_str, bool& expect_val) {
  if (std::isalpha(token_str.at(0))) {
    TORCH_CHECK(
        expect_val,
        "Syntax error: not expecting identifier but get ",
        token_str);
    expect_val = false;
    return parseIdentifier(token_str);
  } else if (token_str == "-") {
    if (expect_val) {
      return fun1_t(&neg);
    } else {
      expect_val = true;
      return fun2_t(&sub);
    }
  }
  if (token_str.at(0) == '!' || token_str.at(0) == '~') {
    TORCH_CHECK(
        expect_val, "Syntax error: not expecting unary op but get ", token_str);
    return fun1_t(&notOp);
  } else if (token_str.at(0) == '-' || std::isdigit(token_str.at(0))) {
    TORCH_CHECK(
        expect_val, "Syntax error: not expecting number but get ", token_str);
    expect_val = false;
    return parseNumber(token_str);
  } else {
    TORCH_CHECK(
        !expect_val,
        "Syntax error: not expecting operator but get ",
        token_str);
    expect_val = true;
    if (token_str.size() == 1) {
      switch (token_str.at(0)) {
        case '+':
          return fun2_t(&add);
        case '*':
          return fun2_t(&mul);
        case '/':
          return fun2_t(&div);
        case '%':
          return fun2_t(&mod);
        case '>':
          return fun2_t(&gt);
        case '<':
          return fun2_t(&lt);
      }
    } else if (token_str == "==") {
      return fun2_t(&eq);
    } else if (token_str == "!=") {
      return fun2_t(&eq);
    } else if (token_str == ">=") {
      return fun2_t(&ge);
    } else if (token_str == "<=") {
      return fun2_t(&le);
    } else if (token_str == "&&") {
      return fun2_t(&bitwise_and);
    } else if (token_str == "||") {
      return fun2_t(&bitwise_or);
    }
    TORCH_CHECK(false, "Unrecognized token: ", token_str);
  }
}

// https://en.cppreference.com/w/cpp/language/operator_precedence
int getOpPrecedence(token_t op) {
  if (std::holds_alternative<LowestPrecedence>(op)) {
    return std::numeric_limits<int>::max();
  }
  if (std::holds_alternative<Comma>(op)) {
    return 17;
  }
  if (std::holds_alternative<fun1_t>(op)) {
    auto uop = std::get<fun1_t>(op);
    if (uop == fun1_t(neg) || uop == fun1_t(notOp)) {
      return 3;
    }
    TORCH_CHECK(false, "Unexpected unary op");
  }

  if (std::holds_alternative<fun2_t>(op)) {
    auto bop = std::get<fun2_t>(op);
    if (bop == fun2_t(&mul) || bop == fun2_t(&div) || bop == fun2_t(&mod)) {
      return 5;
    }
    if (bop == fun2_t(&add) || bop == fun2_t(&sub)) {
      return 6;
    }
    if (bop == fun2_t(&lt) || bop == fun2_t(&le) || bop == fun2_t(&gt) ||
        bop == fun2_t(&ge)) {
      return 9;
    }
    if (bop == fun2_t(&eq) || bop == fun2_t(&ne)) {
      return 10;
    }
    if (bop == fun2_t(&bitwise_and)) {
      return 14;
    }
    if (bop == fun2_t(&bitwise_or)) {
      return 15;
    }
    TORCH_CHECK(false, "Unexpected binary op");
  }
  TORCH_CHECK(false, "Unexpected token");
}

Val* parse(const char* str) {
  std::string_view remaining(str);
  std::vector<token_t> op_stack;
  std::vector<Val*> value_stack;

  Val* current = nullptr;

  auto eval_top = [&]() {
    token_t op = op_stack.back();
    op_stack.pop_back();
    std::visit(
        [&](auto&& op) {
          using T = std::decay_t<decltype(op)>;
          if constexpr (std::is_same_v<T, fun1_t>) {
            current = op(current);
          } else if constexpr (std::is_same_v<T, fun2_t>) {
            TORCH_CHECK(!value_stack.empty(), "Missing operand for binary op");
            current = op(value_stack.back(), current);
            value_stack.pop_back();
          } else {
            TORCH_CHECK(false, "Unexpected token");
          }
        },
        op);
  };

  auto eval_all_top = [&](token_t token) {
    TORCH_CHECK(current != nullptr, "Expect value to evaluate top");
    while (!op_stack.empty() &&
           (std::holds_alternative<fun1_t>(op_stack.back()) ||
            std::holds_alternative<fun2_t>(op_stack.back())) &&
           getOpPrecedence(op_stack.back()) <= getOpPrecedence(token)) {
      eval_top();
    }
  };

  bool expect_val = true;
  int64_t last_lparen_pos = -1;

  while (!remaining.empty()) {
    const auto end_pos = remaining.find_first_of(' ');
    const auto token_str = remaining.substr(0, end_pos);

    if (token_str == "(") {
      TORCH_CHECK(
          expect_val, "Syntax error: not expecting ( but get ", token_str);
      op_stack.push_back(LeftParenthesis{last_lparen_pos});
      last_lparen_pos = op_stack.size() - 1;
    } else if (token_str.back() == '(') {
      TORCH_CHECK(
          expect_val,
          "Syntax error: not expecting function call but get ",
          token_str);
      op_stack.push_back(FunctionCall{
          last_lparen_pos, token_str.substr(0, token_str.size() - 1)});
      last_lparen_pos = op_stack.size() - 1;
    } else if (token_str == ",") {
      TORCH_CHECK(!expect_val, "Syntax error: not expecting comma");
      expect_val = true;
      auto comma = Comma{};
      eval_all_top(comma);
      value_stack.emplace_back(current);
      op_stack.emplace_back(comma);
      current = nullptr;
    } else if (token_str == ")") {
      TORCH_CHECK(
          !expect_val, "Syntax error: not expecting ) but get ", token_str);
      eval_all_top(LowestPrecedence{});
      auto last_lparen = op_stack.at(last_lparen_pos);
      TORCH_CHECK(!op_stack.empty(), "Unmatched )");
      if (std::holds_alternative<LeftParenthesis>(last_lparen)) {
        TORCH_INTERNAL_ASSERT(last_lparen_pos == (int64_t)op_stack.size() - 1);
        auto lparen = std::get<LeftParenthesis>(op_stack.back());
        last_lparen_pos = lparen.prev_lparen_pos;
        op_stack.pop_back();
      } else if (std::holds_alternative<FunctionCall>(last_lparen)) {
        std::deque<Val*> args{current};
        while (std::holds_alternative<Comma>(op_stack.back())) {
          op_stack.pop_back();
          args.push_front(value_stack.back());
          value_stack.pop_back();
        }
        auto fc = std::get<FunctionCall>(op_stack.back());
        last_lparen_pos = fc.prev_lparen_pos;
        op_stack.pop_back();
        current = functionCall(fc.name, std::move(args));
      } else {
        TORCH_CHECK(false, "Unknown left parenthesis type");
      }
    } else {
      token_t token = parseToken(token_str, expect_val);
      if (std::holds_alternative<Val*>(token)) {
        TORCH_CHECK(current == nullptr, "Don't expect value");
        current = std::get<Val*>(token);
      } else if (std::holds_alternative<fun1_t>(token)) {
        op_stack.push_back(token);
      } else if (std::holds_alternative<fun2_t>(token)) {
        eval_all_top(token);
        value_stack.push_back(current);
        op_stack.push_back(token);
        current = nullptr;
      } else {
        TORCH_CHECK(false, "Unexpected token");
      }
    }

    remaining = (end_pos != std::string_view::npos)
        ? remaining.substr(end_pos + 1)
        : "";
  }

  while (!op_stack.empty()) {
    eval_top();
  }

  return current;
}

// syntatic sugar to conveniently compile string into Val*
namespace ops {

Val* operator""_(const char* str, size_t) {
  return parse(str);
}

Bool* operator""_b(const char* str, size_t) {
  return parse(str)->as<Bool>();
}

} // namespace ops

} // namespace stupid_simple_compiler

using namespace stupid_simple_compiler::ops;

class ExprSimplifierTest : public NVFuserTest {};

TEST_F(ExprSimplifierTest, StupidSimpleCompiler_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  ASSERT_EQ(
      "( ( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0 )"_
          ->toInlineString(),
      "( ( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0 )");
  ASSERT_EQ(
      "( ( i1 * i2 ) - ( i2 * i1 ) )"_->toInlineString(),
      "( ( i1 * i2 ) - ( i2 * i1 ) )");
}

TEST_F(ExprSimplifierTest, AssociativeAndCommutativeReordering_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<VarInfo> variables(6);
  variables[0].variable = "i0"_;
  variables[1].variable = "i1"_;
  variables[2].variable = "i2"_;
  variables[3].variable = "i3"_;
  variables[4].variable = "i4"_;
  variables[5].variable = "i5"_;

  {
    auto val = "( ( i3 * i2 ) + i4 ) + ( i1 + 3 )"_;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    auto expect = "( ( ( 3 + i1 ) + ( i2 * i3 ) ) + i4 )"_;
    TORCH_CHECK(
        expect->sameAs(simplified) && simplified->sameAs(expect),
        "Expect the simplified expression ",
        simplified->toInlineString(),
        " to be the same as ",
        expect->toInlineString());
  }

  {
    auto val =
        "( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0"_;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    auto expect =
        "( i0 * ( ( ( 8 + i0 ) + i1 ) + i2 ) ) * ( ( ( 6 + ( i2 * i3 ) ) + i4 ) + i5 )"_;
    TORCH_CHECK(
        // Use isEquivalent to check equivalence because distributeMul will
        // expand the expression.
        isEquivalent(simplified, expect),
        "Expect the simplified expression ",
        simplified->toInlineString(),
        " to be the same as ",
        expect->toInlineString());
  }
}

TEST_F(ExprSimplifierTest, EliminateTrivialComputation_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption->as<Bool>()});
  };

  // constant folding
  ASSERT_TRUE(simplifyExpr("ceilDiv( 5 , 3 ) * 5"_)->sameAs("10"_));

  ASSERT_TRUE(simplifyExpr("1 * i"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("1.0 * d"_)->sameAs("d"_));
  ASSERT_TRUE(simplifyExpr("i * 1"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("d * 1.0"_)->sameAs("d"_));
  ASSERT_EQ(simplifyExpr("0 * i"_)->getInt(), 0);
  ASSERT_EQ(simplifyExpr("i * 0"_)->getInt(), 0);

  ASSERT_TRUE(simplifyExpr("0 + i"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("0.0 + d"_)->sameAs("d"_));
  ASSERT_TRUE(simplifyExpr("i + 0"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("d + 0.0"_)->sameAs("d"_));

  ASSERT_TRUE(simplifyExpr("true && b"_)->sameAs("b"_));
  ASSERT_TRUE(simplifyExpr("b && true"_)->sameAs("b"_));
  ASSERT_EQ(simplifyExpr("false && b"_)->getBool(), false);
  ASSERT_EQ(simplifyExpr("b && false"_)->getBool(), false);

  ASSERT_EQ(simplifyExpr("true || b"_)->getBool(), true);
  ASSERT_EQ(simplifyExpr("b || true"_)->getBool(), true);
  ASSERT_TRUE(simplifyExpr("false || b"_)->sameAs("b"_));
  ASSERT_TRUE(simplifyExpr("b || false"_)->sameAs("b"_));

  ASSERT_TRUE(simplifyExpr("b && b"_)->sameAs("b"_));
  ASSERT_TRUE(simplifyExpr("b || b"_)->sameAs("b"_));
  ASSERT_TRUE(simplifyExpr("max( i , i )"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("min( i , i )"_)->sameAs("i"_));
  ASSERT_TRUE(simplify("max( i1 , i2 )"_, "i1 <= i2"_)->sameAs("i2"_));
  ASSERT_TRUE(simplify("max( i2 , i1 )"_, "i1 <= i2"_)->sameAs("i2"_));
  ASSERT_TRUE(simplify("min( i1 , i2 )"_, "i1 <= i2"_)->sameAs("i1"_));
  ASSERT_TRUE(simplify("min( i2 , i1 )"_, "i1 <= i2"_)->sameAs("i1"_));

  ASSERT_TRUE(simplifyExpr("i / 1"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("d / 1.0"_)->sameAs("d"_));

  ASSERT_EQ(simplifyExpr("0 / i"_)->getInt(), 0);
  ASSERT_EQ(simplifyExpr("i % 1"_)->getInt(), 0);

  // -(-a) -> a
  ASSERT_TRUE(simplifyExpr("- - i"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("~ ~ i"_)->sameAs("i"_));
  ASSERT_TRUE(simplifyExpr("! ! b"_)->sameAs("b"_));

  // Test constant folding
  ASSERT_TRUE(simplifyExpr("1 + i + 1"_)->sameAs("i + 2"_));
  ASSERT_TRUE(simplifyExpr("1.0 + d + 1.0"_)->sameAs("d + 2.0"_));

  // Test that FlattenedAssocCommOp::sameAs ignores order
  ASSERT_TRUE(simplifyExpr("( i1 * i2 ) - ( i2 * i1 )"_)->isZeroInt());
}

TEST_F(ExprSimplifierTest, SimplifyDivisibleDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // assert that our system can correctly find that x is a multiple of y and z,
  // and simplify:
  // x % y -> 0
  // x % z -> 0
  // x / y -> z
  // and if x_div_z is true, also test
  // x / z -> y
  auto assertSimplifiedDivMod = [&fusion](Val* x, Val* y, Val* z) {
    assertSimplifiedMod(x, y, fusion.zeroVal());
    assertSimplifiedMod(x, z, fusion.zeroVal());
    assertSimplifiedDiv(x, y, z);
    assertSimplifiedDiv(x, z, y);
  };

  assertSimplifiedDivMod("6"_, "3"_, "2"_);
  assertSimplifiedDivMod("i1 * i2"_, "i1"_, "i2"_);
  assertSimplifiedDivMod("i1 * i2"_, "i1 * i2"_, "1"_);
  assertSimplifiedDivMod("i1 * i2 * i3"_, "i1"_, "i2 * i3"_);
  assertSimplifiedDivMod("i1 * i2 * i3"_, "i2"_, "i1 * i3"_);
  assertSimplifiedDivMod("i1 * i2 * i3"_, "i3"_, "i1 * i2"_);
  assertSimplifiedDivMod("i1 * i2 * i3"_, "i1 * ( i2 * i3 )"_, "1"_);
  assertSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i1"_, "i2 * i3 + i2 * i4"_);
  assertSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i2"_, "i1 * i3 + i1 * i4"_);
  assertSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i1 * i2"_, "i3 + i4"_);
  assertSimplifiedDivMod(
      "( i1 + i2 ) * i3 + ( i1 + i2 ) * i4"_, "i1 + i2"_, "i3 + i4"_);
  assertSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "( i1 * i2 ) * ( i3 * 6 )"_, "1"_);
  assertSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "i1 * ( i2 * i3 )"_, "6"_);
  assertSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "3"_, "( i1 * i2 ) * ( i3 * 2 )"_);
  assertSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "( i1 * i2 ) * ( i3 * 3 )"_, "2"_);
  assertSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "i1 * ( i3 * 3 )"_, "i2 * 2"_);
  assertSimplifiedDivMod("( i1 + ( i1 * i3 ) ) * i2"_, "i1 * i2"_, "1 + i3"_);
  assertSimplifiedDivMod(
      "( ( i1 * i2 ) + ( i1 * i3 ) ) * ( ( i2 * i1 ) + ( i2 * i4 ) )"_,
      "i1 * i2"_,
      "( i2 + i3 ) * ( i1 + i4 )"_);
  assertSimplifiedDivMod(
      "( 3 * i2 + 6 * i3 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( i2 + 2 * i3 ) * ( i1 + i4 )"_);
  assertSimplifiedDivMod(
      "( 3 * i2 + 6 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( i2 + 2 ) * ( i1 + i4 )"_);
  assertSimplifiedDivMod(
      "( 6 * i2 + 3 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( 2 * i2 + 1 ) * ( i1 + i4 )"_);
  assertSimplifiedDivMod("i1 * i2 * 3 + i2 * i1 * 6"_, "3 * i2 * i1"_, "3"_);
}

TEST_F(ExprSimplifierTest, SignProve_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  auto assertProvedPositive = [&fusion](
                                  Val* x,
                                  const std::vector<Bool*>& assumptions = {}) {
    auto proved =
        (simplifyExpr(IrBuilder::gtExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::geExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::ltExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::leExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::leExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == false) &&
        (simplifyExpr(IrBuilder::ltExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == false) &&
        (simplifyExpr(IrBuilder::geExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == false) &&
        (simplifyExpr(IrBuilder::gtExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == false);
    TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " > 0");
  };
  auto assertProvedNonNegative = [&fusion](
                                     Val* x,
                                     const std::vector<Bool*>& assumptions =
                                         {}) {
    auto proved =
        (simplifyExpr(IrBuilder::geExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::leExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::ltExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == false) &&
        (simplifyExpr(IrBuilder::gtExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == false);
    TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " >= 0");
  };
  auto assertProvedNonZero = [&fusion](
                                 Val* x,
                                 const std::vector<Bool*>& assumptions = {}) {
    auto proved =
        (simplifyExpr(IrBuilder::neExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::neExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == true) &&
        (simplifyExpr(IrBuilder::eqExpr(x, fusion.zeroVal()), {}, assumptions)
             ->getBool() == false) &&
        (simplifyExpr(IrBuilder::eqExpr(fusion.zeroVal(), x), {}, assumptions)
             ->getBool() == false);
    TORCH_CHECK(proved, "Unable to prove ", x->toInlineString(), " != 0");
  };

  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedPositive(NamedScalar::getParallelDim(ParallelType::TIDz));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedNonNegative(NamedScalar::getParallelDim(ParallelType::TIDz));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDx));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDy));
  assertProvedNonZero(NamedScalar::getParallelDim(ParallelType::TIDz));

  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDx));
  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDy));
  assertProvedNonNegative(NamedScalar::getParallelIndex(ParallelType::TIDz));

  assertProvedPositive("1"_);
  assertProvedPositive("2"_);
  assertProvedNonNegative("1"_);
  assertProvedNonNegative("2"_);
  assertProvedNonZero("1"_);
  assertProvedNonZero("2"_);

  assertProvedNonNegative("T123.size[3]"_);
  assertProvedNonNegative("T123.stride[3]"_);

  std::vector<Bool*> assumptions{
      "i1 < 2 && i1 >= 0"_b,
      "i2 < 2 && i2 >= 0"_b,
      "i3 < 2 && i3 >= 0"_b,
      "i4 < 2 && i4 >= 0"_b,
  };

  assertProvedNonNegative("i1"_, assumptions);
  assertProvedNonNegative("i2"_, assumptions);
  assertProvedNonNegative("i3"_, assumptions);
  assertProvedNonNegative("i4"_, assumptions);
  assertProvedNonNegative("i1 + i2"_, assumptions);
  assertProvedNonNegative("i1 + ( i2 + i3 )"_, assumptions);
  assertProvedNonNegative("( i1 + i4 ) * ( i2 + i3 )"_, assumptions);
  assertProvedNonNegative("( i1 + 2 ) * ( i2 + i3 )"_, assumptions);

  assertProvedPositive("( i1 + 2 ) + ( i2 + i3 )"_, assumptions);
  assertProvedNonZero("( i1 + 2 ) + ( i2 + i3 )"_, assumptions);

  assertProvedNonNegative(
      "( i4 + 1 ) / ( ( i1 + 2 ) + ( i2 + i3 ) )"_, assumptions);
  assertProvedNonNegative(
      "( i4 + 1 ) % ( ( i1 + 2 ) + ( i2 + i3 ) )"_, assumptions);
}

TEST_F(ExprSimplifierTest, PredicateProve_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<Bool*> assumptions{"i1 < 5 && i2 <= 5 && i3 > 5 && i4 >= 5"_b};
  ASSERT_EQ(simplifyExpr("i1 < 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("i1 <= 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 > i1"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 >= i1"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("i2 <= 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 >= i2"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("i3 > 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("i3 >= 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 < i3"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 <= i3"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("i4 >= 5"_, {}, assumptions)->getBool(), true);
  ASSERT_EQ(simplifyExpr("5 <= i4"_, {}, assumptions)->getBool(), true);
}

TEST_F(ExprSimplifierTest, EquivalenceSimplification_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto assertProvedEquiv = [](Val* x, Val* y) {
    auto proved = (simplifyExpr(IrBuilder::eqExpr(x, y))->getBool() == true) &&
        (simplifyExpr(IrBuilder::neExpr(x, y))->getBool() == false);
    TORCH_CHECK(
        proved,
        "Unable to prove ",
        x->toInlineString(),
        " == ",
        y->toInlineString());
  };

  assertProvedEquiv("i"_, "i"_);
  assertProvedEquiv("i1 * i2"_, "i2 * i1"_);
  assertProvedEquiv("( i1 * i3 ) % i2"_, "( i3 * i1 ) % i2"_);
}

TEST_F(ExprSimplifierTest, CancelDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  assertSimplifiedDiv(
      "6 * ( i1 * i3 )"_, "15 * ( i1 * i2 )"_, "( 2 * i3 ) / ( 5 * i2 )"_);
  assertSimplifiedMod(
      "6 * ( i1 * i3 )"_,
      "15 * ( i1 * i2 )"_,
      "( ( 2 * i3 ) % ( 5 * i2 ) ) * ( 3 * i1 )"_);
  assertSimplifiedDiv("( 3 * i1 )"_, "15 * ( i1 * i2 )"_, "1 / ( 5 * i2 )"_);
  assertSimplifiedMod(
      "( 3 * i1 )"_, "15 * ( i1 * i2 )"_, "( 1 % ( 5 * i2 ) ) * ( 3 * i1 )"_);
}

TEST_F(ExprSimplifierTest, DistributeDivisibleDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<Bool*> assumptions{"i1 >= 0 && i2 >= 0 && i3 >= 0"_b};

  assertSimplifiedDiv("i1 * i2 + i3"_, "i1"_, "i2 + i3 / i1"_, assumptions);
  assertSimplifiedMod("i1 * i2 + i3"_, "i1"_, "i3 % i1"_, assumptions);
}

TEST_F(ExprSimplifierTest, DistributeGcdRemainderDivMod_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  assertSimplifiedDiv("i1 * 3 + 2"_, "6"_, "i1 / 2"_, {"i1 >= 0"_b});
  assertSimplifiedMod(
      "i1 * 3 + 2"_, "6"_, "( i1 % 2 ) * 3 + 2"_, {"i1 >= 0"_b});
  assertSimplifiedDiv(
      "i1 * 4 + 3"_,
      "32 * T0.size[0]"_,
      "i1 / ( 8 * T0.size[0] )"_,
      {"i1 >= 0"_b});
  assertSimplifiedMod(
      "i1 * 4 + 3"_,
      "32 * T0.size[0]"_,
      "( i1 % ( 8 * T0.size[0] ) ) * 4 + 3"_,
      {"i1 >= 0"_b});
  assertSimplifiedDiv(
      "( ( ( blockIdx.x * 128 + threadIdx.x ) % ( T0.size[3] * 24 ) ) * 4 ) + 3"_,
      "32 * T0.size[3]"_,
      "( ( blockIdx.x * 128 + threadIdx.x ) % ( T0.size[3] * 24 ) ) / ( 8 * T0.size[3] )"_,
      {});
}

TEST_F(ExprSimplifierTest, DistributeMul_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TORCH_CHECK(isEquivalent("i1 * ( i2 + i3 )"_, "( i1 * i2 ) + ( i1 * i3 )"_));
  TORCH_CHECK(isEquivalent(
      "i1 * ( i2 + i3 + i4 )"_, "( i1 * i2 ) + ( i1 * i3 ) + ( i1 * i4 )"_));
}

TEST_F(ExprSimplifierTest, Compare_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption->as<Bool>()})->getBool();
  };

  ASSERT_TRUE(*simplify("i1 <= i1"_, "i1 < i2"_));

  ASSERT_TRUE(*simplify("i1 < i3"_, "i1 < i2 && i2 < i3"_));
  ASSERT_TRUE(*simplify("i1 < i3"_, "i1 < i2 && i2 <= i3"_));
  ASSERT_TRUE(*simplify("i1 < i3"_, "i1 <= i2 && i2 < i3"_));
  ASSERT_FALSE(simplify("i1 < i3"_, "i1 <= i2 && i2 <= i3"_).has_value());

  ASSERT_TRUE(*simplify("i1 > i3"_, "i1 > i2 && i2 > i3"_));
  ASSERT_TRUE(*simplify("i1 > i3"_, "i1 > i2 && i2 >= i3"_));
  ASSERT_TRUE(*simplify("i1 > i3"_, "i1 >= i2 && i2 > i3"_));
  ASSERT_FALSE(simplify("i1 > i3"_, "i1 >= i2 && i2 >= i3"_).has_value());

  ASSERT_TRUE(*simplify("i1 <= i3"_, "i1 < i2 && i2 < i3"_));
  ASSERT_TRUE(*simplify("i1 <= i3"_, "i1 < i2 && i2 <= i3"_));
  ASSERT_TRUE(*simplify("i1 <= i3"_, "i1 <= i2 && i2 < i3"_));
  ASSERT_TRUE(*simplify("i1 <= i3"_, "i1 <= i2 && i2 <= i3"_));

  ASSERT_TRUE(*simplify("i1 >= i3"_, "i1 > i2 && i2 > i3"_));
  ASSERT_TRUE(*simplify("i1 >= i3"_, "i1 > i2 && i2 >= i3"_));
  ASSERT_TRUE(*simplify("i1 >= i3"_, "i1 >= i2 && i2 > i3"_));
  ASSERT_TRUE(*simplify("i1 >= i3"_, "i1 >= i2 && i2 >= i3"_));

  ASSERT_TRUE(*simplify(
      "i1 < 3"_,
      "i1 < i2 && i2 <= i3 && i3 < i4 && i4 <= i5 && i5 <= i6 && i6 < i7 && i7 <= i8 && i8 <= 2"_));

  ASSERT_TRUE(*simplify("i1 <= i1 * i2"_, "i1 >= 0 && i2 > 0"_));
  ASSERT_TRUE(*simplify("i1 >= i1 * i2"_, "i1 <= 0 && i2 > 0"_));
  ASSERT_TRUE(*simplify("d1 <= d1 * d2"_, "d1 >= 0.0 && d2 >= 1.0"_));
  ASSERT_TRUE(*simplify("d1 >= d1 * d2"_, "d1 <= 0.0 && d2 >= 1.0"_));
  ASSERT_TRUE(
      *simplifyExpr(
           "ceilDiv( T0.size[0] , 128 ) * 4 >= ceilDiv( T0.size[0] , 128 )"_)
           ->getBool());

  ASSERT_TRUE(*simplify("ceilDiv( i1 , i2 ) > 0"_, "i1 > 0 && i2 > 0"_));
  ASSERT_TRUE(*simplify("ceilDiv( i1 , i2 ) >= 1"_, "i1 > 0 && i2 > 0"_));

  ASSERT_TRUE(*simplify(
      "blockIdx.x < ceilDiv( T0.size[0] , 128 ) * 4"_,
      "blockIdx.x < ceilDiv( T0.size[0] , 128 ) * 4"_));

  ASSERT_TRUE(*simplify("i1 % i2 < i2"_, "i2 >= 0"_));
}

TEST_F(ExprSimplifierTest, FundamentalDivisionWithRemainderProperty_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TORCH_CHECK(
      isEquivalent("i1 / T1.size[0] * T1.size[0] + i1 % T1.size[0]"_, "i1"_));
  TORCH_CHECK(isEquivalent(
      "( i2 + i1 / T1.size[0] * T1.size[0] ) + i1 % T1.size[0]"_, "i1 + i2"_));
  TORCH_CHECK(isEquivalent(
      "( i1 / T1.size[0] ) * ( T1.size[0] * T1.size[1] ) + T1.size[1] * ( i1 % T1.size[0] )"_,
      "i1 * T1.size[1]"_));
  TORCH_CHECK(isEquivalent(
      "i2 + ( i1 / T1.size[0] ) * ( T1.size[0] * T1.size[1] ) + T1.size[1] * ( i1 % T1.size[0] )"_,
      "i1 * T1.size[1] + i2"_));
}

TEST_F(ExprSimplifierTest, ReducePredicateRegisterUsage_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto u1 = IrBuilder::create<NamedScalar>("u1", DataType::Int);
  auto u2 = IrBuilder::create<NamedScalar>("u2", DataType::Int);
  auto tidx = NamedScalar::getParallelIndex(ParallelType::TIDx);
  auto zero = fusion.zeroVal();
  auto five = IrBuilder::create<Int>(5);
  auto neg_five = IrBuilder::create<Int>(-5);

  auto unroll_gp1 = mul(tidx, u1);
  auto unroll_uniform1 = mul(a, u1);
  auto unroll_imm1 = mul(five, u1);
  auto unroll_gp2 = mul(tidx, u2);
  auto unroll_uniform2 = mul(a, u2);
  auto unroll_imm2 = mul(five, u2);

  std::list<VarInfo> variables{VarInfo{u1, true}, VarInfo{u2, true}};

  // unroll + other == unroll + other
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), add(five, unroll_gp2));
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), add(five, unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), add(unroll_uniform2, five));
    auto v3_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_uniform1), unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), add(unroll_imm2, five));
    auto v4_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), add(unroll_imm2, five));
    auto v5_simplify = eq(add(a, neg_five), add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == unroll
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), unroll_gp2);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), unroll_uniform2);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), unroll_uniform2);
    auto v3_simplified = eq(tidx, add(neg(unroll_uniform1), unroll_uniform2));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), unroll_imm2);
    auto v4_simplified = eq(tidx, add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), unroll_imm2);
    auto v5_simplify = eq(a, add(neg(unroll_imm1), unroll_imm2));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == other
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), five);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), five);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), five);
    auto v3_simplified = eq(add(tidx, neg_five), neg(unroll_uniform1));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), five);
    auto v4_simplified = eq(add(tidx, neg_five), neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), five);
    auto v5_simplify = eq(add(a, neg_five), neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == 0
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), zero);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), zero);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), zero);
    auto v3_simplified = eq(tidx, neg(unroll_uniform1));
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), zero);
    auto v4_simplified = eq(tidx, neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), zero);
    auto v5_simplify = eq(a, neg(unroll_imm1));
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll == other
  {
    auto v1 = eq(unroll_gp1, five);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_uniform1, five);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_imm1, five);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == unroll
  {
    auto v1 = eq(five, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(five, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == other
  {
    auto v1 = eq(five, tidx);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, a);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
  }

  // unroll == unroll
  {
    auto v1 = eq(unroll_gp2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_gp2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_gp2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v3, variables)->sameAs(v3));
    auto v4 = eq(unroll_uniform2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v4, variables)->sameAs(v4));
    auto v5 = eq(unroll_uniform2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v5, variables)->sameAs(v5));
    auto v6 = eq(unroll_uniform2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v6, variables)->sameAs(v6));
    auto v7 = eq(unroll_imm2, unroll_gp1);
    TORCH_CHECK(simplifyExpr(v7, variables)->sameAs(v7));
    auto v8 = eq(unroll_imm2, unroll_uniform1);
    TORCH_CHECK(simplifyExpr(v8, variables)->sameAs(v8));
    auto v9 = eq(unroll_imm2, unroll_imm1);
    TORCH_CHECK(simplifyExpr(v9, variables)->sameAs(v9));
  }
}

TEST_F(ExprSimplifierTest, MinMax_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption->as<Bool>()});
  };

  auto expr =
      "max( max( ceilDiv( T0.size[0] , 128 ) * 4 , ceilDiv( T0.size[0] , 128 ) ) , 4 )"_;
  ASSERT_TRUE(simplify(expr, assume::tensorsAreNotEmpty(expr))
                  ->sameAs("ceilDiv( T0.size[0] , 128 ) * 4"_));
}

TEST_F(ExprSimplifierTest, Assume_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto expr =
      "max( max( ceilDiv( T0.size[0] , 128 ) * 4 , ceilDiv( T0.size[1] , 128 ) ) , 4 )"_;
  ASSERT_EQ(
      simplifyExpr(IrBuilder::eqExpr(
                       assume::tensorsAreNotEmpty(expr),
                       "T0.size[0] > 0 && T0.size[1] > 0"_))
          ->getBool(),
      true);
  expr = "ceilDiv( T0.size[0] , T0.size[0] ) * T0.size[0]"_;
  ASSERT_TRUE(assume::tensorsAreNotEmpty(expr)->sameAs("T0.size[0] > 0"_));
}

TEST_F(ExprSimplifierTest, PredicateDivToMul_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto simplified = simplifyExpr("i1 / T0.size[0] < i2"_, {}, {"i1 >= 0"_b});
  auto expect = "i1 < ( i2 * T0.size[0] )"_;

  ASSERT_TRUE(simplified->sameAs(expect));
}

} // namespace nvfuser
