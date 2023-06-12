// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <expr_simplifier.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

#include <cctype>
#include <deque>
#include <memory>
#include <random>
#include <regex>
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
  return simplifyExpr(IrBuilder::eqExpr(x, y))->getBool() == true;
}

// assert that x/y -> z
void expectSimplifiedDiv(
    Val* x,
    Val* y,
    Val* z,
    std::vector<Val*> assumptions = {}) {
  auto simplified = simplifyExpr(IrBuilder::divExpr(x, y), {}, assumptions);
  EXPECT_TRUE(isEquivalent(simplified, z))
      << "Expect " << x->toInlineString() << " / " << y->toInlineString()
      << " to be simplified to " << z->toInlineString() << ", but get "
      << simplified->toInlineString();
}

// assert that x % y -> z
void expectSimplifiedMod(
    Val* x,
    Val* y,
    Val* z,
    std::vector<Val*> assumptions = {}) {
  auto simplified = simplifyExpr(IrBuilder::modExpr(x, y), {}, assumptions);
  EXPECT_TRUE(isEquivalent(simplified, z))
      << "Expect " << x->toInlineString() << " % " << y->toInlineString()
      << " to be simplified to " << z->toInlineString() << ", but get "
      << simplified->toInlineString();
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
using fun3_t = Val* (*)(Val*, Val*, Val*);
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
    fun3_t, // ternary op
    LeftParenthesis,
    FunctionCall,
    Comma,
    LowestPrecedence>;

Val* randomlyReuseOrCreateNamedScalar(
    std::string_view name_view,
    DataType dtype) {
  Fusion* fusion = FusionGuard::getCurFusion();
  auto name = std::string(name_view);
  if (!fusion->hasManaged(name)) {
    auto ns = IrBuilder::create<NamedScalar>(name, dtype);
    fusion->manage(name, std::vector<Val*>{ns});
    return ns;
  }
  auto& existing_vals = fusion->getManaged<std::vector<Val*>>(name);
  auto index = (size_t)std::rand() % (existing_vals.size() + 1);
  if (index == existing_vals.size()) {
    // create a new one
    auto ns = IrBuilder::create<NamedScalar>(name, dtype);
    existing_vals.push_back(ns);
    return ns;
  } else {
    // reuse an existing one
    return existing_vals[index];
  }
}

Val* parseIdentifier(std::string_view token_str) {
  if (token_str == "true") {
    return IrBuilder::newConstant(true, DataType::Bool);
  } else if (token_str == "false") {
    return IrBuilder::newConstant(false, DataType::Bool);
  } else if (token_str.at(0) == 'T') {
    std::string s(token_str);
    std::regex regex(R"((T\d+)\.(\w+)\[(\d+)\])");
    std::smatch match;
    std::regex_search(s, match, regex);
    NVF_CHECK(match.size() == 4, "Invalid tensor access: ", token_str);
    auto tensor_name = match[1];
    auto attr = match[2];
    auto index = std::stol(match[3]);
    Fusion* fusion = FusionGuard::getCurFusion();
    TensorView* tv = nullptr;
    if (fusion->hasManaged(tensor_name)) {
      tv = fusion->getManaged<TensorView*>(tensor_name);
    } else {
      tv = makeSymbolicTensor(10);
      fusion->manage(tensor_name, tv);
    }
    return IrBuilder::getItemExpr(
        IrBuilder::getAttrExpr(IrBuilder::metadataExpr(tv), attr), index);
  } else if (
      token_str.at(0) == 'i' || token_str == "threadIdx.x" ||
      token_str == "threadIdx.y" || token_str == "threadIdx.z" ||
      token_str == "blockIdx.x" || token_str == "blockIdx.y" ||
      token_str == "blockIdx.z" || token_str == "blockDim.x" ||
      token_str == "blockDim.y" || token_str == "blockDim.z" ||
      token_str == "gridDim.x" || token_str == "gridDim.y" ||
      token_str == "gridDim.z") {
    return randomlyReuseOrCreateNamedScalar(token_str, DataType::Int);
  } else if (token_str.at(0) == 'b') {
    return randomlyReuseOrCreateNamedScalar(token_str, DataType::Bool);
  } else if (token_str.at(0) == 'd') {
    return randomlyReuseOrCreateNamedScalar(token_str, DataType::Double);
  } else {
    NVF_ERROR(false, "Identifier with unknown type: ", token_str);
  }
}

Val* parseNumber(std::string_view token_str) {
  auto s = token_str;
  bool neg = (s.at(0) == '-');
  if (neg) {
    s = s.substr(1);
  }
  NVF_CHECK(!s.empty(), "Invalid number: ", token_str);
  int64_t i = 0;
  while (!s.empty()) {
    auto ch = s.at(0);
    if (ch == '.') {
      break;
    }
    NVF_CHECK(std::isdigit(ch), "Invalid number: ", token_str)
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
      NVF_CHECK(std::isdigit(ch), "Invalid number: ", token_str)
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
  if (name == "gcd") {
    NVF_CHECK(args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::gcdExpr(args.at(0), args.at(1));
  } else if (name == "max") {
    NVF_CHECK(args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::maxExpr(args.at(0), args.at(1));
  } else if (name == "min") {
    NVF_CHECK(args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::minExpr(args.at(0), args.at(1));
  } else if (name == "ceilDiv") {
    NVF_CHECK(args.size() == 2, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::ceilDivExpr(args.at(0), args.at(1));
  } else if (name == "where") {
    NVF_CHECK(args.size() == 3, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::whereExpr(args.at(0), args.at(1), args.at(2));
  } else if (name == "abs") {
    NVF_CHECK(args.size() == 1, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::absExpr(args.at(0));
  }
  NVF_CHECK(false, "Unknown function: ", name);
}

token_t parseToken(std::string_view token_str, bool& expect_val) {
  if (std::isalpha(token_str.at(0))) {
    NVF_CHECK(
        expect_val,
        "Syntax error: not expecting identifier but get ",
        token_str);
    expect_val = false;
    return parseIdentifier(token_str);
  } else if (token_str == "-") {
    if (expect_val) {
      return fun1_t(&IrBuilder::negExpr);
    } else {
      expect_val = true;
      return fun2_t(&IrBuilder::subExpr);
    }
  }
  if (token_str.at(0) == '!' || token_str.at(0) == '~') {
    NVF_CHECK(
        expect_val, "Syntax error: not expecting unary op but get ", token_str);
    return fun1_t(&IrBuilder::logicalNotExpr);
  } else if (token_str.at(0) == '-' || std::isdigit(token_str.at(0))) {
    NVF_CHECK(
        expect_val, "Syntax error: not expecting number but get ", token_str);
    expect_val = false;
    return parseNumber(token_str);
  } else {
    NVF_CHECK(
        !expect_val,
        "Syntax error: not expecting operator but get ",
        token_str);
    expect_val = true;
    if (token_str.size() == 1) {
      switch (token_str.at(0)) {
        case '+':
          return fun2_t(&IrBuilder::addExpr);
        case '*':
          return fun2_t(&IrBuilder::mulExpr);
        case '/':
          return fun2_t(&IrBuilder::divExpr);
        case '%':
          return fun2_t(&IrBuilder::modExpr);
        case '>':
          return fun2_t(&IrBuilder::gtExpr);
        case '<':
          return fun2_t(&IrBuilder::ltExpr);
      }
    } else if (token_str == "==") {
      return fun2_t(&IrBuilder::eqExpr);
    } else if (token_str == "!=") {
      return fun2_t(&IrBuilder::neExpr);
    } else if (token_str == ">=") {
      return fun2_t(&IrBuilder::geExpr);
    } else if (token_str == "<=") {
      return fun2_t(&IrBuilder::leExpr);
    } else if (token_str == "&&") {
      return fun2_t(&IrBuilder::logicalAndExpr);
    } else if (token_str == "||") {
      return fun2_t(&IrBuilder::logicalOrExpr);
    }
    NVF_CHECK(false, "Unrecognized token: ", token_str);
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
    if (uop == fun1_t(IrBuilder::negExpr) ||
        uop == fun1_t(IrBuilder::logicalNotExpr)) {
      return 3;
    }
    NVF_CHECK(false, "Unexpected unary op");
  }

  if (std::holds_alternative<fun2_t>(op)) {
    auto bop = std::get<fun2_t>(op);
    if (bop == fun2_t(&IrBuilder::mulExpr) ||
        bop == fun2_t(&IrBuilder::divExpr) ||
        bop == fun2_t(&IrBuilder::modExpr)) {
      return 5;
    }
    if (bop == fun2_t(&IrBuilder::addExpr) ||
        bop == fun2_t(&IrBuilder::subExpr)) {
      return 6;
    }
    if (bop == fun2_t(&IrBuilder::ltExpr) ||
        bop == fun2_t(&IrBuilder::leExpr) ||
        bop == fun2_t(&IrBuilder::gtExpr) ||
        bop == fun2_t(&IrBuilder::geExpr)) {
      return 9;
    }
    if (bop == fun2_t(&IrBuilder::eqExpr) ||
        bop == fun2_t(&IrBuilder::neExpr)) {
      return 10;
    }
    if (bop == fun2_t(&IrBuilder::logicalAndExpr)) {
      return 14;
    }
    if (bop == fun2_t(&IrBuilder::logicalOrExpr)) {
      return 15;
    }
    NVF_CHECK(false, "Unexpected binary op");
  }
  NVF_CHECK(false, "Unexpected token");
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
            NVF_CHECK(!value_stack.empty(), "Missing operand for binary op");
            current = op(value_stack.back(), current);
            value_stack.pop_back();
          } else {
            NVF_CHECK(false, "Unexpected token");
          }
        },
        op);
  };

  auto eval_all_top = [&](token_t token) {
    NVF_CHECK(current != nullptr, "Expect value to evaluate top");
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
      NVF_CHECK(
          expect_val, "Syntax error: not expecting ( but get ", token_str);
      op_stack.push_back(LeftParenthesis{last_lparen_pos});
      last_lparen_pos = op_stack.size() - 1;
    } else if (token_str.back() == '(') {
      NVF_CHECK(
          expect_val,
          "Syntax error: not expecting function call but get ",
          token_str);
      op_stack.push_back(FunctionCall{
          last_lparen_pos, token_str.substr(0, token_str.size() - 1)});
      last_lparen_pos = op_stack.size() - 1;
    } else if (token_str == ",") {
      NVF_CHECK(!expect_val, "Syntax error: not expecting comma");
      expect_val = true;
      auto comma = Comma{};
      eval_all_top(comma);
      value_stack.emplace_back(current);
      op_stack.emplace_back(comma);
      current = nullptr;
    } else if (token_str == ")") {
      NVF_CHECK(
          !expect_val, "Syntax error: not expecting ) but get ", token_str);
      eval_all_top(LowestPrecedence{});
      auto last_lparen = op_stack.at(last_lparen_pos);
      NVF_CHECK(!op_stack.empty(), "Unmatched )");
      if (std::holds_alternative<LeftParenthesis>(last_lparen)) {
        NVF_ERROR(last_lparen_pos == (int64_t)op_stack.size() - 1);
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
        NVF_CHECK(false, "Unknown left parenthesis type");
      }
    } else {
      token_t token = parseToken(token_str, expect_val);
      if (std::holds_alternative<Val*>(token)) {
        NVF_CHECK(current == nullptr, "Don't expect value");
        current = std::get<Val*>(token);
      } else if (std::holds_alternative<fun1_t>(token)) {
        op_stack.push_back(token);
      } else if (std::holds_alternative<fun2_t>(token)) {
        eval_all_top(token);
        value_stack.push_back(current);
        op_stack.push_back(token);
        current = nullptr;
      } else {
        NVF_CHECK(false, "Unexpected token");
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

} // namespace ops

} // namespace stupid_simple_compiler

using namespace stupid_simple_compiler::ops;

class ExprSimplifierTest : public NVFuserTest {
  std::unique_ptr<Fusion> fusion_ptr;
  std::unique_ptr<FusionGuard> fusion_guard_ptr;

  void SetUp() override {
    NVFuserTest::SetUp();
    fusion_ptr = std::make_unique<Fusion>();
    fusion_guard_ptr = std::make_unique<FusionGuard>(fusion_ptr.get());
  }
};

TEST_F(ExprSimplifierTest, StupidSimpleCompiler) {
  EXPECT_EQ(
      "( ( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0 )"_
          ->toInlineString(),
      "( ( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0 )");
  EXPECT_EQ(
      "( ( i1 * i2 ) - ( i2 * i1 ) )"_->toInlineString(),
      "( ( i1 * i2 ) - ( i2 * i1 ) )");
}

TEST_F(ExprSimplifierTest, AssociativeAndCommutativeReordering) {
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
    EXPECT_TRUE(expect->sameAs(simplified) && simplified->sameAs(expect))
        << "Expect the simplified expression " << simplified->toInlineString()
        << " to be the same as " << expect->toInlineString();
  }

  {
    auto val = "i3 + i2 - i1 + i0"_;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    auto expect = "i0 - i1 + i2 + i3"_;
    EXPECT_TRUE(expect->sameAs(simplified) && simplified->sameAs(expect))
        << "Expect the simplified expression " << simplified->toInlineString()
        << " to be the same as " << expect->toInlineString();
  }

  {
    auto val = "i3 + i2 + i1 - i0"_;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    auto expect = "- i0 + i1 + i2 + i3"_;
    EXPECT_TRUE(expect->sameAs(simplified) && simplified->sameAs(expect))
        << "Expect the simplified expression " << simplified->toInlineString()
        << " to be the same as " << expect->toInlineString();
  }

  {
    auto val =
        "( ( ( ( i2 * i3 ) + ( ( i4 + i5 ) + 3 ) ) + 3 ) * ( ( ( ( i0 + i1 ) + 3 ) + 5 ) + i2 ) ) * i0"_;
    auto simplified = simplifyExpr(val, {variables.begin(), variables.end()});
    auto expect =
        "( i0 * ( ( ( 8 + i0 ) + i1 ) + i2 ) ) * ( ( ( 6 + ( i2 * i3 ) ) + i4 ) + i5 )"_;
    EXPECT_TRUE(
        // Use isEquivalent to check equivalence because distributeMul will
        // expand the expression.
        isEquivalent(simplified, expect))
        << "Expect the simplified expression " << simplified->toInlineString()
        << " to be the same as " << expect->toInlineString();
  }
}

TEST_F(ExprSimplifierTest, EliminateTrivialComputation) {
  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption});
  };

  // constant folding
  EXPECT_TRUE(simplifyExpr("ceilDiv( 5 , 3 ) * 5"_)->sameAs("10"_));

  EXPECT_TRUE(simplifyExpr("1 * i"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("1.0 * d"_)->sameAs("d"_));
  EXPECT_TRUE(simplifyExpr("i * 1"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("d * 1.0"_)->sameAs("d"_));
  EXPECT_EQ(simplifyExpr("0 * i"_)->getInt(), 0);
  EXPECT_EQ(simplifyExpr("i * 0"_)->getInt(), 0);
  EXPECT_TRUE(simplifyExpr("gcd( i , 0 )"_)->sameAs("abs( i )"_));

  EXPECT_TRUE(simplifyExpr("0 + i"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("0.0 + d"_)->sameAs("d"_));
  EXPECT_TRUE(simplifyExpr("i + 0"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("d + 0.0"_)->sameAs("d"_));
  EXPECT_EQ(simplifyExpr("gcd( i , 1 )"_)->getInt(), 1);

  EXPECT_TRUE(simplifyExpr("true && b"_)->sameAs("b"_));
  EXPECT_TRUE(simplifyExpr("b && true"_)->sameAs("b"_));
  EXPECT_EQ(simplifyExpr("false && b"_)->getBool(), false);
  EXPECT_EQ(simplifyExpr("b && false"_)->getBool(), false);

  EXPECT_EQ(simplifyExpr("true || b"_)->getBool(), true);
  EXPECT_EQ(simplifyExpr("b || true"_)->getBool(), true);
  EXPECT_TRUE(simplifyExpr("false || b"_)->sameAs("b"_));
  EXPECT_TRUE(simplifyExpr("b || false"_)->sameAs("b"_));

  EXPECT_TRUE(simplifyExpr("b && b"_)->sameAs("b"_));
  EXPECT_TRUE(simplifyExpr("b || b"_)->sameAs("b"_));
  EXPECT_TRUE(simplifyExpr("max( i , i )"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("min( i , i )"_)->sameAs("i"_));
  EXPECT_TRUE(simplify("max( i1 , i2 )"_, "i1 <= i2"_)->sameAs("i2"_));
  EXPECT_TRUE(simplify("max( i2 , i1 )"_, "i1 <= i2"_)->sameAs("i2"_));
  EXPECT_TRUE(simplify("min( i1 , i2 )"_, "i1 <= i2"_)->sameAs("i1"_));
  EXPECT_TRUE(simplify("min( i2 , i1 )"_, "i1 <= i2"_)->sameAs("i1"_));

  EXPECT_TRUE(simplifyExpr("i / 1"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("d / 1.0"_)->sameAs("d"_));

  EXPECT_EQ(simplifyExpr("0 / i"_)->getInt(), 0);
  EXPECT_EQ(simplifyExpr("i % 1"_)->getInt(), 0);

  // -(-a) -> a
  EXPECT_TRUE(simplifyExpr("- - i"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("~ ~ i"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("! ! b"_)->sameAs("b"_));

  // Test constant folding
  EXPECT_TRUE(simplifyExpr("1 + i + 1"_)->sameAs("i + 2"_));
  EXPECT_TRUE(simplifyExpr("1.0 + d + 1.0"_)->sameAs("d + 2.0"_));

  // Test that FlattenedAssocCommOp::sameAs ignores order
  EXPECT_TRUE(simplifyExpr("( i1 * i2 ) == ( i2 * i1 )"_)->isTrue());

  // where(true, x, y) -> x, where(false, x, y) -> y
  EXPECT_TRUE(simplifyExpr("where( true , i1 , i2 )"_)->sameAs("i1"_));
  EXPECT_TRUE(simplifyExpr("where( false , i1 , i2 )"_)->sameAs("i2"_));

  // abs(x) -> x, if x >= 0
  EXPECT_TRUE(simplifyExpr("abs( i )"_, {}, {"i >= 0"_})->sameAs("i"_));

  // x - x -> 0
  EXPECT_TRUE(simplifyExpr("i - i"_)->isZeroInt());
  EXPECT_TRUE(simplifyExpr("i - i - i"_)->sameAs("- i"_));
  EXPECT_TRUE(simplifyExpr("i - i + i"_)->sameAs("i"_));
  EXPECT_TRUE(simplifyExpr("i - i + i - i"_)->isZeroInt());
  EXPECT_TRUE(simplifyExpr("i1 - ( i2 + i3 ) + i2"_)->sameAs("i1 - i3"_));
  EXPECT_TRUE(simplifyExpr("i2 - ( i2 - i3 ) - i3"_)->isZeroInt());
  EXPECT_TRUE(simplifyExpr("i1 - ( i2 - i3 ) - i3"_)->sameAs("i1 - i2"_));
  // Using the same Val* multiple times in FlattenedAdd so that we can test if
  // our passes are working correctly with the same Val* appearing multiple
  // times
  auto i = "i"_;
  EXPECT_TRUE(
      simplifyExpr(IrBuilder::subExpr(IrBuilder::addExpr(i, i), i))->sameAs(i));
}

TEST_F(ExprSimplifierTest, SimplifyDivisibleDivMod) {
  // assert that our system can correctly find that x is a multiple of y and z,
  // and simplify:
  // x % y -> 0
  // x % z -> 0
  // x / y -> z
  // and if x_div_z is true, also test
  // x / z -> y
  auto expectSimplifiedDivMod = [](Val* x, Val* y, Val* z) {
    expectSimplifiedMod(x, y, "0"_);
    expectSimplifiedMod(x, z, "0"_);
    expectSimplifiedDiv(x, y, z);
    expectSimplifiedDiv(x, z, y);
  };

  expectSimplifiedDivMod("6"_, "3"_, "2"_);
  expectSimplifiedDivMod("i1 * i2"_, "i1"_, "i2"_);
  expectSimplifiedDivMod("i1 * i2"_, "i1 * i2"_, "1"_);
  expectSimplifiedDivMod("i1 * i2 * i3"_, "i1"_, "i2 * i3"_);
  expectSimplifiedDivMod("i1 * i2 * i3"_, "i2"_, "i1 * i3"_);
  expectSimplifiedDivMod("i1 * i2 * i3"_, "i3"_, "i1 * i2"_);
  expectSimplifiedDivMod("i1 * i2 * i3"_, "i1 * ( i2 * i3 )"_, "1"_);
  expectSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i1"_, "i2 * i3 + i2 * i4"_);
  expectSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i2"_, "i1 * i3 + i1 * i4"_);
  expectSimplifiedDivMod(
      "i1 * i2 * i3 + i1 * i2 * i4"_, "i1 * i2"_, "i3 + i4"_);
  expectSimplifiedDivMod(
      "( i1 + i2 ) * i3 + ( i1 + i2 ) * i4"_, "i1 + i2"_, "i3 + i4"_);
  expectSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "( i1 * i2 ) * ( i3 * 6 )"_, "1"_);
  expectSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "i1 * ( i2 * i3 )"_, "6"_);
  expectSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "3"_, "( i1 * i2 ) * ( i3 * 2 )"_);
  expectSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "( i1 * i2 ) * ( i3 * 3 )"_, "2"_);
  expectSimplifiedDivMod(
      "( i1 * i2 ) * ( i3 * 6 )"_, "i1 * ( i3 * 3 )"_, "i2 * 2"_);
  expectSimplifiedDivMod("( i1 + ( i1 * i3 ) ) * i2"_, "i1 * i2"_, "1 + i3"_);
  expectSimplifiedDivMod(
      "( ( i1 * i2 ) + ( i1 * i3 ) ) * ( ( i2 * i1 ) + ( i2 * i4 ) )"_,
      "i1 * i2"_,
      "( i2 + i3 ) * ( i1 + i4 )"_);
  expectSimplifiedDivMod(
      "( 3 * i2 + 6 * i3 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( i2 + 2 * i3 ) * ( i1 + i4 )"_);
  expectSimplifiedDivMod(
      "( 3 * i2 + 6 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( i2 + 2 ) * ( i1 + i4 )"_);
  expectSimplifiedDivMod(
      "( 6 * i2 + 3 ) * ( i2 * i1 + i2 * i4 )"_,
      "3 * i2"_,
      "( 2 * i2 + 1 ) * ( i1 + i4 )"_);
  expectSimplifiedDivMod("i1 * i2 * 3 + i2 * i1 * 6"_, "3 * i2 * i1"_, "3"_);
}

TEST_F(ExprSimplifierTest, SignProve) {
  auto assertProvedPositive = [](Val* x,
                                 const std::vector<Val*>& assumptions = {}) {
    auto proved =
        (simplifyExpr(IrBuilder::gtExpr(x, "0"_), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::geExpr(x, "0"_), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::ltExpr("0"_, x), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::leExpr("0"_, x), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::leExpr(x, "0"_), {}, assumptions)->getBool() ==
         false) &&
        (simplifyExpr(IrBuilder::ltExpr(x, "0"_), {}, assumptions)->getBool() ==
         false) &&
        (simplifyExpr(IrBuilder::geExpr("0"_, x), {}, assumptions)->getBool() ==
         false) &&
        (simplifyExpr(IrBuilder::gtExpr("0"_, x), {}, assumptions)->getBool() ==
         false);
    EXPECT_TRUE(proved) << "Unable to prove " << x->toInlineString() << " > 0";
  };
  auto assertProvedNonNegative = [](Val* x,
                                    const std::vector<Val*>& assumptions = {}) {
    auto proved =
        (simplifyExpr(IrBuilder::geExpr(x, "0"_), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::leExpr("0"_, x), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::ltExpr(x, "0"_), {}, assumptions)->getBool() ==
         false) &&
        (simplifyExpr(IrBuilder::gtExpr("0"_, x), {}, assumptions)->getBool() ==
         false);
    EXPECT_TRUE(proved) << "Unable to prove " << x->toInlineString() << " >= 0";
  };
  auto assertProvedNonZero = [](Val* x,
                                const std::vector<Val*>& assumptions = {}) {
    auto proved =
        (simplifyExpr(IrBuilder::neExpr(x, "0"_), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::neExpr("0"_, x), {}, assumptions)->getBool() ==
         true) &&
        (simplifyExpr(IrBuilder::eqExpr(x, "0"_), {}, assumptions)->getBool() ==
         false) &&
        (simplifyExpr(IrBuilder::eqExpr("0"_, x), {}, assumptions)->getBool() ==
         false);
    EXPECT_TRUE(proved) << "Unable to prove " << x->toInlineString() << " != 0";
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

  assertProvedNonNegative("T123.logical_size[3]"_);
  assertProvedNonNegative("T123.alloc_stride[3]"_);

  std::vector<Val*> assumptions{
      "i1 < 2 && i1 >= 0"_,
      "i2 < 2 && i2 >= 0"_,
      "i3 < 2 && i3 >= 0"_,
      "i4 < 2 && i4 >= 0"_,
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

TEST_F(ExprSimplifierTest, PredicateProve) {
  std::vector<Val*> assumptions{"i1 < 5 && i2 <= 5 && i3 > 5 && i4 >= 5"_};
  EXPECT_EQ(simplifyExpr("i1 < 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("i1 <= 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 > i1"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 >= i1"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("i2 <= 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 >= i2"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("i3 > 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("i3 >= 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 < i3"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 <= i3"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("i4 >= 5"_, {}, assumptions)->getBool(), true);
  EXPECT_EQ(simplifyExpr("5 <= i4"_, {}, assumptions)->getBool(), true);
}

TEST_F(ExprSimplifierTest, EquivalenceSimplification) {
  auto assertProvedEquiv = [](Val* x, Val* y) {
    auto proved = (simplifyExpr(IrBuilder::eqExpr(x, y))->getBool() == true) &&
        (simplifyExpr(IrBuilder::neExpr(x, y))->getBool() == false);
    EXPECT_TRUE(proved) << "Unable to prove " << x->toInlineString()
                        << " == " << y->toInlineString();
  };

  assertProvedEquiv("i"_, "i"_);
  assertProvedEquiv("i1 * i2"_, "i2 * i1"_);
  assertProvedEquiv("( i1 * i3 ) % i2"_, "( i3 * i1 ) % i2"_);
}

TEST_F(ExprSimplifierTest, CancelDivMod) {
  expectSimplifiedDiv(
      "6 * ( i1 * i3 )"_, "15 * ( i1 * i2 )"_, "( 2 * i3 ) / ( 5 * i2 )"_);
  expectSimplifiedMod(
      "6 * ( i1 * i3 )"_,
      "15 * ( i1 * i2 )"_,
      "( ( 2 * i3 ) % ( 5 * i2 ) ) * ( 3 * i1 )"_);
  expectSimplifiedDiv("( 3 * i1 )"_, "15 * ( i1 * i2 )"_, "1 / ( 5 * i2 )"_);
  expectSimplifiedMod(
      "( 3 * i1 )"_, "15 * ( i1 * i2 )"_, "( 1 % ( 5 * i2 ) ) * ( 3 * i1 )"_);
}

TEST_F(ExprSimplifierTest, DistributeDivisibleDivMod) {
  std::vector<Val*> assumptions{"i1 >= 0 && i2 >= 0 && i3 >= 0"_};

  expectSimplifiedDiv("i1 * i2 + i3"_, "i1"_, "i2 + i3 / i1"_, assumptions);
  expectSimplifiedMod("i1 * i2 + i3"_, "i1"_, "i3 % i1"_, assumptions);
}

TEST_F(ExprSimplifierTest, DistributeGcdRemainderDivMod) {
  expectSimplifiedDiv("i1 * 3 + 2"_, "6"_, "i1 / 2"_, {"i1 >= 0"_});
  expectSimplifiedMod("i1 * 3 + 2"_, "6"_, "( i1 % 2 ) * 3 + 2"_, {"i1 >= 0"_});
  expectSimplifiedDiv(
      "i1 * 4 + 3"_,
      "32 * T0.logical_size[0]"_,
      "i1 / ( 8 * T0.logical_size[0] )"_,
      {"i1 >= 0"_});
  expectSimplifiedMod(
      "i1 * 4 + 3"_,
      "32 * T0.logical_size[0]"_,
      "( i1 % ( 8 * T0.logical_size[0] ) ) * 4 + 3"_,
      {"i1 >= 0"_});
  expectSimplifiedDiv(
      "( ( ( blockIdx.x * 128 + threadIdx.x ) % ( T0.logical_size[3] * 24 ) ) * 4 ) + 3"_,
      "32 * T0.logical_size[3]"_,
      "( ( blockIdx.x * 128 + threadIdx.x ) % ( T0.logical_size[3] * 24 ) ) / ( 8 * T0.logical_size[3] )"_,
      {});
}

TEST_F(ExprSimplifierTest, DistributeMul) {
  EXPECT_TRUE(isEquivalent("i1 * ( i2 + i3 )"_, "( i1 * i2 ) + ( i1 * i3 )"_));
  EXPECT_TRUE(isEquivalent(
      "i1 * ( i2 + i3 + i4 )"_, "( i1 * i2 ) + ( i1 * i3 ) + ( i1 * i4 )"_));
}

TEST_F(ExprSimplifierTest, Compare) {
  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption})->getBool();
  };

  EXPECT_TRUE(*simplify("i1 <= i1"_, "i1 < i2"_));

  EXPECT_TRUE(*simplify("i1 < i3"_, "i1 < i2 && i2 < i3"_));
  EXPECT_TRUE(*simplify("i1 < i3"_, "i1 < i2 && i2 <= i3"_));
  EXPECT_TRUE(*simplify("i1 < i3"_, "i1 <= i2 && i2 < i3"_));
  EXPECT_FALSE(simplify("i1 < i3"_, "i1 <= i2 && i2 <= i3"_).has_value());

  EXPECT_TRUE(*simplify("i1 > i3"_, "i1 > i2 && i2 > i3"_));
  EXPECT_TRUE(*simplify("i1 > i3"_, "i1 > i2 && i2 >= i3"_));
  EXPECT_TRUE(*simplify("i1 > i3"_, "i1 >= i2 && i2 > i3"_));
  EXPECT_FALSE(simplify("i1 > i3"_, "i1 >= i2 && i2 >= i3"_).has_value());

  EXPECT_TRUE(*simplify("i1 <= i3"_, "i1 < i2 && i2 < i3"_));
  EXPECT_TRUE(*simplify("i1 <= i3"_, "i1 < i2 && i2 <= i3"_));
  EXPECT_TRUE(*simplify("i1 <= i3"_, "i1 <= i2 && i2 < i3"_));
  EXPECT_TRUE(*simplify("i1 <= i3"_, "i1 <= i2 && i2 <= i3"_));

  EXPECT_TRUE(*simplify("i1 >= i3"_, "i1 > i2 && i2 > i3"_));
  EXPECT_TRUE(*simplify("i1 >= i3"_, "i1 > i2 && i2 >= i3"_));
  EXPECT_TRUE(*simplify("i1 >= i3"_, "i1 >= i2 && i2 > i3"_));
  EXPECT_TRUE(*simplify("i1 >= i3"_, "i1 >= i2 && i2 >= i3"_));

  EXPECT_TRUE(*simplify(
      "i1 < 3"_,
      "i1 < i2 && i2 <= i3 && i3 < i4 && i4 <= i5 && i5 <= i6 && i6 < i7 && i7 <= i8 && i8 <= 2"_));

  EXPECT_TRUE(*simplify("i1 <= i1 * i2"_, "i1 >= 0 && i2 > 0"_));
  EXPECT_TRUE(*simplify("i1 >= i1 * i2"_, "i1 <= 0 && i2 > 0"_));
  EXPECT_TRUE(*simplify("d1 <= d1 * d2"_, "d1 >= 0.0 && d2 >= 1.0"_));
  EXPECT_TRUE(*simplify("d1 >= d1 * d2"_, "d1 <= 0.0 && d2 >= 1.0"_));
  EXPECT_TRUE(
      *simplifyExpr(
           "ceilDiv( T0.logical_size[0] , 128 ) * 4 >= ceilDiv( T0.logical_size[0] , 128 )"_)
           ->getBool());

  EXPECT_TRUE(*simplify("ceilDiv( i1 , i2 ) > 0"_, "i1 > 0 && i2 > 0"_));
  EXPECT_TRUE(*simplify("ceilDiv( i1 , i2 ) >= 1"_, "i1 > 0 && i2 > 0"_));

  EXPECT_TRUE(*simplify(
      "blockIdx.x < ceilDiv( T0.logical_size[0] , 128 ) * 4"_,
      "blockIdx.x < ceilDiv( T0.logical_size[0] , 128 ) * 4"_));

  EXPECT_TRUE(*simplify("i1 % i2 < i2"_, "i2 >= 0"_));

  EXPECT_TRUE(
      *simplifyExpr("T0.logical_size[0] - 1 < T0.logical_size[0]"_)->getBool());
  EXPECT_TRUE(
      *simplifyExpr(
           "T0.logical_size[0] + 1 + 2 + 3 < T0.logical_size[0] + 1 + 2 + 3 + 4"_)
           ->getBool());
  // Two terms of the LHS are both the same as the single RHS term,
  // but the removal should be done only for one of them. If doubly
  // removed, the predicate would be false
  EXPECT_TRUE(*simplify("i1 + i1 > i1"_, "i1 > 0"_));
  EXPECT_TRUE(*simplify("i1 < i1 + i1"_, "i1 > 0"_));
  EXPECT_TRUE(*simplify("i1 + i1 < i1 + i1 + i1"_, "i1 > 0"_));
}

TEST_F(ExprSimplifierTest, FundamentalDivisionWithRemainderProperty) {
  EXPECT_TRUE(isEquivalent(
      "i1 / T1.logical_size[0] * T1.logical_size[0] + i1 % T1.logical_size[0]"_,
      "i1"_));
  EXPECT_TRUE(isEquivalent(
      "( i2 + i1 / T1.logical_size[0] * T1.logical_size[0] ) + i1 % T1.logical_size[0]"_,
      "i1 + i2"_));
  EXPECT_TRUE(isEquivalent(
      "( i1 / T1.logical_size[0] ) * ( T1.logical_size[0] * T1.logical_size[1] ) + T1.logical_size[1] * ( i1 % T1.logical_size[0] )"_,
      "i1 * T1.logical_size[1]"_));
  EXPECT_TRUE(isEquivalent(
      "i2 + ( i1 / T1.logical_size[0] ) * ( T1.logical_size[0] * T1.logical_size[1] ) + T1.logical_size[1] * ( i1 % T1.logical_size[0] )"_,
      "i1 * T1.logical_size[1] + i2"_));
}

TEST_F(ExprSimplifierTest, ReducePredicateRegisterUsage) {
  auto a = IrBuilder::create<NamedScalar>("a", DataType::Int);
  auto b = IrBuilder::create<NamedScalar>("b", DataType::Int);
  auto u1 = IrBuilder::create<NamedScalar>("u1", DataType::Int);
  auto u2 = IrBuilder::create<NamedScalar>("u2", DataType::Int);
  auto tidx = NamedScalar::getParallelIndex(ParallelType::TIDx);
  auto zero = "0"_;
  auto five = IrBuilder::create<Val>(5L);
  auto neg_five = IrBuilder::create<Val>(-5L);

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
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), add(five, unroll_uniform2));
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), add(unroll_uniform2, five));
    auto v3_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_uniform1), unroll_uniform2));
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), add(unroll_imm2, five));
    auto v4_simplified =
        eq(add(tidx, neg_five), add(neg(unroll_imm1), unroll_imm2));
    EXPECT_TRUE(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), add(unroll_imm2, five));
    auto v5_simplify = eq(add(a, neg_five), add(neg(unroll_imm1), unroll_imm2));
    EXPECT_TRUE(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == unroll
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), unroll_gp2);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), unroll_uniform2);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), unroll_uniform2);
    auto v3_simplified = eq(tidx, add(neg(unroll_uniform1), unroll_uniform2));
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), unroll_imm2);
    auto v4_simplified = eq(tidx, add(neg(unroll_imm1), unroll_imm2));
    EXPECT_TRUE(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), unroll_imm2);
    auto v5_simplify = eq(a, add(neg(unroll_imm1), unroll_imm2));
    EXPECT_TRUE(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == other
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), five);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), five);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), five);
    auto v3_simplified = eq(add(tidx, neg_five), neg(unroll_uniform1));
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), five);
    auto v4_simplified = eq(add(tidx, neg_five), neg(unroll_imm1));
    EXPECT_TRUE(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), five);
    auto v5_simplify = eq(add(a, neg_five), neg(unroll_imm1));
    EXPECT_TRUE(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll + other == 0
  {
    // can not save
    auto v1 = eq(add(b, unroll_gp1), zero);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(add(b, unroll_uniform1), zero);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));

    // save general purpose register
    auto v3 = eq(add(unroll_uniform1, tidx), zero);
    auto v3_simplified = eq(tidx, neg(unroll_uniform1));
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3_simplified));
    auto v4 = eq(add(unroll_imm1, tidx), zero);
    auto v4_simplified = eq(tidx, neg(unroll_imm1));
    EXPECT_TRUE(simplifyExpr(v4, variables)->sameAs(v4_simplified));

    // unroll + other == unroll + other, save uniform register
    auto v5 = eq(add(unroll_imm1, a), zero);
    auto v5_simplify = eq(a, neg(unroll_imm1));
    EXPECT_TRUE(simplifyExpr(v5, variables)->sameAs(v5_simplify));
  }

  // unroll == other
  {
    auto v1 = eq(unroll_gp1, five);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_uniform1, five);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_imm1, five);
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == unroll
  {
    auto v1 = eq(five, unroll_gp1);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, unroll_uniform1);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(five, unroll_imm1);
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3));
  }

  // other == other
  {
    auto v1 = eq(five, tidx);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(five, a);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));
  }

  // unroll == unroll
  {
    auto v1 = eq(unroll_gp2, unroll_gp1);
    EXPECT_TRUE(simplifyExpr(v1, variables)->sameAs(v1));
    auto v2 = eq(unroll_gp2, unroll_uniform1);
    EXPECT_TRUE(simplifyExpr(v2, variables)->sameAs(v2));
    auto v3 = eq(unroll_gp2, unroll_imm1);
    EXPECT_TRUE(simplifyExpr(v3, variables)->sameAs(v3));
    auto v4 = eq(unroll_uniform2, unroll_gp1);
    EXPECT_TRUE(simplifyExpr(v4, variables)->sameAs(v4));
    auto v5 = eq(unroll_uniform2, unroll_uniform1);
    EXPECT_TRUE(simplifyExpr(v5, variables)->sameAs(v5));
    auto v6 = eq(unroll_uniform2, unroll_imm1);
    EXPECT_TRUE(simplifyExpr(v6, variables)->sameAs(v6));
    auto v7 = eq(unroll_imm2, unroll_gp1);
    EXPECT_TRUE(simplifyExpr(v7, variables)->sameAs(v7));
    auto v8 = eq(unroll_imm2, unroll_uniform1);
    EXPECT_TRUE(simplifyExpr(v8, variables)->sameAs(v8));
    auto v9 = eq(unroll_imm2, unroll_imm1);
    EXPECT_TRUE(simplifyExpr(v9, variables)->sameAs(v9));
  }
}

TEST_F(ExprSimplifierTest, MinMax) {
  auto simplify = [](Val* x, Val* assumption) {
    return simplifyExpr(x, {}, {assumption});
  };

  auto expr =
      "max( max( ceilDiv( T0.logical_size[0] , 128 ) * 4 , ceilDiv( T0.logical_size[0] , 128 ) ) , 4 )"_;
  EXPECT_TRUE(simplify(expr, "T0.logical_size[0] > 0"_)
                  ->sameAs("ceilDiv( T0.logical_size[0] , 128 ) * 4"_));
}

TEST_F(ExprSimplifierTest, PredicateDivToMul) {
  auto simplified =
      simplifyExpr("i1 / T0.logical_size[0] < i2"_, {}, {"i1 >= 0"_});
  auto expect = "i1 < ( i2 * T0.logical_size[0] )"_;

  EXPECT_TRUE(simplified->sameAs(expect));
}

TEST_F(ExprSimplifierTest, FactorizeGcd) {
  EXPECT_TRUE(simplifyExpr("gcd( i1 * i2 , i3 * i2 )"_)
                  ->sameAs("gcd( i1 , i3 ) * abs( i2 )"_));
  EXPECT_TRUE(simplifyExpr("gcd( i1 * i2 , i3 * i2 )"_, {}, {"i2 >= 0"_})
                  ->sameAs("gcd( i1 , i3 ) * i2"_));
  EXPECT_TRUE(simplifyExpr("gcd( i1 * i2 , i2 )"_)->sameAs("abs( i2 )"_));
  EXPECT_TRUE(
      simplifyExpr("gcd( i1 * i2 , i2 )"_, {}, {"i2 >= 0"_})->sameAs("i2"_));
}

} // namespace nvfuser
