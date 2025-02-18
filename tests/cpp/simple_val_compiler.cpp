// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ir/base_nodes.h>
#include <iter_visitor.h>
#include <ops/all_ops.h>
#include <tests/cpp/simple_val_compiler.h>
#include <tests/cpp/utils.h>

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
    return IrBuilder::create<Val>(true, DataType::Bool);
  } else if (token_str == "false") {
    return IrBuilder::create<Val>(false, DataType::Bool);
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
    NVF_THROW("Identifier with unknown type: ", token_str);
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
    return IrBuilder::create<Val>(i, DataType::Int);
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
    return IrBuilder::create<Val>(d, DataType::Double);
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
  } else if (name == "neg") {
    NVF_CHECK(args.size() == 1, "Invalid argument: ", toDelimitedString(args));
    return IrBuilder::negExpr(args.at(0));
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

} // namespace nvfuser
