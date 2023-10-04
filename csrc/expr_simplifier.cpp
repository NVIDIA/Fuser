// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <expr_simplifier.h>

#include <debug.h>
#include <device_lower/pass/magic_zero.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <options.h>
#include <utils.h>

#include <ranges.h>
#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace nvfuser {

namespace debug_print {

// In order to print transformations from expr simplifier, use:
//   NVFUSER_DUMP="expr_simplify"
// By default (i.e. no trigger specified), enabling the above debug dump option
// will trigger printing when the expression is transformed by at least one
// simplification pass (A simplification pass is a pass that is not a flatten
// or unflatten). If you want to print even if there is no simplification pass,
// applied, use the following:
//   NVFUSER_DUMP="expr_simplify(assoc_comm::flatten)"
// If you want to trigger printing only on a specified set of passes, put the
// pass names as arguments of `expr_simplify`, for example:
//   NVFUSER_DUMP="expr_simplify(eliminateTrivialComputation)"

constexpr const char* kFlattenName = "assoc_comm::flatten";
constexpr const char* kUnflattenName = "assoc_comm::unflatten";

struct Record {
  const char* name;
  Val* result;
};

class NoOpLogger {
 public:
  NoOpLogger(Val*) {}
  virtual ~NoOpLogger() = default;
  virtual void record(const char*, Val*) {}
};

class Logger : public NoOpLogger {
  bool shouldPrint() {
    if (current_val_->sameAs(init_val_)) {
      return false;
    }
    const auto& triggers_arg =
        getDebugDumpArguments(DebugDumpOption::ExprSimplification);
    std::vector<std::regex> triggers;
    triggers.reserve(triggers_arg.size());
    for (const auto& t : triggers_arg) {
      triggers.emplace_back(t);
    }
    if (triggers.empty()) {
      for (auto r : record_) {
        if (r.name != kFlattenName && r.name != kUnflattenName) {
          return true;
        }
      }
      return false;
    }
    for (auto r : record_) {
      auto match = [r](const std::regex& trigger) -> bool {
        return std::regex_match(r.name, trigger);
      };
      if (std::find_if(triggers.begin(), triggers.end(), match) !=
          triggers.end()) {
        return true;
      }
    }
    return false;
  }

 public:
  Logger(Val* value)
      : NoOpLogger(value), init_val_(value), current_val_(value) {}

  ~Logger() override {
    try {
      if (!shouldPrint()) {
        return;
      }

      auto str = [](Val* v) {
        std::stringstream ss;
        ss << ir_utils::varName(v) << " = " << v->toInlineString();
        return ss.str();
      };

      std::string header = "Simplifying expression:\n" + str(init_val_);
      debug() << header << std::endl;
      for (auto r : record_) {
        debug() << r.name << ":\n" << str(r.result) << std::endl;
      }
      debug() << std::string(std::min<size_t>(header.size(), 80), '=')
              << std::endl;
    } catch (...) {
      // clang-tidy don't want this function to throw, but this is just a
      // debugging helper, I don't really care if it has throw or not.
    }
  }

  void record(const char* name, Val* value) override {
    if (value->sameAs(current_val_)) {
      return;
    } else {
      record_.emplace_back(Record{name, value});
      current_val_ = value;
    }
  }

 private:
  std::vector<Record> record_;
  Val* init_val_;
  Val* current_val_;
};

std::unique_ptr<debug_print::NoOpLogger> createLogger(Val* value) {
  if (isDebugDumpEnabled(DebugDumpOption::ExprSimplification)) {
    return std::make_unique<Logger>(value);
  } else {
    return std::make_unique<NoOpLogger>(value);
  }
}

} // namespace debug_print

namespace assoc_comm {
Val* flatten(Val* value);
} // namespace assoc_comm

namespace {

// An ordered mapping of variable -> VarInfo
class Context {
 public:
  Context() = default;

  Context(
      const std::list<VarInfo>& variables,
      std::vector<Val*> assumptions,
      bool preserve_error)
      : preserve_error_(preserve_error) {
    var_order_.reserve(variables.size());
    var_set_.reserve(variables.size());
    less_than_.reserve(assumptions.size());
    less_equal_.reserve(assumptions.size());
    for (const auto& info : variables) {
      auto var = info.variable;
      if (info.is_unrolled_loop_index) {
        unrolled_loop_index_.insert(var);
      }
      var_order_.emplace_back(var);
      var_set_.emplace(var);
    }
    // decompose a && b in assumptions as a and b
    const auto& axioms = FusionGuard::getCurFusion()->axioms();
    assumptions.insert(assumptions.end(), axioms.begin(), axioms.end());
    while (!assumptions.empty()) {
      auto back = assumptions.back();
      assumptions.pop_back();
      auto bop = dynamic_cast<BinaryOp*>(back->definition());
      if (bop == nullptr ||
          bop->getBinaryOpType() != BinaryOpType::LogicalAnd) {
        assume(back);
      } else {
        assumptions.push_back(bop->lhs());
        assumptions.push_back(bop->rhs());
      }
    }
  }

  const std::vector<Val*>& variableOrder() const {
    return var_order_;
  }

  const std::unordered_set<Val*>& variableSet() const {
    return var_set_;
  }

  bool preserveError() const {
    return preserve_error_;
  }

  bool isUnrolledLoopIndex(Val* x) const {
    return unrolled_loop_index_.count(x) > 0;
  }

  const std::vector<std::pair<Val*, Val*>>& getKnownLessThan() const {
    return less_than_;
  }

  const std::vector<std::pair<Val*, Val*>>& getKnownLessEqual() const {
    return less_equal_;
  }

 private:
  void assume(Val* a) {
    auto def = a->definition();
    if (auto bop = dynamic_cast<BinaryOp*>(def)) {
      switch (bop->getBinaryOpType()) {
        case BinaryOpType::LT:
          less_than_.emplace_back(
              assoc_comm::flatten(bop->lhs()), assoc_comm::flatten(bop->rhs()));
          break;
        case BinaryOpType::LE:
          less_equal_.emplace_back(
              assoc_comm::flatten(bop->lhs()), assoc_comm::flatten(bop->rhs()));
          break;
        case BinaryOpType::GT:
          less_than_.emplace_back(
              assoc_comm::flatten(bop->rhs()), assoc_comm::flatten(bop->lhs()));
          break;
        case BinaryOpType::GE:
          less_equal_.emplace_back(
              assoc_comm::flatten(bop->rhs()), assoc_comm::flatten(bop->lhs()));
          break;
        default:
          NVF_ERROR(false, "Unknown operator type ", bop->getBinaryOpType());
      }
    }
  }

 private:
  bool preserve_error_ = false;
  std::vector<Val*> var_order_;
  std::unordered_set<Val*> var_set_;
  std::unordered_set<Val*> unrolled_loop_index_;
  std::vector<std::pair<Val*, Val*>> less_than_;
  std::vector<std::pair<Val*, Val*>> less_equal_;
};

bool hasSimilarType(DataType t1, DataType t2) {
  if (t1 == t2) {
    return true;
  }
  if (isIntegralOrPointerType(t1) && isIntegralOrPointerType(t2)) {
    return true;
  }
  if (isFloatingPointType(t1) && isFloatingPointType(t2)) {
    return true;
  }
  if (isComplexType(t1) && isComplexType(t2)) {
    return true;
  }
  return false;
}

// If `value` is a constant scalar, then evaluate the value of that constant and
// return the evaluated value. Otherwise, returns `value` itself.
Val* foldConstants(Val* value) {
  if (value->isConst()) {
    return value;
  }
  if (value->isConstScalar()) {
    if (value->isIntegralScalar()) {
      return IrBuilder::create<Val>(
          value->evaluateInt(), *value->getDataType());
    }
    if (value->isFloatingPointScalar()) {
      return IrBuilder::create<Val>(
          value->evaluateDouble(), *value->getDataType());
    }
    if (value->isABool()) {
      return IrBuilder::create<Val>(
          value->evaluateBool(), *value->getDataType());
    }
    // TODO: support complex double
  }
  return value;
}

// Get the set of variables that `value` depends on. Items in `variables` are
// considered variables, and items not in `variables` are considered constant.
// For example, if value = a + b + c + d + 3, and `variables` is {a, b, e, f},
// then this function returns {a, b}. All tensors are considered variables.
std::unordered_set<Val*> getSubexprDependency(
    Val* value,
    const std::unordered_set<Val*>& variables) {
  if (value->isA<kir::TensorIndex>()) {
    return {value};
  }
  if (variables.count(value) > 0) {
    return {value};
  }
  auto def = value->definition();
  if (def == nullptr) {
    return {};
  }
  std::unordered_set<Val*> result;
  for (auto i : def->inputs()) {
    auto deps = getSubexprDependency(i, variables);
    result.insert(deps.begin(), deps.end());
  }
  return result;
}

// Apply `rule` to `value`, if `rule` returns a new `Val*` to replace `value`,
// then return that new `Val*`, otherwise recursively goes down to its inputs.
Val* recurseDown(Val* value, std::function<Val*(Val*)> rule) {
  if (value->isOneOf<TensorView, kir::TensorIndex>()) {
    return value;
  }
  auto transformed = rule(value);
  if (transformed != value) {
    return transformed;
  }
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }

  bool changed = false;
  std::vector<Val*> new_inputs;
  new_inputs.reserve(def->inputs().size());
  for (auto v : def->inputs()) {
    new_inputs.emplace_back(recurseDown(v, rule));
    if (new_inputs.back() != v) {
      changed = true;
    }
  }

  if (!changed) {
    return value;
  }

  Val* output = IrBuilder::create<Val>(*value->getDataType());
  auto create_fn = def->newObjectFunc();
  create_fn(
      def->container(), std::move(new_inputs), {output}, def->attributes());
  return output;
}

inline RegisterType promoteRegisterType(RegisterType t1, RegisterType t2) {
  if (t1 == RegisterType::Unknown) {
    return t2;
  }
  if (t2 == RegisterType::Unknown) {
    return t1;
  }
  if (t1 == RegisterType::GeneralPurpose ||
      t2 == RegisterType::GeneralPurpose) {
    return RegisterType::GeneralPurpose;
  }
  if (t1 == RegisterType::Uniform || t2 == RegisterType::Uniform) {
    return RegisterType::Uniform;
  }
  return RegisterType::Immediate;
}

RegisterType getRegisterType(Val* value, const Context& context) {
  NVF_ERROR(value != nullptr);
  if (auto ns = dynamic_cast<NamedScalar*>(value)) {
    if (ns->getParallelIndex() == ParallelType::TIDx ||
        ns->getParallelIndex() == ParallelType::TIDy ||
        ns->getParallelIndex() == ParallelType::TIDz) {
      return RegisterType::GeneralPurpose;
    }
  }
  if (value->isConstScalar()) {
    return RegisterType::Immediate;
  }
  if (context.isUnrolledLoopIndex(value)) {
    return RegisterType::Immediate;
  }
  if (auto def = value->definition()) {
    RegisterType result = RegisterType::Unknown;
    for (auto inp : def->inputs()) {
      auto inp_rtype = getRegisterType(inp, context);
      result = promoteRegisterType(result, inp_rtype);
    }
    return result;
  }
  return RegisterType::Uniform;
}

bool hasUnrolledLoopIndex(Val* value, const Context& context) {
  if (context.isUnrolledLoopIndex(value)) {
    return true;
  }
  auto def = value->definition();
  if (def == nullptr) {
    return false;
  }
  for (auto inp : def->inputs()) {
    if (hasUnrolledLoopIndex(inp, context)) {
      return true;
    }
  }
  return false;
}

inline DataType inferDtypes(const std::vector<Val*>& vals) {
  auto dtype = *vals.at(0)->getDataType();
  for (auto v : vals) {
    dtype = promoteType(dtype, *v->getDataType());
  }
  return dtype;
}

} // namespace

RegisterType getRegisterType(Val* value) {
  return getRegisterType(value, {});
}

namespace assoc_comm {

// Note: [Reordering associative and commutative operators]
//
// For binary operators that is both associative and commutative, we can freely
// change the order of operands and add/remove parenthesis without changing the
// result. For example, + is both associative and commutative, so we have:
// a + b + c := (a + b) + c = a + (b + c) = (b + a) + c = b + (a + c) = ...
// For these operators, the most convenient way for handling them is to flatten
// them. For example, for the above a + b + c, all we need to know is we are
// adding these three variables together. We don't really care whether we are
// adding a and b first, or adding a and c first, or whether we are adding a to
// c or adding c to a. `FlattenedAssocCommOp` is the class that represents this
// flattened perspective.
//
// The reordering of associative and commutative operators is mostly useful for
// index hoisting. For example, if I have a loop structure and index:
//   FOR i1
//     FOR i2
//       FOR i3
//         index = ((i3 + i2) + i1) + 256
// There is no hoisting opportunity for this index in this loop structure.
// However, if I transform the index into index = ((256 + i1) + i2) + i3,
// then I can hoist the index as
//   FOR i1
//     i4 = (256 + i1)
//     FOR i2
//       i5 = i4 + i2
//       FOR i3
//         index = i5 + i3
// This minimizes the total number of computations.

inline bool isAssociativeAndCommutative(BinaryOpType type) {
  // gcd is associative and commutative, see:
  // https://en.wikipedia.org/wiki/Greatest_common_divisor#Properties
  return type == BinaryOpType::Add || type == BinaryOpType::Mul ||
      type == BinaryOpType::LogicalAnd || type == BinaryOpType::LogicalOr ||
      type == BinaryOpType::BitwiseAnd || type == BinaryOpType::BitwiseOr ||
      type == BinaryOpType::BitwiseXor || type == BinaryOpType::Max ||
      type == BinaryOpType::Min || type == BinaryOpType::Gcd;
}

// No-op term `e` is a special number that, for all x:
//   x (op) e = e (op) x = op(x)
// Above, we are slightly abusing terms, because op is a binary operator, in
// principle, op(x) is not well defined. However, because this is only used to
// eliminate trivial computation, for the sake of simplicity, we can consider
// gcd(x) = abs(x) and op(x) = x for all other ops.
// Note that the concept of "no-op term" here is very similar to the concept of
// identity in mathematics, whose definition is
//   x (op) e = e (op) x = x
inline bool isNoOpTerm(Val* v, BinaryOpType type) {
  if (v->isConstScalar()) {
    v = foldConstants(v);
  }
  if (!v->isConst()) {
    return false;
  }
  switch (type) {
    case BinaryOpType::Add:
      return v->isZero();
    case BinaryOpType::Mul:
      return v->isOne();
    case BinaryOpType::LogicalAnd:
      return v->getBool() == true;
    case BinaryOpType::LogicalOr:
      return v->getBool() == false;
    case BinaryOpType::BitwiseAnd:
      return v->getInt() == -1;
    case BinaryOpType::BitwiseOr:
    case BinaryOpType::BitwiseXor:
      return v->getInt() == 0;
    case BinaryOpType::Gcd:
      return v->isZeroInt();
    default:
      return false;
  }
}

// Identity `b` is a special number that, for all x:
// x (op) b = b (op) x = b
inline bool isBlackhole(Val* v, BinaryOpType type) {
  if (v->isConstScalar()) {
    v = foldConstants(v);
  }
  if (!v->isConst()) {
    return false;
  }
  switch (type) {
    case BinaryOpType::Mul:
      return v->getInt() == 0;
    case BinaryOpType::LogicalAnd:
      return v->getBool() == false;
    case BinaryOpType::LogicalOr:
      return v->getBool() == true;
    case BinaryOpType::BitwiseAnd:
      return v->getInt() == 0;
    case BinaryOpType::BitwiseOr:
      return v->getInt() == -1;
    case BinaryOpType::Gcd:
      return v->isOneInt();
    default:
      return false;
  }
}

// If `op` is the inverse of an associative and commutative operator, for
// example:
// - a - b -> a + (-b)
// - a / b -> a * (1 / b) if b is real or complex number
// then return the corresponding associative and commutative operator and the
// inverse operator. Otherwise, return nullopt.
inline std::optional<std::pair<BinaryOpType, UnaryOpType>>
asInverseOfAssocCommOp(BinaryOpType op, DataType rhs_dtype) {
  if (op == BinaryOpType::Sub) {
    return std::make_pair(BinaryOpType::Add, UnaryOpType::Neg);
  }
  if (op == BinaryOpType::Div) {
    if (isFloatingPointType(rhs_dtype) || isComplexType(rhs_dtype)) {
      return std::make_pair(BinaryOpType::Mul, UnaryOpType::Reciprocal);
    }
  }
  return std::nullopt;
}

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
      std::vector<Val*> terms)
      : Expr(passkey) {
    NVF_CHECK(
        isAssociativeAndCommutative(op),
        "Can only flatten associative and commutative ops");
    addDataAttribute(op);
    addOutput(out);
    for (auto v : terms) {
      NVF_CHECK(
          hasSimilarType(dtype(), *v->getDataType()),
          "Input types should be similar, but got: ",
          dtype(),
          ", and ",
          *v->getDataType());
      addInput(v);
    }
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  const char* getOpString() const override {
    switch (getOpType()) {
      case BinaryOpType::Add:
        return "FlattenedAdd";
      case BinaryOpType::Mul:
        return "FlattenedMul";
      case BinaryOpType::LogicalAnd:
        return "FlattenedLogicalAnd";
      case BinaryOpType::LogicalOr:
        return "FlattenedLogicalOr";
      case BinaryOpType::BitwiseAnd:
        return "FlattenedBitwiseAnd";
      case BinaryOpType::BitwiseOr:
        return "FlattenedBitwiseOr";
      case BinaryOpType::BitwiseXor:
        return "FlattenedBitwiseXor";
      case BinaryOpType::Max:
        return "FlattenedMax";
      case BinaryOpType::Min:
        return "FlattenedMin";
      default:
        NVF_ERROR(false, "Unknown operator type ", getOpType());
    }
  }

  // FlattenedAssocCommOp is unordered, so we should have
  // FlattenedAdd(a, b)->sameAs(FlattenedAdd(b, a))
  bool sameAs(const Statement* other) const override {
    if (this == other) {
      return true;
    }
    if (!other->isA<FlattenedAssocCommOp>()) {
      return false;
    }
    auto other_fop = other->as<FlattenedAssocCommOp>();
    if (getOpType() != other_fop->getOpType()) {
      return false;
    }
    // check if we can establish a 1:1 mapping between inputs() and
    // other_fop->inputs()
    std::list<Val*> other_inputs(
        other_fop->inputs().begin(), other_fop->inputs().end());
    for (const auto inp : inputs()) {
      auto it =
          std::find_if(other_inputs.begin(), other_inputs.end(), [inp](Val* v) {
            return v->sameAs(inp);
          });
      if (it == other_inputs.end()) {
        return false;
      }
      other_inputs.erase(it);
    }
    return other_inputs.empty();
  }

  std::string toString(int indent_size = 0) const override {
    std::stringstream ss;
    indent(ss, indent_size) << getOpString() << "(";
    bool needs_comma = false;
    for (auto v : inputs()) {
      if (needs_comma) {
        ss << ", ";
      }
      ss << v->toString();
      needs_comma = true;
    }
    ss << ")\n";
    return ss.str();
  }

  std::string toInlineString(int = 0) const override {
    std::stringstream ss;
    ss << getOpString() << "(";
    bool needs_comma = false;
    for (auto v : inputs()) {
      if (needs_comma) {
        ss << ", ";
      }
      ss << v->toInlineString();
      needs_comma = true;
    }
    ss << ")";
    return ss.str();
  }

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
  std::vector<Val*> sortedInputs(const Context& context) {
    std::vector<Val*> sorted_inputs(inputs().begin(), inputs().end());
    std::unordered_map<Val*, std::unordered_set<Val*>> dependency;
    dependency.reserve(sorted_inputs.size());
    for (auto v : sorted_inputs) {
      dependency[v] = getSubexprDependency(v, context.variableSet());
    }
    auto compare = [&](Val* v1, Val* v2) {
      // Find all variables in context that v1 and v2 depends on. The input (v1
      // or v2) that exclusively has the right most variable in context.order()
      // will be to the right of the other input.
      bool v1_is_left_of_v2 = false;
      auto deps1 = dependency.at(v1);
      auto deps2 = dependency.at(v2);
      auto hasTensorIndex = [](const auto& deps) {
        return std::any_of(deps.begin(), deps.end(), [](auto val) {
          return val->template isA<kir::TensorIndex>();
        });
      };
      if (hasTensorIndex(deps2)) {
        return true;
      }
      if (hasTensorIndex(deps1)) {
        return false;
      }
      for (auto v : context.variableOrder()) {
        if (deps1.count(v) > 0 && deps2.count(v) == 0) {
          v1_is_left_of_v2 = false;
        } else if (deps2.count(v) > 0 && deps1.count(v) == 0) {
          v1_is_left_of_v2 = true;
        }
      }
      return v1_is_left_of_v2;
    };
    std::sort(sorted_inputs.begin(), sorted_inputs.end(), compare);
    return sorted_inputs;
  }

  bool isTrivial() const {
    return inputs().size() == 1;
  }

  std::vector<PolymorphicValue> evaluate(
      const ExpressionEvaluator& ee,
      const std::vector<PolymorphicValue>& inputs) const override {
    using namespace PolymorphicValue_functions;
    std::vector<PolymorphicValue> inputs_ = inputs;
    PolymorphicValue result;
    result = inputs_.back();
    inputs_.pop_back();
    switch (getOpType()) {
      case BinaryOpType::Add:
        for (const auto& i : inputs_) {
          result += i;
        }
        break;
      case BinaryOpType::Mul:
        for (const auto& i : inputs_) {
          result *= i;
        }
        break;
      case BinaryOpType::LogicalAnd:
        for (const auto& i : inputs_) {
          result = result && i;
        }
        break;
      case BinaryOpType::LogicalOr:
        for (const auto& i : inputs_) {
          result = result || i;
        }
        break;
      case BinaryOpType::BitwiseAnd:
        for (const auto& i : inputs_) {
          result = result & i;
        }
        break;
      case BinaryOpType::BitwiseOr:
        for (const auto& i : inputs_) {
          result = result | i;
        }
        break;
      case BinaryOpType::BitwiseXor:
        for (const auto& i : inputs_) {
          result = result ^ i;
        }
        break;
      case BinaryOpType::Min:
        for (const auto& i : inputs_) {
          result = min(result, i);
        }
        break;
      case BinaryOpType::Max:
        for (const auto& i : inputs_) {
          result = max(result, i);
        }
        break;
      default:
        NVF_ERROR(
            "Unexpected operator type encountered"
            "in PolymorphicValue::evaluate: ",
            getOpType());
    }
    return {result};
  }
};

NVFUSER_DEFINE_CLONE_AND_CREATE(FlattenedAssocCommOp)

using FOp = assoc_comm::FlattenedAssocCommOp;

inline Val* maybeFlattenedOpOf(BinaryOpType bop, std::vector<Val*> inputs) {
  std::vector<Val*> nontrivial_inputs;
  std::vector<Val*> trivial_inputs;
  for (auto inp : inputs) {
    if (!assoc_comm::isNoOpTerm(inp, bop)) {
      nontrivial_inputs.emplace_back(inp);
    } else {
      trivial_inputs.emplace_back(inp);
    }
  }
  if (nontrivial_inputs.empty()) {
    return trivial_inputs.at(0);
  }
  if (nontrivial_inputs.size() == 1) {
    if (bop == BinaryOpType::Gcd) {
      return IrBuilder::absExpr(nontrivial_inputs.at(0));
    }
    return nontrivial_inputs.at(0);
  }
  auto result = IrBuilder::create<Val>(inferDtypes(nontrivial_inputs));
  IrBuilder::create<FOp>(bop, result, std::move(nontrivial_inputs));
  return result;
}

// Recursively convert expressions like AddOp(AddOp(a, b), AddOp(c, d)) into
// FlattenedAdd(a, b, c, d). This function recursively transforms the entire
// expression, so divOp(AddOp(AddOp(a, b), AddOp(c, d)), addOp(e, f)) will
// become divOp(FlattenAdd(a, b, c, d), FlattenAdd(e, f))
Val* flatten(Val* value);

Val* flattenRule(Val* value) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }
  value = foldConstants(value);
  if (value->isConst()) {
    return value;
  }

  NVF_ERROR(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  BinaryOpType op = BinaryOpType::Atan2; // Initialize with an arbitrary
                                         // non-associative non-commutative op
  bool changed = false;
  if (auto bop = dynamic_cast<BinaryOp*>(def)) {
    op = bop->getBinaryOpType();
    if (!hasSimilarType(bop->lhs()->dtype(), bop->rhs()->dtype())) {
      // Avoid flattening ops with different types because different dtypes
      // follow different simplification rules, so we don't want to mix them
      // together. For example, 2.0 + 3 + 4 will be easier to analyze if we
      // flatten it as 2.0 + FlattenAdd(3, 4) instead of FlattenAdd(2.0, 3, 4)
      return value;
    }
    if (isAssociativeAndCommutative(op)) {
      changed = true;
    } else if (auto inv = asInverseOfAssocCommOp(op, bop->rhs()->dtype())) {
      // a - b -> FlattenAdd(a, -b)
      // a / b -> FlattenMul(a, b^(-1))
      auto assoc_comm_op = inv->first;
      auto inv_op = inv->second;
      std::vector<Val*> lhs_terms;
      std::vector<Val*> rhs_terms;
      auto collect_terms = [&](Val* operand, std::vector<Val*>& terms) {
        if (auto fop = dynamic_cast<FOp*>(operand->definition());
            fop != nullptr && fop->getOpType() == assoc_comm_op) {
          terms = fop->inputs();
        } else {
          terms.emplace_back(operand);
        }
      };
      collect_terms(flatten(bop->lhs()), lhs_terms);
      collect_terms(flatten(bop->rhs()), rhs_terms);
      for (auto term : rhs_terms) {
        auto inv_term = IrBuilder::create<Val>(bop->rhs()->dtype());
        IrBuilder::create<UnaryOp>(inv_op, inv_term, term);
        lhs_terms.emplace_back(inv_term);
      }
      return maybeFlattenedOpOf(assoc_comm_op, std::move(lhs_terms));
    }
  } else if (auto fop = dynamic_cast<FlattenedAssocCommOp*>(def)) {
    op = fop->getOpType();
  }

  if (isAssociativeAndCommutative(op)) {
    // Handle associative-and-commutative op:
    // Convert binary ops into flattened op, reflatten already flattened
    // FlattenedAssocCommOp
    std::vector<Val*> inputs;

    auto append_or_merge_inputs = [&](Val* operand) {
      auto fop = dynamic_cast<FlattenedAssocCommOp*>(operand->definition());
      if (fop != nullptr && fop->getOpType() == op &&
          hasSimilarType(fop->dtype(), *value->getDataType())) {
        inputs.insert(inputs.end(), fop->inputs().begin(), fop->inputs().end());
        changed = true;
      } else {
        inputs.emplace_back(operand);
      }
    };

    for (auto inp : def->inputs()) {
      auto flattened = flatten(inp);
      if (flattened != inp) {
        changed = true;
      }
      append_or_merge_inputs(flattened);
    }

    if (!changed) {
      return value;
    }

    if (inputs.size() == 1) {
      return inputs.at(0);
    }

    auto output = IrBuilder::create<Val>(inferDtypes(inputs));
    IrBuilder::create<FlattenedAssocCommOp>(op, output, std::move(inputs));
    return output;
  }

  return value;
}

Val* flatten(Val* value) {
  return recurseDown(value, flattenRule);
}

// Recursively convert expressions like FlattenedAdd(a, b, c, d) into
// AddOp(AddOp(AddOp(a, b), c), d))
Val* unflatten(Val* value, const Context& context);

Val* unflattenRule(Val* value, const Context& context) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }

  NVF_ERROR(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  auto fop = dynamic_cast<FlattenedAssocCommOp*>(def);

  if (fop != nullptr) {
    // Handle flattened op:
    // Convert flattened op into original binary ops
    NVF_ERROR(fop->inputs().size() >= 2);
    auto sorted_inputs = fop->sortedInputs(context);
    // We need to recursively unflatten all inputs, because we might have
    // nested flattened expressions like
    // FlattenedAdd(a, b, FlattenedMul(c, d, e))
    Val* lhs = unflatten(sorted_inputs.at(0), context);
    int64_t next = 1;
    while (next < (int64_t)sorted_inputs.size()) {
      auto rhs = unflatten(sorted_inputs.at(next), context);
      if (fop->getOpType() == BinaryOpType::Add) {
        // Convert a + (-b) to a - b for better readibility on generated code
        auto uop = dynamic_cast<UnaryOp*>(rhs->definition());
        if (uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Neg) {
          lhs = IrBuilder::subExpr(lhs, uop->in());
        } else {
          lhs = IrBuilder::addExpr(lhs, rhs);
        }
      } else {
        auto output = IrBuilder::create<Val>(
            promoteType(*lhs->getDataType(), *rhs->getDataType()));
        IrBuilder::create<BinaryOp>(fop->getOpType(), output, lhs, rhs);
        lhs = output;
      }
      next++;
    }
    return lhs;
  }
  return value;
}

Val* unflatten(Val* value, const Context& context) {
  return recurseDown(
      value, [&context](Val* val) { return unflattenRule(val, context); });
}

} // namespace assoc_comm

using FOp = assoc_comm::FlattenedAssocCommOp;
using assoc_comm::maybeFlattenedOpOf;

namespace {

// If x and y are both Null, then return Null. If one of them is Null, then
// return the other one. Otherwise, return the promoted type of x and y.
inline DataType promoteTypeWithNull(DataType x, DataType y) {
  if (x == DataType::Null) {
    return y;
  }
  if (y == DataType::Null) {
    return x;
  }
  return promoteType(x, y);
}

// If x is nullptr, return DataType::Null. Otherwise, return x->dtype().
inline DataType dataTypeOrNull(Val* x) {
  return x == nullptr ? DataType::Null : x->dtype();
}

// If x is nullptr, return 1. Otherwise, return x->getInt().
inline int64_t getIntOrOne(Val* x) {
  return x == nullptr ? 1 : *x->getInt();
}

// If the data type is unknown, return nullptr. Otherwise, return a constant
// with the given value and data type.
template <typename T>
inline Val* getConstOrNullptr(T value, DataType dtype) {
  if (dtype == DataType::Null) {
    return nullptr;
  }
  return IrBuilder::create<Val>(value, dtype);
}

FOp* toFlattenedAdd(Expr* expr) {
  auto fop = dynamic_cast<FOp*>(expr);
  if (!fop) {
    return nullptr;
  }
  if (fop->getOpType() == BinaryOpType::Add) {
    return fop;
  }
  return nullptr;
}

bool isFlattenedAdd(Val* x) {
  return toFlattenedAdd(x->definition()) != nullptr;
}

FOp* toFlattenedMul(Expr* expr) {
  auto fop = dynamic_cast<FOp*>(expr);
  if (!fop) {
    return nullptr;
  }
  if (fop->getOpType() == BinaryOpType::Mul) {
    return fop;
  }
  return nullptr;
}

bool isFlattenedMul(Val* x) {
  return toFlattenedMul(x->definition()) != nullptr;
}

FOp* toFlattenedGcd(Expr* expr) {
  auto fop = dynamic_cast<FOp*>(expr);
  if (!fop) {
    return nullptr;
  }
  if (fop->getOpType() == BinaryOpType::Gcd) {
    return fop;
  }
  return nullptr;
}

bool isFlattenedGcd(Val* x) {
  return toFlattenedGcd(x->definition()) != nullptr;
}

BinaryOp* toDivModOp(Expr* expr) {
  auto bop = dynamic_cast<BinaryOp*>(expr);
  if (!bop) {
    return nullptr;
  }
  if (bop->getBinaryOpType() == BinaryOpType::Div ||
      bop->getBinaryOpType() == BinaryOpType::Mod) {
    // TODO: Add CeilDiv as well? Need mathematiclly prove its rules first
    return bop;
  }
  return nullptr;
}

// Classify terms of a FlattenedMul as (constant, symbolic), the constant may be
// nullptr if it is one. For example:
// a * 3 * b * 5 --> (15, {a, b})
// a * b --> (nullptr, {a, b})
// 1 * a * b --> (1, {a, b})
// 3 * 5 --> (15, {})
// If the given Val `x` is not a flattened mul, then return (nullptr, {x})
std::pair<Val*, std::list<Val*>> getConstAndSymbolicFactors(Val* x) {
  std::vector<Val*> factors;
  if (auto fop = toFlattenedMul(x->definition())) {
    factors = fop->inputs();
  } else {
    factors.emplace_back(x);
  }
  DataType const_dtype = DataType::Null;
  int64_t const_factor = 1;
  std::list<Val*> symbolic_factors;
  for (auto f : factors) {
    f = foldConstants(f);
    if (f->getInt().has_value()) {
      const_dtype = promoteTypeWithNull(const_dtype, f->dtype());
      const_factor *= *f->getInt();
    } else {
      symbolic_factors.emplace_back(f);
    }
  }
  // When const_factor is 1 and the const_dtype is unknown, return a nullptr
  // instead of guessing a dtype.
  return {getConstOrNullptr(const_factor, const_dtype), symbolic_factors};
}

// Given a constant factor and a list of symbolic factors, return a flattened
// mul expression. There are cases where the result dtype can not be inferred
// from the inputs, for example when const_factor is nullptr and
// symbolic_factors is empty, then the result dtype is unknown. In this case,
// return a constant one with the default dtype.
Val* productOfFactors(
    Val* const_factor,
    std::vector<Val*> symbolic_factors,
    DataType default_dtype) {
  if (const_factor == nullptr) {
    if (symbolic_factors.empty()) {
      return IrBuilder::create<Val>(1L, default_dtype);
    }
    return assoc_comm::maybeFlattenedOpOf(
        BinaryOpType::Mul, std::move(symbolic_factors));
  }
  if (*const_factor->getInt() != 1 || symbolic_factors.empty()) {
    symbolic_factors.emplace_back(const_factor);
  }
  return maybeFlattenedOpOf(BinaryOpType::Mul, std::move(symbolic_factors));
}

} // namespace

namespace sym_algebra {

// Common utilities for symbolic algebra.

// Rewrite x in the form x = x1 * x2 * x3 * ...
Val* factorize(Val* x);

// Given that x = x1 * x2 * x3 * x4 * ..., y = x2 * x4 * ..., where x is a
// multiple of y, evaluate x/y as x/y = x1 * x3 * ... Both x and y should have
// already been factorized in the form of products, otherwise, this function
// might not be able to get the desired result. Returns nullptr if not
// divisible. Note that this function does symbolic term cancellation, it does
// not require y to be non-zero.
Val* divideFactorized(Val* x, Val* y) {
  auto x_factors = getConstAndSymbolicFactors(x);
  auto y_factors = getConstAndSymbolicFactors(y);

  int64_t xx = getIntOrOne(x_factors.first);
  int64_t yy = getIntOrOne(y_factors.first);
  auto xdtype = dataTypeOrNull(x_factors.first);
  auto ydtype = dataTypeOrNull(y_factors.first);
  auto const_type = promoteTypeWithNull(xdtype, ydtype);

  if (xx % yy != 0) {
    // not divisible
    return nullptr;
  }
  int64_t quoient_const_factor = xx / yy;

  std::vector<Val*> quotient_symbolic_factors;

  for (auto yf : y_factors.second) {
    auto it = std::find_if(
        x_factors.second.begin(), x_factors.second.end(), [yf](Val* v) {
          return v->sameAs(yf);
        });
    if (it == x_factors.second.end()) {
      // not divisible
      return nullptr;
    }
    x_factors.second.erase(it);
  }
  quotient_symbolic_factors.insert(
      quotient_symbolic_factors.end(),
      x_factors.second.begin(),
      x_factors.second.end());
  return productOfFactors(
      getConstOrNullptr(quoient_const_factor, const_type),
      std::move(quotient_symbolic_factors),
      promoteType(x->dtype(), y->dtype()));
}

// Symbolic gcd, for example: greatestCommonDivisor({6*a*b, 9*b*c}) -> 3*b
Val* greatestCommonDivisor(const std::vector<Val*>& inputs) {
  // The gcd of the constant part. Because gcd(0, a) = gcd(a, 0) = a, it is
  // great to use 0 as initial value because it does not need special handling.
  int64_t common_const_factor = 0;
  // The gcd of the symbolic part. nullptr serve as 0, empty vector serve as 1.
  std::unique_ptr<std::vector<Val*>> common_symbolic_factors = nullptr;

  DataType const_factor_dtype = DataType::Null;
  DataType common_dtype = DataType::Null;

  for (auto inp : inputs) {
    common_dtype = promoteTypeWithNull(common_dtype, inp->dtype());
    auto factors = getConstAndSymbolicFactors(inp);
    const_factor_dtype =
        promoteTypeWithNull(const_factor_dtype, dataTypeOrNull(factors.first));
    common_const_factor =
        std::gcd(common_const_factor, getIntOrOne(factors.first));
    std::vector<Val*> new_common_symbolic_factors;
    if (common_symbolic_factors == nullptr) {
      // gcd(0, x) -> x
      new_common_symbolic_factors.insert(
          new_common_symbolic_factors.end(),
          factors.second.begin(),
          factors.second.end());
    } else {
      for (auto f : (*common_symbolic_factors)) {
        auto it = std::find_if(
            factors.second.begin(), factors.second.end(), [f](Val* v) {
              return v->sameAs(f);
            });
        if (it != factors.second.end()) {
          new_common_symbolic_factors.emplace_back(f);
          factors.second.erase(it);
        }
      }
    }
    common_symbolic_factors = std::make_unique<std::vector<Val*>>(
        std::move(new_common_symbolic_factors));
  }

  NVF_ERROR(common_const_factor != 0);
  NVF_ERROR(common_symbolic_factors != nullptr);
  return productOfFactors(
      getConstOrNullptr(common_const_factor, const_factor_dtype),
      std::move(*common_symbolic_factors),
      common_dtype);
}

namespace {

Val* factorizeFlattenedMul(Val* x) {
  auto fop = toFlattenedMul(x->definition());
  NVF_ERROR(fop != nullptr);
  // Recursively factorize all its inputs, and combine their terms
  int64_t const_factor = 1;
  std::vector<Val*> symbolic_factors;
  DataType const_factor_dtype = DataType::Null;
  bool changed = false;
  for (auto inp : fop->inputs()) {
    auto factorized_inp = factorize(inp);
    auto factors = getConstAndSymbolicFactors(factorized_inp);
    const_factor_dtype =
        promoteTypeWithNull(const_factor_dtype, dataTypeOrNull(factors.first));
    const_factor *= getIntOrOne(factors.first);
    symbolic_factors.insert(
        symbolic_factors.end(), factors.second.begin(), factors.second.end());
    if (factors.second != std::list<Val*>{inp}) {
      changed = true;
    }
  }

  if (!changed) {
    return x;
  }
  return productOfFactors(
      getConstOrNullptr(const_factor, const_factor_dtype),
      std::move(symbolic_factors),
      x->dtype());
}

// Does the following factorization:
// - a * x + b * x -> (a + b) * x
// - gcd(a * x, b * x) -> gcd(a, b) * abs(x)
//
// Note (proof of the second rule): according to
// https://en.wikipedia.org/wiki/Greatest_common_divisor#Properties, We have:
// For m > 0, m*gcd(a,b) = gcd(m*a, m*b). If we further define gcd(0, 0) = 0,
// the above equation holds also for m = 0. Also, observe that gcd(a, b) is a
// even function w.r.t. both a and b, we have:
// - gcd(a * x, b * x) -> gcd(a, b) * abs(x)
Val* factorizeFlattenedAddOrGcd(Val* x) {
  // Warning: This implementation can only factorize out common divisor. It can
  // not factorize FlattenedAdd(x * x, 2 * x, 1) as FlattenedMul(x + 1, x + 1).
  // But I believe factorizing out common divisor is sufficient for index
  // simplification.
  auto fadd = toFlattenedAdd(x->definition());
  auto fgcd = toFlattenedGcd(x->definition());
  bool is_gcd = (fgcd != nullptr);
  auto fop = is_gcd ? fgcd : fadd;
  NVF_ERROR(fop != nullptr);
  std::vector<Val*> factorized_inputs;
  for (auto inp : fop->inputs()) {
    factorized_inputs.emplace_back(factorize(inp));
  }
  // Find common factors
  auto common_factor = greatestCommonDivisor(factorized_inputs);
  if (common_factor->isOne()) {
    return x;
  }
  // divide by common factors
  std::vector<Val*> quotient_inputs;
  quotient_inputs.reserve(factorized_inputs.size());
  for (auto inp : factorized_inputs) {
    auto quotient = divideFactorized(inp, common_factor);
    NVF_ERROR(quotient != nullptr);
    quotient_inputs.emplace_back(quotient);
  }
  auto quotient = IrBuilder::create<Val>(inferDtypes(quotient_inputs));
  IrBuilder::create<FOp>(
      fop->getOpType(), quotient, std::move(quotient_inputs));
  if (is_gcd) {
    common_factor = IrBuilder::absExpr(common_factor);
  }
  auto product = IrBuilder::create<Val>(
      promoteType(*quotient->getDataType(), *common_factor->getDataType()));
  IrBuilder::create<FOp>(
      BinaryOpType::Mul, product, std::vector<Val*>{quotient, common_factor});
  // Quotient might contain nested FlattenedOp, for example, if we have:
  //   FlattenedAdd(a * FlattenedAdd(b, c), a * FlattenedAdd(d, e))
  // then the common_factor will be a, and the quotient will be:
  //   FlattenedAdd(FlattenedAdd(b, c), FlattenedAdd(d, e))
  // So we need to reflatten to get rid of this nested FlattenedOp.
  return assoc_comm::flatten(product);
}

// Rule O
Val* factorizeMod(Val* x) {
  auto bop = dynamic_cast<BinaryOp*>(x->definition());
  NVF_ERROR(bop->getBinaryOpType() == BinaryOpType::Mod);
  auto flhs = factorize(bop->lhs());
  auto frhs = factorize(bop->rhs());
  auto gcd = greatestCommonDivisor({flhs, frhs});
  if (gcd->isOne()) {
    return x;
  }
  auto qlhs = divideFactorized(flhs, gcd);
  auto qrhs = divideFactorized(frhs, gcd);
  auto mod = IrBuilder::create<Val>(*x->getDataType());
  IrBuilder::create<BinaryOp>(BinaryOpType::Mod, mod, qlhs, qrhs);
  auto product = IrBuilder::create<Val>(*x->getDataType());
  IrBuilder::create<FOp>(
      BinaryOpType::Mul, product, std::vector<Val*>{mod, gcd});
  return product;
}

} // namespace

// Rewrite x in the form x = x1 * x2 * x3 * ...
Val* factorize(Val* x) {
  if (x->isConstScalar()) {
    return foldConstants(x);
  }
  if (isProtectedWithMagicZero(x)) {
    return x;
  }
  if (isFlattenedMul(x)) {
    return factorizeFlattenedMul(x);
  }
  if (isFlattenedAdd(x) || isFlattenedGcd(x)) {
    return factorizeFlattenedAddOrGcd(x);
  }
  if (auto bop = dynamic_cast<BinaryOp*>(x->definition())) {
    if (bop->getBinaryOpType() == BinaryOpType::Mod) {
      // Rule O
      return factorizeMod(x);
    }
  }
  return x;
}

} // namespace sym_algebra

namespace {
bool isValidDenominator(Val* denominator, const Context& context);
}

namespace prove {

// Prove properties of values. Note that functions in this namespace return
// boolean values. If true is returned, then this means the property is
// successfully proved. If false is returned, then this means that we are unable
// to prove the property. Be warned that a false being returned does not
// necessarily means the opposite of the property holds. For example, if you get
// isNonZero(x) == false, you shouldn't think that isZero(x) == true, because
// isNonZero(x) == false could mean:
// - x is actually non-zero, but we are not smart enough to prove it
// - x can be either zero or non-zero, it is just a symbolic number that depends
// - x is zero

bool lessThan(Val* x, Val* y, const Context& context);
bool lessEqual(Val* x, Val* y, const Context& context);

bool greaterThan(Val* x, Val* y, const Context& context) {
  return lessThan(y, x, context);
}

bool greaterEqual(Val* x, Val* y, const Context& context) {
  return lessEqual(y, x, context);
}

bool isPositive(Val* value, const Context& context) {
  auto zero = IrBuilder::create<Val>(0L, *value->getDataType());
  return greaterThan(value, zero, context);
}

bool isNonNegative(Val* value, const Context& context) {
  auto zero = IrBuilder::create<Val>(0L, *value->getDataType());
  return greaterEqual(value, zero, context);
}

bool isNonNegativeHelper(Val* value, const Context& context) {
  if (ir_utils::isTensorSize(value) || ir_utils::isTensorStride(value)) {
    return true;
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    auto op = fop->getOpType();
    if (op == BinaryOpType::Add || op == BinaryOpType::Mul) {
      for (auto inp : fop->inputs()) {
        if (!isNonNegative(inp, context)) {
          return false;
        }
      }
      return true;
    }
  } else if (auto bop = dynamic_cast<BinaryOp*>(value->definition())) {
    auto op = bop->getBinaryOpType();
    if (op == BinaryOpType::Mod || op == BinaryOpType::Div ||
        op == BinaryOpType::CeilDiv) {
      return isNonNegative(bop->lhs(), context) &&
          isValidDenominator(bop->rhs(), context) &&
          isNonNegative(bop->rhs(), context);
    }
  }
  for (const auto& [a, b] : context.getKnownLessThan()) {
    if (a->isZero() && b->sameAs(value)) {
      return true;
    }
  }
  for (const auto& [a, b] : context.getKnownLessEqual()) {
    if (a->isZero() && b->sameAs(value)) {
      return true;
    }
  }
  return false;
}

bool isPositiveHelper(Val* value, const Context& context) {
  if (ir_utils::isTensorSize(value)) {
    return true;
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    auto op = fop->getOpType();
    if (op == BinaryOpType::Add) {
      bool has_positive = false;
      for (auto inp : fop->inputs()) {
        if (isPositive(inp, context)) {
          has_positive = true;
        } else if (!isNonNegative(inp, context)) {
          return false;
        }
      }
      return has_positive;
    } else if (op == BinaryOpType::Mul) {
      for (auto inp : fop->inputs()) {
        if (!isPositive(inp, context)) {
          return false;
        }
      }
      return true;
    }
  } else if (auto bop = dynamic_cast<BinaryOp*>(value->definition())) {
    auto op = bop->getBinaryOpType();
    if (op == BinaryOpType::CeilDiv) {
      return isPositive(bop->lhs(), context) &&
          isValidDenominator(bop->rhs(), context) &&
          isNonNegative(bop->rhs(), context);
    }
  }
  for (const auto& [a, b] : context.getKnownLessThan()) {
    if (a->isZero() && b->sameAs(value)) {
      return true;
    }
  }
  return false;
}

bool isNonZero(Val* value, const Context& context) {
  value = foldConstants(value);
  if (value->getInt().has_value() && *value->getInt() != 0) {
    return true;
  }
  if (value->getDouble().has_value() && *value->getDouble() != 0.0) {
    return true;
  }
  if (isPositive(value, context)) {
    return true;
  }
  if (auto fop = toFlattenedMul(value->definition())) {
    for (auto inp : fop->inputs()) {
      if (!isNonZero(inp, context)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// Tries to prove that x is a multiple of y, that is, there exist an integer `k`
// such that x = k*y
bool isMultipleOf(Val* x, Val* y) {
  auto lhs = sym_algebra::factorize(x);
  auto rhs = sym_algebra::factorize(y);
  return sym_algebra::divideFactorized(lhs, rhs) != nullptr;
}

bool hasCompatibleSign(Val* x, Val* y, const Context& context) {
  return isNonNegative(x, context) && isNonNegative(y, context);
}

bool lessThan(Val* x, Val* y, const Context& context) {
  x = foldConstants(x);
  y = foldConstants(y);
  if (x->getInt().has_value() && y->getInt().has_value()) {
    return *x->getInt() < *y->getInt();
  }
  if (x->getDouble().has_value() && y->getDouble().has_value()) {
    return *x->getDouble() < *y->getDouble();
  }
  x = maybeUnwrapMagicZero(x);
  y = maybeUnwrapMagicZero(y);
  if (x->isZero() && isPositiveHelper(y, context)) {
    return true;
  }
  // i1 % i2 < i2
  if (auto bop = dynamic_cast<BinaryOp*>(x->definition());
      bop != nullptr && bop->getBinaryOpType() == BinaryOpType::Mod) {
    auto denominator = bop->rhs();
    if (denominator->sameAs(y) && isValidDenominator(denominator, context) &&
        isNonNegative(y, context)) {
      return true;
    }
  }
  // x <= a & a < b & b <= y  -->  x < y
  for (const auto& [a, b] : context.getKnownLessThan()) {
    if (lessEqual(x, a, context) && lessEqual(b, y, context)) {
      return true;
    }
  }
  return false;
}

bool lessEqual(Val* x, Val* y, const Context& context) {
  x = foldConstants(x);
  y = foldConstants(y);
  if (x->getInt().has_value() && y->getInt().has_value()) {
    return *x->getInt() <= *y->getInt();
  }
  if (x->getDouble().has_value() && y->getDouble().has_value()) {
    return *x->getDouble() <= *y->getDouble();
  }
  x = maybeUnwrapMagicZero(x);
  y = maybeUnwrapMagicZero(y);
  // x == y -> x <= y
  if (x->sameAs(y)) {
    return true;
  }
  if (x->isZero() && isNonNegativeHelper(y, context)) {
    return true;
  }
  for (const auto& [a, b] : context.getKnownLessThan()) {
    // x < y  -->  x <= y
    if (a->sameAs(x) && b->sameAs(y)) {
      return true;
    }
  }
  for (const auto& [a, b] : context.getKnownLessEqual()) {
    if (a->sameAs(x) && b->sameAs(y)) {
      return true;
    }
  }
  for (const auto& [a, b] : context.getKnownLessThan()) {
    // x < b & b <= y  -->  x <= y
    if (a->sameAs(x) && lessEqual(b, y, context)) {
      return true;
    }
  }
  for (const auto& [a, b] : context.getKnownLessEqual()) {
    // x <= b & b <= y  -->  x <= y
    if (a->sameAs(x) && lessEqual(b, y, context)) {
      return true;
    }
  }
  // if i is an integer, i > 0, then i >= 1
  if (x->isOneInt() && y->isIntegralScalar()) {
    if (isPositiveHelper(y, context)) {
      return true;
    }
  }
  // if a >= 0, b >= 1, then a <= a * b
  if (auto fop = toFlattenedMul(y->definition())) {
    std::vector<Val*> remaining_inputs;
    remaining_inputs.reserve(fop->inputs().size());
    bool found = false;
    for (auto inp : fop->inputs()) {
      if (!found && x->sameAs(inp)) {
        found = true;
        continue;
      }
      remaining_inputs.emplace_back(inp);
    }
    if (found) {
      auto zero = IrBuilder::create<Val>(0L, *x->getDataType());
      if (lessEqual(zero, x, context)) {
        auto remaining =
            maybeFlattenedOpOf(BinaryOpType::Mul, std::move(remaining_inputs));
        auto one = IrBuilder::create<Val>(1L, *remaining->getDataType());
        if (lessEqual(one, remaining, context)) {
          return true;
        }
      }
    }
  }
  // if a <= 0, b >= 1, then a * b <= a
  if (auto fop = toFlattenedMul(x->definition())) {
    std::vector<Val*> remaining_inputs;
    remaining_inputs.reserve(fop->inputs().size());
    bool found = false;
    for (auto inp : fop->inputs()) {
      if (!found && y->sameAs(inp)) {
        found = true;
        continue;
      }
      remaining_inputs.emplace_back(inp);
    }
    if (found) {
      auto zero = IrBuilder::create<Val>(0L, *y->getDataType());
      if (lessEqual(y, zero, context)) {
        auto remaining =
            maybeFlattenedOpOf(BinaryOpType::Mul, std::move(remaining_inputs));
        auto one = IrBuilder::create<Val>(1L, *remaining->getDataType());
        if (lessEqual(one, remaining, context)) {
          return true;
        }
      }
    }
  }
  return false;
}

} // namespace prove

namespace {

// If we want to do simplifications like (a * b) / b -> a, depending on whether
// we want to preserve error, the behavior could be different. If we don't care
// about preserving error, we can just go ahead and do the simplification.
// However, if we do want the division-by-zero error to be preserved, then we
// can only do the simplification if we can prove b != 0. This function tells us
// if the value of b is safe to do such optimization. Instead of completely
// ignoring error case, we do a bit extra: if b is proved to be zero, then we
// are sure that there will be an error, then we don't remove the error. That
// is, if we don't know if there will be an error, we procceed assuming no
// error. If we are sure there will be an error, then don't procceed.
bool isValidDenominator(Val* denominator, const Context& context) {
  // We ask two questions:
  // Q1: Can we prove the denominato is nonzero?
  // Q2: Can we prove the denominato is zero?
  // Depending on the outcome of these two question, we have different behavior:
  // | Q1 | Q2 | Div-by-zero err? | Behavior                                |
  // |----|----|------------------|-----------------------------------------|
  // | T  | T  | Not possible     | Not possible                            |
  // | T  | F  | Guaranteed no    | allow proceeding                        |
  // | F  | T  | Guaranteed yes   | disallow proceeding                     |
  // | F  | F  | don't know       | allow if not required to preserve error |
  bool proved_nonzero = prove::isNonZero(denominator, context);
  if (proved_nonzero) {
    return true;
  }
  if (context.preserveError()) {
    return false;
  }
  bool proved_zero = foldConstants(denominator)->isZero();
  if (proved_zero) {
    return false;
  }
  if (isDebugDumpEnabled(DebugDumpOption::ExprSimplification)) {
    TORCH_WARN_ONCE(
        "Assuming ",
        denominator->toInlineString(),
        " to be non-zero does not perserve division-by-zero error");
  }
  return true;
}

} // namespace

namespace rules {

// Sometimes we might have different pointers for the same variable, for
// example, we might have multiple different pointers for threadIdx.x. This pass
// convert all these different pointers into the pointer specified in the
// context. This canonicalization is important, because some passes use set of
// pointers to find variables, without canonicalization, this finding will fail.
Val* canonicalizeVariables(Val* value, const Context& context) {
  for (auto v : context.variableOrder()) {
    if (v->sameAs(value)) {
      return v;
    }
  }
  return value;
}

// Do simplifications like:
// 1 * a -> a
// 0 * a -> 0
// true && a -> a
// false && a -> false
// true || a -> true
// false || a -> a
// a % 1 -> 0
// a / 1 -> a
// x - x -> 0
// b && b -> b (assuming no side effect on b)
// b || b -> b (assuming no side effect on b)
// -(-x) -> x
// !(!x) -> x
// ...
Val* eliminateTrivialComputation(Val* value, const Context& context) {
  auto folded = foldConstants(value);
  if (folded != value) {
    return folded;
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    auto op = fop->getOpType();
    if (fop->isTrivial()) {
      // For gcd, we have gcd(a) -> |a|. For everything else, we have op(a) -> a
      if (op == BinaryOpType::Gcd) {
        return IrBuilder::absExpr(fop->input(0));
      }
      return fop->input(0);
    }
    if (op == BinaryOpType::Add) { // a + (-a) -> 0
      std::vector<std::tuple<Val*, Val*, size_t>> inv_inputs;
      for (size_t idx : irange(fop->inputs().size())) {
        auto inp = fop->input(idx);
        auto def = inp->definition();
        if (auto inv = dynamic_cast<UnaryOp*>(def)) {
          if (inv->getUnaryOpType() == UnaryOpType::Neg) {
            inv_inputs.emplace_back(inp, inv->in(), idx);
          }
        }
      }
      std::unordered_set<size_t> remove;
      for (auto [orig, inv, idx] : inv_inputs) {
        for (size_t idx2 : irange(fop->inputs().size())) {
          auto inp = fop->input(idx2);
          if (remove.count(idx) || remove.count(idx2)) {
            continue;
          }
          if (inp->sameAs(inv)) {
            remove.emplace(idx);
            remove.emplace(idx2);
          }
        }
      }
      if (!remove.empty()) {
        std::vector<Val*> new_inputs;
        for (size_t idx : irange(fop->inputs().size())) {
          if (!remove.count(idx)) {
            new_inputs.emplace_back(fop->input(idx));
          }
        }
        if (new_inputs.empty()) {
          return value->fusion()->zeroVal(value->dtype());
        } else {
          return maybeFlattenedOpOf(op, std::move(new_inputs));
        }
      }
    }
    { // 0 * a -> 0, 1 * a -> a
      std::vector<Val*> new_inputs;
      Val* const_term = nullptr;
      bool changed = false;
      for (auto inp : fop->inputs()) {
        if (inp->isConstScalar()) {
          if (const_term == nullptr) {
            const_term = inp;
          } else {
            auto out = IrBuilder::create<Val>(
                promoteType(*const_term->getDataType(), *inp->getDataType()));
            IrBuilder::create<BinaryOp>(op, out, const_term, inp);
            const_term = out;
            changed = true;
          }
        } else {
          new_inputs.emplace_back(inp);
        }
      }
      if (const_term != nullptr) {
        auto folded_const = foldConstants(const_term);
        if (folded_const != const_term) {
          changed = true;
          const_term = folded_const;
        }
        if (assoc_comm::isBlackhole(const_term, op)) {
          // 0 * a -> 0
          return const_term;
        }
        if (assoc_comm::isNoOpTerm(const_term, op)) {
          // 1 * a -> a
          const_term = nullptr;
          changed = true;
        }
      }
      if (changed) {
        if (const_term != nullptr) {
          new_inputs.emplace_back(const_term);
        }
        return maybeFlattenedOpOf(op, std::move(new_inputs));
      }
    }
    { // b && b -> b, b || b -> b, max(i, i) -> i, min(i, i) -> i
      if (op == BinaryOpType::LogicalAnd || op == BinaryOpType::LogicalOr ||
          op == BinaryOpType::BitwiseAnd || op == BinaryOpType::BitwiseOr ||
          op == BinaryOpType::Max || op == BinaryOpType::Min) {
        std::vector<Val*> dedup_input;
        for (auto v : fop->inputs()) {
          bool found_dup = false;
          for (auto v2 : dedup_input) {
            if (v->sameAs(v2)) {
              found_dup = true;
              break;
            }
          }
          if (!found_dup) {
            dedup_input.emplace_back(v);
          }
        }
        if (dedup_input.size() < fop->inputs().size()) {
          return maybeFlattenedOpOf(op, std::move(dedup_input));
        }
      }
    }
    { // max(a, b) -> a if a >= b, min(a, b) -> b if a >= b
      if (op == BinaryOpType::Max || op == BinaryOpType::Min) {
        std::vector<Val*> simplified_input;
        for (auto v : fop->inputs()) {
          bool found_redundant = false;
          for (auto& v2 : simplified_input) {
            if ((op == BinaryOpType::Max && prove::lessEqual(v, v2, context)) ||
                (op == BinaryOpType::Min && prove::lessEqual(v2, v, context))) {
              found_redundant = true;
              break;
            } else if (
                (op == BinaryOpType::Max && prove::lessEqual(v2, v, context)) ||
                (op == BinaryOpType::Min && prove::lessEqual(v, v2, context))) {
              found_redundant = true;
              v2 = v;
              break;
            }
          }
          if (!found_redundant) {
            simplified_input.emplace_back(v);
          }
        }
        if (simplified_input.size() < fop->inputs().size()) {
          return maybeFlattenedOpOf(op, std::move(simplified_input));
        }
      }
    }
  } else if (auto bop = dynamic_cast<BinaryOp*>(value->definition())) {
    auto lhs = foldConstants(bop->lhs());
    auto rhs = foldConstants(bop->rhs());
    if (bop->getBinaryOpType() == BinaryOpType::Mod) {
      // a % 1 -> 0
      if (rhs->isOneInt()) {
        return IrBuilder::create<Val>(0L, *value->getDataType());
      }
    } else if (
        bop->getBinaryOpType() == BinaryOpType::Div ||
        bop->getBinaryOpType() == BinaryOpType::CeilDiv) {
      // a / 1 -> a
      // 0 / a -> 0
      if (rhs->isOne() ||
          (isValidDenominator(rhs, context) && lhs->getInt() == 0)) {
        return lhs;
      }
    }
  } else if (auto uop = dynamic_cast<UnaryOp*>(value->definition())) {
    auto optype = uop->getUnaryOpType();
    if (optype == UnaryOpType::Neg || optype == UnaryOpType::LogicalNot ||
        optype == UnaryOpType::BitwiseNot) {
      // -(-x) -> x, !(!x) -> x, ~(~x) -> x
      auto uop_in = dynamic_cast<UnaryOp*>(uop->in()->definition());
      if (uop_in != nullptr) {
        auto optype_in = uop_in->getUnaryOpType();
        if (optype == optype_in) {
          return uop_in->in();
        }
      }
    } else if (optype == UnaryOpType::Abs) {
      // abs(x) -> x if x >= 0
      auto abs_in = uop->in();
      if (prove::isNonNegative(abs_in, context)) {
        return abs_in;
      }
    }
  } else if (auto top = dynamic_cast<TernaryOp*>(value->definition())) {
    // where(true, x, y) -> x, where(false, x, y) -> y
    auto optype = top->getTernaryOpType();
    if (optype == TernaryOpType::Where) {
      auto cond = foldConstants(top->in1());
      if (cond->isTrue()) {
        return top->in2();
      } else if (cond->isFalse()) {
        return top->in3();
      }
    }
  }
  return value;
}

// If x can be proved to be non-negative, then replace x >= 0 as true, replace
// x < 0 as false
// If x can be proved to be positive, then replace x >= 0 and x > 0 as true,
// replace x <= 0 and x < 0 as false
// If x can be proved to be nonzero, then replace x != 0 as true, replace x == 0
// as false
// if x->sameAs(y), then replace x == y as true, replace x != y as false
Val* eliminateTrivialPredicate(Val* value, const Context& context) {
  if (!value->isABool()) {
    return value;
  }

  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }

  auto op = bop->getBinaryOpType();
  auto lhs = bop->lhs();
  auto rhs = bop->rhs();
  if (op == BinaryOpType::Eq) {
    if (lhs->sameAs(rhs)) {
      return value->fusion()->trueVal();
    } else if (
        (lhs->isZero() && prove::isNonZero(rhs, context)) ||
        (rhs->isZero() && prove::isNonZero(lhs, context))) {
      return value->fusion()->falseVal();
    }
  } else if (op == BinaryOpType::NE) {
    if ((lhs->isZero() && prove::isNonZero(rhs, context)) ||
        (rhs->isZero() && prove::isNonZero(lhs, context))) {
      return value->fusion()->trueVal();
    } else if (lhs->sameAs(rhs)) {
      return value->fusion()->falseVal();
    }
  } else if (op == BinaryOpType::GE) {
    if (prove::greaterEqual(lhs, rhs, context)) {
      return value->fusion()->trueVal();
    } else if (prove::lessThan(lhs, rhs, context)) {
      return value->fusion()->falseVal();
    }
  } else if (op == BinaryOpType::GT) {
    if (prove::greaterThan(lhs, rhs, context)) {
      return value->fusion()->trueVal();
    } else if (prove::lessEqual(lhs, rhs, context)) {
      return value->fusion()->falseVal();
    }
  } else if (op == BinaryOpType::LE) {
    if (prove::lessEqual(lhs, rhs, context)) {
      return value->fusion()->trueVal();
    } else if (prove::greaterThan(lhs, rhs, context)) {
      return value->fusion()->falseVal();
    }
  } else if (op == BinaryOpType::LT) {
    if (prove::lessThan(lhs, rhs, context)) {
      return value->fusion()->trueVal();
    } else if (prove::greaterEqual(lhs, rhs, context)) {
      return value->fusion()->falseVal();
    }
  }
  return value;
}

// Apply rule 1 in [Simplification of boolean predicates] to convert
// i / d < D into i < d * D
Val* convertDivToMulInPredicate(Val* value, const Context& context) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }
  if (bop->getBinaryOpType() != BinaryOpType::LT) {
    return value;
  }
  auto lhs = bop->lhs();
  auto rhs = bop->rhs();
  bop = dynamic_cast<BinaryOp*>(lhs->definition());
  if (!bop) {
    return value;
  }
  if (bop->getBinaryOpType() != BinaryOpType::Div) {
    return value;
  }
  auto numerator = bop->lhs();
  auto denominator = bop->rhs();
  if (isValidDenominator(denominator, context) &&
      prove::isNonNegative(numerator, context) &&
      prove::isNonNegative(denominator, context)) {
    auto new_rhs = maybeFlattenedOpOf(BinaryOpType::Mul, {rhs, denominator});
    auto out = IrBuilder::create<Val>(DataType::Bool);
    IrBuilder::create<BinaryOp>(BinaryOpType::LT, out, numerator, new_rhs);
    return out;
  }
  return value;
}

// Apply rule L to replace x % y with 0 if x can be proved to be a multiple of y
// Also, according to rule M, if x can be factorized as x = k * y, then x / y
// can be simplified as x / y = (k * y) / y = k * (y / y) = k
Val* simplifyDivisibleDivMod(Val* value, const Context& context) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }
  if (!isValidDenominator(bop->rhs(), context)) {
    return value;
  }
  if (bop->getBinaryOpType() == BinaryOpType::Mod) {
    if (prove::isMultipleOf(bop->lhs(), bop->rhs())) {
      return IrBuilder::create<Val>(0L, *value->getDataType());
    }
  } else if (bop->getBinaryOpType() == BinaryOpType::Div) {
    auto lhs = sym_algebra::factorize(bop->lhs());
    auto rhs = sym_algebra::factorize(bop->rhs());
    auto quotient = sym_algebra::divideFactorized(lhs, rhs);
    if (quotient != nullptr) {
      return quotient;
    }
  }
  return value;
}

// Simplify div and mod by canceling common terms
//
// For div, use rule N): a / (b * c) = (a / b) / c to simplify division:
// Let y = gcd(x, y) * y' and x = gcd(x, y) * x'
// then we can simplify x / y as:
// x / y = x / (gcd(x, y) * y') = (x / gcd(x, y)) / y' = x' / y'
//
// For mod, use rule
// O) If d divides a and b, then a % b = ((a / d) % (b / d)) * d
// Let y = gcd(x, y) * y' and x = gcd(x, y) * x'
// If gcd is nonzero, then we can simplify x % y as:
// x' % y' * gcd(x, y)
Val* cancelDivMod(Val* value, const Context& context) {
  auto divmod = toDivModOp(value->definition());
  if (!divmod) {
    return value;
  }
  auto op = divmod->getBinaryOpType();
  if (op != BinaryOpType::Div && op != BinaryOpType::Mod) {
    return value;
  }
  auto lhs = sym_algebra::factorize(divmod->lhs());
  auto rhs = sym_algebra::factorize(divmod->rhs());
  auto gcd = sym_algebra::greatestCommonDivisor({lhs, rhs});
  if (gcd->isOne() || !isValidDenominator(gcd, context)) {
    return value;
  }
  auto numerator = sym_algebra::divideFactorized(lhs, gcd);
  auto denominator = sym_algebra::divideFactorized(rhs, gcd);
  if (op == BinaryOpType::Div) {
    return IrBuilder::divExpr(numerator, denominator);
  } else {
    NVF_ERROR(op == BinaryOpType::Mod);
    return assoc_comm::flatten(
        IrBuilder::mulExpr(IrBuilder::modExpr(numerator, denominator), gcd));
  }
}

// Use the following rule to simplify div and mod:
// J) Distributivity of % over +:
//    If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// Q) If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
//    (a + b) / c = a/c + b/c
// In this pass we distribute div and mod for a special case:
// If compatible_sign(a, b), and a is a multiple of c, then:
//  (a + b) / c = a/c + b/c
//  (a + b) % c = b % c
Val* distributeDivisibleDivMod(Val* value, const Context& context) {
  auto divmod = toDivModOp(value->definition());
  if (!divmod) {
    return value;
  }
  auto lhs = divmod->lhs();
  auto rhs = divmod->rhs();
  if (!lhs->isIntegralScalar() || !rhs->isIntegralScalar() ||
      !isValidDenominator(rhs, context)) {
    return value;
  }
  auto fop = toFlattenedAdd(lhs->definition());
  if (!fop) {
    return value;
  }
  for (auto i : irange(fop->inputs().size())) {
    Val* divisible_term = fop->input(i);
    if (!prove::isMultipleOf(divisible_term, rhs)) {
      continue;
    }
    std::vector<Val*> other_terms;
    other_terms.reserve(fop->inputs().size() - 1);
    for (auto j : irange(fop->inputs().size())) {
      if (j == i) {
        continue;
      }
      other_terms.emplace_back(fop->input(j));
    }
    Val* sum_of_other_terms =
        maybeFlattenedOpOf(BinaryOpType::Add, std::move(other_terms));
    if (prove::hasCompatibleSign(divisible_term, sum_of_other_terms, context)) {
      std::vector<Val*> new_inputs;
      auto term1 = IrBuilder::create<Val>(
          promoteType(*divisible_term->getDataType(), *rhs->getDataType()));
      IrBuilder::create<BinaryOp>(
          divmod->getBinaryOpType(), term1, divisible_term, rhs);
      new_inputs.emplace_back(simplifyDivisibleDivMod(term1, context));
      new_inputs.emplace_back(IrBuilder::create<Val>(promoteType(
          *sum_of_other_terms->getDataType(), *rhs->getDataType())));
      IrBuilder::create<BinaryOp>(
          divmod->getBinaryOpType(), new_inputs[1], sum_of_other_terms, rhs);
      auto output = IrBuilder::create<Val>(inferDtypes(new_inputs));
      IrBuilder::create<FOp>(BinaryOpType::Add, output, std::move(new_inputs));
      return output;
    }
  }
  return value;
}

// Use the following rule to simplify div and mod:
// J) Distributivity of % over +:
//    If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// Q) If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
//    (a + b) / c = a/c + b/c
// In this pass we distribute div and mod for a special case:
// Let g = gcd(a, c). If compatible_sign(a, b), and -|g| < b < |g|, then:
//  (a + b) / c = a/c
//  (a + b) % c = a % c + b
// Proof:
//  Because -|g| < b < |g|, and |g| <= |c|, we know -|c| < b < |c|, according to
//  Theorem 6.5, we have b % c = b.
//  According to rule O, we have a % c = ((a/g) % (c/g)) * g.
//  So, a % c + b % c = ((a/g) % (c/g)) * g + b
//  Because -|c/g| < (a/g) % (c/g) < |c/g|, for integers, we have
//  -|c/g| + 1 <= (a/g) % (c/g) <= |c/g| - 1
//  So, -(|c/g| - 1) * |g| <= a % c <= (|c/g| - 1) * |g|
//  Therefore, we have
//  -(|c/g| - 1) * |g| - |g| < a % c + b % c < (|c/g| - 1) * |g| + |g|
// That is: -|c| < a % c + b % c < |c|
// Therefore:
// (a + b) % c = (a % c + b % c) % c = a % c + b % c = a % c + b
// (a + b) / c = a/c + b/c = a / c
Val* distributeGcdRemainderDivMod(Val* value, const Context& context) {
  auto divmod = toDivModOp(value->definition());
  if (!divmod) {
    return value;
  }
  auto lhs = divmod->lhs();
  auto rhs = divmod->rhs();
  if (!lhs->isIntegralScalar() || !rhs->isIntegralScalar() ||
      !isValidDenominator(rhs, context)) {
    return value;
  }
  auto fop = toFlattenedAdd(lhs->definition());
  if (!fop) {
    return value;
  }
  auto fdivisor = sym_algebra::factorize(rhs);
  // We should partition fop->inputs() into two parts. And we need to try all
  // possible partitions to check if we can find a pattern matching the
  // condition. However, trying all partition can be slow. So we take advantage
  // of our knowledge of our prover:
  // 1. If we can not prove -|c| < x < |c|, then it is unlikely for us to be
  //    able to prove -|gcd(c,...)| < x < |gcd(c,...)|
  // 2. If we can not prove -|c| < x < |c|, then it is unlikely for us to be
  //    able to prove -|c| < x + y < |c|
  // Note that the above observation is just an approximation, it is not
  // guaranteed. So if we use any of the above assumptions, we might lose
  // simplifying opportunities. But in practice, this should be fine.
  //
  // Taking advantage of the above two points, we come up with the following
  // algorithm:
  // We begin by xs = fop->inputs() and g = c; We dynamically update xs and g:
  // if we find an x in xs that we can not prove -|g| < x < |g|, we remove this
  // x from xs and update g = gcd(g, x). We repeat this process until converge.
  // After convergence, we will try all combinations of xs to find matching
  // pattern that satisfy -|gcd(...)| < x1 + x2 + ... < |gcd(...)|.

  // Step 1: find xs
  std::deque<Val*> xs;
  for (auto x : fop->inputs()) {
    xs.push_back(sym_algebra::factorize(x));
  }
  std::vector<Val*> other_terms;
  bool changed = true;
  Val* front = nullptr;
  Val* g = fdivisor;
  Val* abs_g = IrBuilder::absExpr(g);
  Val* neg_abs_g = IrBuilder::negExpr(abs_g);
  while (front != nullptr || changed) {
    if (front == nullptr) {
      changed = false;
      xs.push_back(nullptr);
    } else {
      bool in_range = prove::lessThan(front, abs_g, context) &&
          prove::greaterThan(front, neg_abs_g, context);
      if (in_range) {
        xs.push_back(front);
      } else {
        changed = true;
        other_terms.push_back(front);
        g = sym_algebra::greatestCommonDivisor({g, front});
        abs_g = IrBuilder::absExpr(g);
        neg_abs_g = IrBuilder::negExpr(abs_g);
      }
    }
    front = xs.front();
    xs.pop_front();
  }
  if (xs.empty()) {
    return value;
  }

  // Step 2: try all combinations of xs
  // The number of combinations of xs is exponential to the size of xs, so we
  // limit the maximum size of xs. If xs is larger, then we make another
  // assumptions that split xs into batches and only consider 1 batch each time.
  // In practice, I expect the size of xs to be just one or two elements, so I
  // don't think this approximation makes any trouble. I add this max size just
  // to make the algorithm more robust.
  constexpr size_t max_size_xs = 10;
  static_assert(
      max_size_xs < sizeof(size_t) * 8,
      "max_size_xs can not be larger than the number of bits in size_t, "
      "otherwise num_combinations will overflow");
  for (size_t batch = 0; batch * max_size_xs < xs.size(); batch++) {
    size_t start = batch * max_size_xs;
    size_t end = std::min(xs.size(), (batch + 1) * max_size_xs);
    size_t batch_size = end - start;
    size_t num_combinations = 1 << batch_size;
    for (size_t combo_id = 1; combo_id < num_combinations; combo_id++) {
      std::vector<Val*> combo_xs;
      std::vector<Val*> combo_other = other_terms;
      // dispatch xs into combo_xs and combo_other
      for (size_t i = 0; i < xs.size(); i++) {
        if (i < start || i >= end) {
          combo_other.push_back(xs[i]);
        } else {
          size_t batch_i = i - start;
          if (((combo_id >> batch_i) & size_t(1)) == 1) {
            combo_xs.push_back(xs[i]);
          } else {
            combo_other.push_back(xs[i]);
          }
        }
      }
      // compute sum of combo_xs and sum of combo_other
      Val* sum_xs = maybeFlattenedOpOf(BinaryOpType::Add, std::move(combo_xs));
      Val* sum_other =
          maybeFlattenedOpOf(BinaryOpType::Add, std::move(combo_other));
      // prove -|g| < sum_xs < |g|
      bool allowed_to_simplify =
          prove::hasCompatibleSign(sum_other, sum_xs, context);
      if (allowed_to_simplify) {
        Val* g = sym_algebra::greatestCommonDivisor({fdivisor, sum_other});
        Val* abs_g = IrBuilder::absExpr(g);
        Val* neg_abs_g = IrBuilder::negExpr(abs_g);
        allowed_to_simplify = prove::lessThan(sum_xs, abs_g, context) &&
            prove::greaterThan(sum_xs, neg_abs_g, context);
      }
      if (!allowed_to_simplify) {
        continue;
      }
      // allowed to simplify
      switch (divmod->getBinaryOpType()) {
        case BinaryOpType::Div: {
          // (a + b) / c = a / c
          auto result = IrBuilder::create<Val>(
              promoteType(*sum_other->getDataType(), *fdivisor->getDataType()));
          IrBuilder::create<BinaryOp>(
              BinaryOpType::Div, result, sum_other, fdivisor);
          return result;
        }
        case BinaryOpType::Mod: {
          // (a + b) % c = a % c + b
          auto term1 = IrBuilder::create<Val>(
              promoteType(*sum_other->getDataType(), *fdivisor->getDataType()));
          IrBuilder::create<BinaryOp>(
              BinaryOpType::Mod, term1, sum_other, fdivisor);
          auto result = IrBuilder::create<Val>(
              promoteType(*term1->getDataType(), *sum_xs->getDataType()));
          IrBuilder::create<FOp>(
              BinaryOpType::Add, result, std::vector<Val*>{term1, sum_xs});
          return assoc_comm::flatten(result);
        }
        default:
          NVF_ERROR(false);
      }
    }
  }
  return value;
}

// a * (b + c) -> a * b + a * c
Val* distributeMul(Val* value, const Context& context) {
  auto fop = toFlattenedMul(value->definition());
  if (!fop) {
    return value;
  }
  Val* flattened_add = nullptr;
  std::vector<Val*> other_terms;
  for (auto inp : fop->inputs()) {
    if (flattened_add == nullptr && isFlattenedAdd(inp)) {
      flattened_add = inp;
    } else {
      other_terms.emplace_back(inp);
    }
  }
  if (flattened_add == nullptr) {
    return value;
  }
  auto fadd_op = toFlattenedAdd(flattened_add->definition());
  std::vector<Val*> add_terms;
  for (auto inp : fadd_op->inputs()) {
    std::vector<Val*> inputs = other_terms;
    inputs.emplace_back(inp);
    add_terms.emplace_back(IrBuilder::create<Val>(inferDtypes(inputs)));
    IrBuilder::create<FOp>(
        BinaryOpType::Mul, add_terms.back(), std::move(inputs));
  }
  auto output = IrBuilder::create<Val>(inferDtypes(add_terms));
  IrBuilder::create<FOp>(BinaryOpType::Add, output, std::move(add_terms));
  return output;
}

// This pass optimizes register usage of predicate evaluation, especially for
// unrolled for loop.
// For example, if I have an unrolled for loop like below:
//   base = ...
//   for i in 4
//     for j in 4
//       if base + i* stride0 + j*stride1 < T.size:
//         expr
// Then the above for loop uses 4 * 4 = 16 general purposed registers to
// evaluate the lhs of the predicate.
// However, if I transform the predicate as:
//   base = ...
//   lhs = base - T.size
//   for i in 4
//     for j in 4
//       if lhs < -i* stride0 - j*stride1:
//         expr
// Then it only takes 1 general purposed register for lhs, and the rhs is a
// compile time constant for nvRTC, so it can use the immediate fields of the
// instruction. This optimization results in a great save on registers.
// See also: https://github.com/csarofeen/pytorch/pull/1976
Val* reducePredicateRegisterUsage(Val* value, const Context& context) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!value->isABool() || bop == nullptr) {
    return value;
  }
  auto op_type = bop->getBinaryOpType();
  if (!isLogicalOp(op_type)) {
    return value;
  }
  auto ltype = *bop->lhs()->getDataType();
  auto rtype = *bop->rhs()->getDataType();
  if (!hasSimilarType(ltype, rtype)) {
    return value;
  }

  // Classify terms into unroll and other
  auto classify = [&context](const std::vector<Val*>& terms) {
    std::vector<Val*> unroll, other;
    RegisterType unroll_rtype = RegisterType::Unknown,
                 other_rtype = RegisterType::Unknown;
    for (auto inp : terms) {
      if (inp->isZero()) {
        continue;
      }
      auto rtype = getRegisterType(inp, context);
      if (hasUnrolledLoopIndex(inp, context)) {
        unroll.push_back(inp);
        unroll_rtype = promoteRegisterType(unroll_rtype, rtype);
      } else {
        other.push_back(inp);
        other_rtype = promoteRegisterType(other_rtype, rtype);
      }
    }
    return std::make_tuple(unroll, unroll_rtype, other, other_rtype);
  };

  auto [lhs_unroll, lhs_unroll_rtype, lhs_other, lhs_other_rtype] = classify(
      isFlattenedAdd(bop->lhs()) ? bop->lhs()->definition()->inputs()
                                 : std::vector<Val*>{bop->lhs()});
  auto [rhs_unroll, rhs_unroll_rtype, rhs_other, rhs_other_rtype] = classify(
      isFlattenedAdd(bop->rhs()) ? bop->rhs()->definition()->inputs()
                                 : std::vector<Val*>{bop->rhs()});
  auto unroll_rtype = promoteRegisterType(lhs_unroll_rtype, rhs_unroll_rtype);
  auto other_rtype = promoteRegisterType(lhs_other_rtype, rhs_other_rtype);

  // it makes sense to move:
  // unroll + other == other
  // unroll + other == unroll
  // unroll + other == 0
  // but not:
  // unroll == other
  // unroll == unroll
  // other == other
  // unroll == 0
  // other == 0
  bool can_move = (!lhs_unroll.empty() && !lhs_other.empty()) ||
      (!rhs_unroll.empty() && !rhs_other.empty());
  if (!can_move) {
    return value;
  }

  bool can_save_general_purpose_register =
      (other_rtype == RegisterType::GeneralPurpose &&
       (unroll_rtype == RegisterType::Uniform ||
        unroll_rtype == RegisterType::Immediate));
  bool can_save_uniform_register =
      (other_rtype == RegisterType::Uniform &&
       unroll_rtype == RegisterType::Immediate);
  bool can_save_register =
      (can_save_general_purpose_register || can_save_uniform_register);
  if (!can_save_register) {
    return value;
  }

  std::vector<Val*> new_lhs = std::move(lhs_other);
  for (auto v : rhs_other) {
    new_lhs.push_back(IrBuilder::negExpr(v));
  }

  std::vector<Val*> new_rhs = std::move(rhs_unroll);
  for (auto v : lhs_unroll) {
    new_rhs.push_back(IrBuilder::negExpr(v));
  }

  Val* lhs = nullptr;
  Val* rhs = nullptr;
  if (new_lhs.empty()) {
    lhs = IrBuilder::create<Val>(0L, ltype);
  } else {
    lhs = maybeFlattenedOpOf(BinaryOpType::Add, std::move(new_lhs));
  }
  if (new_rhs.empty()) {
    rhs = IrBuilder::create<Val>(0L, rtype);
  } else {
    rhs = maybeFlattenedOpOf(BinaryOpType::Add, std::move(new_rhs));
  }
  auto output = IrBuilder::create<Val>(DataType::Bool);
  IrBuilder::create<BinaryOp>(op_type, output, lhs, rhs);
  return output;
}

// Merge values using the fundamental division-with-remainder property:
// a = a / b * b + a % b
// This pass do merges of the following patterns:
// Pattern 1: a / b * b + a % b -> a
// Pattern 2: a / b * (b*c) + a % b * c -> a * c
Val* fundamentalDivisionWithRemainderProperty(
    Val* value,
    const Context& context) {
  auto fadd = toFlattenedAdd(value->definition());
  if (!fadd) {
    return value;
  }
  // Get all patterns like: (a op b) * c, for example, if I have
  // (a / b) * (e / f), the this funciton will return:
  // {(a, b, e/f), (e, f, a/b)}
  auto get_a_op_b_mul_c = [](BinaryOpType op, Val* x) {
    std::vector<std::tuple<Val*, Val*, Val*>> result;
    auto fmul = toFlattenedMul(x->definition());
    if (fmul == nullptr) {
      return result;
    }
    for (auto j : irange(fmul->inputs().size())) {
      auto vmul = fmul->input(j);
      if (!isIntegralType(*vmul->getDataType())) {
        continue;
      }
      auto bop = dynamic_cast<BinaryOp*>(vmul->definition());
      if (bop == nullptr) {
        continue;
      }
      if (bop->getBinaryOpType() == op) {
        auto a = bop->lhs();
        auto b = bop->rhs();
        std::vector<Val*> other_terms;
        for (auto k : irange(fmul->inputs().size())) {
          if (j == k) {
            continue;
          }
          other_terms.emplace_back(fmul->input(k));
        }
        Val* c = nullptr;
        if (other_terms.empty()) {
          continue;
        } else {
          c = maybeFlattenedOpOf(BinaryOpType::Mul, std::move(other_terms));
        }
        if (!isIntegralType(*a->getDataType()) ||
            !isIntegralType(*b->getDataType()) ||
            !isIntegralType(*c->getDataType())) {
          continue;
        }
        result.emplace_back(a, b, c);
      }
    }
    return result;
  };
  // Find a / b * b or a / b * (b*c)
  std::vector<std::tuple<size_t, Val*, Val*, Val*>> divmuls;
  for (auto i : irange(fadd->inputs().size())) {
    auto vadd = fadd->input(i);
    if (!isIntegralType(*vadd->getDataType())) {
      continue;
    }
    for (auto& [a, b, bc] : get_a_op_b_mul_c(BinaryOpType::Div, vadd)) {
      divmuls.emplace_back(i, a, b, bc);
    }
  }
  // Find a % b or a % b * c
  std::vector<std::tuple<size_t, Val*, Val*, Val*>> modmuls;
  for (auto i : irange(fadd->inputs().size())) {
    auto vadd = fadd->input(i);
    if (!isIntegralType(*vadd->getDataType())) {
      continue;
    }
    auto bop = dynamic_cast<BinaryOp*>(vadd->definition());
    if (bop != nullptr && bop->getBinaryOpType() == BinaryOpType::Mod) {
      if (!isIntegralType(*bop->lhs()->getDataType()) ||
          !isIntegralType(*bop->rhs()->getDataType())) {
        continue;
      }
      modmuls.emplace_back(
          i,
          bop->lhs(),
          bop->rhs(),
          IrBuilder::create<Val>(1L, *vadd->getDataType()));
    }
    for (auto& [a, b, c] : get_a_op_b_mul_c(BinaryOpType::Mod, vadd)) {
      modmuls.emplace_back(i, a, b, c);
    }
  }
  // Find matching divmul and modmul
  for (auto& [i, a1, b1, bc] : divmuls) {
    for (auto& [j, a2, b2, c] : modmuls) {
      if (i == j) {
        continue;
      }
      if (!a1->sameAs(a2) || !b1->sameAs(b2)) {
        continue;
      }
      if (!isValidDenominator(b1, context)) {
        continue;
      }
      auto factorized_b1c = sym_algebra::factorize(
          maybeFlattenedOpOf(BinaryOpType::Mul, {b1, c}));
      auto factorized_bc = sym_algebra::factorize(bc);
      auto quotient =
          sym_algebra::divideFactorized(factorized_bc, factorized_b1c);
      if (quotient != nullptr && quotient->isOne()) {
        // Found match!
        // Simplify [1] + [2] + ... + [i] + ... + [j] + ...
        // As: [1] + [2] + a * c ... + ...  + ...
        Val* ac = maybeFlattenedOpOf(BinaryOpType::Mul, {a1, c});
        std::vector<Val*> terms{ac};
        for (auto k : irange(fadd->inputs().size())) {
          if (k == i || k == j) {
            continue;
          }
          terms.emplace_back(fadd->input(k));
        }
        return maybeFlattenedOpOf(BinaryOpType::Add, terms);
      }
    }
  }
  return value;
}

// gcd(a * c, b * c) = |c| * gcd(a, b)
Val* factorizeGcd(Val* value, const Context& context) {
  if (isFlattenedGcd(value)) {
    return sym_algebra::factorize(value);
  }
  return value;
}

Val* cancelTermsInPredicate(Val* value, const Context& context) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }
  auto op = bop->getBinaryOpType();
  if (op != BinaryOpType::Eq && op != BinaryOpType::GE &&
      op != BinaryOpType::GT && op != BinaryOpType::LE &&
      op != BinaryOpType::LT && op != BinaryOpType::NE) {
    return value;
  }

  auto get_terms = [](Val* operand) -> std::vector<Val*> {
    if (auto flattened = toFlattenedAdd(operand->definition())) {
      return flattened->inputs();
    } else {
      return {operand};
    }
  };

  const auto& lhs_terms = get_terms(bop->lhs());
  const auto& rhs_terms = get_terms(bop->rhs());

  std::vector<bool> common_lhs_terms(lhs_terms.size(), false);
  std::vector<bool> common_rhs_terms(rhs_terms.size(), false);
  for (const auto lhs_i : irange(lhs_terms.size())) {
    for (const auto rhs_i : irange(rhs_terms.size())) {
      // Make sure no multiple LHS terms are removed for the same RHS term
      if (common_rhs_terms.at(rhs_i)) {
        continue;
      }
      if (lhs_terms[lhs_i]->sameAs(rhs_terms[rhs_i])) {
        common_lhs_terms.at(lhs_i) = true;
        common_rhs_terms.at(rhs_i) = true;
        break;
      }
    }
  }

  if (std::none_of(
          common_lhs_terms.begin(), common_lhs_terms.end(), [](auto flag) {
            return flag;
          })) {
    return value;
  }

  std::vector<Val*> new_lhs_terms;
  for (const auto i : irange(lhs_terms.size())) {
    if (!common_lhs_terms.at(i)) {
      new_lhs_terms.push_back(lhs_terms[i]);
    }
  }

  std::vector<Val*> new_rhs_terms;
  for (const auto i : irange(rhs_terms.size())) {
    if (!common_rhs_terms.at(i)) {
      new_rhs_terms.push_back(rhs_terms[i]);
    }
  }

  Val* new_lhs = nullptr;
  if (new_lhs_terms.empty()) {
    new_lhs = value->fusion()->zeroVal(*bop->lhs()->getDataType());
  } else {
    new_lhs = maybeFlattenedOpOf(BinaryOpType::Add, std::move(new_lhs_terms));
  }

  Val* new_rhs = nullptr;
  if (new_rhs_terms.empty()) {
    new_rhs = value->fusion()->zeroVal(*bop->lhs()->getDataType());
  } else {
    new_rhs = maybeFlattenedOpOf(BinaryOpType::Add, std::move(new_rhs_terms));
  }

  auto new_val = IrBuilder::create<Val>(*value->getDataType());
  IrBuilder::create<BinaryOp>(op, new_val, new_lhs, new_rhs);

  return new_val;
}

} // namespace rules

#define RUN_PASS(pass_name)                                     \
  if (disabled_passes == nullptr ||                             \
      (!disabled_passes->empty() &&                             \
       disabled_passes->count(#pass_name) == 0)) {              \
    simplified = recurseDown(simplified, [&context](Val* val) { \
      return rules::pass_name(val, context);                    \
    });                                                         \
    logger->record(#pass_name, simplified);                     \
  }

// Requires that all the passes before the barrier to be converged before
// procceeding to the passes after the barrier.
#define PASS_BARRIER                \
  if (old_simplified != simplified) \
  continue

Val* simplifyExpr(
    Val* value,
    const std::list<VarInfo>& variables,
    std::vector<Val*> assumptions,
    bool preserve_error) {
  FusionGuard fg(value->fusion());
  const Context context(variables, assumptions, preserve_error);
  auto logger = debug_print::createLogger(value);

  // nullptr -> disable nothing
  // empty set -> disable everything
  // {"abc", "def"} -> disable passes abc and def
  std::unique_ptr<std::unordered_set<std::string>> disabled_passes = nullptr;
  if (isOptionDisabled(DisableOption::ExprSimplify)) {
    const auto& v = getDisableOptionArguments(DisableOption::ExprSimplify);
    disabled_passes =
        std::make_unique<std::unordered_set<std::string>>(v.begin(), v.end());
  }

  Val* simplified = value;
  Val* old_simplified = nullptr;
  while (old_simplified != simplified) {
    old_simplified = simplified;

    // Passes other than assoc_comm::flatten assumes that all
    // associative-and-commutative binary ops (such as + *) are flattened. So
    // that they don't need to worry about things like
    // (a + b) + c vs a + (b + c)
    // So the first step before running all other passes is always flattening
    //
    // Note that, some passes might create nested flattened ops, something like
    // FlattenedAdd(FlattenedAdd(...), ...), so we should rerun flatten at the
    // beginning of each round instead of flattening before the while loop.
    simplified = assoc_comm::flatten(simplified);
    logger->record(debug_print::kFlattenName, simplified);

    RUN_PASS(canonicalizeVariables);
    RUN_PASS(cancelTermsInPredicate);
    RUN_PASS(eliminateTrivialComputation);
    RUN_PASS(eliminateTrivialPredicate);
    RUN_PASS(simplifyDivisibleDivMod);
    RUN_PASS(cancelDivMod);
    RUN_PASS(fundamentalDivisionWithRemainderProperty);
    RUN_PASS(convertDivToMulInPredicate);
    PASS_BARRIER;
    RUN_PASS(distributeDivisibleDivMod);
    RUN_PASS(distributeGcdRemainderDivMod);
    RUN_PASS(factorizeGcd);
    PASS_BARRIER;
    RUN_PASS(distributeMul);
    PASS_BARRIER;
    RUN_PASS(reducePredicateRegisterUsage);
  }

  auto unflattened = assoc_comm::unflatten(simplified, context);
  logger->record(debug_print::kUnflattenName, unflattened);
  return unflattened;
}

#undef RUN_PASS

} // namespace nvfuser
