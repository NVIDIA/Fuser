// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <evaluator_common.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <root_domain_map.h>

#include <iostream>

namespace nvfuser {

namespace {

bool equals(const Val* value, const EvaluatorValue& concrete_value) {
  switch (std::get<PrimDataType>(value->getDataType()->type)) {
    case DataType::Int: {
      if (!concrete_value.is<int64_t>()) {
        return false;
      }
      auto val = value->getInt();
      return val.has_value() && val.value() == concrete_value.as<int64_t>();
    }
    case DataType::Double: {
      if (!concrete_value.is<double>()) {
        return false;
      }
      auto val = value->getDouble();
      return val.has_value() && val.value() == concrete_value.as<double>();
    }
    case DataType::Bool: {
      if (!concrete_value.is<bool>()) {
        return false;
      }
      auto val = value->getBool();
      return val.has_value() && val.value() == concrete_value.as<bool>();
    }
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
}

template <typename T>
EvaluatorValue toOptionalEvaluatorValue(std::optional<T> i) {
  if (!i) {
    return std::monostate{};
  }
  return EvaluatorValue(i.value());
}

} // namespace

void ExpressionEvaluator::bind_(
    const Val* value,
    const EvaluatorValue& concrete_value) {
  if (equals(value, concrete_value)) {
    return;
  }
  TORCH_CHECK(value->isScalar());
  TORCH_CHECK(
      value->dtype() == DataType::Int || value->dtype() == DataType::Double ||
      value->dtype() == DataType::Bool);
  TORCH_CHECK(!value->isConstScalar(), "Tried to bind to a constant value");
  TORCH_CHECK(
      value->definition() == nullptr,
      "Tried to bind to a value that is computed in the Fusion IR: ",
      value->toInlineString(),
      " with ",
      concrete_value);
  if (value->isA<NamedScalar>()) {
    known_named_scalars_[value->as<NamedScalar>()->name()] = concrete_value;
  } else {
    known_values_[value] = concrete_value;
  }
}

void ExpressionEvaluator::bind_(
    const std::string& name,
    const EvaluatorValue& concrete_value) {
  known_named_scalars_[name] = concrete_value;
}

void ExpressionEvaluator::bind(
    ParallelType pt,
    Int::ScalarType concrete_value) {
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt));
  if (precomputed_values_) {
    // Need to bind the thread value to integer machine
    //  in pre-computed mode.
    precomputed_values_->bindConcreteParallelTypeValue(pt, concrete_value);
  } else {
    bind(stringifyThreadSize(pt), EvaluatorValue(concrete_value));
  }
}

EvaluatorValue ExpressionEvaluator::evaluate(const Val* value) {
  if (precomputed_values_ && precomputed_values_->ready()) {
    if (precomputed_values_->getMaybeValueFor(value).hasValue()) {
      return precomputed_values_->getMaybeValueFor(value);
    }
  }

  auto maybe_concrete_value = getValue(value);
  if (!maybe_concrete_value.hasValue()) {
    if (auto def = value->definition()) {
      FUSER_PERF_SCOPE("ExpressionEvaluator::evaluate");
      if (def->isA<kir::BaseAddress>()) {
        return std::monostate{};
      }
      std::vector<EvaluatorValue> inputs;
      inputs.reserve(def->inputs().size());
      for (auto i : def->inputs()) {
        auto eval_i = evaluate(i);
        if (!eval_i.hasValue()) {
          return std::monostate{};
        }
        inputs.emplace_back(eval_i);
      }
      auto outputs = def->evaluate(inputs);
      for (auto i : c10::irange(def->outputs().size())) {
        known_values_[def->output(i)] = outputs[i];
      }
      maybe_concrete_value = getValue(value);
    }
  }
  return maybe_concrete_value;
}

EvaluatorValue ExpressionEvaluator::evaluate(ParallelType pt) {
  auto it = known_named_scalars_.find(stringifyThreadSize(pt));
  if (it != known_named_scalars_.end()) {
    return it->second;
  }
  return std::monostate{};
}

EvaluatorValue ExpressionEvaluator::getValue(const Val* value) {
  TORCH_INTERNAL_ASSERT(
      value->isIntegralScalar() || value->isFloatingPointScalar() ||
          value->isABool(),
      value->toInlineString(),
      " is not a supported type in expression evaluation.");

  if (value->isScalar() && value->isConst()) {
    if (value->isFloatingPointScalar()) {
      return toOptionalEvaluatorValue(value->as<Double>()->value());
    }
    if (value->isABool()) {
      return toOptionalEvaluatorValue(value->as<Bool>()->value());
    }
    if (value->isIntegralScalar()) {
      return toOptionalEvaluatorValue(value->as<Int>()->value());
    }
    TORCH_INTERNAL_ASSERT(
        false, "Data type not supported by ExpressionEvaluator");
  }

  if (value->isA<NamedScalar>()) {
    const auto it = known_named_scalars_.find(value->as<NamedScalar>()->name());
    if (it != known_named_scalars_.end()) {
      return it->second;
    }
  }

  const auto it = known_values_.find(value);
  return it != known_values_.end() ? it->second
                                   : EvaluatorValue(std::monostate{});
}

void ExpressionEvaluator::print() const {
  std::cout << "\nEvaluation context\n";
  std::cout << "--------------------\n";
  for (const auto& kv : known_values_) {
    TORCH_INTERNAL_ASSERT(!kv.first->isConstScalar());
    std::cout << kv.first << " = " << kv.second << " ; "
              << *kv.first->getValType() << "\n";
  }

  for (const auto& kv : known_named_scalars_) {
    std::cout << kv.first << " = " << kv.second << " ;\n";
  }

  std::cout << "\nPre-computed Values\n";
  if (precomputed_values_ != nullptr) {
    precomputed_values_->print();
  }
  std::cout << "--------------------\n\n";
}

void ExpressionEvaluator::propagateBoundValuesThroughExactMaps(Fusion* fusion) {
  const auto mapped_sets = ExactRootDomainMap(fusion).getMappedSets();

  for (const auto& set : mapped_sets.disjointSets()) {
    int64_t known_size = -1;
    std::vector<Val*> unknown_vals;
    for (const auto id : *set) {
      auto eval_val = evaluate(id->extent());
      if (eval_val.hasValue()) {
        TORCH_INTERNAL_ASSERT(eval_val.is<int64_t>(), "Invalid extent value");
        int64_t this_size = eval_val.as<int64_t>();
        if (known_size != -1) {
          TORCH_INTERNAL_ASSERT(
              known_size == this_size,
              "Conflicting sizes: ",
              known_size,
              ", ",
              this_size);
        } else {
          known_size = this_size;
        }
      } else {
        unknown_vals.push_back(id->extent());
      }
    }

    if (known_size == -1 || unknown_vals.empty()) {
      continue;
    }

    // Binding unknown vals to known_val
    for (auto unknown_val : unknown_vals) {
      bind(unknown_val, known_size);
    }
  }
}

ExpressionEvaluator ExpressionEvaluator::clone(IrCloner& ir_cloner) const {
  ExpressionEvaluator expr_eval;
  TORCH_INTERNAL_ASSERT(
      !precomputed_values_,
      "Cannot clone ExpressionEvaluator with bound PrecomputedValues");
  for (const auto& kv : known_values_) {
    expr_eval.known_values_[ir_cloner.clone(kv.first)] = kv.second;
  }
  expr_eval.known_named_scalars_.insert(
      known_named_scalars_.begin(), known_named_scalars_.end());
  return expr_eval;
}

} // namespace nvfuser
