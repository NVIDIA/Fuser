// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <debug.h>
#include <evaluator_common.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <root_domain_map.h>

#include <functional>
#include <iostream>

namespace nvfuser {

namespace {

void validateValWithConcreteValue(
    const Val* value,
    const PolymorphicValue& concrete_value) {
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    TORCH_CHECK(
        concrete_value.is<at::Tensor>(),
        "Expected ",
        tv->toString(),
        " to be bound to an at::Tensor, but got ",
        concrete_value.type().name());
    const auto& t = concrete_value.as<at::Tensor>();
    auto expect_dim =
        (int64_t)TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size();
    TORCH_CHECK(
        t.dim() == expect_dim,
        "Expected ",
        tv->toString(),
        " to be bound to a tensor of rank ",
        expect_dim,
        ", but got a tensor of rank ",
        t.dim());
    auto actual_dtype = aten_to_data_type(t.scalar_type());
    TORCH_CHECK(
        (value->dtype() == DataType::Index && isIntegralType(actual_dtype)) ||
            (value->dtype() == actual_dtype),
        "Expected ",
        tv->toString(),
        " to be bound to a tensor of dtype ",
        value->dtype(),
        ", but got a tensor of dtype ",
        actual_dtype);
    if (tv->isCpuScalar()) {
      TORCH_CHECK(
          is_cpu_scalar(t),
          "Expected ",
          tv->toString(),
          " to be bound to a CPU scalar tensor "
          ", but got a tensor on device ",
          t.device(),
          " with ",
          t.numel(),
          " elements");
    } else {
      TORCH_CHECK(
          t.is_cuda() || t.is_meta(),
          "Expected ",
          tv->toString(),
          " to be bound to a CUDA or meta tensor, but got a tensor on device ",
          t.device());
    }
  } else {
    TORCH_CHECK(
        hasCompatibleDataType(concrete_value, value->dtype()),
        "Scalar value is not compatible with the given data type.");
  }
}

} // namespace

void ExpressionEvaluator::bind_(
    const Val* value,
    PolymorphicValue concrete_value,
    bool evaluate_validate) {
  using namespace PolymorphicValue_functions;
  TORCH_CHECK(concrete_value.hasValue(), "Cannot bind to undefined value");
  if (value->isConst()) {
    TORCH_CHECK(
        value->value() == concrete_value,
        "Tried to bind to a constant value: ",
        toString(value->value()),
        " as ",
        toString(concrete_value));
    return;
  }
  validateValWithConcreteValue(value, concrete_value);
  if (evaluate_validate &&
      ir_utils::dependenciesSatisfied(value, known_values_)) {
    auto evaluated_value = evaluate(value);
    using namespace PolymorphicValue_functions;
    auto same = isSame(evaluated_value, concrete_value);
    TORCH_CHECK(
        same,
        "Tried to bind to a value: ",
        value->toInlineString(),
        "(which evaluated to ",
        toString(evaluated_value),
        ") as ",
        toString(concrete_value));
  }
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    const auto& t = concrete_value.as<at::Tensor>();
    auto rfactor_domain =
        TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    TORCH_INTERNAL_ASSERT(
        t.dim() == (int64_t)rfactor_domain.size(),
        "Expected ",
        tv->toString(),
        " to be bound to a tensor of rank ",
        rfactor_domain.size(),
        ", but got a tensor of rank ",
        t.dim());
    for (auto i : c10::irange(t.dim())) {
      bind_(
          rfactor_domain[i]->getMaybeExpandedExtent(),
          t.size(i),
          evaluate_validate);
    }
  }
  if (value->isA<NamedScalar>()) {
    known_named_scalars_[value->as<NamedScalar>()->name()] =
        std::move(concrete_value);
  } else {
    known_values_[value] = std::move(concrete_value);
  }
}

void ExpressionEvaluator::bind_(
    const std::string& name,
    PolymorphicValue concrete_value) {
  known_named_scalars_[name] = std::move(concrete_value);
}

void ExpressionEvaluator::bind(
    ParallelType pt,
    PolymorphicValue concrete_value) {
  TORCH_INTERNAL_ASSERT(isParallelTypeThread(pt));
  if (precomputed_values_) {
    // Need to bind the thread value to integer machine
    //  in pre-computed mode.
    precomputed_values_->bindConcreteParallelTypeValue(
        pt, std::move(concrete_value));
  } else {
    bind(stringifyThreadSize(pt), std::move(concrete_value));
  }
}

const PolymorphicValue& ExpressionEvaluator::evaluate(const Val* value) {
  if (precomputed_values_ && precomputed_values_->ready()) {
    if (precomputed_values_->getMaybeValueFor(value).hasValue()) {
      return precomputed_values_->getMaybeValueFor(value);
    }
  }

  std::reference_wrapper<const PolymorphicValue> maybe_concrete_value =
      getValue(value);
  if (!maybe_concrete_value.get().hasValue()) {
    if (auto def = value->definition()) {
      FUSER_PERF_SCOPE("ExpressionEvaluator::evaluate");
      std::vector<PolymorphicValue> inputs;
      inputs.reserve(def->inputs().size());
      for (auto i : def->inputs()) {
        const auto& eval_i = evaluate(i);
        if (!eval_i.hasValue()) {
          return null_;
        }
        inputs.emplace_back(eval_i);
      }
      auto outputs = def->evaluate(*this, inputs);
      for (auto i : c10::irange(def->outputs().size())) {
        known_values_[def->output(i)] = std::move(outputs[i]);
      }
      maybe_concrete_value = getValue(value);
    }
  }
  return maybe_concrete_value;
}

const PolymorphicValue& ExpressionEvaluator::evaluate(ParallelType pt) {
  auto it = known_named_scalars_.find(stringifyThreadSize(pt));
  if (it != known_named_scalars_.end()) {
    return it->second;
  }
  return null_;
}

const PolymorphicValue& ExpressionEvaluator::getValue(const Val* value) {
  if (value->isScalar() && value->isConst()) {
    return value->value();
  }

  if (value->isA<NamedScalar>()) {
    const auto it = known_named_scalars_.find(value->as<NamedScalar>()->name());
    if (it != known_named_scalars_.end()) {
      return it->second;
    }
  }

  const auto it = known_values_.find(value);
  return it != known_values_.end() ? it->second : null_;
}

void ExpressionEvaluator::print() const {
  using namespace PolymorphicValue_functions;

  debug() << "\nEvaluation context\n";
  debug() << "--------------------\n";

  for (const auto& kv : known_values_) {
    TORCH_INTERNAL_ASSERT(!kv.first->isConstScalar());
    debug() << kv.first << " = " << toString(kv.second) << " ; "
            << *kv.first->getValType() << "\n";
  }

  for (const auto& kv : known_named_scalars_) {
    debug() << kv.first << " = " << toString(kv.second) << " ;\n";
  }

  debug() << "\nPre-computed Values\n";
  if (precomputed_values_ != nullptr) {
    precomputed_values_->print();
  }
  debug() << "--------------------\n\n";
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
