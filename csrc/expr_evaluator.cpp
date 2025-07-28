// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <functional>
#include <iostream>

#include <debug.h>
#include <evaluator_common.h>
#include <expr_evaluator.h>
#include <instrumentation.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <logical_domain_map.h>
#include <multidevice/utils.h>
#include <polymorphic_value.h>

namespace nvfuser {

namespace {

// Given a value, if it is not a fusion input, return the empty string. If it is
// a fusion input return a string like "input 2 ". This helper is used to
// provide more informative error messages when a malformed input is received.
std::string getInputPosString(const Val* val) {
  if (!val->isFusionInput()) {
    return "";
  }
  // Get position
  const std::vector<Val*>& inputs = val->fusion()->inputs();
  int64_t pos = -1;
  for (size_t i : arange(inputs.size())) {
    if (inputs[i] == val) {
      pos = (int64_t)i;
      break;
    }
  }
  NVF_ERROR(
      pos != -1,
      "val->isFusionInput() is true but val cannot be found in fusion inputs: ",
      val->toString());
  std::stringstream ss;
  return "input " + std::to_string(pos) + ", ";
}

void validateValWithConcreteValue(
    const Val* value,
    const PolymorphicValue& concrete_value) {
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    NVF_CHECK(
        concrete_value.is<at::Tensor>(),
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be an at::Tensor but got scalar ",
        concrete_value);
    const auto& t = concrete_value.as<at::Tensor>();
    int64_t expect_dim =
        std::ssize(TensorDomain::noReductions(tv->getLogicalDomain()));
    NVF_CHECK(
        t.dim() == expect_dim,
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be bound to a tensor of rank ",
        expect_dim,
        ", but got a tensor of rank ",
        t.dim());
    NVF_CHECK(
        (value->dtype() == DataType::Index &&
         (t.scalar_type() == torch::kInt64 ||
          t.scalar_type() == torch::kInt32)) ||
            (t.scalar_type() == data_type_to_aten(value->dtype())),
        "Expected ",
        getInputPosString(tv),
        tv->toString(),
        ", to be bound to a tensor of dtype ",
        value->dtype(),
        ", but got a tensor of dtype ",
        aten_to_data_type(concrete_value.as<at::Tensor>().scalar_type()));
    // Intermediate tensorviews marked as CPU scalars will be created as meta
    // tensors during compilation. For example, for fusions containing SDPA fwd
    // and bwd, some outputs of the fwd op (philox seed, philox offset) are CPU
    // scalars.
    if (tv->isCpuScalar()) {
      NVF_CHECK(
          is_cpu_scalar(t) || is_meta_scalar(t),
          "Expected ",
          getInputPosString(tv),
          tv->toString(),
          ", to be bound to a CPU or meta scalar tensor "
          ", but got a tensor on device ",
          t.device(),
          " with ",
          t.numel(),
          " elements");
    } else {
      NVF_CHECK(
          !t.defined() || t.is_cuda() || t.is_meta(),
          "Expected ",
          getInputPosString(tv),
          tv->toString(),
          ", to be bound to a CUDA or meta tensor, but got a tensor on device ",
          t.device());
    }
  } else {
    NVF_CHECK(
        !concrete_value.is<at::Tensor>(),
        "Expected ",
        getInputPosString(value),
        value->toString(),
        ", to be a scalar but got ",
        aten_to_data_type(concrete_value.as<at::Tensor>().scalar_type()),
        " tensor of rank ",
        concrete_value.as<at::Tensor>().dim());

    NVF_CHECK(
        hasCompatibleDataType(concrete_value, value->dtype()),
        "Scalar value ",
        concrete_value,
        " is not compatible with the expected data type: ",
        value->dtype(),
        ".");
  }
}

} // namespace

void ExpressionEvaluator::bindTensorDomain(
    const TensorView* tv,
    const at::Tensor& t,
    const bool evaluate_validate) {
  auto logical_domain = TensorDomain::noReductions(tv->getLogicalDomain());
  NVF_ERROR(
      t.dim() == (int64_t)logical_domain.size(),
      "Expected ",
      getInputPosString(tv),
      tv->toString(),
      ", to be bound to a tensor of rank ",
      logical_domain.size(),
      ", but got a tensor of rank ",
      t.dim());

  std::vector<int64_t> logical_sizes = unshardedSizes(tv, t.sizes());

  // Adjust the last dimension of the logical domain to support DataType
  // that is not supported by PyTorch. See the comment of getLastDimAdjustment
  // in type.h for more details.
  const auto adjust_last_dim = getLastDimAdjustment(tv->dtype());
  if (adjust_last_dim.denominator != 1 || adjust_last_dim.numerator != 1) {
    NVF_ERROR(!logical_sizes.empty(), "DataType not supported");
    int64_t last_id_index = -1;
    for (const auto& [i, id] : enumerate(tv->getLogicalDomain())) {
      if (id == tv->getMaybeAllocationDomain().back()) {
        last_id_index = i;
        break;
      }
    }
    NVF_ERROR(last_id_index != -1, "could not find the last ID in allocation for sub byte data types.");
    auto& last_dim = logical_sizes[last_id_index];
    last_dim = adjust_last_dim.fromATenToNVF(last_dim);
  }

  for (auto i : arange(t.dim())) {
    auto id = logical_domain[i];
    if (id->isBroadcast()) {
      bind_(id->extent(), 1, evaluate_validate);
      if (id->hasExpandedExtent()) {
        // Verify that t is also expanded
        NVF_ERROR(
            logical_sizes[i] == 1 || t.stride(i) == 0,
            "IterDomain ",
            id->toString(),
            " in ",
            getInputPosString(tv),
            "TensorView ",
            tv->toString(),
            " has expanded extent but input tensor has size ",
            logical_sizes[i],
            " and stride ",
            t.stride(i),
            " in dimension ",
            i);
        bind_(id->expandedExtent(), logical_sizes[i], evaluate_validate);
      }
    } else {
      bind_(id->extent(), logical_sizes[i], evaluate_validate);
    }
  }
}

void ExpressionEvaluator::bind_(
    const Val* value,
    PolymorphicValue concrete_value,
    bool evaluate_validate) {
  using namespace PolymorphicValue_functions;
  NVF_CHECK(concrete_value.hasValue(), "Cannot bind to undefined value");
  if (value->isConst()) {
    NVF_CHECK(
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
    NVF_CHECK(
        same,
        "Tried to bind to a value: ",
        getInputPosString(value),
        value->toInlineString(),
        "(which evaluated to ",
        toString(evaluated_value),
        ") as ",
        toString(concrete_value));
  }
  if (auto tv = dynamic_cast<const TensorView*>(value)) {
    const auto& t = concrete_value.as<at::Tensor>();
    bindTensorDomain(tv, t, evaluate_validate);
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
  NVF_ERROR(isParallelTypeThread(pt));
  if (precomputed_values_) {
    // Need to bind the thread value to integer machine
    //  in pre-computed mode.
    precomputed_values_->bindConcreteParallelTypeValue(
        pt, std::move(concrete_value));
  } else {
    bind(stringifyThreadSize(pt), std::move(concrete_value));
  }
}

const PolymorphicValue& ExpressionEvaluator::evaluate(ParallelType pt) {
  auto it = known_named_scalars_.find(stringifyThreadSize(pt));
  if (it != known_named_scalars_.end()) {
    return it->second;
  }
  return null_;
}

const PolymorphicValue& ExpressionEvaluator::evaluate(const Val* value) {
  return evaluate(value, known_values_);
}

PolymorphicValue ExpressionEvaluator::evaluate(const Val* value) const {
  std::unordered_map<const Val*, PolymorphicValue> known_values;
  return evaluate(value, known_values);
}

const PolymorphicValue& ExpressionEvaluator::evaluate(
    const Val* value,
    std::unordered_map<const Val*, PolymorphicValue>& known_values) const {
  FUSER_PERF_SCOPE("ExpressionEvaluator::evaluate");
  if (precomputed_values_ && precomputed_values_->hasValidValues()) {
    if (precomputed_values_->getMaybeValueFor(value).hasValue()) {
      return precomputed_values_->getMaybeValueFor(value);
    }
  }

  std::reference_wrapper<const PolymorphicValue> maybe_concrete_value =
      getValue(value, known_values);
  if (!maybe_concrete_value.get().hasValue()) {
    if (auto def = value->definition()) {
      auto outputs = def->evaluate(*this, known_values);
      for (auto i : arange(def->outputs().size())) {
        known_values[def->output(i)] = std::move(outputs[i]);
      }
      maybe_concrete_value = getValue(value, known_values);
    }
  }
  return maybe_concrete_value;
}

const PolymorphicValue& ExpressionEvaluator::getValue(
    const Val* value,
    const std::unordered_map<const Val*, PolymorphicValue>&
        additional_known_values) const {
  if (value->isScalar() && value->isConst()) {
    return value->value();
  }

  if (value->isA<NamedScalar>()) {
    const auto it = known_named_scalars_.find(value->as<NamedScalar>()->name());
    if (it != known_named_scalars_.end()) {
      return it->second;
    }
  }

  auto it = known_values_.find(value);
  if (it != known_values_.end()) {
    return it->second;
  }

  if (&additional_known_values != &known_values_) {
    it = additional_known_values.find(value);
    return it != additional_known_values.end() ? it->second : null_;
  }

  return null_;
}

void ExpressionEvaluator::print() const {
  using namespace PolymorphicValue_functions;

  debug() << "\nEvaluation context\n";
  debug() << "--------------------\n";

  for (const auto& kv : known_values_) {
    NVF_ERROR(!kv.first->isConstScalar());
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

namespace {
// Error handling for
// ExpressionEvaluator::propagateBoundValuesThroughExactMaps(Fusion* fusion)
void handlePropagateError(
    Fusion* fusion,
    ExpressionEvaluator* expr_eval,
    const std::shared_ptr<VectorOfUniqueEntries<const IterDomain*>>& id_set) {
  std::unordered_map<const IterDomain*, int64_t> id_to_size;
  std::set<int64_t> sizes;

  for (const auto id : *id_set) {
    auto eval_val = expr_eval->evaluate(id->extent());
    if (eval_val.hasValue()) {
      NVF_ERROR(
          eval_val.is<int64_t>(),
          "Invalid extent value found, while processing ID: ",
          id);
      id_to_size[id] = eval_val.as<int64_t>();
      sizes.insert(eval_val.as<int64_t>());
    }
  }

  std::stringstream err_msg;
  err_msg << "When trying to propagate constant tensor sizes through the graph "
             "a conflict was found with "
          << sizes.size()
          << " different sizes across dimensions that are expected to match.\n";

  // Track which size is associated with which TV and IterDomain
  std::unordered_map<
      int64_t,
      std::vector<std::pair<TensorView*, const IterDomain*>>>
      size_to_info;

  for (auto tv : fusion->allTvs()) {
    for (const IterDomain* id : tv->domain()->allIDs()) {
      if (auto it = id_to_size.find(id); it != id_to_size.end()) {
        size_to_info[it->second].push_back({tv, id});
      }
    }
  }

  // Check which TensorViews mismatch and check if they're directly related.
  // If so, the expression between them may be the problematic expression, and
  // the error will point to that expression(s). Don't bother to speed up this
  // check as it only runs after an error is detected.
  bool found_producer_consumer_issue = false;
  // Assume producer/consumer relationship
  for (auto [dim_size_1, tv_id_pairs_1] : size_to_info) {
    for (auto [dim_size_2, tv_id_pairs_2] : size_to_info) {
      if (dim_size_1 <= dim_size_2) {
        // N^2 algorithm, only process when one size is less than the other,
        // avoids duplicate entries.
        continue;
      }

      for (const auto& [tv1, id1] : tv_id_pairs_1) {
        for (const auto& [tv2, id2] : tv_id_pairs_2) {
          bool tv1_is_consumer = false;

          // Check for producer-consumer relationship
          auto producer_tvs_of_tv1 = ir_utils::producerTvsOf(tv1);
          auto producer_tvs_of_tv2 = ir_utils::producerTvsOf(tv2);
          if (std::find(
                  producer_tvs_of_tv1.begin(),
                  producer_tvs_of_tv1.end(),
                  tv2) != producer_tvs_of_tv1.end()) {
            tv1_is_consumer = true;
          } else if (
              std::find(
                  producer_tvs_of_tv2.begin(),
                  producer_tvs_of_tv2.end(),
                  tv1) == producer_tvs_of_tv2.end()) {
            // No relationship found, skip
            continue;
          }

          Expr* relationship =
              tv1_is_consumer ? tv1->definition() : tv2->definition();

          // Found at least one consumer/producer relationship with mismatched
          // sizes.
          found_producer_consumer_issue = true;

          // Produce error message. Normally I'd just use swap but keys in an
          // unordered map are const, so doing some juggling with the strings
          // instead.
          std::stringstream tv1_error;
          std::stringstream tv2_error;
          tv1_error << " TV: " << tv1 << " id: " << id1
                    << " found size: " << dim_size_1 << "\n";
          tv2_error << " TV: " << tv2 << " id: " << id2
                    << " found size: " << dim_size_2 << "\n";
          err_msg << "  For Producer"
                  << (tv1_is_consumer ? tv2_error.str() : tv1_error.str());
          err_msg << "  For Consumer"
                  << (tv1_is_consumer ? tv1_error.str() : tv2_error.str());
          err_msg << "  With producer-consumer relationship through the "
                     "expression: "
                  << relationship << "\n";
        }
      }
    }
  }

  if (found_producer_consumer_issue) {
    NVF_THROW(err_msg.str());
  }

  if (size_to_info.size() > 1) {
    for (const auto& [size, info_pairs] : size_to_info) {
      err_msg << "Size " << size << " found for ID, in TV:\n";
      for (auto info_pair : info_pairs) {
        err_msg << "  " << info_pair.second << ", " << info_pair.first << "\n";
      }
    }
    err_msg << "These sizes should all match.\n";
  } else {
    err_msg
        << "Something went wrong trying to detect what went wrong!"
        << " There should have been ID's in TVs that should match, but don't."
        << " Somehow IDs were registered with the exact graph that aren't used "
           "in the Fusion."
        << std::endl;
  }

  NVF_THROW(err_msg.str());
}
} // namespace

void ExpressionEvaluator::propagateBoundValuesThroughExactMaps(
    Fusion* fusion,
    ExactLogicalDomainMap* exact_map) {
  // We map Symbolic IterDomains here only if their extents match. This avoids
  // mapping between symbolic domains that might concretize to an (Iteration,
  // Broadcast) pair from a resolved broadcast.
  std::unique_ptr<ExactLogicalDomainMap> exact_map_ptr;
  if (exact_map == nullptr) {
    exact_map_ptr = std::make_unique<ExactLogicalDomainMap>(fusion);
    exact_map = exact_map_ptr.get();
  }
  const auto mapped_sets = exact_map->getMappedSets();

  for (const auto& set : mapped_sets.disjointSets()) {
    int64_t known_size = -1;
    std::vector<Val*> unknown_vals;
    for (const auto id : *set) {
      auto eval_val = evaluate(id->extent());
      if (eval_val.hasValue()) {
        NVF_ERROR(eval_val.is<int64_t>(), "Invalid extent value");
        int64_t this_size = eval_val.as<int64_t>();
        if (known_size == -1) {
          known_size = this_size;
        } else if (known_size != this_size) {
          handlePropagateError(fusion, this, set);
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
  NVF_ERROR(
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
