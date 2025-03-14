// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <evaluator_common.h>
#include <exceptions.h>
#include <ir/cloner.h>
#include <ir/interface_nodes.h>
#include <iter_visitor.h>
#include <logical_domain_map.h>
#include <polymorphic_value.h>
#include <visibility.h>

#include <string>
#include <unordered_map>

namespace nvfuser {

class PrecomputedValues;

//! Calculate Fusion IR expressions
class ExpressionEvaluator {
 public:
  //! Bind a concrete value to an IR variable
  //! If evaluate_validate is true, and value is evaluatable with the
  //! information already known, then evaluate and validate the value with the
  //! concrete value.
  void bind(
      const Val* value,
      PolymorphicValue concrete_value,
      bool evaluate_validate = false) {
    bind_(value, std::move(concrete_value), evaluate_validate);
  }

  //! Bind a concrete value to a named scalar
  void bind(const std::string& name, PolymorphicValue concrete_value) {
    bind_(name, std::move(concrete_value));
  }

  //! Set a concrete value for a parallel dimension
  void bind(ParallelType pt, PolymorphicValue concrete_value);

  //! Try to evaluate a Fusion IR value
  NVF_API const PolymorphicValue& evaluate(const Val* value);

  //! Try to evaluate a parallel dimension
  const PolymorphicValue& evaluate(ParallelType pt);

  //! Evaluates a value through a const evaluator reference.
  //! Initializes a known_values map to store intermediate values in lieu of
  //! known_values_.
  NVF_API PolymorphicValue evaluate(const Val* value) const;

  //! Base evaluate method called by other overloads and Expr::evaluate.
  const PolymorphicValue& evaluate(
      const Val* value,
      std::unordered_map<const Val*, PolymorphicValue>& known_values) const;

  bool isKnown(const Val* value) const {
    return known_values_.count(value) > 0;
  }

  void invalidate(const Val* value) {
    known_values_.erase(value);
  }

  //! Debugging helper, prints all the currently known values
  void print() const;

  void bindPrecomputedValues(PrecomputedValues* precomputed_values) {
    precomputed_values_ = precomputed_values;
  }

  auto& precomputedValues() {
    return precomputed_values_;
  }

  //! Augment the evaluator with the exact root-domain map such that
  //! if the extent of a root ID is known, the extents of all other
  //! root IDs that are exactly mapped also get bound to the same
  //! value. This is currently just done with ExactLogicalDomainMap, but
  //! can be similarly done with the Exact CA map as well.
  void propagateBoundValuesThroughExactMaps(
      Fusion* fusion,
      ExactLogicalDomainMap* exact_map = nullptr);

  ExpressionEvaluator clone(IrCloner& ir_cloner) const;

 private:
  void bind_(
      const Val* value,
      PolymorphicValue concrete_value,
      bool evaluate_validate);

  void bind_(const std::string& name, PolymorphicValue concrete_value);

  void bindTensorDomain(
      const TensorView* tv,
      const at::Tensor& t,
      bool evaluate_validate);

  const PolymorphicValue& getValue(
      const Val* value,
      const std::unordered_map<const Val*, PolymorphicValue>&
          additional_known_values) const;

 private:
  // TODO: Consider make this const. It can't be const as bind() of
  // this class calls
  // PrecomputedValuess::bindConcreteParallelTypeValue, but it's
  // unclear why the precompute values cannot be kept constant and
  // binding a value to ExpressionEvaluator just updates
  // known_named_scalars_.
  PrecomputedValues* precomputed_values_ = nullptr;
  std::unordered_map<const Val*, PolymorphicValue> known_values_;
  std::unordered_map<std::string, PolymorphicValue> known_named_scalars_;
  PolymorphicValue null_ = std::monostate{};
};

} // namespace nvfuser
