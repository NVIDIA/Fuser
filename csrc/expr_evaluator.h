// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <dynamic_type.h>
#include <evaluator_common.h>
#include <ir/interface_nodes.h>
#include <iter_visitor.h>

#include <c10/util/Optional.h>

#include <string>
#include <unordered_map>

namespace nvfuser {

class PrecomputedValues;

//! Calculate Fusion IR expressions
class TORCH_CUDA_CU_API ExpressionEvaluator {
  void bind_(const Val* value, const EvaluatorValue& concrete_value);
  void bind_(const std::string& name, const EvaluatorValue& concrete_value);

 public:
  //! Bind a concrete value to an IR variable
  template <typename T>
  void bind(const Val* value, const T& concrete_value) {
    bind_(value, EvaluatorValue(concrete_value));
  }

  //! Bind a concrete value to a named scalar
  template <typename T>
  void bind(const std::string& name, const T& concrete_value) {
    bind_(name, EvaluatorValue(concrete_value));
  }

  //! Set a concrete value for a parallel dimension
  void bind(ParallelType pt, Int::ScalarType concrete_value);

  //! Try to evaluate a Fusion IR value
  c10::optional<EvaluatorValue> evaluate(const Val* value);

  //! Try to evaluate a parallel dimension
  c10::optional<EvaluatorValue> evaluate(ParallelType pt);

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
  //! value. This is currently just done with ExactRootDomainMap, but
  //! can be similarly done with the Exact CA map as well.
  void propagateBoundValuesThroughExactMaps(Fusion* fusion);

 private:
  c10::optional<EvaluatorValue> getValue(const Val* value);

 private:
  // TODO: Consider make this const. It can't be const as bind() of
  // this class calls
  // PrecomputedValuess::bindConcreteParallelTypeValue, but it's
  // unclear why the precompute values cannot be kept constant and
  // binding a value to ExpressionEvaluator just updates
  // known_named_scalars_.
  PrecomputedValues* precomputed_values_ = nullptr;
  std::unordered_map<const Val*, EvaluatorValue> known_values_;
  std::unordered_map<std::string, EvaluatorValue> known_named_scalars_;
};

} // namespace nvfuser
