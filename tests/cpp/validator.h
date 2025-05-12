// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>

#include <string>
#include <vector>

#include <exceptions.h>
#include <fusion.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/scheduler_types.h>
#include <validator_utils.h>

namespace nvfuser {

// Validation will look through the fusion and figure out how many elements were
// reduced to create each output. It will then compute a tolernace to use for
// allclose based on experimental results. The experimental results were based
// on adding two tensors then summing them. This of course has an assumption
// that we're always summing values between -2 and 2. If we start summing values
// larger than that this approach might not hold.
// If aten_outputs is empty, then infer the expected outputs from the fusion
// using expr evaluator.
//
// `fusion_outputs` is the return value of
// `FusionExecutorCache::runFusionWithInputs(aten_inputs)`. It's not
// always `fusion->outputs().size()` because `runFusionWithInputs`
// hides outputs that are inputs in-place updated.
void testValidate(
    Fusion* fusion,
    const KernelArgumentHolder& fusion_outputs,
    const KernelArgumentHolder& aten_inputs,
    std::vector<at::Tensor> aten_outputs,
    int line_number,
    const char* file_name,
    std::string err_msg = "",
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants());

// The variant with automatically inferred aten outputs. The `evaluate` method
// of the exprs in the fusion must be overriden to handle at::Tensor.
void testValidate(
    Fusion* fusion,
    const KernelArgumentHolder& fusion_outputs,
    const KernelArgumentHolder& aten_inputs,
    int line_number,
    const char* file_name,
    std::string err_msg = "",
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants());

// A gmock matcher for matching heuristics.
MATCHER_P(HeuristicIs, heuristic, "") {
  return arg->schedulerType() == heuristic;
}

// Matches any subclass of T.
//
// See
// https://google.github.io/googletest/gmock_cook_book.html#writing-new-monomorphic-matchers
// for how to write a matcher.
template <typename T>
class IsAMatcher {
 public:
  using is_gtest_matcher = void;

  bool MatchAndExplain(const PolymorphicBase* pb, std::ostream*) const {
    return pb->isA<T>();
  }

  void DescribeTo(std::ostream* os) const {
    *os << "is " << demangle(typeid(T).name());
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "is not " << demangle(typeid(T).name());
  }
};

template <typename T>
inline testing::Matcher<const PolymorphicBase*> IsA() {
  return IsAMatcher<T>();
}

// Validate that the fusion is segmented with desired scheduler, currently only
// supporting two segments
void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<SchedulerType>& expected_heuristics);

} // namespace nvfuser
