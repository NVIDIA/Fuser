// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <string>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>

#include "exceptions.h"
#include "fusion.h"
#include "runtime/fusion_executor_cache.h"
#include "scheduler/scheduler_types.h"
#include "validator_utils.h"

namespace nvfuser {
// A gmock matcher for matching heuristics.
MATCHER_P(HeuristicIs, expected, "") {
  const SchedulerType actual = arg->schedulerType();
  if (actual != expected) {
    *result_listener << "Expected " << expected << " but got " << actual;
  }
  return actual == expected;
}

MATCHER_P(IsParallelized, expected, "") {
  const ParallelType actual = arg->getParallelType();
  if (actual != expected) {
    *result_listener << "Expected " << expected << " but got " << actual;
  }
  return actual == expected;
}

// Matches any subclass of T.
//
// See
// https://google.github.io/googletest/gmock_cook_book.html#writing-new-monomorphic-matchers
// for how to write a matcher.
//
// testing::An doesn't work for checking subclasses -- `EXPECT_THAT(foo,
// testing::An<Bar>())` requires `foo` to be implicitly convertible to `Bar`.
template <typename T, bool kStrictly>
class IsAMatcher {
 public:
  using is_gtest_matcher = void;

  bool MatchAndExplain(const PolymorphicBase* pb, std::ostream*) const {
    if (kStrictly) {
      return pb->isStrictlyA<T>();
    } else {
      return pb->isA<T>();
    }
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
  return IsAMatcher<T, false>();
}

template <typename T>
inline testing::Matcher<const PolymorphicBase*> IsStrictlyA() {
  return IsAMatcher<T, true>();
}

// Validate that the fusion is segmented with desired scheduler, currently only
// supporting two segments
void validateSegmentation(
    FusionKernelRuntime* runtime,
    const std::vector<SchedulerType>& expected_heuristics);

} // namespace nvfuser
