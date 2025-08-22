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
// A gmock matcher for matching heuristics.
MATCHER_P(HeuristicIs, heuristic, "") {
  return arg->schedulerType() == heuristic;
}

// Matches whether the loop domain of the tensorview is parallelized with the
// given parallel types at the given indices.
MATCHER_P2(DomainIsParallelized, parallel_types, indices, "") {
  const auto& domain = arg->getLoopDomain();
  if (indices.size() > domain.size()) {
    *result_listener << "Indices size " << indices.size()
                     << " is greater than domain size " << domain.size();
    return false;
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices.at(i) >= static_cast<int64_t>(domain.size())) {
      *result_listener << "Index " << indices.at(i)
                       << " is out of bounds for domain size " << domain.size();
      return false;
    }
    if (domain.at(indices.at(i))->getParallelType() != parallel_types.at(i)) {
      *result_listener << "Parallel type of " << arg->domain() << " at index "
                       << indices.at(i)
                       << " does not match expected parallel type "
                       << parallel_types.at(i);
      return false;
    }
  }
  return true;
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
