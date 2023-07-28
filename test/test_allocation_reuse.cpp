// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <device_lower/pass/interval_tree.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class AllocationReuse : public NVFuserTest {};

// Simple exercises for CenteredIntervalTree
TEST_F(AllocationReuse, CenteredIntervalTree) {
  // For this test case, we will have the following intervals
  //
  //     ----------
  //  -----
  //        ------
  //          -----
  // ---------------
  // -
  //    -
  //        -
  //            -
  //               -
  using TimeType = int16_t;
  std::vector<std::pair<TimeType, TimeType>> intervals = {
      {2, 9},
      {1, 3},
      {4, 7},
      {5, 9},
      {0, 10},
      {0, 0},
      {1, 1},
      {4, 4},
      {6, 6},
      {10, 10}};

  auto cit = CenteredIntervalTree<TimeType>(intervals);

  std::cout << cit.toString() << std::endl;

  auto overlaps = cit.getIntervalsContainingPoint(4);

  std::cout << "Intervals containing time point 4: "
            << std::to_string(overlaps.size()) << std::endl;
  for (const auto ind : overlaps) {
    const auto [start, stop] = intervals.at(ind);
    std::cout << "  [" << start << " , " << stop << "]" << std::endl;
  }
  EXPECT_EQ(overlaps.size(), 4);
}

} // namespace nvfuser
