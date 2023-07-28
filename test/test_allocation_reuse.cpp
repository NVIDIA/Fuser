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
  using TimeType = int16_t;
  std::vector<std::pair<TimeType, TimeType>> intervals = {
      {2, 9}, {1, 3}, {4, 7}, {5, 9}, {0, 10}};

  auto cit = CenteredIntervalTree<TimeType>(intervals);

  std::cout << cit.toString() << std::endl;

  auto overlaps = cit.getIntervalsContainingPoint(5);

  std::cout << "Intervals containing 5: " << std::to_string(overlaps.size())
            << std::endl;
  TORCH_CHECK(overlaps.size() == 1);
}

} // namespace nvfuser
