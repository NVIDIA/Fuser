// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using LoopSchedulingTest = NVFuserTest;

TEST_F(LoopSchedulingTest, Test1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {10}, {2, 5});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  fusion.printMath();

  {
    std::vector<IterDomain*> ref = tv0->getLogicalDomain();

    scheduler_utils::scheduleLoopDomainsLike({tv1, tv2, tv3}, ref);
  }
  
}

} // namespace nvfuser
