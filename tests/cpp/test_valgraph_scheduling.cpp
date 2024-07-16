// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>

#include <id_model/schedule.h>
#include <ir/internal_base_nodes.h>

namespace nvfuser {

using ValGraphSchedulingTest = NVFuserTest;

// Create a ValGroup with both Reduction and Iteration domains. Test that the
// Reduction IterDomains are not used as representatives when merging.
TEST_F(ValGraphSchedulingTest, MergeIterationWithReduction) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard guard(fusion);

  auto id0 = IterDomainBuilder(
                 fusion->zeroVal(), IrBuilder::create<Val>(DataType::Index))
                 .iter_type(IterType::Reduction)
                 .build();
  auto id1 = IterDomainBuilder(
                 fusion->zeroVal(), IrBuilder::create<Val>(DataType::Index))
                 .iter_type(IterType::Iteration)
                 .build();
  auto id2 = IterDomainBuilder(
                 fusion->zeroVal(), IrBuilder::create<Val>(DataType::Index))
                 .iter_type(IterType::Broadcast)
                 .build();
  auto id3 = IterDomainBuilder(
                 fusion->zeroVal(), IrBuilder::create<Val>(DataType::Index))
                 .iter_type(IterType::Iteration)
                 .build();

  ValGraph graph;
  graph.initializeVal(id0);
  ValGroup g0 = graph.toGroup(id0);
  graph.initializeVal(id1, g0);

  graph.initializeVal(id2);
  ValGroup g1 = graph.toGroup(id2);
  graph.initializeVal(id3, g1);

  EXPECT_TRUE(g0->front()->as<IterDomain>()->isReduction());
  EXPECT_TRUE(g1->front()->as<IterDomain>()->isBroadcast());

  // merge {id0, id1} with {id2}
  // If this is done by cloning the first Val in each group then applying
  // IterDomain::merge, an error will be encountered due to the IterTypes being
  // incompatible. Each group contains a well-behaved IterType::Iteration ID
  // that should be used instead.
  ValGroup gmerge = merge(&graph, g0, g1);

  ASSERT_FALSE(gmerge->empty());
  ASSERT_TRUE(gmerge->front()->isA<IterDomain>());
  EXPECT_FALSE(gmerge->front()->as<IterDomain>()->isReduction());
}

} // namespace nvfuser
