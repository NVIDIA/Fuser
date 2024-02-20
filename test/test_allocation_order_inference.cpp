// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

using testing::_;
using testing::ElementsAre;

using AllocationOrderInferenceTest = NVFuserTest;

TEST_F(AllocationOrderInferenceTest, BroadcastOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 =
      broadcast(tv0, {true, false, false, true, false, true, false, true});
  fusion.addOutput(tv2); // (0, 2, 3, 1) -> (0, 3, 5, 7, 1, 4, 6, 2)
  auto tv3 = broadcast(tv1, {true, false, true, true});
  fusion.addOutput(tv3);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  auto updated_layout = inferenceAllocationOrder(&fusion);
  EXPECT_THAT(updated_layout[tv2], ElementsAre(0, 3, 5, 7, 1, 4, 6, 2));
  EXPECT_THAT(updated_layout[tv3], ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, UnaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = relu(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  auto updated_layout = preseg_passes::inferenceAllocationOrder(&fusion);
  EXPECT_THAT(updated_layout[tv1], ElementsAre(0, 2, 3, 1));
}

} // namespace nvfuser
