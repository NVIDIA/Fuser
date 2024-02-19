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

  auto updated_layout = inferenceAllocationOrder(&fusion);
  EXPECT_THAT(updated_layout[tv1], ElementsAre(0, 2, 3, 1));
}

TEST_F(LayoutInferenceTest, BinaryOpPropagation) {
  {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // Testing propagation between tensor and a scalar
    auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
    fusion.addInput(tv0);
    auto s1 = IrBuilder::create<Val>(0L);
    auto tv2 = add(tv0, s1);
    fusion.addOutput(tv2);
    auto tv3 = add(s1, tv0);
    fusion.addOutput(tv3);

    std::vector<IterDomain*> tv0_nhwc = {
        tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
    tv0->setAllocationDomain(tv0_nhwc, true);

    auto updated_layout = inferenceAllocationOrder(&fusion);
    EXPECT_THAT(updated_layout[tv2], ElementsAre(0, 2, 3, 1));
    EXPECT_THAT(updated_layout[tv3], ElementsAre(0, 2, 3, 1));
  }
  {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // Testing propagation between two tensors
    auto tv0 = makeSymbolicTensor({-1, 1, 1, -1});
    fusion.addInput(tv0);
    // tv1 has more non-broadcast iter domain and dominates output memory format
    auto tv1 = makeSymbolicTensor({-1, -1, -1, 1});
    fusion.addInput(tv1);
    auto tv2 = add(tv0, tv1);
    fusion.addOutput(tv2);
    auto tv3 = add(tv1, tv0);
    fusion.addOutput(tv3);

    std::vector<IterDomain*> tv0_format = {
        tv0->axis(0), tv0->axis(2), tv0->axis(1), tv0->axis(3)};
    tv0->setAllocationDomain(tv0_format, true);
    std::vector<IterDomain*> tv1_format = {
        tv1->axis(1), tv1->axis(0), tv1->axis(2), tv1->axis(3)};
    tv1->setAllocationDomain(tv1_format, true);

    auto updated_layout = inferenceAllocationOrder(&fusion);
    EXPECT_THAT(updated_layout[tv2], ElementsAre(1, 0, 2, 3));
    EXPECT_THAT(updated_layout[tv3], ElementsAre(1, 0, 2, 3));
  }
  {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // Testing propagation between two tensors
    // tv0 and tv1 has the same number of non-broadcast iter domains, so lhs operand would propagate its allocation order.
    auto tv0 = makeSymbolicTensor({-1, -1, 1, 1});
    fusion.addInput(tv0);
    auto tv1 = makeSymbolicTensor({-1, -1, 1, 1});
    fusion.addInput(tv1);
    // tv2 should have allocation order from tv0
    auto tv2 = add(tv0, tv1);
    fusion.addOutput(tv2);
    // tv3 should have allocation order from tv1
    auto tv3 = add(tv1, tv0);
    fusion.addOutput(tv3);

    std::vector<IterDomain*> tv0_format = {
        tv0->axis(0), tv0->axis(2), tv0->axis(1), tv0->axis(3)};
    tv0->setAllocationDomain(tv0_format, true);
    std::vector<IterDomain*> tv1_format = {
        tv1->axis(1), tv1->axis(0), tv1->axis(2), tv1->axis(3)};
    tv1->setAllocationDomain(tv1_format, true);

    auto updated_layout = inferenceAllocationOrder(&fusion);
    EXPECT_THAT(updated_layout[tv2], ElementsAre(0, 2, 1, 3));
    EXPECT_THAT(updated_layout[tv3], ElementsAre(1, 0, 2, 3));
  }
}

} // namespace nvfuser
