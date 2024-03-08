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

  auto updated_layout = preseg_passes::inferenceAllocationOrder(&fusion);
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

  const auto inferred_layout = preseg_passes::inferenceAllocationOrder(&fusion);
  EXPECT_THAT(inferred_layout.at(tv1), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, BinaryOpPropagation) {
  {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
    fusion.addInput(tv0);
    auto s1 = IrBuilder::create<Val>(1L);
    // Testing propagation between tensor and a scalar
    auto tv2 = add(tv0, s1);
    fusion.addOutput(tv2);
    // Testing propagation between tensor and a scalar
    auto tv3 = add(s1, tv0);
    fusion.addOutput(tv3);
    auto s4 = IrBuilder::create<Val>(3L);
    // binary op between scalars
    auto s5 = add(s1, s4);
    auto tv6 = add(tv0, s5);
    fusion.addOutput(tv6);
    auto tv7 = add(s5, tv0);
    fusion.addOutput(tv7);

    std::vector<IterDomain*> tv0_nhwc = {
        tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
    tv0->setAllocationDomain(tv0_nhwc, true);

    const auto inferred_layout =
        preseg_passes::inferenceAllocationOrder(&fusion);
    EXPECT_THAT(inferred_layout.at(tv2), ElementsAre(0, 2, 3, 1));
    EXPECT_THAT(inferred_layout.at(tv3), ElementsAre(0, 2, 3, 1));
    EXPECT_THAT(inferred_layout.at(tv6), ElementsAre(0, 2, 3, 1));
    EXPECT_THAT(inferred_layout.at(tv7), ElementsAre(0, 2, 3, 1));
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

    const auto inferred_layout =
        preseg_passes::inferenceAllocationOrder(&fusion);
    EXPECT_THAT(inferred_layout.at(tv2), ElementsAre(1, 0, 2, 3));
    EXPECT_THAT(inferred_layout.at(tv3), ElementsAre(1, 0, 2, 3));
  }
  {
    auto fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    // Testing propagation between two tensors
    // tv0 and tv1 has the same number of non-broadcast iter domains, so lhs
    // operand would propagate its allocation order.
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

    const auto inferred_layout =
        preseg_passes::inferenceAllocationOrder(&fusion);
    EXPECT_THAT(inferred_layout.at(tv2), ElementsAre(0, 2, 1, 3));
    EXPECT_THAT(inferred_layout.at(tv3), ElementsAre(1, 0, 2, 3));
  }
}

TEST_F(AllocationOrderInferenceTest, TensorFactoryBinaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, 1});
  fusion.addInput(tv0);
  auto s1 = IrBuilder::create<Val>(16L);
  auto s2 = IrBuilder::create<Val>(32L);
  auto fill_value = IrBuilder::create<Val>(1.0);
  // factory method
  auto tv1 = full({s1, s2}, fill_value, DataType::Float);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);
  auto tv3 = add(tv1, tv0);
  fusion.addOutput(tv3);

  std::vector<IterDomain*> tv0_c_last = {tv0->axis(1), tv0->axis(0)};
  tv0->setAllocationDomain(tv0_c_last, true);

  // tv1 is tensor created by factory method, its layout shouldn't be propagated
  // to output
  std::vector<IterDomain*> tv1_c_last = {tv1->axis(0), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_c_last, true);

  const auto inferred_layout = preseg_passes::inferenceAllocationOrder(&fusion);
  EXPECT_THAT(inferred_layout.at(tv2), ElementsAre(1, 0));
  EXPECT_THAT(inferred_layout.at(tv3), ElementsAre(1, 0));
}

TEST_F(AllocationOrderInferenceTest, TernaryOpPropagation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor({-1, -1, -1, -1});
  fusion.addInput(tv2);
  auto tv3 = gt(tv0, IrBuilder::create<Val>(0.0));
  auto tv4 = where(tv3, tv1, tv2);
  fusion.addOutput(tv4);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);
  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);
  std::vector<IterDomain*> tv2_nhwc = {
      tv2->axis(0), tv2->axis(2), tv2->axis(3), tv2->axis(1)};
  tv2->setAllocationDomain(tv2_nhwc, true);

  const auto inferred_layout =
      preseg_passes::inferenceAllocationOrder(&fusion);
  EXPECT_THAT(inferred_layout.at(tv3), ElementsAre(0, 2, 3, 1));
  EXPECT_THAT(inferred_layout.at(tv4), ElementsAre(0, 2, 3, 1));
}

TEST_F(AllocationOrderInferenceTest, EnableInRuntime) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(4);
  fusion->addInput(tv0);
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);

  std::vector<IterDomain*> tv0_nhwc = {
      tv0->axis(0), tv0->axis(2), tv0->axis(3), tv0->axis(1)};
  tv0->setAllocationDomain(tv0_nhwc, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor in_tensor = at::randn({2, 4, 8, 8}, options);
  at::Tensor in_nhwc =
      in_tensor.as_strided({2, 4, 8, 8}, {4 * 8 * 8, 1, 4 * 8, 4});
  FusionExecutorCache fec(std::move(fusion));

  auto cg_outputs = fec.runFusionWithInputs({in_nhwc});
  auto ref_out = in_nhwc.relu();

  EXPECT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));
  EXPECT_TRUE(ref_out.allclose(cg_outputs[0]));
}

} // namespace nvfuser
