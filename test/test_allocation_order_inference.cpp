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
