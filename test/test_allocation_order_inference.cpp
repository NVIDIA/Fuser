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

using LayoutInferenceTest = NVFuserTest;

TEST_F(LayoutInferenceTest, UnaryOpPropagation) {
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

} // namespace nvfuser
