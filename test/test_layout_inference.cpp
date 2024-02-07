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
#include <optimization/layout_inference.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

class LayoutInferenceTest : public NVFuserTest {};

// A global->global copy kernel converting NCHW memory format into NHWC, with a
// 4d allocation domain in output.
TEST_F(AllocationDomainTest, NCHW4d_To_NHWC4d) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::vector<IterDomain*> tv1_nhwc = {
      tv1->axis(0), tv1->axis(2), tv1->axis(3), tv1->axis(1)};
  tv1->setAllocationDomain(tv1_nhwc, true);

  auto updated_layout = inferenceMemoryFormat(fusion);

  // auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // int n = 31, h = 64, w = 103, c = 21;

  // at::Tensor t0 = at::randn({n, c, h, w}, options);

  // FusionExecutor fe;
  // fe.compileFusion(fusion_ptr.get(), {t0});

  // auto cg_outputs = fe.runFusion({t0});

  // ASSERT_TRUE(cg_outputs[0].is_contiguous(at::MemoryFormat::ChannelsLast));

  // testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
