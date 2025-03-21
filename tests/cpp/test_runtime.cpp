// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// delete intermediate tensors between segments to reduce memory usage of large
// segmented graphs
#include <gtest/gtest.h>

#include <fusion.h>
#include <fusion_guard.h>
#include <ops/arith.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using RuntimeTest = NVFuserTest;

TEST_F(RuntimeTest, FusionClearGmemBetweenSegments_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> input_shape{32, 64, 8, 128};
  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0}); // Group 0
  auto tv3 = sum(tv2, {-1}); // Group 1
  auto output = sum(tv3, {0}); // Group 2
  fusion->addOutput(output);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({at_x});
  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  auto args_num = optimized_fusion->getArgsNumAfterSegmentRuns();

  NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen");
  NVF_CHECK(
      optimized_fusion->fusionSegments()->groups().size() == 3,
      "segmentation didn't happen as expected");
  // group-0: tv1 -> tv2
  // group-1: tv2 -> tv3
  // group-2: tv3 -> tv4
  // -----------without args erase------------------------
  // after group-0, args: {t0, 32, 64, 8, 128, t2}
  // after group-1, args: {t0, 32, 64, 8, 128, t2, t3}
  // after group-2, args: {t0, 32, 64, 8, 128, t2, t3, t4}
  // -----------with args erase---------------------------
  // after group-0, args: {t0, 32, 64, 8, 128, t2}
  // after group-1, args: {t0, 32, 64, 8, 128, t3} (t2 is erased)
  // after group-2, args: {t0, 32, 64, 8, 128, t4} (t3 is erased)
  NVF_CHECK(
      args_num[1] == args_num[0] && args_num[2] == args_num[0],
      "unused intermediate args should be deleted");
  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);
}

} // namespace nvfuser
