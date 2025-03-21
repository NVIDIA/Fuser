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

TEST_F(RuntimeTest, ClearGmemBetweenSegments) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> input_shape{32, 64, 8, 128};
  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0)); // Group 0
  auto tv2 = sum(tv1, {0}); // Group 0
  auto tv3 = sum(tv2, {-1}); // Group 1
  auto output = sum(tv3, {0}); // Group 2
  fusion->addOutput(output);

  resetPeakMemoryStats(0);
  ASSERT_EQ(maxMemoryAllocated(0), 0) << "No tensors are allocated so far.";

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({at_x});
  const int64_t max_memory_allocated = maxMemoryAllocated(0);

  auto runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 3)
      << "segmentation didn't happen as expected";
  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);

  EXPECT_EQ(
      max_memory_allocated,
      (32 * 64 * 8 * 128 + 64 * 8 * 128 + 64 * 8) * sizeof(double))
      << "tv0 (32 * 64 * 8 * 128) outlived the execution, so it contributes "
         "to the peak memory. tv1 was never allocated because it's internal "
         "to group 0. tv2 (64 * 8 * 128) and tv3 (64 * 8) were both alive "
         "when executing group 1.";
}

} // namespace nvfuser
