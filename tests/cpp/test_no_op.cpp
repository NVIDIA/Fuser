// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using NoOpTest = NVFuserTest;

using testing::IsEmpty;
using testing::UnorderedElementsAre;

// Simple test case exercising the null scheduler path.
TEST_F(NoOpTest, FusionNullScheduler) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 1, 1});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.sum({0, 1, 2});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->schedulerType(), SchedulerType::ExprEval);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NoOpTest, FusionNullScheduler2) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({0, 1, 9223372036854775807L});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({0, 1, 9223372036854775807L}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->schedulerType(), SchedulerType::NoOp);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NoOpTest, FusionNullScheduler3) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = TensorViewBuilder().ndims(0).build();
  auto tv1 = TensorViewBuilder().ndims(0).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);
  at::Tensor t1 = at::randn({}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->schedulerType(), SchedulerType::NoOp);
  }
}

TEST_F(NoOpTest, FusionReducingZeroElements) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({0, 1, 9223372036854775807L});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({0, 1, 9223372036854775807L}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NoOpTest, View) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  // Verify the NoOp scheduler was kicked in.
  const std::vector<SegmentedGroup*>& groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();
  ASSERT_EQ(groups.size(), 1);
  SegmentedGroup* group = groups[0];
  EXPECT_EQ(group->schedulerType(), SchedulerType::ExprEval);
}

TEST_F(NoOpTest, ExpandedReduction) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .ndims(2)
                       .dtype(DataType::Float)
                       .contiguity({std::nullopt, std::nullopt})
                       .shape({2, 3})
                       .expanded({true, true})
                       .build();
  fusion->addInput(in);
  TensorView* out = sum(in, {0});
  out = segment_set(out);
  fusion->addOutput(out);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::ones({}).cuda().as_strided({2, 3}, {0, 0});
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::NoOp)));
  EXPECT_TRUE(runtime->executors().front()->isA<KernelExecutor>());
}

} // namespace nvfuser
