// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

using NoOpTest = NVFuserTest;

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

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  std::cerr << cg_outputs[0].sizes() << std::endl;
  std::cerr << t1.sizes() << std::endl;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->heuristic(), ScheduleHeuristic::NoOp);
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

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->heuristic(), ScheduleHeuristic::NoOp);
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

  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0 + t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->heuristic(), ScheduleHeuristic::NoOp);
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

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NoOpTest, FusionEmpty) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({10, 10, 10});
  auto tv1 = makeConcreteTensor({10, 10, 10});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 10, 10}, options);
  at::Tensor t1 = at::randn({10, 10, 10}, options);

  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0, t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    EXPECT_EQ(group->heuristic(), ScheduleHeuristic::NoOp);
  }
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
  fusion->aliasOutputToInput(out, in, AliasType::PointerCast);

  FusionExecutor fe;
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  fe.compileFusion(fusion.get(), {in_tensor});
  at::Tensor out_tensor = fe.runFusion({in_tensor})[0];
  EXPECT_EQ(in_tensor.data_ptr<float>(), out_tensor.data_ptr<float>());
  testValidate(
      fusion.get(),
      {out_tensor},
      {in_tensor},
      {in_tensor.view({2, 12})},
      __LINE__,
      __FILE__);

  FusionExecutorCache fec(std::move(fusion));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  out_tensor = out_tensors[0];
  EXPECT_EQ(in_tensor.data_ptr<float>(), out_tensor.data_ptr<float>());

  const std::vector<SegmentedGroup*>& groups =
      fec.getMostRecentKernelRuntime()->fusionSegments()->groups();
  ASSERT_EQ(groups.size(), 1);
  SegmentedGroup* group = groups[0];

  EXPECT_EQ(group->heuristic(), ScheduleHeuristic::NoOp);
}

} // namespace nvfuser
