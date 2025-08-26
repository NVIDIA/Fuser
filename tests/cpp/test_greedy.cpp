// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <gtest/gtest.h>

namespace nvfuser {

class GreedySchedulerTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
    EnableOptionsGuard::getCurOptions().set(EnableOption::GreedyScheduler);
  }
};

// Scan, followed by pad. Same fusion as
// SgLangMoETest.ComputeExpertOffsets
TEST_F(GreedySchedulerTest, ScanPad1D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = cumsum(tv1, -1);
  auto tv3 =
      pad(tv2, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, ScanPad3D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{3, 4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = cumsum(tv1, -1);
  auto tv3 =
      pad(tv2, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Based on SgLangMoETest.ComputeArgSort
TEST_F(GreedySchedulerTest, ArgsortArith) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = argsort(tv0, -1, /*descending=*/true, /*stable=*/true);
  auto tv2 = mul(tv1, IrBuilder::create<Val>(100, DataType::Int));
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Currently, argsort requires TIDx to be exact, so this fusion is
// currently segmented.
TEST_F(GreedySchedulerTest, ArgsortPadScan) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = argsort(tv0, -1, /*descending=*/true, /*stable=*/true);
  auto tv2 =
      pad(tv1, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  auto tv3 = cumsum(tv2, -1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // TODO: Extend the greedy scheduler to accept the fusion without
  // segmentation
  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

} // namespace nvfuser
