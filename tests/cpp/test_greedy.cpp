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

  // This set is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
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

  // This set is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
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

// Adding a conforming reshape to the ScanPad3D test. This fusion
// should not be segmented.
TEST_F(GreedySchedulerTest, ScanPad3DReshape1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{3, 4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  // This set is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
  auto tv1 = set(tv0);
  auto tv2 = cumsum(tv1, -1);
  auto tv3 =
      pad(tv2, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  auto tv4 = reshape(tv3, {3, 4, 129}, {4, 3, 129});
  auto tv5 = add(tv4, fusion.oneVal(DataType::Int));

  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Merging constrained and unconstrained IDs is not allowed
TEST_F(GreedySchedulerTest, ScanPad3DReshape2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{3, 4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  // This set is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
  auto tv1 = set(tv0);
  auto tv2 = cumsum(tv1, -1);
  auto tv3 =
      pad(tv2, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  auto tv4 = flatten(tv3);
  auto tv5 = add(tv4, fusion.oneVal(DataType::Int));

  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Similar to ScanPad3DReshape2, this fusion should be segmented since
// the constrained and unconstrained IDs would be merged if the
// reshape were propagated.
TEST_F(GreedySchedulerTest, ScanPad3DReshape3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{3, 4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  // The tv4 reshape, while it doesn't have straight producer-consumer
  // relationship with cumsum, would merge the scanned ID and
  // non-scanned ID, which violates one of the requrements of the
  // greedy scheduler. This fusion should be segmented.
  auto tv1 = set(tv0);
  // This op is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
  auto tv2 = add(tv1, fusion.oneVal(DataType::Int));
  auto tv3 = cumsum(tv2, -1);
  auto tv4 =
      pad(tv3, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  auto tv5 = reshape(tv1, {3, 4, 128}, {3, 4 * 128});
  auto tv6 = add(tv5, fusion.oneVal(DataType::Int));

  fusion.addOutput(tv4);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Having conflicting reshapes is not allowed (yet). This fusion
// should be segmented.
TEST_F(GreedySchedulerTest, ScanPad3DReshape4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{3, 4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  // This set is currently necessary to avoid out-of-bounds accesses
  // in scan (Issue #5080)
  auto tv1 = set(tv0);
  auto tv2 = cumsum(tv1, -1);
  auto tv3 =
      pad(tv2, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  auto tv4 = reshape(tv3, {3, 4, 129}, {4, 3, 129});
  auto tv5 = add(tv4, fusion.oneVal(DataType::Int));
  fusion.addOutput(tv5);

  // Non-matching another reshape. This should cause segmentation.
  auto tv6 = reshape(tv3, {3, 4, 129}, {2, 6, 129});
  auto tv7 = add(tv6, fusion.oneVal(DataType::Int));
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Based on SgLangMoETest.ComputeArgSort
TEST_F(GreedySchedulerTest, ArgsortArith) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{4, 128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = flatten(tv0);
  auto tv2 = argsort(tv1, -1, /*descending=*/true, /*stable=*/true);
  auto tv3 = mul(tv2, IrBuilder::create<Val>(100, DataType::Int));
  fusion.addOutput(tv3);

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

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, ArgsortNonLocalOutput) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, fusion.oneVal(DataType::Int));
  auto tv2 = argsort(tv1, -1, /*descending=*/true, /*stable=*/true);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, ScanNonLocalOutput) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{128};

  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, fusion.oneVal(DataType::Int));
  auto tv2 = scan(tv1, -1, BinaryOpType::Add);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, 100, shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Based on SgLangMoETest.ComputeArgSort
TEST_F(GreedySchedulerTest, Scatter) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t size = 128;

  auto tv0 = makeContigConcreteTensor({size}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = zeros({IrBuilder::create<Val>(size)}, DataType::Int);
  auto tv2 = arange(
      fusion.zeroVal(DataType::Int),
      IrBuilder::create<Val>(size, DataType::Int),
      DataType::Int);
  auto tv3 = scatter(tv1, 0, tv0, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(size, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, TopK) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, fusion.oneVal(DataType::Int));
  // Create topk operation along dimension 1, k=3, largest=true, sorted=true
  // Create k as a constant Val (not a fusion input)
  auto k_val = IrBuilder::create<Val>(3L, DataType::Int);
  auto topk_result = topk(tv1, k_val, 1, /*largest=*/true, /*sorted=*/true);
  auto tv_values = topk_result.values;
  auto tv_indices = topk_result.indices;
  auto tv_values_out = add(tv_values, fusion.oneVal(DataType::Int));
  auto tv_indices_out = add(tv_indices, fusion.oneVal(DataType::Int));
  fusion.addOutput(tv_values_out);
  fusion.addOutput(tv_indices_out);

  at::Tensor t0 = at::randint(
      -100,
      100,
      shape,
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

} // namespace nvfuser
