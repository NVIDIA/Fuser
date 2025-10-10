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
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
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

class GreedySchedulerTestConstraintSize
    : public GreedySchedulerTest,
      public ::testing::WithParamInterface<int64_t> {
 protected:
  void SetUp() override {
    GreedySchedulerTest::SetUp();
    size = GetParam();
  }

 protected:
  int64_t size = 0;
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

TEST_F(GreedySchedulerTest, TopKPad) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, fusion.oneVal(DataType::Int));
  auto k_val = IrBuilder::create<Val>(3L, DataType::Int);
  auto topk_result = topk(tv1, k_val, 1, /*largest=*/true, /*sorted=*/true);
  auto tv_indices = topk_result.indices;
  auto tv_indices_padded =
      pad(tv_indices,
          {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});
  fusion.addOutput(tv_indices_padded);

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

// Extracted from test_moe.py
// clang-format off
/*
inputs:
  T12_g___bfloat[iS32{2048}, iS33{128}, rS34{5120}] __bfloat
outputs:
  T20_g_int64_t[iS55{2048}, bS56{1}] int64_t
  T35_g___bfloat[iS91{2048}, bS92{1}] __bfloat


( T19_g___bfloat[iS53{2048}, bS54{1}], T20_g_int64_t[iS55{2048}, bS56{1}] ) = topk( T12_g___bfloat[iS32{2048}, iS33{128}, rS34{5120}], 1, dim = 1, largest = True, sorted = True )
(11)
T24_l_float[iS66{2048}, bS67{1}]
   = __bfloat2float(T19_g___bfloat[iS53{2048}, bS54{1}]);
(15)
T25_l_float[iS68{2048}, bS69{1}]
   = -T24_l_float[iS66{2048}, bS67{1}];
(16)
T26_l_float[iS70{2048}, bS71{1}]
   = expf(T25_l_float[iS68{2048}, bS69{1}]);
(17)
T27_l_float[iS72{2048}, bS73{1}]
   = double(1)
   + T26_l_float[iS70{2048}, bS71{1}];
(18)
T28_l_float[iS74{2048}, bS75{1}]
   = reciprocal(T27_l_float[iS72{2048}, bS73{1}]);
(19)
T29_g___bfloat[iS76{2048}, bS77{1}]
   = __float2bfloat(T28_l_float[iS74{2048}, bS75{1}]);
(20)
T35_g___bfloat[iS91{2048}, bS92{1}]
   = Set( T29_g___bfloat[iS76{2048}, bS77{1}], cache_op=Streaming )
(26)
T8_l_int[iS20{2048}, iS21{128}]
   = full({2048, 128}, 0);
T30_l_int[iS80{2048}, bS81{1}]
   = scatter(in = T8_l_int[iS20{2048}, iS21{128}], dim = 1, src = 1, idx = T20_l_int64_t[iS55{2048}, bS56{1}] )
*/
// clang-format on
TEST_F(GreedySchedulerTest, TopKLlama4) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2048, 128};
  auto tv0 = makeContigConcreteTensor(shape, DataType::BFloat16);
  fusion.addInput(tv0);

  auto topk_result = topk(
      tv0, fusion.oneVal(DataType::Int), 1, /*largest=*/true, /*sorted=*/true);
  auto t19 = topk_result.values;
  auto t20 = topk_result.indices;
  auto t8 = zeros(
      {IrBuilder::create<Val>(shape[0]), IrBuilder::create<Val>(shape[1])},
      DataType::Int);
  auto t30 = scatter(t8, 1, t20, fusion.oneVal(DataType::Int));
  fusion.addOutput(t30);
  auto t24 = castOp(DataType::Float, t19);
  auto t25 = neg(t24);
  auto t26 = exp(t25);
  auto t27 = add(fusion.oneVal(DataType::Double), t26);
  auto t28 = reciprocal(t27);
  auto t29 = castOp(DataType::BFloat16, t28);
  auto t35 = set(t29);
  fusion.addOutput(t35);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      testing::UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::Greedy)));
}

TEST_F(GreedySchedulerTest, ConstrainedIDAndBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{4, 128};

  auto tv0 = makeContigConcreteTensor({shape[0]});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv2, tv1);
  auto tv4 = flatten(tv3);
  auto tv5 = argsort(tv4, -1, /*descending=*/true, /*stable=*/true);
  auto tv6 = mul(tv5, IrBuilder::create<Val>(100));
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({shape[0]}, options);
  auto t1 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, ConstrainedIDAndSqueeze) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{4, 128, 1};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = squeeze(tv0, {-1});
  auto tv2 = argsort(tv1, -1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, UnconstrainedIDAndBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{4, 8, 128};

  auto tv0 = makeContigConcreteTensor({shape[0], shape[2]});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true, false});
  auto tv3 = add(tv2, tv1);
  auto tv4 = flatten(tv3, 0, 1);
  auto tv5 = argsort(tv4, -1, /*descending=*/true, /*stable=*/true);
  auto tv6 = mul(tv5, IrBuilder::create<Val>(100));
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({shape[0], shape[2]}, options);
  auto t1 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_F(GreedySchedulerTest, UnconstrainedIDAndSqueeze) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{1, 128};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = argsort(tv0, -1);
  auto tv2 = squeeze(tv1, {0});
  auto tv3 = cumsum(tv2, -1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Test translating scatter + sum to scatter-accumulate
TEST_F(GreedySchedulerTest, TranslateScatterAndReductionToScatterAccumulate) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 128;
  const int64_t n = 1024;

  // Each element is [0, m).
  auto tv0 = makeContigConcreteTensor({n, 1}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros(
      {IrBuilder::create<Val>(n), IrBuilder::create<Val>(m)}, DataType::Int);
  auto tv3 = scatter(tv2, 1, tv1, fusion.oneVal(DataType::Int));
  auto tv4 = sum(tv3, {0});
  fusion.addOutput(tv4);

  // Just to force the bottom-up segmenter to kick in and test if the
  // above ops can be fused
  auto tv5 = segment_set(tv0);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, m, {n, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // There must not be a reduction segment
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      testing::UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::Greedy)));
}

TEST_F(
    GreedySchedulerTest,
    TranslateScatterAndReductionToScatterAccumulateWithCast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 128;
  const int64_t n = 1024;

  // Each element is [0, m).
  auto tv0 = makeContigConcreteTensor({n, 1}, DataType::Int32);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros(
      {IrBuilder::create<Val>(n), IrBuilder::create<Val>(m)}, DataType::Int32);
  auto tv3 = scatter(tv2, 1, tv1, fusion.oneVal(DataType::Int32));
  auto tv4 = castOp(DataType::Int, tv3);
  auto tv5 = sum(tv4, {0});
  fusion.addOutput(tv5);

  // Just to force the bottom-up segmenter to kick in and test if the
  // above ops can be fused
  auto tv6 = segment_set(tv0);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0);
  auto t0 = at::randint(0, m, {n, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  // There must not be a reduction segment
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      testing::UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::Greedy)));
}

TEST_P(GreedySchedulerTestConstraintSize, ArgsortLargeConstrainedIDs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, size};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = argsort(tv0, -1);
  auto tv2 = add(tv1, fusion.oneVal());
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_P(GreedySchedulerTestConstraintSize, ScanLargeConstrainedIDs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, size};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = scan(tv0, -1, BinaryOpType::Add);
  auto tv2 = add(tv1, fusion.oneVal());
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

TEST_P(GreedySchedulerTestConstraintSize, ScatterLargeConstrainedIDs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {128};
  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = makeContigConcreteTensor({size}, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 =
      scatter(tv0, 0, tv1, fusion.oneVal(DataType::Int), BinaryOpType::Add);
  auto tv3 = add(tv2, fusion.oneVal());
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::zeros({128}, options);
  auto t1 = at::randint(0, 128, {size}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

// Pattern appearing in test_moe.py
TEST_P(GreedySchedulerTestConstraintSize, ArgsortArgsort) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({size}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = argsort(tv0, 0, /*descending=*/false, /*stable=*/true);
  auto tv2 = argsort(tv1, 0, /*descending=*/false, /*stable=*/true);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(size, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
}

INSTANTIATE_TEST_SUITE_P(
    ,
    GreedySchedulerTestConstraintSize,
    testing::Values(1024, 2048, 4096),
    [](const testing::TestParamInfo<int64_t>& info) {
      std::ostringstream os;
      os << info.param;
      return os.str();
    });

class GreedySchedulerTestShmemSize : public GreedySchedulerTest,
                                     public ::testing::WithParamInterface<int> {
};

// Simplified version of
// ArgsortParameterizedWithBlockandBatch.SharedMemoryRequirement. The
// test may be segmented but should not fail as long as the
// expectation of the shared memory usage is accurate.
TEST_P(GreedySchedulerTestShmemSize, Argsort) {
  DisableOptionsGuard disable_options_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  const auto size = GetParam();

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  DataType dtype = DataType::Int;
  DataType dtype_extra = DataType::Float;

  std::vector<int64_t> shape = {size};

  auto tv0 = makeContigConcreteTensor(shape, dtype);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, 0);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Duplicate the above call but should not change the usage as it's
  // the same template instantiation
  auto tv4 = set(tv0);
  auto tv5 = argsort(tv4, 0);
  auto tv6 = set(tv5);
  fusion.addOutput(tv6);

  // Create a different instantiation
  auto tv7 = castOp(dtype_extra, tv0);
  auto tv8 = argsort(tv7, 0);
  auto tv9 = set(tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, shape[0], shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

// Simplified version of
// TopKParameterizedWithBlockandBatch.SharedMemoryRequirement. The
// test may be segmented but should not fail as long as the
// expectation of the shared memory usage is accurate.
TEST_P(GreedySchedulerTestShmemSize, TopK) {
  DisableOptionsGuard disable_options_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  const auto size = GetParam();

  // topk doesn't support batching, so the maximum is 1024
  if (size > 1024) {
    return;
  }

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  DataType dtype = DataType::Int;
  DataType dtype_extra = DataType::Float;

  std::vector<int64_t> shape = {size};

  auto tv0 = makeContigConcreteTensor(shape, dtype);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = topk(tv1, fusion.oneVal(DataType::Int), 0).values;
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Duplicate the above call but should not change the usage as it's
  // the same template instantiation
  auto tv4 = set(tv0);
  auto tv5 = topk(tv4, fusion.oneVal(DataType::Int), 0).values;
  auto tv6 = set(tv5);
  fusion.addOutput(tv6);

  // Create a different instantiation
  auto tv7 = castOp(dtype_extra, tv0);
  auto tv8 = topk(tv7, fusion.oneVal(DataType::Int), 0).values;
  auto tv9 = set(tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, shape[0], shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

// Simplified version of
// ScanParameterizedWithBlockandBatch.SharedMemoryRequirement. The
// test may be segmented but should not fail as long as the
// expectation of the shared memory usage is accurate.
TEST_P(GreedySchedulerTestShmemSize, Scan) {
  DisableOptionsGuard disable_options_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  const auto size = GetParam();

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  DataType dtype = DataType::Int;
  DataType dtype_extra = DataType::Float;

  std::vector<int64_t> shape = {size};

  auto tv0 = makeContigConcreteTensor(shape, dtype);
  fusion.addInput(tv0);

  auto tv1 = cumsum(tv0, 0);
  fusion.addOutput(tv1);

  // Duplicate the above call but should not change the usage as it's
  // the same template instantiation
  auto tv2 = cumsum(tv0, 0);
  fusion.addOutput(tv2);

  // Create a different instantiation
  auto tv3 = castOp(dtype_extra, tv0);
  auto tv4 = cumsum(tv3, 0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, shape[0], shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    GreedySchedulerTestShmemSize,
    testing::Values(128, 256, 512, 1024, 2048, 4096),
    [](const auto& info) {
      std::ostringstream os;
      os << info.param;
      return os.str();
    });

TEST_F(GreedySchedulerTest, TMP) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->setMemoryType(MemoryType::Shared);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->setMemoryType(MemoryType::Shared);
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({100}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
