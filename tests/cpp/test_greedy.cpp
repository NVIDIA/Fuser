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

// Similar to ArgsortPadScan, this is segmented due to the exactness
// requirement of TopKOp
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

  // TODO: Extend the greedy scheduler to accept the fusion without
  // segmentation
  EXPECT_TRUE(executor_cache.getMostRecentKernelRuntime()->isSegmented());
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
}
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
  fusion.addOutput(t20);
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

} // namespace nvfuser
