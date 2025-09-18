// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

namespace nvfuser {

using ScanTest = NVFuserTest;

// Basic functionality test for scan with Add operation (cumsum)
TEST_F(ScanTest, BasicScanAdd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Add).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Max operation (cummax)
TEST_F(ScanTest, BasicScanMax) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Max).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Min operation (cummin)
TEST_F(ScanTest, BasicScanMin) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Min).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Basic functionality test for scan with Mul operation (cumprod)
TEST_F(ScanTest, BasicScanMul) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Mul).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test different tensor shapes and scan dimensions
TEST_F(ScanTest, ScanDifferentDimensions) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({2, 4, 6});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 4, 6}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test 1D tensor scan
TEST_F(ScanTest, Scan1D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add).inclusive;
  fusion.addOutput(tv_result);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({10}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// NOTE: Complex arithmetic + scan fusion is limited by ExprEval scheduler's
// single expression constraint. For complex fusions with ScanOp, nvFuser would
// need fusion segmentation or different scheduler approaches.

// Test simple ScanOp with just one additional operation
TEST_F(ScanTest, ScanWithSimpleArithmetic) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Single arithmetic operation before scan
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));

  // Scan operation
  auto tv2 = scan(tv1, /*dim=*/1, BinaryOpType::Add).inclusive;

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

// Test ScanOp with multiple arithmetic operations - investigating complex
// fusion behavior
TEST_F(ScanTest, ScanWithArithmeticOps) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);

  // Multiple arithmetic operations
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = sub(tv2, IrBuilder::create<Val>(0.5));

  // Scan operation
  auto tv4 = scan(tv3, /*dim=*/1, BinaryOpType::Add).inclusive;

  // Additional operation after scan
  auto tv5 = div(tv4, IrBuilder::create<Val>(3.0));

  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({4, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({input});

  testValidate(executor_cache.fusion(), outputs, {input}, __LINE__, __FILE__);
}

//============================================================================
// KernelExecutor Tests - Testing Code Generation Path
//============================================================================

// Test class for KernelExecutor-based scan tests
class ScanCodeGenTest : public NVFuserTest,
                        public ::testing::WithParamInterface<DataType> {
 protected:
  void runBasicCodeGenTest(DataType data_type) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Create input tensor [4, 8] with specified data type
    std::vector<int64_t> shape = {4, 8};
    auto tv0 = makeContigConcreteTensor(shape, data_type);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    // Create scan operation along dimension 1
    auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Add).inclusive;
    auto tv_output = set(tv_result);
    fusion.addOutput(tv_output);

    // Parallelization strategy - all tensors get same parallelization
    for (auto tv : {tv1, tv_result, tv_output}) {
      tv->axis(0)->parallelize(ParallelType::BIDx);
      tv->axis(1)->parallelize(ParallelType::TIDx);
    }

    at::Tensor input =
        at::randint(
            -100,
            100,
            {4, 8},
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0))
            .to(data_type_to_aten(data_type));

    // Execute the fusion using KernelExecutor (tests code generation path)
    KernelExecutor ke;
    ke.compile(&fusion, {input});
    auto outputs = ke.run({input});

    // Validate against PyTorch reference
    testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
  }
};

TEST_P(ScanCodeGenTest, ParameterizedCodeGenExecution) {
  runBasicCodeGenTest(GetParam());
}

// Instantiate parameterized tests for different data types
INSTANTIATE_TEST_SUITE_P(
    ScanTest,
    ScanCodeGenTest,
    ::testing::Values(DataType::Float, DataType::Double, DataType::Int),
    [](const ::testing::TestParamInfo<DataType>& info) {
      auto data_type = info.param;
      if (data_type == DataType::Float)
        return std::string("Float");
      if (data_type == DataType::Double)
        return std::string("Double");
      if (data_type == DataType::Int)
        return std::string("Int");
      return std::string("Unknown");
    });

// Testing scan with KernelExecutor for Add operation
TEST_F(ScanTest, KernelExecutorScanAdd) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1, Add operation
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Add).inclusive;
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

// Testing scan with KernelExecutor for Max operation
TEST_F(ScanTest, KernelExecutorScanMax) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1, Max operation
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Max).inclusive;
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

// Testing scan with KernelExecutor for Min operation
TEST_F(ScanTest, KernelExecutorScanMin) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1, Min operation
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Min).inclusive;
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

// Testing scan with KernelExecutor for Mul operation
TEST_F(ScanTest, KernelExecutorScanMul) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1, Mul operation
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Mul).inclusive;
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

// Testing multiple scan operations with KernelExecutor
TEST_F(ScanTest, KernelExecutorMultipleScan) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [2, 6]
  std::vector<int64_t> shape = {2, 6};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // First scan operation (Add)
  auto tv_scan1 = scan(tv1, /*dim=*/1, BinaryOpType::Add).inclusive;

  // Add operation between scans
  auto tv_add = add(tv_scan1, IrBuilder::create<Val>(1.0));

  // Second scan operation (Max)
  auto tv_scan2 = scan(tv_add, /*dim=*/1, BinaryOpType::Max).inclusive;

  auto tv_output = set(tv_scan2);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_scan1, tv_add, tv_scan2, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({2, 6}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

TEST_F(ScanTest, Predication) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {100};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = scan(tv0, -1, BinaryOpType::Add).inclusive;
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  // Non-divisible split. 128 threads will be launched. The last 28
  // threads need to be predicated out.
  for (auto tv : {tv1, tv2}) {
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::TIDy);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Simple test case for defining a scan
TEST_F(ScanTest, Concrete1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(tv0, /*dim=*/-1, /*discount_factor=*/nullptr).inclusive;

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScanTest, DiscountFactorScalar1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);

  auto tv1 =
      prefixSum(
          tv0,
          /*dim=*/-1,
          /*discount_factor=*/IrBuilder::create<Val>(0.8, DataType::Float))
          .inclusive;

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScanTest, DiscountFactorTensor1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);
  auto tv1 = makeConcreteTensor({32});
  fusion->addInput(tv1);

  auto tv2 = prefixSum(
                 tv0,
                 /*dim=*/-1,
                 /*discount_factor=*/tv1)
                 .inclusive;

  fusion->addOutput(tv2);

  tv0->cacheAfter();
  tv2->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32}, options).exp();

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1});

  auto cg_outputs = ke.run({t0, t1});

  testValidate(fusion.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Simple test case for defining a scan
TEST_F(ScanTest, Concrete2D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({4, 32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(tv0, /*dim=*/0, /*discount_factor=*/nullptr).inclusive;

  fusion->addOutput(tv1);

  tv0->cacheAfter();
  tv1->cacheBefore();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// This is similar to what's needed for serial online softmax
TEST_F(ScanTest, OnlineSoftmax) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = makeSymbolicTensor(1);
  fusion->addInput(x);

  int64_t scan_dim = 0;

  // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
  //
  // Given x[i] for i=0 .. N-1:
  //
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   for j = 0 .. N-1
  //     m[j] = max(m[j-1], x[j])
  //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
  //
  // Final denominator is d[N-1]

  auto* neg_infty = IrBuilder::create<Val>(
      -std::numeric_limits<double>::infinity(), DataType::Double);
  ScanResult max_scan_result = scan(
      set(x),
      scan_dim,
      BinaryOpType::Max,
      /*return_exclusive=*/true,
      /*discount_factor=*/nullptr,
      /*init=*/neg_infty); // max x[j] over j = 0 .. i
  TensorView* m = max_scan_result.inclusive;
  TensorView* m_prev = max_scan_result.exclusive;
  // normalize by running max and exponentiate
  TensorView* exp_x_m = exp(sub(x, m));
  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  TensorView* discount = exp(sub(m_prev, m));

  auto denoms = prefixSum(exp_x_m, scan_dim, discount).inclusive;

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion->zeroVal(DataType::Float),
      denoms);

  auto full_max = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/neg_infty,
      m);

  auto max_bcast = broadcast(full_max, {true});
  auto norm_factor_bcast = broadcast(norm_factor, {true});
  // Recompute numerator
  auto numer = exp(sub(set(x), max_bcast));

  auto result = div(numer, norm_factor_bcast);

  fusion->addOutput(result);

  // Don't cache inputs for this fusion because we will need to recompute
  // exp((x-m)) using the final max in a separate loop so caching would mean
  // we'd need to hold the whole tensor in registers. Instead, we manually call
  // set(x) twice in the definition above to give us two separate caches.
  scheduler_utils::cacheAndForkOutputs(fusion.get(), /*unroll=*/true);

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (TensorView* tv : {m, m_prev, denoms}) {
    for (IterDomain* id : tv->getLoopDomain()) {
      uninlineable_ids.insert(id);
    }
  }

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (TensorView* tv : {m, m_prev, denoms}) {
    tv->computeWith(-1);
    for (Val* v : tv->definition()->inputs()) {
      // By using `uninlineable_ids` above, we prevent producers of scan from
      // inlining with the ScanOp past the scan dim, even though this is
      // desired. Here we do this inlining manually instead.
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  auto ref = at::softmax(t0, 0);
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
      << " returned " << cg_outputs[0].as<at::Tensor>().item()
      << " but expected " << ref.item();

  // Test automatic evaluation also
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

//
TEST_F(ScanTest, OnlineSoftmaxOuter) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // TODO: Allow outer dim to be symbolic
  auto x = makeConcreteTensor({-1, 32});
  fusion->addInput(x);

  int64_t scan_dim = 0;

  // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
  //
  // Given x[i] for i=0 .. N-1:
  //
  //   m[-1] = -infinity
  //   d[-1] = 0
  //   for j = 0 .. N-1
  //     m[j] = max(m[j-1], x[j])
  //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
  //
  // Final denominator is d[N-1]

  auto* neg_infty = IrBuilder::create<Val>(
      -std::numeric_limits<double>::infinity(), DataType::Double);
  ScanResult max_scan_result = scan(
      set(x),
      scan_dim,
      BinaryOpType::Max,
      /*return_exclusive=*/true,
      /*discount_factor=*/nullptr,
      /*init=*/neg_infty); // max x[j] over j = 0 .. i
  TensorView* m = max_scan_result.inclusive;
  TensorView* m_prev = max_scan_result.exclusive;
  // normalize by running max and exponentiate
  TensorView* exp_x_m = exp(sub(x, m));
  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  TensorView* discount = exp(sub(m_prev, m));

  auto denoms = prefixSum(exp_x_m, scan_dim, discount).inclusive;

  auto norm_factor = reductionOp(
      BinaryOpType::RHS,
      {scan_dim},
      /*init=*/fusion->zeroVal(DataType::Float),
      denoms);

  fusion->addOutput(norm_factor);

  scheduler_utils::cacheInputs(fusion.get(), /*unroll=*/true);
  scheduler_utils::cacheAndForkOutputs(fusion.get(), /*unroll=*/true);

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (TensorView* tv : {m, m_prev, denoms}) {
    for (IterDomain* id : tv->getLoopDomain()) {
      uninlineable_ids.insert(id);
    }
  }

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (TensorView* tv : {m, m_prev, denoms}) {
    tv->computeWith(-1);
    for (Val* v : tv->definition()->inputs()) {
      // By using `uninlineable_ids` above, we prevent producers of scan from
      // inlining with the ScanOp past the scan dim, even though this is
      // desired. Here we do this inlining manually instead.
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  at::Tensor ref = (t0 - std::get<0>(t0.max(/*dim=*/0, /*keepdim=*/true)))
                       .exp()
                       .sum(/*dim=*/0);
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
      << " returned " << cg_outputs[0].as<at::Tensor>() << " but expected "
      << ref;

  // Test automatic evaluation also
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// This is a simplified version of FlashAttention that does not circular buffer
// the inputs or use mma instructions, but has the same general computation
// pattern.
//
// Dao et al. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention
// with IO-Awareness. https://arxiv.org/abs/2205.14135
TEST_F(ScanTest, FlashAttentionNoMma) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Inputs are Q, K, V
  // Normally each of these would be 2D, shaped N-by-d
  // Output is softmax(Q@K.T, dim=1)@V
  //
  // Here, I am hard-coding a tiling split and transpose:
  //
  //   Q: N1o, d1o, N1i, d1i
  //   K: N2o, d1o, d1i, N2i
  //   V: N2o, d2o, d2i, N2i
  //
  // For the Q@K.T matmul, we have the following dim roles:
  //
  //   M: N1o, N1i
  //   N: N2o, N2i
  //   K: d1o, d1i
  //
  // For the second matmul S@V the roles are:
  //
  //   M: N1o, N1i
  //   N: d2o, d2i
  //   K: N2o, N2i
  //
  // The overall output should be of size N1o, N1i, d2o, d2i since it is the
  // result of that final matmul
  //
  // The general strategy for ordinary flash attention is to have two main
  // serial loops: over the outer "K" dims d1o and N2o. The inner dims are
  // computed via a tile matmul, equivalent to two more loops d1i and N2i.
  //
  // We grid parallelize across CTAs in the N1o and d2o dimensions and each CTA
  // computes a final output of size N1i by d2i.
  int64_t N1o = 2;
  int64_t N1i = 3;
  int64_t N2o = 4;
  int64_t N2i = 5;
  int64_t d1o = 6;
  int64_t d1i = 7;
  int64_t d2o = 8;
  int64_t d2i = 9;

  // [N1o, N2o, d1o, d2o, N1i, N2i, d1i, d2i]
  auto Q = makeConcreteTensor({N1o, 1, d1o, 1, N1i, 1, d1i, 1});
  auto K = makeConcreteTensor({1, N2o, d1o, 1, 1, N2i, d1i, 1});
  auto V = makeConcreteTensor({1, N2o, 1, d2o, 1, N2i, 1, d2i});
  fusion->addInput(Q);
  fusion->addInput(K);
  fusion->addInput(V);

  // Notation is from Algorithm 1 of Dao et al. 2022

  // TODO: mma
  auto S =
      sum(mul(Q, K),
          /*dims=*/{-2},
          /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, N2i, (1), 1]

  auto m_tilde =
      max(S, {-3}, /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, (1), 1, 1]

  auto* neg_infty = IrBuilder::create<Val>(
      -std::numeric_limits<double>::infinity(), DataType::Double);
  ScanResult max_scan_result = scan(
      m_tilde,
      2,
      BinaryOpType::Max,
      /*return_exclusive=*/true,
      /*discount_factor=*/nullptr,
      /*init=*/neg_infty);
  TensorView* m =
      max_scan_result.inclusive; // [N1o, N2o, (d1o), 1, N1i, 1, 1, 1]
  TensorView* m_prev = max_scan_result.exclusive;

  auto P_tilde = exp(sub(S, m_tilde)); // [N1o, N2o, d1o, 1, N1i, N2i, 1, 1]

  auto l_tilde =
      sum(P_tilde,
          {-3},
          /*keep_dim=*/true); // [N1o, N2o, d1o, 1, N1i, (1), 1, 1]

  auto first_discount = exp(sub(m_prev, m)); // [N1o, N2o, d1o, 1, N1i, 1, 1, 1]

  auto l_tilde_factor =
      exp(sub(m_tilde, m)); // [N1o, N2o, d1o, 1, N1i, 1, 1, d2i]
  auto next_l =
      mul(l_tilde_factor, l_tilde); // [N1o, N2o, d1o, 1, N1i, 1, 1, d2i]

  ScanResult sum_scan_result = scan(
      next_l,
      2,
      BinaryOpType::Add,
      /*return_exclusive=*/true,
      /*discount_factor=*/first_discount,
      /*init=*/fusion->zeroVal(DataType::Float));
  TensorView* l =
      sum_scan_result.inclusive; // [N1o, N2o, (d1o), 1, N1i, 1, 1, 1]
  TensorView* l_prev = sum_scan_result.exclusive;

  auto O_discount =
      mul(div(l_prev, l), first_discount); // [N1o, N2o, d1o, 1, N1i, 1, 1, 1]

  // P_tilde = [N1o, N2o, d1o, 1, N1i, N2i, 1, d2i]
  // V       = [1,   N2o, 1, d2o, 1,   N2i, 1, d2i]
  auto PtildeV =
      sum(mul(P_tilde, V),
          /*dims=*/{-3},
          /*keep_dim=*/true); // [N1o, N2o, d1o, d2o, N1i, (1), 1, d2i]

  auto O = prefixSum(
               mul(div(l_tilde_factor, l), PtildeV),
               2,
               /*discount_factor=*/O_discount)
               .inclusive; // [N1o, N2o, (d1o), d20, N1i, 1, 1, d2i]

  auto O_final = reductionOp(
      BinaryOpType::RHS,
      {1, 2},
      /*init=*/fusion->zeroVal(DataType::Float),
      set(O), // TODO: is this set really needed to avoid computeWith error on
              // O?
      /*keepdim=*/true); // [N1o, (1), (1), d2o, N1i, 1, 1, d2i]

  fusion->addOutput(O_final);

  fusion->printMath();

  // We don't inline the scans past the scan dimension
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<ScanOp>()) {
      for (auto tv : ir_utils::filterByType<TensorView>(expr->outputs())) {
        for (IterDomain* id : tv->getLoopDomain()) {
          uninlineable_ids.insert(id);
        }
      }
    }
  }

  Q->cacheAfter();
  K->cacheAfter();
  V->cacheAfter();
  O_final->cacheBefore();

  inlineMost(uninlineable_ids);

  // These TVs are not inlined, but instead we set computeWith on them
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<ScanOp>()) {
      expr->output(0)->as<TensorView>()->computeWith(-1);
      for (Val* v : expr->inputs()) {
        // By using `uninlineable_ids` above, we prevent producers of scan from
        // inlining with the ScanOp past the scan dim, even though this is
        // desired. Here we do this inlining manually instead.
        v->as<TensorView>()->inlineAt(-1);
      }
    }
  }

  O_final->axis(0)->parallelize(ParallelType::BIDx);
  O_final->axis(3)->parallelize(ParallelType::BIDy);
  scheduler_utils::parallelizeAllLike(O_final);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor q = at::randn({N1o, 1, d1o, 1, N1i, 1, d1i, 1}, options);
  at::Tensor k = at::randn({1, N2o, d1o, 1, 1, N2i, d1i, 1}, options);
  at::Tensor v = at::randn({1, N2o, 1, d2o, 1, N2i, 1, d2i}, options);
  std::vector<c10::IValue> inputs{q, k, v};

  auto qorig = q.transpose(2, 4).reshape({N1o * N1i, d1o * d1i}); // 6, 42
  auto korig = k.transpose(2, 5).reshape({N2o * N2i, d1o * d1i}); // 20, 42
  auto vorig = v.transpose(3, 5).reshape({N2o * N2i, d2o * d2i}); // 20, 72
  auto qktref = at::matmul(qorig, korig.t()); // 6, 20
  auto sref = at::softmax(qktref, 1); // 6, 20
  auto ref =
      at::matmul(sref, vorig)
          .reshape({N1o, 1, 1, d2o, N1i, 1, 1, d2i}); // 2, 1, 1, 8, 3, 1, 1, 9

  KernelExecutor ke;
  ke.compile(fusion.get(), inputs);

  auto cg_outputs = ke.run(inputs);

  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>().squeeze(), ref.squeeze()));
  //<< " returned " << cg_outputs[0].as<at::Tensor>()[0] << " but expected "
  //<< ref[0];
}

TEST_F(ScanTest, BlockedAttention) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t N = 12; // sequence length
  int64_t D = 5; // hidden dimension size
  int64_t Br = 2; // block size for rows
  int64_t Bc = 3; // block size for columns
  int64_t Tr = N / Br; // Total number of row blocks for Q, 4
  int64_t Tc = N / Bc; // Total number of column blocks for K^T, 8
  // [N,D] --> [Tr, Br, D] for Q and [Tc, Bc, D] for K,V
  auto Q = makeConcreteTensor({Tr, Br, 1, 1, D});
  auto K = makeConcreteTensor({1, 1, Tc, Bc, D});
  auto V = makeConcreteTensor({1, 1, Tc, Bc, D});
  fusion->addInput(Q);
  fusion->addInput(K);
  fusion->addInput(V);

  // QK --> [Tr, Br, Tc, Bc, D]
  auto QK = mul(Q, K);
  // S --> [Tr, Br, Tc, Bc] (sum over hidden dimension for dot product)
  auto Sij = sum(QK, {4});

  // Line 10, [Tr, Br, Tc, Bc] -> [Tr, Br, Tc, 1]
  auto mij_tilde = max(Sij, {3}, /*keep_dim=*/true); // [Tr, Br, Tc, 1]
  auto pij_tilde = exp(sub(Sij, mij_tilde)); // [Tr, Br, Tc, Bc]
  auto lij_tilde = sum(pij_tilde, {3}, /*keep_dim=*/true); // [Tr, Br, Tc, 1]

  // Line 11, m_i_new, [Tr, Br, Tc, 1] -> [Tr, Br, Tc, 1]
  auto m_i_result =
      scan(mij_tilde, 2, BinaryOpType::Max, /*return_exclusive=*/true);
  auto m_i = m_i_result.exclusive;
  auto m_i_new = m_i_result.inclusive;
  // Line 11, l_i_new, [Tr, Br, Tc, 1]
  auto lij_tidle_factor = exp(sub(mij_tilde, m_i_new)); // [Tr, Br, Tc, 1]
  auto l_i_factor = exp(sub(m_i, m_i_new)); // [Tr, Br, Tc, 1]
  auto next_l = mul(lij_tidle_factor, lij_tilde); // [Tr, Br, Tc, 1]
  // prefix sum is always inclusive, for exlcusive, l[i] = l_new[i-1]
  auto l_i_result = prefixSum(next_l, 2, l_i_factor, /*return_exclusive=*/true);
  auto l_i = l_i_result.exclusive;
  auto l_i_new = l_i_result.inclusive;

  // Line 12, o_i, [Tr, Br, Tc, D]
  // [Tr, Br, Tc, Bc, 1] X [1, 1, Tc, Bc, D] --> [Tr, Br, Tc, Bc, D]
  auto pij_tilde_vj_dot =
      mul(broadcast(pij_tilde, {false, false, false, false, true}), V);
  auto pij_tilde_vj = sum(pij_tilde_vj_dot, {3}); // [Tr, Br, Tc, D]
  auto next_o =
      mul(div(lij_tidle_factor, l_i_new), pij_tilde_vj); // [Tr, Br, Tc, D]
  auto o_discount = div(mul(l_i_factor, l_i), l_i_new);
  auto O = prefixSum(next_o, 2, o_discount).inclusive;
  fusion->addOutput(O);
  fusion->printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aQ = at::randn({Tr, Br, 1, 1, D}, options);
  auto aK = at::randn({1, 1, Tc, Bc, D}, options);
  auto aV = at::randn({1, 1, Tc, Bc, D}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aQ, aK, aV});
  auto outputs = ke.run({aQ, aK, aV});
  // get last element of dimension Tc from outputs[0] which has shape [Tr, Br,
  // Tc, D] The prefix scan produces outputs for each step, we want the final
  // result
  auto final_output = at::select(
      outputs[0].as<at::Tensor>(), /*dim=*/2, /*index=*/-1); // [Tr, Br, D]
  std::cout << "final_output: " << final_output << std::endl;

  auto aten_Q = aQ.reshape({N, D});
  auto aten_K = aK.reshape({N, D});
  auto aten_V = aV.reshape({N, D});
  auto aten_S = at::matmul(aten_Q, aten_K.transpose(-2, -1));
  auto aten_S_softmax = at::softmax(aten_S, 1);
  auto aten_O = at::matmul(aten_S_softmax, aten_V).reshape({Tr, Br, D});
  std::cout << "aten_O: " << aten_O << std::endl;

  EXPECT_TRUE(at::allclose(final_output, aten_O, 1e-4, 1e-6, true));
}

TEST_F(ScanTest, BlockedAttentionInline1) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t N = 20; // sequence length
  int64_t D = 6; // hidden dimension size
  int64_t Br = 10; // block size for rows
  int64_t Bc = 5; // block size for columns
  int64_t Tr = N / Br; // 2, Total number of row blocks for Q, 4
  int64_t Tc = N / Bc; // 4, Total number of column blocks for K^T, 8
  // [N,D] --> [Tr, Br, D] for Q and [Tc, Bc, D] for K,V
  auto Q = makeConcreteTensor({Tr, Br, 1, 1, D});
  auto K = makeConcreteTensor({1, 1, Tc, Bc, D});
  auto V = makeConcreteTensor({1, 1, Tc, Bc, D});
  fusion->addInput(Q);
  fusion->addInput(K);
  fusion->addInput(V);

  // QK --> [Tr, Br, Tc, Bc, D] --> [2, 10, 4, 5, 6]
  auto QK = mul(Q, K);
  // S --> [Tr, Br, Tc, Bc, 1] (sum over hidden dimension for dot product)
  // Sij is consumed by both max and sub, needs to store reduction dimension
  // in max(Sij, {5}), which is Bc
  auto Sij = sum(QK, {4}, /*keep_dim=*/true);

  // Line 10, [Tr, Br, Tc, Bc, 1] -> [Tr, Br, Tc, 1, 1]
  auto mij_tilde = max(Sij, {3}, /*keep_dim=*/true); // [Tr, Br, Tc, 1, 1]
  // expr sort error if directly pass mij_tilde_raw to scan
  // auto mij_tilde = set(mij_tilde_raw);
  auto pij_tilde = exp(sub(Sij, mij_tilde)); // [Tr, Br, Tc, Bc, 1]
  auto lij_tilde = sum(pij_tilde, {3}, /*keep_dim=*/true); // [Tr, Br, Tc, 1, 1]

  // Line 12, o_i, [Tr, Br, Tc, D]
  // [Tr, Br, Tc, Bc, 1] X [1, 1, Tc, Bc, D] --> [Tr, Br, Tc, Bc, D]
  auto pij_tilde_vj_dot = mul(pij_tilde, V);
  auto pij_tilde_vj =
      sum(pij_tilde_vj_dot, {3}, /*keep_dim=*/true); // [Tr, Br, Tc, 1, D]

  // Line 11, m_i_new, [Tr, Br, Tc, 1, 1] -> [Tr, Br, Tc, 1, 1]
  auto m_i_result =
      scan(mij_tilde, 2, BinaryOpType::Max, /*return_exclusive=*/true);
  auto m_i = m_i_result.exclusive;
  auto m_i_new = m_i_result.inclusive;

  // Line 11, l_i_new, [Tr, Br, Tc, 1, 1]
  auto lij_tidle_factor = exp(sub(mij_tilde, m_i_new)); // [Tr, Br, Tc, 1, 1]
  auto l_i_factor = exp(sub(m_i, m_i_new)); // [Tr, Br, Tc, 1, 1]
  auto next_l = mul(lij_tidle_factor, lij_tilde); // [Tr, Br, Tc, 1, 1]
  // prefix sum is always inclusive, for exlcusive, l[i] = l_new[i-1]
  auto l_i_result = prefixSum(next_l, 2, l_i_factor, /*return_exclusive=*/true);
  auto l_i = l_i_result.exclusive;
  auto l_i_new = l_i_result.inclusive;
  auto lij_div_l_i_new = div(lij_tilde, l_i_new);

  auto next_o = mul(lij_div_l_i_new, set(pij_tilde_vj)); // [Tr, Br, Tc, 1, D]
  auto o_discount = div(mul(l_i_factor, l_i), l_i_new);
  // [Tr, Br, Tc, 1, D]
  auto O = prefixSum(next_o, 2, o_discount).inclusive;
  // // [Tr, Br, Tc, 1, D] -> [Tr, Br, D]
  // // auto O_final = max(set(O), {2, 3});
  // auto O_final = reductionOp(
  //   BinaryOpType::RHS,
  //   {2, 3},
  //   /*init=*/fusion->zeroVal(DataType::Float),
  //   set(O),
  //   /*keepdim=*/false);
  fusion->addOutput(set(O));
  fusion->printMath();

  // Same as InclusiveScan
  const auto& scan_outputs = {m_i, m_i_new, l_i, l_i_new, O};

  // Similar to InclusiveScan and InclusiveExclusiveScan
  // Avoid inlining the scanned dimensions
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (auto tv : scan_outputs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isScan()) {
        uninlineable_ids.insert(id);
      }
    }
  }
  // use inlineMost to auto detect max inline position
  inlineMost(uninlineable_ids);

  // control compute position
  // manual inline the producers
  for (auto tv : scan_outputs) {
    int compute_with_pos = -1;
    tv->computeWith(compute_with_pos);
    std::cout << "\ntv: " << tv->toString() << std::endl;
    for (auto v : tv->definition()->inputs()) {
      std::cout << "pv: " << v->toString() << std::endl;
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aQ = at::randn({Tr, Br, 1, 1, D}, options);
  auto aK = at::randn({1, 1, Tc, Bc, D}, options);
  auto aV = at::randn({1, 1, Tc, Bc, D}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aQ, aK, aV});
  auto outputs = ke.run({aQ, aK, aV});
  auto final_output = at::select(
      outputs[0].as<at::Tensor>(), /*dim=*/2, /*index=*/-1); // [Tr, Br, 1, D]
  final_output =
      at::select(final_output, /*dim=*/2, /*index=*/-1); // [Tr, Br, D]
  std::cout << "final_output: " << final_output << std::endl;

  auto aten_Q = aQ.reshape({N, D});
  auto aten_K = aK.reshape({N, D});
  auto aten_V = aV.reshape({N, D});
  auto aten_S = at::matmul(aten_Q, aten_K.transpose(-2, -1));
  auto aten_S_softmax = at::softmax(aten_S, 1);
  auto aten_O = at::matmul(aten_S_softmax, aten_V).reshape({Tr, Br, D});
  std::cout << "aten_O: " << aten_O << std::endl;

  EXPECT_TRUE(at::allclose(final_output, aten_O, 1e-4, 1e-6, true));
}

TEST_F(ScanTest, BlockedAttentionInline2) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t N = 20; // sequence length
  int64_t D = 6; // hidden dimension size
  int64_t Br = 10; // block size for rows
  int64_t Bc = 5; // block size for columns
  int64_t Tr = N / Br; // 2, Total number of row blocks for Q, 4
  int64_t Tc = N / Bc; // 4, Total number of column blocks for K^T, 8
  // [N,D] --> [Tr, Br, D] for Q and [Tc, Bc, D] for K,V
  auto Q = makeConcreteTensor({Tr, 1, Br, 1, D});
  auto K = makeConcreteTensor({1, Tc, 1, Bc, D});
  auto V = makeConcreteTensor({1, Tc, 1, Bc, D});
  fusion->addInput(Q);
  fusion->addInput(K);
  fusion->addInput(V);

  // [Tr, Tc, Br, Bc, 1] = sum([Tr, 1, Br, 1, D] X [1, Tc, 1, Bc, D])
  auto sij = sum(mul(Q, K), {4}, /*keep_dim=*/true);
  // [Tr, Tc, Br, 1, 1] = [Tr, Tc, Br, Bc, 1]
  auto row_max_sij = max(sij, {3}, /*keep_dim=*/true);
  // [Tr, Tc, Br, 1, 1] = scan([Tr, Tc, Br, 1, 1])
  auto [m_i_new, m_i, _] =
      scan(row_max_sij, 1, BinaryOpType::Max, /*return_exclusive=*/true);
  // [Tr, Tc, Br, Bc, 1] = [Tr, Tc, Br, Bc, 1], [Tr, Tc, Br, 1, 1]
  auto pij_tilde = exp(sub(sij, m_i_new));
  // [Tr, Tc, Br, 1, 1] = sum([Tr, Tc, Br, Bc, 1])
  auto row_sum_pij_tilde = sum(pij_tilde, {3}, /*keep_dim=*/true);
  // [Tr, Tc, Br, 1, 1] = [Tr, Tc, Br, 1, 1]
  auto l_i_discount = exp(sub(m_i, m_i_new));
  // [Tr, Tc, Br, 1, 1] = [Tr, Tc, Br, 1, 1]
  auto l_i = prefixSum(row_sum_pij_tilde, 1, l_i_discount).inclusive;
  // [Tr, Tc, Br, 1, 1] = [Tr, Tc, Br, 1, 1]
  auto O_i_discount = reciprocal(l_i_discount);
  // [Tr, Tc, Br, 1, D] = sum([Tr, Tc, Br, Bc, 1] X [1, Tc, 1, Bc, D])
  auto O_i_val = sum(mul(pij_tilde, V), {3}, /*keep_dim=*/true);
  // [Tr, Tc, Br, 1, D] = [Tr, Tc, Br, 1, 1] X [Tr, Tc, Br, 1, D]
  auto O_i = prefixSum(O_i_val, 1, O_i_discount).inclusive;
  // [Tr, Tc, Br, 1, D] = [Tr, Tc, Br, 1, D] / [Tr, Tc, Br, 1, 1]
  auto O_i_final = div(O_i, l_i);

  fusion->addOutput(set(O_i_final));
  fusion->printMath();

  // Same as InclusiveScan
  const auto& scan_outputs = {m_i, m_i_new, l_i, O_i};

  // Similar to InclusiveScan and InclusiveExclusiveScan
  // Avoid inlining the scanned dimensions
  std::unordered_set<IterDomain*> uninlineable_ids;
  for (auto tv : scan_outputs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isScan()) {
        uninlineable_ids.insert(id);
      }
    }
  }
  // use inlineMost to auto detect max inline position
  inlineMost(uninlineable_ids);

  // control compute position
  // manual inline the producers
  for (auto tv : scan_outputs) {
    int compute_with_pos = -1;
    tv->computeWith(compute_with_pos);
    for (auto v : tv->definition()->inputs()) {
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aQ = at::randn({Tr, 1, Br, 1, D}, options);
  auto aK = at::randn({1, Tc, 1, Bc, D}, options);
  auto aV = at::randn({1, Tc, 1, Bc, D}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aQ, aK, aV});
  auto outputs = ke.run({aQ, aK, aV});
  // [Tr, Br, 1, D] = [Tr, Tc, Br, 1, D]
  auto final_output =
      at::select(outputs[0].as<at::Tensor>(), /*dim=*/1, /*index=*/-1);
  // [Tr, Br, D] = [Tr, Br, 1, D]
  final_output =
      at::select(final_output, /*dim=*/2, /*index=*/-1); // [Tr, Br, D]
  std::cout << "final_output: " << final_output << std::endl;

  auto aten_Q = aQ.reshape({N, D});
  auto aten_K = aK.reshape({N, D});
  auto aten_V = aV.reshape({N, D});
  auto aten_S = at::matmul(aten_Q, aten_K.transpose(-2, -1));
  auto aten_S_softmax = at::softmax(aten_S, 1);
  auto aten_O = at::matmul(aten_S_softmax, aten_V).reshape({Tr, Br, D});
  std::cout << "aten_O: " << aten_O << std::endl;

  EXPECT_TRUE(at::allclose(final_output, aten_O, 1e-4, 1e-6, true));
}

TEST_F(ScanTest, ScanMiddleDimension) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto t0 = makeConcreteTensor({2, 3, 4});
  fusion->addInput(t0);

  auto t1 = set(t0);
  auto t2 = scan(t1, 1, BinaryOpType::Max).inclusive;
  auto t3 = set(t2);
  fusion->addOutput(t3);

  const auto& scan_outputs = {t2};

  std::unordered_set<IterDomain*> uninlineable_ids;
  for (auto tv : scan_outputs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isScan()) {
        uninlineable_ids.insert(id);
      }
    }
  }
  // use inlineMost to auto detect max inline position
  inlineMost(uninlineable_ids);

  // control compute position
  // manual inline the producers
  for (auto tv : scan_outputs) {
    int compute_with_pos = -1;
    tv->computeWith(compute_with_pos);
    for (auto v : tv->definition()->inputs()) {
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aten_t0 = at::randn({2, 3, 4}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aten_t0});
  auto outputs = ke.run({aten_t0});
  testValidate(fusion.get(), outputs, {aten_t0}, __LINE__, __FILE__);
}

TEST_F(ScanTest, ScanMiddleDimensionExclusive) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto t0 = makeConcreteTensor({2, 3});
  fusion->addInput(t0);

  auto t1 = set(t0);
  auto [t2, t3, _] = scan(t1, 0, BinaryOpType::Max, /*return_exclusive=*/true);
  auto t4 = set(t2);
  auto t5 = set(t3);
  fusion->addOutput(t4);
  fusion->addOutput(t5);

  const auto& scan_outputs = {t2, t3};

  std::unordered_set<IterDomain*> uninlineable_ids;
  for (auto tv : scan_outputs) {
    for (auto id : tv->getLoopDomain()) {
      if (id->isScan()) {
        uninlineable_ids.insert(id);
      }
    }
  }
  // use inlineMost to auto detect max inline position
  inlineMost(uninlineable_ids);

  // control compute position
  // manual inline the producers
  for (auto tv : scan_outputs) {
    int compute_with_pos = -1;
    tv->computeWith(compute_with_pos);
    for (auto v : tv->definition()->inputs()) {
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aten_t0 = at::randn({2, 3}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aten_t0});
  auto outputs = ke.run({aten_t0});
  std::cout << "outputs: " << outputs[0].as<at::Tensor>() << std::endl;
  std::cout << "outputs: " << outputs[1].as<at::Tensor>() << std::endl;
  testValidate(fusion.get(), outputs, {aten_t0}, __LINE__, __FILE__);
}
} // namespace nvfuser
