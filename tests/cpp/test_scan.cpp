// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <csrc/scheduler/tools/inlining.h>
#include <fusion.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include "ir/internal_nodes.h"
#include "ir/utils.h"
#include "type.h"
namespace nvfuser {

using ScanTest = NVFuserTest;

// Basic functionality test for scan with Add operation (cumsum)
TEST_F(ScanTest, BasicScanAdd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto result = scan(tv0, /*dim=*/1, BinaryOpType::Add);
  fusion.addOutput(result.inclusive);

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
  auto result = scan(tv0, /*dim=*/1, BinaryOpType::Max);
  fusion.addOutput(result.inclusive);

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
  auto result = scan(tv0, /*dim=*/1, BinaryOpType::Min);
  fusion.addOutput(result.inclusive);

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
  auto result = scan(tv0, /*dim=*/1, BinaryOpType::Mul);
  fusion.addOutput(result.inclusive);

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
  auto result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion.addOutput(result.inclusive);

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
  auto result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
  fusion.addOutput(result.inclusive);

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
  auto result2 = scan(tv1, /*dim=*/1, BinaryOpType::Add);
  auto tv2 = result2.inclusive;

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
  auto result4 = scan(tv3, /*dim=*/1, BinaryOpType::Add);
  auto tv4 = result4.inclusive;

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
    auto result = scan(tv1, /*dim=*/1, BinaryOpType::Add);
    auto tv_output = set(result.inclusive);
    fusion.addOutput(tv_output);

    // Parallelization strategy - all tensors get same parallelization
    for (auto tv : {tv1, result.inclusive, tv_output}) {
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
  auto result = scan(tv1, /*dim=*/1, BinaryOpType::Add);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
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
  auto result = scan(tv1, /*dim=*/1, BinaryOpType::Max);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
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
  auto result = scan(tv1, /*dim=*/1, BinaryOpType::Min);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
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
  auto result = scan(tv1, /*dim=*/1, BinaryOpType::Mul);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
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
  auto result_scan1 = scan(tv1, /*dim=*/1, BinaryOpType::Add);
  auto tv_scan1 = result_scan1.inclusive;

  // Add operation between scans
  auto tv_add = add(tv_scan1, IrBuilder::create<Val>(1.0));

  // Second scan operation (Max)
  auto result_scan2 = scan(tv_add, /*dim=*/1, BinaryOpType::Max);
  auto tv_scan2 = result_scan2.inclusive;

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

TEST_F(ScanTest, KernelExecutorSerialScanMax) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1, Mul operation
  auto result = scan(tv1, /*dim=*/1, BinaryOpType::Max);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

TEST_F(ScanTest, KernelExecutorSerialScanMaxExclusive) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto result =
      scan(tv1, /*dim=*/1, BinaryOpType::Max, /*return_exclusive=*/true);
  auto tv2 = result.exclusive;
  auto tv_output = set(tv2);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv2, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // [1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1]
  auto input = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}},
      options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  std::cout << "outputs[0]: " << outputs[0] << std::endl;
}

TEST_F(ScanTest, KernelExecutorPrefixSum) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 =
      prefixSum(tv1, /*dim=*/1, IrBuilder::create<Val>(0.5, DataType::Float))
          .inclusive;
  auto tv_output = set(tv2);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv2, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // [1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1]
  auto input = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}},
      options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  std::cout << "outputs[0]: " << outputs[0] << std::endl;
}

TEST_F(ScanTest, KernelExecutorPrefixSumTensorDiscount) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = prefixSum(tv1, /*dim=*/1, tv1).inclusive;
  auto tv_output = set(tv2);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv2, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // [1,2,3,4,5,6,7,8], [8,7,6,5,4,3,2,1]
  auto input = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}},
      options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  std::cout << "outputs[0]: " << outputs[0] << std::endl;
}
// Parameterized test for different BinaryOpType and data types
class SerialScanTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<std::tuple<BinaryOpType, DataType>> {
};
TEST_P(SerialScanTest, BinaryOpTypeAndDataType) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto [binary_op_type, dtype] = GetParam();

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8]
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape, dtype);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create scan operation along dimension 1 with parameterized operation
  auto result = scan(tv1, /*dim=*/1, binary_op_type);
  auto tv_output = set(result.inclusive);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, result.inclusive, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  // Use appropriate input generation based on data type
  at::Tensor input =
      at::randint(-100, 100, {4, 8}, options).to(data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ScanTest,
    SerialScanTest,
    ::testing::Combine(
        ::testing::Values(
            BinaryOpType::Add,
            BinaryOpType::Max,
            BinaryOpType::Min,
            BinaryOpType::Mul),
        ::testing::Values(DataType::Float, DataType::Double, DataType::Int)));

TEST_F(ScanTest, RFactorBlockReduction) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4, 8192};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {-1});
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);
  auto unscheduled_fusion = fusion;

  int64_t vect = 4, threads = 256;
  for (auto tv : {tv1, tv2}) {
    tv->split(1, vect);
    tv->split(1, threads);
  }
  // [I, R, threads, vect]
  // thread local reduction
  auto tv4 = tv2->rFactor({3});
  std::cout << "tv4: " << tv4->toString() << std::endl;
  tv4->printTransforms();

  // tv2: [I, R, threads]
  // block reduction
  auto tv5 = tv2->rFactor({2});
  std::cout << "tv5: " << tv5->toString() << std::endl;
  tv5->printTransforms();

  // Parallelization strategy
  for (auto tv : {tv1, tv2, tv3, tv4, tv5}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    if (tv != tv3 && tv != tv2) {
      tv->axis(2)->parallelize(ParallelType::TIDx);
    }
    if (tv == tv1) {
      tv->axis(3)->parallelize(ParallelType::Vectorize);
    }
  }
  inlineMost(std::vector<TensorView*>{tv1, tv3, tv4});
  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8192}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&unscheduled_fusion, outputs, {input}, __LINE__, __FILE__);
}

// online computation of sum(exp(x - max(x)))
// split x into 8 partitions xp1, xp2, ..., xp8
// compute max and sum for each partition
// allocate m[] and d[]
// for i = 1, 2, ..., 8
//   mi = max(xpi)
//   di = sum(exp(xpi - mi))
// for i = 1, 2, ..., 8
//   m_final = max(mi)
// for i = 1, 2, ..., 8
//   d_final = sum(di * exp(m - mi))
// return d

// pro: pure reduction, can use warp reduce when number of partitions is large
// con: needs extra memory for m[] and d[]
TEST_F(ScanTest, OnlineSumExpXMinusMaxNoScan) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4, 8, 1024};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = max(tv1, {2});
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = sub(tv1, tv3);
  auto tv5 = exp(tv4);
  auto tv6 = sum(tv5, {2});
  // m = max(tv2)
  // d = sum(d * exp(m - max(m)))
  auto tv7 = max(tv2, {1}); // {4,8} -> {4}
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 = sub(tv2, tv8); // mi - m
  auto tv10 = exp(tv9); // exp(mi - m)
  auto tv11 = mul(tv6, tv10); // di * exp(mi - m)
  auto tv12 = sum(tv11, {1}); // sum(di * exp(mi - m))
  auto tv13 = set(tv12);
  fusion.addOutput(tv13);
  auto unscheduled_fusion = fusion;

  int64_t vect = 4, threads = 256;
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(2, vect);
    tv->split(2, threads);
  }

  fusion.printMath();

  // Parallelization strategy
  for (auto tv :
       {tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12, tv13}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->axis(3)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(4)->parallelize(ParallelType::Vectorize);

  // [I1, I2, R, threads, vect]
  // thread local reduction
  auto tv14 = tv2->rFactor({2, 4});
  auto tv15 = tv6->rFactor({2, 4});
  std::cout << "tv14: " << tv14->toString() << std::endl;
  tv14->printTransforms();
  std::cout << "tv15: " << tv15->toString() << std::endl;
  tv15->printTransforms();

  // inlineMost also works, want to clearly split the kernel into 2 parts
  // block reduction to get mi and di
  // online merge of m[] and d[] to get the final m and d
  tv6->inlineAt(1);

  inlineMost(std::vector<TensorView*>{
      tv1,
      tv2,
      tv3,
      tv4,
      tv5,
      tv7,
      tv8,
      tv9,
      tv10,
      tv11,
      tv12,
      tv13,
      tv14,
      tv15});
  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  // equivalent to sum(exp(x - max(x)))
  auto input_reshaped = input.reshape({4, 8192});
  auto aten_max = at::amax(input_reshaped, {1});
  std::cout << "aten_max: " << aten_max << std::endl;
  auto aten_exp = at::exp(input_reshaped - aten_max.unsqueeze(1));
  auto aten_sum = at::sum(aten_exp, {1});
  std::cout << "aten_sum: " << aten_sum << std::endl;
  std::cout << "outputs[0]: " << outputs[0] << std::endl;

  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_sum, 1e-5, 1e-8, true));
}
// Given x[i] for i=0 .. N-1:
//
//   m[-1] = -infinity
//   d[-1] = 0
//   for j = 0 .. N-1
//     m[j] = max(m[j-1], x[j])
//     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
//
// Final denominator is d[N-1]
// online computation of sum(exp(x - max(x)))
// split x into 8 partitions xp1, xp2, ..., xp8
// compute max and sum for each partition
// m[-1] = -inf, d[-1] = 0
// for j = 0 .. N-1
//   mp = max(xpi)
//   m[j] = max(m[j-1], mp)
//   dp = sum(exp(xpi - m[j]))
//   d[j] = d[j-1] * exp(m[j-1] - m[j]) + dp
// return d[N-1]
TEST_F(ScanTest, OnlineSumExpXMinusMaxSerialScan) {
  GTEST_SKIP();
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4, 8, 1024};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = max(tv1, {2}); // mp = max(xpi)
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = sub(tv1, tv3);
  auto tv5 = exp(tv4);
  auto tv6 = sum(tv5, {2}); // dp = sum(exp(xpi - mp))

  // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
  auto result7 =
      scan(tv2, {1}, BinaryOpType::Max, /*return_exclusive=*/true); // m[i-1]
  auto tv7 = result7.exclusive;
  auto tv8 = binaryOp(BinaryOpType::Max, tv2, tv7); // m[i]
  auto tv9 = sub(tv7, tv8); //  exp(m[i-1] - m[i])
  auto tv10 = exp(tv9);
  auto tv11 = prefixSum(tv6, {1}, tv10).inclusive;
  auto tv12 = set(tv11);
  fusion.addOutput(tv8);
  fusion.addOutput(tv12);
  auto unscheduled_fusion = fusion;

  int64_t vect = 4, threads = 256;
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->split(2, vect);
    tv->split(2, threads);
  }

  fusion.printMath();

  // Parallelization strategy
  for (auto tv :
       {tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8, tv9, tv10, tv11, tv12}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  for (auto tv : {tv1, tv2, tv3, tv4, tv5, tv6}) {
    tv->axis(3)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(4)->parallelize(ParallelType::Vectorize);

  // [I1, I2, R, threads, vect]
  // thread local reduction
  auto tv13 = tv2->rFactor({2, 4});
  auto tv14 = tv6->rFactor({2, 4});

  // We don't inline the scans past the scan dimension
  // std::unordered_set<IterDomain*> uninlineable_ids;
  // for (TensorView* tv : {tv7, tv8, tv10}) {
  //   for (IterDomain* id : tv->getLoopDomain()) {
  //     uninlineable_ids.insert(id);
  //   }
  // }
  tv7->inlineAt(1);
  tv11->inlineAt(1);
  inlineMost(std::vector<TensorView*>{
      tv1, tv2, tv3, tv4, tv5, tv6, tv8, tv9, tv10, tv12, tv13, tv14});
  // // These TVs are not inlined, but instead we set computeWith on them
  // for (TensorView* tv : {tv7, tv8, tv11}) {
  //   tv->computeWith(-1);
  //   for (Val* v : tv->definition()->inputs()) {
  //     // By using `uninlineable_ids` above, we prevent producers of scan from
  //     // inlining with the ScanOp past the scan dim, even though this is
  //     // desired. Here we do this inlining manually instead.
  //     v->as<TensorView>()->inlineAt(-1);
  //   }
  // }

  fusion.printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  // equivalent to sum(exp(x - max(x)))
  auto input_reshaped = input.reshape({4, 8192});
  auto aten_max = at::amax(input_reshaped, {1});
  std::cout << "aten_max: " << aten_max << std::endl;
  auto aten_exp = at::exp(input_reshaped - aten_max.unsqueeze(1));
  auto aten_sum = at::sum(aten_exp, {1});
  std::cout << "aten_sum: " << aten_sum << std::endl;
  std::cout << "outputs[0]: " << outputs[0] << std::endl;
  std::cout << "outputs[1]: " << outputs[1] << std::endl;

  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_sum, 1e-5, 1e-8, true));
}

// Online normalizer for softmax: https://arxiv.org/abs/1805.02867

// Given x[i] for i=0 .. N-1:

//   m[-1] = -infinity
//   d[-1] = 0
//   for j = 0 .. N-1
//     m[j] = max(m[j-1], x[j])
//     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])

// sum(exp(x[i] - max(x)))
// is equivalent to:
// m[i] = max(m[i-1], x[i])
// d[i] = d[i-1] * exp(m[i-1] - m[i]) + exp(x[i] - m[i])
// return d[N-1]
TEST_F(ScanTest, InclusiveScan) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 =
      scan(tv1, {1}, BinaryOpType::Max, /*return_exclusive=*/false).inclusive;
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);
  auto unscheduled_fusion = fusion;

  // tv2 is a consumer of scan, don't inline the scan dim as its previous value
  // is required for the current iteration
  int scan_dim = 1;
  for (auto tv : {tv1, tv3}) {
    tv->inlineAt(-1);
  }
  tv2->inlineAt(scan_dim);
  /*========== Generated code ======================
  Array<float, 8, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i2 = 0; i2 < 8; ++i2) {
    Array<float, 1, 1> T1;
    T1[0] = 0;
    T1[0]
       = T0[(i1 + (i2 + nvfuser_zero))];
    T2[i2] = fmax(
      ((i2 > 0) ? (T2[(-1 + i2)]) : NEG_INFINITY),
      (T1[0]));
  }
  #pragma unroll
  for(nvfuser_index_t i3 = 0; i3 < 8; ++i3) {
    T3[(i1 + (i3 + nvfuser_zero))]
       = T2[i3];
  }
  ===============================================*/
  // To aviod the scanned domain is allocated, compute with its consumer.
  tv2->computeWith(-1);
  /*========== Generated code ======================
  Array<float, 1, 1> T2;
  #pragma unroll
  for(nvfuser_index_t i2 = 0; i2 < 8; ++i2) {
    nvfuser_index_t i3;
    i3 = i1 + (i2 + nvfuser_zero);
    Array<float, 1, 1> T1;
    T1[0] = 0;
    T1[0]
       = T0[i3];
    T2[0] = fmax(
      ((i2 > 0) ? (T2[0]) : NEG_INFINITY),
      (T1[0]));
    T3[i3]
       = T2[0];
  }
  ===============================================*/
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}},
      options);
  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  // inclusive scan
  auto aten_output = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f}},
      options);
  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_output, 1e-5, 1e-8, true));
}

// for(nvfuser_index_t i0 = 0; i0 < 2; ++i0) {
//   nvfuser_index_t i1;
//   i1 = 8 * i0;
//   Array<float, 1, 1> T2;
//   Array<float, 1, 1> T3;
//   #pragma unroll
//   for(nvfuser_index_t i2 = 0; i2 < 8; ++i2) {
//     nvfuser_index_t i3;
//     i3 = i1 + (i2 + nvfuser_zero);
//     Array<float, 1, 1> T1;
//     T1[0] = 0;
//     T1[0]
//        = T0[i3];
//     T3[0] = ((i2 > 0) ? (T2[0]) : NEG_INFINITY);
//     T2[0] = fmax(
//       ((i2 > 0) ? (T2[0]) : NEG_INFINITY),
//       (T1[0]));
//     T4[i3]
//        = T2[0];
//     T5[i3]
//        = T3[0];
//   }
// }
TEST_F(ScanTest, InclusiveExclusiveScan) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto res_scan = scan(tv1, {1}, BinaryOpType::Max, /*return_exclusive=*/true);
  auto tv2 = res_scan.inclusive;
  auto tv3 = res_scan.exclusive;
  auto tv4 = set(tv2);
  auto tv5 = set(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  auto unscheduled_fusion = fusion;

  // Same as InclusiveScan
  // use inlineAt to control allocation position.
  // use computeWith to avoid allocating the scanned domain.
  int scan_dim = 1;
  const auto& scan_outputs = {tv2, tv3};
  const auto& all_other_tvs = ir_utils::allTvsExcept(
      &fusion, {scan_outputs.begin(), scan_outputs.end()});
  for (auto tv : all_other_tvs) {
    tv->inlineAt(-1);
  }
  for (auto tv : scan_outputs) {
    tv->inlineAt(scan_dim);
    tv->computeWith(-1);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}},
      options);
  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});
  // inclusive scan
  auto aten_inclusive = torch::tensor(
      {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
       {8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f}},
      options);
  auto neg_inf = -std::numeric_limits<float>::infinity();
  auto aten_exclusive = torch::tensor(
      {{neg_inf, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
       {neg_inf, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f, 8.0f}},
      options);
  EXPECT_TRUE(at::allclose(
      outputs[0].as<at::Tensor>(), aten_inclusive, 1e-5, 1e-8, true));
  EXPECT_TRUE(at::allclose(
      outputs[1].as<at::Tensor>(), aten_exclusive, 1e-5, 1e-8, true));
}

// Online normalizer for softmax: https://arxiv.org/abs/1805.02867

// Given x[i] for i=0 .. N-1:

//   m[-1] = -infinity
//   d[-1] = 0
//   for j = 0 .. N-1
//     m[j] = max(m[j-1], x[j])
//     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])

// sum(exp(x[i] - max(x)))
// is equivalent to:
// m[i] = max(m[i-1], x[i])
// d[i] = d[i-1] * exp(m[i-1] - m[i]) + exp(x[i] - m[i])
// return d[N-1]
TEST_F(ScanTest, SumExpScan) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {2, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  // m[i-1]
  auto scan_res1 = scan(tv1, {1}, BinaryOpType::Max, /*return_exclusive=*/true);
  auto tv2 = scan_res1.inclusive;
  // m[i]
  auto tv3 = scan_res1.exclusive;
  //  exp(m[i-1] - m[i])
  auto tv4 = sub(tv3, tv2);
  auto tv5 = exp(tv4);
  auto tv6 = sub(tv1, tv2);
  auto tv7 = exp(tv6);
  auto scan_res2 = prefixSum(
      tv7, {1}, tv5, /*return_exclusive=*/false, /*return_reduction=*/true);
  auto tv8 = scan_res2.inclusive;
  auto tv9 = scan_res2.reduction;
  auto tv10 = set(tv8);
  auto tv11 = set(tv9);
  fusion.addOutput(tv11);
  fusion.addOutput(tv10);
  auto unscheduled_fusion = fusion;

  fusion.printMath();

  // Same as InclusiveScan
  const auto& scan_outputs = {tv2, tv3, tv8, tv9};

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
    if (tv == tv9 || tv == tv8) {
      compute_with_pos = 1;
    }
    tv->computeWith(compute_with_pos);
    std::cout << "\ntv: " << tv->toString() << std::endl;
    for (auto v : tv->definition()->inputs()) {
      std::cout << "pv: " << v->toString() << std::endl;
      v->as<TensorView>()->inlineAt(-1);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  auto aten_max = at::amax(input, {1});
  auto aten_exp = at::exp(input - aten_max.unsqueeze(1));
  auto aten_sum = at::sum(aten_exp, {1});
  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_sum, 1e-5, 1e-8, true));
}
// This is a simplified version of FlashAttention that does not circular buffer
// the inputs or use mma instructions, but has the same general computation
// pattern.
//
// Dao et al. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention
// with IO-Awareness. https://arxiv.org/abs/2205.14135
TEST_F(ScanTest, BlockedQKTranspose) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int64_t N = 12; // sequence length
  int64_t D = 5; // hidden dimension size
  int64_t Br = 2; // block size for rows
  int64_t Bc = 3; // block size for columns
  int64_t Tr = N / Br; // Total number of row blocks for Q, 4
  int64_t Tc = N / Bc; // Total number of column blocks for K^T, 8
  // [N,D] --> [Tr, Br, D]
  auto Q = makeConcreteTensor({Tr, Br, 1, 1, D});
  auto K = makeConcreteTensor({1, 1, Tc, Bc, D});
  fusion->addInput(Q);
  fusion->addInput(K);

  // Sij = Q_i @ K_j^T
  // QK --> [Tr, Br, Tc, Bc, D]
  auto QK = mul(Q, K);
  // S --> [Tr, Br, Tc, Bc]
  auto S = sum(QK, {4});

  fusion->addOutput(S);
  fusion->printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aQ = at::randn({Tr, Br, 1, 1, D}, options);
  auto aK = at::randn({1, 1, Tc, Bc, D}, options);
  auto aten_Q = aQ.reshape({N, D});
  auto aten_K = aK.reshape({N, D});
  auto aten_S =
      at::matmul(aten_Q, aten_K.transpose(-2, -1)).reshape({Tr, Br, Tc, Bc});
  std::cout << "aten_S: " << aten_S << std::endl;
  KernelExecutor ke;
  ke.compile(fusion.get(), {aQ, aK});
  auto outputs = ke.run({aQ, aK});
  std::cout << "outputs[0]: " << outputs[0] << std::endl;
  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_S, 1e-4, 1e-6, true));
}

TEST_F(ScanTest, BlockedQKTransposeV) {
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

  // Sij = Q_i @ K_j^T (this computes attention scores)
  // QK --> [Tr, Br, Tc, Bc, D]
  auto QK = mul(Q, K);
  // S --> [Tr, Br, Tc, Bc] (sum over hidden dimension for dot product)
  auto S = sum(QK, {4});

  // Apply attention weights S to values V
  // S: [Tr, Br, Tc, Bc], V: [1, 1, Tc, Bc, D]
  // Need to broadcast S to [Tr, Br, Tc, Bc, 1] and multiply with V
  auto S_bcast =
      broadcast(S, {false, false, false, false, true}); // [Tr, Br, Tc, Bc, 1]
  auto SV = mul(S_bcast, V); // [Tr, Br, Tc, Bc, D]

  // Sum over key/value sequence dimensions (Tc, Bc) to get final output
  auto O = sum(SV, {2, 3}); // [Tr, Br, D]

  fusion->addOutput(O);
  fusion->printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto aQ = at::randn({Tr, Br, 1, 1, D}, options);
  auto aK = at::randn({1, 1, Tc, Bc, D}, options);
  auto aV = at::randn({1, 1, Tc, Bc, D}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {aQ, aK, aV});
  auto outputs = ke.run({aQ, aK, aV});
  std::cout << "outputs[0]: " << outputs[0] << std::endl;

  auto aten_Q = aQ.reshape({N, D});
  auto aten_K = aK.reshape({N, D});
  auto aten_V = aV.reshape({N, D});
  auto aten_S = at::matmul(aten_Q, aten_K.transpose(-2, -1));
  auto aten_O = at::matmul(aten_S, aten_V).reshape({Tr, Br, D});
  std::cout << "aten_O: " << aten_O << std::endl;

  EXPECT_TRUE(
      at::allclose(outputs[0].as<at::Tensor>(), aten_O, 1e-4, 1e-6, true));
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
// TEST_F(ScanTest, OnlineSoftmax) {
//   EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

//   auto fusion = std::make_unique<Fusion>();
//   FusionGuard fg(fusion.get());

//   auto x = makeSymbolicTensor(1);
//   fusion->addInput(x);
//   auto x_cache = set(x);
//   int64_t scan_dim = 0;

//   // Online normalizer for softmax: https://arxiv.org/abs/1805.02867
//   //
//   // Given x[i] for i=0 .. N-1:
//   //
//   //   m[-1] = -infinity
//   //   d[-1] = 0
//   //   for j = 0 .. N-1
//   //     m[j] = max(m[j-1], x[j])
//   //     d[j] = d[j-1] * exp(m[j-1] - m[j]) + exp(x[j] - m[j])
//   //
//   // Final denominator is d[N-1]
//   TensorView* m_prev =  scan(x_cache, {scan_dim}, BinaryOpType::Max,
//   /*is_exclusive=*/true); // m[i-1] TensorView* m =
//   binaryOp(BinaryOpType::Max, x_cache, m_prev);
//   // normalize by running max and exponentiate
//   TensorView* exp_x_m = exp(sub(x_cache, m));
//   // Discount factor is exponentiated delta: exp(m[i-1] - m[i])
//   TensorView* discount = exp(sub(m_prev, m));

//   auto denoms = prefixSum(exp_x_m, scan_dim, discount);

//   auto norm_factor = reductionOp(
//       BinaryOpType::RHS,
//       {scan_dim},
//       /*init=*/fusion->zeroVal(DataType::Float),
//       denoms);

//   auto full_max = reductionOp(
//       BinaryOpType::RHS,
//       {scan_dim},
//       /*init=*/neg_infty,
//       m);

//   auto max_bcast = broadcast(full_max, {true});
//   auto norm_factor_bcast = broadcast(norm_factor, {true});
//   // Recompute numerator
//   auto numer = exp(sub(set(x), max_bcast));

//   auto result = div(numer, norm_factor_bcast);

//   fusion->addOutput(result);

//   // Don't cache inputs for this fusion because we will need to recompute
//   // exp((x-m)) using the final max in a separate loop so caching would mean
//   // we'd need to hold the whole tensor in registers. Instead, we manually
//   call
//   // set(x) twice in the definition above to give us two separate caches.
//   scheduler_utils::cacheAndForkOutputs(fusion.get(), /*unroll=*/true);

//   // We don't inline the scans past the scan dimension
//   std::unordered_set<IterDomain*> uninlineable_ids;
//   for (TensorView* tv : {m, m_prev, denoms}) {
//     for (IterDomain* id : tv->getLoopDomain()) {
//       uninlineable_ids.insert(id);
//     }
//   }

//   inlineMost(uninlineable_ids);

//   // These TVs are not inlined, but instead we set computeWith on them
//   for (TensorView* tv : {m, m_prev, denoms}) {
//     tv->computeWith(-1);
//     for (Val* v : tv->definition()->inputs()) {
//       // By using `uninlineable_ids` above, we prevent producers of scan from
//       // inlining with the ScanOp past the scan dim, even though this is
//       // desired. Here we do this inlining manually instead.
//       v->as<TensorView>()->inlineAt(-1);
//     }
//   }

//   auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
//   at::Tensor t0 = at::randn({32}, options);

//   KernelExecutor ke;
//   ke.compile(fusion.get(), {t0});

//   auto cg_outputs = ke.run({t0});

//   auto ref = at::softmax(t0, 0);
//   EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), ref))
//       << " returned " << cg_outputs[0].as<at::Tensor>().item()
//       << " but expected " << ref.item();

//   // Test automatic evaluation also
//   testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
// }
} // namespace nvfuser
