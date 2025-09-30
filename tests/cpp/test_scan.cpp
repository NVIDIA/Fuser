// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <fusion.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ScanTest = NVFuserTest;

// Basic functionality test for scan with Add operation (cumsum)
TEST_F(ScanTest, BasicScanAdd) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Add);
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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Max);
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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Min);
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
  auto tv_result = scan(tv0, /*dim=*/1, BinaryOpType::Mul);
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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
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
  auto tv_result = scan(tv0, /*dim=*/0, BinaryOpType::Add);
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
  auto tv2 = scan(tv1, /*dim=*/1, BinaryOpType::Add);

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
  auto tv4 = scan(tv3, /*dim=*/1, BinaryOpType::Add);

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
    auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Add);
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
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Add);
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
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Max);
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
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Min);
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
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Mul);
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
  auto tv_scan1 = scan(tv1, /*dim=*/1, BinaryOpType::Add);

  // Add operation between scans
  auto tv_add = add(tv_scan1, IrBuilder::create<Val>(1.0));

  // Second scan operation (Max)
  auto tv_scan2 = scan(tv_add, /*dim=*/1, BinaryOpType::Max);

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

  auto tv1 = set(tv0);
  auto tv2 = scan(tv1, -1, BinaryOpType::Add);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Non-divisible split. 128 threads will be launched. The last 28
  // threads need to be predicated out.
  for (auto tv : {tv1, tv2, tv3}) {
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

TEST_F(ScanTest, Grouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 101};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = scan(tv1, -1, BinaryOpType::Add);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(-1, items_per_thread);
    // [i0, i1/S, S]

    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(-2)->parallelize(ParallelType::TIDx);
    if (tv->definition()->isA<ScanOp>()) {
      tv->axis(-1)->parallelize(ParallelType::Group);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Grouping must be done with the innermost subregion of the scan ID
TEST_F(ScanTest, InvalidGrouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 101};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = scan(tv1, -1, BinaryOpType::Add);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(-1, items_per_thread, true);
    // [i0, S, i1/S]

    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
    if (tv->definition()->isA<ScanOp>()) {
      tv->axis(-2)->parallelize(ParallelType::Group);
    }
  }

  // The use of the group type is invalid. GpuLower should issue an
  // exception.
  EXPECT_THAT(
      [&]() { GpuLower lower(&fusion); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Invalid ID to group")));
}

// Outer scan with grouping. Scheduling is not ideal at all but
// should work.
TEST_F(ScanTest, OuterScanWithGrouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 20};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = scan(tv1, 0, BinaryOpType::Add);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(0, items_per_thread);
    // [i0/S, S, i1]

    // The scan dimension must be parallelized with TID, so map BID
    // to the inner dimension, which is not ideal but this is required
    // for now.
    tv->axis(0)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    if (tv->definition()->isA<ScanOp>()) {
      tv->axis(1)->parallelize(ParallelType::Group);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
