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
  auto tv_result = scan(tv1, /*dim=*/1, BinaryOpType::Max);
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto outputs = ke.run({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
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
  auto tv_result = scan(tv1, /*dim=*/1, binary_op_type);
  auto tv_output = set(tv_result);
  fusion.addOutput(tv_output);

  // Parallelization strategy
  for (auto tv : {tv1, tv_result, tv_output}) {
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
} // namespace nvfuser
