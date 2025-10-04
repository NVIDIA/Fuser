// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/executor.h>
#include <scheduler/tools/cub_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class ArgsortTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

// Parameterized test fixture for BasicExecution with different data types
class ArgsortTestBasicExecution
    : public ArgsortTest,
      public ::testing::WithParamInterface<DataType> {
 protected:
  void runBasicExecutionTest(DataType data_type) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Create input tensor [4, 8] with specified data type
    std::vector<int64_t> shape = {4, 8};
    auto tv0 = makeContigConcreteTensor(shape, data_type);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    // Create argsort operation along dimension 1
    auto tv2 = argsort(tv1, 1, /*descending=*/false, /*stable=*/true);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    // Create test input data with appropriate tensor options
    at::TensorOptions options;
    at::Tensor input =
        at::randint(
            -100,
            100,
            {4, 8},
            at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0))
            .to(data_type_to_aten(data_type));

    for (auto tv : {tv1, tv2, tv3}) {
      tv->axis(0)->parallelize(ParallelType::BIDx);
      tv->axis(1)->parallelize(ParallelType::TIDx);
    }

    // Execute the fusion
    KernelExecutor ke;
    ke.compile(&fusion, {input});
    auto outputs = ke.run({input});

    // Verify the output
    testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
  }
};

TEST_P(ArgsortTestBasicExecution, ParameterizedBasicExecution) {
  runBasicExecutionTest(GetParam());
}

// Instantiate parameterized tests for different data types
INSTANTIATE_TEST_SUITE_P(
    ArgsortTest,
    ArgsortTestBasicExecution,
    ::testing::Values(
        DataType::Float,
        DataType::Half,
        DataType::BFloat16,
        DataType::Int),
    [](const ::testing::TestParamInfo<DataType>& info) {
      auto data_type = info.param;
      if (data_type == DataType::Float)
        return std::string("Float");
      if (data_type == DataType::Half)
        return std::string("Half");
      if (data_type == DataType::BFloat16)
        return std::string("BFloat16");
      if (data_type == DataType::Int)
        return std::string("Int");
      return std::string("Unknown");
    });

TEST_F(ArgsortTest, ZeroDimensionalInput) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);

  auto tv2 = argsort(tv0, -1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ArgsortTest, Predication) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {100};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, -1);
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

TEST_F(ArgsortTest, Grouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 101};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, -1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(-1, items_per_thread);
    // [i0, i1/S, S]

    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(-2)->parallelize(ParallelType::TIDx);
    if (tv->definition()->isA<ArgsortOp>()) {
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

// Grouping must be done with the innermost subregion of the argsort ID
TEST_F(ArgsortTest, InvalidGrouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 101};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, -1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(-1, items_per_thread, true);
    // [i0, S, i1/S]

    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
    if (tv->definition()->isA<ArgsortOp>()) {
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

// Outer argsort with grouping. Scheduling is not ideal at all but
// should work.
TEST_F(ArgsortTest, OuterArgsortWithGrouping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {10, 20};

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, 0);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  const int64_t items_per_thread = 4;

  for (auto tv : {tv1, tv2, tv3}) {
    // [i0, i1]
    tv->split(0, items_per_thread);
    // [i0/S, S, i1]

    // The argsort dimension must be parallelized with TID, so map BID
    // to the inner dimension, which is not ideal but this is required
    // for now.
    tv->axis(0)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::BIDx);
    if (tv->definition()->isA<ArgsortOp>()) {
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

// Make sure the shared memory work buffer is reused correctly
TEST_F(ArgsortTest, BufferSync) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape = {4096};
  auto tv0 = makeContigConcreteTensor(shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = argsort(tv1, 0, /*descending=*/false, /*stable=*/true);
  auto tv3 = argsort(tv2, 0, /*descending=*/false, /*stable=*/true);
  auto tv4 = argsort(tv3, 0, /*descending=*/false, /*stable=*/true);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  for (auto tv : fusion.allTvs()) {
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::TIDx);
    if (tv->definition()->isA<ArgsortOp>()) {
      tv->axis(1)->parallelize(ParallelType::Group);
    }
  }

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, shape[0], shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // Verify the output
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

class ArgsortParameterizedWithBlockandBatch
    : public ArgsortTest,
      public ::testing::WithParamInterface<std::tuple<int, int, bool>> {};

TEST_P(ArgsortParameterizedWithBlockandBatch, SharedMemoryRequirement) {
  DisableOptionsGuard disable_options_guard;
  // Avoid using magic zero to make the estimation simpler
  DisableOptionsGuard::getCurOptions().set(DisableOption::MagicZero);
  // Avoid insertion of segmenter_set
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);

  const auto [size, batch, has_extra] = GetParam();

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
  if (has_extra) {
    auto tv7 = castOp(dtype_extra, tv0);
    auto tv8 = argsort(tv7, 0);
    auto tv9 = set(tv8);
    fusion.addOutput(tv9);
  }

  for (auto tv : fusion.allTvs()) {
    if (batch > 1) {
      tv->split(-1, batch);
      if (tv->isDefinitionType<ArgsortOp>()) {
        tv->axis(-1)->parallelize(ParallelType::Group);
      }
    }
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, shape[0], shape, options);

  scheduler_tools::CubSharedMemoryBuffer smem_buffer;
  smem_buffer.registerArgsort(ceilDiv(size, batch), batch, dtype);
  if (has_extra) {
    smem_buffer.registerArgsort(ceilDiv(size, batch), batch, dtype_extra);
  }
  const int64_t expected_size = smem_buffer.getTotalSizeInBytes();

  const int64_t available_capacity =
      at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;

  const bool has_enough_capacity = expected_size <= available_capacity;

  KernelExecutor ke;
  if (has_enough_capacity) {
    ke.compile(&fusion, {t0});
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
    EXPECT_EQ(expected_size, ke.getStaticSmemSize())
        << "Actual static shared memory size was different";
  } else {
    // Compilation should fail
    EXPECT_THAT(
        [&]() { ke.compile(&fusion, {t0}); },
        testing::Throws<nvfuser::nvfError>());
  }
};

INSTANTIATE_TEST_SUITE_P(
    ,
    ArgsortParameterizedWithBlockandBatch,
    testing::Combine(
        testing::Values(128, 256, 512, 1024),
        testing::Range(1, 8),
        testing::Bool(),
        testing::Bool()),
    [](const auto& info) {
      std::ostringstream os;
      os << std::get<0>(info.param) << "_" << std::get<1>(info.param) << "_"
         << std::get<2>(info.param);
      return os.str();
    });

} // namespace nvfuser
