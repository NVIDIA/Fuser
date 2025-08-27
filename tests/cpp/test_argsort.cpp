// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
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

} // namespace nvfuser
