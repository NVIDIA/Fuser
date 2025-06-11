// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
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

class TopKTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

// Parameterized test fixture for BasicExecution with different data types
class TopKTestBasicExecution : public TopKTest,
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
    // Create topk operation along dimension 1, k=3, largest=true, sorted=true
    // Create k as a constant Val (not a fusion input)
    auto k_val = IrBuilder::create<Val>(3L, DataType::Int);
    auto topk_result = topk(tv1, k_val, 1, /*largest=*/true, /*sorted=*/true);
    auto tv_values = topk_result.values;
    auto tv_indices = topk_result.indices;
    auto tv_values_out = set(tv_values);
    auto tv_indices_out = set(tv_indices);
    fusion.addOutput(tv_values_out);
    fusion.addOutput(tv_indices_out);

    // Create test input data with appropriate tensor options
    at::TensorOptions options;
    at::Tensor input;

    if (data_type == DataType::Float) {
      options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
      input = at::randn({4, 8}, options);
    } else if (data_type == DataType::Half) {
      options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
      input = at::randn({4, 8}, options);
    } else if (data_type == DataType::BFloat16) {
      options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
      input = at::randn({4, 8}, options);
    } else if (data_type == DataType::Int) {
      options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
      // For integer types, use randint to avoid floating point values
      input = at::randint(-100, 100, {4, 8}, options);
    } else {
      NVF_ERROR(false, "Unsupported data type for TopKTestBasicExecution");
    }

    // Parallelization strategy - all tensors get same parallelization
    for (auto tv :
         {tv1, tv_values, tv_indices, tv_values_out, tv_indices_out}) {
      tv->axis(0)->parallelize(ParallelType::BIDx);
      tv->axis(1)->parallelize(ParallelType::TIDx);
    }

    // Execute the fusion
    KernelExecutor ke;
    ke.compile(&fusion, {input});
    auto outputs = ke.run({input});

    // Verify the dual outputs
    testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
  }
};

TEST_P(TopKTestBasicExecution, ParameterizedBasicExecution) {
  runBasicExecutionTest(GetParam());
}

// Instantiate parameterized tests for different data types
INSTANTIATE_TEST_SUITE_P(
    TopKTest,
    TopKTestBasicExecution,
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

} // namespace nvfuser