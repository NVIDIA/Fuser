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

TEST_F(ArgsortTest, BasicExecution) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input tensor [4, 8] with float data
  std::vector<int64_t> shape = {4, 8};
  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  // Create argsort operation along dimension 1
  auto tv2 = argsort(tv1, 1, /*descending=*/false, /*stable=*/false);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // Create test input data
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::randn({4, 8}, options);

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

} // namespace nvfuser
