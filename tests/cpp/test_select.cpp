// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class SelectTest : public NVFuserTest {
 protected:
  void SetUp() override {
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
    NVFuserTest::SetUp();
  }
};

TEST_F(SelectTest, Pointwise) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
}

TEST_F(SelectTest, Reduction) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0});
  auto tv5 = sum(tv2, {1});
  auto tv6 = sum(tv3, {0, 1});

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
}

TEST_F(SelectTest, Persistent) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  auto index = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(tv0);
  fusion.addInput(index);

  auto tv1 = select(tv0, 0, index);
  auto tv2 = select(tv0, 1, index);
  auto tv3 = select(tv0, 2, index);

  auto tv4 = sum(tv1, {0}, true);
  auto tv5 = sum(tv2, {1}, true);
  auto tv6 = sum(tv3, {0, 1}, true);

  auto tv7 = add(tv1, tv4);
  auto tv8 = add(tv2, tv5);
  auto tv9 = add(tv3, tv6);

  fusion.addOutput(tv7);
  fusion.addOutput(tv8);
  fusion.addOutput(tv9);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  int x = 31, y = 65, z = 103, idx = 21;

  at::Tensor t0 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, idx});

  testValidate(&fusion, cg_outputs, {t0, idx}, __LINE__, __FILE__);
}

} // namespace nvfuser
