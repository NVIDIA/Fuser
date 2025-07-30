// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

TEST_F(NVFuserTest, FusionHashBasic) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  size_t first_hash = fusion.hash();
  NVF_ERROR(first_hash != 0, "Fusion hash is 0");
  NVF_ERROR(first_hash == fusion.hash(), "Fusion hash is not stable");
  NVF_ERROR(
      fusion.checkDefinition(fusion),
      "Fusion definition does not match itself");

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  auto ref = t0 + 1;
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionHashSameDefinition) {
  std::unique_ptr<Fusion> fusion_first_ptr = std::make_unique<Fusion>();
  Fusion& fusion_first = *fusion_first_ptr.get();
  std::unique_ptr<Fusion> fusion_second_ptr = std::make_unique<Fusion>();
  Fusion& fusion_second = *fusion_second_ptr.get();

  // Create the first fusion.
  {
    FusionGuard fg(&fusion_first);
    auto fusion_first_tv0 = makeContigTensor(1);
    fusion_first.addInput(fusion_first_tv0);
    auto fusion_first_tv1 = add(fusion_first_tv0, IrBuilder::create<Val>(1.0));
    fusion_first.addOutput(fusion_first_tv1);
  }

  // Create a second fusion with the same definition.
  {
    FusionGuard fg(&fusion_second);
    auto fusion_second_tv0 = makeContigTensor(1);
    fusion_second.addInput(fusion_second_tv0);
    auto fusion_second_tv1 =
        add(fusion_second_tv0, IrBuilder::create<Val>(1.0));
    fusion_second.addOutput(fusion_second_tv1);
  }

  // Check that the fusion definitions match and have the same hash value.
  // NVF_ERROR(fusion_first.checkDefinition(fusion_second), "The fusion
  // definitions do not match.");
  // NVF_ERROR(fusion_second.checkDefinition(fusion_first), "The fusion
  // definitions do not match.");
  NVF_ERROR(
      fusion_first.hash() == fusion_second.hash(),
      "The hash values do not match.");

  FusionExecutorCache executor_cache(std::move(fusion_first_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  auto ref = t0 + 1;
  testValidate(&fusion_first, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionHashDifferentDefinition) {
  std::unique_ptr<Fusion> fusion_first_ptr = std::make_unique<Fusion>();
  Fusion& fusion_first = *fusion_first_ptr.get();
  std::unique_ptr<Fusion> fusion_second_ptr = std::make_unique<Fusion>();
  Fusion& fusion_second = *fusion_second_ptr.get();

  // Create the first fusion.
  {
    FusionGuard fg(&fusion_first);
    auto fusion_first_tv0 = makeContigTensor(1);
    fusion_first.addInput(fusion_first_tv0);
    auto fusion_first_tv1 = add(fusion_first_tv0, IrBuilder::create<Val>(1.0));
    fusion_first.addOutput(fusion_first_tv1);
  }

  // Create a second fusion with the same definition.
  {
    FusionGuard fg(&fusion_second);
    auto fusion_second_tv0 = makeContigTensor(1);
    fusion_second.addInput(fusion_second_tv0);
    auto fusion_second_tv1 =
        add(fusion_second_tv0, IrBuilder::create<Val>(10.0));
    fusion_second.addOutput(fusion_second_tv1);
  }

  // Check that the fusion definitions do not have the same hash value.
  NVF_ERROR(
      fusion_first.hash() != fusion_second.hash(),
      "The hash values do not match.");

  FusionExecutorCache executor_cache(std::move(fusion_first_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  auto ref = t0 + 1;
  testValidate(&fusion_first, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

} // namespace nvfuser
