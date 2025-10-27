// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include "fusion.h"
#include "fusion_guard.h"
#include "ops/all_ops.h"
#include "runtime/executor.h"
#include "tests/cpp/utils.h"
#include "validator_utils.h"

namespace nvfuser {

using MathOptTest = NVFuserTest;

TEST_F(MathOptTest, FastMathTanh) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = tanh(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    EnableOptionsGuard enable_opt_guard;
    EnableOptionsGuard::getCurOptions().set(EnableOption::FastMath);
    ke.compile(fusion.get(), {t0});
  }

  // Verify PTX, result validation is skipped since reference won't use fast
  // math.
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx_string(
      compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  EXPECT_TRUE(ptx_string.find("tanh.approx.f32") != std::string::npos);
}

using NanReductionTest = NVFuserFixtureParamTest<BinaryOpType>;

TEST_P(NanReductionTest, Test) {
  // Check NAN reduction behavior for several cases:
  // 1. No NAN input -> no NAN output
  // 2. Single NAN input -> NAN output only for min/max (not fmin/fmax)
  // 3. All NAN input -> NAN output

  BinaryOpType opType = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  Val* init = ops::binOpIdentity(opType, tv0->dtype());
  auto tv1 = reductionOp(opType, {0}, init, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  // No-NAN input
  auto cg_outputs = ke.run({t0});
  EXPECT_FALSE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());

  // Single NAN input
  t0[0] = std::numeric_limits<float>::quiet_NaN();
  cg_outputs = ke.run({t0});
  bool any_nan =
      at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>();
  if (opType == BinaryOpType::FMax || opType == BinaryOpType::FMin) {
    EXPECT_FALSE(any_nan);
  } else {
    EXPECT_TRUE(any_nan);
  }

  // All NAN input
  t0 = at::full({32}, std::numeric_limits<float>::quiet_NaN(), options);
  cg_outputs = ke.run({t0});
  EXPECT_TRUE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    MathOptTest,
    NanReductionTest,
    ::testing::Values(
        BinaryOpType::Max,
        BinaryOpType::FMax,
        BinaryOpType::Min,
        BinaryOpType::FMin),
    [](const testing::TestParamInfo<BinaryOpType>& info) -> std::string {
      std::stringstream ss;
      ss << info.param;
      return sanitizeTestName(ss.str());
    });

using FMinFMaxPromotionTest = NVFuserFixtureParamTest<int>;
TEST_P(FMinFMaxPromotionTest, Test) {
  // Test and validate max reductions under a couple different topologies.
  // Ensure some topologies include "fmax" in the kernel.

  int testIndex = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion->addInput(tv2);

  bool expectFMax = false;

  if (testIndex == 1) {
    TensorView* tv3 = add(max(tv0, {0, 1}), tv0);
    TensorView* tv4 = add(tv3, sum(tv3, {0, 1}));
    fusion->addOutput(tv4);
    expectFMax = true;
  }

  if (testIndex == 2) {
    TensorView* tv3 = add(max(tv0, {0, 1}), sum(tv0, {0, 1}));
    fusion->addOutput(tv3);
    expectFMax = true;
  }

  if (testIndex == 3) {
    TensorView* tv3 = add(tv0, tv1);
    TensorView* tv4 = add(tv3, broadcast(max(tv3, {1}), {false, true}));
    TensorView* tv5 = broadcast(sum(add(tv4, tv2), {1}), {false, true});
    TensorView* tv6 = add(tv4, tv5);
    fusion->addOutput(tv6);
    expectFMax = true;
  }

  if (testIndex == 4) {
    TensorView* tv3 = add(max(tv0, {1}), sum(tv1, {1}));
    fusion->addOutput(tv3);
    expectFMax = false;
  }

  if (testIndex == 5) {
    TensorView* tv3 = add(tv0, tv1);
    TensorView* tv4 = broadcast(max(tv3, {1}), {false, true});
    TensorView* tv5 = broadcast(sum(add(tv4, tv2), {1}), {false, true});
    TensorView* tv6 = add(tv4, tv5);
    fusion->addOutput(tv6);
    expectFMax = false;
  }

  if (testIndex == 6) {
    TensorView* tv3 = add(tv0, broadcast(max(tv0, {1}), {false, true}));
    fusion->addOutput(tv3);
    expectFMax = false;
  }

  if (testIndex == 7) {
    TensorView* tv3 = add(abs(max(tv0, {0, 1})), sum(tv0, {0, 1}));
    fusion->addOutput(tv3);
    expectFMax = true;
  }

  if (testIndex == 8) {
    TensorView* tv3 = add(max(tv0, {0, 1}), max(tv0, {0, 1}));
    fusion->addOutput(tv3);
    expectFMax = true;
  }

  if (testIndex == 9) {
    TensorView* tv3 = add(broadcast(abs(max(tv0, {1})), {false, true}), tv0);
    TensorView* tv4 = add(tv3, broadcast(abs(sum(tv3, {0})), {true, false}));
    fusion->addOutput(tv4);
    expectFMax = false;
  }

  fusion->printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 32}, options);
  t0[0][0] = std::numeric_limits<float>::quiet_NaN();

  auto t1 = at::randn({16, 32}, options);
  auto t2 = at::randn({16, 32}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1, t2}, __LINE__, __FILE__);

  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();

  auto& group = kernel_runtime->fusionSegments()->groups()[0];

  const auto* ke =
      kernel_runtime->executors().at(group->groupId())->as<KernelExecutor>();
  std::string kernel_code = ke->compiledKernel()->kernelString();

  if (expectFMax) {
    EXPECT_THAT(kernel_code, ::testing::HasSubstr("fmax("));
  } else {
    EXPECT_THAT(kernel_code, ::testing::Not(::testing::HasSubstr("fmax(")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MathOptTest,
    FMinFMaxPromotionTest,
    ::testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 9),
    [](const testing::TestParamInfo<int>& info) -> std::string {
      std::stringstream ss;
      ss << info.param;
      return sanitizeTestName(ss.str());
    });

} // namespace nvfuser
