// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "ops/all_ops.h"
#include "scheduler/tools/inlining.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

namespace nvfuser {

using testing::_;
using testing::ElementsAre;

class ProgrammaticDependentLaunchTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  }
};

TEST_F(ProgrammaticDependentLaunchTest, Basic) {
  std::shared_ptr<Fusion> fusion_ptr = std::make_shared<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  // Clone prescheduled fusion for validation
  std::shared_ptr<Fusion> presched_fusion = std::make_shared<Fusion>(fusion);

  // Cache the inputs and output
  TensorView* tv0_cached = tv0->cacheAfter();
  TensorView* tv1_cached = tv1->cacheAfter();
  TensorView* tv2_cached = tv2->cacheBefore();

  TensorView* grid_wait = wait_for_prior_grid({tv0, tv1});
  tv0_cached->addDependency(grid_wait);
  tv1_cached->addDependency(grid_wait);

  TensorView* grid_launch = launch_dependent_grid({tv2_cached});
  tv2->addDependency(grid_launch);

  constexpr int tdx = 128;
  constexpr int vectorize_factor = 4;
  tv2->merge(0, 1);
  tv2->split(-1, vectorize_factor);
  tv2->split(-2, tdx);

  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  // Parallelize the cached tensor
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-3)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  tv0_cached->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cached->axis(-1)->parallelize(ParallelType::Vectorize);
  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  at::Tensor t0 = at::randn({8, 512}).cuda();
  at::Tensor t1 = at::randn({8, 512}).cuda();

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});

  // Validate that the kernel is compiled with PDL support
  const kir::KernelSummary& summary = ke.compiledKernel()->kernel()->summary();
  EXPECT_TRUE(summary.enable_programmatic_dependent_launch);

  // Validate that the kernel contains the expected PDL operations
  const auto& kernel_exprs = ke.compiledKernel()->kernel()->exprs();
  EXPECT_EQ(
      std::count_if(
          kernel_exprs.begin(),
          kernel_exprs.end(),
          [](const Expr* expr) { return expr->isA<LaunchDependentGridOp>(); }),
      1);
  EXPECT_EQ(
      std::count_if(
          kernel_exprs.begin(),
          kernel_exprs.end(),
          [](const Expr* expr) { return expr->isA<WaitForPriorGridOp>(); }),
      1);

  auto cg_outputs = ke.run({t0, t1});
  testValidate(presched_fusion.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
