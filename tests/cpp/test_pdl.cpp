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

#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::_;
using testing::ElementsAre;

using ProgrammaticDependentLaunchTest = NVFuserTest;

TEST_F(ProgrammaticDependentLaunchTest, Basic) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  fusion.printMath();

  // Cache the inputs and output
  TensorView* tv0_cached = tv0->cacheAfter();
  TensorView* tv1_cached = tv1->cacheAfter();
  tv2->cacheBefore();

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

  fusion.printMath();

  at::Tensor t0 = at::randn({8, 512}).cuda();
  at::Tensor t1 = at::randn({8, 512}).cuda();

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto cg_outputs = ke.run({t0, t1});
  testValidate(fusion_ptr.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
