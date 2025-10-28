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
#include <preseg_passes/decompose_reshardings.h>
#include <preseg_passes/propagate_shardings.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::_;
using testing::ElementsAre;

using ProgrommaticDependentLaunchTest = NVFuserTest;

TEST_F(ProgrommaticDependentLaunchTest, Basic) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = broadcast(tv1, {true, true});
  TensorView* tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  at::Tensor t0 = at::randn({2, 4}).cuda();
  at::Tensor t1 = at::randn({}).cuda();

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
