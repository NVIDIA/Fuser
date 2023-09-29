// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class MBarrierTest : public NVFuserTest {};

TEST_F(MBarrierTest, Simple) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({32, 32});
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;
  fe.compileFusion(&fusion);

  auto input = at::randn(
      {32, 32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputs = fe.runFusion({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

} // namespace nvfuser
