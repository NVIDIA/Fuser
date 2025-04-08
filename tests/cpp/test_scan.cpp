// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/arith.h>
#include <options.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using ScanTest = NVFuserTest;

// Simple test case for defining a scan
TEST_F(ScanTest, Concrete1D) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({32});
  fusion->addInput(tv0);

  auto tv1 = prefixSum(tv0, /*dim=*/-1, /*discount_factor=*/nullptr);

  fusion->addOutput(tv1);

  fusion->printMath();

  tv0->cacheAfter();
  tv1->cacheBefore();
  inlineMost();

  fusion->printMath();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
