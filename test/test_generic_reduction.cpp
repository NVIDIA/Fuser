// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <inlining.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <executor_utils.h>
#include <fusion.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>


namespace nvfuser {

TEST_F(NVFuserTest, ArgMax) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto [tvmax, tvpos] = argmax(tv0);

  fusion->addOutput(tvmax);
  fusion->addOutput(tvpos);

  fusion->printMath();

  /*
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({15}, options);

  std::vector<c10::IValue> inputs = {t0};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(inputs);

  auto [refmax, refpos] = at::max(t0, 0);

  testValidate(
      fec.fusion(), outputs, inputs, {refmax, refpos}, __LINE__, __FILE__);
  */
}

} // namespace nvfuser
