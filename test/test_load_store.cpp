// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <regex>

#include <fusion.h>
#include <ops/arith.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

TEST_F(NVFuserTest, LoadCache) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  // DO NOT SUBMIT: is dtype necessary?
  TensorView* tv1 = add(tv0, FusionGuard::getCurFusion()->oneVal());
  fusion.addOutput(tv1);

  at::Tensor t0 = at::randn(
      {128}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor expected_t1 = t0 + 1.0f;

  FusionExecutor fe;
  fe.setSaveCompiledBinaryFlag(true);
  {
    DisableOptionsGuard og;
    DisableOptionsGuard::getCurOptions().set(DisableOption::CompileToSass);
    fe.compileFusion(&fusion, {t0});
  }
  std::vector<char> compiled_binary = fe.compiledBinary();
  std::string ptx(compiled_binary.begin(), compiled_binary.end());

  std::regex regex(R"(ld\.global\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  std::vector<at::Tensor> actual_ts = fe.runFusion({t0});
  testValidate(&fusion, actual_ts, {t0}, {expected_t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
