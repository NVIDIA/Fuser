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
#include <ops/alias.h>
#include <ops/arith.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

// TODO: delta debug test_gpu1.cpp FusionCacheAfter_CUDA.
TEST_F(NVFuserTest, Foo) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(2.0));
  fusion.addOutput(tv2);

  tv0->cacheAfter();

  tv2->split(-1, 4);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  // tv2->axis(3)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({16, 8}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1});
  auto outputs = fe.runFusion({input1});

  at::Tensor tv2_ref = input1 + 2.0;

  testValidate(&fusion, outputs, {input1}, {tv2_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, LoadCache) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, FusionGuard::getCurFusion()->oneVal());
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  // tv1->axis(2)->parallelize(ParallelType::Vectorize);
  // scheduler_utils::parallelizeAllLike(tv1, {tv2});

  at::Tensor input = at::randn(
      {128}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor expected_output = input + 1.0f;

  FusionExecutor fe;
  fe.setSaveCompiledBinaryFlag(true);
  {
    DisableOptionsGuard og;
    DisableOptionsGuard::getCurOptions().set(DisableOption::CompileToSass);
    fe.compileFusion(&fusion, {input});
  }
  std::vector<char> compiled_binary = fe.compiledBinary();
  std::string ptx(compiled_binary.begin(), compiled_binary.end());

  std::regex regex(R"(ld\.global\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  std::vector<at::Tensor> actual_ts = fe.runFusion({input});
  testValidate(&fusion, actual_ts, {input}, {expected_output}, __LINE__, __FILE__);
}

} // namespace nvfuser
