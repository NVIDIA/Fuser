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
#include <inlining.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <type.h>

namespace nvfuser {

class LoadTest : public NVFuserTest {};

TEST_F(LoadTest, LoadCache) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 =
      ops::newValLike(tv0, tv0->getDataType().value())->as<TensorView>();
  IrBuilder::create<LoadStoreOp>(
      LoadStoreOpType::Set, tv1, tv0, CacheOp::AllLevels);
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  TensorView* tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->split(0, 4);
  tv1->split(0, 32);
  TransformPropagatorWithCheck propagator(tv1);
  MaxRootDomainInfoSpanningTree(tv1).traverse(&propagator);

  // Parallelize LoadStoreOps. Other TensorViews don't support vectorization.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  scheduler_utils::parallelizeAllLike(tv1, {tv3});

  // The vector dimension can't be inlined.
  std::unordered_set<IterDomain*> uninlinable;
  for (TensorView* tv : ir_utils::allTvs(&fusion)) {
    if (tv->nDims() == 3) {
      uninlinable.insert(tv->axis(2));
    }
  }
  inlineMost(uninlinable);

  at::Tensor input = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
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

  std::regex regex(R"(ld\.global\.ca\.\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  std::vector<at::Tensor> actual_ts = fe.runFusion({input});
  testValidate(&fusion, actual_ts, {input}, {expected_output}, __LINE__, __FILE__);
}

} // namespace nvfuser
