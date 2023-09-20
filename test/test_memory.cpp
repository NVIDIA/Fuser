// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <regex>

#include <debug.h>
#include <fusion.h>
#include <inlining.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <options.h>
#include <scheduler/cache_policy_refiner.h>
#include <test/utils.h>
#include <test/validator.h>
#include <type.h>

namespace nvfuser {

class MemoryTest
    : public NVFuserTest,
      public testing::WithParamInterface<std::tuple<CacheOp, std::string>> {
 protected:
  void expectMatchCount(
      const std::string& text,
      const std::string& pattern,
      const int num_matches) {
    std::regex regex(pattern);
    std::smatch match;
    std::regex_search(text, match, regex);
    EXPECT_EQ(match.size(), num_matches)
        << "Expect " << pattern << " to occur " << num_matches << " time(s).";
  }
};

TEST_P(MemoryTest, LoadCache) {
  CacheOp cache_op = std::get<0>(GetParam());
  std::string cache_op_str = std::get<1>(GetParam());

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 =
      ops::newValLike(tv0, tv0->getDataType().value())->as<TensorView>();
  IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, tv1, tv0, cache_op);
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

  inlineMost();

  at::Tensor input = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor expected_output = input + 1.0f;

  FusionExecutor fe;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    fe.compileFusion(&fusion, {input});
  }

  // Verify PTX.
  const executor_utils::CompiledKernel compiled_kernel = fe.compiledKernel();
  std::string ptx(compiled_kernel.ptx.begin(), compiled_kernel.ptx.end());
  std::regex regex(R"(ld\.global\.)" + cache_op_str + R"(\.\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel.ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel.ptx_filename);

  // Verify output tensors.
  std::vector<at::Tensor> actual_ts = fe.runFusion({input});
  testValidate(
      &fusion, actual_ts, {input}, {expected_output}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    CacheGlobalLoads,
    MemoryTest,
    testing::Values(
        std::make_tuple(CacheOp::AllLevels, "ca"),
        std::make_tuple(CacheOp::Global, "cg"),
        std::make_tuple(CacheOp::Streaming, "cs")),
    [](const testing::TestParamInfo<std::tuple<CacheOp, std::string>>& info) {
      std::ostringstream os;
      os << std::get<0>(info.param);
      return os.str();
    });

// Use ld.cs when loading streaming data and ld.ca otherwise.
TEST_F(MemoryTest, RefineCachePolicy) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv_a = makeContigTensor(2);
  TensorView* tv_b = makeContigTensor(1);
  fusion.addInput(tv_a);
  fusion.addInput(tv_b);
  TensorView* tv_a2 = set(tv_a);
  TensorView* tv_b2 = set(tv_b);
  TensorView* tv_c = add(tv_a2, tv_b2);
  TensorView* tv_c2 = set(tv_c);
  fusion.addOutput(tv_c2);

  tv_a2->split(1, 4); // Vector.
  tv_a2->split(1, 32); // Thread.
  // TODO: can the propagator automatically split tv_b2?
  tv_b2->split(0, 4); // Vector.
  tv_b2->split(0, 32); // Thread.
  TransformPropagatorWithCheck propagator(tv_a2);
  MaxRootDomainInfoSpanningTree(tv_a2).traverse(&propagator);

  tv_a2->axis(0)->parallelize(ParallelType::BIDy);
  tv_a2->axis(1)->parallelize(ParallelType::BIDx);
  tv_a2->axis(2)->parallelize(ParallelType::TIDx);
  tv_a2->axis(3)->parallelize(ParallelType::Vectorize);
  // TODO: can parallelizeAllLike parallelize b2 automatically?
  tv_b2->axis(0)->parallelize(ParallelType::BIDx);
  tv_b2->axis(1)->parallelize(ParallelType::TIDx);
  tv_b2->axis(2)->parallelize(ParallelType::Vectorize);
  scheduler_utils::parallelizeAllLike(tv_a2, {tv_c2});

  refineCachePolicy(&fusion);

  inlineMost();

  at::Tensor a = at::randn(
      {1024, 1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor b = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor c = a + b;

  FusionExecutor fe;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    fe.compileFusion(&fusion, {a, b});
  }

  // Verify PTX.
  const executor_utils::CompiledKernel compiled_kernel = fe.compiledKernel();
  std::string ptx(compiled_kernel.ptx.begin(), compiled_kernel.ptx.end());
  expectMatchCount(ptx, R"(ld\.global\.ca\.v4\.\S+)", 1);
  expectMatchCount(ptx, R"(ld\.global\.cs\.v4\.\S+)", 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel.ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel.ptx_filename);

  std::vector<at::Tensor> actual_outputs = fe.runFusion({a, b});
  testValidate(&fusion, actual_outputs, {a, b}, {c}, __LINE__, __FILE__);
}

class TMATest : public NVFuserTest {
  void SetUp() override {
    // requires Hopper or newer
    if (!deviceMajorMinorCheck(9)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

TEST_F(TMATest, StoreCompleteTensor1D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor2D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor3D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor4D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMATest, StoreCompleteTensor5D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(5);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);
  tv2->axis(4)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, {DataType::Int32});
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

} // namespace nvfuser
