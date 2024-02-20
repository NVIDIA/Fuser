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

using MemoryTestParams = std::tuple<CacheOp, std::string>;

class MemoryTest : public NVFuserFixtureParamTest<MemoryTestParams> {
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
  const executor_utils::CompiledKernel& compiled_kernel = fe.compiledKernel();
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
    [](const testing::TestParamInfo<MemoryTestParams>& info) {
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

  tv_a2->merge(0);
  tv_a2->split(0, 4);
  tv_a2->split(0, 32);
  TransformPropagatorWithCheck propagator(tv_a2);
  MaxRootDomainInfoSpanningTree(tv_a2).traverse(&propagator);

  tv_a2->axis(0)->parallelize(ParallelType::BIDx);
  tv_a2->axis(1)->parallelize(ParallelType::TIDx);
  tv_a2->axis(2)->parallelize(ParallelType::Vectorize);
  tv_b2->axis(2)->parallelize(ParallelType::Vectorize);
  tv_c2->axis(2)->parallelize(ParallelType::Vectorize);

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
  const executor_utils::CompiledKernel& compiled_kernel = fe.compiledKernel();
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
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// Check that there is an xor "^" somewhere in the kernel
class XorFinder : private kir::IrVisitor {
 private:
  using kir::IrVisitor::dispatch;
  using kir::IrVisitor::handle;
  bool found = false;

  // We recursively goes into val's definition and its inputs and outputs, this
  // is used to prevent infinite recursion
  std::unordered_set<Expr*> visited;

  void handle(kir::TensorIndex* ti) final {
    handle(ti->index());
  }

  void handle(Val* v) {
    if (v->definition() != nullptr) {
      dispatch(v->definition());
    }
  }

  void dispatch(Expr* expr) final {
    if (found || !visited.insert(expr).second) {
      return;
    }
    if (expr->isA<kir::ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }
    if (expr->isA<BinaryOp>()) {
      auto bin_op = expr->as<BinaryOp>();
      if (bin_op->getBinaryOpType() == BinaryOpType::BitwiseXor) {
        found = true;
        return;
      }
    }
    for (auto val : expr->inputs()) {
      dispatch(val);
    }
    for (auto val : expr->outputs()) {
      dispatch(val);
    }
  }

 public:
  static bool findXor(kir::Kernel* kernel) {
    XorFinder finder;
    finder.handle(kernel->topLevelExprs());
    return finder.found;
  }
};

using TMATestParams = std::tuple<MmaInputSmemSwizzle, DataType>;

class TMALdstTest : public TMATest,
                    public ::testing::WithParamInterface<TMATestParams> {
 protected:
  MmaInputSmemSwizzle swizzle;
  DataType dtype;

  int64_t innerDimSize() const {
    return getBytesFromSwizzle(swizzle) / dataTypeSize(dtype);
  }

  int64_t swizzleSize() const {
    return getBytesFromSwizzle(swizzle) / 16;
  }

  void SetUp() override {
    TMATest::SetUp();
    swizzle = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
  }
};

// Assuming tv is currently scheduled as a 1D flat tensor, schedule the TMA
// swizzle for it.
void scheduleTMASwizzle(TensorView* tv, int64_t swizzle_size) {
  // split as core matrices of 8 x 16B
  tv->split(0, 16 / dataTypeSize(tv->dtype()));
  tv->split(0, 8);
  // [N, 8, 16B]
  // swizzle the inner dim of rows of different core matrices
  tv->split(1, swizzle_size);
  tv->split(0, swizzle_size);
  // [N/swizzle_size, swizzle_size, 8/swizzle_size, swizzle_size, 16B]
  tv->swizzle(SwizzleType::XOR, 1, 3);
}

TEST_P(TMALdstTest, LoadCompleteTensor1D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->axis(0)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, LoadCompleteTensor2D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    for (auto tv : {tv1, tv2}) {
      tv->merge(0);
      scheduleTMASwizzle(tv, swizzleSize());
    }
    tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  }
  for (auto id : tv1->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({32, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, LoadCompleteTensor3D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    for (auto tv : {tv1, tv2}) {
      tv->merge(0);
      tv->merge(0);
      scheduleTMASwizzle(tv, swizzleSize());
    }
    tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  }
  for (auto id : tv1->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, LoadCompleteTensor4D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    for (auto tv : {tv1, tv2}) {
      tv->merge(0);
      tv->merge(0);
      tv->merge(0);
      scheduleTMASwizzle(tv, swizzleSize());
    }
    tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  }
  for (auto id : tv1->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, LoadCompleteTensor5D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, -1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    for (auto tv : {tv1, tv2}) {
      tv->merge(0);
      tv->merge(0);
      tv->merge(0);
      tv->merge(0);
      scheduleTMASwizzle(tv, swizzleSize());
    }
    tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  }
  for (auto id : tv1->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, StoreCompleteTensor1D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv2->axis(0)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, StoreCompleteTensor2D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    GTEST_SKIP() << "Swizzle for TMA store is not supported yet";
  }
  for (auto id : tv2->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({32, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, StoreCompleteTensor3D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    GTEST_SKIP() << "Swizzle for TMA store is not supported yet";
  }
  for (auto id : tv2->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, StoreCompleteTensor4D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    GTEST_SKIP() << "Swizzle for TMA store is not supported yet";
  }
  for (auto id : tv2->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, StoreCompleteTensor5D) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({-1, -1, -1, -1, innerDimSize()}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  if (swizzle != MmaInputSmemSwizzle::None) {
    GTEST_SKIP() << "Swizzle for TMA store is not supported yet";
  }
  for (auto id : tv2->getLeafDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 4, 4, 4, innerDimSize()}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

std::string testNameTMALdstTest(
    const testing::TestParamInfo<TMATestParams>& info) {
  auto swizzle = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  std::stringstream ss;
  ss << toString(swizzle) << "_" << dtype;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    TMALdstTest,
    TMALdstTest,
    testing::Combine(
        kAllSmemSwizzleModes,
        testing::Values(DataType::Half, DataType::Float, DataType::Double)),
    testNameTMALdstTest);

class TMAMiscTest : public TMATest {};

// Basically just StoreCompleteTensor1D, but with index hoisting disabled.
// Because index hoisting is responsible making sure that tensor maps are
// created on the host and passed as kernel argument, we need to make sure
// that disabling index hoisting doesn't break this.
TEST_F(TMAMiscTest, DisableIndexHoisting) {
  DisableOptionsGuard opt_guard;
  opt_guard.getCurOptions().set(DisableOption::IndexHoist);

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
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

using LdMatrixTestParam = std::tuple<MmaMacro, MmaOperand>;

class LdMatrixTest : public NVFuserFixtureParamTest<LdMatrixTestParam> {
 protected:
  void SetUp() override {
    // requires Turing or newer
    if (cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }
    NVFuserTest::SetUp();
  }
};

TEST_P(LdMatrixTest, Regular) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto operand = std::get<1>(GetParam());

  bool is_a = operand == MmaOperand::A;

  int size1 = (is_a ? getM(macro) : getN(macro));

  auto tv0 = makeConcreteTensor({size1, getK(macro)}, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  auto tv2 = set(tv1);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdMatrix);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->applyMmaSwizzle(operand);
  tv3->applyMmaSwizzle(operand);

  tv3->merge(0);
  if (is_a) {
    tv3->merge(0);
  }
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({size1, getK(macro)}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_P(LdMatrixTest, Transpose) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto operand = std::get<1>(GetParam());

  bool is_a = operand == MmaOperand::A;

  int size2 = (is_a ? getM(macro) : getN(macro));

  auto tv0 = makeConcreteTensor({getK(macro), size2}, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  auto tv2 = transpose(tv1, 0, 1);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv2->applyMmaSwizzle(operand);
  tv3->applyMmaSwizzle(operand);

  tv3->merge(0);
  if (is_a) {
    tv3->merge(0);
  }
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({getK(macro), size2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    CopyUsingLdMatrix,
    LdMatrixTest,
    testing::Values(
        std::make_tuple(MmaMacro::Turing_16_8_8, MmaOperand::A),
        std::make_tuple(MmaMacro::Turing_16_8_16, MmaOperand::A),
        std::make_tuple(MmaMacro::Turing_16_8_8, MmaOperand::B),
        std::make_tuple(MmaMacro::Turing_16_8_16, MmaOperand::B),
        std::make_tuple(MmaMacro::Turing_16_16_16, MmaOperand::B),
        std::make_tuple(MmaMacro::Hopper_64_8_16, MmaOperand::A)),
    [](const testing::TestParamInfo<LdMatrixTestParam>& info) {
      std::ostringstream os;
      auto macro = std::get<0>(info.param);
      bool is_a = std::get<1>(info.param) == MmaOperand::A;
      os << (is_a ? "A" : "B") << "_" << (is_a ? getM(macro) : getN(macro))
         << "x" << getK(macro);
      return os.str();
    });

} // namespace nvfuser
