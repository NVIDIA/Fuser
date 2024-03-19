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
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
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

// Begin TMA tests

using TMATestParams = std::tuple<MmaInputSmemSwizzle, DataType, int64_t>;

class TMALdstTest : public TMATest,
                    public ::testing::WithParamInterface<TMATestParams> {
 protected:
  MmaInputSmemSwizzle swizzle;
  DataType dtype;
  int64_t dim;
  std::vector<int64_t> shape;
  std::vector<int64_t> tile;

  int64_t innerDimSize() const {
    return getBytesFromSwizzle(swizzle) / dataTypeSize(dtype);
  }

  void SetUp() override {
    TMATest::SetUp();
    swizzle = std::get<0>(GetParam());
    dtype = std::get<1>(GetParam());
    dim = std::get<2>(GetParam());

    // TODO: test less-nice shapes, for example 128 + 1 instead of 128
    // TODO: When shapes are large, I see failures in the SimpleStore test.
    // Needs to investigate why.
    switch (dim) {
      case 1:
        tile = {innerDimSize()};
        shape = {128};
        break;
      case 2:
        tile = {2, innerDimSize()};
        shape = {4, 128};
        break;
      case 3:
        tile = {2, 4, innerDimSize()};
        shape = {4, 8, 128};
        break;
      case 4:
        tile = {2, 4, 8, innerDimSize()};
        shape = {4, 8, 16, 128};
        break;
      case 5:
        tile = {2, 4, 8, 16, innerDimSize()};
        shape = {4, 8, 16, 32, 128};
        break;
      default:
        NVF_ERROR(false, "Invalid dimension");
    }
  }
};

// Assuming tv is currently scheduled as a 1D flat tensor, schedule the TMA
// swizzle for it.
void scheduleTMASwizzle(TensorView* tv, int64_t swizzle_size) {
  // split as core matrices of 8 x 16B
  tv->split(-1, core_matrix_width_bytes / dataTypeSize(tv->dtype()));
  tv->split(-2, 8);
  // [N, 8, 16B]
  // swizzle the inner dim of rows of different core matrices
  tv->split(-3, swizzle_size);
  tv->split(-2, swizzle_size);
  // [N/swizzle_size, swizzle_size, 8/swizzle_size, swizzle_size, 16B]
  tv->swizzle(SwizzleType::XOR, -4, -2);
}

void markAllDimsExceptFirstAsBulk(const TensorView* tv) {
  bool skip = true;
  for (auto id : tv->getLeafDomain()) {
    if (skip) {
      skip = false;
      continue;
    }
    id->parallelize(ParallelType::Bulk);
  }
}

// Simple load/store tests:
//
// These tests launches a <<<N, 1>>> copy kernel to do global -> smem -> global
// copying. Because each block only have 1 thread, there is no need to worry
// about thread predication and synchronization.

void scheduleTile(
    std::vector<TensorView*> tvs,
    std::vector<int64_t> tile_sizes,
    MmaInputSmemSwizzle swizzle) {
  const int64_t dim = tile_sizes.size();
  const int64_t swizzle_size = getBytesFromSwizzle(swizzle) / 16;

  for (auto tv : tvs) {
    NVF_ERROR(
        dim == (int64_t)tv->nDims(), "Tile sizes must match tensor dimensions");
    // [M, N, ...]
    for (int64_t i = dim - 1; i >= 0; i--) {
      tv->split(i, tile_sizes[i]);
    }
    // [M/tile_sizes[0], tile_sizes[0], N/tile_sizes[1], tile_sizes[1], ...]
    std::unordered_map<int, int> old2new;
    for (int64_t i = 0; i < dim; i++) {
      old2new[2 * i] = i;
      old2new[2 * i + 1] = i + dim;
    }
    tv->reorder(old2new);
    // [M/tile_sizes[0], N/tile_sizes[1], ..., tile_sizes[0], tile_sizes[1],
    // ...]
    for (int64_t i = 0; i < dim - 1; i++) {
      tv->merge(0);
    }
    for (int64_t i = 0; i < dim - 1; i++) {
      tv->merge(1);
    }
    // [M/tile_sizes[0] * N/tile_sizes[1] * ..., tile_sizes[0] * tile_sizes[1] *
    // ...]
    tv->axis(0)->parallelize(ParallelType::BIDx);
    if (swizzle != MmaInputSmemSwizzle::None) {
      scheduleTMASwizzle(tv, swizzle_size);
    }
  }
}

TEST_P(TMALdstTest, SimpleLoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(dim, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  scheduleTile({tv1, tv2}, tile, swizzle);
  tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  markAllDimsExceptFirstAsBulk(tv1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_P(TMALdstTest, SimpleStore) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(dim, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  scheduleTile({tv1, tv2}, tile, swizzle);
  tv1->setAllocationDomain(tv1->getLeafDomain(), true);
  markAllDimsExceptFirstAsBulk(tv2);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_FALSE(PredicatedChecker::isPredicated(tv2, fe.kernel()));
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));

  auto cg_outputs = fe.runFusion({t0});
  // std::cout << "t0:\n" << t0 << std::endl;
  // std::cout << "cg_outputs[0]:\n" << cg_outputs[0] << std::endl;
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

std::string testNameTMALdstTest(
    const testing::TestParamInfo<TMATestParams>& info) {
  auto swizzle = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto dim = std::get<2>(info.param);
  std::stringstream ss;
  ss << dim << "D"
     << "_" << toString(swizzle) << "_" << dtype;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    TMALdstTest,
    TMALdstTest,
    testing::Combine(
        kAllSmemSwizzleModes,
        testing::Values(DataType::Half, DataType::Float, DataType::Double),
        testing::Values(1, 2, 3, 4, 5)),
    testNameTMALdstTest);

// Advanced indexing of TMA
class TMAIndexingTest : public TMATest {};

TEST_F(TMAIndexingTest, Load2DTensorWith1DTMA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024, 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, Load1DTensorWith2DTMA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 1024);
    tv->split(1, 32);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024 * 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, NonZeroElementStride) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 32);
    tv->split(0, 32);
    tv->split(1, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
    tv->axis(3)->parallelize(ParallelType::BIDz);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(4)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024, 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// TODO: improve validation of TMA, and add tests for invalid cases.

class TMAMiscTest : public TMATest {};

// Basically just SimpleStore, but with index hoisting disabled. Because index
// hoisting is responsible making sure that tensor maps are created on the host
// and passed as kernel argument, we need to make sure that disabling index
// hoisting doesn't break this.
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

  tv2->split(0, 32);
  tv2->axis(1)->parallelize(ParallelType::Bulk);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// End TMA tests

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
