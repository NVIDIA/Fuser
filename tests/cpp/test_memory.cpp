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

// Begin TMA tests

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
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
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

class TMAPredicateChecker : private kir::IrVisitor {
  int64_t num_threads_;
  TMAPredicateChecker(int64_t num_threads) : num_threads_(num_threads) {}

  kir::Predicate* pred_ = nullptr;

  using kir::IrVisitor::dispatch;

  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::Predicate* prev_pred = nullptr;
      if (expr->isA<kir::IfThenElse>()) {
        auto ite = expr->as<kir::IfThenElse>();
        prev_pred = pred_;
        pred_ = ite->predicate();
      }
      kir::IrVisitor::dispatch(expr);
      if (expr->isA<kir::IfThenElse>()) {
        pred_ = prev_pred;
      }
      return;
    }
    if (!ir_utils::isCpAsyncBulk(expr)) {
      return;
    }

    if (num_threads_ == 0) {
      if (pred_ == nullptr) {
        return;
      }
    }

    ASSERT_NE(pred_, nullptr);
    auto cond = pred_->value();
    ASSERT_NE(cond, nullptr);
    if (num_threads_ == 0) {
      EXPECT_TRUE(cond->isTrue());
    } else if (num_threads_ == 1) {
      auto def = dynamic_cast<BinaryOp*>(cond->definition());
      ASSERT_TRUE(def != nullptr);
      EXPECT_TRUE(def->getBinaryOpType() == BinaryOpType::Eq);
      auto lhs = dynamic_cast<NamedScalar*>(def->lhs());
      auto rhs = def->rhs();
      ASSERT_TRUE(lhs != nullptr);
      ASSERT_TRUE(rhs != nullptr);
      EXPECT_TRUE(lhs->isThreadIdx());
      EXPECT_TRUE(rhs->isZeroInt());
    } else {
      auto def = dynamic_cast<BinaryOp*>(cond->definition());
      ASSERT_TRUE(def != nullptr);
      EXPECT_TRUE(def->getBinaryOpType() == BinaryOpType::LT);
      auto lhs = dynamic_cast<NamedScalar*>(def->lhs());
      auto rhs = def->rhs();
      ASSERT_TRUE(lhs != nullptr);
      ASSERT_TRUE(rhs != nullptr);
      EXPECT_TRUE(lhs->isThreadIdx());
      EXPECT_TRUE(rhs->isConstInt());
      EXPECT_EQ(rhs->value(), num_threads_);
    }
  }

 public:
  // Check that TMA is predicated with things like "tidx < num_threads".
  // num_threads == 0 is reserved for no predication.
  static void checkPredicate(kir::Kernel* kernel, int64_t num_threads) {
    TMAPredicateChecker checker(num_threads);
    checker.handle(kernel->topLevelExprs());
  }
};

class TMADimChecker : private kir::IrVisitor {
  int64_t dim_ = -1;

  using kir::IrVisitor::dispatch;

  void dispatch(Expr* expr) final {
    if (expr->isA<ForLoop>() || expr->isA<kir::IfThenElse>()) {
      kir::IrVisitor::dispatch(expr);
      return;
    }
    kir::TensorIndex* gmem_ti = nullptr;
    if (ir_utils::isCpAsyncBulkLoad(expr)) {
      gmem_ti = expr->input(0)->as<kir::TensorIndex>();
    } else if (ir_utils::isCpAsyncBulkStore(expr)) {
      gmem_ti = expr->output(0)->as<kir::TensorIndex>();
    }
    if (gmem_ti == nullptr) {
      return;
    }
    auto dtype = std::get<StructType>(gmem_ti->index()->dtype().type);
    auto field_it = std::find_if(
        dtype.fields.begin(), dtype.fields.end(), [](const auto& f) {
          return f.name == "coordinate";
        });
    auto field_dtype = std::get<ArrayType>(field_it->type->type);
    dim_ = (int64_t)field_dtype.size;
  }

 public:
  // Check the dimension of TMA
  static int64_t getDim(kir::Kernel* kernel) {
    TMADimChecker checker;
    checker.handle(kernel->topLevelExprs());
    return checker.dim_;
  }
};

// Simple load/store tests:
//
// Do a gmem -> smem -> gmem copy. Either the load or the store is a TMA.
// For TMA, use one thread in the block to copy the entire tile. For the
// non-TMA copy, use all threads in the block to parallelize the copy.

using TMASimpleLdstTestParam =
    std::tuple<MmaInputSmemSwizzle, DataType, int64_t>;

class TMASimpleLdstTest
    : public TMATest,
      public ::testing::WithParamInterface<TMASimpleLdstTestParam> {
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

    switch (dim) {
      case 1:
        tile = {innerDimSize()};
        shape = {1024 * 1024};
        break;
      case 2:
        tile = {3, innerDimSize()};
        shape = {1024, 128};
        break;
      case 3:
        tile = {1, 5, innerDimSize()};
        shape = {1024, 8, 128};
        break;
      case 4:
        tile = {3, 5, 1, innerDimSize()};
        shape = {4, 8, 1024, 1024};
        break;
      case 5:
        tile = {1, 3, 1, 5, innerDimSize()};
        shape = {4, 8, 1024, 32, 128};
        break;
      default:
        NVF_ERROR(false, "Invalid dimension");
    }
  }
};

// Assuming the tile of tv is currently scheduled as a 1D flat dim placed at
// position -1, schedule the TMA swizzle for it.
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
  for (auto id : tv->getLoopDomain()) {
    if (skip) {
      skip = false;
      continue;
    }
    id->parallelize(ParallelType::Bulk);
  }
}

void parallelizeAllDimsExceptFirstAsTIDx(TensorView* tv) {
  tv->flatten(1);
  tv->axis(1)->parallelize(ParallelType::TIDx);
}

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
    std::unordered_map<int64_t, int64_t> old2new;
    for (int64_t i = 0; i < dim; i++) {
      old2new[2 * i] = i;
      old2new[2 * i + 1] = i + dim;
    }
    tv->reorder(old2new);
    // [M/tile_sizes[0], N/tile_sizes[1], ..., tile_sizes[0], tile_sizes[1],
    // ...]
    tv->flatten(0, dim - 1);
    tv->flatten(1);
    // [M/tile_sizes[0] * N/tile_sizes[1] * ..., tile_sizes[0] * tile_sizes[1] *
    // ...]
    tv->axis(0)->parallelize(ParallelType::BIDx);
    if (swizzle != MmaInputSmemSwizzle::None) {
      scheduleTMASwizzle(tv, swizzle_size);
    }
  }
}

TEST_P(TMASimpleLdstTest, Load) {
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
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  markAllDimsExceptFirstAsBulk(tv1);
  parallelizeAllDimsExceptFirstAsTIDx(tv2);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), dim);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));
  TMADimChecker::getDim(fe.kernel());

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

class TMALoadTestWithABroadcastDim
    : public NVFuserFixtureParamTest<
          std::tuple<std::vector<int64_t>, DataType, MmaInputSmemSwizzle>> {
 protected:
  MmaInputSmemSwizzle swizzle;
  DataType dtype;

  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
  }

  void schedule(TensorView* tv) {
    // We move the broadcast dim to be the left most.
    moveInnerBroadcastLeft(tv);

    // {B, N, K}
    // {B, NO, N_dim, K}
    tv->split(-2, tv->axis(-2)->extent());

    if (swizzle == MmaInputSmemSwizzle::None) {
      return;
    }
    // {B, NO, N_dim, KO, KI (32/64/128)}
    tv->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));

    // {B, NO, KO, N_dim, KI }
    tv->reorder({{2, 3}, {3, 2}});

    // {B, NO * KO, N_dim, KI}
    tv->merge(1);

    // {B, NO * KO, N_dim_O, N_128/(32, 64, 128),  KI}
    tv->split(-2, (128 / (getBytesFromSwizzle(swizzle))));

    // {B, NO * KO, N_dim_O, N_128/(32, 64, 128),  KIO, KII}
    tv->split(-1, (core_matrix_width_bytes / dataTypeSize(dtype)));

    // split N_dim_O by N/16 N =swizzle size (32/64/128)
    // {B, NO * KO, N_dim_O/(swizzle_size/16), N_128/(32, 64, 128),  KIO, KII}
    tv->split(-4, (getBytesFromSwizzle(swizzle) / 16));

    tv->swizzle(SwizzleType::XOR, -4, -2);
  }

  void markAllDimsExceptFirstTwoAsBulk(const TensorView* tv) {
    int skip = 0;
    for (auto id : tv->getLoopDomain()) {
      if (skip < 2) {
        skip++;
        continue;
      }
      id->parallelize(ParallelType::Bulk);
    }
  }

  TMALoadTestWithABroadcastDim() {
    dtype = std::get<1>(GetParam());
    swizzle = std::get<2>(GetParam());
  }
};

TEST_P(TMALoadTestWithABroadcastDim, LoadWithBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto shape = std::get<0>(GetParam());

  auto tv0 = makeContigConcreteTensor(shape, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  schedule(tv1);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  markAllDimsExceptFirstTwoAsBulk(tv1);

  schedule(tv2);
  // Naively parallelize an outer dim of tv2.
  // We use a single CTA. Inputs are small enough not to error out.
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

std::vector<std::vector<int64_t>> shapes_to_load =
    {{1, 64, 16}, {1, 16, 64}, {1, 128, 64}, {1, 128, 128}, {64, 1, 16}};

INSTANTIATE_TEST_SUITE_P(
    ,
    TMALoadTestWithABroadcastDim,
    testing::Combine(
        testing::ValuesIn(shapes_to_load),
        testing::Values(DataType::Half, DataType::Float, DataType::Double),
        testing::Values(
            MmaInputSmemSwizzle::None,
            MmaInputSmemSwizzle::B128,
            MmaInputSmemSwizzle::B64,
            MmaInputSmemSwizzle::B32)));

TEST_P(TMASimpleLdstTest, Store) {
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
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  markAllDimsExceptFirstAsBulk(tv2);
  parallelizeAllDimsExceptFirstAsTIDx(tv1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), dim);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);
  ASSERT_EQ(
      XorFinder::findXor(fe.kernel()), (swizzle != MmaInputSmemSwizzle::None));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

std::string testNameTMASimpleLdstTest(
    const testing::TestParamInfo<TMASimpleLdstTestParam>& info) {
  auto swizzle = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto dim = std::get<2>(info.param);
  std::stringstream ss;
  ss << dim << "D"
     << "_" << toString(swizzle) << "_" << dtype;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TMASimpleLdstTest,
    testing::Combine(
        kAllSmemSwizzleModes,
        testing::Values(DataType::Half, DataType::Float, DataType::Double),
        testing::Values(1, 2, 3, 4, 5)),
    testNameTMASimpleLdstTest);

// TMA indexing tests:
// Test advanced scheduling strategies for TMA. Make sure that its indexing
// is working correctly.

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
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024, 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

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
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024 * 1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, NonOneElementStride) {
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

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, Advanced) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(8)
                 .dtype(DataType::Float)
                 .contiguity({true, true, false, true, true, false, true, true})
                 .build();
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // View the 8D tensor as 4D
    // [I1, I2, I3, I4, I5, I6, I7, I8]
    tv->merge(0);
    tv->merge(0);
    tv->merge(1);
    tv->merge(1);
    tv->merge(2);
    // [I1*I2*I3, I4*I5*I6, I7*I8]
    tv->split(0, 16);
    // [I1*I2*I3/16, 16, I4*I5*I6, I7*I8]

    // Create tiles
    tv->split(3, 32);
    tv->split(4, 8);
    tv->split(3, 1);
    tv->split(2, 32);
    tv->split(3, 3);
    tv->split(1, 2);
    tv->split(0, 4);
    // [I1*I2*I3/16/4, 4, 16/2, 2, I4*I5*I6/32, 32/3, 3,
    //  I7*I8/32/1, 1, 32/8, 8]

    // Reorder the axes as [non-tile..., tile...]
    tv->reorder({{1, 6}, {3, 7}, {5, 8}, {8, 9}});
    // [I1*I2*I3/16/4, 16/2, I4*I5*I6/32, 3, I7*I8/32/1, 32/8,
    //  4, 2, 32/3, 1, 8]

    // Merge all non-tile axes together, and all tile axes together
    tv->flatten(0, 5);
    tv->flatten(1);
    // [I1*I2*I3/16/4 * 16/2 * I4*I5*I6/32 * 3 * I7*I8/32/1 * 32/8,
    //  4 * 2 * 32/3 * 1 * 8]

    // Parallelize the non-tile axes
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  // Parallelize the tile axes
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 32, 2, 8, 8, 8, 32, 8}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 4);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, DefineBoxByCompositing1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = TensorViewBuilder()
                 .ndims(8)
                 .dtype(DataType::Float)
                 .contiguity({true, true, false, true, false, true, true, true})
                 .build();
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // [I1, I2, I3, I4, I5, I6, I7, I8]
    tv->merge(6);
    tv->split(6, 32);
    // [I1, I2, I3, I4, I5, I6, I7*I8/32, 32]
    // Will use 4D TMA:
    // [ I1, I2, I3,
    //   I4, I5,
    //   I6,
    //   I7*I8/32, 32]
    // Where the first 3 dims are implicitly tiled 1x1x1
    tv->flatten(0, -2);
    // [I1*I2*I3*I4*I5*I6*(I7*I8/32), 32]
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  // Parallelize the tile axes
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  // tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 32, 2, 8, 8, 8, 32, 8}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 4);
  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, DefineBoxByCompositing2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 =
      TensorViewBuilder()
          .ndims(9)
          .dtype(DataType::Float)
          .contiguity({true, true, false, true, false, true, true, false, true})
          .build();
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // [I1, I2, I3, I4, I5, I6, I7, I8, I9]
    tv->merge(3);
    tv->split(3, 3);
    // [I1, I2, I3, I4*I5/3, 3, I6, I7, I8, I9]
    tv->reorder({{1, -5}, {3, -4}});
    // [I1, I3, 3, I6, I2, I4*I5/3, I7, I8, I9]
    tv->flatten(0, 3);
    tv->flatten(1);
    // [I1*I3*3*I6, I2*(I4*I5/3)*I7*I8*I9]
    tv->axis(0)->parallelize(ParallelType::BIDx);
    // Will use 5D TMA:
    // [ I1, I2,
    //   I3,
    //   I4, I5,
    //   I6, I7, I8,
    //   I9]
  }
  // Parallelize the tile axes
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  // tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 4, 2, 8, 8, 8, 2, 8, 4}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 5);
  EXPECT_FALSE(PredicatedChecker::isPredicated(tv1, fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, DefineBoxByRotation1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(3, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // [M, N, K]
    tv->split(-1, 256);
    tv->split(-1, 128);
    tv->split(-1, 64);
    tv->split(-1, 32);
    // [M, N, K/256, 2, 2, 2, 32]
    tv->split(1, 3);
    tv->split(1, 3);
    tv->split(1, 3);
    // [M, N/27, 3, 3, 3, K/256, 2, 2, 2, 32]
    tv->split(0, 3);
    tv->split(0, 3);
    tv->split(0, 64);
    tv->split(1, 32);
    tv->split(2, 4);
    // [M/9/64, 2, 8, 4, 3, 3, N/27, 3, 3, 3, K/256, 2, 2, 2, 32]
    tv->reorder({{3, -7}, {4, -6}, {5, -5}, {7, -4}, {8, -3}, {9, -2}});
    // [M/9/64, 2, 8, N/27, K/256, 2, 2, 2, 4, 3, 3, 3, 3, 3, 32]
    tv->flatten(-7);
    tv->flatten(0, -2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->split(1, 256);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  // Make the shape of the input tensor as prime as possible so splits are
  // mostly not divisible
  int64_t prime_number = 599;
  int64_t multiple_of_16B_but_not_more = 4 * 67;
  auto t0 = at::randn(
      {prime_number, prime_number, multiple_of_16B_but_not_more}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 3);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, DefineBoxByRotation2) {
  // Test that strided box can not be merged with other bulk axes by rotation
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // [M]
    tv->split(0, 8);
    tv->split(0, 4);
    tv->split(1, 2);
    // [M/8/4, 4/2, 2, 8]
    tv->reorder({{1, 2}});
    // [M/8/4, 2, 4/2, 8]
    tv->merge(0);
    tv->merge(1);
    // [M/8/4*2, 4/2*8]
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  int64_t multiple_of_8_but_not_more = 8 * 997;
  auto t0 = at::randn({multiple_of_8_but_not_more}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  // We will be using 2D TMA instead of 1D, because strided box can not be
  // merged with other bulk axes by rotation. So, this schedule will be
  // interpreted as viewing then tensor as 2D (M/8, 8) and then applying 2D TMA.
  // The outer dim of TMA is defined by boxing and striding splits, and the
  // inner dim is defined as implicit whole.
  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // The tensor shape is not a multiple of 8, so the view should fail.
  EXPECT_THAT(
      [&]() {
        auto options = at::TensorOptions()
                           .dtype(data_type_to_aten(dtype))
                           .device(at::kCUDA, 0);
        int64_t prime_number = 997;
        auto t0 = at::randn({prime_number}, options);
        auto cg_outputs = fe.runFusion({t0});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("must be divisible by 8")));
}

TEST_F(TMAIndexingTest, DefineBoxByRotation3) {
  // Test that indivisible split can not be moved up by rotation
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // [M, N]
    tv->split(0, 23);
    tv->split(1, 8);
    // [M/23, 23/8, 8, N]
    tv->merge(0);
    tv->merge(1);
    // [M/23*23/8, 8*N]
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  int64_t multiple_of_23 = 23 * 997;
  auto t0 = at::randn({multiple_of_23, 8}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  // We will be using 3D TMA instead of 2D, because split(23, 8) is indivisible,
  // we can not consider this schedule as a 2D TMA whose first dimension has box
  // size 8. Instead, we must view the tensor as 2D (M/23, 23, N) and apply 3D
  // TMA. The dim 0 of TMA is as implicit size-one, and the dim 1 is defined by
  // a boxing split whose box size is 8, and dim 2 is an implicit whole box with
  // size N.
  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 3);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // The tensor shape is not a multiple of 23, so the view should fail.
  EXPECT_THAT(
      [&]() {
        auto options = at::TensorOptions()
                           .dtype(data_type_to_aten(dtype))
                           .device(at::kCUDA, 0);
        int64_t prime_number = 997;
        auto t0 = at::randn({prime_number, 8}, options);
        auto cg_outputs = fe.runFusion({t0});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("must be divisible by 23")));
}

TEST_F(TMAIndexingTest, NonTrivialGmemAllocationDomain) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_32_bytes = 32 / dataTypeSize(dtype);

  auto tv0 = makeContigTensor(3, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv0, tv1, tv2}) {
    tv->merge(0);
    tv->reorder({{0, 1}});
  }
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);
  scheduleTile({tv1, tv2}, {128, items_of_32_bytes}, MmaInputSmemSwizzle::B32);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  markAllDimsExceptFirstAsBulk(tv1);
  parallelizeAllDimsExceptFirstAsTIDx(tv2);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128, 1024 * 128}, options)
                .transpose(0, 1)
                .view({128, 1024, 128});
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);
  ASSERT_TRUE(XorFinder::findXor(fe.kernel()));

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// TODO: improve validation of TMA, and add tests for invalid cases.
// TODO: test that broadcasting IterDomains are correctly handled by TMA.

class TMAMiscTest : public TMATest {};

TEST_F(TMAMiscTest, AdvancedThreadParallelizationLoad) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // Use 4 threads to issue TMA simultaneously
  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->split(0, 4);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  // Use 512 threads to do the plain store simultaneously
  tv2->split(0, 128);
  tv2->split(0, 4);
  tv2->merge(1);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({100000}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 4);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAMiscTest, AdvancedThreadParallelizationStore) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // Use 512 threads to do the plain load simultaneously
  tv1->split(0, 128);
  tv1->split(0, 4);
  tv1->merge(1);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  // Use 4 threads to issue TMA store simultaneously
  tv2->split(0, 128);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->split(0, 4);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({100000}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 4);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

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

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAMiscTest, Repro1977) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->split(0, 128);
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({1024}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

#if 0
TEST_F(TMAMiscTest, LoadStrongCorrectness) {
  // See doc/reading/tma-modeling-in-depth.md
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({32}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 16);
    tv->split(0, 1);
    tv->split(1, 2);
    // [2, 1, 2, 16]
  }
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  // Use a hacky way to get the "raw data" in smem, including valid items and
  // holes, from the smem buffer.
  tv2->commitLeafToLogical();
  fusion.manage(
      "don't predicate", std::unordered_set<Expr*>{tv2->definition()});

  tv1->axis(-1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::arange(1, 33, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);

  auto cg_outputs = fe.runFusion({t0});

  auto expect = at::zeros({2, 1, 2, 16}, options);
  expect.flatten(0, 2).select(0, 0) = at::arange(1, 17, options);
  expect.flatten(0, 2).select(0, 2) = at::arange(17, 33, options);

  // TODO: remove the line below. The line below is here only to make the test
  // pass. The result is actually wrong.
  expect.flatten(0, 2).select(0, 1) = at::arange(17, 33, options);

  std::cout << cg_outputs[0] << std::endl;
  std::cout << expect << std::endl;

  EXPECT_TRUE(at::equal(cg_outputs[0], expect));
}
#endif

// Testing invalid cases are correctly detected and reported.

// It is not required to run compile-time invalid case tests on Hopper or newer
// GPUs. Detecting invalid cases does not even require a GPU.
class TMACompileTimeInvalidTest : public NVFuserTest {};
class TMARuntimeInvalidTest : public TMATest {};

TEST_F(TMACompileTimeInvalidTest, BulkNotInTMA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // There is a Bulk parallel axis, but there is no TMA
  tv1->axis(0)->parallelize(ParallelType::Bulk);

  EXPECT_THAT(
      [&]() {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        auto t0 = at::randn({32}, options);
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "ParallelType::Bulk is only supported for cp.async.bulk.")));
}

TEST_F(TMARuntimeInvalidTest, MisalignedGlobalAddress) {
  // According to the CUDA programming guide, the global address must be
  // aligned 16 byte:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_16_bytes = 16 / dataTypeSize(dtype);

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 128);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0_aligned = at::randn({128 + items_of_16_bytes}, options)
                        .narrow(0, items_of_16_bytes, 128);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0_aligned}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0_aligned});
  testValidate(
      &fusion, cg_outputs, {t0_aligned}, {t0_aligned}, __LINE__, __FILE__);

  EXPECT_THAT(
      [&]() {
        auto t0_misaligned = at::randn({128 + items_of_16_bytes / 2}, options)
                                 .narrow(0, items_of_16_bytes / 2, 128);
        fe.runFusion({t0_misaligned});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "globalAddress, which specifies the starting address of the memory region described, "
          "must be 32 byte aligned when interleave is CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.")));
}

TEST_F(TMARuntimeInvalidTest, MisalignedGlobalStride) {
  // According to the CUDA programming guide, the global strides must be
  // aligned 16 byte:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_16_bytes = 16 / dataTypeSize(dtype);

  auto tv0 = makeContigTensor(2, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 128);
    tv->split(0, 128);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0_aligned =
      at::randn({128, 128 + items_of_16_bytes}, options).narrow(1, 0, 128);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0_aligned}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0_aligned});
  testValidate(
      &fusion, cg_outputs, {t0_aligned}, {t0_aligned}, __LINE__, __FILE__);

  EXPECT_THAT(
      [&]() {
        auto t0_misaligned =
            at::randn({128, 128 + items_of_16_bytes / 2}, options)
                .narrow(1, 0, 128);
        fe.runFusion({t0_misaligned});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "globalStrides array, which specifies tensor stride of each of the lower tensorRank - 1 dimensions in bytes, "
          "must be a multiple of 16 and less than 2^40.")));
}

TEST_F(TMACompileTimeInvalidTest, SizeOfTransfer) {
  // According to the CUDA programming guide, the size of the transfer must be
  // a multiple of 16 bytes:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_16_bytes = 16 / dataTypeSize(dtype);

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, items_of_16_bytes / 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "The expected bytes must be a multiple of 16 bytes, but 8 is not.")));
}

TEST_F(TMARuntimeInvalidTest, SizeOfTransfer) {
  // According to the CUDA programming guide, the size of the transfer must be
  // a multiple of 16 bytes:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_16_bytes = 16 / dataTypeSize(dtype);

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tile_size = IrBuilder::create<Val>(DataType::Index);
  fusion.addInput(tile_size);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, tile_size);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, items_of_16_bytes}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 1);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0, items_of_16_bytes});
  testValidate(
      &fusion, cg_outputs, {t0, items_of_16_bytes}, {t0}, __LINE__, __FILE__);

  EXPECT_THAT(
      [&]() {
        fe.runFusion({t0, items_of_16_bytes / 2});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "The expected bytes must be a multiple of 16 bytes, but ")));
}

TEST_F(TMARuntimeInvalidTest, InvalidView) {
  // According to the CUDA programming guide, the size of the transfer must be
  // a multiple of 16 bytes:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    // view as 2D
    tv->split(0, 1024);
    // create tile
    tv->split(1, 32);
    tv->split(0, 32);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  // (10240,) can be viewed as (10, 1024)
  auto t0_valid = at::randn({10240}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0_valid}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);

  auto cg_outputs = fe.runFusion({t0_valid});
  testValidate(&fusion, cg_outputs, {t0_valid}, {t0_valid}, __LINE__, __FILE__);

  EXPECT_THAT(
      [&]() {
        // it is impossible to view (10249,) as (?, 1024)
        auto t0_inval = at::randn({10249}, options);
        fe.runFusion({t0_inval});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Invalid view in TMA: the extent of")));
}

TEST_F(TMACompileTimeInvalidTest, InnermostDiscontiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeSymbolicTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 16);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "The innermost dimension of the TMA domain must be contiguous")));
}

TEST_F(TMACompileTimeInvalidTest, MergeDiscontiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = TensorViewBuilder()
                 .ndims(2)
                 .dtype(DataType::Float)
                 .contiguity({false, true})
                 .build();
  ;
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->split(0, 16);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Can not merge discontiguous dimensions, but")));
}

TEST_F(TMACompileTimeInvalidTest, InnermostElementStrideNotOne) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 32);
    tv->split(1, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE "
          "(this is always the case for nvFuser now), "
          "the first element of elementStrides must be one.")));
}

TEST_F(TMACompileTimeInvalidTest, SwizzleBulkWithNonBulk) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(0, 16);
    tv->split(0, 16);
    tv->swizzle(SwizzleType::XOR, 1, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(2)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "TMA domain must be a view of the allocation domain of the gmem tensor")));
}

// Tests for the examples in doc/dev/tma.md

class TMADocTest : public TMATest {};

TEST_F(TMADocTest, Figure13a) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 200}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 128);
    tv->split(0, 4);
    tv->split(1, 3);
    tv->axis(3)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(4)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 200}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure14a) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 200}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 128);
    tv->split(0, 4);
    tv->split(1, 3);
    tv->axis(3)->parallelize(ParallelType::BIDx);
    tv->reorder({{1, 2}});
  }
  tv1->axis(2)->parallelize(ParallelType::Bulk);
  tv1->axis(4)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure13b) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 6);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure14b) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 10);
    tv->split(0, 4);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure13c) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 200}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 128);
    tv->split(0, 1);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 200}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure14c) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 200}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 128);
    tv->split(0, 1);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure13d) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 12}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->split(1, 8);
  tv2->split(0, 4);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 12}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure14d) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 12}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->split(0, 4);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 12}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure13e) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 6);
    tv->split(0, 4);
  }
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure14e) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 16);
    tv->split(0, 4);
  }
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::TIDy);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 1);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure15a) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 4);
    tv->split(0, 4);
    // 16/4, 4, 10/4, 4
    tv->reorder({{1, 2}});
    // 16/4, 10/4, 4, 4
    tv->split(-1, 2);
    // 16/4, 10/4, 4, 2, 2
    tv->merge(2);
    // 16/4, 10/4, 8, 2
  }
  tv1->axis(2)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 0);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure15b) {
  GTEST_SKIP() << "TODO: requires IdModel based indexing.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 12}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->merge(0);
  tv1->split(0, 2);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  tv2->split(1, 4);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 12}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(fe.kernel()), 2);
  TMAPredicateChecker::checkPredicate(fe.kernel(), 4);

  auto cg_outputs = fe.runFusion({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMADocTest, Figure15c) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 4);
    tv->split(0, 4);
    // 16/4, 4, 10/4, 4
    tv->reorder({{1, 2}});
    // 16/4, 10/4, 4, 4
    tv->split(-1, 3);
    // 16/4, 10/4, 4, 2, 3
    tv->merge(2);
    // 16/4, 10/4, 8, 3
  }
  tv1->axis(2)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure15d) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({16, 10}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2}) {
    tv->split(1, 4);
    tv->split(0, 4);
    // 16/4, 4, 10/4, 4
    tv->reorder({{1, 2}});
    // 16/4, 10/4, 4, 4
    tv->split(-1, 3);
    // 16/4, 10/4, 4, 2, 2'
    tv->reorder({{-1, -2}});
    // 16/4, 10/4, 4, 2', 2
    tv->merge(2);
    // 16/4, 10/4, 8, 2
  }
  tv1->axis(2)->parallelize(ParallelType::Bulk);
  tv1->axis(3)->parallelize(ParallelType::Bulk);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 10}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
}

TEST_F(TMADocTest, Figure15e) {
  GTEST_SKIP() << "TODO: add check for this invalid case.";
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Double;

  auto tv0 = makeContigConcreteTensor({15, 12}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->merge(0);
  tv1->split(0, 8);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  tv2->split(1, 4);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(2)->parallelize(ParallelType::Bulk);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 12}, options);

  EXPECT_THAT(
      [&]() {
        FusionExecutor fe;
        fe.compileFusion(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Some error message")));
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
  auto t0 = at::randn({getK(macro), size2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
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
