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
#include <ir/utils.h>
#include <mma_type.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <options.h>
#include <scheduler/cache_policy_refiner.h>
#include <scheduler/mma_utils.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <type.h>
#include <utils.h>

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
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  // Parallelize LoadStoreOps. Other TensorViews don't support vectorization.
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Vectorize);
  scheduler_utils::parallelizeAllLike(tv1, {tv3});

  inlineMost();

  at::Tensor input = at::randn(
      {1024}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor expected_output = input + 1.0f;

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    ke.compile(&fusion, {input});
  }

  // Verify PTX.
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx(compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  std::regex regex(R"(ld\.global\.)" + cache_op_str + R"(\.\S+)");
  std::smatch match;
  std::regex_search(ptx, match, regex);
  EXPECT_EQ(match.size(), 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel->ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel->ptx_filename);

  // Verify output tensors.
  auto cg_outputs = ke.run({input});
  testValidate(
      &fusion, cg_outputs, {input}, {expected_output}, __LINE__, __FILE__);
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
  MaxLogicalDomainInfoSpanningTree(tv_a2).traverse(&propagator);

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

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    ke.compile(&fusion, {a, b});
  }

  // Verify PTX.
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx(compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  expectMatchCount(ptx, R"(ld\.global\.ca\.v4\.\S+)", 1);
  expectMatchCount(ptx, R"(ld\.global\.cs\.v4\.\S+)", 1);

  // Clean up the dumped PTX file.
  debug() << "Removing " << compiled_kernel->ptx_filename << std::endl;
  std::filesystem::remove(compiled_kernel->ptx_filename);

  auto actual_outputs = ke.run({a, b});
  testValidate(&fusion, actual_outputs, {a, b}, {c}, __LINE__, __FILE__);
}

// Begin TMA tests

using TMATest = TmaBase;

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
  int64_t cta_threads_;
  bool is_tma_store_;
  TMAPredicateChecker(
      int64_t num_threads,
      int64_t cta_threads,
      bool is_tma_store)
      : num_threads_(num_threads),
        cta_threads_(cta_threads),
        is_tma_store_(is_tma_store) {}

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

    // Handle TMA Store first
    if (is_tma_store_) {
      if (cta_threads_ <= 32) {
        EXPECT_TRUE(cond->isTrue());
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
        EXPECT_EQ(rhs->value(), 32);
      }
      return;
    }

    // Then, handle TMA Load
    if (num_threads_ == 0) {
      EXPECT_TRUE(cond->isTrue());
    } else if (is_tma_store_ && cta_threads_ <= 32) {
      EXPECT_TRUE(cond->isTrue());
    } else if (num_threads_ == 1 && cta_threads_ > 32) {
      auto def = dynamic_cast<BinaryOp*>(cond->definition());
      ASSERT_TRUE(def != nullptr);
      EXPECT_TRUE(def->getBinaryOpType() == BinaryOpType::LogicalAnd);
      auto lhs = def->lhs();
      auto rhs = def->rhs();
      ASSERT_TRUE(lhs != nullptr);
      auto lhs_def = dynamic_cast<UnaryOp*>(lhs->definition());
      EXPECT_TRUE(lhs_def->getUnaryOpType() == UnaryOpType::ElectSync);
      ASSERT_TRUE(rhs != nullptr);
      auto rhs_def = dynamic_cast<BinaryOp*>(rhs->definition());
      EXPECT_TRUE(rhs_def->getBinaryOpType() == BinaryOpType::LT);
      auto lhs_rhs = dynamic_cast<NamedScalar*>(rhs_def->lhs());
      auto rhs_rhs = rhs_def->rhs();
      ASSERT_TRUE(lhs_rhs != nullptr);
      ASSERT_TRUE(rhs_rhs != nullptr);
      EXPECT_TRUE(lhs_rhs->isThreadIdx());
      EXPECT_TRUE(rhs_rhs->isConstInt());
      EXPECT_EQ(rhs_rhs->value(), 32);
    } else if (num_threads_ == 1 && cta_threads_ == 32) {
      auto def = dynamic_cast<UnaryOp*>(cond->definition());
      ASSERT_TRUE(def != nullptr);
      EXPECT_TRUE(def->getUnaryOpType() == UnaryOpType::ElectSync);
    } else if (num_threads_ == 1 && cta_threads_ < 32) {
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
  static void checkPredicate(
      kir::Kernel* kernel,
      int64_t num_threads,
      int64_t cta_threads = -1,
      bool is_tma_store = false) {
    TMAPredicateChecker checker(num_threads, cta_threads, is_tma_store);
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
        NVF_THROW("Invalid dimension");
    }

    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

void parallelizeAllDimsExceptFirstAsTIDx(TensorView* tv) {
  tv->flatten(1);
  tv->axis(1)->parallelize(ParallelType::TIDx);
}

void scheduleTile(
    std::vector<TensorView*> tvs,
    std::vector<int64_t> tile_sizes,
    MmaInputSmemSwizzle swizzle) {
  const int64_t dim = tile_sizes.size();

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
      // In our implementation of swizzling, we work on 2D box where the
      // inner-dim is the size of the swizzle in Bytes (at most).
      tv->split(-1, tile_sizes[dim - 1]);
      tv->swizzleTMABox(swizzle);
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
  mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
      tv1, 1 /* skip the first ID*/);
  parallelizeAllDimsExceptFirstAsTIDx(tv2);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), dim);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
  ASSERT_EQ(
      XorFinder::findXor(ke.compiledKernel()->kernel()),
      (swizzle != MmaInputSmemSwizzle::None));
  TMADimChecker::getDim(ke.compiledKernel()->kernel());
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

  matmul_utils::moveInnerBroadcastLeft(tv1);
  matmul_utils::moveInnerBroadcastLeft(tv2);

  // [B, K, N] -> [B, KO, K8, N]
  tv1->split(-2, 8);
  tv2->split(-2, 8);
  // [B, KO, K8, N] ->  [B, KO, K8, NO, NI ]
  tv1->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));
  tv2->split(-1, getBytesFromSwizzle(swizzle) / dataTypeSize(dtype));
  // [B, KO, K8, NO, NI ] -> [B, KO, NO, K8, NI ] (Box: K8, NI)
  tv1->reorder({{-2, -3}});
  tv2->reorder({{-2, -3}});
  if (swizzle != MmaInputSmemSwizzle::None) {
    tv1->swizzleTMABox(swizzle);
    tv2->swizzleTMABox(swizzle);
  }
  mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
      tv1, 3 /* skip the first three IDs*/);

  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  // Naively parallelize an outer dim of tv2.
  // We use a single CTA. Inputs are small enough not to error out.
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  auto cg_outputs = ke.run({t0});
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
        testing::ValuesIn(kAllSmemSwizzleModes)));

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
  mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(tv2, 1);
  parallelizeAllDimsExceptFirstAsTIDx(tv1);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), dim);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(),
      1,
      ke.lastLaunchParams().nThreads(),
      /*is_tma_store=*/true);
  ASSERT_EQ(
      XorFinder::findXor(ke.compiledKernel()->kernel()),
      (swizzle != MmaInputSmemSwizzle::None));
}

std::string testNameTMASimpleLdstTest(
    const testing::TestParamInfo<TMASimpleLdstTestParam>& info) {
  auto swizzle = std::get<0>(info.param);
  auto dtype = std::get<1>(info.param);
  auto dim = std::get<2>(info.param);
  std::stringstream ss;
  ss << dim << "D" << "_" << toString(swizzle) << "_" << dtype;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TMASimpleLdstTest,
    testing::Combine(
        testing::ValuesIn(kAllSmemSwizzleModes),
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 4);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 4);
  EXPECT_FALSE(
      PredicatedChecker::isPredicated(tv1, ke.compiledKernel()->kernel()));

  auto cg_outputs = ke.run({t0});
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

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32, 4, 2, 8, 8, 8, 2, 8, 4}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 5);
  EXPECT_FALSE(
      PredicatedChecker::isPredicated(tv1, ke.compiledKernel()->kernel()));

  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(TMAIndexingTest, DefineBoxByCompositingShouldNotMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 256, 2, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // Use 1 thread and a single instruction to load the entire tensor to smem
  for (auto id : tv1->getLoopDomain()) {
    id->parallelize(ParallelType::Bulk);
  }

  // Then use 32 threads to dump results out
  tv2->axis(3)->parallelize(ParallelType::TIDx);

  // Schedule the allocation domain of tv1 to use 128B swizzle
  AbstractTensor alloc1(tv1->getLoopDomain());
  alloc1.merge(0);
  alloc1.merge(0);
  // [1024, 32]
  alloc1.split(1, 4);
  alloc1.split(0, 8);
  // [128, 8, 8, 4]
  alloc1.swizzle(SwizzleType::XOR, 1, 2);
  tv1->setAllocationDomain(alloc1.as<IterDomain*>(), true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 256, 2, 32}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  // Because merging dims will violate hardware requirement, we do not merge
  // dims.
  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 4);

  EXPECT_TRUE(
      PredicatedChecker::isPredicated(tv1, ke.compiledKernel()->kernel()));

  auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 3);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // We will be using 2D TMA instead of 1D, because strided box can not be
  // merged with other bulk axes by rotation. So, this schedule will be
  // interpreted as viewing then tensor as 2D (M/8, 8) and then applying 2D TMA.
  // The outer dim of TMA is defined by boxing and striding splits, and the
  // inner dim is defined as implicit whole.
  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());

  // The tensor shape is not a multiple of 8, so the view should fail.
  EXPECT_THAT(
      [&]() {
        auto options = at::TensorOptions()
                           .dtype(data_type_to_aten(dtype))
                           .device(at::kCUDA, 0);
        int64_t prime_number = 997;
        auto t0 = at::randn({prime_number}, options);
        auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // We will be using 3D TMA instead of 2D, because split(23, 8) is indivisible,
  // we can not consider this schedule as a 2D TMA whose first dimension has box
  // size 8. Instead, we must view the tensor as 2D (M/23, 23, N) and apply 3D
  // TMA. The dim 0 of TMA is as implicit size-one, and the dim 1 is defined by
  // a boxing split whose box size is 8, and dim 2 is an implicit whole box with
  // size N.
  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 3);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());

  // The tensor shape is not a multiple of 23, so the view should fail.
  EXPECT_THAT(
      [&]() {
        auto options = at::TensorOptions()
                           .dtype(data_type_to_aten(dtype))
                           .device(at::kCUDA, 0);
        int64_t prime_number = 997;
        auto t0 = at::randn({prime_number, 8}, options);
        auto cg_outputs = ke.run({t0});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("must be divisible by 23")));
}

TEST_F(TMAIndexingTest, NonTrivialGmemAllocationDomain1) {
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
  mma_utils::MmaSwizzler::parallelizeAsBulkSkippingFirstIDs(
      tv1, 1 /* skip the first ID*/);
  parallelizeAllDimsExceptFirstAsTIDx(tv2);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({128, 1024 * 128}, options)
                .transpose(0, 1)
                .view({128, 1024, 128});
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
  ASSERT_TRUE(XorFinder::findXor(ke.compiledKernel()->kernel()));
}

TEST_F(TMAIndexingTest, NonTrivialGmemAllocationDomain2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(6, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // Schedule like this:
  // 0   1   2   3   4   5
  //  \   \ /   /     \ /
  //   \   6   /       7
  //    \ /   /
  //     8   /
  //      \ /
  //       9
  // where 1 and 5 are bulk IDs. This way, [merge 1, 2 -> 6] is a "striding
  // split", and [merge 0, 6 -> 8] and [merge 4, 5 -> 7] are "boxing splits".
  tv0->merge(1);
  tv0->merge(0);
  tv0->merge(-2);
  tv0->merge(0);
  tv0->setAllocationDomain(tv0->getLoopDomain(), true);

  for (auto tv : {tv1, tv2}) {
    tv->reorder({{1, -2}});
    tv->merge(-2);
    tv->flatten(0, -2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }
  tv1->axis(1)->parallelize(ParallelType::Bulk);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 3, 5, 7, 11, 32}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 3);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 4);

  auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 4);

  auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Test that if the async wait expressions for TMA store is inserted correctly
TEST_F(TMAMiscTest, StoreSyncInsertion) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;

  auto tv0 = makeContigTensor(1, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  // gmem -> smem -> gmem -> gmem copy kernel
  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Global);

  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);
  tv2->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  for (auto tv : {tv1, tv2, tv3}) {
    tv->split(0, 128);
    tv->split(0, 4);
  }
  tv1->axis(-1)->parallelize(ParallelType::Bulk);
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto input = at::randn({8192}, options);

  auto is_commit = [](Expr* expr) {
    auto asm_ = dynamic_cast<kir::Asm*>(expr);
    return asm_ != nullptr && asm_->code() == "cp.async.bulk.commit_group";
  };
  auto is_wait = [](Expr* expr) {
    auto asm_ = dynamic_cast<kir::Asm*>(expr);
    return asm_ != nullptr && asm_->code() == "cp.async.bulk.wait_group.read";
  };

  {
    // No inline, the kernel should look like:
    //   for N/128/4: (loop 1)
    //     for 4:
    //       for 128:
    //         TMA load;
    //   for N/128/4: (loop 2)
    //     for 4:
    //       for 128:
    //         TMA store;
    //   for N/128/4: (loop 3)
    //     for 4:
    //       for 128:
    //         gmem->gmem copy;
    // There should only be a RAW sync inserted before loop 3
    GpuLower gpulw(&fusion);
    auto kernel = gpulw.run();

    auto commit_it = std::find_if(
        kernel->topLevelExprs().begin(),
        kernel->topLevelExprs().end(),
        is_commit);
    ASSERT_NE(commit_it, kernel->topLevelExprs().end());
    ASSERT_TRUE((*std::next(commit_it))->isA<kir::Asm>());
    EXPECT_TRUE(is_wait(*std::next(commit_it)));
    EXPECT_EQ((*std::next(commit_it))->input(0)->value(), 0);

    auto flattened_exprs =
        ir_utils::flattenScopedExprs(kernel->topLevelExprs());
    EXPECT_EQ(
        std::count_if(
            flattened_exprs.begin(), flattened_exprs.end(), is_commit),
        1);
    EXPECT_EQ(
        std::count_if(flattened_exprs.begin(), flattened_exprs.end(), is_wait),
        1);

    KernelExecutor ke;
    ke.compile(&fusion, {input}, {}, matmul_cparams);
    auto cg_outputs = ke.run({input});
    testValidate(&fusion, cg_outputs, {input}, {input}, __LINE__, __FILE__);
  }

  tv1->inlineAt(1);

  {
    // tv1 inlined to tv2, the kernel should look like:
    //   for N/128/4: (loop 1)
    //     for 4:
    //       for 128:
    //         TMA load;
    //     for 4:
    //       for 128:
    //         TMA store;
    //   for N/128/4: (loop 2)
    //     for 4:
    //       for 128:
    //         gmem->gmem copy;
    // There must be a WAR async wait at the end of loop 1. In theory,
    // We do not need a RAW async wait before loop 2, because in this example
    // the WAR async wait should helped RAW as well. But we are not smartly
    // enough right now to avoid this RAW async wait.
    GpuLower gpulw(&fusion);
    auto kernel = gpulw.run();

    auto fl_it = std::find_if(
        kernel->topLevelExprs().begin(),
        kernel->topLevelExprs().end(),
        [](Expr* expr) { return expr->isA<ForLoop>(); });
    ASSERT_NE(fl_it, kernel->topLevelExprs().end());
    const auto& body = (*fl_it)->as<ForLoop>()->body().exprs();
    EXPECT_TRUE(is_wait(body.back()));
    EXPECT_EQ(body.back()->input(0)->value(), 0);
    EXPECT_TRUE(is_commit(body.at(body.size() - 2)));

    auto flattened_exprs = ir_utils::flattenScopedExprs(body);
    EXPECT_EQ(
        std::count_if(
            flattened_exprs.begin(), flattened_exprs.end(), is_commit),
        1);
    EXPECT_EQ(
        std::count_if(flattened_exprs.begin(), flattened_exprs.end(), is_wait),
        1);

    // TODO: For this case, the WAR sync is already sufficient to cover the RAW.
    // However, we are still inserting a RAW sync because at the time when the
    // RAW sync is inserted, the WAR pass has not run yet. We should be able to
    // remove the RAW sync by adding a cleanup pass.

    KernelExecutor ke;
    ke.compile(&fusion, {input}, {}, matmul_cparams);
    auto cg_outputs = ke.run({input});
    testValidate(&fusion, cg_outputs, {input}, {input}, __LINE__, __FILE__);
  }

  tv1->circularBuffer(/*stage=*/10, /*prefetch=*/4);

  {
    // tv1 inlined to tv2 and circular buffered, the kernel should look like:
    //   for N/128/4: (loop 1.prologue)
    //     for 4:
    //       for 128:
    //         TMA load;
    //   for N/128/4: (loop 1.main)
    //     for 4:
    //       for 128:
    //         TMA load;
    //     for 4:
    //       for 128:
    //         TMA store;
    //   for N/128/4: (loop 1.epilogue)
    //     for 4:
    //       for 128:
    //         TMA store;
    //   for N/128/4: (loop 2)
    //     for 4:
    //       for 128:
    //         gmem->gmem copy;
    // We need a WAR async wait at the end of loop 1.main, and a RAW async wait
    // before loop.
    GpuLower gpulw(&fusion);
    auto kernel = gpulw.run();

    auto fl_it = std::find_if(
        kernel->topLevelExprs().begin(),
        kernel->topLevelExprs().end(),
        [](Expr* expr) {
          auto fl = dynamic_cast<ForLoop*>(expr);
          return fl != nullptr &&
              fl->circularBufferLoopStage() == CircularBufferLoopStage::Main;
        });
    ASSERT_NE(fl_it, kernel->topLevelExprs().end());
    const auto& body = (*fl_it)->as<ForLoop>()->body().exprs();
    EXPECT_TRUE(is_wait(body.back()));
    EXPECT_EQ(body.back()->input(0)->value(), 5);
    EXPECT_TRUE(is_commit(body.at(body.size() - 2)));

    auto commit_it = std::find_if(
        kernel->topLevelExprs().begin(),
        kernel->topLevelExprs().end(),
        is_commit);
    ASSERT_NE(commit_it, kernel->topLevelExprs().end());
    ASSERT_TRUE((*std::next(commit_it))->isA<kir::Asm>());
    EXPECT_TRUE(is_wait(*std::next(commit_it)));
    EXPECT_EQ((*std::next(commit_it))->input(0)->value(), 0);

    auto flattened_exprs =
        ir_utils::flattenScopedExprs(kernel->topLevelExprs());
    EXPECT_EQ(
        std::count_if(
            flattened_exprs.begin(), flattened_exprs.end(), is_commit),
        2);
    EXPECT_EQ(
        std::count_if(flattened_exprs.begin(), flattened_exprs.end(), is_wait),
        2);

    KernelExecutor ke;
    ke.compile(&fusion, {input}, {}, matmul_cparams);
    auto cg_outputs = ke.run({input});
    testValidate(&fusion, cg_outputs, {input}, {input}, __LINE__, __FILE__);
  }
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);

  auto cg_outputs = ke.run({t0});

  auto expect = at::zeros({2, 1, 2, 16}, options);
  expect.flatten(0, 2).select(0, 0) = at::arange(1, 17, options);
  expect.flatten(0, 2).select(0, 2) = at::arange(17, 33, options);

  // TODO: remove the line below. The line below is here only to make the test
  // pass. The result is actually wrong.
  expect.flatten(0, 2).select(0, 1) = at::arange(17, 33, options);

  EXPECT_TRUE(cg_outputs[0].as<at::Tensor>().equal(expect));
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "ParallelType::Bulk is only supported for cp.async.bulk.")));
}

TEST_F(TMACompileTimeInvalidTest, BulkBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  // ParallelType on Broadcast
  tv1->axis(0)->parallelize(ParallelType::Bulk);

  EXPECT_THAT(
      [&]() {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        auto t0 = at::randn({32}, options);
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "ParallelType::Bulk is only supported for IterType::Iteration.")));
}

TEST_F(TMACompileTimeInvalidTest, InvalidParallelType) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::CpAsyncBulkTensorTile);

  tv1->axis(0)->parallelize(ParallelType::Vectorize);

  EXPECT_THAT(
      [&]() {
        auto options =
            at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
        auto t0 = at::randn({32}, options);
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Invalid parallel type for cp.async.bulk: V")));
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0_aligned}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0_aligned});
  testValidate(
      &fusion, cg_outputs, {t0_aligned}, {t0_aligned}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());

  EXPECT_THAT(
      [&]() {
        auto t0_misaligned = at::randn({128 + items_of_16_bytes / 2}, options)
                                 .narrow(0, items_of_16_bytes / 2, 128);
        ke.run({t0_misaligned});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "globalAddress, which specifies the starting address of the memory "
          "region described, "
          "must be 32 byte aligned when interleave is "
          "CU_TENSOR_MAP_INTERLEAVE_32B and 16 byte aligned otherwise.")));
}

TEST_F(TMARuntimeInvalidTest, MisalignedGlobalStride) {
  // According to the CUDA programming guide, the global strides must be
  // aligned 16 byte:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-one-dim-tma
  Fusion fusion;
  FusionGuard fg(&fusion);

  const DataType dtype = DataType::Float;
  const int64_t items_of_16_bytes = 16 / dataTypeSize(dtype);

  auto tv0 = makeSymbolicTensor(2, dtype);
  tv0->setContiguity({false, true});
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0_aligned}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0_aligned});
  testValidate(
      &fusion, cg_outputs, {t0_aligned}, {t0_aligned}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, ke.lastLaunchParams().nThreads());

  EXPECT_THAT(
      [&]() {
        auto t0_misaligned =
            at::randn({128, 128 + items_of_16_bytes / 2}, options)
                .narrow(1, 0, 128);
        ke.run({t0_misaligned});
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "globalStrides array, which specifies tensor stride of each of the "
          "lower tensorRank - 1 dimensions in bytes, "
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, items_of_16_bytes}, {}, matmul_cparams);
  EXPECT_THAT(
      [&]() { ke.run({t0, items_of_16_bytes}); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("Expected blockDim.x >= 32 but found 4")));

  // The blockDim.x size is determined at runtime, so a kernel with the
  // elect-sync predicate is generated and a runtime check is used to determine
  // correctness.
  constexpr int64_t num_threads = 64;
  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 1);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(), 1, num_threads);

  EXPECT_THAT(
      [&]() { ke.run({t0, items_of_16_bytes / 2}); },
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
  KernelExecutor ke;
  ke.compile(&fusion, {t0_valid}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);

  auto cg_outputs = ke.run({t0_valid});
  testValidate(&fusion, cg_outputs, {t0_valid}, {t0_valid}, __LINE__, __FILE__);

  EXPECT_THAT(
      [&]() {
        // it is impossible to view (10249,) as (?, 1024)
        auto t0_inval = at::randn({10249}, options);
        ke.run({t0_inval});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("TMA domain must be a view of the allocation "
                               "domain of the gmem tensor")));
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(),
      1,
      ke.lastLaunchParams().nThreads(),
      /*is_tma_store=*/true);

  auto cg_outputs = ke.run({t0});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(
      ke.compiledKernel()->kernel(),
      1,
      ke.lastLaunchParams().nThreads(),
      /*is_tma_store=*/true);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 0);

  auto cg_outputs = ke.run({t0});
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, {}, matmul_cparams);

  EXPECT_EQ(TMADimChecker::getDim(ke.compiledKernel()->kernel()), 2);
  TMAPredicateChecker::checkPredicate(ke.compiledKernel()->kernel(), 4);

  auto cg_outputs = ke.run({t0});
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
        KernelExecutor ke;
        ke.compile(&fusion, {t0}, {}, matmul_cparams);
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
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// We get shapes M and N from MmaMacrao. The vector of ints are
// the tile_m and tile_n factors (8x8, 16x8 and 16x16).
using StMatrixTestParams = std::tuple<MmaMacro, std::vector<int>, DataType>;

class StMatrixTest : public NVFuserFixtureParamTest<StMatrixTestParams> {
 protected:
  void SetUp() override {
    if (cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "skipping tests on pre-Hopper GPUs";
    }
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_P(StMatrixTest, Regular) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto macro = std::get<0>(GetParam());
  auto tile_sizes = std::get<1>(GetParam());
  auto dtype = std::get<2>(GetParam());
  auto sizeM = getM(macro);
  auto sizeN = getN(macro);
  int64_t tile_m = tile_sizes.at(0);
  int64_t tile_n = tile_sizes.at(1);

  if (sizeM % tile_m || sizeN % tile_n) {
    GTEST_SKIP() << "Fractional tiling is not supported/tested";
  }

  fusion.manage("ldst_matrix_m_tile", tile_m);
  fusion.manage("ldst_matrix_n_tile", tile_n);
  fusion.manage("ldst_matrix_m_smem", sizeM);
  fusion.manage("ldst_matrix_n_smem", sizeN);

  auto tv0 = makeContigConcreteTensor({sizeM, sizeN}, dtype);
  fusion.addInput(tv0);
  // tv0 (global) -> tv1 (registers)
  auto tv1 = set(tv0);
  // tv1 (register) -> tv2 (shared)
  auto tv2 = set(tv1);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StMatrix);
  tv2->setMemoryType(MemoryType::Shared);
  // tv2 (shared) -> tv3(global)
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv0->merge(0);
  tv0->split(0, 32);
  tv0->axis(1)->parallelize(ParallelType::TIDx);

  for (auto tv : {tv1, tv2}) {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv->getLoopDomain());
    tv->setLoopDomain(s.as<IterDomain*>());
  }
  tv1->setAllocationDomain(tv1->getLoopDomain(), true);

  mma_utils::scheduleLdStMatrixForMmaOutput(tv2, tile_m, tile_n);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  tv3->merge(0);
  tv3->split(0, 32);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({sizeM, sizeN}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

std::string testNameStMatrixTest(
    const testing::TestParamInfo<StMatrixTestParams>& info) {
  std::ostringstream os;
  auto macro = std::get<0>(info.param);
  auto tile_sizes = std::get<1>(info.param);
  auto dtype = std::get<2>(info.param);
  auto sizeM = getM(macro);
  auto sizeN = getN(macro);
  auto tile_m = tile_sizes.at(0);
  auto tile_n = tile_sizes.at(1);

  os << "m_" << sizeM << "_n_" << sizeN << "_tile_m_" << tile_m << "_tile_n_"
     << tile_n << "_" << mma_utils::dtypeToChar(dtype);
  return os.str();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    StMatrixTest,
    testing::Combine(
        testing::ValuesIn(kAllHopperMacros),
        testing::Values(
            // tile_m, tile_n
            std::vector<int>{16, 8},
            std::vector<int>{16, 16}),
        testing::Values(DataType::Half, DataType::BFloat16)),
    testNameStMatrixTest);

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({t0});

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

TEST_F(TMATest, CpAsyncBulk1D) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};

  constexpr int dim0 = 16384, dim1 = 16384;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  auto tv1 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  // cp.async.bulk copies 1D data and allows more than 256 elements
  // cp.async.bulk.tensor.nd copies n-d data and each dimension must <= 256
  // using CpAsyncBulkTensorTile will trigger the following error:
  // boxDim array, which specifies number of elements to be traversed along each
  // of the tensorRank dimensions, must be non-zero and less than or equal to
  // 256. box_dim_val = 512
  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  auto tv1a = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  auto tv2b = tv2->cacheBefore();
  tv0a->setMemoryType(MemoryType::Shared);
  tv1a->setMemoryType(MemoryType::Shared);
  tv2b->setMemoryType(MemoryType::Shared);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsyncBulk);

  tv2->merge(0);
  tv2->split(0, 512);
  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(tv2);

  /// TIDx for computation, Bulk for load
  tv2->axis(-1)->parallelize(ParallelType::Bulk);
  tv2b->axis(-1)->parallelize(ParallelType::TIDx);
  tv0a->axis(-1)->parallelize(ParallelType::Bulk);
  tv1a->axis(-1)->parallelize(ParallelType::Bulk);
  inlineMost();

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);
  at::Tensor at_tv1 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0, at_tv1}, {}, index32bit);
  auto outputs = ke.run({at_tv0, at_tv1});
  auto at_output = at_tv0 + at_tv1;
  testValidate(
      fusion.get(), outputs, {at_tv0, at_tv1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(TMATest, CpAsyncBulk1dNonDivisibleSplit) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};

  constexpr int dim0 = 2, dim1 = 1023;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0a->setMemoryType(MemoryType::Shared);

  tv1->merge(0);
  tv1->split(0, 512);
  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  /// TIDx for computation, Bulk for load
  tv0a->axis(-1)->parallelize(ParallelType::Bulk);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  inlineMost();

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  try {
    ke.run({at_tv0});
  } catch (const std::exception& e) {
    const char* reference =
        R"(If split output domain is loaded with 1D TMA, the split must be divisible)";
    const char* str_match_pointer = strstr(e.what(), reference);
    EXPECT_TRUE(str_match_pointer != nullptr);
  }
}

TEST_F(TMATest, CpAsyncBulk1dNonDivisibleUnroll) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};

  constexpr int dim0 = 1023, dim1 = 128;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0a->setMemoryType(MemoryType::Shared);

  tv1->split(0, 2);
  tv1->split(0, 137);
  tv1->reorder({{0, 1}});
  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // circular buffer loop
  tv1->axis(1)->parallelize(ParallelType::Serial);
  // synchronized loop, expect arrive bytes is on top of this loop
  tv1->axis(2)->parallelize(ParallelType::Serial);
  scheduler_utils::parallelizeAllLike(tv1);

  /// TIDx for computation, Bulk for load
  tv1->axis(3)->parallelize(ParallelType::TIDx);
  tv0a->axis(3)->parallelize(ParallelType::Bulk);

  // inline
  inlineSelectedAt({tv0a}, tv0a, 2);
  inlineMost(std::unordered_set<TensorView*>{tv1});

  tv0a->circularBuffer(2, 1, WarpSpecialized(ParallelType::TIDy));

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  try {
    // input is concrete tensor, we can detect the error at compile time
    ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  } catch (const std::exception& e) {
    const char* reference =
        R"(Loop domains between circular buffer and 1D TMA load requires divisible split)";
    const char* str_match_pointer = strstr(e.what(), reference);
    EXPECT_TRUE(str_match_pointer != nullptr) << e.what();
  }
}

TEST_F(TMATest, CpAsyncBulk1dPipplined) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};

  constexpr int dim0 = 1023, dim1 = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0a->setMemoryType(MemoryType::Shared);

  tv1->split(0, 2);
  tv1->split(0, 137);
  tv1->reorder({{0, 1}});
  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  // circular buffer loop
  tv1->axis(1)->parallelize(ParallelType::Serial);
  // synchronized loop, expect arrive bytes is on top of this loop
  tv1->axis(2)->parallelize(ParallelType::Serial);
  scheduler_utils::parallelizeAllLike(tv1);

  /// TIDx for computation, Bulk for load
  tv1->axis(3)->parallelize(ParallelType::TIDx);
  tv0a->axis(3)->parallelize(ParallelType::Bulk);

  // inline
  inlineSelectedAt({tv0a}, tv0a, 2);
  inlineMost(std::unordered_set<TensorView*>{tv1});

  tv0a->circularBuffer(2, 1, Pipelined());

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  try {
    ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  } catch (const std::exception& e) {
    const char* reference =
        R"(1D TMA load can only be used with WarpSpecialized circular buffer:)";
    const char* str_match_pointer = strstr(e.what(), reference);
    EXPECT_TRUE(str_match_pointer != nullptr) << e.what();
  }
}

TEST_F(TMATest, CpAsyncBulk1dNonCircularBuffer) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};

  constexpr int dim0 = 1023, dim1 = 512;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0a->setMemoryType(MemoryType::Shared);

  tv1->split(0, 2);
  tv1->split(0, 137);
  tv1->reorder({{0, 1}});
  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::Serial);
  tv1->axis(2)->parallelize(ParallelType::Unroll);
  scheduler_utils::parallelizeAllLike(tv1);

  /// TIDx for computation, Bulk for load
  tv1->axis(3)->parallelize(ParallelType::TIDx);
  tv0a->axis(3)->parallelize(ParallelType::Bulk);

  // inline
  inlineSelectedAt({tv0a}, tv0a, 2);
  inlineMost(std::unordered_set<TensorView*>{tv1});

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);
  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  auto outputs = ke.run({at_tv0});
  auto at_output = at_tv0 + at_tv0;
  testValidate(
      fusion.get(), outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}

using TMA1dPredicateTestParams = std::tuple<bool, bool>;
using TMA1dPredicateTest = NVFuserFixtureParamTest<TMA1dPredicateTestParams>;
TEST_P(TMA1dPredicateTest, testUnrollCircularBuffer) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  constexpr at::ScalarType dtype = at::ScalarType::Float;
  CompileParams index32bit{DataType::Int32, 255, false};
  auto [has_unroll, has_circular_buffer] = GetParam();
  int64_t circular_stages = 2;
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t outer_unroll = has_unroll ? 2 : 1;
  // Ensure dim0 is divisible by outer_unroll but not
  // divisible by sm_count after divide by outer_unroll
  const int64_t dim0 = (sm_count + 1) * outer_unroll * circular_stages;
  const int64_t dim1 = 128;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto tv0a = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulk);
  tv0a->setMemoryType(MemoryType::Shared);

  if (has_unroll) {
    tv1->split(0, outer_unroll);
  }
  tv1->split(0, sm_count);
  tv1->reorder({{0, 1}});
  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  /// TIDx for computation, Bulk for load
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv0a->axis(-1)->parallelize(ParallelType::Bulk);

  // inline
  inlineSelectedAt({tv0a}, tv0a, 2);
  inlineMost(std::unordered_set<TensorView*>{tv1});

  if (has_circular_buffer) {
    tv0a->circularBuffer(
        circular_stages, 1, WarpSpecialized(ParallelType::TIDy));
  }
  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor at_tv0 = at::randn({dim0, dim1}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {at_tv0}, {}, index32bit);
  auto outputs = ke.run({at_tv0});
  auto at_output = at_tv0 + at_tv0;
  testValidate(
      fusion.get(), outputs, {at_tv0}, {at_output}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    TMATest,
    TMA1dPredicateTest,
    ::testing::Combine(
        testing::Values(true, false),
        testing::Values(true, false)),
    [](const testing::TestParamInfo<TMA1dPredicateTestParams>& info)
        -> std::string {
      std::stringstream ss;
      ss << "has_unroll_" << std::get<0>(info.param);
      ss << "_has_circular_buffer_" << std::get<1>(info.param);
      return sanitizeTestName(ss.str());
    });
} // namespace nvfuser
