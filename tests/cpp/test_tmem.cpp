// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <scheduler/tools/inlining.h>
#include <type.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

// Tensor memory tests
using TMemTest = BlackwellBase;

TEST_F(TMemTest, GmemRegTMemRegGmemCopy) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0); // register
  auto tv2 = set(tv1); // tmem
  auto tv3 = set(tv2); // register
  auto tv4 = set(tv3); // gmem
  fusion.addOutput(tv4);

  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->split(0, 32);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv4, {tv1, tv2, tv3});

  tv2->setAllocationDomain(tv2->getLoopDomain(), true);
  tv2->setTMemDimSepPos(-1);

  inlineMost();

  KernelExecutor ke;
  ke.compile(&fusion);
  auto t0 = at::randn(
      {12800}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

void testTMemAddKernel(bool same_region) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0); // register
  auto tv2 = set(tv1); // tmem
  auto tv3 = set(tv2); // register
  auto tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);
  auto tv5 = set(tv4); // register
  auto tv6 = set(tv5); // tmem
  auto tv7 = set(tv6); // register
  auto tv8 = add(tv3, tv7); // register
  auto tv9 = set(tv8); // gmem
  fusion.addOutput(tv9);

  if (same_region) {
    using Region = std::vector<TensorView*>;
    Region region1{tv2, tv6};
    std::vector<Region> regions{region1};
    fusion.manage("tmem_regions", regions);
  }

  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv6->setMemoryType(MemoryType::Tensor);
  tv6->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv7->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv9->split(0, 32);

  TransformPropagator propagator(tv9);
  MaxLogicalDomainInfoSpanningTree(tv9).traverse(&propagator);

  tv9->axis(0)->parallelize(ParallelType::BIDx);
  tv9->axis(1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv9);

  for (auto tv : {tv2, tv6}) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
    tv->setTMemDimSepPos(-1);
  }

  inlineMost();

  KernelExecutor ke;

  // check number of tcgen05.alloc calls
  ke.registerLoweringHook([same_region](GpuLower* lower) {
    auto check_pass = [same_region](const std::vector<Expr*>& exprs) {
      int64_t num_allocs =
          std::count_if(exprs.begin(), exprs.end(), [](Expr* expr) {
            std::string str = expr->toString();
            return str.find("tcgen05.alloc") != std::string::npos;
          });
      EXPECT_EQ(num_allocs, same_region ? 1 : 2);
      int64_t num_deallocs = 0;
      for (auto expr : exprs) {
        std::string str = expr->toString();
        std::string sub = "tcgen05.dealloc";
        // count number of sub in str
        size_t pos = 0;
        while ((pos = str.find(sub, pos)) != std::string::npos) {
          ++num_deallocs;
          pos += sub.length();
        }
      }
      EXPECT_EQ(num_deallocs, same_region ? 1 : 2);
      return exprs;
    };
    lower->passes().push_back({"Check result", check_pass});
  });

  ke.compile(&fusion);
  auto t0 = at::randn(
      {12800}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto t1 = at::randn(
      {12800}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto cg_outputs = ke.run({t0, t1});
  testValidate(&fusion, cg_outputs, {t0, t1}, {t0 + t1}, __LINE__, __FILE__);
}

TEST_F(TMemTest, AddKernelMultipleRegions) {
  testTMemAddKernel(false);
}

TEST_F(TMemTest, AddKernelSameRegion) {
  testTMemAddKernel(true);
}

TEST_F(TMemTest, dtypes) {
  const std::vector<DataType> data_types{
      DataType::Char,
      DataType::Half,
      DataType::Float,
      DataType::Double,
      DataType::ComplexDouble};

  const std::vector<int64_t> vec_factors = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

  int64_t expect_dtype_bytes = 1;
  for (auto dtype : data_types) {
    EXPECT_EQ(dataTypeSizeByte(dtype), expect_dtype_bytes);
    for (int64_t vec : vec_factors) {
      int64_t vec_bytes = expect_dtype_bytes * vec;
      if (vec_bytes > 512) {
        continue;
      }
      Fusion fusion;
      FusionGuard fg(&fusion);

      auto tv0 = makeContigConcreteTensor({128, 512}, dtype);
      fusion.addInput(tv0);
      auto tv1 = set(tv0);
      auto tv2 = set(tv1);
      auto tv3 = set(tv2);
      auto tv4 = set(tv3);
      fusion.addOutput(tv4);
      tv2->setMemoryType(MemoryType::Tensor);
      tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
      tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

      for (auto tv : {tv1, tv2, tv3, tv4}) {
        tv->axis(0)->parallelize(ParallelType::TIDx);
        tv->split(1, vec);
      }
      tv2->axis(-1)->parallelize(ParallelType::Vectorize);
      tv3->axis(-1)->parallelize(ParallelType::Vectorize);

      tv2->setAllocationDomain(tv2->getLoopDomain(), true);
      tv2->setTMemDimSepPos(1);

      inlineMost();

      KernelExecutor ke;

      if (vec_bytes % 4 != 0) {
        std::string message = vec_bytes == 1
            ? "Tried to vectorize a dim resulting in a word size of 1 however, "
              "vector sizes only upto and including 512 bytes are supported."
            : "Vectorize size is not a multiple of 4 bytes";
        EXPECT_THAT(
            [&]() { ke.compile(&fusion); },
            ::testing::ThrowsMessage<nvfuser::nvfError>(
                ::testing::HasSubstr(message)));
        continue;
      }

      // check allocation size of tcgen05.alloc calls
      ke.registerLoweringHook([vec_bytes](GpuLower* lower) {
        auto check_pass = [vec_bytes](const std::vector<Expr*>& exprs) {
          bool found_alloc = false;
          for (Expr* expr : ir_utils::flattenScopedExprs(exprs)) {
            std::string str = expr->isA<kir::Asm>()
                ? expr->as<kir::Asm>()->code()
                : std::string();
            if (str.find("tcgen05.alloc") != std::string::npos) {
              EXPECT_FALSE(found_alloc);
              found_alloc = true;
            } else {
              continue;
            }
            Val* alloc_size = expr->input(1);
            Val* expected_size =
                IrBuilder::create<Val>(static_cast<int64_t>(std::max<int64_t>(
                    std::bit_ceil(static_cast<uint64_t>(vec_bytes / 4)), 32)));
            EXPECT_TRUE(
                simplifyExpr(IrBuilder::eqExpr(alloc_size, expected_size))
                    ->isTrue());
          }
          EXPECT_TRUE(found_alloc);
          return exprs;
        };
        lower->passes().push_back({"Check result", check_pass});
      });

      ke.compile(&fusion);

      at::TensorOptions options = at::TensorOptions()
                                      .dtype(data_type_to_aten(dtype))
                                      .device(at::kCUDA, 0);
      at::Tensor t0 = dtype == DataType::Char
          ? at::randint(-128, 128, {128, 512}, options)
          : at::randn({128, 512}, options);
      auto out = ke.run({t0});
      EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));

      // Check that vectorized PTX instructions are used
      GpuLower gpulw(&fusion);
      auto kernel_str = codegen::generateCudaKernel(gpulw.run());
      std::stringstream expect_st, expect_ld;
      expect_st << "tcgen05.st.sync.aligned.32x32b.x"
                << ir_utils::getTMemLdStVectorizeSize(tv2) << ".b32";
      expect_ld << "tcgen05.ld.sync.aligned.32x32b.x"
                << ir_utils::getTMemLdStVectorizeSize(tv3) << ".b32";
      EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_st.str()));
      EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_ld.str()));
    }
    expect_dtype_bytes *= 2;
  }
}

using TMemTestCompileOnly = NVFuserTest;

TEST_F(TMemTestCompileOnly, SetTMemDimSepPosNonTMem) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 33});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  EXPECT_THAT(
      [&]() { tv1->setTMemDimSepPos(-1); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "TMem dimension separator is only supported for tensor memory")));
}

// Test that we are checking the stride of the "outer parallel types".
// If in a kernel, the parallel dimension map is [TIDy, TIDx] = [2, 33],
// But in the TMem load/store's loop domain, Ix (the ID parallelized on TIDx)
// have extent 32. Then we will generate code like:
//   if (threadIdx.x < 32) {
//     tcgen05::load
//   }
// For threadIdx.y == 0, it is correct. But for threadIdx.y == 1, it is wrong
// because we are using the thread id 33-65 for the load, which is not a warp.
TEST_F(TMemTestCompileOnly, WrongStride) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 33});
  fusion.addInput(tv0);
  auto tv1 = set(tv0); // gmem
  auto tv2 = set(tv1); // register
  auto tv3 = set(tv2); // tmem
  auto tv4 = set(tv3); // register
  auto tv5 = set(tv4); // gmem
  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Tensor);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv4->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  // [TIDy{2}, TIDx{33}]
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  // [TIDy{2}, Serial{2}, TIDx{32}]
  for (auto tv : {tv2, tv3, tv4, tv5}) {
    tv->split(1, 32);
    tv->axis(0)->parallelize(ParallelType::TIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  tv3->setAllocationDomain(tv3->getLoopDomain(), true);
  tv3->setTMemDimSepPos(-1);

  inlineMost();

  KernelExecutor ke;

  EXPECT_THAT(
      [&]() { ke.compile(&fusion); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store: "
          "Outer parallel types' strides must be a multiple of 32.")));
}

// This test is a variant of the WrongStride test, but this test is valid.
// Test a case where the parallel types are not exact. The parallel dimension
// map is [TIDy, TIDx] = [2, 33], but in the TMem load/store's loop domain,
// we have Iy{1}, Ix{32}. the generated code will be like:
//   if (threadIdx.x < 32 && threadIdx.y < 1) {
//     tcgen05::load
//   }
// This is valid because we are using a whole warp for the load.
TEST_F(TMemTest, InexactParallelType) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 33});
  fusion.addInput(tv0);
  auto tv1 = set(tv0); // smem
  auto tv2 = set(tv1); // register
  auto tv3 = set(tv2); // tmem
  auto tv4 = set(tv3); // register
  auto tv5 = set(tv4); // gmem
  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Tensor);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv4->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  // [TIDy{2}, TIDx{33}]
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  // [Serial{2}, TIDy{1}, Serial{2}, TIDx{32}]
  for (auto tv : {tv2, tv3, tv4, tv5}) {
    tv->split(1, 32);
    tv->split(0, 1);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  tv3->setAllocationDomain(tv3->getLoopDomain(), true);
  tv3->setTMemDimSepPos(-1);

  inlineMost();

  KernelExecutor ke;
  ke.compile(&fusion);
  auto t0 = at::randn(
      {2, 33}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

using AllocationSizeTestParams = int64_t; // num cols

class TMemAllocationSize
    : public TMemTest,
      public ::testing::WithParamInterface<AllocationSizeTestParams> {
 protected:
  int64_t num_cols;

  void SetUp() override {
    TMemTest::SetUp();
    num_cols = GetParam();
  }
};

TEST_P(TMemAllocationSize, CopyKernel) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t size = 32 * num_cols;

  auto tv0 = makeContigConcreteTensor({size});
  fusion.addInput(tv0);
  auto tv1 = set(tv0); // register
  auto tv2 = set(tv1); // tmem
  auto tv3 = set(tv2); // register
  auto tv4 = set(tv3); // gmem
  fusion.addOutput(tv4);

  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->split(0, num_cols);

  TransformPropagator propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv4, {tv1, tv2, tv3});

  tv2->setAllocationDomain(tv2->getLoopDomain(), true);
  tv2->setTMemDimSepPos(1);

  KernelExecutor ke;

  // check allocation size of tcgen05.alloc calls
  ke.registerLoweringHook([this](GpuLower* lower) {
    auto check_pass = [this](const std::vector<Expr*>& exprs) {
      bool found_alloc = false;
      for (Expr* expr : ir_utils::flattenScopedExprs(exprs)) {
        std::string str = expr->isA<kir::Asm>() ? expr->as<kir::Asm>()->code()
                                                : std::string();
        if (str.find("tcgen05.alloc") != std::string::npos) {
          EXPECT_FALSE(found_alloc);
          found_alloc = true;
        } else {
          continue;
        }
        Val* alloc_size = expr->input(1);
        Val* expected_size = IrBuilder::create<Val>(static_cast<int64_t>(
            std::bit_ceil(static_cast<uint64_t>(num_cols))));
        EXPECT_TRUE(simplifyExpr(IrBuilder::eqExpr(alloc_size, expected_size))
                        ->isTrue());
      }
      EXPECT_TRUE(found_alloc);
      return exprs;
    };
    lower->passes().push_back({"Check result", check_pass});
  });

  ke.compile(&fusion);
  auto t0 = at::randn(
      {size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

std::string allocationSizeTestName(
    const ::testing::TestParamInfo<AllocationSizeTestParams>& info) {
  return std::to_string(info.param) + "cols";
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TMemAllocationSize,
    ::testing::Values(32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257),
    allocationSizeTestName);

} // namespace nvfuser
