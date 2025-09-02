// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <exceptions.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class ScatterTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

// Counting of non-duplicated integers on gmem with TIDx
TEST_F(ScatterTest, BlockCountingWithGmem) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 64;
  const int64_t n = 5;

  auto tv0 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros({IrBuilder::create<Val>(m)}, DataType::Int);
  auto tv3 = ones({IrBuilder::create<Val>(n)}, DataType::Int);
  auto tv4 = scatter(tv2, 0, tv1, tv3);
  fusion.addOutput(tv4);

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Counting of non-duplicated integers on shmem with TIDx
TEST_F(ScatterTest, BlockCountingWithShmem) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 64;
  const int64_t n = 5;

  auto tv0 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros({IrBuilder::create<Val>(m)}, DataType::Int);
  auto tv3 = ones({IrBuilder::create<Val>(n)}, DataType::Int);
  auto tv4 = scatter(tv2, 0, tv1, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Shared);
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Counting of non-duplicated integers on gmem with BIDx
TEST_F(ScatterTest, GridCounting) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 64;
  const int64_t n = 5;

  auto tv0 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros({IrBuilder::create<Val>(m)}, DataType::Int);
  auto tv3 = ones({IrBuilder::create<Val>(n)}, DataType::Int);
  auto tv4 = scatter(tv2, 0, tv1, tv3);
  fusion.addOutput(tv4);

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScatterTest, BlockCountingWithShmem2D) {
  // Scatter allows the non-indexed domains of the index tensor to
  // have smaller extents, which causes indexing error as there's not
  // traversal path. It is not currently supported.
  GTEST_SKIP() << "Scatter with multi-dimensional tensors not supported yet";

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const std::vector<int64_t> self_shape{4, 100};
  const std::vector<int64_t> index_shape{2, 10};

  auto tv0 = makeContigConcreteTensor(index_shape, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros(
      {IrBuilder::create<Val>(self_shape[0]),
       IrBuilder::create<Val>(self_shape[1])},
      DataType::Int);
  auto tv3 = ones(
      {IrBuilder::create<Val>(index_shape[0]),
       IrBuilder::create<Val>(index_shape[1])},
      DataType::Int);
  auto tv4 = scatter(tv2, 1, tv1, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Shared);
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(self_shape[1], options).slice(0, 0, index_shape[1]);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScatterTest, CacheBefore) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 64;
  const int64_t n = 5;

  auto tv0 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = zeros({IrBuilder::create<Val>(m)}, DataType::Int);
  auto tv3 = ones({IrBuilder::create<Val>(n)}, DataType::Int);
  auto tv4 = scatter(tv2, 0, tv1, tv3);
  fusion.addOutput(tv4);

  auto output_cache = tv4->cacheBefore();

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Shared);
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);
  output_cache->setMemoryType(MemoryType::Shared);
  output_cache->setAllocationDomain(output_cache->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(ScatterTest, CacheAfter) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 64;
  const int64_t n = 5;

  auto tv0 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({m}, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = ones({IrBuilder::create<Val>(n)}, DataType::Int);
  auto tv3 = scatter(tv1, 0, tv0, tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);

  auto input_cache = tv1->cacheAfter();

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  input_cache->setMemoryType(MemoryType::Shared);
  input_cache->setAllocationDomain(input_cache->getLogicalDomain(), true);
  tv3->setMemoryType(MemoryType::Shared);
  tv3->setAllocationDomain(tv3->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);
  auto t1 = at::zeros({m}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Repro of #4929. In ScatterOp, the logical domain of the output tensor is not
// automatically mapped with its loop domain, but it's possible they
// happen to be mapped. Make sure proper syncronizations are inserted
// even in that case.
TEST_F(ScatterTest, MappedLogicalAndLoop) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int64_t m = 8;

  auto tv0 = makeContigConcreteTensor({m}, DataType::Int);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({m}, DataType::Int);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = arange(IrBuilder::create<Val>(8));
  auto tv4 = scatter(tv2, 0, tv0, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  // At this point, tv4's loop ID is not mapped with its sole logical
  // ID but mapped with tv0's logical ID. This means that the loop
  // ID is not mapped with the loop ID of the input of the op,
  // tv2. When parallelized, this difference of the loop IDs should
  // cause the sync analysis to flag a potential RAW race. However, it
  // is possible they happen to be mapped, e.g., by an additional op
  // like below:
  auto tv6 = add(tv0, tv1);
  fusion.addOutput(tv6);

  // The binary add op maps the logical domains of tv0 and tv1, which
  // in turn maps the loop domain of tv4 with its logical domain. Make
  // sure that the proper synchronizations are inserted even in cases
  // like this.

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  tv2->setMemoryType(MemoryType::Shared);
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options);
  auto t1 = at::randint(0, 100, {m}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);

  // There must be a block sync both before and after the scatter
  // op.
  bool pre_scatter_sync_found = false;
  bool scatter_found = false;
  bool post_scatter_sync_found = false;
  for (const auto expr :
       KernelExprVisitor::getAllExprs(ke.compiledKernel()->kernel())) {
    if (auto scatter = dynamic_cast<ScatterOp*>(expr)) {
      EXPECT_TRUE(pre_scatter_sync_found)
          << "Sync before scatter not found: " << scatter->toString();
      scatter_found = true;
    }
    if (auto sync = dynamic_cast<kir::BlockSync*>(expr)) {
      if (!scatter_found) {
        EXPECT_FALSE(pre_scatter_sync_found)
            << "Only one sync before scatter expected: " << sync->toString();
        pre_scatter_sync_found = true;
      } else {
        EXPECT_FALSE(post_scatter_sync_found)
            << "Only one sync after scatter expected: " << sync->toString();
        post_scatter_sync_found = true;
      }
    }
  }
  EXPECT_TRUE(pre_scatter_sync_found) << "Sync before scatter not found";
  EXPECT_TRUE(scatter_found) << "Scatter not found in Kernel";
  EXPECT_TRUE(post_scatter_sync_found) << "Sync after scatter not found";
}

class ScatterAccumulateTest
    : public NVFuserFixtureParamTest<
          std::tuple<int64_t, int64_t, PrimDataType, BinaryOpType>> {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

    std::tie(m, n, dtype, accumulate_op) = GetParam();
  }

 protected:
  int64_t m = 8;
  int64_t n = 128;
  PrimDataType dtype = PrimDataType::Int;
  BinaryOpType accumulate_op = BinaryOpType::Add;
};

TEST_P(ScatterAccumulateTest, BlockParallelScatterAccumulate) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({m}, dtype);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({n}, DataType::Int);
  fusion.addInput(tv1);
  auto tv2 = makeContigConcreteTensor({n}, dtype);
  fusion.addInput(tv2);

  auto tv3 = set(tv0);
  auto tv4 = scatter(tv3, 0, tv1, tv2, accumulate_op);

  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // Scatter input must use the same memory as the output
  tv3->setMemoryType(MemoryType::Shared);
  tv3->setAllocationDomain(tv3->getLogicalDomain(), true);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = isIntegralType(dtype) ? at::randint(0, 100, {m}, options)
                                  : at::randn({m}, options);
  auto t1 = at::randint(0, m, {n}, options_int);
  auto t2 = isIntegralType(dtype) ? at::randint(0, 100, {n}, options)
                                  : at::randn({n}, options);

  KernelExecutor ke;
  if (isFloatingPointType(dtype) &&
      (accumulate_op == BinaryOpType::Max ||
       accumulate_op == BinaryOpType::Min)) {
    EXPECT_THAT(
        [&]() { ke.compile(&fusion, {t0, t1, t2}); },
        testing::ThrowsMessage<nvfuser::nvfError>(
            testing::HasSubstr("accumulation not supported")));
  } else {
    ke.compile(&fusion, {t0, t1, t2});
    auto outputs = ke.run({t0, t1, t2});
    testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    ScatterAccumulateTest,
    testing::Combine(
        testing::Values(8, 32),
        testing::Values(8, 32, 128),
        testing::Values(PrimDataType::Int, PrimDataType::Float),
        testing::Values(
            BinaryOpType::Add,
            BinaryOpType::Max,
            BinaryOpType::Min)),
    [](const testing::TestParamInfo<
        std::tuple<int64_t, int64_t, PrimDataType, BinaryOpType>>& info) {
      std::stringstream ss;
      ss << std::get<0>(info.param) << "_" << std::get<1>(info.param) << "_"
         << std::get<2>(info.param) << "_" << std::get<3>(info.param);
      return ss.str();
    });

} // namespace nvfuser
