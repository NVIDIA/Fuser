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

  // Maps the iter domains of tv0 and tv1, which in turn maps the loop
  // domain of tv4 with its logical domain
  if (getenv("MAP")) {
    auto tv6 = add(tv0, tv1);
    fusion.addOutput(tv6);
  }

  for (auto tv : fusion.allTvs()) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
  }

  tv2->setMemoryType(MemoryType::Shared);
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  fusion.print();

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options);
  auto t1 = at::zeros({m}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

} // namespace nvfuser
