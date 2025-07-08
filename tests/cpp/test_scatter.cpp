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

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Global);

  fusion.print();
  fusion.printKernel();

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

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);
  tv4->setAllocationDomain(tv4->getLogicalDomain(), true);

  fusion.print();
  fusion.printKernel();

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

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(0)->parallelize(ParallelType::BIDx);

  // Scatter input must use the same memory as the output
  tv2->setMemoryType(MemoryType::Global);

  fusion.print();
  fusion.printKernel();

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randperm(m, options).slice(0, 0, n);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, SyncthreadsWithGmem) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Global);

  // [TIDx, TIDy]
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  // [TIDy, TIDx]
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  fusion.printKernel();
}

} // namespace nvfuser
