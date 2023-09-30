// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <fusion.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class MBarrierTest : public NVFuserTest {
  void SetUp() override {
    // requires Ampere or newer
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    }
    NVFuserTest::SetUp();
  }
};

TEST_F(MBarrierTest, Simple) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({32, 32});
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  fe.registerPostLoweringHook([](kir::Kernel* kernel) {
    // Replace block sync with mbarrier
    FusionGuard fg(kernel);

    std::vector<Expr*>& top_level_exprs =
        const_cast<std::vector<Expr*>&>(kernel->topLevelExprs());
    kir::KernelSummary& summary =
        const_cast<kir::KernelSummary&>(kernel->summary());

    // Allocate mbarrier
    std::vector<const nvfuser::kir::Allocate*>& dynamic_smem_allocations =
        summary.dynamic_smem_allocations;
    ASSERT_EQ(dynamic_smem_allocations.size(), 1);

    TensorView* mbarrier = makeContigConcreteTensor({}, DataType::UInt);
    mbarrier->setMemoryType(MemoryType::Shared);
    kir::Allocate* mbarrier_alloc =
        IrBuilder::create<kir::Allocate>(mbarrier, MemoryType::Shared);
    dynamic_smem_allocations.push_back(mbarrier_alloc);

    Val* mbarrier_address = SimplifyingIrBuilder::mulExpr(
        dynamic_smem_allocations.at(0)->size(),
        dataTypeSize(dynamic_smem_allocations.at(0)->buffer()->dtype()));
    mbarrier_alloc->setAddress(mbarrier_address);

    auto smem_alloc_it = std::find_if(
        top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
          if (auto alloc = dynamic_cast<kir::Allocate*>(expr)) {
            if (auto tv = dynamic_cast<TensorView*>(alloc->buffer())) {
              return tv->getMemoryType() == MemoryType::Shared;
            } else {
              return false;
            }
          }
          return false;
        });
    smem_alloc_it++;
    ASSERT_NE(smem_alloc_it, top_level_exprs.end());
    smem_alloc_it = top_level_exprs.insert(smem_alloc_it, mbarrier_alloc);

    // Indexing mbarrier
    auto mbarrier_smem_addr = IrBuilder::create<Val>(DataType::SMemAddress);
    IrBuilder::create<UnaryOp>(
        UnaryOpType::ToUnsignedSmemAddr,
        mbarrier_smem_addr,
        IrBuilder::metadataExpr(mbarrier));
    auto mbarrier_index =
        IrBuilder::create<kir::TensorIndex>(mbarrier, mbarrier_smem_addr);

    // Initialize mbarrier
    smem_alloc_it++;
    ASSERT_NE(smem_alloc_it, top_level_exprs.end());
    auto init = IrBuilder::create<kir::MBarrierInit>(
        mbarrier_index, IrBuilder::create<Val>(1024, DataType::UInt32));
    top_level_exprs.insert(smem_alloc_it, init);

    // Arrive and wait
    auto sync_it = std::find_if(
        top_level_exprs.begin(), top_level_exprs.end(), [](Expr* expr) {
          return expr->isA<kir::BlockSync>();
        });
    ASSERT_NE(sync_it, top_level_exprs.end());
    auto state = IrBuilder::create<Val>(DataType::UInt);
    auto alloc_state = IrBuilder::create<kir::Allocate>(
        state, MemoryType::Local, kernel->oneVal());
    auto arrive = IrBuilder::create<kir::MBarrierArrive>(state, mbarrier_index);
    auto wait = IrBuilder::create<kir::MBarrierWait>(mbarrier_index, state);
    *sync_it = wait;
    sync_it = top_level_exprs.insert(sync_it, arrive);
    top_level_exprs.insert(sync_it, alloc_state);

    // Invalidate mbarrier
    auto invalidate =
        IrBuilder::create<kir::MBarrierInvalidate>(mbarrier_index);
    top_level_exprs.push_back(invalidate);
  });

  fe.compileFusion(&fusion);

  auto input = at::randn(
      {32, 32}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputs = fe.runFusion({input});

  testValidate(&fusion, outputs, {input}, __LINE__, __FILE__);
}

} // namespace nvfuser
