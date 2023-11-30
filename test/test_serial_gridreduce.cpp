// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <grouped_reduction.h>
#include <inlining.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using SerialGridReductionTest = NVFuserTest;

// Test that we are able to generate code for a serial reduction
// TODO: remove this test once lowering of serial grid reductions is implemented
TEST_F(SerialGridReductionTest, CodegenNodes) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Unreduced dimensions should be concrete. Reduced dimension may be symbolic
  TensorView* tv0 = TensorViewBuilder()
                        .shape({16384, -1})
                        .dtype(DataType::Float)
                        .contiguity(true)
                        .build();
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  fusion->addOutput(tv1);

  // Schedule grid reduction directly on last axis. Split unreduced axis to
  // generate a loop nest. Each thread needs to do 8 grid reductions.
  //   [ {16384}, {i0} ]
  // becomes
  //   [ iBIDx{8}, iBIDy{8}, iS{4}, iS{2}, iTIDx{32}, rBIDz{i0} ]
  tv0->cacheAfter();
  auto tv3 = tv1->cacheBefore();

  tv3->split(0, 32);
  tv3->split(0, 2);
  tv3->split(0, 4);
  tv3->split(0, 8);
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::BIDy);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::BIDz);

  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);

  inlineMost();

  FusionExecutor fe;
  fe.registerPostLoweringHook([](kir::Kernel* kernel) {
    FusionGuard fg(kernel);

    std::vector<Expr*>& top_level_exprs =
        const_cast<std::vector<Expr*>&>(kernel->topLevelExprs());
    kir::KernelSummary& summary =
        const_cast<kir::KernelSummary&>(kernel->summary());
    std::vector<const kir::Allocate*>& global_allocations =
        summary.global_allocations;
    // There should be a work buffer and a sync buffer allocated
    ASSERT_EQ(global_allocations.size(), 2);

    // Create new TensorView and Allocate
    auto output = kernel->outputs().at(0)->as<TensorView>();
    Val* i0 = output->getRootDomain().at(0)->extent();
    auto new_work_buf =
        TensorViewBuilder().shape(std::vector<Val*>{i0}).build();
    new_work_buf->setMemoryType(MemoryType::Global);
    auto new_work_buf_alloc = IrBuilder::create<kir::Allocate>(
        new_work_buf, MemoryType::Global, std::vector<Val*>{i0});
    auto orig_work_buf_alloc = global_allocations[0];
    global_allocations[0] = new_work_buf_alloc;
    // replace work buf alloc expr in top_level_exprs
    for (auto i : c10::irange(top_level_exprs.size())) {
      if (top_level_exprs[i] == orig_work_buf_alloc) {
        top_level_exprs[i] = new_work_buf_alloc;
      }
    }
    // replace work buf in kernel->parameters()
    std::vector<Val*>& params =
        const_cast<std::vector<Val*>&>(kernel->parameters());
    for (auto i : c10::irange(params.size())) {
      if (params[i] == orig_work_buf_alloc->buffer()) {
        params[i] = new_work_buf;
      }
    }

    // TODO:
    // - set allocation size to be same as output of reduction op
    // - codegen serial GridReduction

    // There should be a single top-level ForLoop. Find its position and check
    // that there is only one.
    size_t top_level_loop_pos = -1;
    for (size_t i : c10::irange(top_level_exprs.size())) {
      Expr* expr = top_level_exprs.at(i);
      if (expr->isA<kir::ForLoop>()) {
        ASSERT_EQ(top_level_loop_pos, -1);
        top_level_loop_pos = i;
      }
    }

    // This is a poor approximation of a traversal that would appear in a
    // lowering pass to both set the isSerial() flag on grid reductions and
    // insert wait/release syncs.
    kir::Scope& scope = top_level_exprs.at(top_level_loop_pos)
                            ->as<kir::ForLoop>()
                            ->body()
                            .at(0)
                            ->as<kir::ForLoop>()
                            ->body() // BIDy
                            .at(0)
                            ->as<kir::ForLoop>()
                            ->body() // i131
                            .exprs()
                            .back()
                            ->as<kir::ForLoop>()
                            ->body() // i130
                            .exprs()
                            .back()
                            ->as<kir::ForLoop>()
                            ->body() // TIDx
                            .at(2)
                            ->as<kir::ForLoop>()
                            ->body(); // BIDz
    // Now scope holds inner scope. Replace the grid reduction there
    auto old_grop = scope.at(3)->as<kir::GridReduction>();
    auto new_grop = IrBuilder::create<kir::GridReduction>(
        old_grop->getReductionOpType(),
        old_grop->init(),
        old_grop->out(),
        old_grop->in(),
        new_work_buf_alloc,
        old_grop->sync_buffer(),
        old_grop->entrance_index(),
        old_grop->entrances(),
        old_grop->isAllreduce(),
        /*is_serial=*/true);
    new_grop = new_grop->withPredicate(old_grop->predicate())
                   ->as<kir::GridReduction>();
    new_grop = new_grop->withWritePredicate(old_grop->writePredicate())
                   ->as<kir::GridReduction>();
    scope.at(3) = new_grop;

    auto sync_buf = global_allocations.at(1)->buffer();

    top_level_exprs.insert(
        top_level_exprs.end() - 1,
        IrBuilder::create<kir::SerialReductionPreSync>(
            ParallelTypeBitmap(ParallelType::BIDz), sync_buf));

    top_level_exprs.push_back(IrBuilder::create<kir::SerialReductionPostSync>(
        ParallelTypeBitmap(ParallelType::BIDz), sync_buf));
  });
  fe.compileFusion(fusion);

  auto input = at::randn(
      {16384, 256}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  auto outputs = fe.runFusion({input});

  testValidate(fusion, outputs, {input}, __LINE__, __FILE__);
}

} // namespace nvfuser
