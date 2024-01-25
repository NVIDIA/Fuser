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
  for (bool serial : {true, false}) {
    for (int64_t num_warps : {4, 8}) {
      // B is size of inner serial loop. Outer loop is hardcoded at A=4
      // Here we set B to a small value of 8 instead of 32 (i.e. 128 elements
      // per thread), so that the non-serial compilation does not take too
      // long.
      for (int64_t B : {8}) {
        std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
        auto fusion = fusion_ptr.get();
        FusionGuard fg(fusion);

        int64_t blocks_x = 8;
        int64_t blocks_y = 8;
        int64_t blocks_z = 5;
        int64_t A = 4; // Size of outer serial loop
        int64_t H = blocks_z;
        int64_t W = A * B * blocks_x * blocks_y * num_warps * 32;

        // Unreduced dimensions should be concrete. Reduced dimension could be
        // symbolic, but is concrete here so that we can read tv0 to registers
        TensorView* tv0 = TensorViewBuilder()
                              .shape({H, W})
                              .dtype(DataType::Float)
                              .contiguity(true)
                              .build();
        fusion->addInput(tv0);

        auto tv1 = sum(tv0, {0});
        fusion->addOutput(tv1);

        // Start with
        //   [ rS{H}, iS{W} ]
        // We are grid reducing the H dimension and we want to coalesce
        // accesses in the W dimension. So we first reorder to
        //   [ iS{W}, rS{H} ]
        // then schedule as
        //   [ iBIDx{blocks_x}, iBIDy{blocks_y}, iS{A}, iS{B}, iTIDy{num_warps},
        //   iTIDx{32}, rBIDz{blocks_z} ]
        auto tv2 = tv0->cacheAfter();
        auto tv3 = tv1->cacheBefore();

        tv3->reorder({{1, 0}, {0, 1}}); // blocks_x*blocks_y*A*B*num_warps*32, H
        tv3->split(0, 32); // blocks_x*blocks_y*A*B*num_warps, 32, H
        tv3->split(0, num_warps); // blocks_x*blocks_y*A*B, num_warps, 32, H
        tv3->split(0, B); // blocks_x*blocks_y*A, B, num_warps, 32, H
        tv3->split(0, A); // blocks_x*blocks_y, A, B, num_warps, 32, H
        tv3->split(0, blocks_y); // blocks_x, blocks_y, A, B, num_warps, 32, H
        tv3->axis(0)->parallelize(ParallelType::BIDx);
        tv3->axis(1)->parallelize(ParallelType::BIDy);
        tv3->axis(4)->parallelize(ParallelType::TIDy);
        tv3->axis(5)->parallelize(ParallelType::TIDx);
        tv3->axis(6)->parallelize(ParallelType::BIDz);
        // Reorder to put parallel dims first for better inlining
        tv3->reorder({
            {4, 2},
            {5, 3},
            {2, 4},
            {3, 5},
        });

        TransformPropagator propagator(tv3);
        MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
        scheduler_utils::parallelizeAllLike(tv3);

        // Here we just transpose A and B in tv2, so that it will be partially
        // inlined with tv3, resulting in a separate loop to load tv0 into
        // registers (tv2).
        tv2->reorder({
            {-2, -3},
            {-3, -2},
        });

        inlineMost();

        FusionExecutor fe;
        if (serial) {
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

            // Find the position of the last outer loop
            size_t top_level_loop_pos = -1;
            for (size_t i : c10::irange(top_level_exprs.size())) {
              Expr* expr = top_level_exprs.at(i);
              if (expr->isA<kir::ForLoop>()) {
                top_level_loop_pos = i;
              }
            }

            // This is a poor approximation of a traversal that would appear in
            // a lowering pass to both set the isSerial() flag on grid
            // reductions and insert wait/release syncs.
            //
            // tidx_scope is the inner-most fully parallelized scope. It is
            // "top-level" in that its loops appear as top-level in the
            // generated kernel
            kir::Scope& tidx_scope = top_level_exprs.at(top_level_loop_pos)
                                         ->as<kir::ForLoop>()
                                         ->body() // BIDx
                                         .at(0)
                                         ->as<kir::ForLoop>()
                                         ->body() // BIDy
                                         .at(0)
                                         ->as<kir::ForLoop>()
                                         ->body() // TIDy
                                         .at(0)
                                         ->as<kir::ForLoop>()
                                         ->body(); // TIDx
            kir::Scope& B_scope = tidx_scope.exprs()
                                      .at(5)
                                      ->as<kir::ForLoop>()
                                      ->body() // A (reduction loop)
                                      .exprs()
                                      .back()
                                      ->as<kir::ForLoop>()
                                      ->body(); // B
            // We will need the store op output TensorIndex
            LoadStoreOp* output_store_expr = B_scope.exprs()
                                                 .back()
                                                 ->as<kir::IfThenElse>()
                                                 ->thenBody()
                                                 .at(0)
                                                 ->as<LoadStoreOp>();
            // bidz_scope is the scope containing the GridReduction expression
            kir::Scope& bidz_scope =
                B_scope.exprs().at(4)->as<kir::ForLoop>()->body(); // BIDz
            auto old_grop = bidz_scope.at(0)->as<kir::GridReduction>();
            // Store the TensorIndex for the output tensor T1_g, so that we can
            // re-use its index
            auto t1_idx = output_store_expr->output(0)->as<kir::TensorIndex>();

            // Create new TensorView and Allocate
            auto output = kernel->outputs().at(0)->as<TensorView>();
            Val* i0 = output->getRootDomain().at(0)->extent();
            auto new_work_buf_tv =
                TensorViewBuilder().shape(std::vector<Val*>{i0}).build();
            new_work_buf_tv->setMemoryType(MemoryType::Global);
            // associate the index of the output tensor with the work buffer
            // NOTE: in actual lowering we would generate an index ourselves
            // here, but this works for this test since the T1 store is inlined
            // fully with the serial grid reduction.
            Val* idx = t1_idx->index();

            auto new_work_buf_idx =
                IrBuilder::create<kir::TensorIndex>(new_work_buf_tv, idx);
            auto new_work_buf_alloc = IrBuilder::create<kir::Allocate>(
                new_work_buf_tv, MemoryType::Global, std::vector<Val*>{i0});
            const kir::Allocate* orig_work_buf_alloc = global_allocations[0];
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
                params[i] = new_work_buf_tv;
              }
            }
            // replace the grid reduction Expr
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
                new_work_buf_idx);
            new_grop = new_grop->withPredicate(old_grop->predicate())
                           ->as<kir::GridReduction>();
            new_grop = new_grop->withWritePredicate(old_grop->writePredicate())
                           ->as<kir::GridReduction>();
            bidz_scope.at(0) = new_grop;

            auto sync_buf = global_allocations.at(1)->buffer();

            std::vector<Expr*>& nonpar_top_level_exprs =
                const_cast<std::vector<Expr*>&>(tidx_scope.exprs());
            nonpar_top_level_exprs.insert(
                nonpar_top_level_exprs.end() - 2,
                IrBuilder::create<kir::BlockSerializeWait>(
                    ParallelTypeBitmap(ParallelType::BIDz), sync_buf));

            nonpar_top_level_exprs.insert(
                nonpar_top_level_exprs.end() - 1,
                IrBuilder::create<kir::BlockSerializeRelease>(
                    ParallelTypeBitmap(ParallelType::BIDz), sync_buf));
          });
        }
        fe.compileFusion(fusion);

        auto input = at::randn(
            {H, W}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
        auto outputs = fe.runFusion({input});

        if (serial) {
          testValidate(fusion, outputs, {input}, __LINE__, __FILE__);
        }
      }
    }
  }
}

TEST_F(SerialGridReductionTest, Scheduling) {
  for (bool serial : {true, false}) {
    for (int64_t num_warps : {4, 8}) {
      // B is size of inner serial loop. Outer loop is hardcoded at A=4
      // Here we set B to a small value of 8 instead of 32 (i.e. 128 elements
      // per thread), so that the non-serial compilation does not take too
      // long.
      for (int64_t B : {8}) {
        std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
        auto fusion = fusion_ptr.get();
        FusionGuard fg(fusion);

        int64_t blocks_x = 8;
        int64_t blocks_y = 8;
        int64_t blocks_z = 5;
        int64_t A = 4; // Size of outer serial loop
        int64_t H = blocks_z;
        int64_t W = A * B * blocks_x * blocks_y * num_warps * 32;

        // Unreduced dimensions should be concrete. Reduced dimension could be
        // symbolic, but is concrete here so that we can read tv0 to registers
        TensorView* tv0 = TensorViewBuilder()
                              .shape({H, W})
                              .dtype(DataType::Float)
                              .contiguity(true)
                              .build();
        fusion->addInput(tv0);

        auto tv1 = sum(tv0, {0});
        fusion->addOutput(tv1);

        // Start with
        //   [ rS{H}, iS{W} ]
        // We are grid reducing the H dimension and we want to coalesce
        // accesses in the W dimension. So we first reorder to
        //   [ iS{W}, rS{H} ]
        // then schedule as
        //   [ iBIDx{blocks_x}, iBIDy{blocks_y}, iS{A}, iS{B}, iTIDy{num_warps},
        //   iTIDx{32}, rBIDz{blocks_z} ]
        auto tv2 = tv0->cacheAfter();
        auto tv3 = tv1->cacheBefore();

        tv3->reorder({{1, 0}, {0, 1}}); // blocks_x*blocks_y*A*B*num_warps*32, H
        tv3->split(0, 32); // blocks_x*blocks_y*A*B*num_warps, 32, H
        tv3->split(0, num_warps); // blocks_x*blocks_y*A*B, num_warps, 32, H
        tv3->split(0, B); // blocks_x*blocks_y*A, B, num_warps, 32, H
        tv3->split(0, A); // blocks_x*blocks_y, A, B, num_warps, 32, H
        tv3->split(0, blocks_y); // blocks_x, blocks_y, A, B, num_warps, 32, H
        tv3->axis(0)->parallelize(ParallelType::BIDx);
        tv3->axis(1)->parallelize(ParallelType::BIDy);
        tv3->axis(4)->parallelize(ParallelType::TIDy);
        tv3->axis(5)->parallelize(ParallelType::TIDx);
        tv3->axis(6)->parallelize(ParallelType::BIDz);
        // Reorder to put parallel dims first for better inlining
        tv3->reorder({
            {4, 2},
            {5, 3},
            {2, 4},
            {3, 5},
        });

        TransformPropagator propagator(tv3);
        MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
        scheduler_utils::parallelizeAllLike(tv3);

        // Here we just transpose A and B in tv2, so that it will be partially
        // inlined with tv3, resulting in a separate loop to load tv0 into
        // registers (tv2).
        tv2->reorder({
            {-2, -3},
            {-3, -2},
        });

        inlineMost();

        FusionExecutor fe;
        if (serial) {
          tv3->definition()->as<ReductionOp>()->requestSerialGridReduction();
        }
        fe.compileFusion(fusion);

        auto input = at::randn(
            {H, W}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
        auto outputs = fe.runFusion({input});

        if (serial) {
          testValidate(fusion, outputs, {input}, __LINE__, __FILE__);
        }
      }
    }
  }
}

} // namespace nvfuser
