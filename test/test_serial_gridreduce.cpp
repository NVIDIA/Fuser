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
    for (int64_t num_warps : {1, 2, 4, 8, 16, 32}) {
      // B is size of inner serial loop
      for (int64_t B : {1, 2, 4, 8, 16, 32}) {
        std::cout << "serial=" << serial << " num_warps=" << num_warps
                  << " B=" << B << std::endl;
        std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
        auto fusion = fusion_ptr.get();
        FusionGuard fg(fusion);

        int64_t blocks_x = 8;
        int64_t blocks_y = 8;
        int64_t blocks_z = 5;
        int64_t A = 4; // Size of outer serial loop
        int64_t H = blocks_z;
        int64_t W = A * B * blocks_x * blocks_y * num_warps * 32;

        // Unreduced dimensions should be concrete. Reduced dimension may be
        // symbolic
        TensorView* tv0 = TensorViewBuilder()
                              .shape({H, -1})
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
        tv0->cacheAfter();
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

        TransformPropagator propagator(tv3);
        MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
        scheduler_utils::parallelizeAllLike(tv3);

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

            // There should be a single top-level ForLoop. Find its position and
            // check that there is only one.
            size_t top_level_loop_pos = -1;
            for (size_t i : c10::irange(top_level_exprs.size())) {
              Expr* expr = top_level_exprs.at(i);
              if (expr->isA<kir::ForLoop>()) {
                ASSERT_EQ(top_level_loop_pos, -1);
                top_level_loop_pos = i;
              }
            }

            // This is a poor approximation of a traversal that would appear in
            // a lowering pass to both set the isSerial() flag on grid
            // reductions and insert wait/release syncs.
            kir::Scope& tidx_scope = top_level_exprs.at(top_level_loop_pos)
                                         ->as<kir::ForLoop>()
                                         ->body() // BIDx
                                         .at(0)
                                         ->as<kir::ForLoop>()
                                         ->body() // BIDy
                                         .at(0)
                                         ->as<kir::ForLoop>()
                                         ->body() // A
                                         .exprs()
                                         .back()
                                         ->as<kir::ForLoop>()
                                         ->body() // B
                                         .exprs()
                                         .back()
                                         ->as<kir::ForLoop>()
                                         ->body() // TIDy
                                         .exprs()
                                         .back()
                                         ->as<kir::ForLoop>()
                                         ->body(); // TIDx
            kir::Scope& bidz_scope =
                tidx_scope.at(2)->as<kir::ForLoop>()->body(); // BIDz
            // Now scope holds inner scope
            auto old_grop = bidz_scope.at(3)->as<kir::GridReduction>();
            auto output_store_expr =
                tidx_scope.at(3)->as<kir::IfThenElse>()->thenBody().at(0);
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
            bidz_scope.at(3) = new_grop;

            auto sync_buf = global_allocations.at(1)->buffer();

            top_level_exprs.insert(
                top_level_exprs.end() - 1,
                IrBuilder::create<kir::BlockSerializeWait>(
                    ParallelTypeBitmap(ParallelType::BIDz), sync_buf));

            top_level_exprs.push_back(
                IrBuilder::create<kir::BlockSerializeRelease>(
                    ParallelTypeBitmap(ParallelType::BIDz), sync_buf));
          });
        }
        fe.compileFusion(fusion);

        auto input = at::randn(
            {H, W}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
        auto outputs = fe.runFusion({input});

        testValidate(fusion, outputs, {input}, __LINE__, __FILE__);
      }
    }
  }
}

} // namespace nvfuser
