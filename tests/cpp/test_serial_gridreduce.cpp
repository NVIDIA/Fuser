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
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
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
