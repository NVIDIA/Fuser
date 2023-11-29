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
  // [ {16384}, {i0} ] -> [ iBIDx{8}, iBIDy{8}, iS{4}, iS{2}, iTIDx{32},
  // rBIDz{i0} ]
  tv0->cacheAfter();
  auto tv3 = tv1->cacheBefore();

  tv3->split(0, 2);
  tv3->split(0, 4);
  tv3->split(0, 4);
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::BIDy);

  TransformPropagator propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);

  inlineMost();

  // Lower then insert sync nodes manually around top-level loop
  GpuLower gpulw(fusion);
  auto kernel = gpulw->run();

  // TODO: insert syncs and modify node to enable serial reduction codegen
  // - set allocation size to be same as output of reduction op
  // - swap GridReduction node with one having isSerial() == true
  // - insert {Pre,Post}SerialReductionSync nodes before and after main loop

  auto kernel_code_ =
      codegen::generateCudaKernel(kernel, "serial_gridreduce_kernel");

  std::cout << kernel_code_ << std::endl;
}

} // namespace nvfuser
