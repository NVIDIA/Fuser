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

#include <executor.h>
#include <executor_utils.h>
#include <fusion.h>
#include <fusion_profiler.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class FusionProfilerTest : public NVFuserTest {};

// RUN CMD: bin/nvfuser_tests --gtest_filter="*Profile1Segment*"
TEST_F(FusionProfilerTest, Profile1Segment) {
  FusionProfiler* fp = nullptr;
  try {
    fp = FusionProfiler::get();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Getting the FusionProfiler singleton failed!" << e.what();
  }

  ASSERT_FALSE(fp == nullptr);

  try {
    fp->start();
    Fusion fusion;
    FusionGuard fg(&fusion);
 
    auto shape = std::vector<int64_t>({4, 4});
    auto tv0 = makeConcreteTensor(shape);
    auto tv1 = makeConcreteTensor(shape);
    fusion.addInput(tv0);
    fusion.addInput(tv1);
 
    auto tv2 = add(tv0, tv1);
    fusion.addOutput(tv2);

    fp->createSegments(1);
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn(shape, options);
    auto t1 = at::randn(shape, options);
    std::vector<c10::IValue> aten_inputs({t0, t1});
    FusionExecutor fe;
    fp->segment(0).startCompile(/*device*/0);
    fe.compileFusion(&fusion, aten_inputs);
    fp->segment(0).stopCompile();

    fp->segment(0).startKernel(/*device*/0);
    auto cg_outputs = fe.runFusion(aten_inputs);
    fp->segment(0).stopKernel();
    int64_t bytes = 0;
    for (auto size : shape) {
      if (bytes == 0) {
        bytes = size * (int64_t)sizeof(float);
      } else {
        bytes *= size * (int64_t)sizeof(float);
      }
    }
    fp->segment(0).inputBytesAccessed(2 * bytes);
    fp->segment(0).outputBytesAccessed(bytes);
    fp->inputBytesAccessed(2 * bytes);
    fp->outputBytesAccessed(bytes);
    fp->stop();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Defining and profiling the fusion failed!" << e.what();
  }

  auto fprof = fp->profile();
  ASSERT_TRUE(fprof.kernel_profiles.size() == 1);
}
} // namespace nvfuser
