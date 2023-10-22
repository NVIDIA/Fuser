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
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    EnableOptionsGuard::getCurOptions().set(EnableOption::FusionProfiler);
 
    auto shape = std::vector<int64_t>({4, 4});
    auto tv0 = makeConcreteTensor(shape);
    auto tv1 = makeConcreteTensor(shape);
    fusion->addInput(tv0);
    fusion->addInput(tv1);
 
    auto tv2 = add(tv0, tv1);
    fusion->addOutput(tv2);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn(shape, options);
    auto t1 = at::randn(shape, options);
  
    FusionExecutorCache executor_cache(std::move(fusion));
    auto outputs = executor_cache.runFusionWithInputs({t0, t1});
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Defining and profiling the fusion failed!" << e.what();
  }

  auto fprof = fp->profile();
  ASSERT_TRUE(fprof.kernel_profiles.size() == 1);
}

TEST_F(FusionProfilerTest, Profile3Segments) {
  FusionProfiler* fp = nullptr;
  try {
    fp = FusionProfiler::get();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Getting the FusionProfiler singleton failed!" << e.what();
  }

  ASSERT_FALSE(fp == nullptr);

  try {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    EnableOptionsGuard::getCurOptions().set(EnableOption::FusionProfiler);
 
    auto shape1 = std::vector<int64_t>({11});
    auto shape2 = std::vector<int64_t>({13});
    auto shape3 = std::vector<int64_t>({17});
    auto tv0 = makeConcreteTensor(shape1);
    auto tv1 = makeConcreteTensor(shape2);
    auto tv2 = makeConcreteTensor(shape3);
    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
 
    auto s0 = IrBuilder::create<Val>(2.0);
    auto tv3 = mul(tv0, s0);
    auto tv4 = mul(tv1, s0);
    auto tv5 = mul(tv2, s0);
    fusion->addOutput(tv3);
    fusion->addOutput(tv4);
    fusion->addOutput(tv5);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn(shape1, options);
    auto t1 = at::randn(shape2, options);
    auto t2 = at::randn(shape3, options);
    
    FusionExecutorCache executor_cache(std::move(fusion));
    auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Defining and profiling the fusion failed!" << e.what();
  }

  auto fprof = fp->profile();
  ASSERT_TRUE(fprof.kernel_profiles.size() == 3);
}

TEST_F(FusionProfilerTest, FusionProfilerErrorChecks) {
  FusionProfiler* fp = nullptr;
  try {
    fp = FusionProfiler::get();
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Getting the FusionProfiler singleton failed!" << e.what();
  }

  ASSERT_FALSE(fp == nullptr);

  // Make error checks for state when it is Ready and methods expect
  // something else.

  try {
    fp->stop();
    FAIL() << "Expected FusionProfiler::stop to assert because state is not Running! " << fp->state();
  } catch (const std::exception& e) {
    SUCCEED();
  }
  
  try {
    fp->profile();
    FAIL() << "Expected FusionProfiler::profile to assert because state is not Processed! " << fp->state();
  } catch (const std::exception& e) {
    SUCCEED();
  }

  fp->start();
  
  // Make error checks for state when it is Running and methods expect
  // something else.
  
  try {
    fp->profile();
    FAIL() << "Expected FusionProfiler::profile to assert because state is not Processed! " << fp->state();
  } catch (const std::exception& e) {
    SUCCEED();
  }
}

} // namespace nvfuser
