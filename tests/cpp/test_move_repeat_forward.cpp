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

#include <fusion.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using MoveRepeatForwardTest = NVFuserTest;

TEST_F(MoveRepeatForwardTest, Simple) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8, 128};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {2, 1});
  auto tv2 = sin(tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
  Fusion* scheduled_fusion = runtime->executors()
                                 .at(0)
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();

  for (auto e : scheduled_fusion->exprs()) {
    // The sin op should operate on a tensor of the pre-repeat size
    if (auto uop = dynamic_cast<UnaryOp*>(e);
        uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Sin) {
      auto repeated_id = uop->out()->as<TensorView>()->getLogicalDomain().at(1);
      EXPECT_EQ(repeated_id->extent()->evaluate().as<int64_t>(), 128);
    }
  }
}

TEST_F(MoveRepeatForwardTest, MoveOverSlice) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8, 128};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {2, 1});
  auto tv2 = slice(
      tv1,
      {{fusion.zeroVal(), tv1->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(64)}});
  auto tv3 = sin(tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::Resize);
  Fusion* scheduled_fusion = runtime->executors()
                                 .at(0)
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();

  for (auto e : scheduled_fusion->exprs()) {
    // The sin op should operate on a tensor of the pre-repeat size
    if (auto uop = dynamic_cast<UnaryOp*>(e);
        uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Sin) {
      auto repeated_id = uop->out()->as<TensorView>()->getLogicalDomain().at(0);
      EXPECT_EQ(repeated_id->extent()->evaluate().as<int64_t>(), shape1.at(0));
    }
  }
}

// Disabled for now due to an unrelated bug. Will be fixed in a
// follow-up PR.
TEST_F(MoveRepeatForwardTest, DISABLED_ConflictingSlice) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8, 128};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {1, 2});
  auto tv2 = sin(tv1);
  auto tv3 = slice(
      tv2,
      {{fusion.zeroVal(), tv2->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(64)}});
  auto tv4 = cos(tv3);
  fusion.addOutput(tv4);

  // Should be moved past sin but not slice as it resizes the repeated ID

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::Resize);

  auto exprs = runtime->executors()
                   .at(0)
                   ->as<KernelExecutor>()
                   ->compiledKernel()
                   ->kernel()
                   ->exprs();
  // The sin should operate on the pre-repeat size
  auto sin_it = std::ranges::find_if(exprs, [](Expr* e) {
    auto uop = dynamic_cast<UnaryOp*>(e);
    return uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Sin;
  });
  ASSERT_NE(sin_it, exprs.end());
  auto sin_repeated_id =
      (*sin_it)->input(0)->as<TensorView>()->getLogicalDomain().at(1);
  EXPECT_EQ(sin_repeated_id->extent()->evaluate().as<int64_t>(), shape1.at(1));

  // The slice should operate on the post-repeat size
  auto slice_it =
      std::ranges::find_if(exprs, [](Expr* e) { return e->isA<SliceOp>(); });
  ASSERT_NE(slice_it, exprs.end());
  EXPECT_EQ(
      (*slice_it)
          ->input(0)
          ->as<TensorView>()
          ->getLogicalDomain()
          .at(1)
          ->extent()
          ->evaluate()
          .as<int64_t>(),
      shape1.at(1) * 2);

  // The cos should follow the slice
  auto cos_it = std::ranges::find_if(exprs, [](Expr* e) {
    auto uop = dynamic_cast<UnaryOp*>(e);
    return uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Cos;
  });
  ASSERT_NE(cos_it, exprs.end());
  EXPECT_EQ((*cos_it)->input(0)->definition(), *slice_it);
}

TEST_F(MoveRepeatForwardTest, MoveOverRotation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8, 128};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {2, 1});

  // Rotation pattern
  auto tv2 = slice(
      tv1,
      {{fusion.zeroVal(), tv1->getLogicalDomain().at(0)->extent()},
       {fusion.zeroVal(), IrBuilder::create<Val>(64)}});
  auto tv3 = slice(
      tv1,
      {{fusion.zeroVal(), tv1->getLogicalDomain().at(0)->extent()},
       {IrBuilder::create<Val>(64), tv1->getLogicalDomain().at(1)->extent()}});
  auto tv4 = cat({tv3, tv2}, 1);
  auto tv5 = sin(tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::Resize);
  Fusion* scheduled_fusion = runtime->executors()
                                 .at(0)
                                 ->as<KernelExecutor>()
                                 ->compiledKernel()
                                 ->kernel();

  for (auto e : scheduled_fusion->exprs()) {
    // The sin op should operate on a tensor of the pre-repeat size
    if (auto uop = dynamic_cast<UnaryOp*>(e);
        uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Sin) {
      auto repeated_id = uop->out()->as<TensorView>()->getLogicalDomain().at(0);
      EXPECT_EQ(repeated_id->extent()->evaluate().as<int64_t>(), shape1.at(0));
    }
  }
}

// Repeat should be moved across exprs even when other non-repeated
// operands are not repeated as long as their corresponding IDs are
// broadcast.
TEST_F(MoveRepeatForwardTest, MoveRepeatWithNonRepeatedInputs) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  std::vector<int64_t> shape1{8};
  std::vector<int64_t> shape2{1};
  std::vector<int64_t> shape3{16};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape2);
  fusion.addInput(tv1);
  auto tv2 = makeContigConcreteTensor(shape3);
  fusion.addInput(tv2);

  auto tv3 = repeat(tv0, {2});
  auto tv4 = add(tv3, tv1);
  auto tv5 = sin(tv4);
  auto tv6 = add(tv5, tv2);
  auto tv7 = cos(tv6);
  fusion.addOutput(tv7);

  // Should be moved past the sin but not cos since the other operand
  // of the cos has the extent equal to the repeated size

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);
  auto t2 = at::randn(shape3, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
  const auto& heuristic_param =
      runtime->schedulerHeuristics()->heuristicsList().front();
  EXPECT_EQ(heuristic_param->scheduler_type, SchedulerType::PointWise);
  auto exprs = runtime->executors()
                   .at(0)
                   ->as<KernelExecutor>()
                   ->compiledKernel()
                   ->kernel()
                   ->exprs();

  // The sin should operate on the pre-repeat size
  auto sin_it = std::ranges::find_if(exprs, [](Expr* e) {
    auto uop = dynamic_cast<UnaryOp*>(e);
    return uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Sin;
  });
  ASSERT_NE(sin_it, exprs.end());
  auto sin_repeated_id =
      (*sin_it)->input(0)->as<TensorView>()->getLogicalDomain().at(0);
  EXPECT_EQ(sin_repeated_id->extent()->evaluate().as<int64_t>(), shape1.at(0));

  // The cos should operate on the pre-repeat size
  auto cos_it = std::ranges::find_if(exprs, [](Expr* e) {
    auto uop = dynamic_cast<UnaryOp*>(e);
    return uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Cos;
  });
  ASSERT_NE(sin_it, exprs.end());
  auto cos_repeated_id =
      (*cos_it)->input(0)->as<TensorView>()->getLogicalDomain().at(0);
  EXPECT_EQ(
      cos_repeated_id->extent()->evaluate().as<int64_t>(), shape1.at(0) * 2);
}

} // namespace nvfuser
