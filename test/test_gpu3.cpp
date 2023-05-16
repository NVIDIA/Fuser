// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <device_lower/magic_zero.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <grouped_reduction.h>
#include <inlining.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_graphviz.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <mutator.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, FusionNonDivisibleSplit1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  // [I]
  tv1->split(0, 5);
  // [ceilDiv(I, 5), 5]

  // This second split is non-divisible. The split domain must be predicated.
  tv1->split(1, 3);
  // [ceilDiv(I, 5), 2, 3]

  auto tv2 = sum(tv0, {0});
  fusion.addOutput(tv2);

  // tv2 shouldn't need to have another predicate
  tv2->split(0, 4);
  tv2->split(1, 2);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv1 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref, ref}, __LINE__, __FILE__);
}

// Repro of issue #1074
TEST_F(NVFuserTest, FusionNonDivisibleSplit2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  fusion.addOutput(tv2);

  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->reorder({{1, 2}, {2, 1}});
  tv0->computeAt(tv2, 2);

  tv2->split(-1, 3);

  // To make the sanitizer catch the invalid accesses. Not necessary
  // to expose the bug.
  tv1->setMemoryType(MemoryType::Shared);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({13, 17}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Similar to FusionNonDivisibleSplit1 but with unswitch
TEST_F(NVFuserTest, FusionNonDivisibleSplit3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::Unswitch);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Non-divisible split through merge
TEST_F(NVFuserTest, FusionNonDivisibleSplit4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0, 1});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->merge(1, 2);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Nested splits
TEST_F(NVFuserTest, FusionNonDivisibleSplit5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  // [I]
  tv2->split(0, 8);
  // [I/8, 8]
  tv2->split(1, 2);
  // [I/8, 4, 2]
  tv2->split(1, 3); // non-divisible split of outer output
  // [I/8, 2, 3, 2]

  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    TORCH_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    TORCH_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({24}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Vectorized non-divisible split. Must be validated at run time
TEST_F(NVFuserTest, FusionNonDivisibleSplitVectorize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->split(0, 8, false);
  tv1->split(1, 4);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    TORCH_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  auto t0_non_divisible = at::randn({8}, options);
  // Since ceilDiv(8, 8) is not divisible by 4, the vectorization is
  // illegal. The run-time validation of vectorization should throw an error.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0_non_divisible}));
}

// If a split is validated at run time, it's not necessary to predicate.
TEST_F(NVFuserTest, FusionNonDivisibleSplitVectorize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1));
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv3->split(0, 8, false);
  tv3->split(1, 4);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, {tv1, tv2});

  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    TORCH_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto t0 = at::randn({1024}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1284Repro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {10, 20};
  std::vector<int64_t> input_shape_1 = {15};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  fusion.addInput(in_0);
  fusion.addInput(in_1);

  TensorView* out_0 = add(in_0, IrBuilder::create<Double>(0.f));
  TensorView* out_1 = add(in_1, IrBuilder::create<Double>(2.f));

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  std::vector<c10::IValue> aten_inputs = {at_in_0, at_in_1};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t1 = at_in_1 + 2;

  auto runtime = fec.getMostRecentKernelRuntime();
  TORCH_INTERNAL_ASSERT(runtime->isSegmented());
  TORCH_INTERNAL_ASSERT(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion, outputs, {at_in_0, at_in_1}, {at_in_0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1284Repro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape_0 = {4, 4};
  std::vector<int64_t> input_shape_1 = {3, 4, 4};
  std::vector<int64_t> input_shape_2 = {2, 8, 4, 4};

  TensorView* in_0 = makeSymbolicTensor(input_shape_0.size());
  TensorView* in_1 = makeSymbolicTensor(input_shape_1.size());
  TensorView* in_2 = makeSymbolicTensor(input_shape_2.size());

  fusion.addInput(in_0);
  fusion.addInput(in_1);
  fusion.addInput(in_2);

  TensorView* out_0 = add(in_0, in_1);
  TensorView* out_1 = add(in_0, in_2);

  fusion.addOutput(out_0);
  fusion.addOutput(out_1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_in_0 = at::randn(input_shape_0, options);
  at::Tensor at_in_1 = at::randn(input_shape_1, options);
  at::Tensor at_in_2 = at::randn(input_shape_2, options);

  std::vector<c10::IValue> aten_inputs = {at_in_0, at_in_1, at_in_2};

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto outputs = fec.runFusionWithInputs(aten_inputs);

  auto t0 = at_in_0 + at_in_1;
  auto t1 = at_in_0 + at_in_2;

  auto runtime = fec.getMostRecentKernelRuntime();
  TORCH_INTERNAL_ASSERT(runtime->isSegmented());
  TORCH_INTERNAL_ASSERT(runtime->fusionSegments()->groups().size() == 2);

  testValidate(
      &fusion,
      outputs,
      {at_in_0, at_in_1, at_in_2},
      {t0, t1},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1305Repro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto t0 = makeContigTensor(1);
  auto t1 = makeContigTensor(2);

  fusion.addInput(t0);
  fusion.addInput(t1);

  auto t2 = broadcast(t0, {true, false});
  auto t3 = add(t1, t2);
  auto t4 = add(t3, t2);
  auto t5 = sum(t4, {1});
  auto t6 = broadcast(t5, {false, true});
  auto t7 = add(t3, t6);

  fusion.addOutput(t7);

  t3->computeAt(t7, -1, ComputeAtMode::MostInlined);

  TORCH_INTERNAL_ASSERT(t3->getComputeAtPosition() == 1);
}

TEST_F(NVFuserTest, FusionDoubleBuffering1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDoubleBuffering2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  auto tv3 = set(tv2);
  fusion.addOutput(tv3);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, -1);

  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDoubleBuffering3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);

  // tv2 is invalid to double-buffer as its producer, tv1, is
  // computed inside the double-buffering loop.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv2->doubleBuffer());

  // Moving tv2 inner makes tv1 large enough to double-buffer tv2
  tv2->computeAt(tv3, 2);

  tv2->doubleBuffer();

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unswitch
TEST_F(NVFuserTest, FusionDoubleBuffering4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 32);
  tv3->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 2);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv3);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering gmem to shared and unswitch
TEST_F(NVFuserTest, FusionDoubleBuffering5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 128);
  tv2->split(-1, 32);
  tv2->split(-1, 8);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  scheduler_utils::parallelizeAllLike(tv2);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1000}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering smem to local and unroll
TEST_F(NVFuserTest, FusionDoubleBuffering6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Shared);

  tv3->split(-1, 128);
  tv3->split(-1, 16);
  tv3->split(-2, 4);
  tv3->split(-2, 2);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv0->computeAt(tv3, 1);
  tv2->computeAt(tv3, -1);

  tv3->axis(2)->parallelize(ParallelType::Unroll);
  tv3->axis(4)->parallelize(ParallelType::TIDx);

  tv2->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({199}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Double buffering and vectorize
TEST_F(NVFuserTest, FusionDoubleBuffering7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Double>(1.0));
  fusion.addOutput(tv2);

  tv2->split(-1, 128);
  tv2->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv1->computeAt(tv2, 2);

  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({200}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Multiple tensors to double-buffer
TEST_F(NVFuserTest, FusionDoubleBuffering8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 32);
  tv4->split(0, 4);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({100}, options);
  auto t1 = at::randn({100}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Nested double buffering from gmem to smem and smem to register
TEST_F(NVFuserTest, FusionDoubleBuffering9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto out = tv1;
  fusion.addOutput(out);

  auto tv2 = tv0->cacheAfter();
  auto tv3 = tv2->cacheAfter();

  out->split(0, 32);
  out->split(0, 4);
  TransformPropagatorWithCheck propagator(out);
  MaxRootDomainInfoSpanningTree(out).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Shared);

  tv2->computeAt(out, 1);
  tv3->computeAt(out, -1);

  out->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(out);

  tv2->doubleBuffer();
  tv3->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1001}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0 + 1;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// FusionSmemBlockGemmCache + double buffering at both smem and local
TEST_F(NVFuserTest, FusionSmemBlockGemmCacheDoubleBuffer_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(2); // (M, K)
  TensorView* tv1 = makeSymbolicTensor(2); // (K, N)
  TensorView* tv2 = broadcast(tv0, {false, false, true}); // (M, K, B)
  TensorView* tv3 = broadcast(tv1, {true, false, false}); // (B, K, N)
  TensorView* tv4 = mul(tv2, tv3); // M, K, N
  TensorView* tv5 = sum(tv4, {1}); // M, R, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  TensorView* tv6 = tv5->cacheBefore();

  // For smem double buffering
  auto tv0_cache_local = tv0->cacheAfter();
  auto tv1_cache_local = tv1->cacheAfter();

  // For register double buffering
  auto tv0_cache_smem = tv0->cacheAfter();
  auto tv1_cache_smem = tv1->cacheAfter();

  const int BSX = 32;
  const int TSX = 8;

  // [M, K, N]
  tv6->split(-1, BSX);
  tv6->split(-1, TSX);
  tv6->split(1, BSX);
  tv6->split(0, BSX);
  tv6->split(1, TSX);
  // [M/BSX, BSX/TSX, TSX, K/BSX, BSX, N/BSX, BSX/TSX, TSX]
  tv6->reorder(
      {{4, 7}, {7, 6}, {6, 5}, {2, 4}, {1, 3}, {3, 2}, {5, 1}, {0, 0}});
  // [M/BSX, N/BSX, K/BSX, BSX/TSX, BSX/TSX, TSX, TSX, BSX]

  auto tv6_rf = tv6->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv6_rf);
  MaxRootDomainInfoSpanningTree(tv6_rf).traverse(&propagator);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);

  tv6_rf->computeAt(tv6, -1);
  tv0_cache_local->computeAt(tv6_rf, -1);
  tv1_cache_local->computeAt(tv6_rf, -1);

  tv0_cache_smem->setMemoryType(MemoryType::Shared);
  tv1_cache_smem->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(-3)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv5);

  tv0_cache_local->doubleBuffer();
  tv1_cache_local->doubleBuffer();

  tv0_cache_smem->doubleBuffer();
  tv1_cache_smem->doubleBuffer();

  constexpr int M = 154, K = 45, N = 1524;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);
  at::Tensor aten_output = matmul(t0.to(at::kDouble), t1.to(at::kDouble));

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
  // The smem cache write in this test case is redundant predicated,
  //   and also double buffered. Currently we are relying on WAR sync
  //   insertion to ensure ordering of double buffered tensor access.
  // The check below makes sure that the sync is inserted so that the
  //   test isn't running on a race condition.
  TORCH_CHECK(fe.kernel()->summary().war_hazard_syncs_count > 0);
}

TEST_F(NVFuserTest, FusionIntermediateTensorVectorize_CUDA) {
  GTEST_SKIP();
  std::vector<MemoryType> mem_types = {MemoryType::Shared, MemoryType::Local};

  for (auto mem_type : mem_types) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigTensor(1);
    fusion.addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    auto tv3 = set(tv2);
    fusion.addOutput(tv3);

    tv1->setMemoryType(mem_type);

    tv3->split(-1, 4);
    TransformPropagatorWithCheck propagator(tv3);
    MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

    tv1->computeAt(tv3, -2);

    tv2->axis(-1)->parallelize(ParallelType::Vectorize);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::manual_seed(0);
    auto t0 = at::randn({15}, options);
    FusionExecutor fe;
    fe.compileFusion(&fusion);

    // This should throw an exception as the extent of t0 is not
    // divisible by the vector width
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(fe.runFusion({t0}));

    auto t1 = at::randn({16}, options);
    auto cg_outputs = fe.runFusion({t1});

    testValidate(&fusion, cg_outputs, {t1}, {t1}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionBroadcastConcretization1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10, 1});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({10, 20});
  fusion.addInput(tv1);
  auto tv2 = makeConcreteTensor({10, 10});
  fusion.addInput(tv2);

  // Not concretized
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv0, tv4);
  fusion.addOutput(tv5);

  // Concretized
  auto tv6 = sum(tv2, {1});
  auto tv7 = broadcast(tv6, {false, true});
  auto tv8 = add(tv1, tv7);
  fusion.addOutput(tv8);

  for (auto tv : {tv3, tv4, tv5, tv6, tv7, tv8}) {
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  GpuLower gpulw(&fusion);
  TORCH_CHECK(!gpulw.concretizedBroadcastDomains()->isConcretized(
      loweredTv(tv4, gpulw)->axis(1)));
  TORCH_CHECK(gpulw.concretizedBroadcastDomains()->isConcretized(
      loweredTv(tv7, gpulw)->axis(1)));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({10, 1}, options);
  auto t1 = at::randn({10, 20}, options);
  auto t2 = at::randn({10, 10}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t5 = t0 + t2.sum({1}).unsqueeze(-1);
  auto t8 = t1 + t2.sum({1}).unsqueeze(-1);

  testValidate(&fusion, outputs, aten_inputs, {t5, t8}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastConcretization2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0, 1});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = broadcast(tv2, {false, true});
  fusion.addOutput(tv3);

  // tv1 is thread-predicated with TIDx and TIDy
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);
  // tv2 broadcasts along TIDx
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  // tv3 broadcasts along TIDy
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);

  // Both tv2 and tv3 broadcast along predicated TID dimensions, but
  // since the broadcast domains are not concretized, there should be
  // no actual parallel broadcast

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({10, 11}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t3 = t0.sum().unsqueeze(-1).unsqueeze(-1);

  testValidate(&fusion, outputs, aten_inputs, {t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastConcretization3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape({10, 4, 8});
  std::vector<int64_t> output_shape({8, 4, 1});

  auto tv0 = makeConcreteTensor(input_shape);
  fusion.addInput(tv0);

  auto tv2 = sum(tv0, {0});
  auto tv3 = set(tv2);
  auto tv4 =
      reshape(tv3, {input_shape.begin() + 1, input_shape.end()}, output_shape);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  // The reshape op adds a broadcast domain in tv4, which is
  // parallelized. Howver, it is never materialized, so there should
  // be no parallel broadcast.

  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  auto t5 = at::native::view(t0.sum(0), output_shape) + 1;

  testValidate(&fusion, outputs, aten_inputs, {t5}, __LINE__, __FILE__);
}

// Merging non-broadcast and broadcast domains
// TODO: Fix use case see issue https://github.com/csarofeen/pytorch/issues/1418
// validateParallelize does not pass. Even if it's skipped,
// generated code is invalid as blockBroadcast is not used.
#if 0
TEST_F(NVFuserTest, FusionBroadcastConcretization4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  fusion.addOutput(tv3);

  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->merge(0, 1);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  // TODO: When set to shared memory, this kernel should be correct, but fails
  // validation and when skipped produces incorrect code
  tv2->setMemoryType(MemoryType::Shared);

  tv3->merge(0, 1);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  fusion.printMath();
  fusion.printKernel();
}
#endif

TEST_F(NVFuserTest, FusionBroadcastConcretization5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = makeSymbolicTensor(1);
  fusion.addInput(tv3);

  // Assert tv2 and tv3 have the same shape
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Concretize a broadcast domain to multiple non-concrete domains
  // through a multi-output expression. It should be considered to be
  // non-uniquely concretized.
  auto tv5 = broadcast(tv0, {false, true});
  // Reduce only the non-broadcast domain.
  auto tvs = Welford(tv5, {0});
  auto tv9 = add(tvs.avg, tv1);
  auto tv10 = add(tvs.var_sum, tv2);
  fusion.addOutput(tv9);
  fusion.addOutput(tv10);

  // Same pattern as the above, but concretize the broadcast domain
  // with tv2 and tv3, which have the exactly same shape, so the
  // broadcast should be considered uniquely concretized.
  auto tv11 = broadcast(tv0, {false, true});
  // Reduce only the non-broadcast domain.
  auto tvs2 = Welford(tv11, {0});
  auto tv15 = add(tvs2.avg, tv2);
  auto tv16 = add(tvs2.var_sum, tv3);
  fusion.addOutput(tv15);
  fusion.addOutput(tv16);

  // Reduce only the broadcast domain. Since it's reduced, it should
  // not be considered to be concretized.
  auto tv17 = broadcast(tv0, {false, true});
  auto tvs3 = Welford(tv17, {1});
  fusion.addOutput(tvs3.avg);

  ConcretizedBroadcastDomains bcast_concretization_info(&fusion);

  TORCH_CHECK(
      bcast_concretization_info.maybeNonUniquelyConcretized(tv5->axis(1)),
      "Failed to detect non-unique concretization of ",
      tv5->toString());

  TORCH_CHECK(
      bcast_concretization_info.isUniquelyConcretized(tv11->axis(1)),
      "Failed to detect unique concretization of ",
      tv11->toString());

  TORCH_CHECK(
      !bcast_concretization_info.isConcretized(tv17->axis(1)),
      "Failed to detect non-concretization of ",
      tv17->toString());
}

TEST_F(NVFuserTest, FusionIssue1430_CUDA) {
  // Derived from an expression sorting issue when using loop map, now expr
  // sorting uses parallel map.
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int V = 2, W = 3, X = 4, Y = 5, Z = 6;

  // setup fusion
  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .dtype(DataType::Half)
                 .contiguity(true)
                 .shape({V, W, X, Y, Z})
                 .build();

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = castOp(DataType::Float, tv1);

  auto tvs = Welford(tv2, {1, 2, 3, 4});
  auto tv3 = tvs.avg;
  auto tv4 = tvs.var_sum;

  // avg
  auto tv6 = broadcast(tvs.avg, {false, true, true, true, true});

  // var
  auto tv7 = mul(tv4, IrBuilder::create<Double>(1. / (W * X * Y * Z)));
  auto tv8 = add(tv7, IrBuilder::create<Double>(1.e-6));
  auto tv9 = broadcast(tv8, {false, true, true, true, true});
  auto tv10 = rsqrt(tv9);

  auto tv11 = castOp(DataType::Float, tv1);
  auto tv12 = sub(tv11, tv6);
  auto tv13 = mul(tv12, tv10);

  auto tv14 = set(tv13);
  fusion.addOutput(tv14);

  tv3->axis(0)->parallelize(ParallelType::BIDy);
  tv3->axis(2)->parallelize(ParallelType::BIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(4)->parallelize(ParallelType::Vectorize);

  // tv3->reorder({{1, -2}});

  auto rfactor = ir_utils::rfactorHelper(tv3, {1, 4});

  scheduler_utils::parallelizeAllLike(rfactor);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv != tv1 || tv != tv3) {
      for (auto i : c10::irange(tv->nDims())) {
        if (isParallelTypeVectorize(tv->axis((int)i)->getParallelType())) {
          tv->axis((int)i)->parallelize(ParallelType::Serial);
        }
      }
    }
  }

  tv0->computeAt(tv14, 1);
  tv13->computeAt(tv14, -2);
  tv2->computeAt(tv14, -1, ComputeAtMode::MostInlined);
  tv11->computeAt(tv14, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({V, W, X, Y, Z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  auto cg_outputs = fe.runFusion({t0}, LaunchParams(X, V, -1, Y, -1, -1));

  auto t0_double = t0.to(at::kDouble);

  auto at_mu = at::mean(t0_double, {1, 2, 3, 4})
                   .unsqueeze(-1)
                   .unsqueeze(-1)
                   .unsqueeze(-1)
                   .unsqueeze(-1);
  auto at_var = at::var(t0_double, {1, 2, 3, 4}, false)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .unsqueeze(-1);

  auto at_out = t0_double.sub(at_mu).div(at_var.add(1.e-6).sqrt());

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {at_out},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(X, V, -1, Y, -1, -1));
}

// Test code generation of allocated scalars
TEST_F(NVFuserTest, FusionCodegenAllocatedScalars_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Fusion is just a dummy container in this test, just used for
  // getting a Kernel container
  auto tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  GpuLower gpulw(&fusion);
  auto kernel = gpulw.kernel();

  // Set the kernel as the current fusion
  FusionGuard kg(kernel);

  // Create alocated scalars
  auto ks0 = add(kernel->zeroVal(), kernel->oneVal());
  auto ks0_alloc = IrBuilder::create<kir::Allocate>(
      ks0, MemoryType::Local, kernel->oneVal());

  auto ks1 = add(ks0, kernel->oneVal());
  auto ks1_alloc = IrBuilder::create<kir::Allocate>(
      ks1, MemoryType::Local, kernel->oneVal());

  auto tk0 = kernel->inputs()[0]->as<TensorView>();
  auto tki0 = IrBuilder::create<kir::TensorIndex>(tk0, ks0);
  auto tki1 = IrBuilder::create<kir::TensorIndex>(tk0, ks1);
  auto tk0_expr =
      IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, tki0, tki1);

  // Insert the scalar expression and the allocation of the
  // output directly to the kernel
  auto proxy = kir::KernelInternalProxy(kernel);

  const auto indent = "  ";
  const auto ks0_name = "i" + std::to_string(ks0->name());
  const auto ks1_name = "i" + std::to_string(ks1->name());
  const auto tk0_name = "T" + std::to_string(tk0->name());

  auto& exprs = proxy.topLevelExprs();
  exprs.push_back(tk0_expr);

  // Invalid code gen
  const auto no_alloc_code = codegen::generateCudaKernel(kernel);

  // Without alloc, Int vals are just inlined, resulting in:
  // t0[(0 + 1)] = t0[((0 + 1) + 1)]
  std::stringstream no_alloc_ref;
  no_alloc_ref << "\n"
               << indent << tk0_name << "[(0 + 1)]\n"
               << indent << indent << " = " << tk0_name << "[((0 + 1) + 1)];\n";

  TORCH_CHECK(
      no_alloc_code.find(no_alloc_ref.str()) != std::string::npos,
      "Invalid code generation. Expected:",
      no_alloc_ref.str(),
      "Actual:\n",
      no_alloc_code);

  // Insert proper allocations and definitions
  exprs.insert(std::find(exprs.begin(), exprs.end(), tk0_expr), ks0_alloc);
  exprs.insert(
      std::find(exprs.begin(), exprs.end(), tk0_expr), ks0->definition());
  exprs.insert(std::find(exprs.begin(), exprs.end(), tk0_expr), ks1_alloc);
  exprs.insert(
      std::find(exprs.begin(), exprs.end(), tk0_expr), ks1->definition());

  const auto valid_code = codegen::generateCudaKernel(kernel);

  std::stringstream valid_ref;
  valid_ref << "\n"
            << indent << tk0_name << "[" << ks0_name << "]\n"
            << indent << indent << " = " << tk0_name << "[" << ks1_name
            << "];\n";

  TORCH_CHECK(
      valid_code.find(valid_ref.str()) != std::string::npos,
      "Invalid code generation. Expected:",
      valid_ref.str(),
      "Actual:\n",
      valid_code);
}

TEST_F(NVFuserTest, FusionIndexHoist1_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv1->split(-1, 4);
  tv2->split(-1, 4);
  tv3->merge(0, 1);
  tv3->split(0, 8);
  tv5->merge(0, 1);
  tv5->split(0, 8);
  tv4->computeAt(tv5, -1);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);

  // Use Int32 as the index type to verify Int32 is used as the type
  // of hoisted indices
  GpuLower gpulw(&fusion, {DataType::Int32});
  auto kernel = gpulw.kernel();

  auto is_index_times_ns = [](Val* val, Val* index, std::string name) -> bool {
    auto def = dynamic_cast<BinaryOp*>(val->definition());
    if (def == nullptr) {
      return false;
    }
    return def->getBinaryOpType() == BinaryOpType::Mul &&
        def->rhs()->isA<NamedScalar>() &&
        def->rhs()->as<NamedScalar>()->name() == name && def->lhs() == index;
  };

  // Validate indices in the kernel are hoisted as
  // intended. Validation could be also done by just string comparison
  // as the parser test, but updating such tests would be tedious.
  for (auto top_level_loop :
       ir_utils::filterByType<kir::ForLoop>(kernel->topLevelExprs())) {
    auto innermost_loop = top_level_loop;
    while (auto first_expr_loop = dynamic_cast<kir::ForLoop*>(
               innermost_loop->body().exprs().at(0))) {
      innermost_loop = first_expr_loop;
    }
    const auto& exprs = innermost_loop->body().exprs();
    TORCH_CHECK(!exprs.empty(), "No expression found");
    TORCH_CHECK(
        exprs.at(0)->isA<kir::Allocate>(),
        "Invalid expression: ",
        exprs.at(0)->toString());
    auto hoisted_index = exprs.at(0)->as<kir::Allocate>()->buffer();
    TORCH_CHECK(
        hoisted_index->dtype() == DataType::Int32,
        "Invalid data type of hoisted indices. Should be Int32 but: ",
        hoisted_index->dtype());
    kir::Predicate* pred = nullptr;
    for (auto expr : exprs) {
      if (expr->isA<kir::IfThenElse>()) {
        pred = expr->as<kir::IfThenElse>()->predicate();
        auto arith_expr = expr->as<kir::IfThenElse>()->thenBody().exprs().at(0);
        auto out_ti = arith_expr->outputs()[0]->as<kir::TensorIndex>();
        if (out_ti->view()->name() == 1) {
          // Ref: T1[*, hoisted_index] = T0[*, hoisted_index * T0.stride];
          auto t1_index =
              out_ti->index()->definition()->as<BinaryOp>()->input(1);
          TORCH_CHECK(
              t1_index == hoisted_index,
              "Invalid index: ",
              t1_index->toInlineString());
          // Pred: hoisted_index < T0.size[1]
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 0);
          // hoisted_index * T0.stride[1]
          auto t0_index = in0->index()->definition()->as<BinaryOp>()->input(1);
          TORCH_CHECK(
              is_index_times_ns(t0_index, hoisted_index, "T0.stride[1]"),
              "Invalid index: ",
              t0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 2) {
          // Ref: T3[*, hoisted_index] = T2[*, hoisted_index];
          auto out_index =
              out_ti->index()->definition()->as<BinaryOp>()->input(1);
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 1);
          auto in0_index = in0->index()->definition()->as<BinaryOp>()->input(1);
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 3) {
          // Ref: T3[hoisted_index] = T2[hoisted_index];
          auto out_index = out_ti->index();
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 2);
          auto in0_index = in0->index();
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 4) {
          // Ref: T4[0] = T3[hoisted_index];
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(arith_expr->inputs().size() == 1);
          auto in0 = arith_expr->inputs().front()->as<kir::TensorIndex>();
          TORCH_CHECK(in0->view()->name() == 3);
          auto in0_index = in0->index();
          TORCH_CHECK(
              in0_index == hoisted_index,
              "Invalid index: ",
              in0_index->toInlineString(),
              ", ",
              expr->toString());
        } else if (out_ti->view()->name() == 5) {
          // Ref: T5[hoisted_index] = T4[0]
          auto out_index = out_ti->index();
          TORCH_CHECK(
              out_index == hoisted_index,
              "Invalid index: ",
              out_index->toInlineString(),
              ", ",
              expr->toString());
          TORCH_CHECK(
              pred->value()->definition()->as<BinaryOp>()->lhs() ==
                  hoisted_index,
              "Invalid predicate: ",
              pred->value()->toInlineString(),
              ", ",
              expr->toString());
        }
      }
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({15, 17}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Hoist indices for vectorized tensors
TEST_F(NVFuserTest, FusionIndexHoist2_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv5->split(-1, 4);
  TransformPropagatorWithCheck propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  tv4->split(-1, 3);

  tv0->computeAt(tv5, 1);
  tv1->computeAt(tv5, 1);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv5->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({16}, options);
  auto t1 = at::randn({16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexHoist3_CUDA) {
  if (isOptionDisabled(DisableOption::IndexHoist)) {
    GTEST_SKIP() << "Index hoisting disabled";
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto input = makeContigTensor(2);
  fusion->addInput(input);
  auto sin_input = sin(input);
  auto numel = mul(input->axis(0)->extent(), input->axis(1)->extent());
  auto output = add(sin_input, numel);
  fusion->addOutput(output);

  for (auto tv : {output, sin_input}) {
    tv->merge(0);
    tv->split(0, 256);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
  inlineMost();

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(10000, options).view({100, 100});
  at::Tensor t1 = t0.sin() + 10000;

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});

  const std::string expected_kernel = R"(
__global__ void CUDAGeneratedKernel(Tensor<float, 2, 2> T0, Tensor<float, 2, 2> T2) {
  int64_t i75;
  i75 = ((nvfuser_index_t)threadIdx.x) + (256 * ((nvfuser_index_t)blockIdx.x));
  int64_t i7;
  i7 = T0.size[0] * T0.size[1];
  bool b149;
  b149 = i75 < i7;
  float f8;
  f8 = (float)(i7);
  float T1[1];
  if (b149) {
    T1[0]
       = sinf(T0[i75]);
  }
  if (b149) {
    T2[i75]
      = T1[0]
      + f8;
  }
}
)";

  assertCUDAKernel(fusion.get(), expected_kernel);

  testValidate(fusion.get(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTestGridComm_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  int X = 3, Y = 4, Z = 2;
  auto tv0 = makeContigConcreteTensor({X, Y, Z});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({X, Y, Z});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = add(tv2, tv1);
  auto tv4 = set(tv3);
  auto tv5 = set(tv4);
  fusion.addOutput(tv5);

  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv4->setMemoryType(MemoryType::Global);

  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::BIDx);
  tv2->axis(2)->parallelize(ParallelType::Vectorize);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::BIDy);

  tv4->axis(0)->parallelize(ParallelType::BIDy);
  tv4->axis(1)->parallelize(ParallelType::BIDx);

  tv5->axis(0)->parallelize(ParallelType::BIDy);
  tv5->axis(1)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({X, Y, Z}, options);
  auto t1 = at::randn({X, Y, Z}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue https://github.com/csarofeen/pytorch/issues/1497
TEST_F(NVFuserTest, FusionTestGridComm2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t W = 3, X = 4;

  auto tv0 = makeConcreteTensor({X});
  auto tv1 = makeConcreteTensor({W, X});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Double>(1));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv3->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({X}, options);
  auto t1 = at::randn({W, X}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1 + 1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Vectorized reset test for double buffered registers
TEST_F(NVFuserTest, FusionDoubleBufferVector_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = sum(tv1, {0});
  auto tv2c = tv2->cacheBefore();

  fusion.addOutput(tv2);

  auto tv1cw = tv1->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter();

  tv1cw->split(-1, 32);
  tv1cr->split(-1, 32);
  tv1cr->split(-1, 4);
  tv1cr->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1cw->computeAt(tv1cr, 1);
  tv0->computeAt(tv1cw, -1);
  tv2c->split(-1, 32);
  tv2c->split(-1, 4);
  tv1cr->computeAt(tv2c, 2);

  tv1cw->setMemoryType(MemoryType::Shared);
  tv1cr->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({200}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto ref = (t0 + 1).sum({0});

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Request 48KB of data in shared mem,
//  should be large enough not to fit in
//  static allocations, but small enough
//  to fit in supported devices (sm70+).
TEST_F(NVFuserTest, FusionLargeSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  fusion.addOutput(tv2);

  tv2->split(0, 12288);
  tv2->split(1, 128);
  tv1->computeAt(tv2, 1);
  tv1->split(1, 128);
  tv0->computeAt(tv1, -1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({(int)(12288 * 4)}, options);
  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto ref = t0 + 1 + 2;

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Request a smem allocation that is equal to the device limit
TEST_F(NVFuserTest, FusionTooLargeSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int device_limit = (int)properties->sharedMemPerBlockOptin;

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Double>(2.0));
  fusion.addOutput(tv2);

  // 4 byte per float
  tv2->split(0, device_limit / 4);
  tv2->split(1, 128);
  tv1->computeAt(tv2, 1);
  tv1->split(1, 128);
  tv0->computeAt(tv1, -1);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({(int)(12288 * 4)}, options);
  FusionExecutor fe;

  //  NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0}));
}

// Try to test alignment when multiple tensors are
//  in shared mem.
TEST_F(NVFuserTest, FusionSmemAlignment_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({3, 4, 7, 2, 5});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {4});
  auto tv2 = sum(tv1, {3});
  auto tv3 = sum(tv2, {2});
  auto tv4 = sum(tv3, {1});
  fusion.addOutput(tv4);

  auto tv0c = tv0->cacheAfter();
  auto tv1bc = tv1->cacheBefore();
  auto tv2bc = tv2->cacheBefore();
  auto tv3bc = tv3->cacheBefore();
  auto tv4bc = tv4->cacheBefore();

  tv0c->setMemoryType(MemoryType::Shared);
  tv1bc->setMemoryType(MemoryType::Shared);
  tv2bc->setMemoryType(MemoryType::Shared);
  tv3bc->setMemoryType(MemoryType::Shared);
  tv4bc->setMemoryType(MemoryType::Shared);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0->computeAt(tv4, 0);
  tv0->computeAt(tv2, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);
  auto t0 = at::randn({3, 4, 7, 2, 5}, options);
  FusionExecutor fe;

  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto tref = t0.sum({1, 2, 3, 4});

  testValidate(&fusion, cg_outputs, {t0}, {tref}, __LINE__, __FILE__);
}

// Repro of #1521
TEST_F(NVFuserTest, FusionImmediateValueAsInput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto immediate_scalr = IrBuilder::create<Double>(0.1);
  // Adding an immediate scalar value as an input is not allowed
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.addInput(immediate_scalr));

  // Instead, use a symbolic value
  auto symbolic_scalar = IrBuilder::create<Double>();
  fusion.addInput(symbolic_scalar);

  auto tv1 = add(tv0, symbolic_scalar);
  fusion.addOutput(tv1);

  // Make sure the kernel is compiled.
  FusionExecutor fe;
  fe.compileFusion(&fusion);
}

// Repro of #1506
TEST_F(NVFuserTest, FusionVectorizeContigIndex_CUDA) {
  std::vector<int64_t> shape{14, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->merge(0);

  // Vectorize by 4 should be allowed
  tv2->split(0, 4);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv2, 1);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

// Make sure the same fusion as FusionVectorizeContigIndex fails if
// not contig.
TEST_F(NVFuserTest, FusionVectorizeContigIndexFail_CUDA) {
  GTEST_SKIP();
  std::vector<int64_t> shape{14, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = TensorViewBuilder().contiguity({false, true}).ndims(2).build();
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->merge(0);

  tv2->split(0, 4);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv2, 1);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  // This should fail at compile time as we're trying to merge in a
  // non-contiguous dimension, then split and vectorize it.
  ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0}));
}

// Make sure the same fusion as FusionVectorizeContigIndex fails if
// not a correct multiple
TEST_F(NVFuserTest, FusionVectorizeContigIndexFail2_CUDA) {
  GTEST_SKIP();
  std::vector<int64_t> shape{15, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->merge(0);

  tv2->split(0, 4);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv2, 1);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // This should fail at the launch time as 14 is not divisible by the
  // vector word size. The two domains are merged, but they are not
  // contiguous, so contig indexing is not involved in this case.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}));
}

TEST_F(NVFuserTest, FusionVectorizeInputToOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  const int n = 12;
  auto t0 = at::randn({n}, options);
  // Shift by one to make it non-aligned
  auto t0_misaligned =
      at::randn({n + 1}, options).index({at::indexing::Slice(1)});
  auto t1_misaligned =
      at::empty({n + 1}, options).index({at::indexing::Slice(1)});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  TORCH_CHECK(t0.equal(cg_outputs[0]));

  // Pass misaligned input. This must fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0_misaligned}));

  // Pass misaligned output. This must fail too.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}, {t1_misaligned}));
}

// Repro of issue #1530
TEST_F(NVFuserTest, FusionVectorizeContigIndexValidationFail_CUDA) {
  GTEST_SKIP();
  std::vector<int64_t> shape{1, 2, 1};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(shape.size());
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(1);
  tv1->merge(0);

  auto invalid_vec_size = shape[0] * shape[1] * shape[2];
  invalid_vec_size *= invalid_vec_size;

  tv1->split(0, invalid_vec_size);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0}));
}

TEST_F(NVFuserTest, FusionContigIndexingWithBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({4});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({3, 4});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->merge(0);
  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Local);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4}, options);
  auto t1 = at::randn({3, 4}, options);

  auto t3 = t0.unsqueeze(0).add(t1);
  {
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0, t1});
    auto cg_outputs = fe.runFusion({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, {t3}, __LINE__, __FILE__);
  }

  // Make sure tv2 indexing also works when it's stored in global memory
  tv2->setMemoryType(MemoryType::Global);
  {
    FusionExecutor fe;
    fe.compileFusion(&fusion, {t0, t1});
    auto cg_outputs = fe.runFusion({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, {t3}, __LINE__, __FILE__);
  }
}

// TODO: Fix validation
// Repro of #1534. Validation should detect invalid vectorization.
TEST_F(NVFuserTest, FusionVectorizeContigIndexValidationFail2_CUDA) {
  GTEST_SKIP();
  std::vector<int64_t> shape1{2, 3, 2};
  std::vector<int64_t> shape2{2, 2};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor(shape2);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  tv4->merge(1, 2);
  tv4->merge(0, 1);
  tv4->split(0, 4);
  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  // Vectorization of tv2 should be detected as invalid.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fe.runFusion({t0, t1}));
}

TEST_F(NVFuserTest, FusionVectorizeContigIndexWithBroadcast_CUDA) {
  std::vector<int64_t> shape1{2, 2, 2};
  std::vector<int64_t> shape2{1, 2, 2};

  Fusion fusion;
  FusionGuard fg(&fusion);

  // [I0, I1, I2]
  auto tv0 = makeContigTensor(shape1.size());
  fusion.addInput(tv0);

  // [B3, I1, I2]
  auto tv1 = makeContigConcreteTensor(shape2);
  fusion.addInput(tv1);

  auto tv2 = set(tv1);
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  tv3->merge(1, 2);
  tv3->merge(0, 1);
  tv3->split(0, 4);

  // Don't modify tv1 so that it's replayed as tv2 with actual
  // transformations. It would create temporary IterDomains, and the
  // validation should still be able to detect vectorization by 4 is valid.
  // TransformPropagatorWithCheck propagator(tv3);
  // MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->merge(1, 2);
  tv2->merge(0, 1);
  tv2->split(0, 4);

  tv2->computeAt(tv3, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionVectorizeContigIndexPointwiseSchedule_CUDA) {
  std::vector<int64_t> shape0{100, 14, 2, 14};
  std::vector<int64_t> shape1{100, 2, 14};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(shape0.size());
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(shape1.size());
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true, false, false});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape0, options);
  auto t1 = at::randn(shape1, options);

  auto lparams = schedulePointwise(&fusion, {t0, t1});

  GpuLower gpulw(&fusion);
  auto kernel = gpulw.kernel();

  // The innermost two dimensions are merged and contiguous, so
  // vectorization can be done against 2*14=28 rather than 14, so
  // vector word size should be 4. Broadcasting of tv1 should not
  // matter.
  for (const auto& vec_info : kernel->summary().vectorized_set_info) {
    TORCH_CHECK(
        vec_info.word_size == 4,
        "Invalid vector word size: ",
        vec_info.word_size);
  }

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, lparams);
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1.unsqueeze(-3);

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialReductionForwarding4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  // tv4 has a trivial reduction axis
  auto tv4 = sum(tv2, {0});
  auto tv5 = add(tv4, IrBuilder::create<Double>(1));
  fusion.addOutput(tv5);

  tv3->merge(0, 1);
  tv3->split(0, 32);

  // This causes the trivial reduction of tv4 to be merged with
  // another axis of tv4, and then forward computeAt is done from tv4
  // to tv5. The split of the merged id of tv4 should be done on tv5
  // by forwarding the merge of the trivial reduction.
  tv0->computeAt(tv3, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({111}, options);
  auto t1 = at::randn({123, 111}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto t2 = t0.unsqueeze(0);
  auto t3 = t1 + t2;
  auto t5 = sum(t2, {0}) + 1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {t3, t5}, __LINE__, __FILE__);
}

// See issue #1598
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Place tv2 on shared memory
  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->setMemoryType(MemoryType::Shared);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->split(0, 2);
  tv3->split(-1, 4);
  // swap tidx and tidy
  tv3->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDy);

  tv4->split(0, 2);
  tv4->split(-1, 4);
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 1);
  tv3->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 64}, options);
  auto t1 = at::randn({10, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue #1598
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv2->split(0, 2);
  tv2->split(-1, 4);
  tv2->setMemoryType(MemoryType::Shared);

  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv4->split(0, 2);
  tv4->split(-1, 4);
  // Also do unroll for tv3 and tv4
  tv4->split(-2, 8, false);
  tv4->axis(-3)->parallelize(ParallelType::Unroll);
  // swap tidx and tidy
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 1);
  tv3->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 64}, options);
  auto t1 = at::randn({10, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See issue #1599
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  // Use unroll where a RAW-sync tensor is stored

  tv4->split(0, 2);
  tv4->split(0, 3);
  tv4->split(-1, 4);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(-2)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDy);

  tv0->computeAt(tv4, 3);
  tv3->computeAt(tv4, -1);

  tv2->split(-1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({50, 64}, options);
  auto t1 = at::randn({50, 64}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// See #1618
TEST_F(NVFuserTest, FusionRAWSyncInsertionPlace4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({16, 128});
  auto tv1 = makeConcreteTensor({16, 128});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv1);
  auto tv4 = set(tv2);
  auto tv5 = set(tv3);
  auto tv6 = add(tv4, tv5);
  fusion.addOutput(tv6);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv2->computeAt(tv6, 0);
  tv3->computeAt(tv6, 1);
  tv4->computeAt(tv6, 1);
  tv5->computeAt(tv6, -1);
  tv2->split(1, 64);
  tv3->split(1, 64);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv6->axis(-1)->parallelize(ParallelType::TIDx);

  // Check the block sync is inserted at the correct location.
  //  There is exactly one block sync needed in this test case
  //    and the sync needs to be after the 2 expressions
  //    that modify shared memory.
  class SyncInsertionPointChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(LoadStoreOp* uop) final {
      // Record number of load-store ops that modifies shared memory.
      if (uop->out()->isA<kir::TensorIndex>() &&
          uop->out()->as<kir::TensorIndex>()->view()->getMemoryType() ==
              MemoryType::Shared &&
          // Filter out initialization expressions
          uop->in()->isA<kir::TensorIndex>()) {
        number_of_writes_++;
      }
    }
    void handle(kir::BlockSync* bsync) final {
      // Make sure both shared memory modifying expressions
      //  have been observed at the sync insertion point.
      TORCH_INTERNAL_ASSERT(
          number_of_writes_ == 2,
          "FusionRAWSyncInsertionPlace4 test fail:",
          "only 1 sync after the 2 shared mem writes is needed in this test,"
          "either a redundant sync has been inserted or the block sync is not inserted at the right place");
    }

   private:
    int number_of_writes_ = 0;
  } sync_insertion_checker;
  GpuLower gpulw(&fusion);
  sync_insertion_checker.handle(gpulw.kernel()->topLevelExprs());
}

// Test serial write and parallel read of shared mem: mapped case
TEST_F(NVFuserTest, FusionSerialSmemWriteParallelRead1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({128, 6});
  TensorView* tv1 = makeConcreteTensor({128, 6});
  TensorView* tv2 = makeConcreteTensor({128, 6});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = add(tv3, tv2);

  fusion.addOutput(tv4);

  //  Use shared memory
  tv3->setMemoryType(MemoryType::Shared);

  // Parallelize t4, in this case dim 0 on tv3 will
  //  not be parallelized but dim0 of t4 will be.
  // We will need to make sure a sync is inserted
  //  even if these dimensions are mapped.
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({128, 6}, options);
  at::Tensor t1 = at::randn({128, 6}, options);
  at::Tensor t2 = at::randn({128, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t2});
  auto cg_outputs = fe.runFusion({t0, t1, t2});

  auto ref = t0 + t1 + t2;

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, {ref}, __LINE__, __FILE__);
}

// Test serial write and parallel read of shared mem: un-mapped case
TEST_F(NVFuserTest, FusionSerialSmemWriteParallelRead2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({128, 6});
  TensorView* tv1 = makeConcreteTensor({128, 6});
  TensorView* tv2 = makeConcreteTensor({128, 6});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, tv1);
  TensorView* tv4 = add(tv3, tv2);

  fusion.addOutput(tv4);

  //  Use shared memory
  tv3->setMemoryType(MemoryType::Shared);

  // Split and parallelize t4,
  //  the parallelized dimension in t4 will not
  // map across to the shared mem tensor, t3. So
  // there will need to be a sync before use of t3.
  tv4->split(0, 2);
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({128, 6}, options);
  at::Tensor t1 = at::randn({128, 6}, options);
  at::Tensor t2 = at::randn({128, 6}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1, t2});
  auto cg_outputs = fe.runFusion({t0, t1, t2});

  auto ref = t0 + t1 + t2;

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive
TEST_F(NVFuserTest, FusionSimpleCpAsync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int m = 33, n = 31;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv0_shared->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv2, 1);
  tv0_shared->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 1, both block sync and async wait
//  are needed.
TEST_F(NVFuserTest, FusionDoubleBufferCpAsync1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeContigConcreteTensor({m, n});
  TensorView* tv1 = makeContigConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 12);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test of async copy primitive: double buffered
//   Double buffer case 2, only async wait is needed
TEST_F(NVFuserTest, FusionDoubleBufferCpAsync2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  FusionExecutor fe;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0, t1}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Simple test for double buffer in shared mem,
//  where we should not insert redundant syncs when
//  they are not needed.
TEST_F(NVFuserTest, FusionDoubleBufferNoSync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeConcreteTensor({m, n});
  TensorView* tv1 = makeConcreteTensor({m, n});

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);

  fusion.addOutput(tv2);

  auto tv0_shared = tv0->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv2, 1);

  // Asynchronously load a tile in one schedule
  tv0_shared->split(1, 4);
  tv0_shared->axis(-2)->parallelize(ParallelType::TIDx);

  // Consume the loaded tile in another schedule,
  //   triggering the need for a sync.
  tv2->split(1, 4);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);

  // Double buffer the shared mem tensor.
  tv0_shared->doubleBuffer();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.kernel()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  TORCH_INTERNAL_ASSERT(!sync_inserted, "Un-expected block sync inserted");

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test predicate inversion for cp.async
TEST_F(NVFuserTest, FusionCpAsyncPredicate_CUDA) {
  // requires ampere+ GPU

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Using vectorization so need to keep n multiple of 4.
  int m = 33, n = 48;

  TensorView* tv0 = makeContigConcreteTensor({m, n});

  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv0_shared->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv1, 1);

  tv0_shared->split(-1, 32);
  tv0_shared->split(-1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);

  FusionExecutor fe;
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_ANY_THROW(fe.compileFusion(&fusion, {t0}));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.sum({1});

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Test predicate removal on reg-to-reg expressions
TEST_F(NVFuserTest, FusionPredRemovalCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  TensorView* tv4 = set(tv3);

  fusion.addOutput(tv4);
  tv4->split(1, 4);
  tv0->computeAt(tv4, -2);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);

  class PredicateRemovalChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(UnaryOp* uop) final {
      assertOnLocalToLocal(uop);
    }

    // Utility to assert any local-to-local expr is only trivially predicated.
    void assertOnLocalToLocal(Expr* expr) {
      bool is_local = true;
      for (auto in : ir_utils::filterByType<kir::TensorIndex>(expr->inputs())) {
        if (in->view()->getMemoryType() != MemoryType::Local) {
          is_local = false;
        }
      }
      for (auto in :
           ir_utils::filterByType<kir::TensorIndex>(expr->outputs())) {
        if (in->view()->getMemoryType() != MemoryType::Local) {
          is_local = false;
        }
      }

      if (is_local) {
        if (scope_exprs_.empty()) {
          return;
        }
        if (auto ite = dynamic_cast<kir::IfThenElse*>(scope_exprs_.back())) {
          TORCH_INTERNAL_ASSERT(
              ite->predicate()->value()->isConst(),
              "redundant predicate on: ",
              expr);
        }
      }
    }

  } pred_checker;

  GpuLower gpulw(&fusion);
  pred_checker.handle(gpulw.kernel()->topLevelExprs());
}

TEST_F(NVFuserTest, FusionPropagateParallelTypesToSiblings_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tvs = Welford(tv0, {0});
  auto tv_avg = tvs.avg;
  fusion.addOutput(tv_avg);

  tv_avg->split(0, 128);
  TransformPropagatorWithCheck propagator(tv_avg);
  MaxRootDomainInfoSpanningTree(tv_avg).traverse(&propagator);

  tv_avg->axis(0)->parallelize(ParallelType::BIDx);
  tv_avg->axis(1)->parallelize(ParallelType::TIDx);

  // Make sure the parallelization of tv_avg is propagated to the var
  // and count tensors.
  GpuLower gpulw(&fusion);
  for (const auto expr : gpulw.kernel()->exprs()) {
    auto wop = dynamic_cast<WelfordOp*>(expr);
    if (wop == nullptr) {
      continue;
    }
    auto ref = wop->outAvg()->as<TensorView>();
    for (auto sibling : ir_utils::filterByType<TensorView>(wop->outputs())) {
      if (ref == sibling) {
        continue;
      }
      TORCH_CHECK(
          ref->nDims() == sibling->nDims(),
          "Invalid sibling: ",
          sibling->toString());
      for (const auto i : c10::irange(ref->nDims())) {
        TORCH_CHECK(
            ref->axis(i)->getParallelType() ==
                sibling->axis(i)->getParallelType(),
            "Mismatched parallel types between siblings. ",
            ref->toString(),
            ", ",
            sibling->toString());
      }
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({9999}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto outputs = fe.runFusion({t0});

  testValidate(fe.kernel(), outputs, {t0}, {t0.mean({0})}, __LINE__, __FILE__);
}

// Test ExactRootDomainMap
TEST_F(NVFuserTest, FusionExactRootDomainMap_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = transpose(tv2);
  auto tv4 = add(tv2, tv1);
  auto tv5 = add(tv2, tv3);
  auto tv6 = add(tv3, tv1);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  const auto exact_map = ExactRootDomainMap(&fusion);

  // In the exact mapping, the broadcast domain introduced at tv2 is
  // only mapped with the another one in tv3, which is just transposed
  // from tv2. Any other domain, including the second domain of tv4,
  // must not be mapped.

  auto tv2_bc = tv2->axis(1);
  auto tv3_bc = tv3->axis(0);

  TORCH_CHECK(
      exact_map.areMapped(tv2_bc, tv3_bc),
      "Invalid exact root domain map: ",
      exact_map.toString());

  // They must not be mapped with anything else.
  for (auto tv : ir_utils::allTvs(&fusion)) {
    for (auto root_id : tv->getRootDomain()) {
      if (root_id == tv2_bc || root_id == tv3_bc) {
        continue;
      }
      TORCH_CHECK(
          !exact_map.areMapped(root_id, tv2_bc),
          "Invalid exact root domain map: ",
          exact_map.toString());
      TORCH_CHECK(
          !exact_map.areMapped(root_id, tv3_bc),
          "Invalid exact root domain map: ",
          exact_map.toString());
    }
  }
}

class NVFuserMultithreadedTest : public ::testing::Test {
 protected:
  bool was_enabled = false;

  void SetUp() override {
    was_enabled = torch::jit::fuser::cuda::setEnabled(true);
  }

  void TearDown() override {
    torch::jit::fuser::cuda::setEnabled(was_enabled);
  }
};

TEST_F(NVFuserMultithreadedTest, SingleFunction_CUDA) {
  std::string ir = R"IR(
graph(%x.1 : Tensor,
      %y.1 : Tensor):
  %12 : NoneType = prim::Constant()
  %11 : bool = prim::Constant[value=0]()
  %9 : int = prim::Constant[value=1]()
  %3 : Tensor = aten::exp(%x.1)
  %5 : Tensor = aten::relu(%y.1)
  %6 : Tensor = aten::sin(%5)
  %8 : Tensor = aten::add(%3, %6, %9)
  %10 : int[] = prim::ListConstruct(%9)
  %13 : Tensor = aten::sum(%8, %10, %11, %12)
  return (%13)
)IR";
  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::parseIR(ir, g.get());
  torch::jit::GraphFunction fn("nvfuser_test", g, nullptr);

  auto run_kernel = [&fn]() {
    auto x = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    auto y = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    std::vector<c10::IValue> results;
    for (const auto& i : c10::irange(10)) {
      (void)i; // Suppress unused variable warning
      auto stack = createStack({x.clone(), y.clone()});
      fn.run(stack);
      results.push_back(stack.back());
    }
    for (const auto& i : c10::irange(1, 10)) {
      auto t0 = results[0].toTensor();
      auto ti = results[i].toTensor();
      ASSERT_TRUE(at::allclose(t0, ti));
    }
  };

  constexpr size_t kNumThreads = 4;
  std::vector<std::thread> threads;
  for (size_t id = 0; id < kNumThreads; ++id) {
    threads.emplace_back(run_kernel);
  }
  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(NVFuserMultithreadedTest, MultipleFunctions_CUDA) {
  auto run_kernel = []() {
    const std::string ir = R"IR(
  graph(%x.1 : Tensor,
        %y.1 : Tensor):
    %12 : NoneType = prim::Constant()
    %11 : bool = prim::Constant[value=0]()
    %9 : int = prim::Constant[value=1]()
    %3 : Tensor = aten::exp(%x.1)
    %5 : Tensor = aten::relu(%y.1)
    %6 : Tensor = aten::sin(%5)
    %8 : Tensor = aten::add(%3, %6, %9)
    %10 : int[] = prim::ListConstruct(%9)
    %13 : Tensor = aten::sum(%8, %10, %11, %12)
    return (%13)
  )IR";
    auto g = std::make_shared<torch::jit::Graph>();
    torch::jit::parseIR(ir, g.get());
    torch::jit::GraphFunction fn("nvfuser_test", g, nullptr);

    auto x = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    auto y = torch::rand({32, 32}, at::TensorOptions(at::kCUDA));
    std::vector<c10::IValue> results;
    constexpr size_t numRuns = 10;
    for (const auto& i : c10::irange(numRuns)) {
      (void)i; // Suppress unused variable warning
      auto stack = createStack({x.clone(), y.clone()});
      fn.run(stack);
      results.push_back(stack.back());
    }
    for (const auto& i : c10::irange(1, numRuns)) {
      auto t0 = results[0].toTensor();
      auto ti = results[i].toTensor();
      ASSERT_TRUE(at::allclose(t0, ti));
    }
  };

  constexpr size_t kNumThreads = 4;
  std::vector<std::thread> threads;
  for (size_t id = 0; id < kNumThreads; ++id) {
    threads.emplace_back(run_kernel);
  }
  for (auto& t : threads) {
    t.join();
  }
}

// Repro of issue #1655
TEST_F(NVFuserTest, FusionIncompleteConcreteID_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv0, {true, true, false});
  auto tv4 = broadcast(tv1, {false, true, false});
  auto tv5 = broadcast(tv2, {true, false, false});

  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv3, tv5);

  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  tv6->merge(0);
  tv6->merge(0);

  TransformPropagatorWithCheck propagator(tv6);
  MaxRootDomainInfoSpanningTree(tv6).traverse(&propagator);

  tv0->computeAt(tv6, -1, ComputeAtMode::MostInlined);
  tv1->computeAt(tv6, -1, ComputeAtMode::MostInlined);
  tv2->computeAt(tv7, -1, ComputeAtMode::MostInlined);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST_F(NVFuserTest, FusionTestReEntrantGridWelford_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int X = 256, Y = 7, Z = 2048;

  // setup fusion
  auto tv0 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);

  auto tvs = Welford(tv1, {0, 1, 2});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);

  auto cached_input = tv0->cacheAfter();
  tv_avg->cacheBefore();
  tv_M2->cacheBefore();

  auto reduction_tv = scheduler_utils::getReductionTvs(&fusion)[0];

  reduction_tv->merge(0);
  reduction_tv->merge(0);

  int TIDx = 16;
  int vec = 4;

  int TIDy = 16;
  int outer_tidy_fact = 16;

  reduction_tv->split(-1, TIDx * vec);
  reduction_tv->split(-1, vec);
  reduction_tv->axis(-2)->parallelize(ParallelType::TIDx);
  reduction_tv->axis(-1)->parallelize(ParallelType::Vectorize);
  reduction_tv->axis(-3)->parallelize(ParallelType::BIDx);

  reduction_tv->split(0, TIDy);
  reduction_tv->axis(1)->parallelize(ParallelType::TIDy);
  reduction_tv->split(0, outer_tidy_fact);
  reduction_tv->axis(0)->parallelize(ParallelType::BIDy);

  // T2_g[ rblockIdx.y, rS{16}, rthreadIdx.y, iblockIdx.x, ithreadIdx.x24,
  // iV25{4} ]
  reduction_tv->reorder({{3, 0}, {4, 1}, {0, 2}, {2, 3}, {1, 4}, {5, 5}});
  // T2_g[iblockIdx.x, ithreadIdx.x24, rblockIdx.y, rthreadIdx.y, rS{16},
  // iV25{4}]

  TransformPropagatorWithCheck propagator(reduction_tv);
  MaxRootDomainInfoSpanningTree(reduction_tv).traverse(&propagator);
  auto rfactor_tv = ir_utils::rfactorHelper(reduction_tv, {4});
  scheduler_utils::parallelizeAllLike(rfactor_tv);

  tv0->computeAt(tv_avg, 2);
  tv0->computeAt(cached_input, -2);

  cached_input->computeAt(rfactor_tv, 4, ComputeAtMode::BestEffort);

  for (auto tv : ir_utils::allTvs(&fusion)) {
    if (tv == cached_input || tv == tv_avg || tv == tv_M2) {
      continue;
    }
    tv->axis(-1)->parallelize(ParallelType::Serial);
  }

  // Welford inputs and outputs should not be aliased. See PR #2118.
  class AliasChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

    void handle(kir::Allocate* alloc) final {
      if (alloc->alias() == nullptr) {
        return;
      }
      auto tv = dynamic_cast<TensorView*>(alloc->buffer());
      auto alias_tv = dynamic_cast<TensorView*>(alloc->alias()->buffer());
      if (tv != nullptr && alias_tv != nullptr) {
        alias_map_.emplace(tv, alias_tv);
        alias_map_.emplace(alias_tv, tv);
      }
    }

    void handle(kir::GridWelford* gwop) final {
      for (auto out_ti : ir_utils::filterByType<kir::TensorIndex>(
               gwop->welford_op()->outputs())) {
        auto out_tv = out_ti->view();
        if (alias_map_.count(out_tv) == 0) {
          continue;
        }
        auto alias_tv = alias_map_.at(out_tv);
        for (auto inp_ti : ir_utils::filterByType<kir::TensorIndex>(
                 gwop->welford_op()->inputs())) {
          TORCH_CHECK(
              inp_ti->view() != alias_tv,
              "Invalid alias found between GridWelford input and output. Out tv: ",
              out_tv->toString(),
              ", In tv: ",
              alias_tv->toString());
        }
      }
    }

    std::unordered_map<TensorView*, TensorView*> alias_map_;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.kernel()->topLevelExprs());

  FusionExecutor fe;
  fe.compileFusion(&fusion, {}, LaunchParams());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({X, Y, Y, Z}, options);

  auto cg_outputs = fe.runFusion({t0}, LaunchParams(-1, -1, -1, -1, -1, -1));

  // by default Welford outputs sum of square diff so need to divide to get var
  cg_outputs[1] = cg_outputs[1].div((float)(X * Y * Y));

  auto at_mu = at::mean(t0.to(at::kDouble), {0, 1, 2});
  auto at_var = at::var(t0.to(at::kDouble), {0, 1, 2}, false);

  testValidate(
      &fusion,
      cg_outputs,
      {t0},
      {at_mu, at_var},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(-1, -1, -1, -1, -1, -1));
}

// Test sync insertion with redundant predicates
TEST_F(NVFuserTest, FusionRedundantPredSync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);

  fusion.addOutput(tv3);

  auto tv0c = tv0->cacheAfter();

  // Make a redundant write through smem
  tv0c->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv3, 0);
  tv1->computeAt(tv3, 0);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  GpuLower gpulw(&fusion);
  auto flattened_exprs =
      ir_utils::flattenScopedExprs(gpulw.kernel()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  TORCH_INTERNAL_ASSERT(sync_inserted, "Expected block sync not inserted");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test case for removing syncs on chain of redundant uses.
TEST_F(NVFuserTest, FusionRedundantPredSync2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = add(tv2, tv1);

  fusion.addOutput(tv3);

  auto tv0c = tv0->cacheAfter();

  // Make a redundant write through smem
  tv0c->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv3, 0);
  tv1->computeAt(tv3, 0);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  // Utility class to make sure one block sync
  //  is inserted by RAW pass.
  class SyncChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;
    int result() {
      return sync_seen_;
    }

   private:
    void handle(kir::BlockSync*) final {
      sync_seen_++;
    }

   private:
    int sync_seen_ = 0;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.kernel()->topLevelExprs());
  TORCH_INTERNAL_ASSERT(
      checker.result() < 2, "More syncs were inserted than expected");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
}

// Test case for sync insertion after redundant predicated smem write
//  Check that syncs are removed only when all paths are redundant.
TEST_F(NVFuserTest, FusionRedundantPredSync3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32});
  TensorView* tv1 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false});
  auto tv3 = set(tv2);
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv2, tv1);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  auto tv0c = tv0->cacheAfter();

  // In this scheduling config,
  //  tv0c -> tv2 -> tv3 is a redundant path for tidy
  //  tv0c -> tv2 -> tv5 is not.
  //  So we need a RAW sync in tv0c->tv2 to make sure
  //  tv2 has the correct value to produce tv5.
  tv0c->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv0c->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  tv5->axis(0)->parallelize(ParallelType::TIDy);
  tv5->axis(1)->parallelize(ParallelType::TIDx);

  // Utility class to make sure one block sync
  //  is inserted by RAW pass.
  class SyncChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;
    int result() {
      return sync_seen_;
    }

   private:
    void handle(kir::BlockSync* sync) final {
      if (!sync->isWarHazardSync()) {
        sync_seen_++;
      }
    }

   private:
    int sync_seen_ = 0;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.kernel()->topLevelExprs());

  // This is implicit checking. There are exactly 2 places
  //  where RAW hazards happen: one producing tv2 and the other
  //  producing tv3. This test case expect syncs in both of
  //  these places so we check that 2 RAW syncs are inserted.
  TORCH_INTERNAL_ASSERT(
      checker.result() == 2,
      "Exactly 2 RAW sync expected for the two shared memory transfers");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  auto ref = t0 + t1;

  testValidate(&fusion, cg_outputs, {t0, t1}, {ref, ref}, __LINE__, __FILE__);
}

// Unit test case for detecting thread redundant usage of shared tensors.
TEST_F(NVFuserTest, FusionRedundantUseCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({32, 32});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);

  auto tv5 = set(tv4);

  auto tv6 = set(tv4);
  auto tv7 = set(tv6);

  fusion.addOutput(tv5);
  fusion.addOutput(tv7);

  tv2->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);

  tv7->axis(-1)->parallelize(ParallelType::TIDx);

  // Thread pred map cannot be built without an active lower
  //  object. So would need to lower the whole fusion for
  //  testing. However, lower also keeps an copy of the fusion
  //  so the original pointers cannot be used to querry the
  //  thread pred map. So have to traverse the new expr list
  //  to find the pointers;
  GpuLower gpulw(&fusion);

  TensorView *lowered_tv2 = nullptr, *lowered_tv4 = nullptr;
  auto used_vals = gpulw.kernel()->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->name() == 2) {
      lowered_tv2 = tv;
    }
    if (tv->name() == 4) {
      lowered_tv4 = tv;
    }
  }

  TORCH_INTERNAL_ASSERT(
      lowered_tv2 != nullptr && lowered_tv4 != nullptr,
      "tv2 or tv4 not lowered or mangled");

  auto tv2_info = gpulw.threadPredMap().getPredicateInfo(lowered_tv2);
  auto tv4_info = gpulw.threadPredMap().getPredicateInfo(lowered_tv4);

  // tv2 -> tv3 -> tv4 (shared) is the only use chain for tv2,
  //  and tv4 is redundantly written in tidx so tv2 is redundantly
  //  consumed in tidx.
  TORCH_INTERNAL_ASSERT(
      tv2_info.redundant_use_types.get(ParallelType::TIDx),
      "TV2 is redundantly used but not detected.");

  // tv4->tv5 (global) is a redundant use chain, but
  // tv4->tv6->tv7 is not, so tv4 should not be detected as
  // a redundant used tensor in tidx.
  TORCH_INTERNAL_ASSERT(
      !tv4_info.redundant_use_types.get(ParallelType::TIDx),
      "TV4 is not redundantly used but not detected.");
}

TEST_F(NVFuserTest, FusionUnsqueeze1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // [I, R]
  auto tv1 = sum(tv0, {1});
  // [I, B]
  auto tv2 = unsqueeze(tv1, -1);
  fusion.addOutput(tv2);

  TORCH_CHECK(
      tv2->nDims() == 2, "Unpected unsqueeze result: ", tv2->toString());
  TORCH_CHECK(
      tv2->axis(1)->isBroadcast(),
      "Unexpected unsqueeze result: ",
      tv2->toString());

  // tv1 has only one non-reduction axis. An exception should be
  // thrown.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(unsqueeze(tv1, 2));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.sum(1).unsqueeze(-1);

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSqueeze1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({10, 11});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  // [I, B]
  auto tv1 = sum(tv0, {1}, true);
  // [I]
  auto tv2 = squeeze(tv1, std::vector<int64_t>{shape[0], 1});
  fusion.addOutput(tv2);

  TORCH_CHECK(
      tv2->nDims() == 1, "Unexpected squeeze result: ", tv2->toString());

  // [I, R]
  auto tv3 = sum(tv0, {1});
  // tv3 has only one non-reduction axis. The extent of the first axis
  // is not one, so squeeze should fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(squeeze(tv3, std::vector<int64_t>{shape[0], 1}));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  auto ref = t0.sum(1, true).squeeze(-1);

  testValidate(&fusion, cg_outputs, aten_inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionContigPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, true, false});
  fusion.addOutput(tv2);

  tv2->merge(-2, -1);
  tv2->merge(-2, -1);
  tv2->split(-1, 100);
  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  TORCH_CHECK(PredicatedChecker::isPredicated(tv1, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 4}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = t0.unsqueeze(1);

  testValidate(fe.kernel(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Repro of https://github.com/csarofeen/pytorch/issues/1777
TEST_F(NVFuserTest, FusionDivScalarLhs_CUDA) {
  // tv1 = 2.0 / tv0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = div(IrBuilder::create<Double>(2.0), tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 3}, options);
  // There's no overload div(Scalar, Tensor) in ATen
  auto aten_output = at::div(
      at::native::wrapped_scalar_tensor(at::Scalar(2.0), options.device()), t0);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {aten_output}, __LINE__, __FILE__);
}

// Repro of an issue of the reduction scheduler with a broadcast
// domain concretized to multiple domains that are not proven to have
// the same extent
TEST_F(NVFuserTest, FusionRepro1713_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = broadcast(tv2, {false, true});

  auto tv4 = add(tv3, tv0);

  auto tv5 = add(tv3, tv1);
  auto tv6 = sum(tv5, {0});
  fusion->addOutput(tv4);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 204800}, options);
  // Original repro had the same shape as t0, but this should work
  // with a different extent at the second axis
  at::Tensor t1 = at::randn({1024, 123}, options);
  at::Tensor t2 = at::randn({1024}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1, t2});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t3 = t2.unsqueeze(-1);
  auto t4 = t3 + t0;
  auto t5 = t3 + t1;
  auto t6 = sum(t5, {0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1, t2},
      {t4, t6},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionExpand_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto w = 2, x = 3, y = 4, z = 5;

  // Test
  // a simple expand
  // Expand that's propagated
  // expand_as
  // symbolic expand

  // x
  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = expand(tv1, {tv0->axis(0)->extent(), IrBuilder::create<Int>(y)});

  // x
  auto tv3 = makeSymbolicTensor(1);
  fusion->addInput(tv3);
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv4, tv2);
  // [x, e_y]

  // [x, y, z]
  auto tv6 = makeSymbolicTensor(3);
  fusion->addInput(tv6);

  // Disjoint set op will cause a segmentation for just this op.
  auto tmp_7 = set(tv6);
  fusion->addOutput(tmp_7);

  auto tv7 = broadcast(tv5, {false, false, true});

  auto tv8 = expand_as(tv7, tv6);
  // [x, e_y, e_z]

  auto w_symbolic = IrBuilder::create<Int>();
  fusion->addInput(w_symbolic);

  auto tv9 = broadcast(tv8, {true, false, false, false});
  //[1, x, e_y, e_z]

  auto tv10 = expand(
      tv9,
      {w_symbolic,
       tv9->axis(1)->extent(),
       tv9->axis(2)->expandedExtent(),
       tv9->axis(3)->expandedExtent()});

  fusion->addOutput(tv10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t3 = at::randn({x}, options);
  at::Tensor t6 = at::randn({x, y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t3, t6, w});
  auto cg_out = cg_outputs[1];

  TORCH_INTERNAL_ASSERT(cg_out.size(0) == w);
  TORCH_INTERNAL_ASSERT(cg_out.size(1) == x);
  TORCH_INTERNAL_ASSERT(cg_out.size(2) == y);
  TORCH_INTERNAL_ASSERT(cg_out.size(3) == z);
  TORCH_INTERNAL_ASSERT(cg_out.stride(0) == 0);
  TORCH_INTERNAL_ASSERT(cg_out.stride(1) == 1);
  TORCH_INTERNAL_ASSERT(cg_out.stride(2) == 0);
  TORCH_INTERNAL_ASSERT(cg_out.stride(3) == 0);

  auto t10 = t0.unsqueeze(-1)
                 .expand({x, y})
                 .add(t3.unsqueeze(-1))
                 .unsqueeze(-1)
                 .expand_as(t6)
                 .unsqueeze(0)
                 .expand({w, x, y, z});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t3, t6, w},
      {t6, t10},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionExpandIssue1751_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3, y = 4, z = 5;

  // y, z
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});

  // Two ways to propagate extents as is: use -1 or explicitly pass
  // the extent vals.

  auto tv2 = expand(
      tv1,
      {IrBuilder::create<Int>(x),
       IrBuilder::create<Int>(-1),
       IrBuilder::create<Int>(-1)});

  auto tv3 = expand(
      tv1,
      {IrBuilder::create<Int>(x),
       tv0->axis(0)->extent(),
       tv0->axis(1)->extent()});

  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    TORCH_INTERNAL_ASSERT(cg_out.size(0) == x);
    TORCH_INTERNAL_ASSERT(cg_out.size(1) == y);
    TORCH_INTERNAL_ASSERT(cg_out.size(2) == z);
  }

  auto t2 = t0.expand({x, y, z});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t2, t2}, __LINE__, __FILE__);
}

// TODO: Make sure the kernel uses the expanded concrete size instead
// of the symbolic size
TEST_F(NVFuserTest, FusionExpandToConcrete_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3, y = 4;

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false});

  auto tv2 =
      expand(tv1, {IrBuilder::create<Int>(x), IrBuilder::create<Int>(y)});

  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    TORCH_INTERNAL_ASSERT(cg_out.size(0) == x);
    TORCH_INTERNAL_ASSERT(cg_out.size(1) == y);
  }

  auto t2 = t0.expand({x, y});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReproNoncontigBroadcast_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 32, 16, 112, 112}, options).transpose(-1, -2);
  at::Tensor t1 = at::randn({32, 1, 112, 1}, options).transpose(-1, -2);

  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .contiguity({true, true, false, false, false}) // ttfff
                 .shape({-1, -1, -1, -1, -1})
                 .dtype(DataType::Half)
                 .build();
  auto tv1 = TensorViewBuilder()
                 .ndims(4)
                 .contiguity({true, std::nullopt, std::nullopt, true})
                 .shape({-1, 1, 1, -1})
                 .dtype(DataType::Half)
                 .build();

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);

  fusion->addOutput(tv2);

  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t2 = t0 + t1;

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTransformPropagateSibling_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1760
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion.addOutput(tvs.var_sum);

  tvs.avg->split(1, 1);
  tvs.avg->split(1, 2);
  tvs.avg->split(1, 3);
  tvs.var_sum->split(1, 1);
  tvs.var_sum->split(1, 2);
  tvs.var_sum->split(1, 3);
  tvs.n->split(1, 1);
  tvs.n->split(1, 2);
  tvs.n->split(1, 3);

  auto var_sum_rf = ir_utils::rfactorHelper(tvs.var_sum, {1, 4});

  TransformPropagatorWithCheck propagator(var_sum_rf);
  MaxRootDomainInfoSpanningTree(var_sum_rf).traverse(&propagator);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  std::vector<std::vector<TensorView*>> siblings = {
      {tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
  for (const auto& tensors : siblings) {
    for (auto t1 : tensors) {
      for (auto t2 : tensors) {
        TORCH_CHECK(TransformReplay::fullSelfMatching(t1, t2));
      }
    }
  }
}

TEST_F(NVFuserTest, FusionTransformPropagateSelectorSibling_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion.addOutput(tvs.var_sum);

  tvs.avg->split(1, 1);
  tvs.avg->split(1, 2);
  tvs.avg->split(1, 3);
  tvs.var_sum->split(1, 1);
  tvs.var_sum->split(1, 2);
  tvs.var_sum->split(1, 3);
  tvs.n->split(1, 1);
  tvs.n->split(1, 2);
  tvs.n->split(1, 3);

  auto var_sum_rf = ir_utils::rfactorHelper(tvs.var_sum, {1, 4});

  struct DisableTv0 : public MaxInfoSpanningTree::Selector {
    TensorView* tv0;
    bool allowC2P(TensorView* from, TensorView* to) override {
      return from != tv0 && to != tv0;
    };
    bool allowP2C(TensorView* from, TensorView* to) override {
      return from != tv0 && to != tv0;
    };
    bool allowSibling(TensorView* from, TensorView* to) override {
      return true;
    }
    DisableTv0(TensorView* tv0) : tv0(tv0) {}
  } selector1(tv0);

  struct DisableTv0AndSibling : public DisableTv0 {
    bool allowSibling(TensorView* from, TensorView* to) override {
      return false;
    }
    using DisableTv0::DisableTv0;
  } selector2(tv0);

  TransformPropagatorWithCheck propagator(var_sum_rf);
  MaxRootDomainInfoSpanningTree good_path(var_sum_rf, &selector1);
  MaxRootDomainInfoSpanningTree bad_path(var_sum_rf, &selector2);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  auto check = [&]() {
    std::vector<std::vector<TensorView*>> siblings = {
        {tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
    for (const auto& tensors : siblings) {
      for (auto t1 : tensors) {
        for (auto t2 : tensors) {
          TORCH_CHECK(TransformReplay::fullSelfMatching(t1, t2));
        }
      }
    }
  };

  bad_path.traverse(&propagator);
  ASSERT_ANY_THROW(check());
  good_path.traverse(&propagator);
  check();
}

TEST_F(NVFuserTest, FusionTransformPropagatePosition_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeSymbolicTensor(6);
  fusion.addInput(tv0);

  auto tv2 = broadcast(tv0, {false, false, true, false, false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv0->merge(2);
  tv0->merge(0);
  TransformPropagatorWithCheck propagator(tv0);
  MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);

  TORCH_CHECK(tv1->nDims() == 4);
}

TEST_F(NVFuserTest, FusionIgnoreZeroDimReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0});
  // tv1 is effectively a zero-dim tensor as it only has a reduction
  // axis.
  // Reducing it further is converted to just a set op.
  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto tv2_def = dynamic_cast<LoadStoreOp*>(tv2->definition());
  TORCH_CHECK(
      tv2_def != nullptr,
      "Expected LoadStoreOp but found ",
      tv2->definition()->toString());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({12345}, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = sum(t0, {0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Repro of issue #1770
TEST_F(NVFuserTest, FusionIssue1770Repro_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion->addInput(tv1);

  auto tv2 = ge(tv0, tv1);
  auto tv3 =
      where(tv2, IrBuilder::create<Double>(1), IrBuilder::create<Double>(2));
  fusion->addOutput(tv3);

  std::vector<int64_t> shape({999});
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto ref = where(t0 >= t1, 1.0, 2.0);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionTransformPropagatorSelector_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion->addInput(tv1);

  auto tv2 = add(tv0, tv1);

  auto tv3 = sin(tv2);
  auto tv4 = cos(tv2);

  fusion->addOutput(tv3);
  fusion->addOutput(tv4);

  tv2->split(0, 10);

  struct Selector : public MaxInfoSpanningTree::Selector {
    TensorView* tv0;
    TensorView* tv3;
    bool allowC2P(TensorView* from, TensorView* to) override {
      return to == tv0;
    }
    bool allowP2C(TensorView* from, TensorView* to) override {
      return to == tv3;
    }
    bool allowSibling(TensorView* from, TensorView* to) override {
      return false;
    }
    Selector(TensorView* tv0, TensorView* tv3) : tv0(tv0), tv3(tv3) {}
  } selector(tv0, tv3);

  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2, &selector).traverse(&propagator);

  TORCH_CHECK(tv0->nDims() == 2);
  TORCH_CHECK(tv1->nDims() == 1);
  TORCH_CHECK(tv2->nDims() == 2);
  TORCH_CHECK(tv3->nDims() == 2);
  TORCH_CHECK(tv4->nDims() == 1);
}

TEST_F(NVFuserTest, FusionTransformPropagatorPos_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({22, 105});
  fusion->addInput(tv0);

  auto tv1 = sin(tv0);
  fusion->addOutput(tv1);

  tv1->split(0, 2);
  tv1->split(-1, 3);
  tv1->split(-1, 5);

  TransformPropagatorWithCheck propagator(tv1, 2);
  MaxRootDomainInfoSpanningTree(tv1, 2).traverse(&propagator);

  auto expect = makeConcreteTensor({22, 105});
  expect->split(0, 2);
  TORCH_CHECK(TransformReplay::fullSelfMatching(expect, tv0));
}

TEST_F(NVFuserTest, FusionMaxRootDomainInfoSpanningTreePrintTwice_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = neg(tv1);

  fusion->addOutput(tv2);

  tv1->split(0, 10);

  struct Printer : public MaxInfoSpanningTree::Propagator {
    std::stringstream ss;
    void propagateC2P(TensorView* from, TensorView* to) override {
      ss << "propagateC2P" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
    void propagateP2C(TensorView* from, TensorView* to) override {
      ss << "propagateP2C" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
    void propagateSibling(TensorView* from, TensorView* to) override {
      ss << "propagateSibling" << std::endl;
      ss << "from: " << from->name() << std::endl;
      ss << "to: " << to->name() << std::endl;
    }
  } printer1, printer2;
  printer1.ss << std::endl;
  printer2.ss << std::endl;

  MaxRootDomainInfoSpanningTree path(tv1);
  path.traverse(&printer1);
  path.traverse(&printer2);

  auto expect = R"ESCAPE(
propagateC2P
from: 1
to: 0
propagateP2C
from: 1
to: 2
)ESCAPE";
  TORCH_CHECK(printer1.ss.str() == expect);
  TORCH_CHECK(printer2.ss.str() == expect);
}

TEST_F(NVFuserTest, FusionTransformPropagatorNoOverwrite_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = broadcast(tv0, {true, false, true});
  auto tv2 = sin(tv1);
  fusion->addOutput(tv2);

  tv0->split(0, 2);
  tv2->split(1, 2);
  tv2->split(0, 4);

  MaxRootDomainInfoSpanningTree path1(tv2);
  TransformPropagatorWithCheck propagator1(tv2);
  path1.traverse(&propagator1);

  MaxRootDomainInfoSpanningTree path2(tv0);
  TransformPropagatorWithCheck propagator2(tv0);
  path2.traverse(&propagator2);

  TORCH_CHECK(tv1->axis(0)->isBroadcast());
  TORCH_CHECK(tv1->axis(1)->isBroadcast());
  TORCH_CHECK(!tv1->axis(2)->isBroadcast());
  TORCH_CHECK(!tv1->axis(3)->isBroadcast());
  TORCH_CHECK(tv1->axis(4)->isBroadcast());

  auto expect = makeSymbolicTensor(3);
  expect->split(1, 2);
  expect->split(0, 4);
  TORCH_CHECK(TransformReplay::fullSelfMatching(expect, tv1));
}

TEST_F(NVFuserTest, FusionIssue1785Repro_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(1);
  TensorView* tv1 = makeContigTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  // [B, I]
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  auto tv5 = set(tv4);

  // Register your outputs
  fusion.addOutput(tv5);

  tv5->split(0, 8);
  tv5->split(-1, 8);

  // [Serial, TIDy, TIDX, Serial]

  tv4->computeAt(tv5, -2);
  tv3->computeAt(tv4, -1);
  tv2->computeAt(tv3, 0);
  tv2->split(0, 8);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv1->computeAt(tv5, -2);

  tv5->axis(1)->parallelize(ParallelType::TIDy);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor in1 = at::randn({16}, options);
  at::Tensor in2 = at::randn({12, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {in1, in2});
  auto cg_outputs = fe.runFusion({in1, in2});

  auto tv_ref = in1 + in2;

  testValidate(&fusion, cg_outputs, {in1, in2}, {tv_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSkipReplay_CUDA) {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeContigTensor(1);
    TensorView* tv1 = makeContigTensor(2);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = broadcast(tv0, {false, true});
    auto tv3 = add(tv2, tv1);
    fusion.addOutput(tv3);

    tv3->split(1, 2, false);

    TransformPropagatorWithCheck propagator(tv3);
    MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  }

  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeContigTensor(3);
    fusion.addInput(tv0);

    auto tv1 = sum(tv0, {0, 2});
    auto tv2 = sin(tv1);
    fusion.addOutput(tv2);

    tv0->split(1, 2, false);

    TransformPropagatorWithCheck propagator(tv0);
    MaxRootDomainInfoSpanningTree(tv0).traverse(&propagator);
  }
}

TEST_F(NVFuserTest, FusionInlineRepro1803_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tvs = Welford(tv1, {1});
  auto tvo = set(tvs.var_sum);
  fusion.addOutput(tvo);

  tvo->split(0, 16);
  tvo->axis(1)->parallelize(ParallelType::Unroll);

  tv0->computeAt(tvo, -1, ComputeAtMode::BestEffort);

  TORCH_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.avg->getComputeAtPosition());
  TORCH_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.n->getComputeAtPosition());
  TORCH_CHECK(tvs.var_sum->getComputeAtPosition() == 1);
}

// Unit test for the transform selection logic
TEST_F(NVFuserTest, FusionBoundedDirectionSelection1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigTensor(2);

  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  tv3->split(-1, 5);
  tv3->split(-1, 8);

  scheduler_utils::BoundedDirectionalTransformPropagator::backward(
      tv3, -1, {tv0, tv2});

  // Check that the splits are replayed on tv2
  TORCH_INTERNAL_ASSERT(
      tv2->nDims() == tv3->nDims(),
      "Propagator didn't propagate to tv2: ",
      tv2->toString());

  // Check that the splits are replayed on tv1 as well. Even though
  //  one of its consumers, tv2, is part of the boundary, another
  //  consumer is not a boundary, so tv1 should be transformed as well.
  TORCH_INTERNAL_ASSERT(
      tv1->nDims() == tv3->nDims(),
      "Propagator didn't propagate to tv1: ",
      tv1->toString());
}

TEST_F(NVFuserTest, FusionIssueRepro1844_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> shape = {2, 1, 768};
  std::vector<int64_t> sum_to_shape = {768};
  std::vector<int64_t> sum_to_axes = {0, 1};
  double kProb = 0.5;

  std::vector<Int*> sum_to_symb;
  std::transform(
      sum_to_shape.begin(),
      sum_to_shape.end(),
      std::back_inserter(sum_to_symb),
      [](int s) -> Int* { return IrBuilder::create<Int>(s); });

  TensorView* tv0 = makeContigConcreteTensor(shape);
  TensorView* tv1 = makeContigConcreteTensor(shape);
  TensorView* tv2 = makeContigConcreteTensor(shape, DataType::Bool);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  Double* prob = IrBuilder::create<Double>(kProb);
  auto grad_input = dropout_backward(tv1, tv2, prob);
  auto grad_gelu = gelu_backward(grad_input, tv0);
  auto grad_bias = sum_to(grad_gelu, sum_to_symb);

  fusion->addOutput(grad_gelu);
  fusion->addOutput(grad_bias);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  at::Tensor a = at::randn(shape, options);
  at::Tensor b = at::randn(shape, options);
  at::Tensor c = at::randn(shape, options);
  auto mask = at::gt(c, 0.0f);
  std::vector<c10::IValue> aten_inputs = {a, b, mask};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto dinput = at::native_dropout_backward(b, mask, kProb);
  auto dgelu = at::gelu_backward(dinput, a, "none");
  auto dbias = dgelu.sum(sum_to_axes);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {dgelu, dbias},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionInsertMagicZero1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv2->split(-1, 2);
  tv2->reorder({{1, 2}, {2, 1}});
  tv2->merge(0);

  TransformPropagatorWithCheck propagator(tv2);
  MaxRootDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 1);

  // The predicate of tv2 should be protected with magic zero
  GpuLower gpulw(&fusion);
  TORCH_CHECK(
      PredicateMagicZeroChecker::isProtected(tv2, gpulw),
      "Failed to protect the predicates of ",
      tv2->toString());
}

TEST_F(NVFuserTest, FusionExpandRepro1860_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);
  std::vector<std::optional<bool>> contiguity(3, std::nullopt);

  std::vector<int64_t> shape{1, -1, -1};
  TensorView* tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);
  TensorView* tv1 = makeContigConcreteTensor(shape);
  fusion.addInput(tv1);
  TensorView* tv2 = makeContigConcreteTensor(shape);
  fusion.addInput(tv2);

  std::vector<IterDomain*> domain1(3, nullptr);
  for (const auto i : c10::irange(3)) {
    if (i == 0) {
      domain1[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
              .iter_type(IterType::Broadcast)
              .build();
    } else {
      domain1[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
              .expanded_extent(IrBuilder::create<Int>(1 + i))
              .iter_type(IterType::Broadcast)
              .build();
    }
  }

  TensorView* tv22 = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domain1, contiguity), DataType::Float);

  fusion.addInput(tv22);

  auto tv3 = add(tv0, tv1);
  auto tv4 = softmax(tv3, 0);
  auto tv5 = add(tv4, tv22);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({1, 2, 3}, options);
  at::Tensor input2 = at::randn({1, 2, 3}, options);
  at::Tensor input3 = at::randn({1, 2, 3}, options);
  at::Tensor input4 = at::randn({1, 1, 1}, options).expand({1, 2, 3});
  std::vector<c10::IValue> aten_inputs = {input1, input2, input3, input4};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs(aten_inputs);
}

TEST_F(NVFuserTest, FusionExpandReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 8});
  fusion->addInput(tv0);

  auto tv1 =
      expand(tv0, {IrBuilder::create<Int>(12), IrBuilder::create<Int>(8)});

  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.expand({12, 8}).sum({0});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Predicate elimination issue repro:
TEST_F(NVFuserTest, FusionExpandReduce2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 4});
  fusion->addInput(tv0);

  auto tv1 =
      expand(tv0, {IrBuilder::create<Int>(3), IrBuilder::create<Int>(4)});

  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  // tv2[r{3}, i{4}]
  tv2->split(0, NamedScalar::getParallelDim(ParallelType::TIDy));
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv2->split(0, NamedScalar::getParallelDim(ParallelType::BIDy), false);
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->split(-1, NamedScalar::getParallelDim(ParallelType::TIDx));
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::BIDx);
  // [rBIDy, rO, rTIDy, iBIDx, iTIDx]
  tv2->reorder({{-2, 0}, {-1, 1}, {2, 2}});
  // [iBIDx, iTIDx, rTIDy, rBIDy, rO]
  auto tv3 = tv2->rFactor({-1});

  TransformPropagatorWithCheck propagator(tv3);
  MaxRootDomainInfoSpanningTree(tv3).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv3);
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1, 4}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0}, LaunchParams(-1, 2, -1, 4, 2, 1));
  auto cg_outputs = fe.runFusion({t0}, LaunchParams(-1, 2, -1, 4, 2, 1));

  auto ref = t0.expand({3, 4}).sum({0});

  testValidate(
      fusion.get(),
      cg_outputs,
      {t0},
      {ref},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(-1, 2, -1, 4, 2, 1));
}

TEST_F(NVFuserTest, FusionVectorComponentReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::ComplexFloat);
  fusion->addInput(tv0);
  auto tv1 = view_as_real(tv0);
  auto tv2 = sum(tv1, {-1});
  fusion->addOutput(tv2);

  inlineMost();

  auto options =
      at::TensorOptions().dtype(at::kComplexFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  auto t0 = at::randn({1024}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref = at::view_as_real(t0).sum({-1});

  testValidate(fusion.get(), cg_outputs, {t0}, {ref}, __LINE__, __FILE__, "");
}

TEST_F(NVFuserTest, FusionExpandBadShapeTest_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(&fusion);
  std::vector<std::optional<bool>> contiguity{false, std::nullopt};

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  std::vector<IterDomain*> domains = {
      IterDomainBuilder(
          FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>())
          .build(),
      IterDomainBuilder(
          FusionGuard::getCurFusion()->zeroVal(), IrBuilder::create<Int>(1))
          .expanded_extent(IrBuilder::create<Int>(10))
          .iter_type(IterType::Broadcast)
          .build()};

  // expand to 10
  TensorView* tv22 = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(domains, contiguity), DataType::Float);

  fusion.addInput(tv22);

  auto tv3 = add(tv0, tv22);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // Incompatible shapes
  at::Tensor input1 = at::randn({2, 3}, options);
  // Passing expand size of 5, not 10. Should cause an error
  at::Tensor input4 = at::randn({2, 1}, options).expand({2, 5});

  std::vector<c10::IValue> aten_inputs = {input1, input4};

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  ASSERT_ANY_THROW(executor_cache.runFusionWithInputs(aten_inputs));
}

TEST_F(
    NVFuserTest,
    FusionPointwiseScheduleWithBroadcastAndTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv3 = sin(tv2);
  auto tv4 = add(tv3, tv1);
  auto tv5 = sum(tv4, {1});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100, 100, 10}, options);
  at::Tensor t1 = at::randn({10, 20}, options);

  auto aten_output = (t0.view({100, 1, 100, 1, 10, 1}).sin() + t1).squeeze(1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineMost();

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().cos().transpose(1, 2).exp().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = exp(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  inlineAllAt(tv5, -1, true);

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().cos().transpose(1, 2).exp().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningMismatchedDims4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = exp(tv1);
  auto tv3 = relu(tv2);
  auto tv4 = cos(tv3);
  auto tv5 = tan(tv4);
  fusion.addOutput(tv5);

  tv3->merge(1);
  inlineMost();

  TORCH_CHECK(tv5->getComputeAtPosition() == 3);
  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 1);
  TORCH_CHECK(tv2->getComputeAtPosition() == 1);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().exp().relu().cos().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInliningBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = sin(tv0);
  // broadcasting
  auto tv2 = broadcast(tv1, {false, true, false, true, false, true});
  auto tv3 = cos(tv2);
  auto tv4 = tan(tv3);
  fusion.addOutput(tv4);

  for (auto tv : {tv2, tv3, tv4}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  inlineMost();

  TORCH_CHECK(tv4->getComputeAtPosition() == 3);
  TORCH_CHECK(tv3->getComputeAtPosition() == 3);
  TORCH_CHECK(tv2->getComputeAtPosition() == 3);
  TORCH_CHECK(tv1->getComputeAtPosition() == 3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({2, 3, 4}, options);
  auto output = input.sin().view({2, 1, 3, 1, 4, 1}).cos().tan();

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input});
  auto cg_outputs = fe.runFusion({input});

  testValidate(&fusion, cg_outputs, {input}, {output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMatchedLeafPosWithoutReplayBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 3, 4});
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true, false, true, false, true});
  auto tv2 = sin(tv1);
  fusion.addOutput(tv2);

  for (auto tv : {tv1, tv2}) {
    tv->merge(0);
    tv->merge(1);
    tv->merge(2);
  }

  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv0, tv1, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv1, tv0, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayPasC(tv1, tv2, 3) == 3);
  TORCH_CHECK(
      TransformReplay::getMatchedLeafPosWithoutReplayCasP(tv2, tv1, 3) == 3);
}

TEST_F(NVFuserTest, FusionPrint_CUDA) {
  auto dtypes = {
      at::kFloat,
      at::kDouble,
      at::kHalf,
      at::kBFloat16,
      at::kInt,
      at::kLong,
      at::kBool};
  for (auto dtype : dtypes) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeSymbolicTensor(1, aten_to_data_type(dtype));
    fusion->addInput(tv0);
    auto tv1 = print(tv0);
    auto tv2 = sin(tv1);
    fusion->addOutput(tv2);

    // There is no way to check if anything is printed to the console, but we
    // can validate that when print exist, compilation and computation are not
    // broken.
    auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
    at::Tensor t0 = at::arange(2, options).to(dtype);

    FusionExecutorCache executor_cache(std::move(fusion));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    testValidate(
        executor_cache.fusion(),
        cg_outputs,
        {t0},
        {t0.sin()},
        __LINE__,
        __FILE__);
  }
}

TEST_F(NVFuserTest, FusionCheckedSymbolicShape_CUDA) {
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor a = at::randn({123, 456}, options);
  at::Tensor b = at::randn({123, 456}, options);
  at::Tensor c = at::randn({321, 654}, options);

  using return_t =
      std::pair<std::unique_ptr<FusionExecutorCache>, std::vector<at::Tensor>>;
  auto matched_add = [](at::Tensor a, at::Tensor b) -> return_t {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* s1 = IrBuilder::create<Int>();
    Val* s2 = IrBuilder::create<Int>();
    auto builder = TensorViewBuilder().shape(std::vector<Val*>{s1, s2});
    TensorView* tv0 = builder.build();
    TensorView* tv1 = builder.build();

    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);

    fusion->addOutput(tv2);

    auto executor_cache =
        std::make_unique<FusionExecutorCache>(std::move(fusion));
    auto cg_outputs = executor_cache->runFusionWithInputs({a, b});
    return {std::move(executor_cache), std::move(cg_outputs)};
  };

  {
    auto ret1 = matched_add(a, b);
    testValidate(
        ret1.first->fusion(), ret1.second, {a, b}, {a + b}, __LINE__, __FILE__);
  }

  {
    EXPECT_THAT(
        [&]() { matched_add(a, c); },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Attempting to bind")));
  }
}

TEST_F(NVFuserTest, FusionSizeDependentData_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* s1 = IrBuilder::create<Int>();
  auto builder = TensorViewBuilder().shape(std::vector<Val*>{s1});
  TensorView* tv0 = builder.build();

  fusion->addInput(tv0);

  auto tv1 = add(tv0, s1);

  fusion->addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor a = at::zeros({123}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({a});

  testValidate(
      executor_cache.fusion(), cg_outputs, {a}, {a + 123}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDependencyCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(1);
  TensorView* tv3 = makeSymbolicTensor(1);

  auto tv4 = add(tv0, tv1);
  auto tv5 = add(tv0, tv2);
  auto tv6 = add(tv0, tv3);

  auto tv7 = add(tv1, tv2);
  auto tv8 = add(tv1, tv3);

  auto tv9 = add(tv2, tv3);

  {
    auto all_vals = DependencyCheck::getAllValsBetween(
        {tv0, tv1}, {tv4, tv5, tv6, tv7, tv8, tv9});
    std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
    std::vector<Val*> results({tv0, tv1, tv4, tv5, tv6, tv7, tv8});
    for (auto result : results) {
      TORCH_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    TORCH_CHECK(all_vals_set.empty());
  }

  auto tv10 = add(tv6, tv7);
  {
    auto all_vals = DependencyCheck::getAllValsBetween({tv0, tv1}, {tv10});
    std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
    std::vector<Val*> results({tv0, tv1, tv6, tv7, tv10});
    for (auto result : results) {
      TORCH_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    TORCH_CHECK(all_vals_set.empty());
  }
}

// Repro for issue #1925
TEST_F(NVFuserTest, FusionScheduleTransposeRepro1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeConcreteTensor({-1, -1, -1, 1});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1, 1, 333, 1}, options);
  at::Tensor input1 = at::randn({1, 1, 333, 1}, options);

  auto lparams = scheduleTranspose(&fusion, {input0, input1});

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, lparams);
  auto outputs = fe.runFusion({input0, input1}, lparams);

  auto tv_ref = input0 + input1;

  testValidate(
      &fusion, outputs, {input0, input1}, {tv_ref}, __LINE__, __FILE__);
}

// Repro for issue #1873
TEST_F(NVFuserTest, FusionInlineBroadcastIndexing0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  tv0->computeAt(tv4, 1);

  tv2->split(-1, 8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({123}, options);
  at::Tensor t1 = at::randn({3, 123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  auto outputs = fe.runFusion({t0, t1});

  auto tv_ref = t0 + t1;

  testValidate(&fusion, outputs, {t0, t1}, {tv_ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPredicateUnshare_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1926
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion->addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);
  for (auto tv : {tv1, tv2}) {
    tv->split(0, 4);
    tv->reorder({{1, -1}});
    tv->split(1, 8);
    tv->merge(0);
    tv->split(0, 1);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::Unswitch);
  }
  tv1->merge(2);
  tv2->reorder({{2, 3}});
  tv2->merge(2);
  for (auto tv : {tv1, tv2}) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 5}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto out = cg_outputs[0];

  testValidate(fusion, {out}, {t0}, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, AsyncCompilation_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Double>(1)); // Group 0
  TensorView* tv4 =
      max(tv3, {0}); // Group 0 (use max instead to avoid numerical issues)
  TensorView* tv5 = add(tv4, tv1); //  Group 0 (Non Broadcast after reduce,
                                   //  keeps normalization scheduler away)
  TensorView* tv6 = add(tv5, tv2); //  Group 1 (Broadcast after reduce)

  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({8, 5}, options);
  at::Tensor t1 = at::randn({5}, options);
  at::Tensor t2 = at::randn({8, 5}, options);

  auto t3 = t0.add(1.0);
  auto t4 = std::get<0>(at::max(t3, 0));
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);

  FusionExecutorCache executor_cache(std::move(fusion));

  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  executor_cache.compileFusionAsync(aten_inputs);

  while (!executor_cache.isCompiled(aten_inputs)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    printf(".");
  }

  auto outputs = executor_cache.runFusionWithInputs(aten_inputs);

  TORCH_CHECK(
      executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation didn't happen");
  TORCH_CHECK(
      executor_cache.getMostRecentKernelRuntime()
              ->fusionSegments()
              ->groups()
              .size() == 2,
      "segmentation didn't happen as expected");

  testValidate(
      executor_cache.fusion(), outputs, aten_inputs, {t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMergeBroadcastingTrivialReduction1_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({1, 1});
  TensorView* tv1 = makeConcreteTensor({-1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = sum(tv0, {1});
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  tv0->merge(0);

  MaxRootDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({10}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  testValidate(
      fusion, {out}, {t0, t1}, {t1 + t0.flatten()}, __LINE__, __FILE__);
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 1, 1});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1, 1}, options);

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({0, 1, 9223372036854775807L});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({0, 1, 9223372036854775807L}, options);

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

// Simple test case exercising the null scheduler path.
TEST_F(NVFuserTest, FusionNullScheduler3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = TensorViewBuilder().ndims(0).build();
  auto tv1 = TensorViewBuilder().ndims(0).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);
  at::Tensor t1 = at::randn({}, options);

  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0 + t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

TEST_F(NVFuserTest, FusionReducingZeroElements_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({0, 1, 9223372036854775807L});
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {0, 1, 2});

  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({0, 1, 9223372036854775807L}, options);

  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto t1 = t0.sum({0, 1, 2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionEmpty_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({10, 10, 10});
  auto tv1 = makeConcreteTensor({10, 10, 10});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 10, 10}, options);
  at::Tensor t1 = at::randn({10, 10, 10}, options);

  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1},
      {t0, t1},
      __LINE__,
      __FILE__);

  auto groups =
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups();

  // Check that all groups on the resulting runtime are null.
  for (auto group : groups) {
    TORCH_INTERNAL_ASSERT(group->heuristic() == ScheduleHeuristic::NoOp);
  }
}

TEST_F(NVFuserTest, FusionMappingRelation_CUDA) {
  // See https://github.com/csarofeen/pytorch/pull/1960
  // and https://github.com/csarofeen/pytorch/pull/2113
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({1, 1});
  TensorView* tv1 = makeConcreteTensor({-1, 1, 1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false, false});
  auto tv4 = add(tv3, tv1);

  fusion->addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-1);

  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  ComputeAtMap ca_map(fusion);

  auto tv4_inner_node = tv4->axis(0)->definition()->input(1)->as<IterDomain>();
  TORCH_CHECK(
      ca_map.areMapped(tv2->axis(0), tv4_inner_node, IdMappingMode::EXACT));
  TORCH_CHECK(ca_map.areMapped(
      tv2->axis(0), tv4_inner_node, IdMappingMode::PERMISSIVE));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({2, 1, 1}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});
  auto out = cg_outputs[0];

  testValidate(
      fusion, {out}, {t0, t1}, {t1 + t0.squeeze(0)}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInlineAt_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = sin(tv0);
  auto tv2 = cos(tv1);
  fusion->addOutput(tv2);

  tv1->inlineAt(-1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});
  auto out = cg_outputs[0];

  testValidate(fusion, {out}, {t0}, {t0.sin().cos()}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialInputForwarding_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeConcreteTensor({-1, -1});
  TensorView* tv1 = makeConcreteTensor({-1, -1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  // Note: output of add is not used. Kept it here since previously there was an
  // assertion from sorting in codegen.
  add(tv1, IrBuilder::create<Double>(3.141));
  fusion->addOutput(tv0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 4}, options);
  at::Tensor t1 = at::randn({10, 4}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0, t1});

  testValidate(fusion, cg_outputs, {t0, t1}, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  TORCH_CHECK(fec.isCompiled({t0, t1}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0, t1});
  testValidate(fusion, cg_outputs2, {t0, t1}, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialInputForwarding2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(0);
  fusion->addInput(tv0);
  fusion->addOutput(tv0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0});

  testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  TORCH_CHECK(fec.isCompiled({t0}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0});
  testValidate(fusion, cg_outputs2, {t0}, {t0}, __LINE__, __FILE__);
}

// Simplified repro of issue #2008
TEST_F(NVFuserTest, FusionReplayTrivialReductionAndBroadcast2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({10, 1, 1});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Double>(1));
  auto tv2 = sum(tv1, {1, 2});
  auto tv3 = broadcast(tv2, {false, true, true});
  fusion.addOutput(tv3);

  tv0->merge(-2, -1)->merge(-2, -1)->split(0, 4);

  MaxRootDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), aten_inputs);
  auto outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, outputs, aten_inputs, {t0 + 1}, __LINE__, __FILE__);
}

namespace {

size_t getVecSizeForPointwise(FusionExecutorCache& fec) {
  auto most_recent_params =
      fec.getMostRecentKernelRuntime()->getMostRecentExecutorLog().params;
  auto params = dynamic_cast<PointwiseParams*>(most_recent_params.get());
  if (params->vectorize) {
    return params->unroll_factor;
  }
  return 1;
}

} // namespace

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity2D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(2).contiguity({false, true}).build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity3D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(3).contiguity({false, true, true}).build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto pair : size_and_vec) {
    auto size = pair.first;
    auto vec = pair.second;
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguity5D_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int>> sizes_and_vec{
      {9, 17, 1}, {9, 10, 2}, {9, 16, 4}};

  for (auto tup : sizes_and_vec) {
    auto size1 = std::get<0>(tup);
    auto size2 = std::get<1>(tup);
    auto vec = std::get<2>(tup);
    at::Tensor t0 = at::randn({4, size1, 12345, size2, 3}, options)
                        .narrow(1, 0, 8)
                        .narrow(3, 0, 4);
    auto cg_outputs = fec.runFusionWithInputs({t0});

    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);

    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionVectorizeStrideContiguitySelfOverlapping_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache fec(std::move(fusion_ptr));
  fec.profile(true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int>> sizes_strides_and_vec{
      {4, 4, 4, 4},
      {4, 4, 2, 2},
      {4, 2, 4, 2},
      {2, 4, 4, 2},
      {4, 4, 1, 1},
      {4, 1, 4, 1},
      {1, 4, 4, 1},
      {2, 2, 2, 2},
      {2, 2, 1, 1},
      {2, 1, 2, 1},
      {1, 2, 2, 1}};

  for (auto tup : sizes_strides_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto vec = std::get<3>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {
        stride1, (int64_t)stride2 * 12345, (int64_t)stride2, 3, 1};
    at::Tensor t0 = at::empty_strided(shape, stride, options);
    t0.random_();
    auto cg_outputs = fec.runFusionWithInputs({t0});
    TORCH_CHECK(getVecSizeForPointwise(fec) == (size_t)vec);
    testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionSimpleAmperePipeline_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
    return;
  }

  auto tv0 = makeContigTensor(1);

  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  fusion.addOutput(tv1);

  auto tv_cache = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv_cache->setMemoryType(MemoryType::Shared);

  tv1->split(0, 16);
  tv0->computeAt(tv1, 1);

  tv_cache->circularBuffer(10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input1 = at::randn({255}, options);

  // Add check that the cp async op has an inlined predicate.
  class InlinedCpAsyncPredChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;

   private:
    void handle(kir::IfThenElse* ite) final {
      auto prev_within_ite = within_ite_;
      within_ite_ = true;
      kir::IrVisitor::handle(ite);
      within_ite_ = prev_within_ite;
    }

    void handle(LoadStoreOp* ldst) final {
      if (ldst->opType() == LoadStoreOpType::CpAsyncCa) {
        TORCH_INTERNAL_ASSERT(!within_ite_, "CPASYNC predicate not inlined");
        TORCH_INTERNAL_ASSERT(
            ldst->predicate()->hasValue() &&
                !ldst->predicate()->value()->isConst(),
            "CPASYNC predicate is not generated");
      }
    }

   private:
    bool within_ite_ = false;
  } pred_checker;

  // Check that cp async is inlined:
  GpuLower gpulw(&fusion);
  pred_checker.handle(gpulw.kernel()->topLevelExprs());

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1});
  auto cg_outputs = fe.runFusion({input1});

  testValidate(&fusion, cg_outputs, {input1}, {input1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpandedInput_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .shape({-1, -1, -1})
                        .contiguity({false, std::nullopt, true})
                        .expanded({false, true, false})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4096, 1, 4}, options).expand({-1, 7, -1});

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0});

  testValidate(fusion, cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Repro for
// https://github.com/csarofeen/pytorch/issues/1843#issuecomment-1270759724
TEST_F(NVFuserTest, FusionVectorizeRepro1843_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv1 =
      TensorViewBuilder().ndims(2).contiguity({true, true}).build();
  TensorView* tv0 =
      TensorViewBuilder().ndims(2).contiguity({true, true}).build();
  fusion->addInput(tv1);
  fusion->addInput(tv0);

  auto tv7 = sum(tv0, {1}, true);
  auto tv_exp =
      expand(tv7, {tv0->axis(0)->extent(), IrBuilder::create<Int>(32128)});
  auto tv3 = exp(tv1);
  auto tv8 = mul(tv3, tv_exp);
  auto tv13 = sub(tv0, tv8);
  fusion->addOutput(tv13);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 =
      at::empty_strided({4096, 32128}, {32128, 1}, options).random_();
  at::Tensor t0 =
      at::empty_strided({4096, 32128}, {32128, 1}, options).random_();

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t1, t0});

  auto ref = t0 - t1.exp() * t0.sum((1), true);
  testValidate(fusion, cg_outputs, {t1, t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastPersistentReduction_CUDA) {
  // Simplified repro for
  // https://github.com/csarofeen/pytorch/issues/2094
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = broadcast(tv1, {true, true, false, false});
  auto tv3 = sum(tv2, {-1}, true);
  auto tv4 = add(tv2, tv3); // TODO: changing this to tv1 there is still errors
  auto tv5 = sum(tv4, {-1});
  fusion->addInput(tv0);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({1024, 768}, options);
  auto t1 = t0.view({1, 1, 1024, 768}).to(at::kFloat);
  auto t3 = t1.sum({-1}, true);
  auto t4 = t1 + t3;
  auto t5 = t4.sum({-1});

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({t0});
  testValidate(fusion, cg_outputs, {t0}, {t5}, __LINE__, __FILE__);
}

// Repro for
// https://github.com/csarofeen/pytorch/issues/2094
TEST_F(NVFuserTest, FusionRepro2094_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  std::vector<int64_t> neg_one_vec = {-1};
  {
    auto tv0 = TensorViewBuilder()
                   .ndims(1)
                   .shape(neg_one_vec)
                   .contiguity(true)
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv0);
    auto tv1 = TensorViewBuilder()
                   .ndims(1)
                   .shape(neg_one_vec)
                   .contiguity(true)
                   .dtype(DataType::Float)
                   .build();
    fusion->addInput(tv1);
    auto tv2 = TensorViewBuilder()
                   .ndims(2)
                   .shape(std::vector<int64_t>{-1, -1})
                   .contiguity({true, true})
                   .dtype(DataType::Half)
                   .build();
    fusion->addInput(tv2);
    auto tv3 = expand(
        broadcast(tv0, {true, true, false}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(768)});
    auto tv4 = expand(
        broadcast(tv1, {true, true, false}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(768)});
    auto tv5 = reshape(tv2, {1024, 768}, {1, 1024, 768});
    auto tv6 = castOp(DataType::Float, tv5);
    auto s7 = IrBuilder::create<Double>(0.5);
    auto tv8 = mul(tv6, s7);
    auto s9 = IrBuilder::create<Double>(0.707107);
    auto tv10 = mul(tv6, s9);
    auto tv11 = erf(tv10);
    auto s12 = IrBuilder::create<Double>(1.0);
    auto tv13 = add(tv11, s12);
    auto tv14 = mul(tv8, tv13);
    auto tv15 = castOp(DataType::Half, tv14);
    auto tv16 = castOp(DataType::Float, tv15);
    auto tv17_tv18 = variance_mean(tv16, {2}, 0, false);
    auto tv17 = std::get<0>(tv17_tv18);
    auto tv18 = std::get<1>(tv17_tv18);
    auto tv19 = expand(
        broadcast(tv17, {false, false, true}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(1)});
    auto tv20 = expand(
        broadcast(tv18, {false, false, true}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(1)});
    auto s21 = IrBuilder::create<Double>(1e-05);
    auto tv22 = add(tv19, s21);
    auto tv23 = expand(
        broadcast(tv20, {false, false, false}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(768)});
    auto tv24 = rsqrt(tv22);
    auto tv25 = sub(tv16, tv23);
    auto tv26 = expand(
        broadcast(tv24, {false, false, false}),
        {IrBuilder::create<Int>(1),
         IrBuilder::create<Int>(1024),
         IrBuilder::create<Int>(768)});
    auto tv27 = mul(tv25, tv26);
    auto tv28 = mul(tv27, tv3);
    auto tv29 = add(tv28, tv4);
    auto tv30 = castOp(DataType::Float, tv29);
    auto tv31 = castOp(DataType::Half, tv30);
    auto tv32 = reshape(tv31, {1, 1024, 768}, {1024, 768});
    fusion->addOutput(tv5);
    fusion->addOutput(tv16);
    fusion->addOutput(tv20);
    fusion->addOutput(tv24);
    fusion->addOutput(tv32);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  std::vector<c10::IValue> inputs;
  std::vector<at::Tensor> outputs;

  {
    auto t0 = at::randn({768}, options);
    inputs.push_back((c10::IValue)t0);
    auto t1 = at::randn({768}, options);
    inputs.push_back((c10::IValue)t1);
    auto t2 = at::randn({1024, 768}, options).to(at::ScalarType::Half);
    inputs.push_back((c10::IValue)t2);
    auto t3 = t0.unsqueeze(0).unsqueeze(1).expand({1, 1024, 768});
    auto t4 = t1.unsqueeze(0).unsqueeze(1).expand({1, 1024, 768});
    auto t5 = t2.view({1, 1024, 768});
    auto t6 = t5.to(at::ScalarType::Float);
    auto s7 = 0.5;
    auto t8 = at::mul(t6, s7);
    auto s9 = 0.707107;
    auto t10 = at::mul(t6, s9);
    auto t11 = at::erf(t10);
    auto s12 = 1.0;
    auto t13 = at::add(t11, s12);
    auto t14 = at::mul(t8, t13);
    auto t15 = t14.to(at::ScalarType::Half);
    auto t16 = t15.to(at::ScalarType::Float);
    auto t17_t18 =
        at::var_mean(t16, {2}, /*unbiased*/ false, /*keepdim*/ false);
    auto t17 = std::get<0>(t17_t18);
    auto t18 = std::get<1>(t17_t18);
    auto t19 = t17.unsqueeze(2).expand({1, 1024, 1});
    auto t20 = t18.unsqueeze(2).expand({1, 1024, 1});
    auto s21 = 1e-05;
    auto t22 = at::add(t19, s21);
    auto t23 = t20.expand({1, 1024, 768});
    auto t24 = at::rsqrt(t22);
    auto t25 = at::sub(t16, t23);
    auto t26 = t24.expand({1, 1024, 768});
    auto t27 = at::mul(t25, t26);
    auto t28 = at::mul(t27, t3);
    auto t29 = at::add(t28, t4);
    auto t30 = t29.to(at::ScalarType::Float);
    auto t31 = t30.to(at::ScalarType::Half);
    auto t32 = t31.view({1024, 768});
    outputs.push_back(t5);
    outputs.push_back(t16);
    outputs.push_back(t20);
    outputs.push_back(t24);
    outputs.push_back(t32);
  }

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);
  testValidate(fusion, cg_outputs, inputs, outputs, __LINE__, __FILE__);
}

// https://github.com/csarofeen/pytorch/issues/2068
TEST_F(NVFuserTest, FusionIssue2068_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int w = 32, x = 56, y = 56, z = 128;

  auto tv0 = makeContigTensor(3);
  auto tv1 = makeContigTensor(1);
  auto tv2 = makeContigTensor(3);
  auto tv3 = makeContigTensor(1);
  auto tv4 = makeContigTensor(4);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addInput(tv4);

  auto tv5 = broadcast(tv0, {false, false, false, true});
  auto tv6 = broadcast(tv1, {true, true, true, false});
  auto tv7 = expand(
      tv6,
      {IrBuilder::create<Int>(w),
       IrBuilder::create<Int>(x),
       IrBuilder::create<Int>(y),
       tv6->axis(3)->extent()});
  auto tv8 = broadcast(tv2, {false, false, false, true});
  auto tv9 = broadcast(tv3, {true, true, true, false});
  auto tv10 = expand(
      tv9,
      {IrBuilder::create<Int>(w),
       IrBuilder::create<Int>(x),
       IrBuilder::create<Int>(y),
       tv9->axis(3)->extent()});
  auto tv11 = set(tv5);
  auto tv12 = expand(
      tv11,
      {tv11->axis(0)->extent(),
       tv11->axis(1)->extent(),
       tv11->axis(2)->extent(),
       IrBuilder::create<Int>(z)});

  auto tv13 = add(tv8, IrBuilder::create<Double>(1.e-6));
  auto tv14 = sub(tv4, tv12);
  auto tv15 = rsqrt(abs(tv13));
  auto tv16 = set(tv15);
  auto tv17 = expand(
      tv16,
      {tv16->axis(0)->extent(),
       tv16->axis(1)->extent(),
       tv16->axis(2)->extent(),
       IrBuilder::create<Int>(z)});
  auto tv18 = mul(tv14, tv17);
  auto tv19 = mul(tv18, tv7);
  auto tv20 = add(tv19, tv10);
  auto tv21 = set(tv20);

  fusion.addOutput(tv5);
  fusion.addOutput(tv15);
  fusion.addOutput(tv21);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({w, x, y}, options);
  auto t1 = at::randn({z}, options);
  auto t2 = at::randn({w, x, y}, options);
  auto t3 = at::randn({z}, options);
  auto t4 = at::randn({w, x, y, z}, options);

  auto t5 = t0.unsqueeze(-1);
  auto t12 = t5.expand({-1, -1, -1, z});
  auto t7 = t1.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand({w, x, y, -1});
  auto t8 = t2.unsqueeze(-1);
  auto t10 = t3.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand({w, x, y, -1});

  auto t13 = t8 + 1.e-6;
  auto t14 = t4 - t12;
  auto t15 = t13.abs().rsqrt();
  auto t17 = t15.expand({-1, -1, -1, z});
  auto t18 = mul(t14, t17);
  auto t19 = mul(t18, t7);
  auto t21 = add(t19, t10);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1, t2, t3, t4},
      {t5, t15, t21},
      __LINE__,
      __FILE__,
      "");
}

// Similar to the following HuggingFace repro:
// https://github.com/csarofeen/pytorch/issues/2064
// but with the trivial reduction replaced with squeeze
TEST_F(NVFuserTest, FusionHuggingFaceRepro2064Squeeze_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});
  auto tv2 = mul(tv1, IrBuilder::create<Double>(0.5));
  auto tv3 = mul(tv1, IrBuilder::create<Double>(0.707107));
  auto tv4 = erf(tv3);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1.0));
  auto tv6 = mul(tv2, tv5);
  auto tv7 = squeeze(tv6, std::vector<bool>{true, false, false});

  fusion.addOutput(tv1);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 8}, options);
  auto t1 = t0.unsqueeze(0);
  auto t2 = t1 * 0.5;
  auto t5 = (t1 * 0.707107).erf() + 1.0;
  auto t6 = t2 * t5;
  auto t7 = t6.squeeze(0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0},
      {t1, t7},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionSqueezeTransformPropagation_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({5, 1, 1, 1, 1});
  fusion.addInput(tv0);
  auto tv1 = squeeze(tv0, std::vector<bool>{false, true, false, true, false});
  auto tv2 = squeeze(tv0, std::vector<bool>{false, false, true, false, true});
  auto tv3 = squeeze(tv0, std::vector<bool>{false, false, false, false, true});
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->merge(0);
  tv3->merge(0);

  MaxRootDomainInfoSpanningTree tree(tv3);
  TransformPropagatorWithCheck tp(tv3);
  tree.traverse(&tp);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 1, 1, 1, 1}, options);
  auto t1 = t0.squeeze(1).squeeze(2);
  auto t2 = t0.squeeze(2).squeeze(-1);
  auto t3 = t0.squeeze(-1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t1, t2, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSqueezeInlining_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1, -1});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, std::vector<bool>{true, false});
  fusion.addOutput(tv2);

  tv0->merge(0);
  tv0->split(0, 128);

  {
    MaxRootDomainInfoSpanningTree tree(tv0);
    TransformPropagatorWithCheck tp(tv0);
    tree.traverse(&tp);
    TORCH_CHECK(tv2->nDims() == 2);
    TORCH_CHECK(tv1->nDims() == 2);
    TORCH_CHECK(tv0->nDims() == 2);
  }

  {
    // The propagation here should be a no-op, I am adding it here just to test
    // if transformation propagation works for squeeze on both direction.
    MaxRootDomainInfoSpanningTree tree(tv2);
    TransformPropagatorWithCheck tp(tv2);
    tree.traverse(&tp);
    TORCH_CHECK(tv2->nDims() == 2);
    TORCH_CHECK(tv1->nDims() == 2);
    TORCH_CHECK(tv0->nDims() == 2);
  }

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  inlineMost();

  TORCH_CHECK(tv1->getComputeAtPosition() == 2);
  TORCH_CHECK(tv2->getComputeAtPosition() == 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1024}, options);
  auto t1 = t0.squeeze(0);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  testValidate(&fusion, cg_outputs, {t0}, {t1}, __LINE__, __FILE__);
}

// HuggingFace repro:
// https://github.com/csarofeen/pytorch/issues/2064
TEST_F(NVFuserTest, FusionHuggingFaceRepro2064_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});
  auto tv2 = mul(tv1, IrBuilder::create<Double>(0.5));
  auto tv3 = mul(tv1, IrBuilder::create<Double>(0.707107));
  auto tv4 = erf(tv3);
  auto tv5 = add(tv4, IrBuilder::create<Double>(1.0));
  auto tv6 = mul(tv2, tv5);
  auto tv7 = sum(tv6, {0});

  fusion.addOutput(tv1);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 8}, options);
  auto t1 = t0.expand({1, 2, 8});
  auto t2 = t1 * 0.5;
  auto t5 = (t1 * 0.707107).erf() + 1.0;
  auto t6 = t2 * t5;
  auto t7 = t6.sum(0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0},
      {t1, t7},
      __LINE__,
      __FILE__,
      "");
}

#ifndef USE_ROCM

TEST_F(NVFuserTest, FusionCastings_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 4, y = 1024;

  std::vector<DataType> data_types{
      DataType::Double,
      DataType::Float,
      DataType::Half,
      DataType::Int,
      DataType::Int32,
      DataType::Bool,
      DataType::ComplexFloat,
      DataType::ComplexDouble};

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (at::cuda::getDeviceProperties(0)->major >= 8) {
    data_types.emplace_back(DataType::BFloat16);
  }
#endif

  for (const auto& input_type : data_types) {
    auto tv_in = makeContigTensor(2, input_type);
    fusion.addInput(tv_in);
    for (const auto& output_type : data_types) {
      auto tv_out = castOp(output_type, tv_in);
      fusion.addOutput(tv_out);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<c10::IValue> inputs;
  std::vector<at::Tensor> outputs;
  for (const auto& input_type : data_types) {
    at::Tensor t = at::randn({x, y}, options).to(data_type_to_aten(input_type));
    inputs.emplace_back(t);
    for (const auto& output_type : data_types) {
      outputs.emplace_back(t.to(data_type_to_aten(output_type)));
    }
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(inputs);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      inputs,
      outputs,
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionIssue2074_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 4, y = 1024;

  auto tv0 = makeContigTensor(2, DataType::Int32);
  fusion.addInput(tv0);
  auto tv1 = ne(tv0, IrBuilder::create<Int>(0));
  auto tv2 = castOp(DataType::Int32, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = sub(tv3, IrBuilder::create<Int>(1));
  fusion.addOutput(tv0);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x, y}, options).to(at::kInt);
  auto t1 = t0.ne(0);
  auto t2 = t1.to(at::kInt);
  auto t3 = t2.sum({1});
  auto t4 = t3 - 1;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  ASSERT_TRUE(at::allclose(cg_outputs[1], t4));
}

TEST_F(NVFuserTest, FusionIssue2077_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv3 = mul(tv1, IrBuilder::create<Int>(1));
  auto tv5 = sub(IrBuilder::create<Double>(1.), tv3);
  auto tv6 = castOp(DataType::Half, tv5);
  auto tv7 = castOp(DataType::Bool, tv6);

  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2, 4, 6}, options);
  auto t1 = t0.to(at::kFloat);
  auto t3 = t1 * 1;
  auto t5 = 1 - t3;
  auto t6 = t5.to(at::kHalf);
  auto t7 = t6.to(at::kBool);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  ASSERT_TRUE(at::equal(cg_outputs[0], t7));
}

#endif

TEST_F(NVFuserTest, FusionIssue2372_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tx = makeContigTensor(5, DataType::Float);
  fusion.addInput(tx);
  auto tmean = makeContigTensor(1, DataType::Float);
  fusion.addInput(tmean);
  auto tvar = makeContigTensor(1, DataType::Float);
  fusion.addInput(tvar);
  auto seps = IrBuilder::newScalar(DataType::Double);
  fusion.addInput(seps);

  auto tmean_bcast = broadcast(tmean, {true, true, true, true, false});
  auto tmean_expand = expand_as(tmean_bcast, tx);
  auto diff = sub(tx, tmean_expand);
  auto regvar = add(tvar, seps);
  auto invstd = rsqrt(regvar);
  auto invstd_bcast = broadcast(invstd, {true, true, true, true, false});
  auto invstd_expand = expand_as(invstd_bcast, tx);
  auto x_normed = mul(diff, invstd_expand);

  fusion.addOutput(x_normed);
  // This output is not necessary for a normalization function, but should not
  // cause compilation to fail
  fusion.addOutput(tmean); // Contiguous even-size input added as output
  fusion.addOutput(invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::manual_seed(0);

  int C = 2;
  at::Tensor x = at::randn({1, 5, 5, 5, C}, options);
  at::Tensor mean = at::randn({C}, options);
  at::Tensor var = at::rand({C}, options);
  double eps = 1e-5;

  std::vector<c10::IValue> inputs = {x, mean, var, eps};

  auto lparams = schedulePointwise(&fusion, inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs, lparams);
  auto cg_outputs = fe.runFusion(inputs, lparams);

  auto eager_diff = x - mean.view({1, 1, 1, 1, -1});
  auto eager_invstd = at::rsqrt(var + eps);
  auto eager_x_normed = eager_diff * eager_invstd.view({1, 1, 1, 1, -1});

  // testValidate currently fails since cg_outputs[1] is an empty tensor
  ASSERT_TRUE(at::allclose(cg_outputs[0], eager_x_normed));
  // ASSERT_TRUE(at::equal(cg_outputs[1], mean));
  ASSERT_TRUE(at::allclose(cg_outputs[2], eager_invstd));
}

TEST_F(NVFuserTest, FusionIssue2075_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 2, y = 128, z = 128;

  auto tv0 = makeContigConcreteTensor({1, -1, 1});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({1, 1, -1});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = expand(
      tv2,
      {IrBuilder::create<Int>(x),
       tv2->axis(1)->extent(),
       IrBuilder::create<Int>(z)});

  // [1, 1, 128] -> [1, 1, 1, 1, 1, 128]
  auto tv4 = broadcast(tv1, {{false, false, true, true, true, false}});
  // [1, 1, 1, 1, 1, 128] -> [2, 128, 1, 1, 1, 128]
  auto tv5 = expand(
      tv4,
      {IrBuilder::create<Int>(x),
       IrBuilder::create<Int>(y),
       tv4->axis(2)->extent(),
       tv4->axis(3)->extent(),
       tv4->axis(4)->extent(),
       tv4->axis(5)->extent()});
  auto tv6 = set(tv5);
  // [2, 128, 1, 1, 1, 128] -> [2, 1, 128, 1, 1, 128]
  auto tv7 = permute(tv6, {0, 3, 1, 2, 4, 5});
  auto tv8 = sum(tv7, {1, 3, 4});
  auto tv9 = le(tv8, tv3);
  auto tv10 = castOp(DataType::Float, tv9);
  fusion.addOutput(tv10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({1, y, 1}, options);
  at::Tensor t1 = at::randn({1, 1, z}, options);
  auto t3 = t0.expand({x, y, z});
  auto t4 = t1.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2);
  auto t5 = t4.expand({x, y, 1, 1, 1, z});
  auto t7 = t5.permute({0, 3, 1, 2, 4, 5});
  auto t8 = t7.squeeze(-2).squeeze(-2).squeeze(-3);
  auto t9 = t8 < t3;
  auto t10 = t9.to(at::kFloat);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(&fusion, cg_outputs, {t0, t1}, {t10}, __LINE__, __FILE__);
}

// Simple test of propagating vectorize predicates through the Exact
// CA map
TEST_F(NVFuserTest, FusionPropagateVectorizePredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);

  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  const int vec_factor = 4;
  tv1->split(-1, vec_factor);

  MaxRootDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->setMemoryType(MemoryType::Shared);

  // The predicate tv2 should look like (i * 4) + j < tv0.extent(0),
  // where i and j are the loop indices of the two leaf axes,
  // respectively. PredChecker checks if the second loop index is
  // indeed used in the predicate of tv2.

  class PredChecker : public kir::IrVisitor {
   public:
    PredChecker(bool vectorized) : vectorized_(vectorized) {}

    using kir::IrVisitor::handle;

    void handle(LoadStoreOp* ldst) final {
      if (ldst->out()->as<kir::TensorIndex>()->view()->name() == 2) {
        // Make sure the index of the inner loop isn't used in the
        // predicate of the tv2 expression
        TORCH_INTERNAL_ASSERT(!scope_exprs_.empty());
        TORCH_INTERNAL_ASSERT(scope_exprs_.back()->isA<kir::IfThenElse>());
        auto ite = scope_exprs_.back()->as<kir::IfThenElse>();
        auto cond = ite->predicate()->value();
        // Make sure the index of the inner loop isn't used in the predicate
        TORCH_INTERNAL_ASSERT(!for_loops_.empty());
        auto loop_index = for_loops_.back()->index();
        auto cond_inputs = InputsOf::output(cond->fusion(), cond);
        auto index_it =
            std::find(cond_inputs.begin(), cond_inputs.end(), loop_index);
        auto vec_factor_it =
            std::find_if(cond_inputs.begin(), cond_inputs.end(), [](Val* inp) {
              auto int_val = inp->getInt();
              return int_val.has_value() &&
                  (int_val.value() == vec_factor - 1 ||
                   int_val.value() == -(vec_factor - 1));
            });
        // If vectorized, the predicate should use (vec_factor - 1) or
        // -(vec_factor - 1) rather than the loop index.
        if (vectorized_) {
          TORCH_CHECK(
              index_it == cond_inputs.end(),
              "Not expected to have ",
              loop_index->toInlineString(),
              " in ",
              cond->toInlineString());
          TORCH_CHECK(
              vec_factor_it != cond_inputs.end(),
              "Expected to have ",
              vec_factor - 1,
              " in ",
              cond->toInlineString());
        } else {
          TORCH_CHECK(
              index_it != cond_inputs.end(),
              "Expected to have ",
              loop_index->toInlineString(),
              " in ",
              cond->toInlineString());
          TORCH_CHECK(
              vec_factor_it == cond_inputs.end(),
              "Not expected to have ",
              vec_factor - 1,
              " in ",
              cond->toInlineString());
        }
      }
    }

    bool vectorized_ = false;
  };

  GpuLower gpulw_wo_vec(&fusion);
  PredChecker(false).handle(gpulw_wo_vec.kernel()->topLevelExprs());

  // Vectorize the second axis of tv1
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // Now, the predicate tv2 should look like (i * 4) + 3 <
  // tv0.extent(0), i.e., j should be replaced with 3 since the second
  // axis is exactly mapped with the vectorized axis of tv1. It is
  // sufficient to check the condition using the last value of j,
  // i.e., 3.

  GpuLower gpulw_w_vec(&fusion);
  PredChecker(true).handle(gpulw_w_vec.kernel()->topLevelExprs());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  TORCH_CHECK(t0.equal(cg_outputs[0]));
}

TEST_F(NVFuserTest, FusionSqueezeOnlyWelford_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({-1, -1, 1, 1, 1});
  fusion.addInput(tv0);

  // welford with squeeze and reduction
  auto w1 = Welford(tv0, {1, 2, 3, 4});
  // welford with only squeeze
  auto w2 = Welford(tv0, {2, 3, 4});
  // feed w2 to a new welfword
  auto new_result_tv = [&](DataType dtype) -> TensorView* {
    auto dim0 = IterDomainBuilder(w1.avg->axis(0)).build();
    auto dim1 = IterDomainBuilder(w1.avg->axis(1)).build();
    auto td = IrBuilder::create<TensorDomain>(
        std::vector<IterDomain*>{dim0, dim1},
        std::vector<std::optional<bool>>{true, std::nullopt});
    auto tv = IrBuilder::create<TensorView>(td, dtype);
    return tv;
  };
  auto avg = new_result_tv(DataType::Float);
  auto var_sum = new_result_tv(DataType::Float);
  auto n = new_result_tv(DataType::Index);
  IrBuilder::create<WelfordOp>(
      avg,
      var_sum,
      n,
      w2.avg,
      w2.var_sum,
      w2.n,
      IrBuilder::create<Double>(0),
      IrBuilder::create<Double>(0),
      fusion.zeroVal());

  fusion.addOutput(w1.avg);
  fusion.addOutput(w1.var_sum);
  fusion.addOutput(w1.n);
  fusion.addOutput(avg);
  fusion.addOutput(var_sum);
  fusion.addOutput(n);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({10, 4, 1, 1, 1}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  ASSERT_TRUE(at::allclose(cg_outputs[0], cg_outputs[3]));
  ASSERT_TRUE(at::allclose(cg_outputs[1], cg_outputs[4]));
  ASSERT_TRUE(at::allclose(cg_outputs[2], cg_outputs[5]));
}

TEST_F(NVFuserTest, FusionIssue2163ReproInvalidAlias_CUDA) {
  int64_t N = 10, C = 16;

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());

  // setup fusion
  auto input = makeConcreteTensor({N, C});
  auto weight = makeConcreteTensor({C});
  fusion_ptr->addInput(input);
  fusion_ptr->addInput(weight);

  // This seems to confuse the alias analysis
  auto weight_copy1 = set(weight);
  auto weight_copy2 = set(weight_copy1);

  auto input_sum = sum(input, {0});
  auto sub_bcast = broadcast(input_sum, {true, false});
  auto input_sub_sum = sub(input, sub_bcast);
  auto weight_bcast = broadcast(weight_copy2, {true, false});
  auto output = mul(input_sub_sum, weight_bcast);
  fusion_ptr->addOutput(output);

  output->cacheBefore();

  auto ref = input;
  ref->split(-1, 8);
  ref->reorder({{0, 1}, {1, 0}, {2, 2}});
  TransformPropagator propagator(ref);
  MaxRootDomainInfoSpanningTree(ref).traverse(&propagator);

  // Don't inline the innermost axes
  std::unordered_set<IterDomain*> uninlinable;
  uninlinable.insert(output->axis(-1));
  uninlinable.insert(weight_copy1->axis(-1));

  inlineMost(uninlinable);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);

  auto at_input = at::randn({N, C}, options_float);
  auto at_weight = at::randn({C}, options_float);

  std::vector<c10::IValue> aten_inputs({at_input, at_weight});

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);
  auto cg_output = cg_outputs.at(0);

  auto ref_x_sub_mean = at_input - at_input.sum({0}).unsqueeze(0);
  auto ref_y = ref_x_sub_mean * at_weight.unsqueeze(0);

  testValidate(
      fe.kernel(), {cg_output}, aten_inputs, {ref_y}, __LINE__, __FILE__, "");
}

// Testing scalar FP types
TEST_F(NVFuserTest, FusionFloatingPointType_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const float float_val = 0.1f;
  const double double_val = 0.2;

  {
    auto tv0 = makeConcreteTensor({2}, DataType::Float);
    fusion.addInput(tv0);

    auto f2 = IrBuilder::create<Double>(float_val, DataType::Float);
    TORCH_CHECK(
        f2->getDataType() == DataType::Float,
        "Invalid data type: ",
        f2->getDataType().value());

    auto d3 = IrBuilder::create<Double>(double_val, DataType::Double);
    TORCH_CHECK(
        d3->getDataType() == DataType::Double,
        "Invalid data type: ",
        d3->getDataType().value());

    // Adding two Floats produces a Float
    auto f4 = add(f2, f2);
    TORCH_CHECK(
        f4->getDataType() == DataType::Float,
        "Invalid data type: ",
        f4->getDataType().value());

    // Adding a Double and a Float produces a Double
    auto d5 = add(f2, d3);
    TORCH_CHECK(
        d5->getDataType() == DataType::Double,
        "Invalid data type: ",
        d5->getDataType().value());

    // Adding a Float and a Double produces a Double
    auto d6 = add(d3, f2);
    TORCH_CHECK(
        d6->getDataType() == DataType::Double,
        "Invalid data type: ",
        d6->getDataType().value());

    // Adding two Doubles produce a Double
    auto d7 = add(d5, d6);
    TORCH_CHECK(
        d7->getDataType() == DataType::Double,
        "Invalid data type: ",
        d7->getDataType().value());

    // Adding a Float to a Float tensor produces a Float tensor
    auto tv1 = add(tv0, f4);
    TORCH_CHECK(
        tv1->getDataType() == DataType::Float,
        tv1->toString(),
        " has an invalid data type: ",
        tv1->getDataType().value());

    // Adding a Double to a Float tensor still produces a Float tensor
    auto tv2 = add(tv1, d7);
    TORCH_CHECK(
        tv2->getDataType() == DataType::Float,
        tv2->toString(),
        " has an invalid data type: ",
        tv2->getDataType().value());

    fusion.addOutput(tv2);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2}, options);

  std::vector<c10::IValue> inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);

  auto f2 = float_val;
  auto d3 = double_val;
  auto f4 = f2 + f2;
  auto d5 = f2 + d3;
  auto d6 = d3 + f2;
  auto d7 = d5 + d6;
  auto t1 = t0 + f4;
  auto t2 = t1 + d7;

  testValidate(&fusion, cg_outputs, inputs, {t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIntegerType_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t int64_val = 1;
  const int int_val = 2;

  {
    auto tv0 = makeConcreteTensor({10}, DataType::Int32);
    fusion.addInput(tv0);

    auto i2 = IrBuilder::create<Int>(int64_val, DataType::Int);
    auto i3 = IrBuilder::create<Int>(int_val, DataType::Int32);

    // Adding two Ints produces an Int
    auto i4 = add(i2, i2);
    TORCH_CHECK(
        i4->getDataType() == DataType::Int,
        "Invalid result: ",
        i4->toInlineString());

    // Adding two Int32s produces an Int32
    auto i5 = add(i3, i3);
    TORCH_CHECK(
        i5->getDataType() == DataType::Int32,
        "Invalid result: ",
        i5->toInlineString());

    // Adding an Int and an Int32 produces an Int
    auto i6 = add(i4, i5);
    TORCH_CHECK(
        i6->getDataType() == DataType::Int,
        "Invalid result: ",
        i6->toInlineString());

    // Adding an Int32 to an Int32 tensor produces an Int32 tensor
    auto tv1 = add(tv0, i4);
    TORCH_CHECK(
        tv1->getDataType() == DataType::Int32,
        tv1->toString(),
        " has an invalid data type: ",
        tv1->getDataType().value());

    // Adding an Int to an Int32 tensor still produces an Int32 tensor
    auto tv2 = add(tv1, i6);
    TORCH_CHECK(
        tv2->getDataType() == DataType::Int32,
        tv2->toString(),
        " has an invalid data type: ",
        tv2->getDataType().value());

    fusion.addOutput(tv2);
  }

  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(10, {10}, options);

  std::vector<c10::IValue> inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);

  auto i2 = int64_val;
  auto i3 = int_val;
  auto i4 = i2 + i2;
  auto i5 = i3 + i3;
  auto i6 = i4 + i5;
  auto t1 = t0 + i4;
  auto t2 = t1 + i6;

  TORCH_CHECK(cg_outputs.at(0).equal(t2));
}

TEST_F(NVFuserTest, FusionVectorizeWelford1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({7, 32});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tvs = Welford(tv1, {0});
  fusion.addOutput(tvs.avg);
  fusion.addOutput(tvs.var_sum);
  fusion.addOutput(tvs.n);

  tv1->split(1, 4);

  MaxRootDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->computeWith(-1, true);

  GpuLower gpulw(&fusion);
  auto all_exprs = KernelExprVisitor::getAllExprs(gpulw.kernel());
  auto num_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<WelfordOp>();
      });
  TORCH_CHECK(
      num_welford_ops == 0,
      "All WelfordOp exprs should be converted to VectorizedWelfordOp");

  auto num_vectorized_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<kir::VectorizedWelfordOp>();
      });
  TORCH_CHECK(
      num_vectorized_welford_ops == 1,
      "There must be two VectorizedWelfordOp exprs");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref_avg = t0.mean({0});
  auto ref_var = t0.var({0}, false) * shape[0];
  auto ref_N = at::ones({shape[1]}, options_int) * shape[0];

  testValidate(
      fe.kernel(),
      cg_outputs,
      {t0},
      {ref_avg, ref_var, ref_N},
      __LINE__,
      __FILE__);
}

// Unswitched welford
TEST_F(NVFuserTest, FusionVectorizeWelford2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({7, 32});

  auto tv0 = makeContigConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tvs = Welford(tv1, {0});
  fusion.addOutput(tvs.avg);
  fusion.addOutput(tvs.var_sum);
  fusion.addOutput(tvs.n);

  tv1->split(1, 4);
  tv1->split(0, 5);
  tv1->split(0, 1);

  tv1->reorder({{-2, 1}});

  MaxRootDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->computeAt(tvs.avg, 3);
  tvs.avg->axis(2)->parallelize(ParallelType::Unswitch);

  tv1->computeWith(-1, true);

  GpuLower gpulw(&fusion);
  auto all_exprs = KernelExprVisitor::getAllExprs(gpulw.kernel());
  auto num_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<WelfordOp>();
      });
  TORCH_CHECK(
      num_welford_ops == 0,
      "All WelfordOp exprs should be converted to VectorizedWelfordOp");

  auto num_vectorized_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<kir::VectorizedWelfordOp>();
      });
  TORCH_CHECK(
      num_vectorized_welford_ops == 2,
      "There must be two VectorizedWelfordOp exprs");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn(shape, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});
  auto cg_outputs = fe.runFusion({t0});

  auto ref_avg = t0.to(at::kDouble).mean({0});
  auto ref_var = t0.to(at::kDouble).var({0}, false) * shape[0];
  auto ref_N = at::ones({shape[1]}, options_int) * shape[0];

  testValidate(
      fe.kernel(),
      cg_outputs,
      {t0},
      {ref_avg, ref_var, ref_N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionRepro2241_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  {
    TensorView* t6 = makeContigConcreteTensor({1}, DataType::Int);
    TensorView* t15 = makeContigConcreteTensor({3, 2, 1, 2}, DataType::Double);
    TensorView* t20 = makeContigConcreteTensor({1, 1, 1, 1}, DataType::Int);
    fusion->addInput(t6);
    fusion->addInput(t15);
    fusion->addInput(t20);
    auto sample_total = sum(t15, {0, 1, 2, 3}, true);
    auto sample_mean = div(sample_total, t20);
    auto x = sub(t15, sample_mean);
    auto input = mul(x, x);
    auto total = sum(input, {0, 1, 2, 3});
    auto t7 = div(total, t6);
    fusion->addOutput(t7);
  }

  FusionExecutorCache fec(std::move(fusion_ptr));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t6 = at::tensor({15}, options.dtype(at::kLong));
  at::Tensor t15 = at::randn({3, 2, 1, 2}, options.dtype(at::kDouble));
  at::Tensor t20 =
      at::tensor({12}, options.dtype(at::kLong)).expand({1, 1, 1, 1});

  auto cg_outputs = fec.runFusionWithInputs({t6, t15, t20});

  auto sample_total = at::sum(t15, {0, 1, 2, 3}, true);
  auto sample_mean = at::div(sample_total, t20);
  auto x = at::sub(t15, sample_mean);
  auto input = at::mul(x, x);
  auto total = at::sum(input, {0, 1, 2, 3}, false);
  auto t7 = at::div(total, t6);

  testValidate(
      fec.fusion(), cg_outputs, {t6, t15, t20}, {t7}, __LINE__, __FILE__);
}

// https://github.com/csarofeen/pytorch/issues/2321
TEST_F(
    NVFuserTest,
    FusionPersistentBufferProjectionAfterWelfordTranslate_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  DataType dtype = DataType::Half;
  constexpr int64_t dim0 = 2048;
  constexpr int64_t dim1 = 10240;
  std::vector<int64_t> input_shape{dim0, dim1};
  std::vector<int64_t> norm_shape{dim1};
  auto input_half = makeContigTensor(2, dtype);
  auto weight_half = makeContigTensor(1, dtype);
  auto bias_half = makeContigTensor(1, dtype);
  fusion.addInput(input_half);
  fusion.addInput(weight_half);
  fusion.addInput(bias_half);
  auto input = castOp(DataType::Float, input_half);
  auto weight = castOp(DataType::Float, weight_half);
  auto bias = castOp(DataType::Float, bias_half);
  auto result = layer_norm(input, norm_shape, weight, bias, eps_ptr);
  auto result_output = castOp(dtype, result.output);
  fusion.addOutput(result_output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = at::randn({input_shape[1]}, options);
  c10::optional<at::Tensor> aten_bias = at::randn({input_shape[1]}, options);
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  // welford translate
  KernelArgumentHolder runtime_inputs =
      KernelArgumentHolder::createKernelArgumentHolder(
          {aten_input, aten_weight, aten_bias});
  bool isTranslated =
      SegmentCandidateFinder::translateWelfordInFusion(&fusion, runtime_inputs);
  TORCH_INTERNAL_ASSERT(isTranslated);

  // persistent buffer should be projected to input
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  TORCH_CHECK(
      persistent_buffer_info.projectable_persistent_buffers.size() == 1,
      "should have only one projectable_persistent_buffer!");
  TORCH_CHECK(
      persistent_buffer_info.projectable_buffer_inputs.size() == 1,
      "should have only one projectable_buffer_inputs!");
  TORCH_CHECK(
      persistent_buffer_info.projectable_buffer_inputs[0] == input_half,
      "persistent buffer should be projected to input!");

  // Check reduction axis is same for all reductions
  // Generate Launch Parameters
  auto reduction_params =
      getPersistentHeuristics(&fusion, {aten_input, aten_weight, aten_bias});
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs =
      fec.runFusionWithInputs({aten_input, aten_weight, aten_bias});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, aten_weight, aten_bias},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__,
      "");
}

TEST_F(NVFuserTest, FusionExprSortMatmulLikeSchedule_CUDA) {
  // See https://github.com/csarofeen/pytorch/pull/2366
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M1 = 5, M2 = 5, N1 = 6, N2 = 6, K1 = 2, K2 = 2;

  auto tv0 = makeContigConcreteTensor({M1, M2, K1, K2}, DataType::Double);
  auto tv1 = makeContigConcreteTensor({N1, N2, K1, K2}, DataType::Double);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true, false, true, false, false});
  auto tv3 = broadcast(tv1, {true, false, true, false, false, false});
  auto tv4 = mul(tv2, tv3);
  auto tv5 = sum(tv4, {-1, -2});
  fusion.addOutput(tv5);

  auto tv6 = tv0->cacheAfter();
  auto tv7 = tv1->cacheAfter();
  auto tv8 = tv6->cacheAfter();
  auto tv9 = tv7->cacheAfter();
  auto tv10 = tv5->cacheBefore();

  tv6->inlineAt(3);
  tv7->inlineAt(3);
  tv8->inlineAt(4);
  tv9->inlineAt(4);
  tv2->inlineAt(6);
  tv3->inlineAt(6);
  tv4->inlineAt(6);
  tv10->inlineAt(4);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M1, M2, K1, K2}, options);
  at::Tensor t1 = at::randn({N1, N2, K1, K2}, options);
  auto expect = at::mm(
                    t0.view({(int64_t)(M1 * M2), (int64_t)(K1 * K2)}),
                    t1.view({(int64_t)(N1 * N2), (int64_t)(K1 * K2)}).t())
                    .view({M1, M2, N1, N2})
                    .transpose(1, 2);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});
  auto cg_outputs = fe.runFusion({t0, t1});

  testValidate(fe.kernel(), cg_outputs, {t0, t1}, {expect}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionFloatConstantWhere_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1, DataType::Bool);
  fusion.addInput(tv0);

  auto tv1 = where(
      tv0,
      IrBuilder::create<Double>(3.0, DataType::Float),
      IrBuilder::create<Double>(5.0, DataType::Float));

  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::arange(4, options) > 1.0;

  std::vector<c10::IValue> inputs = {t0};

  auto lparams = schedulePointwise(&fusion, inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs, lparams);
  auto cg_outputs = fe.runFusion(inputs, lparams);

  auto ref = at::where(t0, (float)3.0, (float)5.0);

  // testValidate does not check that dtypes match
  TORCH_CHECK(cg_outputs[0].dtype() == ref.dtype());
  testValidate(&fusion, cg_outputs, inputs, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCpAsyncCommitWait_CUDA) {
  // Repro for https://github.com/csarofeen/pytorch/issues/2463
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({12800, 8, 8, 8}, DataType::Half);
  auto tv1 = set(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::TIDx);

  auto tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsyncCa);
  tv2->axis(-1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);

  tv2->inlineAt(1);
  tv2->circularBuffer(8);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({12800, 8, 8, 8}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0});

  auto cg_outputs = fe.runFusion({t0});
  testValidate(fe.kernel(), cg_outputs, {t0}, {t0}, __LINE__, __FILE__);
}

// Repro of issue #2459
TEST_F(NVFuserTest, FusionClearThreadPredicateByRAWSync_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  // test with gmem
  auto tv4 = sum(tv0, {1});
  auto tv5 = set(tv4);
  auto tv6 = set(tv5);
  fusion.addOutput(tv6);

  // tv1 is predicated with tidx
  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  // Upload to shmem. Still predicated with tidx, so only the threads
  // with tidx == 0 should be active.
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);

  // Remap the parallelization from tidy to tidx. This should work as
  // tv2 is in shared memory and SyncMap should correctly insert a RAW
  // sync between tv2 and tv3. However, ThreadPredicateMap still marks
  // tv3 as predicated by tidx, and since it is invalid to parallelize
  // by a predicated parallel type, this resulted in an error (#2459).
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  // Test with gmem
  tv4->split(0, 4);
  tv5->split(0, 4);
  tv6->split(0, 4);

  // Make tv4 predicated with tidx
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  // Upload to gmem
  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::TIDy);
  tv5->setMemoryType(MemoryType::Global);

  // RAW sync should be inserted after tv5

  tv6->axis(0)->parallelize(ParallelType::BIDy);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({10, 11}, options);

  std::vector<c10::IValue> inputs = {t0};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);

  auto t3 = t0.sum({1}).sum({0});
  auto t6 = t0.sum({1});

  testValidate(fe.kernel(), cg_outputs, inputs, {t3, t6}, __LINE__, __FILE__);
}

namespace {

class ThreadPredChecker : public kir::IrVisitor {
 public:
  static bool isPredicatedBy(
      StmtNameType tv_name_to_check,
      ParallelTypeBitmap pt_map,
      kir::Kernel* kernel) {
    ThreadPredChecker checker(tv_name_to_check, pt_map);
    checker.handle(kernel->topLevelExprs());
    return checker.pt_map_.none();
  }

  ThreadPredChecker(StmtNameType tv_name_to_check, ParallelTypeBitmap pt_map)
      : tv_name_to_check_(tv_name_to_check), pt_map_(pt_map) {}

  using kir::IrVisitor::handle;

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      auto tv_output = ir_utils::getTvOutput(expr);
      if (tv_output != nullptr && tv_output->name() == tv_name_to_check_ &&
          expr->isA<LoadStoreOp>() && ite->predicate()->hasValue()) {
        handle(ite->predicate()->value());
      }
    }
  }

  void handle(Bool* val) final {
    if (val->definition()) {
      handle(val->definition());
    }
  }

  void handle(BinaryOp* bop) final {
    if (bop->getBinaryOpType() == BinaryOpType::And) {
      handle(bop->lhs());
      handle(bop->rhs());
    } else if (bop->getBinaryOpType() == BinaryOpType::Eq) {
      if (bop->lhs()->isZeroInt() || bop->rhs()->isZeroInt()) {
        auto non_zero_arg = bop->lhs()->isZeroInt() ? bop->rhs() : bop->lhs();

        // It can be changed like (-threadIdx.x) by expr simplifier
        if (auto uop = dynamic_cast<UnaryOp*>(non_zero_arg->definition())) {
          if (uop->getUnaryOpType() == UnaryOpType::Neg) {
            non_zero_arg = uop->in();
          }
        }

        if (auto ns = dynamic_cast<NamedScalar*>(non_zero_arg)) {
          if (ns->getParallelIndex().has_value()) {
            auto predicated_type = ns->getParallelIndex().value();
            pt_map_.clear(predicated_type);
          }
        }
      }
    }
  }

 private:
  StmtNameType tv_name_to_check_;
  ParallelTypeBitmap pt_map_;
};

} // namespace

// Repro of issue #2487
TEST_F(NVFuserTest, FusionPredicateReductionInitShared_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(1);
  fusion.addInput(tv3);

  auto tv4 = exp(tv3);
  fusion.addOutput(tv4);

  tv1->setMemoryType(MemoryType::Shared);

  tv4->split(0, 1024);
  tv4->axis(-2)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  // tv4 is parallelized with both BIDx and TIDx, but tv1 is not at
  // all, so tv1 is predicated with both BIDx and TIDx as they are
  // redundant. That means that the initialization of the reduction
  // has to be predicated as well. Since tv1 is on shared memory, only
  // the TIDx predicate is required.

  // Make sure the initialization of tv1 is predicated with
  // threadIdx.x == 0
  GpuLower gpulw(&fusion);
  ParallelTypeBitmap predicated_types(ParallelType::TIDx);
  TORCH_CHECK(
      ThreadPredChecker::isPredicatedBy(
          tv1->name(), predicated_types, gpulw.kernel()),
      "Validation of lowered kernel failed");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({2}, options);
  at::Tensor t1 = at::randn({10000}, options);

  std::vector<c10::IValue> inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);

  auto ref_t1 = t0.sum({0});
  auto ref_t4 = t1.exp();

  testValidate(
      fe.kernel(), cg_outputs, inputs, {ref_t1, ref_t4}, __LINE__, __FILE__);
}

// Repro of issue #2487
TEST_F(NVFuserTest, FusionPredicateReductionInitGlobal_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape({100});

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = exp(tv2);
  fusion.addOutput(tv3);

  tv3->split(0, 32);
  tv3->axis(-2)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // tv3 is parallelized with both BIDx and TIDx, but tv1 is not at
  // all, so tv1 is predicated with both BIDx and TIDx as they are
  // redundant. That means that the initialization of the reduction
  // has to be predicated as well.

  // Make sure the initialization of tv1 is predicated with
  // threadIdx.x == 0 and blockIdx.x == 0
  GpuLower gpulw(&fusion);
  ParallelTypeBitmap predicated_types({ParallelType::TIDx, ParallelType::BIDx});
  TORCH_CHECK(
      ThreadPredChecker::isPredicatedBy(
          tv1->name(), predicated_types, gpulw.kernel()),
      "Validation of lowered kernel failed");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({2}, options);
  at::Tensor t1 = at::randn({10000}, options);

  std::vector<c10::IValue> inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);

  auto ref_t1 = t0.sum({0});
  auto ref_t3 = t1.exp();

  testValidate(
      fe.kernel(), cg_outputs, inputs, {ref_t1, ref_t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTypePromotionATenConsistency_CUDA) {
  auto convertible_to_aten = {
      DataType::Bool,
      DataType::Double,
      DataType::Float,
      DataType::Half,
      DataType::BFloat16,
      DataType::Int,
      DataType::Int32,
      DataType::ComplexFloat,
      DataType::ComplexDouble};
  for (auto t1 : convertible_to_aten) {
    for (auto t2 : convertible_to_aten) {
      auto t1_aten = data_type_to_aten(t1);
      auto t2_aten = data_type_to_aten(t2);
      auto result_aten = c10::promoteTypes(t1_aten, t2_aten);
      auto result = promoteType(t1, t2);
      ASSERT_EQ(data_type_to_aten(result), result_aten);
    }
  }
}

// Make sure invalid usage of index type is detected
TEST_F(NVFuserTest, FusionCompileIndexType_CUDA) {
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeSymbolicTensor(1, DataType::Bool);
    fusion.addInput(tv0);

    auto tv2 = neg(tv0);
    fusion.addOutput(tv2);

    tv2->split(0, 256);
    tv2->split(0, 1024);

    MaxRootDomainInfoSpanningTree tree(tv2);
    TransformPropagator tp(tv2);
    tree.traverse(&tp);

    inlineMost();

    tv2->axis(1)->parallelize(ParallelType::BIDx);
    tv2->axis(2)->parallelize(ParallelType::TIDx);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::manual_seed(0);
    at::Tensor t0 = at::randn({999}, options).ge(0);
    std::vector<c10::IValue> small_inputs = {t0};

    at::Tensor t0_large =
        at::randn({std::numeric_limits<int>::max()}, options).ge(0);
    std::vector<c10::IValue> large_inputs = {t0_large};

    TORCH_CHECK(
        KernelArgumentHolder::createKernelArgumentHolder(large_inputs)
            .getSmallestIndexTypeOfArguments() == PrimDataType::Int);
    TORCH_CHECK(
        KernelArgumentHolder::createKernelArgumentHolder(small_inputs)
            .getSmallestIndexTypeOfArguments() == PrimDataType::Int32);

    {
      FusionExecutor fe;
      // Lower the kernel with large inputs and int64 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int};
      fe.compileFusion(&fusion, large_inputs, LaunchParams(), compile_opts);

      TORCH_CHECK(
          fe.kernel()->indexType() == PrimDataType::Int,
          "Unexpected kernel index type: ",
          fe.kernel()->indexType());

      // Since the index type is int64, both small and large inputs
      // should work fine
      fe.runFusion(small_inputs);
      fe.runFusion(large_inputs);
    }

    {
      FusionExecutor fe;
      // Lower the kernel with small inputs and int64 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int};
      fe.compileFusion(&fusion, small_inputs, LaunchParams(), compile_opts);

      TORCH_CHECK(
          fe.kernel()->indexType() == PrimDataType::Int,
          "Unexpected kernel index type: ",
          fe.kernel()->indexType());

      // Since the index type is int64, both small and large inputs
      // should work fine
      fe.runFusion(small_inputs);
      fe.runFusion(large_inputs);
    }

    {
      FusionExecutor fe;
      LaunchParams launch_params;
      CompileParams compile_opts = {.index_type = PrimDataType::Int32};
      fe.compileFusion(&fusion, small_inputs, launch_params, compile_opts);

      TORCH_CHECK(
          fe.kernel()->indexType() == PrimDataType::Int32,
          "Unexpected kernel index type: ",
          fe.kernel()->indexType());

      // This should complete successfully as the arguments are small
      // enough to use the int32 index type
      fe.runFusion(small_inputs);

      // This should fail as the Kernel is already compiled for Int32, but
      // the arguments are too large
      CompileParams compile_opts_large = {.index_type = PrimDataType::Int};
      EXPECT_THAT(
          [&]() {
            fe.runFusion(large_inputs, launch_params, compile_opts_large);
          },
          testing::ThrowsMessage<c10::Error>(testing::HasSubstr(
              "Kernel index type and compilation index type don't match")));
    }

    {
      FusionExecutor fe;
      // Lower the kernel with large inputs and int32 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int32};
      // This should fail due to the conflict
      EXPECT_THAT(
          [&]() {
            fe.compileFusion(
                &fusion, large_inputs, LaunchParams(), compile_opts);
          },
          testing::ThrowsMessage<c10::Error>(testing::HasSubstr(
              "Compilation with int32 is requested but int64 is required for the arguments")));
    }
  }

  c10::cuda::CUDACachingAllocator::emptyCache();
}

// Make sure the index type is determined both fusion inputs and outputs
TEST_F(NVFuserTest, FusionExecutorCacheIndexType1_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv1);

  auto tv2 = castOp(DataType::Float, tv0);
  auto tv3 = castOp(DataType::Float, tv1);
  auto tv4 = broadcast(tv2, {false, true, false});
  auto tv5 = broadcast(tv3, {true, false, false});
  auto tv6 = add(tv4, tv5);
  auto tv7 = castOp(DataType::Half, tv6);

  fusion.addOutput(tv7);

  c10::cuda::CUDACachingAllocator::emptyCache();

  // Inputs are small enough to use 32-bit indexing, but the output is
  // not
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2024, 1024}, options);
  at::Tensor t1 = at::randn({2024, 1024}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int);

  c10::cuda::CUDACachingAllocator::emptyCache();
}

// Make sure the index type is also determined by intermediate
// tensors. This is not ideal but just tests if the logic produces
// what is expected at this moment
TEST_F(NVFuserTest, FusionExecutorCacheIndexType2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true, false});
  auto tv3 = broadcast(tv1, {true, false, false});
  auto tv4 = add(tv2, tv3);
  auto tv5 = sum(tv4, {-1});

  fusion.addOutput(tv5);

  // Inputs and outputs are small enough to use 32-bit indexing,
  // however the intermediate, tv4, should cause the kernel to use
  // 64-bit indexing. This is not ideal as tv4 should be inlined, and
  // its allocation size should be small enough to use 32-bit
  // indexing. However, the current logic should result in forcing
  // 64-bit indexing. This would need to be fixed for matmul for
  // example.
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2024, 1024}, options);
  at::Tensor t1 = at::randn({2024, 1024}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  executor_cache.runFusionWithInputs(aten_inputs);
  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int);

  // Running again with forced type of Int32
  executor_cache.runFusionWithInputs(aten_inputs, PrimDataType::Int32);
  kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  TORCH_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int32);
}

//! Test whether we can create and use float16 scalars
TEST_F(NVFuserTest, FusionHalfScalars_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::Half);
  fusion->addInput(tv0);

  auto tv2 = full_like(tv0, IrBuilder::create<Double>(1.5, DataType::Half));
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::zeros({5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0},
      {at::ones_like(t0) * 1.5},
      __LINE__,
      __FILE__);
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
//! Test whether we can create and use BFloat16 scalars
TEST_F(NVFuserTest, FusionBFloat16Scalars_CUDA) {
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping BFloat16Scalars test on pre-AMPERE GPUs";
  }
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::BFloat16);
  fusion->addInput(tv0);

  auto tv2 = full_like(tv0, IrBuilder::create<Double>(1.5, DataType::BFloat16));
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::zeros({5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0},
      {at::ones_like(t0) * 1.5},
      __LINE__,
      __FILE__);
}
#endif

// Quick test of traversing attributes with IterVisitor
TEST_F(NVFuserTest, IterVisitorTraverseAttributes_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{IrBuilder::create<Int>(1),
        sub(tv0->axis(0)->extent(), IrBuilder::create<Int>(1))}});
  fusion.addOutput(tv1);

  auto tv1_resize = tv1->axis(0)->definition()->as<Resize>();

  auto stmts = StmtSort::getStmts(&fusion, true, true);

  // Make sure the expansion parameters of tv1_resize are visited
  TORCH_CHECK(
      std::find(stmts.begin(), stmts.end(), tv1_resize->leftExpand()) !=
          stmts.end(),
      "Resize left expand parameter not found");
  TORCH_CHECK(
      std::find(stmts.begin(), stmts.end(), tv1_resize->rightExpand()) !=
          stmts.end(),
      "Resize right expand parameter not found");
}

TEST_F(NVFuserTest, FusionManagedData_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2});
  auto tv1 = set(set(set(set(set(set(set(set(set(set(set(set(tv0))))))))))));
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  using T1 = std::vector<Val*>;
  T1 data1 = {tv0, tv1};

  struct T2 {
    Val* input;
    Val* output;
    size_t magic_number;
  } data2{tv0, tv1, 0x123456789abcdef};
  auto clone_fn = [](IrCloner& cloner, std::any data) -> std::any {
    auto d = std::any_cast<T2>(data);
    return T2{cloner.clone(d.input), cloner.clone(d.output), d.magic_number};
  };

  auto i1 = fusion.manage(data1);
  auto i2 = fusion.manage(data2, clone_fn);
  fusion.manage("data1", data1);
  fusion.manage("data2", data2, clone_fn);

  GpuLower lower(&fusion);
  auto kernel = lower.kernel();

  T1 expect1{kernel->inputs().at(0), kernel->outputs().at(0)};
  ASSERT_EQ(kernel->getManaged<T1>(i1), expect1);
  ASSERT_EQ(kernel->getManaged<T1>("data1"), expect1);
  ASSERT_EQ(kernel->getManaged<T2>(i2).input, kernel->inputs().at(0));
  ASSERT_EQ(kernel->getManaged<T2>(i2).output, kernel->outputs().at(0));
  ASSERT_EQ(kernel->getManaged<T2>("data2").input, kernel->inputs().at(0));
  ASSERT_EQ(kernel->getManaged<T2>("data2").output, kernel->outputs().at(0));
  ASSERT_EQ(kernel->getManaged<T2>("data2").magic_number, 0x123456789abcdef);
}

// Test for ir_utils::validateDomainEquivalence. We could consider
// it well tested as it's always used when TensorDomain is created, but
// here's some corner cases.
TEST_F(NVFuserTest, FusionDomainEquivalence_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [I0, I1]
  tv1->split(0, 4);
  // [I0/4, 4, I1]

  // Initial domain: root domain
  // Derived domain: [4, I1]
  // Should fail as the derived domain only partially covers the
  // root domain
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            tv1->getRootDomain(), {tv1->axis(1), tv1->axis(2)});
      },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Invalid derived domain")));

  tv1->merge(0);
  // [I0/4*4, I1]

  // Initial domain: root domain
  // Derived domain: leaf domain
  // Should succeed.
  ir_utils::validateDomainEquivalence(
      tv1->getRootDomain(), tv1->getLeafDomain());

  auto tv1_intermediate_id = tv1->axis(0);

  tv1->split(0, 3);
  // [I0/4*4/3, 3, I1]

  // Initial domain: root domain
  // Derived domain: leaf + tv1_intermediate_id
  // Should fail as the intermediate ID and the first two leaves are redundant
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            tv1->getRootDomain(),
            {tv1_intermediate_id, tv1->axis(0), tv1->axis(1), tv1->axis(2)});
      },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Invalid derived domain")));

  // Testing symbolic domains
  auto tv2 = reshape(tv0, {IrBuilder::create<Int>(), IrBuilder::create<Int>()});

  ir_utils::validateDomainEquivalence(
      tv2->getRootDomain(), tv2->getLeafDomain());

  // create a 2D tensor with one symbolid and another non-symbolic
  auto tv4 = broadcast(sum(tv2, {1}), {false, true});
  fusion.addOutput(tv4);

  // [S0, B0]
  tv4->split(1, 4);
  // [S0, B0/4, 4]

  ir_utils::validateDomainEquivalence(
      tv4->getRootDomain(), tv4->getLeafDomain());

  // Initial domain: root domain
  // Derived domain: [S0, B0/4]
  // Should fail as the derived domain only partially covers the
  // root domain
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            tv4->getRootDomain(), {tv4->axis(0), tv4->axis(1)});
      },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Invalid derived domain")));
}

// Repro for issue #236 (https://github.com/NVIDIA/Fuser/issues/236)
TEST_F(NVFuserTest, DoublePrecisionNorm_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  DataType dt = DataType::Float;

  auto tv0 = makeSymbolicTensor(1, dt);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(1, dt);
  fusion->addInput(tv1);

  auto tv2 = sum(tv1, {0});
  auto tv3 = broadcast(tv2, {true});
  auto tv4 = sub(tv1, tv3);
  auto tv5 = mul(tv4, tv0);
  fusion->addOutput(tv5);

  // The persistent scheduler with this problem size resulted in an
  // error as reported in #236
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dt)).device(at::kCUDA, 0);
  at::manual_seed(0);
  at::Tensor t0 = at::randn({11}, options);
  at::Tensor t1 = at::randn({11}, options);
  std::vector<c10::IValue> aten_inputs({t0, t1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  t1 = t1.to(at::kDouble);
  auto ref = (t1 - t1.sum().unsqueeze(0)) * t0;

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      aten_inputs,
      {ref},
      __LINE__,
      __FILE__);
}

// Test for void IterDomain::parallelize(ParallelType t)
// Only allowed to parallelize a leaf domain.
TEST_F(NVFuserTest, FusionIllegalParallelizeNonLeafDomain_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  // [I0, I1]
  tv1->split(1, 4);
  // [I0, I1/4, 4]

  const auto& root_domain = tv1->getRootDomain();

  // legal, as I0 is also a leaf domain
  root_domain[0]->parallelize(ParallelType::BIDx);

  // llegal, as I1 is not a leaf domain
  EXPECT_THAT(
      [&]() { root_domain[1]->parallelize(ParallelType::BIDy); },
      testing::ThrowsMessage<c10::Error>(
          testing::HasSubstr("Only allowed to parallelize a leaf domain")));
}

// delete intermediate tensors between segments to reduce memory usage of large
// segmented graphs
TEST_F(NVFuserTest, FusionClearGmemBetweenSegments_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> input_shape{32, 64, 8, 128};
  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Double>(1.0));
  auto tv2 = sum(tv1, {0}); // Group 0
  auto tv3 = sum(tv2, {-1}); // Group 1
  auto output = sum(tv3, {0}); // Group 2
  fusion->addOutput(output);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({at_x});
  auto t1 = at_x.add(1.0);
  auto t2 = t1.sum({0});
  auto t3 = t2.sum({-1});
  auto t4 = t3.sum({0});
  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  auto args_num = optimized_fusion->getArgsNumAfterSegmentRuns();

  TORCH_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen");
  TORCH_CHECK(
      optimized_fusion->fusionSegments()->groups().size() == 3,
      "segmentation didn't happen as expected");
  // group-0: tv1 -> tv2
  // group-1: tv2 -> tv3
  // group-2: tv3 -> tv4
  // -----------without args erase------------------------
  // after group-0, args: {t0, 32, 64, 8, 128, t2}
  // after group-1, args: {t0, 32, 64, 8, 128, t2, t3}
  // after group-2, args: {t0, 32, 64, 8, 128, t2, t3, t4}
  // -----------with args erase---------------------------
  // after group-0, args: {t0, 32, 64, 8, 128, t2}
  // after group-1, args: {t0, 32, 64, 8, 128, t3} (t2 is erased)
  // after group-2, args: {t0, 32, 64, 8, 128, t4} (t3 is erased)
  TORCH_CHECK(
      args_num[1] == args_num[0] && args_num[2] == args_num[0],
      "unused intermediate args should be deleted");
  testValidate(
      executor_cache.fusion(), outputs, {at_x}, {t4}, __LINE__, __FILE__);
}

// Test nan propagation during min/max with floats and doubles
TEST_F(NVFuserTest, FusionMinMaxNanPropagation_CUDA) {
  for (auto dtype : {DataType::Float, DataType::Double}) {
    for (auto do_min : {true, false}) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());

      auto tv0 = makeSymbolicTensor(2, dtype);
      fusion->addInput(tv0);
      auto tv1 = do_min ? min(tv0, {1}) : max(tv0, {1});
      fusion->addOutput(tv1);

      FusionExecutorCache executor_cache(std::move(fusion));

      auto options =
          at::TensorOptions()
              .dtype(dtype == DataType::Float ? at::kFloat : at::kDouble)
              .device(at::kCUDA, 0);
      // Test size 1 since it will have a single comparison, which checks
      // missing propagation in one position even if it propagates properly in
      // the other position
      for (auto size : {1, 2, 5}) {
        // To check nans in multiple positions along reduction axis create a 2D
        // tensor that is ones except the diagonal, which are nans
        auto at_x = at::eye(size, options);
        at_x = (1 - at_x) / (1 - at_x);
        std::vector<c10::IValue> inputs{at_x};

        std::vector<at::Tensor> at_outputs(
            {do_min ? at_x.amin(1) : at_x.amax(1)});
        auto nvf_outputs = executor_cache.runFusionWithInputs(inputs);

        testValidate(
            executor_cache.fusion(),
            nvf_outputs,
            inputs,
            at_outputs,
            __LINE__,
            __FILE__);
      }
    }
  }
}

class ExpandedBroadcastGlobalIntermediateTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    // Do not fill allocation with NaN. The logical output size of this test is
    // huge, although they are just because of expand, the pointwise kernel in
    // PyTorch eager mode is not smart enough to not iterating on the entire
    // logical space
    setFillAllocationWithNan(false);
  }
};

TEST_F(ExpandedBroadcastGlobalIntermediateTest, TheTest_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 1, 2});
  fusion.addInput(tv0);
  auto tv1 = expand(
      tv0,
      {IrBuilder::create<Int>(2),
       IrBuilder::create<Int>(1L << 60L),
       IrBuilder::create<Int>(2)});
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Global);

  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2, 1, 2}, options);

  FusionExecutor fe;
  fe.compileFusion(fusion_ptr.get(), {t0});
  auto cg_output = fe.runFusion({t0}).at(0);

  ASSERT_EQ(cg_output.size(0), 2);
  ASSERT_EQ(cg_output.size(1), (1L << 60L));
  ASSERT_EQ(cg_output.size(2), 2);
  ASSERT_EQ(cg_output.stride(0), 2);
  ASSERT_EQ(cg_output.stride(1), 0);
  ASSERT_EQ(cg_output.stride(2), 1);
  ASSERT_TRUE(at::eq(t0.squeeze(1), cg_output.select(1, 0)).all().item<bool>());
}

// Test forced segmentation hint
TEST_F(NVFuserTest, FusionTestSegmenterHint_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  std::vector<int64_t> input_shape{32, 64, 8, 128};
  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();
  fusion->addInput(tv0);
  auto tv1 = relu(tv0);
  auto tv2 = segment_set(tv1);
  auto tv3 = neg(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({at_x});
  auto ref_out = at_x.clone().relu().neg();

  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();

  TORCH_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen");
  auto groups = optimized_fusion->fusionSegments()->groups();
  TORCH_CHECK(
      groups.size() == 2, "segmentation hint isn't working as expected");
  // with the hint, segment_set should be grouped with its producer
  // [relu, segment_set], [neg]
  for (auto& group : groups) {
    // we only check the group with a single node
    if (group->exprs().size() == 1) {
      auto relu_expr = group->exprs()[0];
      TORCH_CHECK(
          relu_expr->isA<UnaryOp>() &&
              relu_expr->as<UnaryOp>()->getUnaryOpType() == UnaryOpType::Neg,
          "segmentation result is not expected");
    }
  }
  testValidate(
      executor_cache.fusion(), outputs, {at_x}, {ref_out}, __LINE__, __FILE__);
}

// Test file size should be up to 10K LoC. Create a new file for more tests.

} // namespace nvfuser
