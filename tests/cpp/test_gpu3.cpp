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

#include <codegen.h>
#include <debug.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <device_lower/pass/replace_size.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <grouped_reduction.h>
#include <id_model/id_model.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/abstract_tensor.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/tools/loop_domain_scheduler.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include "parallel_dimension_map.h"

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
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv1 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    NVF_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    NVF_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({24}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = t0.sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref, ref}, __LINE__, __FILE__);
}

// Repro of issue #1074
TEST_F(NVFuserTest, FusionNonDivisibleSplit2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
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
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 1,
      "Only tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    NVF_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    NVF_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({13, 17}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Similar to FusionNonDivisibleSplit1 but with unswitch
TEST_F(NVFuserTest, FusionNonDivisibleSplit3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  tv2->axis(0)->parallelize(ParallelType::Unswitch);

  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    NVF_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    NVF_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({24}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Non-divisible split through merge
TEST_F(NVFuserTest, FusionNonDivisibleSplit4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0, 1});
  fusion.addOutput(tv2);

  tv2->split(0, 5);
  tv2->merge(1, 2);
  tv2->split(1, 3);

  tv0->computeAt(tv2, -1);

  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    NVF_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    NVF_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({24, 2}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

// Nested splits
TEST_F(NVFuserTest, FusionNonDivisibleSplit5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
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
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().empty(),
      "There must be no split to validate");
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToPredicate().size() == 2,
      "Both tv1 and tv2 should have a non-divisible predicate.");
  for (auto tv : {loweredTv(tv1, gpulw), loweredTv(tv2, gpulw)}) {
    auto it = gpulw.nonDivisibleSplitInfo().splitsToPredicate().find(tv);
    NVF_CHECK(
        it != gpulw.nonDivisibleSplitInfo().splitsToPredicate().end(),
        "No info found for ",
        tv);
    const auto& splits_to_predicate = it->second;
    NVF_CHECK(
        splits_to_predicate.size() == 1,
        "There must be one split to predicate");
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({24}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

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
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    NVF_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);

  auto t0_non_divisible = at::randn({8}, options);
  // Since ceilDiv(8, 8) is not divisible by 4, the vectorization is
  // illegal. The run-time validation of vectorization should throw an error.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0_non_divisible}));
}

// If a split is validated at run time, it's not necessary to predicate.
TEST_F(NVFuserTest, FusionNonDivisibleSplitVectorize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv3->split(0, 8, false);
  tv3->split(1, 4);
  TransformPropagatorWithCheck propagator(tv3);
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv3->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv3, {tv1, tv2});

  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      gpulw.nonDivisibleSplitInfo().splitsToValidate().size() == 1,
      "There should be one split to validate");
  for (const auto& kv : gpulw.nonDivisibleSplitInfo().splitsToPredicate()) {
    const auto& splits_to_predicate = kv.second;
    NVF_CHECK(
        splits_to_predicate.empty(),
        "There must be no split to predicate, but tensor t",
        kv.first->name(),
        " has:",
        splits_to_predicate);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto t0 = at::randn({1024}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref = (t0 + 1).sum();

  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
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

  NVF_ERROR(t3->getComputeAtPosition() == 1);
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
    MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

    tv1->computeAt(tv3, -2);

    tv2->axis(-1)->parallelize(ParallelType::Vectorize);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({15}, options);
    KernelExecutor ke;
    ke.compile(&fusion);

    // This should throw an exception as the extent of t0 is not
    // divisible by the vector width
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(ke.run({t0}));

    auto t1 = at::randn({16}, options);
    auto cg_outputs = ke.run({t1});

    testValidate(&fusion, cg_outputs, {t1}, __LINE__, __FILE__);
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
  gpulw.run();
  NVF_CHECK(!gpulw.info().concretizedBroadcastDomains().isConcretized(
      loweredTv(tv4, gpulw)->axis(1)));
  NVF_CHECK(gpulw.info().concretizedBroadcastDomains().isConcretized(
      loweredTv(tv7, gpulw)->axis(1)));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 1}, options);
  auto t1 = at::randn({10, 20}, options);
  auto t2 = at::randn({10, 10}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2});
  auto outputs = ke.run({t0, t1, t2});

  testValidate(&fusion, outputs, {t0, t1, t2}, __LINE__, __FILE__);
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
  gpulw.run();
  NVF_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10, 11}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  auto t3 = t0.sum().unsqueeze(-1).unsqueeze(-1);

  testValidate(&fusion, outputs, {t0}, {t3}, __LINE__, __FILE__);
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
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);

  // The reshape op adds a broadcast domain in tv4, which is
  // parallelized. Howver, it is never materialized, so there should
  // be no parallel broadcast.

  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      !gpulw.kernel()->summary().has_block_broadcasts &&
          !gpulw.kernel()->summary().has_grid_broadcasts,
      "There must be no parallel broadcast in this fusion");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
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

  NVF_CHECK(
      bcast_concretization_info.maybeNonUniquelyConcretized(tv5->axis(1)),
      "Failed to detect non-unique concretization of ",
      tv5->toString());

  NVF_CHECK(
      bcast_concretization_info.isUniquelyConcretized(tv11->axis(1)),
      "Failed to detect unique concretization of ",
      tv11->toString());

  NVF_CHECK(
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
  auto tv7 = mul(tv4, IrBuilder::create<Val>(1. / (W * X * Y * Z)));
  auto tv8 = add(tv7, IrBuilder::create<Val>(1.e-6));
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

  auto rfactor = ir_utils::rFactorHelper(tv3, {1, 4});

  scheduler_utils::parallelizeAllLike(rfactor);

  for (auto tv : fusion.allTvs()) {
    if (tv != tv1 || tv != tv3) {
      for (auto i : arange(tv->nDims())) {
        if (isParallelTypeVectorize(tv->axis(i)->getParallelType())) {
          tv->axis(i)->parallelize(ParallelType::Serial);
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

  KernelExecutor ke;
  ke.compile(&fusion);
  auto cg_outputs = ke.run({t0}, {}, LaunchParams(X, V, -1, Y, -1, -1));

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
  auto kernel = gpulw.run();

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
  const auto ks0_name = "i0";
  const auto ks1_name = "i1";
  const auto tk0_name = "T" + std::to_string(tk0->name());

  auto& exprs = proxy.topLevelExprs();
  exprs.push_back(tk0_expr);

  // Invalid code gen
  const auto no_alloc_code = codegen::generateCudaKernel(kernel);

  // Without alloc, Int vals are just inlined, resulting in:
  // t0[(0 + 1)] = t0[((0 + 1) + 1)]
  std::stringstream no_alloc_ref;
  no_alloc_ref << "\n"
               << indent << tk0_name << "[(0LL + 1LL)]\n"
               << indent << indent << " = " << tk0_name
               << "[((0LL + 1LL) + 1LL)];\n";

  NVF_CHECK(
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

  NVF_CHECK(
      valid_code.find(valid_ref.str()) != std::string::npos,
      "Invalid code generation. Expected:",
      valid_ref.str(),
      "Actual:\n",
      valid_code);
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
  auto t0 = at::randn({X, Y, Z}, options);
  auto t1 = at::randn({X, Y, Z}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv3->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({X}, options);
  auto t1 = at::randn({W, X}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(2.0));
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

  auto t0 = at::randn({(int)(12288 * 4)}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  auto ref = t0 + 1 + 2;

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(2.0));
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

  auto t0 = at::randn({(int)(12288 * 4)}, options);
  KernelExecutor ke;

  //  NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.compile(&fusion, {t0}));
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

  auto t0 = at::randn({3, 4, 7, 2, 5}, options);
  KernelExecutor ke;

  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Repro of #1521
TEST_F(NVFuserTest, FusionImmediateValueAsInput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto immediate_scalr = IrBuilder::create<Val>(0.1);
  // Adding an immediate scalar value as an input is not allowed
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.addInput(immediate_scalr));

  // Instead, use a symbolic value
  auto symbolic_scalar = IrBuilder::create<Val>(DataType::Double);
  fusion.addInput(symbolic_scalar);

  auto tv1 = add(tv0, symbolic_scalar);
  fusion.addOutput(tv1);

  // Make sure the kernel is compiled.
  KernelExecutor ke;
  ke.compile(&fusion);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  NVF_CHECK(t0.equal(cg_outputs[0].as<at::Tensor>()));
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

  KernelExecutor ke;
  // This should fail at compile time as we're trying to merge in a
  // non-contiguous dimension, then split and vectorize it.
  ASSERT_ANY_THROW(ke.compile(&fusion, {t0}));
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // This should fail at the launch time as 14 is not divisible by the
  // vector word size. The two domains are merged, but they are not
  // contiguous, so contig indexing is not involved in this case.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0}));
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

  const int n = 12;
  auto t0 = at::randn({n}, options);
  // Shift by one to make it non-aligned
  auto t0_misaligned =
      at::randn({n + 1}, options).index({at::indexing::Slice(1)});
  auto t1_misaligned =
      at::empty({n + 1}, options).index({at::indexing::Slice(1)});

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  NVF_CHECK(t0.equal(cg_outputs[0].as<at::Tensor>()));

  // Pass misaligned input. This must fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0_misaligned}));

  // Pass misaligned output. This must fail too.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0}, {t1_misaligned}));
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0});

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0}));
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
  MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->setMemoryType(MemoryType::Local);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4}, options);
  auto t1 = at::randn({3, 4}, options);

  {
    KernelExecutor ke;
    ke.compile(&fusion, {t0, t1});
    auto cg_outputs = ke.run({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
  }

  // Make sure tv2 indexing also works when it's stored in global memory
  tv2->setMemoryType(MemoryType::Global);
  {
    KernelExecutor ke;
    ke.compile(&fusion, {t0, t1});
    auto cg_outputs = ke.run({t0, t1});

    testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv4, -2);
  tv1->computeAt(tv4, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});

  // Vectorization of tv2 should be detected as invalid.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.run({t0, t1}));
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
  // MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);

  tv2->merge(1, 2);
  tv2->merge(0, 1);
  tv2->split(0, 4);

  tv2->computeAt(tv3, -2);

  tv2->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  auto cg_results = scheduleAndRun(&fusion, SchedulerType::PointWise, {t0, t1});

  // The innermost two dimensions are merged and contiguous, so
  // vectorization can be done against 2*14=28 rather than 14, so
  // vector word size should be 4. Broadcasting of tv1 should not
  // matter.
  for (const auto& vec_info : cg_results.kernel_executor->compiledKernel()
                                  ->kernel()
                                  ->summary()
                                  .vectorized_set_info) {
    NVF_CHECK(
        vec_info.word_size == 4,
        "Invalid vector word size: ",
        vec_info.word_size);
  }

  testValidate(&fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
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
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
      NVF_ERROR(
          number_of_writes_ == 2,
          "FusionRAWSyncInsertionPlace4 test fail:",
          "only 1 sync after the 2 shared mem writes is needed in this test,"
          "either a redundant sync has been inserted or the block sync is not "
          "inserted at the right place");
    }

   private:
    int number_of_writes_ = 0;
  } sync_insertion_checker;
  GpuLower gpulw(&fusion);
  sync_insertion_checker.handle(gpulw.run()->topLevelExprs());
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2});
  auto cg_outputs = ke.run({t0, t1, t2});

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2});
  auto cg_outputs = ke.run({t0, t1, t2});

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
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

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->setMemoryType(MemoryType::Shared);

  tv0->computeAt(tv2, 1);
  tv0_shared->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);
  at::Tensor t1 = at::randn({m, n}, options);

  KernelExecutor ke;

  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_THAT(
        [&]() { ke.compile(&fusion, {t0, t1}); },
        testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
            "Reason: LoadStoreOpType::CpAsync requires Ampere")));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  } else {
    ke.compile(&fusion, {t0, t1});
  }
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  auto tv0_shared = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv0_shared->cacheAfter();
  tv0_shared->setMemoryType(MemoryType::Shared);
  tv0->computeAt(tv1, 1);

  tv0_shared->split(-1, 32);
  tv0_shared->split(-1, 4);
  tv0_shared->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);

  KernelExecutor ke;
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_THAT(
        [&]() { ke.compile(&fusion, {t0}); },
        testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
            "Reason: LoadStoreOpType::CpAsync requires Ampere")));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  } else {
    ke.compile(&fusion, {t0});
  }

  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

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
          NVF_ERROR(
              ite->predicate()->value()->isConst(),
              "redundant predicate on: ",
              expr);
        }
      }
    }

  } pred_checker;

  GpuLower gpulw(&fusion);
  pred_checker.handle(gpulw.run()->topLevelExprs());
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
  MaxLogicalDomainInfoSpanningTree(tv_avg).traverse(&propagator);

  tv_avg->axis(0)->parallelize(ParallelType::BIDx);
  tv_avg->axis(1)->parallelize(ParallelType::TIDx);

  // Make sure the parallelization of tv_avg is propagated to the var
  // and count tensors.
  GpuLower gpulw(&fusion);
  for (const auto expr : gpulw.run()->exprs()) {
    auto wop = dynamic_cast<WelfordOp*>(expr);
    if (wop == nullptr) {
      continue;
    }
    auto ref = wop->outAvg()->as<TensorView>();
    for (auto sibling : ir_utils::filterByType<TensorView>(wop->outputs())) {
      if (ref == sibling) {
        continue;
      }
      NVF_CHECK(
          ref->nDims() == sibling->nDims(),
          "Invalid sibling: ",
          sibling->toString());
      for (const auto i : arange(ref->nDims())) {
        NVF_CHECK(
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
  at::Tensor t0 = at::randn({9999}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({0})},
      __LINE__,
      __FILE__);
}

// Test ExactLogicalDomainMap
TEST_F(NVFuserTest, FusionExactLogicalDomainMap_CUDA) {
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

  const auto exact_map = ExactLogicalDomainMap(&fusion);

  // In the exact mapping, the broadcast domain introduced at tv2 is
  // only mapped with the another one in tv3, which is just transposed
  // from tv2. Any other domain, including the second domain of tv4,
  // must not be mapped.

  auto tv2_bc = tv2->axis(1);
  auto tv3_bc = tv3->axis(0);

  NVF_CHECK(
      exact_map.areMapped(tv2_bc, tv3_bc),
      "Invalid exact root domain map: ",
      exact_map.toString());

  // They must not be mapped with anything else.
  for (auto tv : fusion.allTvs()) {
    for (auto logical_id : tv->getLogicalDomain()) {
      if (logical_id == tv2_bc || logical_id == tv3_bc) {
        continue;
      }
      NVF_CHECK(
          !exact_map.areMapped(logical_id, tv2_bc),
          "Invalid exact logical domain map: ",
          exact_map.toString());
      NVF_CHECK(
          !exact_map.areMapped(logical_id, tv3_bc),
          "Invalid exact logical domain map: ",
          exact_map.toString());
    }
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
  MaxLogicalDomainInfoSpanningTree(tv6).traverse(&propagator);

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
  MaxLogicalDomainInfoSpanningTree(reduction_tv).traverse(&propagator);
  auto rfactor_tv = ir_utils::rFactorHelper(reduction_tv, {4});
  scheduler_utils::parallelizeAllLike(rfactor_tv);

  tv0->computeAt(tv_avg, 2);
  tv0->computeAt(cached_input, -2);

  cached_input->computeAt(rfactor_tv, 4, ComputeAtMode::BestEffort);

  for (auto tv : fusion.allTvs()) {
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
          NVF_CHECK(
              inp_ti->view() != alias_tv,
              "Invalid alias found between GridWelford input and output. Out "
              "tv: ",
              out_tv->toString(),
              ", In tv: ",
              alias_tv->toString());
        }
      }
    }

    std::unordered_map<TensorView*, TensorView*> alias_map_;
  } checker;

  GpuLower gpulw(&fusion);
  checker.handle(gpulw.run()->topLevelExprs());

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({X, Y, Y, Z}, options);

  auto cg_outputs = ke.run({t0}, {}, LaunchParams(-1, -1, -1, -1, -1, -1));

  // by default Welford outputs sum of square diff so need to divide to get var
  cg_outputs[1] = cg_outputs[1].as<at::Tensor>().div((float)(X * Y * Y));

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
      ir_utils::flattenScopedExprs(gpulw.run()->topLevelExprs());
  bool sync_inserted = std::any_of(
      flattened_exprs.begin(), flattened_exprs.end(), [](Expr* expr) {
        return expr->isA<kir::BlockSync>();
      });
  NVF_ERROR(sync_inserted, "Expected block sync not inserted");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
  checker.handle(gpulw.run()->topLevelExprs());
  NVF_ERROR(checker.result() < 2, "More syncs were inserted than expected");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
  checker.handle(gpulw.run()->topLevelExprs());

  // This is implicit checking. There are exactly 2 places
  //  where RAW hazards happen: one producing tv2 and the other
  //  producing tv3. This test case expect syncs in both of
  //  these places so we check that 2 RAW syncs are inserted.
  NVF_ERROR(
      checker.result() == 2,
      "Exactly 2 RAW sync expected for the two shared memory transfers");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
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
  auto used_vals = gpulw.run()->usedMathVals();

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    if (tv->name() == 2) {
      lowered_tv2 = tv;
    }
    if (tv->name() == 4) {
      lowered_tv4 = tv;
    }
  }

  NVF_ERROR(
      lowered_tv2 != nullptr && lowered_tv4 != nullptr,
      "tv2 or tv4 not lowered or mangled");

  auto tv2_info =
      gpulw.info().threadPredicateMap().getPredicateInfo(lowered_tv2);
  auto tv4_info =
      gpulw.info().threadPredicateMap().getPredicateInfo(lowered_tv4);

  // tv2 -> tv3 -> tv4 (shared) is the only use chain for tv2,
  //  and tv4 is redundantly written in tidx so tv2 is redundantly
  //  consumed in tidx.
  NVF_ERROR(
      tv2_info.redundant_use_types.get(ParallelType::TIDx),
      "TV2 is redundantly used but not detected.");

  // tv4->tv5 (global) is a redundant use chain, but
  // tv4->tv6->tv7 is not, so tv4 should not be detected as
  // a redundant used tensor in tidx.
  NVF_ERROR(
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

  NVF_CHECK(tv2->nDims() == 2, "Unpected unsqueeze result: ", tv2->toString());
  NVF_CHECK(
      tv2->axis(1)->isBroadcast(),
      "Unexpected unsqueeze result: ",
      tv2->toString());

  // tv1 has only one non-reduction axis. An exception should be
  // thrown.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(unsqueeze(tv1, 2));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
  auto tv2 = squeeze(tv1, {1});
  fusion.addOutput(tv2);

  NVF_CHECK(tv2->nDims() == 1, "Unexpected squeeze result: ", tv2->toString());

  // [I, R]
  auto tv3 = sum(tv0, {1});
  // tv3 has only one non-reduction axis. The extent of the first axis
  // is not one, so squeeze should fail.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(squeeze(tv3, {1}));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 11}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
  gpulw.run();
  NVF_CHECK(PredicatedChecker::isPredicated(tv1, gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 4}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(
      ke.compiledKernel()->kernel(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// Repro of https://github.com/csarofeen/pytorch/issues/1777
TEST_F(NVFuserTest, FusionDivScalarLhs_CUDA) {
  // tv1 = 2.0 / tv0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  TensorView* tv1 = div(IrBuilder::create<Val>(2.0), tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({3, 3}, options);
  // There's no overload div(Scalar, Tensor) in ATen
  auto aten_output = at::div(
      at::native::wrapped_scalar_tensor(at::Scalar(2.0), options.device()), t0);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpand_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto w = 2, x = 3, z = 5;
  auto y = 4L;

  // Test
  // a simple expand
  // Expand that's propagated
  // expand_as
  // symbolic expand

  // x
  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = expand(tv1, {tv0->axis(0)->extent(), IrBuilder::create<Val>(y)});

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

  auto w_symbolic = IrBuilder::create<Val>(DataType::Int);
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
  auto cg_out = cg_outputs[1].as<at::Tensor>();

  NVF_ERROR(cg_out.size(0) == w);
  NVF_ERROR(cg_out.size(1) == x);
  NVF_ERROR(cg_out.size(2) == y);
  NVF_ERROR(cg_out.size(3) == z);
  NVF_ERROR(cg_out.stride(0) == 0);
  NVF_ERROR(cg_out.stride(1) == 1);
  NVF_ERROR(cg_out.stride(2) == 0);
  NVF_ERROR(cg_out.stride(3) == 0);

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t3, t6, w}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionExpandIssue1751_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3L;
  auto y = 4, z = 5;

  // y, z
  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false, false});

  // Two ways to propagate extents as is: use -1 or explicitly pass
  // the extent vals.

  auto tv2 = expand(
      tv1,
      {IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(-1L),
       IrBuilder::create<Val>(-1L)});

  auto tv3 = expand(
      tv1,
      {IrBuilder::create<Val>(x),
       tv0->axis(0)->extent(),
       tv0->axis(1)->extent()});

  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    NVF_ERROR(cg_out.as<at::Tensor>().size(0) == x);
    NVF_ERROR(cg_out.as<at::Tensor>().size(1) == y);
    NVF_ERROR(cg_out.as<at::Tensor>().size(2) == z);
  }

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// TODO: Make sure the kernel uses the expanded concrete size instead
// of the symbolic size
TEST_F(NVFuserTest, FusionExpandToConcrete_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto x = 3L, y = 4L;

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = broadcast(tv0, {true, false});

  auto tv2 =
      expand(tv1, {IrBuilder::create<Val>(x), IrBuilder::create<Val>(y)});

  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  for (const auto& cg_out : cg_outputs) {
    NVF_ERROR(cg_out.as<at::Tensor>().size(0) == x);
    NVF_ERROR(cg_out.as<at::Tensor>().size(1) == y);
  }

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  auto var_sum_rf = ir_utils::rFactorHelper(tvs.var_sum, {1, 4});

  TransformPropagatorWithCheck propagator(var_sum_rf);
  MaxLogicalDomainInfoSpanningTree(var_sum_rf).traverse(&propagator);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  std::vector<std::vector<TensorView*>> siblings = {
      {tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
  for (const auto& tensors : siblings) {
    for (auto t1 : tensors) {
      for (auto t2 : tensors) {
        NVF_CHECK(TransformReplay::fullSelfMatching(t1, t2));
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

  auto var_sum_rf = ir_utils::rFactorHelper(tvs.var_sum, {1, 4});

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
  MaxLogicalDomainInfoSpanningTree good_path(var_sum_rf, &selector1);
  MaxLogicalDomainInfoSpanningTree bad_path(var_sum_rf, &selector2);

  auto rf_tvs = ir_utils::producerTvsOf(tvs.var_sum);

  auto check = [&]() {
    std::vector<std::vector<TensorView*>> siblings = {
        {tvs.avg, tvs.var_sum, tvs.n}, rf_tvs};
    for (const auto& tensors : siblings) {
      for (auto t1 : tensors) {
        for (auto t2 : tensors) {
          NVF_CHECK(TransformReplay::fullSelfMatching(t1, t2));
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
  MaxLogicalDomainInfoSpanningTree(tv0).traverse(&propagator);

  NVF_CHECK(tv1->nDims() == 4);
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
  NVF_CHECK(
      tv2_def != nullptr,
      "Expected LoadStoreOp but found ",
      tv2->definition()->toString());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({12345}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
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
      where(tv2, IrBuilder::create<Val>(1.0), IrBuilder::create<Val>(2.0));
  fusion->addOutput(tv3);

  std::vector<int64_t> shape({999});
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto ref = where(t0 >= t1, 1.0, 2.0);

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, {ref}, __LINE__, __FILE__);
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
  MaxLogicalDomainInfoSpanningTree(tv2, &selector).traverse(&propagator);

  NVF_CHECK(tv0->nDims() == 2);
  NVF_CHECK(tv1->nDims() == 1);
  NVF_CHECK(tv2->nDims() == 2);
  NVF_CHECK(tv3->nDims() == 2);
  NVF_CHECK(tv4->nDims() == 1);
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
  MaxLogicalDomainInfoSpanningTree(tv1, 2).traverse(&propagator);

  auto expect = makeConcreteTensor({22, 105});
  expect->split(0, 2);
  NVF_CHECK(TransformReplay::fullSelfMatching(expect, tv0));
}

TEST_F(NVFuserTest, FusionMaxLogicalDomainInfoSpanningTreePrintTwice_CUDA) {
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

  MaxLogicalDomainInfoSpanningTree path(tv1);
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
  NVF_CHECK(printer1.ss.str() == expect);
  NVF_CHECK(printer2.ss.str() == expect);
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

  MaxLogicalDomainInfoSpanningTree path1(tv2);
  TransformPropagatorWithCheck propagator1(tv2);
  path1.traverse(&propagator1);

  MaxLogicalDomainInfoSpanningTree path2(tv0);
  TransformPropagatorWithCheck propagator2(tv0);
  path2.traverse(&propagator2);

  NVF_CHECK(tv1->axis(0)->isBroadcast());
  NVF_CHECK(tv1->axis(1)->isBroadcast());
  NVF_CHECK(!tv1->axis(2)->isBroadcast());
  NVF_CHECK(!tv1->axis(3)->isBroadcast());
  NVF_CHECK(tv1->axis(4)->isBroadcast());

  auto expect = makeSymbolicTensor(3);
  expect->split(1, 2);
  expect->split(0, 4);
  NVF_CHECK(TransformReplay::fullSelfMatching(expect, tv1));
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

  KernelExecutor ke;
  ke.compile(&fusion, {in1, in2});
  auto cg_outputs = ke.run({in1, in2});

  testValidate(&fusion, cg_outputs, {in1, in2}, __LINE__, __FILE__);
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
    MaxLogicalDomainInfoSpanningTree(tv3).traverse(&propagator);
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
    MaxLogicalDomainInfoSpanningTree(tv0).traverse(&propagator);
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

  NVF_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.avg->getComputeAtPosition());
  NVF_CHECK(
      tvs.var_sum->getComputeAtPosition() == tvs.n->getComputeAtPosition());
  NVF_CHECK(tvs.var_sum->getComputeAtPosition() == 1);
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
  NVF_ERROR(
      tv2->nDims() == tv3->nDims(),
      "Propagator didn't propagate to tv2: ",
      tv2->toString());

  // Check that the splits are replayed on tv1 as well. Even though
  //  one of its consumers, tv2, is part of the boundary, another
  //  consumer is not a boundary, so tv1 should be transformed as well.
  NVF_ERROR(
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

  std::vector<Val*> sum_to_symb;
  std::transform(
      sum_to_shape.begin(),
      sum_to_shape.end(),
      std::back_inserter(sum_to_symb),
      [](int64_t s) -> Val* { return IrBuilder::create<Val>(s); });

  TensorView* tv0 = makeContigConcreteTensor(shape);
  TensorView* tv1 = makeContigConcreteTensor(shape);
  TensorView* tv2 = makeContigConcreteTensor(shape, DataType::Bool);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  Val* prob = IrBuilder::create<Val>(kProb);
  auto grad_input = dropout_backward(tv1, tv2, prob);
  auto grad_gelu = gelu_backward(grad_input, tv0);
  auto grad_bias = sum_to(grad_gelu, sum_to_symb);

  fusion->addOutput(grad_gelu);
  fusion->addOutput(grad_bias);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor a = at::randn(shape, options);
  at::Tensor b = at::randn(shape, options);
  at::Tensor c = at::randn(shape, options);
  auto mask = at::gt(c, 0.0f);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({a, b, mask});

  testValidate(
      executor_cache.fusion(), cg_outputs, {a, b, mask}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInsertMagicZero1_CUDA) {
  EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv2->split(-1, 2);
  tv2->reorder({{1, 2}, {2, 1}});
  tv2->merge(0);

  TransformPropagatorWithCheck propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);

  tv0->computeAt(tv2, 1);

  // The predicate of tv2 should be protected with magic zero
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
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
  for (const auto i : arange(3)) {
    if (i == 0) {
      domain1[i] = IterDomainBuilder(
                       FusionGuard::getCurFusion()->zeroVal(),
                       IrBuilder::create<Val>(1L, DataType::Index))
                       .iter_type(IterType::Broadcast)
                       .build();
    } else {
      domain1[i] =
          IterDomainBuilder(
              FusionGuard::getCurFusion()->zeroVal(),
              IrBuilder::create<Val>(1L, DataType::Index))
              .expanded_extent(IrBuilder::create<Val>(1L + i, DataType::Index))
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

  at::Tensor t1 = at::randn({1, 2, 3}, options);
  at::Tensor t2 = at::randn({1, 2, 3}, options);
  at::Tensor t3 = at::randn({1, 2, 3}, options);
  at::Tensor t4 = at::randn({1, 1, 1}, options).expand({1, 2, 3});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t1, t2, t3, t4});
}

TEST_F(NVFuserTest, FusionExpandReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({1, 8});
  fusion->addInput(tv0);

  auto tv1 =
      expand(tv0, {IrBuilder::create<Val>(12L), IrBuilder::create<Val>(8L)});

  auto tv2 = sum(tv1, {0});
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
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
  auto t0 = at::randn({1024}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__, "");
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
          FusionGuard::getCurFusion()->zeroVal(),
          IrBuilder::create<Val>(DataType::Index))
          .build(),
      IterDomainBuilder(
          FusionGuard::getCurFusion()->zeroVal(),
          FusionGuard::getCurFusion()->oneVal())
          .expanded_extent(IrBuilder::create<Val>(10L, DataType::Index))
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
  at::Tensor t1 = at::randn({2, 3}, options);
  // Passing expand size of 5, not 10. Should cause an error
  at::Tensor t4 = at::randn({2, 1}, options).expand({2, 5});

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  ASSERT_ANY_THROW(executor_cache.runFusionWithInputs({t1, t4}));
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

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {t0, t1}).outputs;
  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPrint_CUDA) {
  std::vector<at::ScalarType> dtypes = {
      at::kFloat, at::kDouble, at::kHalf, at::kInt, at::kLong, at::kBool};
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    dtypes.push_back(at::kBFloat16);
  }
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
      std::pair<std::unique_ptr<FusionExecutorCache>, KernelArgumentHolder>;
  auto matched_add = [](at::Tensor a, at::Tensor b) -> return_t {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    Val* s1 = IrBuilder::create<Val>(DataType::Int);
    Val* s2 = IrBuilder::create<Val>(DataType::Int);
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
    testValidate(ret1.first->fusion(), ret1.second, {a, b}, __LINE__, __FILE__);
  }

  {
    ASSERT_THAT(
        [&]() { matched_add(a, c); },
        ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
            "When trying to propagate constant tensor sizes through the graph "
            "a conflict was found with 2 different sizes across dimensions "
            "that are expected to match.")));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }
}

TEST_F(NVFuserTest, FusionSizeDependentData_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  Val* s1 = IrBuilder::create<Val>(DataType::Index);
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

  testValidate(executor_cache.fusion(), cg_outputs, {a}, __LINE__, __FILE__);
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
      NVF_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    NVF_CHECK(all_vals_set.empty());
  }

  auto tv10 = add(tv6, tv7);
  {
    auto all_vals = DependencyCheck::getAllValsBetween({tv0, tv1}, {tv10});
    std::unordered_set<Val*> all_vals_set(all_vals.begin(), all_vals.end());
    std::vector<Val*> results({tv0, tv1, tv6, tv7, tv10});
    for (auto result : results) {
      NVF_CHECK(all_vals_set.count(result) > 0);
      all_vals_set.erase(result);
    }
    NVF_CHECK(all_vals_set.empty());
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
  at::Tensor t1 = at::randn({1, 1, 333, 1}, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::Transpose, {input0, t1}, false)
          .outputs;
  testValidate(&fusion, cg_outputs, {input0, t1}, __LINE__, __FILE__);
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

  KernelExecutor ke;
  ke.compile(fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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

  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(1.0)); // Group 0
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation didn't happen");
  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()
              ->fusionSegments()
              ->groups()
              .size() == 2,
      "segmentation didn't happen as expected");

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
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

  MaxLogicalDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1}, options);
  at::Tensor t1 = at::randn({10}, options);

  KernelExecutor ke;
  ke.compile(fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(
      fusion, cg_outputs, {t0, t1}, {t1 + t0.flatten()}, __LINE__, __FILE__);
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

  KernelExecutor ke;
  ke.compile(fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Simplified repro of issue #2008
TEST_F(NVFuserTest, FusionReplayTrivialReductionAndBroadcast2_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape({10, 1, 1});

  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1, 2});
  auto tv3 = broadcast(tv2, {false, true, true});
  fusion.addOutput(tv3);

  tv0->merge(-2, -1)->merge(-2, -1)->split(0, 4);

  MaxLogicalDomainInfoSpanningTree tree(tv0);
  TransformPropagatorWithCheck tp(tv0);
  tree.traverse(&tp);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleAmperePipeline_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);

  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  fusion.addOutput(tv1);

  auto tv_cache = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv_cache->setMemoryType(MemoryType::Shared);

  tv1->split(0, 16);
  tv0->computeAt(tv1, 1);

  tv_cache->circularBuffer(10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({255}, options);

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
      if (ldst->opType() == LoadStoreOpType::CpAsync &&
          ldst->cacheOp() == CacheOp::AllLevels) {
        NVF_ERROR(!within_ite_, "CPASYNC predicate not inlined");
        NVF_ERROR(
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
  pred_checker.handle(gpulw.run()->topLevelExprs());

  KernelExecutor ke;
  // requires ampere+ GPU
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_THAT(
        [&]() { ke.compile(&fusion, {t1}); },
        testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
            "Reason: LoadStoreOpType::CpAsync requires Ampere")));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  } else {
    ke.compile(&fusion, {t1});
  }

  auto cg_outputs = ke.run({t1});

  testValidate(&fusion, cg_outputs, {t1}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
      expand(tv7, {tv0->axis(0)->extent(), IrBuilder::create<Val>(32128L)});
  auto tv3 = exp(tv1);
  auto tv8 = mul(tv3, tv_exp);
  auto tv13 = sub(tv0, tv8);
  fusion->addOutput(tv13);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 =
      at::empty_strided({4096, 32128}, {32128, 1}, options).random_();
  at::Tensor t0 =
      at::empty_strided({4096, 32128}, {32128, 1}, options).random_();

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t0});

  testValidate(fusion, cg_outputs, {t1, t0}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(768L)});
    auto tv4 = expand(
        broadcast(tv1, {true, true, false}),
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(768L)});
    auto tv5 = reshape(tv2, {1024, 768}, {1, 1024, 768});
    auto tv6 = castOp(DataType::Float, tv5);
    auto s7 = IrBuilder::create<Val>(0.5);
    auto tv8 = mul(tv6, s7);
    auto s9 = IrBuilder::create<Val>(0.707107);
    auto tv10 = mul(tv6, s9);
    auto tv11 = erf(tv10);
    auto s12 = IrBuilder::create<Val>(1.0);
    auto tv13 = add(tv11, s12);
    auto tv14 = mul(tv8, tv13);
    auto tv15 = castOp(DataType::Half, tv14);
    auto tv16 = castOp(DataType::Float, tv15);
    auto tv17_tv18 = variance_mean(tv16, {2}, 0, false);
    auto tv17 = std::get<0>(tv17_tv18);
    auto tv18 = std::get<1>(tv17_tv18);
    auto tv19 = expand(
        broadcast(tv17, {false, false, true}),
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(1L)});
    auto tv20 = expand(
        broadcast(tv18, {false, false, true}),
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(1L)});
    auto s21 = IrBuilder::create<Val>(1e-05);
    auto tv22 = add(tv19, s21);
    auto tv23 = expand(
        broadcast(tv20, {false, false, false}),
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(768L)});
    auto tv24 = rsqrt(tv22);
    auto tv25 = sub(tv16, tv23);
    auto tv26 = expand(
        broadcast(tv24, {false, false, false}),
        {IrBuilder::create<Val>(1L),
         IrBuilder::create<Val>(1024L),
         IrBuilder::create<Val>(768L)});
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
  auto t0 = at::randn({768}, options);
  auto t1 = at::randn({768}, options);
  auto t2 = at::randn({1024, 768}, options).to(at::ScalarType::Half);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

// https://github.com/csarofeen/pytorch/issues/2068
TEST_F(NVFuserTest, FusionIssue2068_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int64_t w = 32, x = 56, y = 56, z = 128;

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
      {IrBuilder::create<Val>(w),
       IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(y),
       tv6->axis(3)->extent()});
  auto tv8 = broadcast(tv2, {false, false, false, true});
  auto tv9 = broadcast(tv3, {true, true, true, false});
  auto tv10 = expand(
      tv9,
      {IrBuilder::create<Val>(w),
       IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(y),
       tv9->axis(3)->extent()});
  auto tv11 = set(tv5);
  auto tv12 = expand(
      tv11,
      {tv11->axis(0)->extent(),
       tv11->axis(1)->extent(),
       tv11->axis(2)->extent(),
       IrBuilder::create<Val>(z)});

  auto tv13 = add(tv8, IrBuilder::create<Val>(1.e-6));
  auto tv14 = sub(tv4, tv12);
  auto tv15 = rsqrt(abs(tv13));
  auto tv16 = set(tv15);
  auto tv17 = expand(
      tv16,
      {tv16->axis(0)->extent(),
       tv16->axis(1)->extent(),
       tv16->axis(2)->extent(),
       IrBuilder::create<Val>(z)});
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2, t3, t4});

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t0, t1, t2, t3, t4},
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
  auto tv2 = mul(tv1, IrBuilder::create<Val>(0.5));
  auto tv3 = mul(tv1, IrBuilder::create<Val>(0.707107));
  auto tv4 = erf(tv3);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  auto tv6 = mul(tv2, tv5);
  auto tv7 = squeeze(tv6, std::vector<bool>{true, false, false});

  fusion.addOutput(tv1);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__, "");
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

  MaxLogicalDomainInfoSpanningTree tree(tv3);
  TransformPropagatorWithCheck tp(tv3);
  tree.traverse(&tp);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 1, 1, 1, 1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
    MaxLogicalDomainInfoSpanningTree tree(tv0);
    TransformPropagatorWithCheck tp(tv0);
    tree.traverse(&tp);
    NVF_CHECK(tv2->nDims() == 2);
    NVF_CHECK(tv1->nDims() == 2);
    NVF_CHECK(tv0->nDims() == 2);
  }

  {
    // The propagation here should be a no-op, I am adding it here just to test
    // if transformation propagation works for squeeze on both direction.
    MaxLogicalDomainInfoSpanningTree tree(tv2);
    TransformPropagatorWithCheck tp(tv2);
    tree.traverse(&tp);
    NVF_CHECK(tv2->nDims() == 2);
    NVF_CHECK(tv1->nDims() == 2);
    NVF_CHECK(tv0->nDims() == 2);
  }

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  inlineMost();

  NVF_CHECK(tv1->getComputeAtPosition() == 2);
  NVF_CHECK(tv2->getComputeAtPosition() == 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1024}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
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
  auto tv2 = mul(tv1, IrBuilder::create<Val>(0.5));
  auto tv3 = mul(tv1, IrBuilder::create<Val>(0.707107));
  auto tv4 = erf(tv3);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  auto tv6 = mul(tv2, tv5);
  auto tv7 = sum(tv6, {0});

  fusion.addOutput(tv1);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 8}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__, "");
}

#ifndef USE_ROCM

TEST_F(NVFuserTest, Castings) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int x = 4, y = 1024;

  std::vector<DataType> data_types{
      DataType::Double,
      DataType::Float,
      DataType::Half,
      DataType::Char,
      DataType::Short,
      DataType::Int32,
      DataType::Int,
      DataType::Byte,
      DataType::UInt16,
      DataType::UInt32,
      DataType::UInt64,
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

  KernelArgumentHolder args;
  std::vector<at::Tensor> outputs;
  for (const auto& input_type : data_types) {
    at::Tensor t = at::randn({x, y}, options)
                       .relu() // Discard negative numbers so that signed and
                               // unsigned types are equivalent. There is no way
                               // to represent unsigned numbers in PyTorch.
                       .to(data_type_to_aten(input_type));
    args.push(t);
    for (const auto& output_type : data_types) {
      outputs.emplace_back(t.to(data_type_to_aten(output_type)));
    }
  }

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs(args);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      args,
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
  auto tv1 = ne(tv0, IrBuilder::create<Val>(0L));
  auto tv2 = castOp(DataType::Int32, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = sub(tv3, IrBuilder::create<Val>(1L));
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
  ASSERT_TRUE(at::allclose(cg_outputs[1].as<at::Tensor>(), t4));
}

TEST_F(NVFuserTest, FusionIssue2077_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv3 = mul(tv1, IrBuilder::create<Val>(1L));
  auto tv5 = sub(IrBuilder::create<Val>(1.), tv3);
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
  ASSERT_TRUE(at::equal(cg_outputs[0].as<at::Tensor>(), t7));
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
  auto seps = IrBuilder::create<Val>(DataType::Double);
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

  int C = 2;
  at::Tensor x = at::randn({1, 5, 5, 5, C}, options);
  at::Tensor mean = at::randn({C}, options);
  at::Tensor var = at::rand({C}, options);
  double eps = 1e-5;

  auto eager_diff = x - mean.view({1, 1, 1, 1, -1});
  auto eager_invstd = at::rsqrt(var + eps);
  auto eager_x_normed = eager_diff * eager_invstd.view({1, 1, 1, 1, -1});

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {x, mean, var, eps});
  // testValidate currently fails since cg_outputs.outputs[0] is an empty tensor
  ASSERT_TRUE(
      at::allclose(cg_outputs.outputs[0].as<at::Tensor>(), eager_x_normed));
  // ASSERT_TRUE(at::equal(cg_outputs.outputs[1], mean));
  ASSERT_TRUE(
      at::allclose(cg_outputs.outputs[2].as<at::Tensor>(), eager_invstd));
}

TEST_F(NVFuserTest, FusionIssue2075_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int64_t x = 2, y = 128, z = 128;

  auto tv0 = makeContigConcreteTensor({1, -1, 1});
  fusion.addInput(tv0);
  auto tv1 = makeContigConcreteTensor({1, 1, -1});
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = expand(
      tv2,
      {IrBuilder::create<Val>(x),
       tv2->axis(1)->extent(),
       IrBuilder::create<Val>(z)});

  // [1, 1, 128] -> [1, 1, 1, 1, 1, 128]
  auto tv4 = broadcast(tv1, {{false, false, true, true, true, false}});
  // [1, 1, 1, 1, 1, 128] -> [2, 128, 1, 1, 1, 128]
  auto tv5 = expand(
      tv4,
      {IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(y),
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

  MaxLogicalDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->setMemoryType(MemoryType::Shared);

  // The predicate tv2 should look like (i * 4) + j < tv0.extent(0),
  // where i and j are the loop indices of the two loop axes,
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
        NVF_ERROR(!scope_exprs_.empty());
        NVF_ERROR(scope_exprs_.back()->isA<kir::IfThenElse>());
        auto ite = scope_exprs_.back()->as<kir::IfThenElse>();
        auto cond = ite->predicate()->value();
        // Make sure the index of the inner loop isn't used in the predicate
        NVF_ERROR(!for_loops_.empty());
        auto loop_index = for_loops_.back()->index();
        auto cond_inputs = InputsOf::output(cond);
        auto index_it =
            std::find(cond_inputs.begin(), cond_inputs.end(), loop_index);
        auto vec_factor_it =
            std::find_if(cond_inputs.begin(), cond_inputs.end(), [](Val* inp) {
              auto int_val = inp->value();
              return int_val.hasValue() &&
                  (int_val.as<int64_t>() == vec_factor - 1 ||
                   int_val.as<int64_t>() == -(vec_factor - 1));
            });
        // If vectorized, the predicate should use (vec_factor - 1) or
        // -(vec_factor - 1) rather than the loop index.
        if (vectorized_) {
          NVF_CHECK(
              index_it == cond_inputs.end(),
              "Not expected to have ",
              loop_index->toInlineString(),
              " in ",
              cond->toInlineString());
          NVF_CHECK(
              vec_factor_it != cond_inputs.end(),
              "Expected to have ",
              vec_factor - 1,
              " in ",
              cond->toInlineString());
        } else {
          NVF_CHECK(
              index_it != cond_inputs.end(),
              "Expected to have ",
              loop_index->toInlineString(),
              " in ",
              cond->toInlineString());
          NVF_CHECK(
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
  gpulw_wo_vec.run();
  PredChecker(false).handle(gpulw_wo_vec.kernel()->topLevelExprs());

  // Vectorize the second axis of tv1
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // Now, the predicate tv2 should look like (i * 4) + 3 <
  // tv0.extent(0), i.e., j should be replaced with 3 since the second
  // axis is exactly mapped with the vectorized axis of tv1. It is
  // sufficient to check the condition using the last value of j,
  // i.e., 3.

  GpuLower gpulw_w_vec(&fusion);
  gpulw_w_vec.run();
  PredChecker(true).handle(gpulw_w_vec.kernel()->topLevelExprs());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  NVF_CHECK(t0.equal(cg_outputs[0].as<at::Tensor>()));
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
      IrBuilder::create<Val>(0.0),
      IrBuilder::create<Val>(0.0),
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
  ASSERT_TRUE(at::allclose(
      cg_outputs[0].as<at::Tensor>(), cg_outputs[3].as<at::Tensor>()));
  ASSERT_TRUE(at::allclose(
      cg_outputs[1].as<at::Tensor>(), cg_outputs[4].as<at::Tensor>()));
  ASSERT_TRUE(at::allclose(
      cg_outputs[2].as<at::Tensor>(), cg_outputs[5].as<at::Tensor>()));
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
  MaxLogicalDomainInfoSpanningTree(ref).traverse(&propagator);

  // Don't inline the innermost axes
  std::unordered_set<IterDomain*> uninlinable;
  uninlinable.insert(output->axis(-1));
  uninlinable.insert(weight_copy1->axis(-1));

  inlineMost(uninlinable);

  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto at_input = at::randn({N, C}, options_float);
  auto at_weight = at::randn({C}, options_float);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {at_input, at_weight});
  auto cg_outputs = ke.run({at_input, at_weight});
  auto cg_output = cg_outputs[0].as<at::Tensor>();

  auto ref_x_sub_mean = at_input - at_input.sum({0}).unsqueeze(0);
  auto ref_y = ref_x_sub_mean * at_weight.unsqueeze(0);

  testValidate(
      ke.compiledKernel()->kernel(),
      {cg_output},
      {at_input, at_weight},
      {ref_y},
      __LINE__,
      __FILE__,
      "");
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

    auto f2 = IrBuilder::create<Val>(float_val, DataType::Float);
    NVF_CHECK(
        f2->getDataType() == DataType::Float,
        "Invalid data type: ",
        f2->getDataType().value());

    auto d3 = IrBuilder::create<Val>(double_val, DataType::Double);
    NVF_CHECK(
        d3->getDataType() == DataType::Double,
        "Invalid data type: ",
        d3->getDataType().value());

    // Adding two Floats produces a Float
    auto f4 = add(f2, f2);
    NVF_CHECK(
        f4->getDataType() == DataType::Float,
        "Invalid data type: ",
        f4->getDataType().value());

    // Adding a Double and a Float produces a Double
    auto d5 = add(f2, d3);
    NVF_CHECK(
        d5->getDataType() == DataType::Double,
        "Invalid data type: ",
        d5->getDataType().value());

    // Adding a Float and a Double produces a Double
    auto d6 = add(d3, f2);
    NVF_CHECK(
        d6->getDataType() == DataType::Double,
        "Invalid data type: ",
        d6->getDataType().value());

    // Adding two Doubles produce a Double
    auto d7 = add(d5, d6);
    NVF_CHECK(
        d7->getDataType() == DataType::Double,
        "Invalid data type: ",
        d7->getDataType().value());

    // Adding a Float to a Float tensor produces a Float tensor
    auto tv1 = add(tv0, f4);
    NVF_CHECK(
        tv1->getDataType() == DataType::Float,
        tv1->toString(),
        " has an invalid data type: ",
        tv1->getDataType().value());

    // Adding a Double to a Float tensor still produces a Float tensor
    auto tv2 = add(tv1, d7);
    NVF_CHECK(
        tv2->getDataType() == DataType::Float,
        tv2->toString(),
        " has an invalid data type: ",
        tv2->getDataType().value());

    fusion.addOutput(tv2);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIntegerType_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t int64_val = 1;
  const int int_val = 2;

  {
    auto tv0 = makeConcreteTensor({10}, DataType::Int32);
    fusion.addInput(tv0);

    auto i2 = IrBuilder::create<Val>(int64_val, DataType::Int);
    auto i3 = IrBuilder::create<Val>((int64_t)int_val, DataType::Int32);

    // Adding two Ints produces an Int
    auto i4 = add(i2, i2);
    NVF_CHECK(
        i4->getDataType() == DataType::Int,
        "Invalid result: ",
        i4->toInlineString());

    // Adding two Int32s produces an Int32
    auto i5 = add(i3, i3);
    NVF_CHECK(
        i5->getDataType() == DataType::Int32,
        "Invalid result: ",
        i5->toInlineString());

    // Adding an Int and an Int32 produces an Int
    auto i6 = add(i4, i5);
    NVF_CHECK(
        i6->getDataType() == DataType::Int,
        "Invalid result: ",
        i6->toInlineString());

    // Adding an Int32 to an Int32 tensor produces an Int32 tensor
    auto tv1 = add(tv0, i4);
    NVF_CHECK(
        tv1->getDataType() == DataType::Int32,
        tv1->toString(),
        " has an invalid data type: ",
        tv1->getDataType().value());

    // Adding an Int to an Int32 tensor still produces an Int32 tensor
    auto tv2 = add(tv1, i6);
    NVF_CHECK(
        tv2->getDataType() == DataType::Int32,
        tv2->toString(),
        " has an invalid data type: ",
        tv2->getDataType().value());

    fusion.addOutput(tv2);
  }

  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(10, {10}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto i2 = int64_val;
  auto i3 = int_val;
  auto i4 = i2 + i2;
  auto i5 = i3 + i3;
  auto i6 = i4 + i5;
  auto t1 = t0 + i4;
  auto t2 = t1 + i6;

  NVF_CHECK(cg_outputs[0].as<at::Tensor>().equal(t2));
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

  MaxLogicalDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->computeWith(-1, true);

  GpuLower gpulw(&fusion);
  auto all_exprs = KernelExprVisitor::getAllExprs(gpulw.run());
  auto num_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<WelfordOp>();
      });
  NVF_CHECK(
      num_welford_ops == 0,
      "All WelfordOp exprs should be converted to VectorizedWelfordOp");

  auto num_vectorized_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<kir::VectorizedWelfordOp>();
      });
  NVF_CHECK(
      num_vectorized_welford_ops == 1,
      "There must be two VectorizedWelfordOp exprs");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref_avg = t0.mean({0});
  auto ref_var = t0.var({0}, false) * shape[0];
  auto ref_N = at::ones({shape[1]}, options_int) * shape[0];

  testValidate(
      ke.compiledKernel()->kernel(),
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

  MaxLogicalDomainInfoSpanningTree tree(tv1);
  TransformPropagator tp(tv1);
  tree.traverse(&tp);

  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1->computeAt(tvs.avg, 3);
  tvs.avg->axis(2)->parallelize(ParallelType::Unswitch);

  tv1->computeWith(-1, true);

  GpuLower gpulw(&fusion);
  auto all_exprs = KernelExprVisitor::getAllExprs(gpulw.run());
  auto num_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<WelfordOp>();
      });
  NVF_CHECK(
      num_welford_ops == 0,
      "All WelfordOp exprs should be converted to VectorizedWelfordOp");

  auto num_vectorized_welford_ops =
      std::count_if(all_exprs.begin(), all_exprs.end(), [](Expr* expr) {
        return expr->isStrictlyA<kir::VectorizedWelfordOp>();
      });
  NVF_CHECK(
      num_vectorized_welford_ops == 2,
      "There must be two VectorizedWelfordOp exprs");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto ref_avg = t0.to(at::kDouble).mean({0});
  auto ref_var = t0.to(at::kDouble).var({0}, false) * shape[0];
  auto ref_N = at::ones({shape[1]}, options_int) * shape[0];

  testValidate(
      ke.compiledKernel()->kernel(),
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  at::Tensor t6 = at::tensor({15}, options.dtype(at::kLong));
  at::Tensor t15 = at::randn({3, 2, 1, 2}, options.dtype(at::kDouble));
  at::Tensor t20 =
      at::tensor({12}, options.dtype(at::kLong)).expand({1, 1, 1, 1});

  auto cg_outputs = executor_cache.runFusionWithInputs({t6, t15, t20});

  auto sample_total = at::sum(t15, {0, 1, 2, 3}, true);
  auto sample_mean = at::div(sample_total, t20);
  auto x = at::sub(t15, sample_mean);
  auto input = at::mul(x, x);
  auto total = at::sum(input, {0, 1, 2, 3}, false);
  auto t7 = at::div(total, t6);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {t6, t15, t20},
      {t7},
      __LINE__,
      __FILE__);
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

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(
      ke.compiledKernel()->kernel(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionFloatConstantWhere_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1, DataType::Bool);
  fusion.addInput(tv0);

  auto tv1 = where(
      tv0,
      IrBuilder::create<Val>(3.0, DataType::Float),
      IrBuilder::create<Val>(5.0, DataType::Float));

  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(4, options) > 1.0;

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {t0}).outputs;
  auto ref = at::where(t0, (float)3.0, (float)5.0);
  // testValidate does not check that dtypes match
  NVF_CHECK(cg_outputs[0].as<at::Tensor>().dtype() == ref.dtype());
  testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCpAsyncCommitWait_CUDA) {
  // Repro for https://github.com/csarofeen/pytorch/issues/2463
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({12800, 8, 8, 8}, DataType::Half);
  auto tv1 = set(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(2)->parallelize(ParallelType::TIDx);

  auto tv2 = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tv2->axis(-1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);

  tv2->inlineAt(1);
  tv2->circularBuffer(8);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({12800, 8, 8, 8}, options);

  KernelExecutor ke;
  if (!deviceMajorMinorCheck(8)) {
    ASSERT_THAT(
        [&]() { ke.compile(&fusion, {t0}); },
        testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
            "Reason: LoadStoreOpType::CpAsync requires Ampere")));
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  } else {
    ke.compile(&fusion, {t0});
  }

  auto cg_outputs = ke.run({t0});
  testValidate(
      ke.compiledKernel()->kernel(), cg_outputs, {t0}, __LINE__, __FILE__);
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
  at::Tensor t0 = at::randn({10, 11}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  auto t3 = t0.sum({1}).sum({0});
  auto t6 = t0.sum({1});

  testValidate(
      ke.compiledKernel()->kernel(),
      cg_outputs,
      {t0},
      {t3, t6},
      __LINE__,
      __FILE__);
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

  using kir::IrVisitor::dispatch;
  using kir::IrVisitor::handle;

  void handle(kir::IfThenElse* ite) final {
    for (auto expr : ite->thenBody().exprs()) {
      auto tv_output = ir_utils::getTvOutput(expr);
      if (tv_output != nullptr && tv_output->name() == tv_name_to_check_ &&
          expr->isA<LoadStoreOp>() && ite->predicate()->hasValue()) {
        dispatch(ite->predicate()->value());
      }
    }
  }

  void dispatch(Val* val) final {
    if (val->definition()) {
      dispatch(val->definition());
    }
  }

  void handle(BinaryOp* bop) final {
    if (bop->getBinaryOpType() == BinaryOpType::LogicalAnd) {
      dispatch(bop->lhs());
      dispatch(bop->rhs());
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
  NVF_CHECK(
      ThreadPredChecker::isPredicatedBy(
          tv1->name(), predicated_types, gpulw.run()),
      "Validation of lowered kernel failed");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2}, options);
  at::Tensor t1 = at::randn({10000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref_t1 = t0.sum({0});
  auto ref_t4 = t1.exp();

  testValidate(
      ke.compiledKernel()->kernel(),
      cg_outputs,
      {t0, t1},
      {ref_t1, ref_t4},
      __LINE__,
      __FILE__);
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
  NVF_CHECK(
      ThreadPredChecker::isPredicatedBy(
          tv1->name(), predicated_types, gpulw.run()),
      "Validation of lowered kernel failed");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2}, options);
  at::Tensor t1 = at::randn({10000}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto ref_t1 = t0.sum({0});
  auto ref_t3 = t1.exp();

  testValidate(
      ke.compiledKernel()->kernel(),
      cg_outputs,
      {t0, t1},
      {ref_t1, ref_t3},
      __LINE__,
      __FILE__);
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

    MaxLogicalDomainInfoSpanningTree tree(tv2);
    TransformPropagator tp(tv2);
    tree.traverse(&tp);

    inlineMost();

    tv2->axis(1)->parallelize(ParallelType::BIDx);
    tv2->axis(2)->parallelize(ParallelType::TIDx);

    auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
    at::Tensor t_small = at::randn({999}, options).ge(0);

    at::Tensor t_large =
        at::randn({std::numeric_limits<int>::max()}, options).ge(0);

    NVF_CHECK(
        KernelArgumentHolder({t_large}).getSmallestIndexTypeOfArguments() ==
        PrimDataType::Int);
    NVF_CHECK(
        KernelArgumentHolder({t_small}).getSmallestIndexTypeOfArguments() ==
        PrimDataType::Int32);

    {
      KernelExecutor ke;
      // Lower the kernel with large inputs and int64 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int};
      ke.compile(&fusion, {t_large}, LaunchParams(), compile_opts);

      NVF_CHECK(
          ke.compiledKernel()->kernel()->indexType() == PrimDataType::Int,
          "Unexpected kernel index type: ",
          ke.compiledKernel()->kernel()->indexType());

      // Since the index type is int64, both small and large inputs
      // should work fine
      ke.run({t_small});
      ke.run({t_large});
    }

    {
      KernelExecutor ke;
      // Lower the kernel with small inputs and int64 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int};
      ke.compile(&fusion, {t_small}, LaunchParams(), compile_opts);

      NVF_CHECK(
          ke.compiledKernel()->kernel()->indexType() == PrimDataType::Int,
          "Unexpected kernel index type: ",
          ke.compiledKernel()->kernel()->indexType());

      // Since the index type is int64, both small and large inputs
      // should work fine
      ke.run({t_small});
      ke.run({t_large});
    }

    {
      KernelExecutor ke;
      LaunchParams launch_params;
      CompileParams compile_opts = {.index_type = PrimDataType::Int32};
      ke.compile(&fusion, {t_small}, launch_params, compile_opts);

      NVF_CHECK(
          ke.compiledKernel()->kernel()->indexType() == PrimDataType::Int32,
          "Unexpected kernel index type: ",
          ke.compiledKernel()->kernel()->indexType());

      // This should complete successfully as the arguments are small
      // enough to use the int32 index type
      ke.run({t_small});

      // This should fail as the Kernel is already compiled for Int32, but
      // the arguments are too large
      CompileParams compile_opts_large = {.index_type = PrimDataType::Int};
      EXPECT_THAT(
          [&]() { ke.run({t_large}, {}, launch_params, compile_opts_large); },
          testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
              "Kernel index type and compilation index type don't match")));
    }

    {
      KernelExecutor ke;
      // Lower the kernel with large inputs and int32 index type.
      CompileParams compile_opts = {.index_type = PrimDataType::Int32};
      // This should fail due to the conflict
      EXPECT_THAT(
          [&]() {
            ke.compile(&fusion, {t_large}, LaunchParams(), compile_opts);
          },
          testing::ThrowsMessage<nvfuser::nvfError>(
              testing::HasSubstr("Compilation with int32 is requested but "
                                 "int64 is required for the arguments")));
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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int);

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

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  executor_cache.runFusionWithInputs({t0, t1});
  auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int);

  // Running again with forced type of Int32
  executor_cache.runFusionWithInputs({t0, t1}, PrimDataType::Int32);
  kernel_runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(kernel_runtime->getIndexType() == PrimDataType::Int32);
}

//! Test whether we can create and use float16 scalars
TEST_F(NVFuserTest, FusionHalfScalars_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::Half);
  fusion->addInput(tv0);

  auto tv2 = full_like(tv0, IrBuilder::create<Val>(1.5, DataType::Half));
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
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

  auto tv2 = full_like(tv0, IrBuilder::create<Val>(1.5, DataType::BFloat16));
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({5}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}
#endif

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
  lower.run();
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

// Repro of issue #2125, 1.45e+03 GB/s on A100-80G
TEST_F(NVFuserTest, FusionAvoidRedundantWriteBroadcastedSoftmaxInput_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape0({2, 512});
  std::vector<int64_t> shape1({2, 64, 512, 512});

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tvb = broadcast(tv0, {false, true, true, false});
  auto tv2 = add(tvb, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv1, tv2);
  auto tv4 = softmax(tv3, -1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::ones(shape0, options);
  at::Tensor t1 = at::ones(shape1, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  // check thread_pred and write_stride
  const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
  auto kernel = ke->compiledKernel()->kernel();
  const auto& thread_pred_map =
      ke->compiledKernel()->lowered()->info().threadPredicateMap();
  for (const auto expr : kernel->exprs()) {
    auto tv = ir_utils::getTvOutput(expr);
    if (tv && tv->name() == 15 && tv->getMemoryType() == MemoryType::Global) {
      const auto& thread_pred = thread_pred_map.getPredicateInfo(tv);
      bool predicted = thread_pred.redundant_types.get(ParallelType::BIDx) &&
          thread_pred.broadcast_ld_indices_map.count(ParallelType::BIDx);
      NVF_CHECK(
          predicted,
          "Tv15 should be predicted by ParallelType::BIDx with a "
          "broadcast_ld_indices_map!");
      break;
    }
  }

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAvoidRedundantWrite_CUDA) {
  auto runTest = [](const std::vector<bool>& is_broadcast) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    std::vector<int64_t> shape0;
    // inner dim should be large enough to trigger the issue
    // 10240 >= vect (4) x threads per block (512) x max unroll (5).
    // In inner reduction, TIDx is used for the reduction axis, if it
    // is less than 512, TIDy is used for iteration axis, this leads
    // to a split in the iteration axis, current redundant write remover
    // can't handle this case since it only checks the loop domain whose
    // definition is a merge.
    std::vector<int64_t> shape1({2, 64, 2, 10240});
    const size_t ndim = shape1.size();
    for (size_t i = 0; i < ndim; i++) {
      if (!is_broadcast[i]) {
        shape0.push_back(shape1[i]);
      }
    }

    auto tv0 = makeSymbolicTensor(shape0.size());
    auto tv1 = makeSymbolicTensor(4);
    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tvb = broadcast(tv0, is_broadcast);
    auto tv2 = add(tvb, IrBuilder::create<Val>(1.0));
    auto tv3 = add(tv1, tv2);
    auto tv4 = sum(tv3, {-1});
    fusion.addOutput(tv2);
    fusion.addOutput(tv4);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn(shape0, options);
    at::Tensor t1 = at::randn(shape1, options);

    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

    // check thread_pred and write_stride
    const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
    auto kernel = ke->compiledKernel()->kernel();
    const auto& thread_pred_map =
        ke->compiledKernel()->lowered()->info().threadPredicateMap();

    for (const auto expr : kernel->exprs()) {
      auto tv = ir_utils::getTvOutput(expr);
      if (tv && tv->name() == 8 && tv->getMemoryType() == MemoryType::Global) {
        const auto& thread_pred = thread_pred_map.getPredicateInfo(tv);
        bool predicted = thread_pred.redundant_types.get(ParallelType::BIDx) &&
            thread_pred.broadcast_ld_indices_map.count(ParallelType::BIDx);
        NVF_CHECK(
            predicted,
            "Tv8 should be predicted by ParallelType::BIDx with a "
            "broadcast_ld_indices_map!");
        break;
      }
    }

    testValidate(
        executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
  };

  // Test case where [B1,I2,I3] is merged to [B1I2I3]
  runTest({true, false, false, false});

  // Test case where [I1,B2,I3] is merged to [I1B2I3]
  runTest({false, true, false, false});

  // Test case where [I1,I2,B3] is merged to [I1I2B3]
  runTest({false, false, true, false});

  // Test case where [I1,B2,B3] is merged to [I1B2B3]
  runTest({false, true, true, false});

  // Test case where [B1,I2,B3] is merged to [B1I2B3]
  runTest({true, false, true, false});

  // Test case where [B1,B2,I3] is merged to [B1B2I3]
  runTest({true, true, false, false});

  // Test case where [B1,B2,B3] is merged to [B1B2B3]
  runTest({true, true, true, false});
}

TEST_F(NVFuserTest, FusionAvoidRedundantWriteDifferentConcretizedDomains_CUDA) {
  // if the broadcasted tensor is concretized to different shapes
  // the fusion will be segmented.
  auto runTest = [](const bool direct_lowering) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);

    const std::vector<bool> is_broadcast = {true, false, true, false};
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1({2, 64, 128, 2048});
    std::vector<int64_t> shape2({4, 64, 256, 2048});
    const size_t ndim = shape1.size();
    for (size_t i = 0; i < ndim; i++) {
      if (!is_broadcast[i]) {
        shape0.push_back(shape1[i]);
      }
    }

    auto tv0 = makeSymbolicTensor(shape0.size());
    auto tv1 = makeSymbolicTensor(4);
    auto tv2 = makeSymbolicTensor(4);
    fusion.addInput(tv0);
    fusion.addInput(tv1);
    fusion.addInput(tv2);

    auto tv3 = broadcast(tv0, is_broadcast);
    auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
    // concretized to shape1
    auto tv5 = add(tv4, tv1);
    // concretized to shape2
    auto tv6 = add(tv4, tv2);
    auto tv7 = sum(tv5, {-1});
    auto tv8 = sum(tv6, {-1});
    fusion.addOutput(tv4);
    fusion.addOutput(tv7);
    fusion.addOutput(tv8);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn(shape0, options);
    at::Tensor t1 = at::randn(shape1, options);
    at::Tensor t2 = at::randn(shape2, options);

    if (direct_lowering) {
      // it should be segmented, if directly lowered, it should throw an error
      EXPECT_THAT(
          [&]() {
            scheduleAndRun(
                &fusion, SchedulerType::Reduction, {t0, t1, t2}, false);
          },
          testing::ThrowsMessage<nvfuser::nvfError>(
              testing::HasSubstr("Producer is required to be in Global Memory "
                                 "based on parallelization strategy.")));
    } else {
      FusionExecutorCache executor_cache(std::move(fusion_ptr));
      auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

      auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
      NVF_CHECK(optimized_fusion->isSegmented(), "segmentation didn't happen!");

      testValidate(
          executor_cache.fusion(),
          cg_outputs,
          {t0, t1, t2},
          __LINE__,
          __FILE__);
    }
  };
  runTest(true);
  runTest(false);
}

TEST_F(NVFuserTest, FusionAvoidRedundantWriteNonOutput_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto tv5 = add(tv3, IrBuilder::create<Val>(1.0));
  tv5->setMemoryType(MemoryType::Global);
  auto tv6 = add(tv5, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv6);

  for (auto tv : {tv3, tv4, tv5, tv6}) {
    tv->merge(0);
  }

  tv2->inlineAt(1);
  tv3->inlineAt(1);
  tv5->inlineAt(1);

  for (auto tv : {tv3, tv4, tv5, tv6}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32}, options);
  at::Tensor t1 = at::randn({32, 64}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  // check thread_pred
  auto kernel = ke.compiledKernel()->kernel();
  const auto& thread_pred_map =
      ke.compiledKernel()->lowered()->info().threadPredicateMap();

  for (const auto expr : kernel->exprs()) {
    auto tv = ir_utils::getTvOutput(expr);
    if (tv->name() == 5 || tv->name() == 6) {
      const auto& thread_pred = thread_pred_map.getPredicateInfo(tv);
      bool predicted = thread_pred.redundant_types.get(ParallelType::BIDx) &&
          thread_pred.broadcast_ld_indices_map.count(ParallelType::BIDx);
      NVF_CHECK(
          predicted,
          "TV5 and TV6 should be predicted by ParallelType::BIDx with a "
          "broadcast_ld_indices_map!");
    }
  }

  testValidate(fusion_ptr.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test case where the merge order is random
TEST_F(NVFuserTest, FusionAvoidRedundantWriteNonNeighbor_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const int ndim = 5;
  const std::vector<bool> is_broadcast = {false, true, false, false, true};
  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(ndim);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, is_broadcast);
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto tv5 = add(tv3, IrBuilder::create<Val>(1.0));
  tv5->setMemoryType(MemoryType::Global);
  auto tv6 = add(tv5, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv6);

  // merge first and last domain
  for (auto tv : {tv3, tv4, tv5, tv6}) {
    tv->merge(0, -1);
  }

  tv2->inlineAt(-1);
  tv3->inlineAt(-1);
  tv5->inlineAt(-1);

  for (auto tv : {tv3, tv4, tv5, tv6}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8, 10, 12}, options);
  at::Tensor t1 = at::randn({8, 7, 10, 12, 9}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  // check thread_pred
  auto kernel = ke.compiledKernel()->kernel();
  const auto& thread_pred_map =
      ke.compiledKernel()->lowered()->info().threadPredicateMap();

  for (const auto expr : kernel->exprs()) {
    auto tv = ir_utils::getTvOutput(expr);
    if (tv->name() == 5 || tv->name() == 6) {
      const auto& thread_pred = thread_pred_map.getPredicateInfo(tv);
      bool predicted = thread_pred.redundant_types.get(ParallelType::BIDx) &&
          thread_pred.broadcast_ld_indices_map.count(ParallelType::BIDx);
      NVF_CHECK(
          predicted,
          "TV5 and TV6 should be predicted by ParallelType::BIDx with a "
          "broadcast_ld_indices_map!");
    }
  }

  testValidate(fusion_ptr.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

  // dom0: logical domain
  // dom1: [4, I1]
  // Should fail as the derived domain only partially covers the
  // logical domain
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            tv1->getLogicalDomain(), {tv1->axis(1), tv1->axis(2)});
      },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("dom0 has unreachable IDs")));

  tv1->merge(0);
  // [I0/4*4, I1]

  // dom0: logical domain
  // dom1: loop domain
  // Should succeed.
  ir_utils::validateDomainEquivalence(
      tv1->getLogicalDomain(), tv1->getLoopDomain());

  auto tv1_intermediate_id = tv1->axis(0);

  tv1->split(0, 3);
  // [I0/4*4/3, 3, I1]

  // dom0: logical domain
  // dom1: loop + tv1_intermediate_id
  // Should fail as the intermediate ID and the first two loop ids are
  // redundant
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            tv1->getLogicalDomain(),
            {tv1_intermediate_id, tv1->axis(0), tv1->axis(1), tv1->axis(2)});
      },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("is redundant")));
  // Same pair but reversed order
  EXPECT_THAT(
      [&]() {
        ir_utils::validateDomainEquivalence(
            {tv1_intermediate_id, tv1->axis(0), tv1->axis(1), tv1->axis(2)},
            tv1->getLogicalDomain());
      },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("is redundant")));

  // Testing symbolic domains
  auto tv2 = reshape(
      tv0,
      {IrBuilder::create<Val>(DataType::Int),
       IrBuilder::create<Val>(DataType::Int)});

  ir_utils::validateDomainEquivalence(
      tv2->getRootDomain(), tv2->getLoopDomain());

  // create a 2D tensor with one symbolid and another non-symbolic
  auto tv4 = broadcast(sum(tv2, {1}), {false, true});
  fusion.addOutput(tv4);

  // [S0, B0]
  tv4->split(1, 4);
  // [S0, B0/4, 4]

  ir_utils::validateDomainEquivalence(
      tv4->getLogicalDomain(), tv4->getLoopDomain());

  // dom0: logical domain
  // dom1: [S0, B0/4]
  // Succeeds because broadcasting IterDomains are just auxiliary placeholders
  // and can be arbitrarily created and annihilated as needed.
  ir_utils::validateDomainEquivalence(
      tv4->getLogicalDomain(), {tv4->axis(0), tv4->axis(1)});
}

TEST_F(NVFuserTest, CompareLogicalAndLoopDomains) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [i0, i1]
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  // [i0]
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  // [i0]
  auto tv2 = set(tv1);
  // [i0, b1]
  auto tv3 = broadcast(tv2, {false, true});
  // [i0, i1]
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Set the loop domain of tv2 the same as tv4. The new loop domain
  // includes an ID that is not reachable from tv2 logical domain
  tv2->setLoopDomain(
      {tv2->getLogicalDomain().at(0),
       tv4->getLoopDomain().at(1)->cloneWithoutRFactor()});

  // Same for tv3
  tv3->setLoopDomain(
      {tv3->getLogicalDomain().at(0),
       tv4->getLoopDomain().at(1)->cloneWithoutRFactor()});

  // Test if the validation can catch an invalid loop domain that
  // cannot reach the concrete domain of tv2
  EXPECT_THAT(
      [&]() {
        tv2->setLoopDomain({tv4->getLoopDomain().at(1)->cloneWithoutRFactor()});
      },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Not all logical IDs are covered by loop domain")));
}

TEST_F(NVFuserTest, CompareDomainWithReference1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  const auto& reference = tv1->getLogicalDomain();

  tv1->merge(0);
  auto domain = tv1->getLoopDomain();
  EXPECT_TRUE(ir_utils::compareDomainWithReference(domain, reference).empty());

  // Adding one of the logical domain, which is redundant.
  domain.push_back(tv1->getLogicalDomain().at(0));
  auto x = ir_utils::compareDomainWithReference(domain, reference);
  EXPECT_FALSE(ir_utils::compareDomainWithReference(domain, reference).empty());

  tv1->split(0, 4);
  domain = tv1->getLoopDomain();
  EXPECT_TRUE(ir_utils::compareDomainWithReference(domain, reference).empty());
  // Replace one of the loop IDs with a logical ID. Dependency is not
  // fully satisfied, but it should detect the redundancy
  domain[0] = tv1->getLogicalDomain()[0];
  EXPECT_FALSE(ir_utils::compareDomainWithReference(domain, reference).empty());

  // Check non-leaf intermedaite IDs
  domain = tv1->getLoopDomain();
  // Adding a parent ID to the current loop domain. This should be
  // detected as redundnt.
  domain.push_back(
      tv1->getLoopDomain().at(0)->definition()->input(0)->as<IterDomain>());
  // Create a further IDs to make the above IDs non leaf
  tv1->merge(0);
  EXPECT_FALSE(ir_utils::compareDomainWithReference(domain, reference)
                   .redundant_ids.empty());

  // Remember the current loop domain
  domain = tv1->getLoopDomain();
  // Reset the loop domain
  tv1->setLoopDomain(tv1->getLogicalDomain());
  // Schedule the tensor again
  tv1->merge(0);
  EXPECT_TRUE(
      ir_utils::compareDomainWithReference(tv1->getLoopDomain(), reference)
          .empty());
  // Combine the previous loop domain and the new loop domain, which
  // should be detected as redundant
  domain.push_back(tv1->getLoopDomain()[0]);
  EXPECT_FALSE(ir_utils::compareDomainWithReference(domain, reference)
                   .redundant_ids.empty());
}

// Pattern to test (see the comment of CompareDomainResult in
// ir/utils.h)
// For example, if we have
//  I0  I1  I2  I3
//   \  /    \  /
//    I4      I5
// then [I0, I1, I2, I3] is equivalent to [I4, I5], but [I1, I2, I3] is not
// equivalent to [I4, I5].
TEST_F(NVFuserTest, CompareDomainWithReference2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->merge(1);

  EXPECT_TRUE(ir_utils::compareDomainWithReference(
                  tv1->getLoopDomain(), tv1->getLogicalDomain())
                  .empty());
  EXPECT_FALSE(
      ir_utils::compareDomainWithReference(
          {tv1->getLogicalDomain().begin() + 1, tv1->getLogicalDomain().end()},
          tv1->getLoopDomain())
          .unreachable_reference_ids.empty());
}

// Pattern to test (see the comment of CompareDomainResult in ir/utils.h)
//  I0  I1  I2  I3
//   \  /    \  /
//    I4      I5
//   /  \    /  \.
//  I6  I7  I8  I9
// Then [I0, I1, I8, I9] is equivalent to [I6, I7, I2, I3]. [I0, I1, I2, I3] is
// equivalent to [I6, I7, I8, I9]. But [I0, I1, I8, I3] is NOT equivalent to
// [I6, I7, I2, I9]
//
// The second case does not work compareDomainWithReference as none
// of the two domains is disjoint.
TEST_F(NVFuserTest, CompareDomainWithReference3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->merge(1);

  tv1->split(0, 3);
  tv1->split(-1, 4);

  EXPECT_TRUE(ir_utils::compareDomainWithReference(
                  {tv1->getLogicalDomain().at(0),
                   tv1->getLogicalDomain().at(1),
                   tv1->getLoopDomain().at(2),
                   tv1->getLoopDomain().at(3)},
                  {tv1->getLoopDomain().at(0),
                   tv1->getLoopDomain().at(1),
                   tv1->getLogicalDomain().at(2),
                   tv1->getLogicalDomain().at(3)})
                  .empty());

  // Testing [I0, I1, I8, I3]. Logical domain is used as a referene
  auto result1 = ir_utils::compareDomainWithReference(
      {tv1->getLogicalDomain().at(0),
       tv1->getLogicalDomain().at(1),
       tv1->getLoopDomain().at(2),
       tv1->getLogicalDomain().at(3)},
      tv1->getLogicalDomain());
  // I2 is unreachable
  EXPECT_EQ(
      result1.unreachable_reference_ids,
      std::vector<IterDomain*>{tv1->getLogicalDomain().at(2)});

  // Testing [I6, I7, I2, I9]. Logical domain is used as a referene
  auto result2 = ir_utils::compareDomainWithReference(
      {tv1->getLoopDomain().at(0),
       tv1->getLoopDomain().at(1),
       tv1->getLogicalDomain().at(2),
       tv1->getLoopDomain().at(3)},
      tv1->getLogicalDomain());
  // I3 is unreachable
  EXPECT_EQ(
      result2.unreachable_reference_ids,
      std::vector<IterDomain*>{tv1->getLogicalDomain().at(3)});
}

// Repro of issue #3502
TEST_F(NVFuserTest, CompareDomainWithReference4) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [i0]
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  // [i1, i0]
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  // [i0]
  auto tv2 = set(tv0);
  // [b2, i0]
  auto tv3 = broadcast(tv2, {true, false});
  // [i1, i0]
  auto tv4 = add(tv1, tv3);
  fusion.addOutput(tv4);

  // [i3]
  tv4->merge(0);

  // tv2
  {
    // Clone i1
    auto missing_id = tv4->getLogicalDomain().at(0)->cloneWithoutRFactor();
    std::vector<IterDomain*> new_loop_domain{
        IterDomain::merge(missing_id, tv2->getLogicalDomain().at(0))};
    auto result = ir_utils::compareDomainWithReference(
        new_loop_domain, tv2->getLogicalDomain());
    EXPECT_TRUE(result.redundant_ids.empty());
    EXPECT_EQ(result.additional_ids, std::vector<IterDomain*>{missing_id});
    EXPECT_TRUE(result.unreachable_reference_ids.empty());
    tv2->setLoopDomain(new_loop_domain);
  }

  // tv3
  {
    // Clone i1
    auto missing_id = tv4->getLogicalDomain().at(0)->cloneWithoutRFactor();
    std::vector<IterDomain*> new_loop_domain{
        IterDomain::merge(missing_id, tv3->getLogicalDomain().at(1))};
    auto result = ir_utils::compareDomainWithReference(
        new_loop_domain, tv3->getLogicalDomain());
    EXPECT_TRUE(result.redundant_ids.empty());
    EXPECT_EQ(
        result.unreachable_reference_ids,
        std::vector<IterDomain*>{tv3->getLogicalDomain().at(0)});
    tv3->setLoopDomain(new_loop_domain);
  }
}

TEST_F(NVFuserTest, AllIDsWithExtraLoopIDs1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [i0, i1]
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  // [i0]
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  // [i0]
  auto tv2 = set(tv1);
  // [i0, b1]
  auto tv3 = broadcast(tv2, {false, true});
  // [i0, i1]
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Set the loop domain of tv2 the same as tv4. The new loop domain
  // includes an ID that is not reachable from tv2 logical domain
  auto tv2_inner_loop_domain =
      tv4->getLoopDomain().at(1)->cloneWithoutRFactor();
  tv2->setLoopDomain({tv2->getLogicalDomain().at(0), tv2_inner_loop_domain});

  tv2->merge(0, 1);
  auto tv2_merge_out = tv2->axis(0);
  tv2->split(0, 32);

  // tv2 logical: [i0]
  //   merge(i0, i1) -> i0*i1
  //   split(i0*i1, 32) -> i0*i1/32, 32
  // tv2 loop: [i0*i1/32, 32]
  //
  // All IDs: [i0, i0*i1, i0*i1/32, 32]

  // This ordering should return nothing as the logical domain does
  // not have i1, thus the merge expr cannot be traversed.
  EXPECT_TRUE(
      getExprsBetween<IRBFS>(
          {tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end()},
          {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()},
          false)
          .first.empty());

  // This ordering should find two exprs (i.e., the merge and the split).
  EXPECT_EQ(
      getExprsBetween<IRBFS>(
          {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()},
          {tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end()},
          false)
          .first.size(),
      2);

  std::unordered_set<IterDomain*> tv2_all_ids_ref;
  tv2_all_ids_ref.insert(
      tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end());
  tv2_all_ids_ref.insert(tv2_inner_loop_domain);
  tv2_all_ids_ref.insert(tv2_merge_out);
  tv2_all_ids_ref.insert(
      tv2->getLoopDomain().begin(), tv2->getLoopDomain().end());

  auto tv2_all_ids = tv2->domain()->allIDs();
  std::unordered_set<IterDomain*> tv2_all_id_set(
      tv2_all_ids.begin(), tv2_all_ids.end());

  EXPECT_EQ(tv2_all_id_set, tv2_all_ids_ref);
}

TEST_F(NVFuserTest, AllIDsWithExtraLoopIDs2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [i0, i1]
  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  // [i0]
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  // [i0]
  auto tv2 = set(tv1);
  // [i0, b1]
  auto tv3 = broadcast(tv2, {false, true});
  // [i0, i1]
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  // Set the loop domain of tv2 the same as tv4. The new loop domain
  // includes an ID that is not reachable from tv2 logical domain
  auto tv2_inner_loop_domain =
      tv4->getLoopDomain().at(1)->cloneWithoutRFactor();
  std::vector<IterDomain*> tv2_initial_loop_domain{
      tv2->getLogicalDomain().at(0), tv2_inner_loop_domain};
  tv2->setLoopDomain(tv2_initial_loop_domain);

  // Schedule only the extra dommain
  tv2->split(1, 4);
  auto tv2_split = tv2->axis(1)->definition();

  // tv2 logical: [i0]
  //   split(i1) -> i1/4, 4
  // tv2 loop: [i0, i1/4, 4]
  //
  // All IDs: [i0, i1, i1/4, 4]

  EXPECT_EQ(tv2->getInitialLoopDomain(), tv2_initial_loop_domain);

  // Because the split only uses the extra ID, getExprsBetween from
  // the loop domain to the logical domain does not traverse the
  // split, just returning an empty vector.
  EXPECT_TRUE(
      getExprsBetween<IRBFS>(
          {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()},
          {tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end()},
          false)
          .first.empty());

  // From the initial loop to the current loop should find the split expr
  auto exprs_between =
      getExprsBetween<IRBFS>(
          {tv2->getInitialLoopDomain().begin(),
           tv2->getInitialLoopDomain().end()},
          {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()},
          false)
          .first;
  EXPECT_EQ(exprs_between.size(), 1);
  EXPECT_EQ(exprs_between.front().first, tv2_split);

  // The initial loop domain and the current loop domain should be
  // reachable to each other with no redundancy
  auto tv2_loop_domain_comparison_results = ir_utils::compareDomains(
      tv2->getInitialLoopDomain(), tv2->getLoopDomain());
  EXPECT_FALSE(tv2_loop_domain_comparison_results.dom0_has_unreachable_ids);
  EXPECT_FALSE(tv2_loop_domain_comparison_results.dom1_has_unreachable_ids);

  // Make sure allIDs finds all the IDs including the extra IDs
  std::unordered_set<IterDomain*> tv2_all_ids_ref;
  tv2_all_ids_ref.insert(
      tv2->getLogicalDomain().begin(), tv2->getLogicalDomain().end());
  tv2_all_ids_ref.insert(
      tv2->getInitialLoopDomain().begin(), tv2->getInitialLoopDomain().end());
  tv2_all_ids_ref.insert(
      tv2->getLoopDomain().begin(), tv2->getLoopDomain().end());

  auto tv2_all_ids = tv2->domain()->allIDs();
  std::unordered_set<IterDomain*> tv2_all_id_set(
      tv2_all_ids.begin(), tv2_all_ids.end());

  EXPECT_EQ(tv2_all_id_set, tv2_all_ids_ref);
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
  at::Tensor t0 = at::randn({11}, options);
  at::Tensor t1 = at::randn({11}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
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

        auto nvf_outputs = executor_cache.runFusionWithInputs({at_x});

        testValidate(
            executor_cache.fusion(), nvf_outputs, {at_x}, __LINE__, __FILE__);
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
      {IrBuilder::create<Val>(2L),
       IrBuilder::create<Val>(1L << 60L),
       IrBuilder::create<Val>(2L)});
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);
  tv1->setMemoryType(MemoryType::Global);

  tv1->axis(2)->parallelize(ParallelType::TIDx);
  tv2->axis(2)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2, 1, 2}, options);

  KernelExecutor ke;
  ke.compile(fusion_ptr.get(), {t0});
  auto out_tensor = ke.run({t0})[0].as<at::Tensor>();

  ASSERT_EQ(out_tensor.size(0), 2);
  ASSERT_EQ(out_tensor.size(1), (1L << 60L));
  ASSERT_EQ(out_tensor.size(2), 2);
  ASSERT_EQ(out_tensor.stride(0), 2);
  ASSERT_EQ(out_tensor.stride(1), 0);
  ASSERT_EQ(out_tensor.stride(2), 1);
  ASSERT_TRUE(
      at::eq(t0.squeeze(1), out_tensor.select(1, 0)).all().item<bool>());
}

TEST_F(NVFuserTest, FusionTestWarnRegisterSpill_CUDA) {
  const int hidden_size = 1024 * 10;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);
  std::vector<int64_t> input_shape{2048, hidden_size};
  std::vector<int64_t> norm_shape{hidden_size};

  auto input = makeSymbolicTensor(input_shape.size());
  fusion.addInput(input);
  auto result = layer_norm(input, norm_shape, nullptr, nullptr, eps_ptr);
  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = c10::nullopt;
  c10::optional<at::Tensor> aten_bias = c10::nullopt;
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  // capture stdout and check stdout contains register spill warning
  captureStdout();
  {
    // generate persistent kernel
    auto heuristic_params = SchedulerEntry::scheduleWith(
        &fusion, SchedulerType::InnerPersistent, {aten_input});

    // compile and run persistent kernel
    // intentionally set maxrregcount to 32 to trigger register spill
    auto compile_opts = heuristic_params->cparams;
    compile_opts.maxrregcount = 32;
    compile_opts.enable_ptxas_verbose = true;

    // nvrtc JIT caching may skip compilation, in which case no warning
    // is produced. Disable caching to test the warning option.
    DisableOptionsGuard disable_opt_guard;
    DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);

    KernelExecutor ke;
    ke.compile(&fusion, {aten_input}, heuristic_params->lparams, compile_opts);
    auto cg_outputs = ke.run({aten_input});

    // validate results
    testValidate(
        &fusion,
        cg_outputs,
        {aten_input},
        {std::get<0>(aten_outputs),
         std::get<1>(aten_outputs),
         std::get<2>(aten_outputs)},
        __LINE__,
        __FILE__,
        "");
  }
  std::string output = getCapturedStdout();
  NVF_CHECK(
      output.find("Register spill detected") != std::string::npos,
      "Register spill is not captured!");
}

// Simple test to check if the aligned block sync is used in aligned
// reductions
TEST_F(NVFuserTest, AlignedSyncReduction1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  const int gdimx = 16;
  const int bdimx = 100;
  const int per_thread_reductions = 8;

  std::vector<int64_t> shape({gdimx * bdimx * per_thread_reductions});

  tv1->split(0, bdimx);
  tv1->split(0, per_thread_reductions);

  // Serial reduction
  auto tv2 = tv1->rFactor({1});
  // Block reduction
  tv1->rFactor({1});

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv2);

  GpuLower gpulw(&fusion);
  const std::string kernel_string = codegen::generateCudaKernel(gpulw.run());

  // The block reduction should use the aligned sync
  NVF_CHECK(
      kernel_string.find("blockReduce<true, false, false, true>(") !=
          std::string::npos,
      "blockReduce with aligned sync not found: ",
      kernel_string);
}

TEST_F(NVFuserTest, IntegerDivision_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1, DataType::Int);
  auto tv1 = makeSymbolicTensor(1, DataType::Int);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = div(tv0, tv1);
  auto tv3 = truediv(tv0, tv1);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = (at::randn({1024 * 1024}, options) * 1024).to(at::kLong);
  at::Tensor t1 = (at::randn({1024 * 1024}, options) * 1024).to(at::kLong);
  auto div_expect = at::div(input0, t1, "trunc");
  auto truediv_expect = at::true_divide(input0, t1);

  auto cg_outputs = executor_cache.runFusionWithInputs({input0, t1});

  ASSERT_TRUE(cg_outputs[0].as<at::Tensor>().scalar_type() == at::kLong);
  ASSERT_TRUE(cg_outputs[1].as<at::Tensor>().scalar_type() == at::kFloat);

  testValidate(
      executor_cache.fusion(),
      cg_outputs,
      {input0, t1},
      {div_expect, truediv_expect},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, IsFinite_CUDA) {
  std::vector<std::pair<DataType, at::ScalarType>> dtypes{
      {DataType::Float, at::kFloat}, {DataType::Half, at::kHalf}};
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    dtypes.push_back({DataType::BFloat16, at::kBFloat16});
  }
  for (const auto& [nvfuser_dtype, aten_dtype] : dtypes) {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    auto fusion = fusion_ptr.get();
    FusionGuard fg(fusion);
    auto tv0 = makeContigTensor(1, nvfuser_dtype);
    fusion->addInput(tv0);
    auto tv1 = isfinite(tv0);
    fusion->addOutput(tv1);

    auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
    std::array<float, 3> data{1.0, INFINITY, NAN};
    const auto input = at::from_blob(data.data(), {3}, {1}).to(options);

    KernelExecutor ke;
    ke.compile(fusion, {input});
    const auto output = ke.run({input});

    testValidate(fusion, output, {input}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, Repro413_CUDA) {
  int64_t n = 10240;

  for (int64_t m : {3, 6, 12, 24}) {
    for (int64_t k : {10, 20, 40}) {
      std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
      Fusion& fusion = *fusion_ptr.get();
      FusionGuard fg(&fusion);

      auto tv0 = makeContigConcreteTensor({n, m}, DataType::Half);
      fusion.addInput(tv0);
      auto tv1 = broadcast(tv0, {false, true, false});
      auto tv2 = expand(
          tv1,
          {IrBuilder::create<Val>(n),
           IrBuilder::create<Val>(k),
           IrBuilder::create<Val>(m)});
      auto tv3 = reshape(tv2, {n, k, m}, {n, k * m});
      auto tv4 = reshape(tv3, {n, k * m}, {n, m, k});
      auto tv5 = transpose(tv4, 0, 1);
      auto tv6 = reshape(tv5, {m, n, k}, {m * n, k});
      fusion.addOutput(tv6);

      auto expect_vec_factor = std::gcd(m, k);

      auto getVectorizationFactor = [](TensorView* tv) -> int64_t {
        for (auto i : tv->getLoopDomain()) {
          if (i->getParallelType() == ParallelType::Vectorize) {
            return i->extent()->evaluate().as<int64_t>();
          }
        }
        return 1;
      };

      auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
      auto t0 = at::randn({n, m}, options);
      auto cg_outputs =
          scheduleAndRun(&fusion, SchedulerType::PointWise, {t0}).outputs;

      for (auto o : fusion.outputs()) {
        EXPECT_EQ(
            getVectorizationFactor(o->as<TensorView>()), expect_vec_factor);
      }
      for (auto i : fusion.inputs()) {
        for (auto c : ir_utils::consumerTvsOf(i->as<TensorView>())) {
          EXPECT_EQ(getVectorizationFactor(c), expect_vec_factor);
        }
      }

      testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
    }
  }
}

// Based on FusionTestWarnRegisterSpill_CUDA but modified to test OptionsGuard
TEST_F(NVFuserTest, FusionOptionsGuard_CUDA) {
  const int hidden_size = 1024 * 10;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);
  std::vector<int64_t> input_shape{2048, hidden_size};
  std::vector<int64_t> norm_shape{hidden_size};

  auto input = makeSymbolicTensor(input_shape.size());
  fusion.addInput(input);
  auto result = layer_norm(input, norm_shape, nullptr, nullptr, eps_ptr);
  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = c10::nullopt;
  c10::optional<at::Tensor> aten_bias = c10::nullopt;
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  auto heuristic_params = SchedulerEntry::scheduleWith(
      &fusion, SchedulerType::InnerPersistent, {aten_input});

  // compile and run persistent kernel
  // intentionally set maxrregcount to 32 to trigger register spill
  heuristic_params->cparams.maxrregcount = 32;

  EnableOptionsGuard enable_opt_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::WarnRegisterSpill);

  // nvrtc JIT caching may skip compilation, in which case no warning
  // is produced. Disable caching to test the warning option.
  DisableOptionsGuard disable_opt_guard;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);

  // capture stdout and check stdout contains register spill warning
  captureStdout();

  KernelExecutor ke;
  ke.compile(
      &fusion,
      {aten_input},
      heuristic_params->lparams,
      heuristic_params->cparams);

  std::string output = getCapturedStdout();
  ASSERT_NE(output.find("Register spill detected"), std::string::npos)
      << "Register spill is not captured! NVRTC output: " << output;
}

// Test that DebugStreamGuard captures output
TEST_F(NVFuserTest, FusionDebugStreamGuard_CUDA) {
  std::stringstream ss;
  std::string text("test debug output");

  debug() << "text before guard";

  { // Test using the guard
    DebugStreamGuard dsg(ss);

    debug() << text;
  }

  debug() << "text after guard";

  // If the guard failed, we might write nothing to ss or we might write the
  // text after the guard to ss.
  ASSERT_EQ(ss.str(), text);
}

// Test that disabling kernel re-use leads to resegmented Fusion
TEST_F(NVFuserTest, FusionDisableKernelReuse_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto a5 = at::zeros({5}, options);
  auto a6 = at::zeros({6}, options);
  auto a7 = at::zeros({7}, options);

  executor_cache.runFusionWithInputs({a5});

  auto numRuntimes = [&executor_cache]() -> size_t {
    // this is map<pair<device, conc_info>, vector<FusionKernelRuntime>>
    const auto& runtime_map = executor_cache.getKernelRuntimes();
    return runtime_map
        .begin() // There should be only one device/concretization pair
        ->second.size();
  };

  {
    DisableOptionsGuard og;
    DisableOptionsGuard::getCurOptions().unset(DisableOption::KernelReuse);

    executor_cache.runFusionWithInputs({a6});

    // Since kernel reuse is enabled, we should not generate a new runtime
    EXPECT_EQ(numRuntimes(), 1);
  }

  {
    DisableOptionsGuard og;
    DisableOptionsGuard::getCurOptions().set(DisableOption::KernelReuse);

    executor_cache.runFusionWithInputs({a7});

    // Disabling reuse means we should get a new runtime
    EXPECT_EQ(numRuntimes(), 2);
  }
}

// Repro of https://github.com/NVIDIA/Fuser/issues/585
TEST_F(NVFuserTest, FusionDanglingUnaryOp_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Create a segmented Fusion. We call segment_set here to ensure the whole
  // Fusion cannot be scheduled. This triggers segmentation, so that
  // forwardInputs() is called. The structure of this Fusion is not important;
  // it is only important that it must be segmented.
  auto size = IrBuilder::create<Val>(5L);
  auto tv0 = full({size}, fusion->zeroVal(), DataType::Int);
  auto tv1 = segment_set(tv0);
  fusion->addOutput(tv1);

  // Now take in an input that has a chain of UnaryOp uses that terminates in a
  // Val with no uses. This triggers a segfault in forwardInputs().
  Val* alpha = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(alpha);
  neg(castOp(DataType::Float, alpha));

  FusionExecutorCache executor_cache(std::move(fusion));

  auto cg_outputs = executor_cache.runFusionWithInputs({11});

  testValidate(executor_cache.fusion(), cg_outputs, {11}, __LINE__, __FILE__);
}

// converted from https://github.com/NVIDIA/Fuser/issues/443
TEST_F(NVFuserTest, FusionInstanceNormNHWC_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  double k_eps = 1e-05;
  auto shape = std::vector<int64_t>{256, 28, 28, 128};
  {
    DataType dtype = DataType::Half;
    auto tv0 = makeContigTensor(4, dtype);
    auto weight = makeContigTensor(1, dtype);
    auto bias = makeContigTensor(1, dtype);
    fusion->addInput(tv0);
    fusion->addInput(weight);
    fusion->addInput(bias);
    tv0 = castOp(DataType::Float, tv0);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);

    auto s1 = IrBuilder::create<Val>(k_eps);
    auto var_mean = variance_mean(tv0, {1, 2}, 0, true);
    auto tv_mean = var_mean.mean;
    auto tv_var = var_mean.var;
    auto tv_var_s1 = add(tv_var, s1);
    auto tv_sqrt = sqrt(tv_var_s1);
    auto tv_diff = sub(tv0, tv_mean);
    auto tv_div = div(tv_diff, tv_sqrt);
    auto tv_mul = mul(tv_div, weight);
    auto tv_out = add(tv_mul, bias);
    tv_out = castOp(DataType::Half, tv_out);

    fusion->addOutput(tv_out);
  }

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  auto t0 = at::randn(shape, options);
  auto t1 = at::randn(shape[3], options);
  auto t2 = at::randn(shape[3], options);

  auto var_mean = at::var_mean(t0, {1, 2}, 0, true);
  auto var = std::get<0>(var_mean);
  auto mean = std::get<1>(var_mean);
  auto t3 = (t0 - mean) / sqrt(var + k_eps);
  auto t4 = t3 * t1 + t2;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(fusion, cg_outputs, {t0, t1, t2}, {t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, VectorizeBackToBackReductions) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_shape{128, 256, 256};

  auto tv0 = makeContigConcreteTensor(input_shape);
  fusion->addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {2});

  auto output = sum(tv2, {1});
  fusion->addOutput(output);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({at_x});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  ASSERT_TRUE(runtime->isSegmented()) << "segmentation didn't happen";
  ASSERT_EQ(runtime->fusionSegments()->groups().size(), 2)
      << "segmentation didn't happen as expected";

  auto& heuristic_params =
      runtime->schedulerHeuristics()->heuristicsList().at(1);
  ASSERT_TRUE(heuristic_params->isA<ReductionParams>());
  auto rparams = heuristic_params->as<ReductionParams>();
  ASSERT_TRUE(rparams->vectorize_inner_reduction) << "Failed to vectorize";
  // On some hardware, the reduction heuristics may choose a
  // vectorization factor of 2.
  EXPECT_THAT(rparams->unroll_factor_inner_reduction, testing::AnyOf(2, 4))
      << "Unexpected vectorization factor";

  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, AllInputDtypes) {
  for (auto index_type : {DataType::Int, DataType::Int32}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(0, DataType::Double);
    auto tv1 = makeContigTensor(0, DataType::Double);
    tv1->setCpuScalar(true);
    auto d = IrBuilder::create<Val>(DataType::Double);
    auto f = IrBuilder::create<Val>(DataType::Float);
    auto h = IrBuilder::create<Val>(DataType::Half);
    auto i = IrBuilder::create<Val>(DataType::Int);
    auto idx = IrBuilder::create<Val>(DataType::Index);
    auto i32 = IrBuilder::create<Val>(DataType::Int32);
    auto b = IrBuilder::create<Val>(DataType::Bool);
    auto cf = IrBuilder::create<Val>(DataType::ComplexFloat);
    auto cd = IrBuilder::create<Val>(DataType::ComplexDouble);
    DataType ptr_type =
        PointerType{std::make_shared<DataType>(DataType::Float)};
    auto ptr = IrBuilder::create<Val>(ptr_type);
    DataType array_type =
        ArrayType{std::make_shared<DataType>(DataType::Float), 2};
    auto array = IrBuilder::create<Val>(array_type);
    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(d);
    fusion->addInput(f);
    fusion->addInput(h);
    fusion->addInput(i);
    fusion->addInput(idx);
    fusion->addInput(i32);
    fusion->addInput(b);
    fusion->addInput(cf);
    fusion->addInput(cd);
    fusion->addInput(ptr);
    fusion->addInput(array);

    auto output = d;
    output = IrBuilder::addExpr(output, f);
    output = IrBuilder::addExpr(output, castOp(DataType::Double, h));
    output = IrBuilder::addExpr(output, i);
    output = IrBuilder::addExpr(output, idx);
    output = IrBuilder::addExpr(output, i32);
    output = IrBuilder::addExpr(output, b);
    if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
      auto bf16 = IrBuilder::create<Val>(DataType::BFloat16);
      fusion->addInput(bf16);
      output = IrBuilder::addExpr(output, castOp(DataType::Double, bf16));
    }
    output = IrBuilder::addExpr(output, abs(cf));
    output = IrBuilder::addExpr(output, abs(cd));
    output = IrBuilder::addExpr(output, IrBuilder::derefExpr(ptr));
    output = IrBuilder::addExpr(
        output, IrBuilder::getItemExpr(array, PolymorphicValue(0L)));
    output = IrBuilder::addExpr(
        output, IrBuilder::getItemExpr(array, PolymorphicValue(1L)));
    output = add(tv0, output);
    output = add(tv1, output);

    fusion->addOutput(output);

    at::Tensor t0 = at::randn(
        {}, at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0));
    at::Tensor t1 =
        at::randn({}, at::TensorOptions().dtype(at::kDouble).device(at::kCPU));
    // Use page-locked memory so the pointer can be accessed both on host and on
    // device.
    at::Tensor t2 = at::randn(
        {},
        at::TensorOptions()
            .dtype(at::kFloat)
            .device(at::kCPU)
            .pinned_memory(true));

    KernelArgumentHolder args;
    args.push(t0);
    args.push(t1);
    args.push(2.3);
    args.push(4.5);
    args.push(6.7);
    args.push(8L);
    args.push(9L);
    args.push(10L);
    args.push(true);
    args.push(std::complex<double>(4.5, 6.7));
    args.push(std::complex<double>(8.9, 10.11));
    args.push(t2.data_ptr<float>());
    args.push(PolymorphicValue(std::vector<PolymorphicValue>{12.3, 45.0}));
    if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
      args.push(12.3); // bf16
    }

    auto ee = executor_utils::bindInputs(args, fusion.get());

    CompileParams opt{.index_type = index_type};

    KernelExecutor ke;
    ke.compile(fusion.get(), args, LaunchParams{}, opt);
    auto outputs = ke.run(args, {}, LaunchParams{}, opt);

    auto kernel_result = outputs[0].as<at::Tensor>().item<double>();
    auto expect = ee.evaluate(output).as<at::Tensor>().item<double>();
    EXPECT_NEAR(kernel_result, expect, 0.1);
  }
}

TEST_F(NVFuserTest, IndexDataTypePromotion) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto a = IrBuilder::create<Val>(DataType::Int);
  auto b = IrBuilder::create<Val>(DataType::Index);
  auto c = add(a, b);

  ExpressionEvaluator ee;
  ee.bind(a, 1L);
  ee.bind(b, 299792458L);
  EXPECT_EQ(ee.evaluate(c), 299792459L);
  EXPECT_EQ(c->dtype(), DataType::Index);
}

TEST_F(NVFuserTest, SymbolicOneBroadcasting) {
  // Test that if a tensor dimension's extent is one, no matter whether this
  // extent is constant 1 or symbolic 1, we always mark this ID as broadcasting.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto one = IrBuilder::create<Val>(1L);
  auto zero = sub(one, one);
  auto symbolic_one = add(zero, one);
  std::vector<Val*> shape{symbolic_one};
  auto tv = TensorViewBuilder()
                .ndims(1)
                .dtype(DataType::Float)
                .contiguity(true)
                .shape(shape)
                .build();
  ASSERT_EQ(tv->nDims(), 1);
  EXPECT_TRUE(tv->axis(0)->isBroadcast());
}

TEST_F(NVFuserTest, OpaqueTupleAsComplex) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  DataType dtype =
      OpaqueType::make<std::array<float, 2>>("Tuple<float, float>");

  auto tuple = IrBuilder::create<Val>(dtype);
  fusion.addInput(tuple);
  auto complex = bitCastOp(DataType::ComplexFloat, tuple);

  auto tv = full(
      {IrBuilder::create<Val>(1L, DataType::Index)},
      complex,
      DataType::ComplexFloat);

  fusion.addOutput(tv);

  KernelArgumentHolder args;
  args.push(Opaque(std::array<float, 2>{1.2, 3.4}));

  KernelExecutor ke;
  ke.compile(&fusion);
  auto outputs = ke.run(args);

  EXPECT_EQ(
      outputs[0].as<at::Tensor>().item<c10::complex<float>>(),
      c10::complex<float>(1.2, 3.4));
}

TEST_F(NVFuserTest, StructConstruct) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto real = IrBuilder::create<Val>(DataType::Float);
  auto imag = IrBuilder::create<Val>(DataType::Float);
  fusion.addInput(real);
  fusion.addInput(imag);

  auto struct_ = IrBuilder::structExpr({{"real", real}, {"imag", imag}});
  auto complex = bitCastOp(DataType::ComplexFloat, struct_);

  auto tv = full(
      {IrBuilder::create<Val>(1L, DataType::Index)},
      complex,
      DataType::ComplexFloat);

  fusion.addOutput(tv);

  KernelExecutor ke;
  ke.compile(&fusion);
  auto outputs = ke.run({1.2, 3.4});

  EXPECT_EQ(
      outputs[0].as<at::Tensor>().item<c10::complex<float>>(),
      c10::complex<float>(1.2, 3.4));
}

// Test that Int constants used in expressions that would overflow for 32-bit
// ints do not overflow in the generated kernel.
TEST_F(NVFuserTest, ConstLongExpressions) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto s0 = IrBuilder::create<Val>(65536L, DataType::Int);
  auto s1 = mul(s0, s0);
  // If s1 is printed in the kernel as "65536 * 65536" then it might be
  // evaluated at compiled time or not, and either way it will _likely_ be
  // evaluated using 32-bit ints instead of 64-bit as intended. The compiler
  // does this because promoting literals to long would change the value of the
  // expression.
  // See https://github.com/NVIDIA/Fuser/pull/998

  auto tv0 = full({}, s1, DataType::Int);
  fusion->addOutput(tv0);

  KernelExecutor ke;
  ke.compile(fusion);

  auto outputs = ke.run({});

  testValidate(fusion, outputs, {}, __LINE__, __FILE__);
}

// Related to https://github.com/NVIDIA/Fuser/issues/1084.
// On H100, the generated kernel is vectorized by 8, has a serial batch
// of 5. It uses 106 threads without padding, nsys shows kernel
// duration is 0.271 ms. If eliminate predicate for RNG ops by comment out
// predicateRNGOp(), the kernel duration is increased to 0.376 ms.
TEST_F(NVFuserTest, PredicateRNGOps) {
  int64_t size = 4224;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion->addInput(tv0);
  auto tv1 = rand_like(tv0);
  auto tv2 = rand_like(tv0);
  auto tv3 = rand_like(tv0);
  auto tv4 = rand_like(tv0);
  auto tv5 = add(tv1, tv2);
  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv5, tv6);
  auto tv8 = castOp(DataType::Half, tv7);
  auto tv9 = set(tv8);
  fusion->addOutput(tv9);

  tv1->split(-1, 8);
  tv1->split(-2, 5);
  tv1->split(-2, 1);
  tv1->axis(-2)->parallelize(ParallelType::Unswitch);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  TransformPropagator propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);

  tv9->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  // check RNGOp is predicated without else branch.
  class PredicateChecker : public kir::IrVisitor {
   public:
    using kir::IrVisitor::handle;
    bool predicate_rngop = false;

   private:
    void handle(kir::RNGOp* uop) final {
      for (auto expr : scope_exprs_) {
        if (!expr->isA<kir::IfThenElse>() ||
            expr->as<kir::IfThenElse>()->hasElse()) {
          continue;
        }
        if (!expr->as<kir::IfThenElse>()->predicate()->isTrivial()) {
          predicate_rngop = true;
        }
      }
    }
  } pred_checker;
  GpuLower gpulw(fusion);
  pred_checker.handle(gpulw.run()->topLevelExprs());
  ASSERT_TRUE(pred_checker.predicate_rngop);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::zeros({2048, size}, options);

  KernelExecutor ke;
  ke.compile(fusion, {t0});

  auto cg_outputs = ke.run({t0});
}

TEST_F(NVFuserTest, LoweringHook) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  GpuLower gpulw(&fusion);
  bool executed = false;
  gpulw.passes().push_back(
      {"test",
       [&executed](const std::vector<Expr*>& exprs) -> std::vector<Expr*> {
         executed = true;
         return exprs;
       }});
  EXPECT_FALSE(executed);
  gpulw.run();
  EXPECT_TRUE(executed);
}

// Test that 3D reductions with broadcasts as the inner-most non-reduction
// dimension are successfully scheduled.
// See https://github.com/NVIDIA/Fuser/issues/1471
TEST_F(NVFuserTest, Reduction3DWithBroadcast) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = TensorViewBuilder()
                 .dtype(DataType::Double)
                 .contiguity({true, true, true, std::nullopt})
                 .shape({-1, -1, -1, 1})
                 .build();
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {2, 0});
  fusion->addOutput(tv1);

  // Copy unscheduled fusion for later use in validation
  auto unsched_fusion_ptr = std::make_unique<Fusion>(*fusion);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  auto t0 = at::randn({8, 7, 5, 1}, options);

  auto cg_outputs =
      scheduleAndRun(fusion, SchedulerType::Reduction, {t0}).outputs;
  testValidate(unsched_fusion_ptr.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test 3D reductions with constant domains.
// https://github.com/NVIDIA/Fuser/issues/1590
TEST_F(NVFuserTest, Reduction3DConstantIterationDomain) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  long x = 2L, y = 8L, z = 8L, w = 16L, h = 512L;
  auto tv0 = TensorViewBuilder()
                 .ndims(5)
                 .shape({-1, -1, -1, -1, -1})
                 .contiguity({true, true, true, true, true})
                 .strideOrder({4, 3, 2, 0, 1})
                 .build();
  fusion->addInput(tv0);
  auto tv1 = full(
      {IrBuilder::create<Val>(x),
       IrBuilder::create<Val>(y),
       IrBuilder::create<Val>(z),
       IrBuilder::create<Val>(w),
       IrBuilder::create<Val>(h)},
      fusion->oneVal(),
      DataType::Float);
  auto tv2 = mul(tv0, tv1);
  auto tv3 = sum(tv2, {2, 4});
  fusion->addOutput(tv3);

  // tv1 is a constant tensor, and its domains are constant.
  // Its constant domains are used in ExactMappedExtentSubstitutionPass
  // to substitute the domains of tv0.
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 =
      at::randn({x, y, z, w, h}, options)
          .as_strided({x, y, z, w, h}, {w * h * z * y, w * h * z, w * h, 1, w});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});

  auto ref = t0.to(at::kDouble).sum({2, 4});
  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);
}

// Test that architectures before Ampere give helpful error message if BFloat16
// is used
TEST_F(NVFuserTest, UnsupportedBFloat) {
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    GTEST_SKIP() << "Requires GPU capability below 8.0 to run.\n";
  }

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3}, DataType::BFloat16);
  auto tv1 = set(tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  KernelExecutor ke;
  EXPECT_THAT(
      [&]() { ke.compile(&fusion); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("Reason: Fusion contains BFloat16")));
}

// Issue #1470 reproduction:
// `nvfuser_index_t T5[4]` is aliased as `Array<float, 4> T9`.
// `float T4[4]` is aliased as `auto& T10 = T4`.
// Using `T9` and `T10` in `welfordGroupOuter` function causes a compilation
// error due to type mismatch: `T9` is an aligned array, while `T10` is a
// regular array. Should generate fun<>(T9.array, T10) instead of
// fun<>(T9, T10).
TEST_F(NVFuserTest, TemplateFunctionTypeMismatch) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 8192;
  const int hidden_size = 1024;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = add(tv1, tv1);
  auto tv3 = Welford(tv2, {0});
  auto tv4 = broadcast(tv3.avg, {true, false});
  auto tv5 = div(tv2, tv4);

  auto tv6 = exp(tv5);
  auto tv7 = Welford(tv6, {0});
  auto tv8 = broadcast(tv7.avg, {true, false});
  auto tv9 = div(tv6, tv8);

  fusion->addOutput(tv5);
  fusion->addOutput(tv9);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  scheduleAndRun(fusion, SchedulerType::OuterPersistent, {t0});
}

// Test block reduction across TIDx and TIDz
TEST_F(NVFuserTest, BlockReduction3D) {
  auto test = [](const int tidx, const int tidy, const int tidz) {
    Fusion fusion;
    FusionGuard fg(&fusion);
    std::vector<int64_t> shape({tidz, tidy, tidx});

    auto tv0 = makeConcreteTensor(shape);
    fusion.addInput(tv0);
    auto tv1 = sum(tv0, {0, 2});
    fusion.addOutput(tv1);

    tv1->axis(0)->parallelize(ParallelType::TIDz);
    tv1->axis(1)->parallelize(ParallelType::TIDy);
    tv1->axis(2)->parallelize(ParallelType::TIDx);

    scheduler_utils::parallelizeAllLike(tv1);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn(shape, options);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto cg_outputs = ke.run({t0});
    auto ref = t0.sum(0).sum(-1);
    testValidate(&fusion, cg_outputs, {t0}, {ref}, __LINE__, __FILE__);
  };
  // tested locally with i,j,k +=2, change to i,j,k *=2 to reduce CI time.
  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int max_threads_per_blk = (int)properties->maxThreadsPerBlock;
  for (int i = 2; i <= 32; i *= 2) {
    for (int j = 2; j <= 32; j *= 2) {
      for (int k = 2; k <= 32; k *= 2) {
        if (i * j * k <= max_threads_per_blk) {
          test(i, j, k);
        }
      }
    }
  }
}

// Simple test to merge an inner domain as an outer input
TEST_F(NVFuserTest, ReverseMerge) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(1, 0);

  ASSERT_EQ(tv1->nDims(), 1);
  auto merge = dynamic_cast<Merge*>(tv1->axis(0)->definition());
  ASSERT_NE(merge, nullptr);
  ASSERT_EQ(merge->outer(), tv1->getLogicalDomain().at(1));
  ASSERT_EQ(merge->inner(), tv1->getLogicalDomain().at(0));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({11, 12}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  ASSERT_TRUE(t0.equal(cg_outputs[0].as<at::Tensor>()));
}

TEST_F(NVFuserTest, FusionCpAsyncPredicateAvoidIllegalMemoryAccess) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int m = 33, n = 48;
  TensorView* tv0 = makeContigTensor(2);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto tvs = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  tvs->setMemoryType(MemoryType::Shared);

  tvs->split(-1, 4);
  tvs->axis(-1)->parallelize(ParallelType::Vectorize);
  tvs->axis(-2)->parallelize(ParallelType::TIDx);
  tvs->axis(-3)->parallelize(ParallelType::BIDx);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({m, n}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  ASSERT_TRUE(t0.equal(cg_outputs[0].as<at::Tensor>()));
}

TEST_F(NVFuserTest, DecoupledDomains1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // XX shape structure:
  //
  // domain 0: [I0, I1...    I2  I3} domain 1
  //             \  /         \  /
  //            merge         merge
  //             /  \         /  \.
  // domain 1: {I4  I5    ...I6, I7] domain 0
  // where domain 0 is [I0, I1, I6, I7], and
  //       domain 1 is [I4, I5, I2, I3]
  auto create_xx_shape_structure = []() {
    auto s0 = IrBuilder::create<Val>(DataType::Index);
    auto s1 = IrBuilder::create<Val>(DataType::Index);
    auto s2 = IrBuilder::create<Val>(DataType::Index);
    auto s3 = IrBuilder::create<Val>(DataType::Index);
    auto id0 = IterDomainBuilder(s0->fusion()->zeroVal(), s0).build();
    auto id1 = IterDomainBuilder(s1->fusion()->zeroVal(), s1).build();
    auto id2 = IterDomainBuilder(s2->fusion()->zeroVal(), s2).build();
    auto id3 = IterDomainBuilder(s3->fusion()->zeroVal(), s3).build();
    std::unordered_set<IterDomain*> all_ids{id0, id1, id2, id3};
    AbstractTensor dom0({id0, id1, id2, id3});
    AbstractTensor dom1 = dom0;
    dom0.merge(2);
    all_ids.insert(dom0[2].as<IterDomain*>());
    dom0.split(2, 256);
    all_ids.insert(dom0[2].as<IterDomain*>());
    all_ids.insert(dom0[3].as<IterDomain*>());
    dom1.merge(0);
    all_ids.insert(dom1[0].as<IterDomain*>());
    dom1.split(0, 256);
    all_ids.insert(dom1[0].as<IterDomain*>());
    all_ids.insert(dom1[1].as<IterDomain*>());
    return std::make_tuple(
        dom0.as<IterDomain*>(), dom1.as<IterDomain*>(), all_ids);
  };
  auto [logical_xx0, logical_xx1, logical_all] = create_xx_shape_structure();
  auto [root_xx0, root_xx1, root_all] = create_xx_shape_structure();
  auto [alloc_xx0, alloc_xx1, alloc_all] = create_xx_shape_structure();
  auto [loop_xx0, loop_xx1, loop_all] = create_xx_shape_structure();

  auto concat = [](auto x, auto y, auto z, auto q) {
    std::vector<IterDomain*> result;
    result.reserve(x.size() + y.size() + z.size() + q.size());
    result.insert(result.end(), x.begin(), x.end());
    result.insert(result.end(), y.begin(), y.end());
    result.insert(result.end(), z.begin(), z.end());
    result.insert(result.end(), q.begin(), q.end());
    return decltype(x)(result.begin(), result.end());
  };
  auto logical_domain = concat(logical_xx1, root_xx0, alloc_xx0, loop_xx0);
  auto root_domain = concat(logical_xx0, root_xx1, alloc_xx0, loop_xx0);
  auto allocation_domain = concat(logical_xx0, root_xx0, alloc_xx1, loop_xx0);
  auto loop_domain = concat(logical_xx0, root_xx0, alloc_xx0, loop_xx1);
  std::vector<std::optional<bool>> contiguity(allocation_domain.size(), true);

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      root_domain, logical_domain, allocation_domain, loop_domain, contiguity);
  TensorView* tv = IrBuilder::create<TensorView>(td, DataType::Float);
  auto all_ids = concat(logical_all, root_all, alloc_all, loop_all);
  auto tv_all_vec = tv->domain()->allIDs();
  std::unordered_set<IterDomain*> tv_all(tv_all_vec.begin(), tv_all_vec.end());
  EXPECT_EQ(tv_all, all_ids);
}

TEST_F(NVFuserTest, DecoupledDomains2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto s0 = IrBuilder::create<Val>(DataType::Index);
  auto s1 = IrBuilder::create<Val>(DataType::Index);
  auto s2 = IrBuilder::create<Val>(DataType::Index);
  auto s3 = IrBuilder::create<Val>(DataType::Index);
  auto id0 = IterDomainBuilder(s0->fusion()->zeroVal(), s0).build();
  auto id1 = IterDomainBuilder(s1->fusion()->zeroVal(), s1).build();
  auto id2 = IterDomainBuilder(s2->fusion()->zeroVal(), s2).build();
  auto id3 = IterDomainBuilder(s3->fusion()->zeroVal(), s3).build();
  std::unordered_set<IterDomain*> all_ids{id0, id1, id2, id3};

  AbstractTensor root_ids({id0, id1, id2, id3});

  auto schedule = [&]() {
    AbstractTensor domain = root_ids;
    domain.merge(2);
    all_ids.insert(domain[2].as<IterDomain*>());
    domain.split(2, 256);
    all_ids.insert(domain[2].as<IterDomain*>());
    all_ids.insert(domain[3].as<IterDomain*>());
    domain.merge(0);
    all_ids.insert(domain[0].as<IterDomain*>());
    domain.split(0, 256);
    all_ids.insert(domain[0].as<IterDomain*>());
    all_ids.insert(domain[1].as<IterDomain*>());
    return domain.as<IterDomain*>();
  };

  // Create a TensorView that, all the domains are transformed from a common
  // topologically root IDs [I0, I1, I2, I3] separately, so that to traverse
  // between any two domains, the traversal requires both forward and backward.
  auto logical_domain = schedule();
  auto root_domain = schedule();
  auto allocation_domain = schedule();
  auto loop_domain = schedule();
  std::vector<std::optional<bool>> contiguity(allocation_domain.size(), true);

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      root_domain, logical_domain, allocation_domain, loop_domain, contiguity);
  TensorView* tv = IrBuilder::create<TensorView>(td, DataType::Float);
  auto tv_all_vec = tv->domain()->allIDs();
  std::unordered_set<IterDomain*> tv_all(tv_all_vec.begin(), tv_all_vec.end());
  EXPECT_EQ(tv_all, all_ids);
}

TEST_F(NVFuserTest, BroadcastFromNowhere) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto us = [](const std::vector<IterDomain*>& b) {
    return std::unordered_set<IterDomain*>(b.begin(), b.end());
  };

  std::unordered_set<IterDomain*> all_ids;
  auto tv0 = makeSymbolicTensor(1); // [I]
  all_ids.insert(tv0->axis(0));
  EXPECT_EQ(all_ids, us(tv0->getLoopDomain()));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->broadcast(0); // [b, I]
  EXPECT_TRUE(tv0->axis(0)->isBroadcast());
  EXPECT_EQ(tv0->axis(0)->extent()->value(), 1);
  EXPECT_EQ(all_ids.count(tv0->axis(0)), 0);
  all_ids.insert(tv0->axis(0));
  EXPECT_EQ(all_ids, us(tv0->getLoopDomain()));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->broadcast(2, 2); // [b, I, b]
  EXPECT_TRUE(tv0->axis(2)->isBroadcast());
  EXPECT_EQ(tv0->axis(2)->extent()->value(), 2);
  EXPECT_EQ(all_ids.count(tv0->axis(2)), 0);
  all_ids.insert(tv0->axis(2));
  EXPECT_EQ(all_ids, us(tv0->getLoopDomain()));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->broadcast(-1, 3); // [b, I, b, b]
  EXPECT_TRUE(tv0->axis(-1)->isBroadcast());
  EXPECT_EQ(tv0->axis(-1)->extent()->value(), 3);
  EXPECT_EQ(all_ids.count(tv0->axis(3)), 0);
  all_ids.insert(tv0->axis(3));
  EXPECT_EQ(all_ids, us(tv0->getLoopDomain()));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->broadcast(1, 4); // [b, b, I, b, b]
  EXPECT_TRUE(tv0->axis(1)->isBroadcast());
  EXPECT_EQ(tv0->axis(1)->extent()->value(), 4);
  EXPECT_EQ(all_ids.count(tv0->axis(1)), 0);
  all_ids.insert(tv0->axis(1));
  EXPECT_EQ(all_ids, us(tv0->getLoopDomain()));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->merge(1); // [b, b*I, b, b]
  all_ids.insert(tv0->axis(1));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  tv0->merge(2); // [b, b*I, b*b]
  EXPECT_TRUE(tv0->axis(2)->isBroadcast());
  all_ids.insert(tv0->axis(2));
  EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv0->getLoopDomain(),
      tv0->getLogicalDomain(),
      tv0->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv0->getLogicalDomain(),
      tv0->getLoopDomain(),
      tv0->domain()->additionalIDs());

  while (tv0->nDims() > 1) {
    tv0->merge(0);
    all_ids.insert(tv0->axis(0));
    EXPECT_EQ(all_ids, us(tv0->domain()->allIDs()));
    ir_utils::validateDomainEquivalence(
        tv0->getLoopDomain(),
        tv0->getLogicalDomain(),
        tv0->domain()->additionalIDs());
    ir_utils::validateDomainEquivalence(
        tv0->getLogicalDomain(),
        tv0->getLoopDomain(),
        tv0->domain()->additionalIDs());
  }

  auto tv1 = makeSymbolicTensor(0);
  EXPECT_EQ(tv1->nDims(), 0);
  EXPECT_TRUE(tv1->getLoopDomain().empty());
  EXPECT_TRUE(tv1->domain()->allIDs().empty());
  ir_utils::validateDomainEquivalence(
      tv1->getLoopDomain(),
      tv1->getLogicalDomain(),
      tv1->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv1->getLogicalDomain(),
      tv1->getLoopDomain(),
      tv1->domain()->additionalIDs());

  tv1->broadcast(0);
  EXPECT_TRUE(tv1->axis(0)->isBroadcast());
  EXPECT_EQ(us({tv1->axis(0)}), us(tv1->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv1->getLoopDomain(),
      tv1->getLogicalDomain(),
      tv1->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv1->getLogicalDomain(),
      tv1->getLoopDomain(),
      tv1->domain()->additionalIDs());

  auto tv2 = makeSymbolicTensor(0);
  EXPECT_EQ(tv2->nDims(), 0);
  EXPECT_TRUE(tv2->getLoopDomain().empty());
  EXPECT_TRUE(tv2->domain()->allIDs().empty());
  ir_utils::validateDomainEquivalence(
      tv2->getLoopDomain(),
      tv2->getLogicalDomain(),
      tv2->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv2->getLogicalDomain(),
      tv2->getLoopDomain(),
      tv2->domain()->additionalIDs());

  tv2->broadcast(-1);
  EXPECT_TRUE(tv2->axis(0)->isBroadcast());
  EXPECT_EQ(us({tv2->axis(0)}), us(tv2->domain()->allIDs()));
  ir_utils::validateDomainEquivalence(
      tv2->getLoopDomain(),
      tv2->getLogicalDomain(),
      tv2->domain()->additionalIDs());
  ir_utils::validateDomainEquivalence(
      tv2->getLogicalDomain(),
      tv2->getLoopDomain(),
      tv2->domain()->additionalIDs());
}

TEST_F(NVFuserTest, BroadcastFromNowhereFusion) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // auto tv0 = makeSymbolicTensor(2);
  auto tv0 = makeConcreteTensor({4});
  fusion.addInput(tv0);
  // auto tv1 = makeSymbolicTensor(2);
  auto tv1 = makeConcreteTensor({2, 4});
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);
  tv2->broadcast(0);
  for (auto tv : {tv1, tv2, tv3, tv4}) {
    tv->merge(0);
    tv->split(0, 256);
#if 0
    // TODO: sync analysis could not handle this yet
    tv->axis(1)->parallelize(ParallelType::TIDx);
    tv->axis(0)->parallelize(ParallelType::BIDx);
#endif
  }

#if 0
  // TODO: Inlining not supported yet
  inlineMost();
  for (auto tv : {tv1, tv2, tv3}) {
    EXPECT_EQ(tv->getComputeAtPosition(), 2);
  }
#endif

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // TODO: use larger tensor size
  at::Tensor t0 = at::randn({4}, options);
  at::Tensor t1 = at::randn({2, 4}, options);
  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});
  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// https://github.com/NVIDIA/Fuser/issues/2488
TEST_F(NVFuserTest, ReplayRFactorMergeBcast) {
  const std::vector<int64_t> input_shape = {256, 1, 1, 4};
  // test rFactor, merge of two bcast IDs generate a bcast ID
  {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);
    TensorView* tv0 = makeConcreteTensor(input_shape);
    fusion.addInput(tv0);
    auto tv1 = sum(tv0, {-1});
    fusion.addOutput(tv1);
    // {256, 1, 1, 4}
    tv1->merge(1, 2);
    // {256, 1*1, 4}
    tv1->merge(0, 1);
    // {256*1*1, 4}
    tv1->split(-1, 2);
    // {256*1*1, 4/2, 2}
    auto tv2 = tv1->rFactor({-2});
    for (auto expr : StmtSort::getExprsTo(
             {tv2->getLoopDomain().begin(), tv2->getLoopDomain().end()})) {
      if (auto merge = dynamic_cast<Merge*>(expr)) {
        if (merge->outer()->isBroadcast() && merge->inner()->isBroadcast()) {
          EXPECT_TRUE(merge->out()->isBroadcast())
              << "Merge of two broadcast IDs should generate a new broadcast "
                 "ID: "
              << merge->toString();
        }
      }
    }
  }
  // end-to-end validation
  {
    std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
    Fusion& fusion = *fusion_ptr.get();
    FusionGuard fg(&fusion);
    TensorView* tv0 = makeConcreteTensor(input_shape);
    fusion.addInput(tv0);
    auto tv1 = sum(tv0, {-1});
    fusion.addOutput(tv1);
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor at_x = at::ones(input_shape, options);
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto outputs = executor_cache.runFusionWithInputs({at_x});

    testValidate(&fusion, outputs, {at_x}, __LINE__, __FILE__);
  }
}

// This tests that we don't hit errors when we have multiple grid reductions
// scheduled with different static-sized extents mapped to the same parallel
// dimension.
// See https://github.com/NVIDIA/Fuser/issues/2634
TEST_F(NVFuserTest, MultipleDifferentSizeGridReduction) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeContigConcreteTensor({128});
  TensorView* tv1 = makeContigConcreteTensor({192});
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  TensorView* tv2 = max(tv0, {0});
  TensorView* tv3 = sum(tv1, {0});
  TensorView* tv4 = add(tv2, tv3);

  fusion.addOutput(tv4);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  inlineMost();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const at::Tensor t0 = at::randn({128}, options);
  const at::Tensor t1 = at::randn({192}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// See PR #2799
TEST_F(NVFuserTest, MoveNonConcretizedBroadcastInNormalization) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1));
  auto tv4 = squeeze(tv3, std::vector<int64_t>{0});
  auto tv5 = add(tv4, IrBuilder::create<Val>(2));
  auto tv6 = broadcast(tv5, {false, true});
  auto tv7 = add(tv0, tv6);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 1024}, options);

  auto fusion_copy = fusion;
  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::InnerPersistent, {t0});
  testValidate(&fusion_copy, cg_outputs.outputs, {t0}, __LINE__, __FILE__);

  // tv2 and tv3 have non-concretized broadcasts. Make sure they are
  // moved to the innermost position of the loop domain
  for (auto tv : {tv2, tv3}) {
    auto broadcast_domain = tv->getLogicalDomain().at(0);
    ASSERT_TRUE(broadcast_domain->isBroadcast());
    EXPECT_EQ(tv->getLoopDomain().back(), broadcast_domain)
        << "Non-concretized broadcast should be moved to the innermost "
           "position: "
        << tv->toString();
  }

  // One of the symptoms of issue 2685 was some tensors got
  // non-concretized broadcast domains at the outermost position of
  // the loop domain, preventing uniform inlining. Check if the
  // outermost loop domains of all tensors are mapped and inlined.
  auto ref_outermost = tv7->getLoopDomain().at(0);
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv->getLoopDomain().at(0), ref_outermost))
        << "Invalid outermost domain: " << tv->toString();

    EXPECT_TRUE(tv->getComputeAtPosition() >= 1)
        << "Invalid inlining position: " << tv->toString();
  }
}

// See PR #2799
TEST_F(NVFuserTest, MoveNonConcretizedBroadcastInPointwise) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, true, false});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1));
  auto tv4 = squeeze(tv3, std::vector<int64_t>{0, 1});
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input0 = at::randn({1024}, options);
  at::Tensor t1 = at::randn({1024}, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {input0, t1}).outputs;
  testValidate(&fusion, cg_outputs, {input0, t1}, __LINE__, __FILE__);

  // tv2 and tv3 have non-concretized broadcasts. Make sure they are
  // moved to the innermost position of the loop domain
  for (auto tv : {tv2, tv3}) {
    for (const auto i : arange(2)) {
      auto broadcast_domain = tv->getLogicalDomain().at(i);
      ASSERT_TRUE(broadcast_domain->isBroadcast());
      EXPECT_EQ(
          tv->getLoopDomain().at(tv->getLoopDomain().size() - 2 + i),
          broadcast_domain)
          << "Non-concretized broadcast should be moved to the innermost "
             "position: "
          << tv->toString();
    }
  }

  // One of the symptoms of issue 2685 was some tensors got
  // non-concretized broadcast domains at the outermost position of
  // the loop domain, preventing uniform inlining. Check if the
  // outermost loop domains of all tensors are mapped and inlined.
  auto ref_outermost = tv5->getLoopDomain().at(0);
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv->getLoopDomain().at(0), ref_outermost))
        << "Invalid outermost domain: " << tv->toString();

    EXPECT_TRUE(tv->getComputeAtPosition() >= 1)
        << "Invalid inlining position: " << tv->toString();
  }
}

// See PR #2799
TEST_F(NVFuserTest, MoveNonConcretizedBroadcastInReduction) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {true, false, false});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1));
  auto tv4 = squeeze(tv3, std::vector<int64_t>{0});
  auto tv5 = add(tv4, tv1);
  auto tv6 = sum(tv5, {1});
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({32, 1024}, options);
  at::Tensor t1 = at::randn({32, 1024}, options);

  Fusion fusion_copy = fusion;

  auto cg_outputs = scheduleAndRun(&fusion, SchedulerType::Reduction, {t0, t1});
  testValidate(&fusion_copy, cg_outputs.outputs, {t0, t1}, __LINE__, __FILE__);

  // tv2 and tv3 have non-concretized broadcasts. Make sure they are
  // moved to the innermost position of the loop domain
  for (auto tv : {tv2, tv3}) {
    auto broadcast_domain = tv->getLogicalDomain().at(0);
    ASSERT_TRUE(broadcast_domain->isBroadcast());
    EXPECT_EQ(tv->getLoopDomain().back(), broadcast_domain)
        << "Non-concretized broadcast should be moved to the innermost "
           "position: "
        << tv->toString();
  }

  // One of the symptoms of issue 2685 was some tensors got
  // non-concretized broadcast domains at the outermost position of
  // the loop domain, preventing uniform inlining. Check if the
  // outermost loop domains of all tensors are mapped and inlined.
  auto ref_outermost = tv6->getLoopDomain().at(0);
  IdModel id_model(&fusion, /*build_graphs=*/false);
  const auto& exact_graph = id_model.buildExactGraph();
  for (auto tv : fusion.allTvs()) {
    if (tv->isFusionInput()) {
      continue;
    }

    EXPECT_TRUE(exact_graph.disjointValSets().strictAreMapped(
        tv->getLoopDomain().at(0), ref_outermost))
        << "Invalid outermost domain: " << tv->toString();

    EXPECT_TRUE(tv->getComputeAtPosition() >= 1)
        << "Invalid inlining position: " << tv->toString();
  }
}

// See issue #2685 and PR #2799
TEST_F(NVFuserTest, Issue2685Repro) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{2, 288, 30, 80};

  auto tv0 = makeContigTensor(4);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);
  auto tv10 = makeContigConcreteTensor({-1, -1, -1, -1, 1});
  fusion.addInput(tv10);

  auto tv11 = squeeze(tv10, std::vector<int64_t>{4});
  auto tv12 = set(tv11);
  auto tv17 = set(tv11);
  auto tv18 = sum(tv17, {0, 2, 3});
  auto tv19 = broadcast(tv18, {true, false, true, true});
  auto tv20 = set(tv19);
  auto tv27 = set(tv20);
  auto tv28 = set(tv27);
  auto tv29 = expand(
      tv28,
      {IrBuilder::create<Val>(shape[0]),
       IrBuilder::create<Val>(-1),
       IrBuilder::create<Val>(shape[2]),
       IrBuilder::create<Val>(shape[3])});
  auto tv30 = mul(IrBuilder::create<Val>(2.60417e-05), tv29);
  auto tv13 = set(tv11);
  auto tv14 = sum(tv13, {0, 2, 3});
  auto tv15 = broadcast(tv14, {true, false, true, true});
  auto tv16 = set(tv15);
  auto tv22 = squeeze(tv16, std::vector<int64_t>{0, 2, 3});
  auto tv23 = mul(IrBuilder::create<Val>(0.5), tv22);
  auto tv24 = add(tv2, IrBuilder::create<Val>(3));
  auto tv25 = mul(tv23, tv24);
  auto tv31 = broadcast(tv25, {true, false, true, true});
  auto tv32 = set(tv31);
  auto tv33 = set(tv32);
  auto tv34 = expand(
      tv33,
      {IrBuilder::create<Val>(shape[0]),
       IrBuilder::create<Val>(-1),
       IrBuilder::create<Val>(shape[2]),
       IrBuilder::create<Val>(shape[3])});
  auto tv37 = mul(IrBuilder::create<Val>(2), tv34);
  auto tv35 = broadcast(tv1, {true, false, true, true});
  auto tv36 = set(tv35);
  auto tv38 = sub(tv0, tv36);
  auto tv39 = mul(tv37, tv38);
  auto tv40 = set(tv39);
  auto tv41 = add(tv30, tv40);
  auto tv42 = add(tv12, tv41);
  auto tv43 = castOp(DataType::Half, tv42);
  fusion.addOutput(tv43);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape[1], options);
  at::Tensor t2 = at::randn(shape[1], options);
  at::Tensor t10 = at::randn(shape, options).unsqueeze(-1);
  Fusion fusion_copy = fusion;
  auto cg_outputs = scheduleAndRun(
      &fusion, SchedulerType::InnerPersistent, {t0, t1, t2, t10});
  testValidate(
      &fusion_copy, cg_outputs.outputs, {t0, t1, t2, t10}, __LINE__, __FILE__);
}

// Check that extents are properly replaced by replaceSymbolicSizes lowering
// pass
TEST_F(NVFuserTest, ReplaceSymbolicSizes) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = add(tv0, tv1);
  auto tv4 = full(
      {IrBuilder::create<Val>(5, DataType::Index)},
      IrBuilder::create<Val>(2.0, DataType::Float),
      DataType::Float);
  auto tv5 = mul(tv2, tv4);

  fusion->addOutput(tv3);
  fusion->addOutput(tv5);

  replaceSymbolicSizes(fusion);

  // tv0's extents map to their corresponding getMetaData expressions
  EXPECT_EQ(
      tv0->axis(0)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[0] )");
  EXPECT_EQ(
      tv0->axis(1)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[1] )");
  EXPECT_EQ(
      tv1->axis(0)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[0] )");
  EXPECT_EQ(
      tv1->axis(1)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[1] )");
  EXPECT_EQ(
      tv3->axis(0)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[0] )");
  EXPECT_EQ(
      tv3->axis(1)->extent()->toInlineString(),
      "( (( (( getMetaData(T0) )).logical_size ))[1] )");

  EXPECT_EQ(tv2->axis(0)->extent()->toInlineString(), "5");
  EXPECT_EQ(tv5->axis(0)->extent()->toInlineString(), "5");
}

// Make sure BestEffortReplay with error_on_failure=false does not
// complain about missing root-to-logical IterDomain ops
TEST_F(NVFuserTest, BestEffortReplayWithMismatchedRootToLogical) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({2, 4});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = reshape(tv1, {2, 4}, {8});
  fusion.addOutput(tv2);

  // This split does not exist in tv2
  tv1->split(0, 1);

  // Due to the split of tv1, BestEffortReplay would not find any
  // matching transformations. If error_on_failure is true, it would
  // result in an error.
  EXPECT_THAT(
      [&]() {
        BestEffortReplay replay(
            tv2->getLoopDomain(),
            tv1->getLoopDomain(),
            PairwiseLogicalDomainMap(tv1, tv2).mapProducerToConsumer(),
            /*replay_forward_id_map=*/{},
            /*target_forward_id_map=*/{},
            /*skip_replay_swizzle=*/false,
            /*skip_target_swizzle=*/false,
            /*skip_resize=*/false,
            /*error_on_failure=*/true);
      },
      ::testing::ThrowsMessage<nvfuser::nvfError>(
          ::testing::HasSubstr("conflicts with an root-to-logical call")));

  // Should not result in an error as error_on_failure is false
  BestEffortReplay replay(
      tv2->getLoopDomain(),
      tv1->getLoopDomain(),
      PairwiseLogicalDomainMap(tv1, tv2).mapProducerToConsumer(),
      /*replay_forward_id_map=*/{},
      /*target_forward_id_map=*/{},
      /*skip_replay_swizzle=*/false,
      /*skip_target_swizzle=*/false,
      /*skip_resize=*/false,
      /*error_on_failure=*/false);
}

TEST_F(NVFuserTest, RAWSync) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv2->merge(0);
  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);

  // Since tv2 is not inlined and tv2 and tv3 are both parallelized,
  // tv2 as a producer of tv3 requires a synchronization with tv2
  // placed on shared memory. Lowering the fusion should fail.
  EXPECT_THAT(
      [&]() { GpuLower(&fusion).run(); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "Producer is required to be in Global, Shared or Tensor Memory based "
          "on parallelization strategy. RAW flags: (threadIdx.x)")));
}

// Test `DistributedTransformerTest.Backward/__bfloat` has bool type tensor
// if copied to shared memory using async copy, will trigger a bug as described
// in https://github.com/NVIDIA/Fuser/issues/3273
// This test checks pointer to bool is not treated as data type bool when
// generating PTX code for kir::Asm, e.g. async copy.
TEST_F(NVFuserTest, CpAsyncDataTypeBool) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto dtype = DataType::Bool;
  int m = 33, n = 128;
  auto tv0 = makeContigConcreteTensor({m, n}, dtype);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  tv1->setMemoryType(MemoryType::Shared);
  tv1->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::CpAsync);
  tv1->definition()->as<LoadStoreOp>()->setCacheOp(CacheOp::Unspecified);
  auto tv2 = castOp(DataType::Int32, tv1);
  fusion.addOutput(tv2);

  for (auto tv : {tv0, tv1, tv2}) {
    tv->split(1, 4);
  }
  for (auto tv : {tv0, tv1, tv2}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }
  tv1->axis(2)->parallelize(ParallelType::Vectorize);

  inlineMost();

  // randn doesn't support bool, ones is used instead
  auto at_dtype = data_type_to_aten(dtype);
  auto options = at::TensorOptions().dtype(at_dtype).device(at::kCUDA, 0);
  at::Tensor t0 = at::ones({m, n}, options);

  // Expected asm code is:
  // asm volatile(
  //   "{\n"
  //   "  .reg .pred p0; \n"
  //   "  setp.ne.b32 p0, %3, 0;\n"
  //   "  cp.async.ca.shared.global [%0], [%1], %2, p0;\n"
  //   "}\n"
  //   :
  //   :"r"((uint32_t)((toSmem(T1) + i0))),
  //    "l"(((T0.data + i0) + i1)),
  //    "n"(4LL),
  //    "r"((uint32_t)((!b3)))
  // );
  // If not correctly lowered, would trigger error in compile
  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// Intermediate IDs generaetd by rFactor should also remain
// reductions. See #3327 for more info.
TEST_F(NVFuserTest, RfactorIntermediateIDs) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  fusion.addOutput(tv1);

  tv1->merge(1, 2);
  tv1->split(1, 4);

  auto tv2 = tv1->rFactor({-1});

  EXPECT_TRUE(tv2->axis(-1)->isReduction());
  EXPECT_FALSE(tv2->axis(-2)->isReduction());

  auto split = dynamic_cast<Split*>(tv2->axis(-1)->definition());
  ASSERT_NE(split, nullptr);

  auto merge_out = split->in();
  EXPECT_TRUE(merge_out->isReduction());
}

// Simple test to make sure replacement with a dependent val is
// detected as an error
TEST_F(NVFuserTest, AvoidReplacingWithDependentVal) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto i0 = IrBuilder::create<Val>(DataType::Int);
  fusion.addInput(i0);

  auto i1 = mul(i0, IrBuilder::create<Val>(1, DataType::Int));

  auto tv0 = TensorViewBuilder().shape({i1}).build();
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  std::unordered_map<Val*, Val*> replacement_map;
  replacement_map.emplace(i0, i1);

  EXPECT_THAT(
      [&]() { ir_utils::replaceValue(&fusion, replacement_map); },
      testing::ThrowsMessage<nvfuser::nvfError>(testing::HasSubstr(
          "not allowed as it would result in a recursive definition")));
}

// Was also a repro of issue #3347
TEST_F(NVFuserTest, ReplaceSymbolicSizesPreferSimplerExtents) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = reshape(
      tv0,
      {mul(
          mul(tv0->axis(0)->extent(), tv0->axis(1)->extent()),
          tv0->axis(2)->extent())});
  auto tv3 =
      reshape(tv1, {mul(tv1->axis(0)->extent(), tv1->axis(1)->extent())});
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  ExpressionEvaluator expr_eval;

  expr_eval.bind(tv0->axis(0)->extent(), 2L);
  expr_eval.bind(tv0->axis(1)->extent(), 4L);
  expr_eval.bind(tv0->axis(2)->extent(), 8L);
  expr_eval.bind(tv1->axis(0)->extent(), 8L);
  expr_eval.bind(tv1->axis(1)->extent(), 8L);

  auto initial_info = DynamicTransform::getInitialInfo(&fusion);
  auto info = DynamicTransformConcretizationInfo(&initial_info, &expr_eval);

  DynamicTransform::concretizeFusion(&fusion, &info);

  replaceSymbolicSizes(&fusion);

  // All expr output tensors should use the extent of the tv3 since it
  // has only one merge, whereas tv2 has two merges
  // All expr output tensors should use the same extent.
  auto ref_ext = fusion.outputs().at(0)->as<TensorView>()->axis(0)->extent();

  // ref_ext should look like getMetaData(T1).logical_size[0] *
  // getMetaData(T1).logical_size[1]
  auto ext_def = dynamic_cast<BinaryOp*>(ref_ext->definition());
  ASSERT_NE(ext_def, nullptr);
  ASSERT_EQ(ext_def->getBinaryOpType(), BinaryOpType::Mul);
  auto lhs = ext_def->input(0);
  auto rhs = ext_def->input(1);
  ASSERT_NE(dynamic_cast<GetItem*>(lhs->definition()), nullptr);
  ASSERT_NE(dynamic_cast<GetItem*>(rhs->definition()), nullptr);

  for (auto expr : fusion.exprs()) {
    auto tv_output = ir_utils::getTvOutput(expr);
    ASSERT_EQ(tv_output->nDims(), 1);
    auto ext = tv_output->axis(0)->extent();
    EXPECT_EQ(ref_ext, ext) << "Reference: " << ref_ext->toString()
                            << ", actual: " << ext->toString();
  }
}

// Test that we are able to infer parallel dimensions even if they are not
// provided in loop domains. This is important for Hopper MMA since we
// parallelize TIDx on an allocation domain for the MmaOp output that is not in
// its loop domain.
TEST_F(NVFuserTest, ParallelDimensionsInAllocation) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  auto tv1 = neg(tv0);
  auto tv2 = exp(tv1);
  fusion.addOutput(tv2);

  IterDomain* merged_id = IterDomain::merge(tv1->axis(0), tv1->axis(1));
  tv1->setAllocationDomain({merged_id}, true);
  merged_id->parallelize(ParallelType::TIDx);

  GpuLower gpulw(&fusion);
  gpulw.run();

  Val* tidx_dim = gpulw.info().parallelDimensionMap().get(ParallelType::TIDx);
  ASSERT_TRUE(tidx_dim != nullptr);
}

// Check the topological ordering of TensorDomain::allIDs(). Repro of
// issue #3583
TEST_F(NVFuserTest, AllIdsMultipleDependencies) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10, 20});
  fusion.addInput(tv0);

  auto tv1 = slice(
      tv0,
      {{fusion.zeroVal(), IrBuilder::create<Val>(2)},
       {fusion.zeroVal(), tv0->getLogicalDomain().at(1)->extent()}});

  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->split(0, 4);
  tv1->split(0, 8);

  auto all_ids = tv1->domain()->allIDs();

  auto split2 = tv1->axis(0)->definition()->as<Split>();
  auto split1 = split2->input(0)->definition()->as<Split>();
  auto merge = split1->input(0)->definition()->as<Merge>();
  auto resize = merge->input(0)->definition()->as<Resize>();

  std::vector<Expr*> exprs{resize, merge, split1, split2};

  for (auto expr : exprs) {
    for (auto inp : ir_utils::filterByType<IterDomain>(expr->inputs())) {
      auto inp_it = std::find(all_ids.begin(), all_ids.end(), inp);
      for (auto out : ir_utils::filterByType<IterDomain>(expr->outputs())) {
        auto out_it = std::find(all_ids.begin(), all_ids.end(), out);
        EXPECT_LT(inp_it, out_it)
            << "Invalid ordering: " << out->toString() << " detected before "
            << inp->toString() << ". All IDs: " << toDelimitedString(all_ids)
            << "\n";
      }
    }
  }
}

// Repeating a broadcast ID. RepeatOp should be used.
TEST_F(NVFuserTest, RepeatBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = repeat(tv1, {1L, 2L});
  fusion.addOutput(tv2);

  EXPECT_TRUE(tv2->definition()->isA<RepeatOp>());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Repeating a non-broadcast ID. Should be translated to broadcast +
// expand + reshape.
TEST_F(NVFuserTest, RepeatNonBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeConcreteTensor({10});
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {2L});
  fusion.addOutput(tv1);

  ASSERT_TRUE(tv1->definition()->isA<ReshapeOp>());
  ASSERT_TRUE(tv1->definition()->input(0)->definition()->isA<ExpandOp>());
  ASSERT_TRUE(tv1->definition()
                  ->input(0)
                  ->definition()
                  ->input(0)
                  ->definition()
                  ->isA<BroadcastOp>());

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({10}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Repeating a mix of broadcast and non-broadcast IDs
TEST_F(NVFuserTest, RepeatBroadcastAndNonBroadcast) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  std::vector<int64_t> shape{2, 1, 3, 1};
  auto tv0 = makeConcreteTensor(shape);
  fusion.addInput(tv0);

  auto tv1 = repeat(tv0, {2L, 2L, 2L, 2L});
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, CastPrecision) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = castOp(DataType::BFloat16, tv1);
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(2, DataType::Index);
  fusion.addInput(tv3);

  auto tv4 = castOp(DataType::Int, tv3);
  fusion.addOutput(tv4);

  auto tv1_precision = ir_utils::getPrecisionOfProducerConsumerTensorsBit(
      tv1->definition()->as<UnaryOp>());
  ASSERT_TRUE(tv1_precision.has_value());
  EXPECT_EQ(tv1_precision->first, 16);
  EXPECT_EQ(tv1_precision->second, 32);

  auto tv2_precision = ir_utils::getPrecisionOfProducerConsumerTensorsBit(
      tv2->definition()->as<UnaryOp>());
  ASSERT_TRUE(tv2_precision.has_value());
  EXPECT_EQ(tv2_precision->first, 32);
  EXPECT_EQ(tv2_precision->second, 16);

  // Precision of type Index is not possible to determine until lowering
  auto tv4_precision = ir_utils::getPrecisionOfProducerConsumerTensorsBit(
      tv4->definition()->as<UnaryOp>());
  ASSERT_FALSE(tv4_precision.has_value());
}

TEST_F(NVFuserTest, RegisteredExactMappingWithExtentReplacment) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({16, 32});
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);

  auto tv3 = set(tv2);
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv1, tv4);
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  // Make the loop domains of tv3 and tv4 exact mapped with tv1's loop
  // domain
  scheduler_tools::scheduleLoopDomainsLike({tv3, tv4}, tv1->getLoopDomain());

  EXPECT_TRUE(fusion.hasRegisteredExactMappings());

  // tv3 and tv4 should have new cloned IDs that are exact mapped with
  // tv1
  auto registered_mappings = fusion.registeredExactMappings();
  auto registered_mappings_it = registered_mappings.find(tv3->axis(1));
  EXPECT_NE(registered_mappings_it, registered_mappings.end());
  const auto& registered_ids = registered_mappings_it->second;
  EXPECT_TRUE(registered_ids->has(tv4->axis(1)));
  EXPECT_TRUE(registered_ids->has(tv1->axis(1)));

  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    for (auto tv : {tv3, tv4}) {
      EXPECT_EQ(
          exact_graph.toGroups(tv->getLoopDomain()),
          exact_graph.toGroups(tv1->getLoopDomain()));
    }
  }

  // tv0 and tv1 are exact mapped. Since tv0 has static extents,
  // replaceSymbolicSizes will replace the symbolic extents of tv1 and tv2
  // with the static extents of tv0.
  replaceSymbolicSizes(&fusion);

  // Check if the exact mapping is still alive
  {
    IdModel id_model(&fusion, /*build_graphs=*/false);
    const auto& exact_graph = id_model.buildExactGraph();
    for (auto tv : {tv3, tv4}) {
      EXPECT_EQ(
          exact_graph.toGroups(tv->getLoopDomain()),
          exact_graph.toGroups(tv1->getLoopDomain()));
    }
  }
}

// Always use sharedMemPerBlockOptin to check memory usage in nvFuser,
// it already considers reservedSharedMemPerBlock
TEST_F(NVFuserTest, DeviceSharedMemoryLimit) {
  auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());
  int device_limit = (int)properties->sharedMemPerBlockOptin;
  int device_total = (int)properties->sharedMemPerMultiprocessor;
  int device_reserved = (int)properties->reservedSharedMemPerBlock;
  EXPECT_EQ(device_limit, device_total - device_reserved);
}

// Check that we can actually make use of every byte of shared memory on the
// device
TEST_F(NVFuserTest, UseAllSharedMemory) {
  const auto properties = at::cuda::getDeviceProperties(
      c10::Device(c10::DeviceType::CUDA, 0).index());

  // This kernel requires some static smem for some reason. We validate that
  // here as well.
  constexpr int64_t expected_static_smem = 16L;
  const int64_t available_dyn_smem_bytes =
      (int64_t)properties->sharedMemPerBlockOptin - expected_static_smem;

  const PrimDataType dtype = DataType::Char;
  EXPECT_EQ(available_dyn_smem_bytes % dataTypeSizeByte(dtype), 0);
  const int64_t len = available_dyn_smem_bytes / dataTypeSizeByte(dtype);

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({len}, dtype);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  auto tv1_smem = tv1->cacheBefore();
  tv1_smem->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kChar).device(at::kCUDA, 0);
  at::Tensor t0 = at::randint(0, 128, {len}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);

  // check that we used the full device
  int64_t actual_smem = ke.lastLaunchParams().smem();
  EXPECT_EQ(actual_smem, available_dyn_smem_bytes);
  EXPECT_EQ(ke.getStaticSmemSize(), expected_static_smem);
}

TEST_F(NVFuserTest, SyncthreadsWithGmemIssue4741) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);

  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Global);

  // [TIDx, TIDy]
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDy);

  // [TIDy, TIDx]
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  GpuLower gpulw(&fusion);
  gpulw.run();
  auto kernel = gpulw.kernel();
  const auto exprs = ir_utils::flattenScopedExprs(kernel->topLevelExprs());
  EXPECT_TRUE(std::any_of(exprs.begin(), exprs.end(), [](Expr* expr) {
    return expr->isA<kir::BlockSync>();
  }));
}

// Repro of issue #4829
TEST_F(NVFuserTest, InliningPosWithVectorizedCastOps) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeContigTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, fusion.oneVal());
  auto tv3 = add(tv2, fusion.oneVal());
  auto tv4 = castOp(DataType::BFloat16, tv3);
  auto tv5 = castOp(DataType::Float, tv4);
  auto tv6 = eq(tv1, fusion.zeroVal());
  auto tv7 = where(tv6, tv2, tv5);
  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1024 * 1024}, options);
  auto t1 = at::randn({1024 * 1024}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Test file size should be up to 10K LoC. Create a new file for more tests.

} // namespace nvfuser
