// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <algorithm>
#include <iostream>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDAStream.h>

#include "c10/core/ScalarType.h"
#include "codegen.h"
#include "device_lower/analysis/bank_conflict.h"
#include "device_lower/lower2device.h"
#include "disjoint_set.h"
#include "exceptions.h"
#include "expr_evaluator.h"
#include "fusion.h"
#include "fusion_profiler.h"
#include "fusion_segmenter.h"
#include "ir/all_nodes.h"
#include "ir/builder.h"
#include "ir/graphviz.h"
#include "ir/iostream.h"
#include "ir/printer.h"
#include "ir/utils.h"
#include "iter_visitor.h"
#include "kernel_ir.h"
#include "kernel_ir_dispatch.h"
#include "logical_domain_map.h"
#include "macros.h"
#include "mma_type.h"
#include "ops/all_ops.h"
#include "preseg_passes/pre_segmenter.h"
#include "runtime/executor.h"
#include "runtime/executor_params.h"
#include "runtime/fusion_executor_cache.h"
#include "scheduler/all_schedulers.h"
#include "scheduler/matmul.h"
#include "scheduler/matmul_heuristic.h"
#include "scheduler/mma_utils.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"
#include "sys_utils.h"
#include "tests/cpp/utils.h"
#include "transform_replay.h"
#include "transform_rfactor.h"
#include "utils.h"
#include "validator_utils.h"

namespace nvfuser {

class MatmulTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
  };
};

class MatmulTestWithLayout : public MatmulTest,
                             public ::testing::WithParamInterface<MmaLayout> {
 protected:
  MmaLayout layout;
  void SetUp() override {
    layout = GetParam();
    MatmulTest::SetUp();
  }
};

using namespace at::indexing;

#define SKIP_IF_INSUFFICIENT_SMEM(params, data_types)                     \
  {                                                                       \
    int64_t estim = mma_utils::computeExpectedSharedMemoryUsage(          \
        params, data_types, true, true);                                  \
    int64_t avail = (int64_t)deviceAvailableSharedMemoryBytes();          \
    if (avail < estim) {                                                  \
      GTEST_SKIP() << "Insufficient shared memory to run test (" << estim \
                   << "B required but only " << avail << "B available)."; \
    }                                                                     \
  }

// Matmul test for Ampere MMA: across supported layouts
TEST_P(MatmulTestWithLayout, AmpereMatmul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Single batch dimension which is broadcast
TEST_P(MatmulTestWithLayout, AmpereMatmulBroadcastBatch) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  // Broadcast inputs to 1, M, 1, K and 1, 1, N, K
  tv0 = broadcast(tv0, {true, false, false, false});
  tv1 = broadcast(tv1, {true, false, false, false});
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref =
      atMatmul(
          inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout)
          .unsqueeze(0);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmperePrologueFusionBroadcast) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput2D(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmpereProloguePointwise) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv0 = castOp(DataType::Half, sin(tv0));
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  tv1 = castOp(DataType::Half, sin(tv1));
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.sin().to(at::kFloat),
      inputs.second.sin().to(at::kFloat),
      layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmpereMatmulBFloat16) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::BFloat16);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::BFloat16);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout, at::kBFloat16);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test for Ampere MMA: with pipelined gmem load
TEST_P(MatmulTestWithLayout, AmpereMatmulPipelineGmem) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  REQUIRE_DEVICE_SMEM_SIZE(70 << 10, 0);

  // Gmem pipeline stage
  for (auto stage : {3, 4}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

    auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
    auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

    fusion.addOutput(tv2);

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.tile_sizes = gemm_tile;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = stage;
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    KernelExecutor ke;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        ke.compile(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
    ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
        ke.compiledKernel()->kernel()));
    auto cg_outputs = ke.run({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(
        at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
  }
}

// Check that mma op is not predicated.
class PredicateChecker : public kir::IrVisitor {
 public:
  using kir::IrVisitor::handle;
  bool found_mma = false;

 private:
  void handle(kir::Asm* asm_) final {
#if IS_CPP20
    if (!asm_->code().starts_with("mma") &&
        !asm_->code().starts_with("wgmma")) {
#else
    if (asm_->code().substr(0, 3) != "mma" &&
        asm_->code().substr(0, 5) != "wgmma") {
#endif
      return;
    }
    found_mma = true;
    for (auto expr : scope_exprs_) {
      NVF_CHECK(
          !expr->isA<kir::IfThenElse>() ||
              expr->as<kir::IfThenElse>()->predicate()->isTrivial(),
          "MmaOp should't be predicated!",
          " Get predicate ",
          expr->as<kir::IfThenElse>()->predicate()->toInlineString());
    }
  }
};

// Matmul test for Ampere MMA: checking CTA Swizzles
TEST_P(MatmulTestWithLayout, AmpereSwizzle) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int dim = 8192;
  int M = dim, N = dim, K = dim;
  const auto all_orders = {
      MatmulParams::TileRasterizationOrder::RowMajor,
      MatmulParams::TileRasterizationOrder::ColumnMajor};

  REQUIRE_DEVICE_SMEM_SIZE(70 << 10, 0);

  auto test = [&](MmaLayout layout,
                  MatmulParams::TileRasterizationOrder order,
                  int swizzle,
                  float& runtime) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

    auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
    auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

    fusion.addOutput(tv2);

    OptimizationPass<preseg_passes::PreSegmenter>::runPass(&fusion);

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 3;

    mparams.cta_order = order;
    mparams.grid_traversal_factor = {swizzle, 1};

    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    if (!detectComputeSanitizer()) {
      ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
    }

    KernelExecutor ke;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        ke.compile(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
    ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
        ke.compiledKernel()->kernel()));
    auto cg_outputs = ke.run({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.01, 0.01));

    int gdimx = ke.lastLaunchParams().gdimx();
    int gdimy = ke.lastLaunchParams().gdimy();

    int expected_gdim_unswizzled = (dim + 128 - 1) / 128;
    int expected_gdimx = expected_gdim_unswizzled * swizzle;
    int expected_gdimy = (expected_gdim_unswizzled + swizzle - 1) / swizzle;

    NVF_CHECK(gdimx == expected_gdimx);
    NVF_CHECK(gdimy == expected_gdimy);

    if (!detectComputeSanitizer()) {
      FusionProfiler::stop();
      runtime = FusionProfiler::profile().kernel_time_ms;
      ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::Enable);
    } else {
      runtime = 0;
    }

    // This is a regression test for https://github.com/NVIDIA/Fuser/issues/95
    GpuLower gpulw(&fusion);
    PredicateChecker pred_checker;
    pred_checker.handle(gpulw.run()->topLevelExprs());
    ASSERT_TRUE(pred_checker.found_mma);
  };

  // Checking only a single layout to keep runtime short (compilation overhead)
  for (auto order : all_orders) {
    float runtime1 = 0;
    test(layout, order, 1, runtime1);

    float runtime4 = 0;
    test(layout, order, 4, runtime4);

    // GRID Swizzle requires further changes to work in main. So for now we
    // don't assert the perf benefit here.
    // if (!detectComputeSanitizer()) {
    // NVF_CHECK(runtime4 < runtime1);
    // }
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulRegCircularBuffer) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  REQUIRE_DEVICE_SMEM_SIZE(70 << 10, 0);

  // Gmem pipeline stage
  for (auto stage : {3, 4}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

    auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
    auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

    fusion.addOutput(tv2);

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = stage;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    KernelExecutor ke;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        ke.compile(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
    ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
        ke.compiledKernel()->kernel()));
    auto cg_outputs = ke.run({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(
        at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
  }
}

// Matmul test for Turing MMA: across supported layouts
TEST_P(MatmulTestWithLayout, TuringMatmul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Turing_16_8_16;
  mparams.tile_sizes = gemm_tile;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      7, 5, ke.compile(&fusion, {inputs.first, inputs.second}));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Test an end-to-end matmul case with swizzled smem
// data layout.

// Matmul test on Ampere using ldmatrix.x4 to load operands
TEST_P(MatmulTestWithLayout, AmpereMatmulLargeLoad) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  REQUIRE_DEVICE_SMEM_SIZE(98384, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 64);
  gemm_tile.warp_tile = GemmTile(64, 64, 64);
  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_16_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test for Turing MMA: across supported layouts
TEST_P(MatmulTestWithLayout, TuringMatmulLargeLoad) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Turing_16_16_16;
  mparams.tile_sizes = gemm_tile;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      7,
      5,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Tile layout check for symmetric 4-warp recipes
TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck4warp) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  REQUIRE_DEVICE_SMEM_SIZE(98384, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  // Symmetric tile with 16x16x16 macro,
  //  supports mn_size of multiple of 32,
  //  and k size multiple of 16.
  for (int mn_size : {32, 64, 96, 128, 160, 192}) {
    for (int k_size : {32, 48, 64}) {
      Fusion fusion;
      FusionGuard fg(&fusion);

      auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

      auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
      auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

      fusion.addOutput(tv2);

      MatMulTileOptions gemm_tile;
      gemm_tile.cta_tile = GemmTile(mn_size, mn_size, k_size);
      gemm_tile.warp_tile = GemmTile(mn_size / 2, mn_size / 2, k_size);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_16_16;
      mparams.tile_sizes = gemm_tile;
      mparams.async_gmem_load_operands = true;
      mparams.circular_buffer_options.circular_buffer_smem_write = true;
      mma_utils::MmaDataTypes data_types = {
          DataType::Half, DataType::Half, DataType::Float};
      std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
          mma_utils::generateSharedMemoryEpilogueHeuristics(
              gemm_tile,
              mparams.circular_buffer_options.smem_circular_buffer_stage,
              data_types,
              true,
              true);
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto inputs = matmulAtInput3DTuring(M, N, K, layout);

      KernelExecutor ke;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          8,
          0,
          ke.compile(
              &fusion,
              {inputs.first, inputs.second},
              LaunchParams(),
              matmul_cparams));
      EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
      auto cg_outputs = ke.run({inputs.first, inputs.second});
      ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
          ke.compiledKernel()->kernel()));
      auto tref = atMatmul(
          inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
      NVF_CHECK(
          at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001),
          "error :",
          (cg_outputs[0].as<at::Tensor>() - tref).abs().max(),
          "tile dim:",
          mn_size,
          " ",
          k_size);
    }
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck8warp) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  REQUIRE_DEVICE_SMEM_SIZE(98384, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  // ASymmetric tile with 16x16x16 macro,
  for (int m_size : {256}) {
    for (int n_size : {32, 64, 96, 128}) {
      for (int k_size : {32, 48, 64}) {
        Fusion fusion;
        FusionGuard fg(&fusion);

        auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

        auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
        auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

        fusion.addInput(tv0);
        fusion.addInput(tv1);

        tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
        tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
        auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

        fusion.addOutput(tv2);

        MatMulTileOptions gemm_tile;
        gemm_tile.cta_tile = GemmTile(m_size, n_size, k_size);
        gemm_tile.warp_tile = GemmTile(m_size / 4, n_size / 2, k_size);

        MatmulParams mparams;
        mparams.supported_vec_size = {8, 8, 4};
        mparams.mma_macro = MmaMacro::Ampere_16_16_16;
        mparams.tile_sizes = gemm_tile;
        mparams.async_gmem_load_operands = true;
        mparams.circular_buffer_options.circular_buffer_smem_write = true;
        mparams.circular_buffer_options.circular_buffer_smem_read = true;
        mparams.circular_buffer_options.smem_circular_buffer_stage = 2;
        mma_utils::MmaDataTypes data_types = {
            DataType::Half, DataType::Half, DataType::Float};
        std::tie(
            mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
            mma_utils::generateSharedMemoryEpilogueHeuristics(
                gemm_tile,
                mparams.circular_buffer_options.smem_circular_buffer_stage,
                data_types);

        SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
            ->schedule(&fusion, &mparams);

        auto inputs = matmulAtInput3DTuring(M, N, K, layout);

        KernelExecutor ke;
        NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
            8,
            0,
            ke.compile(
                &fusion,
                {inputs.first, inputs.second},
                LaunchParams(),
                matmul_cparams));
        ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
        ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
            ke.compiledKernel()->kernel()));
        auto cg_outputs = ke.run({inputs.first, inputs.second});
        auto tref = atMatmul(
            inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
        NVF_CHECK(
            at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
      }
    }
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck6warp) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  REQUIRE_DEVICE_SMEM_SIZE(98384, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  for (int k_size : {32, 48, 64}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

    auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
    auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

    fusion.addOutput(tv2);

    MatMulTileOptions gemm_tile;
    // 2 warp by 3 warp
    gemm_tile.cta_tile = GemmTile(192, 128, k_size);
    gemm_tile.warp_tile = GemmTile(64, 64, k_size);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_16_16;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 2;
    mma_utils::MmaDataTypes data_types = {
        DataType::Half, DataType::Half, DataType::Float};
    std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
        mma_utils::generateSharedMemoryEpilogueHeuristics(
            gemm_tile,
            mparams.circular_buffer_options.smem_circular_buffer_stage,
            data_types);
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    KernelExecutor ke;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        ke.compile(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
    ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
        ke.compiledKernel()->kernel()));
    auto cg_outputs = ke.run({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(
        at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
  }
}

// Matmul test on Ampere using ldmatrix.x4 to load operands
TEST_P(MatmulTestWithLayout, AmpereMatmulLargeLoadLargeK) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 2048;
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 64);
  gemm_tile.warp_tile = GemmTile(64, 64, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_16_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
}

// Matmul test for Ampere MMA: across supported layouts
TEST_P(MatmulTestWithLayout, AmpereSplitKLikeStridedBatchedMatmul) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int B = 2, M = 504, N = 136, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, ke.compile(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({t0, t1});
  auto tref = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmpereMatmulSmemEpilogue) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  constexpr bool ignore_occupancy_drop = true;
  // Keep multiples of 8 to keep vectorizable.
  int M = 4096, N = 4096, K = 4096;
  // This tests num_stages=0, which should be treated identically to
  // num_stages=1. It is put here to exercise this path to ensure we don't
  // crash in generateSharedMemoryEpilogueHeuristics.
  // See https://github.com/NVIDIA/Fuser/pull/1917 for more info
  for (int num_stages : {0, 2}) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

    auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
    auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

    fusion.addOutput(tv2);

    // The settings of cta_tile, warp_tile, and smem_circular_buffer_stage
    // have been purposefully selected to produce a constant occupancy of 25%.
    // This allows us to effectively evaluate the influence of the
    // use_smem_epilogue parameter on performance, since changing its value to
    // either true or false will not affect the occupancy rate.
    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(64, 128, 32);
    gemm_tile.warp_tile = GemmTile(32, 32, 32);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = num_stages > 1;
    mparams.circular_buffer_options.circular_buffer_smem_read = num_stages > 1;
    mparams.circular_buffer_options.smem_circular_buffer_stage = num_stages;
    mma_utils::MmaDataTypes data_types = {
        DataType::Half, DataType::Half, DataType::Float};
    std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
        mma_utils::generateSharedMemoryEpilogueHeuristics(
            gemm_tile,
            mparams.circular_buffer_options.smem_circular_buffer_stage,
            data_types,
            ignore_occupancy_drop);
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    // If use_smem_epilogue is true, there should be 3 shared memory tensors 2
    // for prologue and 1 for epilogue.
    int num_shared_mem_tensors = 0;
    int expected_num_shared_mem_tensors = mparams.use_smem_epilogue ? 3 : 2;
    for (const auto& tv : fusion.allTvs()) {
      if (tv->getMemoryType() == MemoryType::Shared) {
        num_shared_mem_tensors++;
      }
    }
    NVF_CHECK(
        num_shared_mem_tensors == expected_num_shared_mem_tensors,
        "Number of shared memory tensors doesn't match!",
        "Expected: ",
        expected_num_shared_mem_tensors,
        ", Got: ",
        num_shared_mem_tensors);

    at::manual_seed(0);
    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    KernelExecutor ke;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        ke.compile(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    auto cg_outputs = ke.run({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

    // check bank conflicts
    ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
    ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
        ke.compiledKernel()->kernel()));
    // (0.001, 0.001) passed on local A100 but failed on CI A100
    NVF_CHECK(
        at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.01, 0.01),
        "Result validation failed. Max diff: ",
        (cg_outputs[0].as<at::Tensor>() - tref).abs().max());

    if (!mparams.use_smem_epilogue) {
      GTEST_SKIP() << "Test conducted without utilizing shared memory epilogue "
                      "due to the device's constrained shared memory capacity.";
    }

    // Check that smem is allocated as expected.
    // There are three cases that are determined by the current device in
    // mma_utils::generateSharedMemoryEpilogueHeuristics:
    //   - !use_smem_epilogue : A + B (this test is skipped in this case)
    //   - use_smem_epilogue && !promote_prologue_smem_reuse : A + B + C
    //   - use_smem_epilogue && promote_prologue_smem_reuse : max(A + B, C)
    auto smem_allocs =
        ke.compiledKernel()->kernel()->summary().dynamic_smem_allocations;
    NVF_CHECK(smem_allocs.size() == 3);
    if (mparams.promote_prologue_smem_reuse) {
      // Check prologue shared memory re-use
      // smem_allocs = {A, B, C} where C is the epilogue buffer
      // since A and B have no further uses, we should be able to reuse both
      // of them, implying that the address of C is zero. In this case, B will
      // also be allocated at address 0 with A stacked above it at position
      // 8192.
      EXPECT_EQ(
          smem_allocs.at(0)->address()->evaluate(),
          // Assuming B numel times size(dtype) is a multiple of 16 so that
          // this address is aligned
          smem_allocs.at(1)->size()->evaluate() *
              dataTypeSizeByte(smem_allocs.at(1)->buffer()->dtype()));
      EXPECT_EQ(smem_allocs.at(1)->address()->evaluate(), 0L);
      EXPECT_EQ(smem_allocs.at(2)->address()->evaluate(), 0L);
    } else {
      // Prologue shared memory is not re-used. In this case, memory should
      // stack in C, B, A order.
      EXPECT_EQ(
          smem_allocs.at(0)->address()->evaluate(),
          // Assuming for B and C that numel times size(dtype) is a multiple
          // of 16 so that this address is aligned
          smem_allocs.at(1)->size()->evaluate() *
                  dataTypeSizeByte(smem_allocs.at(1)->buffer()->dtype()) +
              smem_allocs.at(2)->size()->evaluate() *
                  dataTypeSizeByte(smem_allocs.at(2)->buffer()->dtype()));
      EXPECT_EQ(
          smem_allocs.at(1)->address()->evaluate(),
          smem_allocs.at(2)->size()->evaluate() *
              dataTypeSizeByte(smem_allocs.at(2)->buffer()->dtype()));
      EXPECT_EQ(smem_allocs.at(2)->address()->evaluate(), 0L);
    }
  }
}

// On A100, this problem is able to make use of smem epilogue but only if we
// promote use.
// See https://github.com/NVIDIA/Fuser/pull/1834
TEST_F(MatmulTest, AmpereMatmulSmemEpiloguePromotionRequiredA100) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 4096, N = 4096, K = 4096;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto layout = MmaLayout::TN;

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  // The settings of cta_tile, warp_tile, and smem_circular_buffer_stage have
  // been purposefully selected to produce a constant occupancy of 25%. This
  // allows us to effectively evaluate the influence of the use_smem_epilogue
  // parameter on performance, since changing its value to either true or
  // false will not affect the occupancy rate.
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(64, 96, 64);
  gemm_tile.warp_tile = GemmTile(16, 32, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 6;
  mma_utils::MmaDataTypes data_types = {
      DataType::Half, DataType::Half, DataType::Float};
  std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          gemm_tile,
          mparams.circular_buffer_options.smem_circular_buffer_stage,
          data_types,
          /*ignore_occupancy_drop=*/false);

  if (deviceMajorMinorCheck(8, 0)) {
    // Test that we promote smem reuse on A100. This might differ on devices
    // with different amounts of smem.
    ASSERT_TRUE(mparams.promote_prologue_smem_reuse);
  }

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // KernelExecutor::compile would fail otherwise.
  SKIP_IF_INSUFFICIENT_SMEM(&mparams, data_types);

  at::manual_seed(0);
  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0].as<at::Tensor>() - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP() << "Test conducted without utilizing shared memory epilogue "
                    "due to the device's constrained shared memory capacity.";
  }
  if (!mparams.promote_prologue_smem_reuse) {
    GTEST_SKIP() << "Test conducted with shared memory epilogue but without "
                    "promoting prologue smem re-use.";
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulSmemEpilogueCast) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  constexpr bool ignore_occupancy_drop = true;
  // Keep multiples of 8 to keep vectorizable.
  int M = 4096, N = 4096, K = 4096;
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion.addOutput(tv3);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mma_utils::MmaDataTypes data_types = {
      DataType::Half, DataType::Half, DataType::Float};
  std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          gemm_tile,
          mparams.circular_buffer_options.smem_circular_buffer_stage,
          data_types,
          ignore_occupancy_drop);
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // If use_smem_epilogue is true, there should be 3 shared memory tensors 2
  // for prologue and 1 for epilogue.
  int num_shared_mem_tensors = 0;
  int expected_num_shared_mem_tensors = mparams.use_smem_epilogue ? 3 : 2;
  for (const auto& tv : fusion.allTvs()) {
    if (tv->getMemoryType() == MemoryType::Shared) {
      num_shared_mem_tensors++;
    }
  }
  NVF_CHECK(
      num_shared_mem_tensors == expected_num_shared_mem_tensors,
      "Number of shared memory tensors doesn't match!",
      "Expected: ",
      expected_num_shared_mem_tensors,
      ", Got: ",
      num_shared_mem_tensors);

  at::manual_seed(0);
  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  tref = tref.to(at::kHalf);
  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0].as<at::Tensor>() - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP() << "Test conducted without utilizing shared memory epilogue "
                    "due to the device's constrained shared memory capacity.";
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulSmemEpilogueRelu) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  constexpr bool ignore_occupancy_drop = true;
  // Keep multiples of 8 to keep vectorizable.
  int M = 4096, N = 4096, K = 4096;
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});
  auto tv3 = relu(tv2);

  fusion.addOutput(tv3);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mma_utils::MmaDataTypes data_types = {
      DataType::Half, DataType::Half, DataType::Float};
  std::tie(mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
      mma_utils::generateSharedMemoryEpilogueHeuristics(
          gemm_tile,
          mparams.circular_buffer_options.smem_circular_buffer_stage,
          data_types,
          ignore_occupancy_drop);
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // If use_smem_epilogue is true, there should be 3 shared memory tensors 2
  // for prologue and 1 for epilogue.
  int num_shared_mem_tensors = 0;
  int expected_num_shared_mem_tensors = mparams.use_smem_epilogue ? 3 : 2;
  for (const auto& tv : fusion.allTvs()) {
    if (tv->getMemoryType() == MemoryType::Shared) {
      num_shared_mem_tensors++;
    }
  }
  NVF_CHECK(
      num_shared_mem_tensors == expected_num_shared_mem_tensors,
      "Number of shared memory tensors doesn't match!",
      "Expected: ",
      expected_num_shared_mem_tensors,
      ", Got: ",
      num_shared_mem_tensors);

  at::manual_seed(0);
  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto t2 = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  auto tref = at::relu(t2).to(at::kFloat);

  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0].as<at::Tensor>() - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP() << "Test conducted without utilizing shared memory epilogue "
                    "due to the device's constrained shared memory capacity.";
  }
}

// Test the matmul scheduler's single-kernel split-K support
TEST_P(MatmulTestWithLayout, FusionAmpereMatmulSplitK_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 8096;

  for (int splitk_factor : {2}) {
    for (int use_smem_epilogue : {false, true}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

      auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
      auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

      fusion.addOutput(tv2);

      MatMulTileOptions gemm_tile;
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      if (use_smem_epilogue) {
        std::tie(
            mparams.use_smem_epilogue, mparams.promote_prologue_smem_reuse) =
            mma_utils::generateSharedMemoryEpilogueHeuristics(
                gemm_tile,
                1,
                {DataType::Half, DataType::Half, DataType::Float},
                true,
                true,
                true);
        if (!mparams.use_smem_epilogue) {
          std::cout << "Skipping smem epilogue due to shared memory "
                       "constraints on this device"
                    << std::endl;
          continue;
        }
        mparams.promote_prologue_smem_reuse = true;
      }

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto inputs = matmulAtInput3DTuring(M, N, K, layout);

      KernelExecutor ke;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, ke.compile(&fusion, {inputs.first, inputs.second}));
      EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
      auto cg_outputs = ke.run({inputs.first, inputs.second});
      ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
          ke.compiledKernel()->kernel()));
      auto tref = atMatmul(
          inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

      // Relax tolerance for larger sum due to large K
      NVF_CHECK(at::allclose(
          cg_outputs[0].as<at::Tensor>(), tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Test splitk with bias epilogue
TEST_P(MatmulTestWithLayout, FusionAmpereMatmulSplitKBias_CUDA) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 8096;

  for (int splitk_factor : {2}) {
    for (int use_smem_epilogue : {false, true}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

      auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
      auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);
      auto tv2 = makeContigTensor(1, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);
      fusion.addInput(tv2);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
      auto tv4 = broadcast(tv2, {false, true});
      auto tv5 = add(tv3, tv4); // bias

      fusion.addOutput(tv5);

      MatMulTileOptions gemm_tile;
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto [t0, t1] = matmulAtInput3DTuring(M, N, K, layout);
      at::Tensor bias = at::randn({M}, t0.options());

      KernelExecutor ke;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, ke.compile(&fusion, {t0, t1, bias}));
      EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
      auto cg_outputs = ke.run({t0, t1, bias});
      ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
          ke.compiledKernel()->kernel()));
      auto tref = atBiasEpilogue(
          atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout), bias);

      // Relax tolerance for larger sum due to large K
      NVF_CHECK(at::allclose(
          cg_outputs[0].as<at::Tensor>(), tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Same as above but has a batch dimension and splitk
TEST_P(MatmulTestWithLayout, AmpereMatmulBatchSplitK) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int B = 2, M = 504, N = 136, K = 2048;

  for (int splitk_factor : {2}) {
    for (int use_smem_epilogue : {false, true}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto tv0 = makeContigTensor(3, DataType::Half);
      auto tv1 = makeContigTensor(3, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

      fusion.addOutput(tv2);

      MatMulTileOptions gemm_tile;
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto t0 =
          matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
      auto t1 =
          matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);

      KernelExecutor ke;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(7, 5, ke.compile(&fusion, {t0, t1}));
      ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
      ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
          ke.compiledKernel()->kernel()));
      auto cg_outputs = ke.run({t0, t1});
      auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

      // Relax tolerance for larger sum due to large K
      NVF_CHECK(at::allclose(
          cg_outputs[0].as<at::Tensor>(), tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Test batch splitk with bias epilogue
TEST_P(MatmulTestWithLayout, AmpereMatmulBatchSplitKBias) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int B = 2, M = 504, N = 136, K = 2048;

  for (int splitk_factor : {2}) {
    for (int use_smem_epilogue : {false, true}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto tv0 = makeContigTensor(3, DataType::Half);
      auto tv1 = makeContigTensor(3, DataType::Half);
      auto tv2 = makeContigTensor(1, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);
      fusion.addInput(tv2);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      auto tv3 = fusedMultiplySum(tv0, tv1, {-1});
      auto tv4 = broadcast(tv2, {true, false, true});
      auto tv5 = add(tv3, tv4);

      fusion.addOutput(tv5);

      MatMulTileOptions gemm_tile;
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto t0 =
          matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
      auto t1 =
          matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
      at::Tensor bias = at::randn({M}, t0.options());

      KernelExecutor ke;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, ke.compile(&fusion, {t0, t1, bias}));
      ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
      ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
          ke.compiledKernel()->kernel()));
      auto cg_outputs = ke.run({t0, t1, bias});
      auto tref = atBiasEpilogue(
          atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout), bias);

      // Relax tolerance for larger sum due to large K
      NVF_CHECK(at::allclose(
          cg_outputs[0].as<at::Tensor>(), tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Avoid lowering error https://github.com/NVIDIA/Fuser/issues/1808
TEST_F(MatmulTest, ReproIssue1808) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  auto layout = MmaLayout::TN;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto shapes = matmulAtInputShape3DTuring(-1, -1, -1, layout);

  auto tv0 = makeContigConcreteTensor(shapes.first, DataType::Half);
  auto tv1 = makeContigConcreteTensor(shapes.second, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
  tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(160, 144, 16);
  gemm_tile.warp_tile = GemmTile(80, 24, 16);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      ke.compile(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Test matmul with sizes that are not divisible by 8 and with misaligned inputs
TEST_P(MatmulTestWithLayout, MisalignedVectorization) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  for (bool add_2d_bias : {false, true}) {
    for (bool downcast_output : {false, true}) {
      for (const auto& [M, N, K, alignA, alignB, alignBias] : std::vector<
               std::
                   tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>{
               {504, 136, 248, 8, 8, 8}, // all fully vectorizable in all
                                         // layouts
               {504, 136, 249, 8, 8, 8}, // odd K, operands not vectorizable
                                         // in TN. output fully vectorizable
               {504, 137, 248, 8, 8, 8}, // A fully vectorizable, B fully
                                         // vectorizable unless transposed,
                                         // output not vectorizable
               {505, 136, 248, 8, 8, 8}, // B fully vectorizable, A
                                         // vectorizable unless transposed,
                                         // output fully vectorizable
               {505, 137, 248, 8, 8, 8}, // none vectorizable

               // Cases with vectorizable strides but misaligned base pointers
               {504, 136, 248, 2, 8, 8}, // A not vectorizable due to offset
               {504, 136, 248, 8, 2, 8}, // B not vectorizable due to offset
               {504, 136, 248, 8, 8, 2}, // epilogue not vectorizable due to
               // offset
           }) {
        const auto maybeUnalign = [](const at::Tensor& t, int64_t offset) {
          if (offset == 16 / t.element_size()) {
            // Already fully aligned
            return t;
          }
          return at::pad(t.ravel(), {{0, offset}})
              .index({at::indexing::Slice(offset, t.numel() + offset, 1)})
              .view({t.size(0), t.size(1)});
        };

        auto t0 = maybeUnalign(
            matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K),
            alignA);
        auto t1 = maybeUnalign(
            matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K),
            alignB);

        auto tref = atMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);

        KernelArgumentHolder inputs = {t0, t1};

        if (add_2d_bias) {
          const auto options =
              at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
          auto bias = maybeUnalign(at::randn({M, N}, options), alignBias);
          tref = tref + bias;
          inputs.push(bias);
        }

        if (downcast_output) {
          tref = tref.to(at::kHalf);
        }

        auto fusion = std::make_unique<Fusion>();
        FusionGuard fg(fusion.get());

        auto tv0 = makeContigTensor(2, DataType::Half);
        auto tv1 = makeContigTensor(2, DataType::Half);
        fusion->addInput(tv0);
        fusion->addInput(tv1);

        tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
        tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
        auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

        if (add_2d_bias) {
          auto bias = makeContigTensor(2, DataType::Half);
          fusion->addInput(bias);
          tv2 = add(tv2, bias);
        }

        if (downcast_output) {
          tv2 = castOp(DataType::Half, tv2);
        }

        fusion->addOutput(tv2);

        const MmaLayout fusion_layout = getMatmulProblemLayout(fusion.get());
        NVF_CHECK(
            fusion_layout == layout,
            "mismatch between test layout (",
            toString(layout),
            ") and layout inferred from fusion definition (",
            toString(fusion_layout),
            ")");

        // determine supported vectorization of an ATen tensor that will be
        // loaded along its innermost dimension
        const auto atenSupportedVectorization =
            [](const at::Tensor& tens) -> int64_t {
          auto data_ptr_int = static_cast<int64_t>(
              reinterpret_cast<std::uintptr_t>(tens.data_ptr()));
          int64_t vec_size =
              scheduler_utils::maxVectorizationWidth(data_ptr_int) /
              tens.element_size();
          std::vector<int64_t> strides = tens.strides().vec();
          std::sort(strides.begin(), strides.end());
          if (strides.front() > 1) {
            // Discontiguous input
            return 1;
          }
          strides.erase(strides.begin());
          NVF_ERROR(!strides.empty());
          // Next smallest stride determines supported vectorization
          vec_size = std::min(
              vec_size,
              scheduler_utils::maxVectorizationWidth(strides.front()));
          return std::min(vec_size, (int64_t)(16 / tens.element_size()));
        };

        MatmulParams mparams;
        mparams.mma_macro = MmaMacro::Ampere_16_8_16;
        mparams.supported_vec_size = {
            .a = atenSupportedVectorization(t0),
            .b = atenSupportedVectorization(t1),
            .epilogue = std::min(
                alignBias,
                std::min(
                    scheduler_utils::maxVectorizationWidth(N),
                    (int64_t)(16 / (downcast_output ? 2 : 4))))};
        // Supported vectorization determines whether we are able to do async
        // gmem->smem loads.
        mparams.async_gmem_load_operands = mparams.supported_vec_size.a >= 4 &&
            mparams.supported_vec_size.b >= 4;
        mparams.circular_buffer_options.circular_buffer_smem_write = true;
        // If we cannot use cp.async, it means we cannot do circular buffering
        mparams.circular_buffer_options.smem_circular_buffer_stage =
            mparams.async_gmem_load_operands ? 4 : 2;

        SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
            ->schedule(fusion.get(), &mparams);

        KernelExecutor ke;
        NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
            8,
            0,
            ke.compile(fusion.get(), inputs, LaunchParams(), matmul_cparams));
        ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
        ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
            ke.compiledKernel()->kernel()));
        auto outputs = ke.run(inputs);

        EXPECT_TRUE(
            at::allclose(outputs[0].as<at::Tensor>(), tref, 0.001, 0.001));
      }
    }
  }
}

// Matmul test with multiple M and N dimensions that are consecutive
TEST_F(MatmulTest, MultipleConsecutiveDims) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);

  int M1 = 126, M2 = 4, N1 = 68, N2 = 2, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(3, DataType::Half); // M1, M2, K
  auto tv1 = makeContigTensor(3, DataType::Half); // N1, N2, K

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // M1, M2, N1, N2, K
  tv0 = broadcast(tv0, {false, false, true, true, false});
  tv1 = broadcast(tv1, {true, true, false, false, false});

  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  // Supported vec size is based on inner dim
  mparams.supported_vec_size = {4, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M1, M2, K}, options);
  auto t1 = at::randn({N1, N2, K}, options);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, ke.compile(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({t0, t1});
  auto tref = at::reshape(
      at::linear(
          at::reshape(t0.to(at::kFloat), {M1 * M2, K}),
          at::reshape(t1.to(at::kFloat), {N1 * N2, K})),
      {M1, M2, N1, N2});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test with multiple M dimensions that are non-consecutive
// TODO: This test currently fails, but it can be run using
//   build/test_matmul --gtest_also_run_disabled_tests
//
// This case fails because the allocation domain of the A smem cached tensor is
// [M1, K, M2], which is incompatible with ldMatrix. We need to gather the
// allocation domains for the smem tensors by role instead, so that this
// becomes [K, M1, M2].
TEST_F(MatmulTest, DISABLED_MultipleNonConsecutiveMDims) {
  int M1 = 126, N = 136, M2 = 4, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Note that M2 is inside K, so this is an NN layout
  auto tv0 = makeContigTensor(3, DataType::Half); // M1, K, M2
  auto tv1 = makeContigTensor(2, DataType::Half); // N, K

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // M1, N, K, M2
  tv0 = broadcast(tv0, {false, true, false, false});
  tv1 = broadcast(tv1, {true, false, false, true});

  auto tv2 = fusedMultiplySum(tv0, tv1, {-2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  // Supported vec size is based on inner dim
  mparams.supported_vec_size = {4, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M1, K, M2}, options);
  auto t1 = at::randn({N, K}, options);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, ke.compile(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({t0, t1});
  auto Apermuted = t0.permute({{1, 2}}).reshape({M1 * M2, K});
  auto tref = at::linear(Apermuted.to(at::kFloat), t1.to(at::kFloat))
                  .reshape({M1, M2, N})
                  .permute({{1, 2}});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// Matmul test with multiple N dimensions that are non-consecutive
// TODO: This test currently fails, but it can be run using
//   build/test_matmul --gtest_also_run_disabled_tests
//
// This case fails because the allocation domain of the B smem cached tensor is
// [N1, K, N2], which is incompatible with ldMatrix. We need to gather the
// allocation domains for the smem tensors by role instead, so that this
// becomes [K, N1, N2].
TEST_F(MatmulTest, DISABLED_MultipleNonConsecutiveNDims) {
  int M = 504, N1 = 68, N2 = 2, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Note that N2 is inside K, so this is a TT layout
  auto tv0 = makeContigTensor(2, DataType::Half); // M, K
  auto tv1 = makeContigTensor(3, DataType::Half); // N1, K, N2

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // M, N1, K, N2
  tv0 = broadcast(tv0, {false, true, false, true});
  tv1 = broadcast(tv1, {true, false, false, false});

  auto tv2 = fusedMultiplySum(tv0, tv1, {-2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  // Output is M, N1, N2, contiguous so fully vectorizable despite size N2=2 in
  // inner dim.
  mparams.supported_vec_size = {8, 2, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N1, K, N2}, options);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, ke.compile(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({t0, t1});
  auto Bpermuted = t1.permute({{1, 2}}).reshape({N1 * N2, K});
  auto tref = at::linear(t0.to(at::kFloat), Bpermuted.to(at::kFloat))
                  .reshape({M, N1, N2});
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

// This is a tougher test where we insert a batch dim between the M dims
// The batch dim is parallelized, so M1 and M2 are consecutive in shared
// memory.
TEST_F(MatmulTest, MultipleMDimsBatch) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);

  int Batch = 2, M1 = 126, N = 136, M2 = 4, K = 248;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Note that M2 is inside K, so this is an NN layout
  auto tv0 = makeContigTensor(4, DataType::Half); // M1, Batch, M2, K
  auto tv1 = makeContigTensor(3, DataType::Half); // Batch, N, K

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  tv0 = broadcast(tv0, {false, false, false, true, false});
  tv1 = broadcast(tv1, {true, false, true, false, false});

  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);

  MatmulParams mparams;
  // Supported vec size is based on inner dim
  mparams.supported_vec_size = {4, 8, 4};
  mparams.mma_macro = MmaMacro::Ampere_16_8_16;
  mparams.tile_sizes = gemm_tile;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = true;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M1, Batch, M2, K}, options);
  auto t1 = at::randn({Batch, N, K}, options);

  KernelExecutor ke;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, ke.compile(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));
  auto cg_outputs = ke.run({t0, t1});
  auto tref =
      at::matmul(t0.to(at::kFloat), at::permute(t1.to(at::kFloat), {0, 2, 1}));
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulTestWithLayout,
    testing::ValuesIn(kAllSupportedMmaLayout),
    mmaLayoutName);

class HopperMatmulTest : public HopperBase {
 protected:
  void SetUp() override {
    HopperBase::SetUp();
  }
};

using BlackwellMatmulTest = BlackwellBase;

// 2 math group, non-persistent, non-warp specialized, no CGA
// TODO: This could be in HopperMatmulTest::SetUp() instead
MatmulParams defaultHopperParams() {
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);
  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::Pipelined;
  mparams.tiling_strategy = MatmulParams::TilingStrategy::OneTilePerCTA;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {1, 1};
  mparams.promote_prologue_smem_reuse = true;
  return mparams;
}

TEST_F(HopperMatmulTest, HSH_NT_128BSwizzle_Basic) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  constexpr auto macro = MmaMacro::Hopper_64_256_16;
  constexpr auto layout = MmaLayout::NT; // [K, M] x [K, N] -> [M, N]
  constexpr auto swizzle = MmaInputSmemSwizzle::B128;
  const auto dtype = DataType::Half;

  constexpr int64_t stages = 4;
  constexpr int64_t prefetch = 3;
  const int64_t cta_m = 2 * getM(macro);
  const int64_t cta_n = 1 * getN(macro);

  constexpr std::tuple<int64_t, int64_t, int64_t> cluster_dims{2, 1, 1};

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
  auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  if constexpr (
      cluster_dims != std::tuple<int64_t, int64_t, int64_t>{1, 1, 1}) {
    fusion.manage("cluster_dims", cluster_dims);
  }

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  // gmem [K, M, 1] x gmem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> gmem [M, N]

  auto tv0c = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0c->setMemoryType(MemoryType::Shared);
  auto tv1c = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv1c->setMemoryType(MemoryType::Shared);

  TensorView* tv3c = nullptr;
  tv3c = tv3->cacheBefore();
  tv3c->setMemoryType(MemoryType::Local);

  // gmem [K, M, 1] -TMA-> smem [K, M, 1]
  // gmem [K, 1, N] -TMA-> smem [K, 1, N]
  // smem [K, M, 1] x smem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> register [M, N] -set-> gmem [M, N]

  // Create tiles
  tv2->split(-3, cta_m);
  tv2->split(-2, cta_n);
  tv2->split(-1, getK(macro));
  // [Mo, Mi, No, Ni, Ko, Ki] -> [Mo, No, Ko, Mi, Ni, Ki]
  tv2->reorder({{-5, -3}, {-3, -2}});
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::BIDx);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv2);

  // [..., Mi, Ni, Ki] -> [..., Ni, Ki, Mi]
  tv0c->reorder({{-3, -1}});
  tv0c->applyMmaSwizzleForTMALoad(swizzle);
  // [..., Mi, Ni, Ki] -> [..., Mi, Ki, Ni]
  tv1c->reorder({{-1, -2}});
  tv1c->applyMmaSwizzleForTMALoad(swizzle);

  {
    tv2->split(-3, getM(macro));
    tv2->split(-2, getN(macro));
    // [Mo, No, Ko, Mio, Mii, Nio, Nii, Ki]
    // -> [Mo, No, Ko, Mio, Nio, Mii, Nii, Ki]
    tv2->reorder({{-4, -3}});
    tv2->merge(-5);
    tv2->axis(-4)->parallelize(ParallelType::TIDy);
    scheduler_utils::BoundedDirectionalTransformPropagator::forward(
        tv2,
        -1,
        {tv3},
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType()
            .propagateToBoundary());
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setAllocationDomain(s.as<IterDomain*>(), true);
    tv2->axis(-1)->parallelize(ParallelType::Mma);
    tv2->axis(-2)->parallelize(ParallelType::Mma);
    tv2->axis(-3)->parallelize(ParallelType::Mma);
  }

  for (auto tv : {tv3c, tv3}) {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv->getLoopDomain());
    tv->setLoopDomain(s.as<IterDomain*>());
  }
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  tv0c->circularBuffer(stages, prefetch, WarpSpecialized(ParallelType::TIDy));
  tv1c->circularBuffer(stages, prefetch, WarpSpecialized(ParallelType::TIDy));

  auto inputs = matmulAtInput3DSS(M, N, K, layout, data_type_to_aten(dtype));

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(inputs.first.squeeze(), inputs.second.squeeze(), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));
}

TEST_F(HopperMatmulTest, HSH_NT_UseScheduler) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // K, M
  auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({K, M, 1}, options);
  auto t1 = at::randn({K, 1, N}, options);
  auto out_ref = at::matmul(t0.squeeze().t(), t1.squeeze()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_F(HopperMatmulTest, HSH_TN_UseScheduler) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {-1});

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto out_ref = at::matmul(t0.squeeze(), t1.squeeze().t()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_F(HopperMatmulTest, HSH_NN_UseScheduler) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({1, -1, -1}, dtype); // K, M
  auto tv1 = makeContigConcreteTensor({-1, -1, 1}, dtype); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {1});

  // Reorder the accumulator as [M, N, K]
  // [M, K, N] -> [M, N, K]
  tv2->reorder({{-1, -3}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({1, K, M}, options);
  auto t1 = at::randn({N, K, 1}, options);
  auto out_ref = at::matmul(t0.squeeze().t(), t1.squeeze().t()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_F(HopperMatmulTest, HSH_TT_UseScheduler) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {1});

  // Reorder the accumulator as [M, N, K]
  // [M, K, N] -> [M, N, K]
  tv2->reorder({{-2, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, K, 1}, options);
  auto t1 = at::randn({1, K, N}, options);
  auto out_ref = at::matmul(t0.squeeze(), t1.squeeze()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

struct MLPBenchmarkTestParams {
  bool warp_specialization;
  bool persistent_kernel;
};

class MLPBenchmarkTest
    : public NVFuserTest,
      public ::testing::WithParamInterface<MLPBenchmarkTestParams> {
 protected:
  MLPBenchmarkTestParams test_params;
  MatmulParams mparams;

  void SetUp() override {
    NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);
    test_params = GetParam();
    NVFuserTest::SetUp();
    if (test_params.warp_specialization || test_params.persistent_kernel) {
      // persistent kernels and warp specialization requires Hopper+
      NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);
    }

    mparams.supported_vec_size = {8, 8, 8};
    MatMulTileOptions gemm_tile;
    if (cudaArchGuardShouldSkip(9, 0)) {
      // Ampere
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      gemm_tile.cta_tile = GemmTile(128, 128, 32);
      gemm_tile.warp_tile = GemmTile(64, 64, 32);
    } else {
      mparams.mma_macro = MmaMacro::Hopper_64_256_16;
      gemm_tile.cta_tile = GemmTile(128, 256, 64);
      gemm_tile.warp_tile = GemmTile(64, 256, 64);
    }
    mparams.tile_sizes = gemm_tile;
    mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffering_strategy = test_params.warp_specialization
        ? MatmulParams::CircularBufferingStrategy::WarpSpecialized
        : MatmulParams::CircularBufferingStrategy::Pipelined;
    mparams.tiling_strategy = test_params.persistent_kernel
        ? MatmulParams::TilingStrategy::DistributeTilesAcrossSMs
        : MatmulParams::TilingStrategy::OneTilePerCTA;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = false;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 2;
    mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
    mparams.splitk_factor = 1;
    mparams.grid_traversal_factor = {8, 1};
    mparams.use_smem_epilogue = true;
    mparams.cluster_dims = {2, 1};
    mparams.promote_prologue_smem_reuse = true;
  }
};

TEST_P(MLPBenchmarkTest, FwdGEMM) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = linear(tv0, tv1);

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);
  auto out_ref = at::linear(t0, t1);

  std::vector<c10::IValue> inputs = {t0, t1};

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_P(MLPBenchmarkTest, FwdGEMM_BroadcastInputs) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {2});
  auto tv3 = castOp(dtype, tv2);

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto out_ref = at::linear(t0.squeeze(), t1.squeeze());

  KernelArgumentHolder inputs = {t0, t1};

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_P(MLPBenchmarkTest, FwdEpilogueBiasFusion) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype); // N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = linear(tv0, tv1, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);
  auto t2 = at::randn({M, N}, options);
  auto tv3_ref = at::linear(t0, t1, t2);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelArgumentHolder inputs = {t0, t1, t2};

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tv3_ref, 5e-2, 5e-2));
}

TEST_P(MLPBenchmarkTest, FwdEpilogueSiluFusion) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype); // N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = linear(tv0, tv1);
  fusion.addOutput(tv3);

  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);
  auto tv10 = mul(tv9, tv2);
  auto tv11 = castOp(DataType::BFloat16, tv10);
  fusion.addOutput(tv11);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);
  auto t2 = at::randn({M, N}, options);

  auto tv3_ref = at::linear(t0, t1);
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv11_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * t2).to(at::kBFloat16);

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelArgumentHolder inputs = {t0, t1, t2};

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(at::allclose(
      cg_outputs[0].as<at::Tensor>(), tv3_ref, 1e-6 * K, 1e-6 * K));
  EXPECT_TRUE(
      at::allclose(cg_outputs[1].as<at::Tensor>(), tv11_ref, 1e-2, 1e-2));
}

TEST_P(MLPBenchmarkTest, FwdEpilogueFusion_BroadcastInputs) {
  GTEST_SKIP() << "THIS TEST IS CURRENTLY FAILING" << std::endl;

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv4 = fusedMultiplySum(tv0, tv1, {2});
  auto tv3 = castOp(dtype, tv4);
  fusion.addOutput(tv3);

  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);
  auto tv10 = mul(tv9, tv2);
  auto tv11 = castOp(dtype, tv10);
  fusion.addOutput(tv11);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({M, N}, options);

  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze());
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv11_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * t2).to(at::kBFloat16);

  std::vector<c10::IValue> inputs = {t0, t1, t2};

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1, t2});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), tv3_ref, 1e-6 * K, 1e-6 * K));
  NVF_CHECK(at::allclose(cg_outputs[1].as<at::Tensor>(), tv11_ref, 1e-2, 1e-2));
}

TEST_P(MLPBenchmarkTest, FwdHorizontalFusion) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);

  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype); // N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = linear(tv0, tv1);
  fusion.addOutput(tv3);

  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);

  auto tv10 = linear(tv0, tv2);
  fusion.addOutput(tv10);

  auto tv11 = mul(tv9, tv10);
  auto tv12 = castOp(DataType::BFloat16, tv11);
  fusion.addOutput(tv12);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);
  auto t2 = at::randn({N, K}, options);

  auto tv3_ref = at::linear(t0, t1);
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv10_ref = at::linear(t0, t2);
  auto tv12_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * tv10_ref.to(at::kFloat))
          .to(at::kBFloat16);

  KernelArgumentHolder inputs = {t0, t1, t2};

  // Adjust parameters in order to fit smem and register constraints
  mparams.tile_sizes.cta_tile = GemmTile(128, 128, 64);
  mparams.tile_sizes.warp_tile = GemmTile(64, 128, 64);
  mparams.mma_macro = MmaMacro::Hopper_64_128_16;
  mparams.promote_prologue_smem_reuse = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 2;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  // TODO: Some of these are failing, perhaps due to improper syncing of
  // horizontally fused kernels?
  EXPECT_TRUE(at::allclose(
      cg_outputs[0].as<at::Tensor>(), tv3_ref, 1e-6 * K, 1e-6 * K));
  EXPECT_TRUE(at::allclose(
      cg_outputs[1].as<at::Tensor>(), tv10_ref, 1e-6 * K, 1e-6 * K));
  EXPECT_TRUE(
      at::allclose(cg_outputs[2].as<at::Tensor>(), tv12_ref, 5e-2, 1e-1));
}

TEST_P(MLPBenchmarkTest, FwdHorizontalFusion_BroadcastInputs) {
  // TODO: This test currently fails on Ampere
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);

  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv4 = fusedMultiplySum(tv0, tv1, {2});
  auto tv3 = castOp(dtype, tv4);
  fusion.addOutput(tv3);

  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);

  auto tv10 = fusedMultiplySum(tv0, tv2, {2});
  auto tv10c = castOp(dtype, tv10);
  fusion.addOutput(tv10c);

  auto tv11 = mul(tv9, tv10);
  auto tv12 = castOp(DataType::BFloat16, tv11);
  fusion.addOutput(tv12);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({1, N, K}, options);

  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze());
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv10_ref = at::linear(t0.squeeze(), t2.squeeze());
  auto tv12_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * tv10_ref.to(at::kFloat))
          .to(at::kBFloat16);

  std::vector<c10::IValue> inputs{t0, t1, t2};

  // Adjust parameters in order to fit smem and register constraints
  mparams.tile_sizes.cta_tile = GemmTile(128, 128, 64);
  mparams.tile_sizes.warp_tile = GemmTile(64, 128, 64);
  mparams.mma_macro = MmaMacro::Hopper_64_128_16;
  mparams.promote_prologue_smem_reuse = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 2;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({t0, t1, t2});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), tv3_ref, 1e-6 * K, 1e-6 * K));
  NVF_CHECK(at::allclose(
      cg_outputs[1].as<at::Tensor>(), tv10_ref, 1e-6 * K, 1e-6 * K));
  NVF_CHECK(at::allclose(cg_outputs[2].as<at::Tensor>(), tv12_ref, 5e-2, 1e-1));
}

TEST_P(MLPBenchmarkTest, BatchGEMM) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t Batch = 3, M = 4096, N = 14336, K = 5120;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, -1, -1}, dtype); // Batch, M, K
  auto tv1 = makeContigConcreteTensor({-1, -1, -1}, dtype); // Batch, K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = matmul(tv0, tv1);

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({Batch, M, K}, options);
  auto t1 = at::randn({Batch, K, N}, options);
  auto out_ref = at::matmul(t0, t1);

  std::vector<c10::IValue> inputs = {t0, t1};

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  // Disable NVRTC caching so that we can reliably observe ptxas output
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().set(DisableOption::NvrtcCaching);
  CompileParams compile_opts;
  compile_opts.enable_ptxas_verbose = true;
  captureStdout();

  KernelExecutor ke;
  ke.compile(&fusion, inputs, LaunchParams(), compile_opts);

  std::string output = getCapturedStdout();
  EXPECT_EQ(output.find("warpgroup"), std::string::npos)
      << "Detected ptxas warpgroup warning";

  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MLPBenchmarkTest,
    ::testing::Values(
        MLPBenchmarkTestParams{
            .warp_specialization = false,
            .persistent_kernel = false},
        MLPBenchmarkTestParams{
            .warp_specialization = true,
            .persistent_kernel = false},
        MLPBenchmarkTestParams{
            .warp_specialization = false,
            .persistent_kernel = true},
        MLPBenchmarkTestParams{
            .warp_specialization = true,
            .persistent_kernel = true}),
    [](const testing::TestParamInfo<MLPBenchmarkTestParams>& info) {
      std::stringstream ss;
      ss << (info.param.persistent_kernel ? "persistent" : "dataparallel");
      ss << (info.param.warp_specialization ? "_warpspec" : "_non_warpspec");
      return ss.str();
    });

// This tests that we can use a small instruction tile with a medium size
// warpgroup tile and a large CTA tile.
// TODO: This test passes on CUDA 13.0 but fails on CUDA 13.1
TEST_F(
    HopperMatmulTest,
    DISABLED_HSH_NT_UseScheduler_MultipleInstructionsPerWarpTile) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // K, M
  auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({K, M, 1}, options);
  auto t1 = at::randn({K, 1, N}, options);
  auto out_ref = at::matmul(t0.squeeze().t(), t1.squeeze()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  // Regardless of the instruction, this should result in 2 warp groups i.e. 256
  // threads
  gemm_tile.cta_tile = GemmTile(256, 256, 16);
  gemm_tile.warp_tile = GemmTile(128, 128, 16);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_64_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 4;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = false;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  EXPECT_TRUE(getBankConflictInfo(kernel).empty());
  EXPECT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(kernel));

  auto cg_outputs = ke.run({t0, t1});

  // Check number of launched threads matches what we expect
  EXPECT_EQ(ke.lastLaunchParams().bdimx(), 128);
  EXPECT_EQ(ke.lastLaunchParams().bdimy(), 4)
      << " expected 4 warp groups (BIDy==4) but found BIDy=="
      << ke.lastLaunchParams().bdimy();

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_F(HopperMatmulTest, ScheduleWithTranslation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype); // K, N
  // Note tv1 has allocation domain
  // tv1->setAllocationDomain({tv1->axis(1), tv1->axis(0)}, true);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = matmul(tv0, tv1);

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({M, K}, options);
  // auto t1 = at::randn({N, K}, options).t();
  auto t1 = at::randn({K, N}, options);
  auto out_ref = at::matmul(t0, t1);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 16);
  gemm_tile.warp_tile = GemmTile(64, 64, 16);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_64_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {1, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  EXPECT_TRUE(getBankConflictInfo(kernel).empty());
  EXPECT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(kernel));

  auto cg_outputs = ke.run({t0, t1});

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

// Test that we can compile matmul kernels with both 32-bit and 64-bit indexing,
// and that if we pass arguments for which this is unsafe (meaning there is
// overflow), that the appropriate exception is raised
TEST_F(HopperMatmulTest, IndexTypeValidation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // M, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {1});

  // Reorder the accumulator as [M, N, K]
  // [M, K, N] -> [M, N, K]
  tv2->reorder({{-2, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = false;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 1;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {1, 1};
  mparams.promote_prologue_smem_reuse = true;

  constexpr int64_t M = 1 << 17, N = 256, K = 1 << 17;
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);

  { // This scope is to help us reclaim memory later
    auto a_ref = at::randn({M, K, 1}, options);
    auto b_ref = at::randn({1, K, N}, options);
    auto out_ref = at::matmul(a_ref.squeeze(), b_ref.squeeze()).to(at::kHalf);
    const std::vector<c10::IValue> inputs = {a_ref, b_ref};

    mparams.cparams.index_type = DataType::Int32;

    at::Tensor int32_output;
    {
      Fusion fusion_clone;
      Fusion::copy(&fusion, &fusion_clone);
      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion_clone, &mparams);

      KernelExecutor ke;
      ke.compile(&fusion_clone, inputs);
      int32_output = ke.run(inputs)[0].as<at::Tensor>();
    }

    mparams.cparams.index_type = DataType::Int;

    Fusion fusion_clone;
    Fusion::copy(&fusion, &fusion_clone);
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion_clone, &mparams);

    KernelExecutor ke;
    ke.compile(&fusion_clone, inputs);
    auto int64_output = ke.run(inputs)[0].as<at::Tensor>();
    EXPECT_TRUE(int64_output.equal(int32_output));
  }

  // Test that passing inputs that are too large in one dimension lead to error
  maybeClearAllocator(/*max_bytes=*/0);
  {
    mparams.cparams.index_type = DataType::Int;

    Fusion fusion_clone;
    Fusion::copy(&fusion, &fusion_clone);
    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion_clone, &mparams);

    constexpr int64_t M_big = 1L << 32, N_big = 2, K_big = 2;
    auto a_big = at::randn({M_big, K_big, 1}, options);
    auto b_big = at::randn({1, K_big, N_big}, options);
    const std::vector<c10::IValue> inputs_big{a_big, b_big};

    KernelExecutor ke;
    ke.compile(&fusion_clone, inputs_big);
    EXPECT_THAT(
        [&]() { ke.run(inputs_big); },
        ::testing::ThrowsMessage<nvfuser::nvfError>(
            ::testing::HasSubstr("Found unsafe casts from DataType::Index")));
  }
}

TEST_F(HopperMatmulTest, HSH_NT_128BSwizzle_BroadcastOp) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  constexpr auto macro = MmaMacro::Hopper_64_256_16;
  constexpr auto layout = MmaLayout::NT; // [K, M] x [K, N] -> [M, N]
  constexpr auto swizzle = MmaInputSmemSwizzle::B128;
  const auto dtype = DataType::Half;

  constexpr bool use_smem_epilogue = false;
  constexpr bool use_warp_specialization = false;

  constexpr int64_t stages = 4;
  constexpr int64_t prefetch = 3;
  const int64_t cta_m = 2 * getM(macro);
  const int64_t cta_n = 1 * getN(macro);

  constexpr std::tuple<int64_t, int64_t, int64_t> cluster_dims{2, 1, 1};

  // auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype);
  // auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype);
  auto tv0 = makeContigConcreteTensor({-1, -1}, dtype);
  auto tv1 = makeContigConcreteTensor({-1, -1}, dtype);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  if constexpr (
      cluster_dims != std::tuple<int64_t, int64_t, int64_t>{1, 1, 1}) {
    fusion.manage("cluster_dims", cluster_dims);
  }

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, "
      "got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  // gmem [K, M, 1] x gmem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> gmem [M, N]

  auto tv0c = tv0b->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0c->setMemoryType(MemoryType::Shared);
  auto tv1c = tv1b->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv1c->setMemoryType(MemoryType::Shared);

  tv0b->setMemoryType(MemoryType::Global);
  tv1b->setMemoryType(MemoryType::Global);

  TensorView *tv3c = nullptr, *tv3_shmem = nullptr;
  if (use_smem_epilogue) {
    tv3_shmem = tv3->cacheBefore();
    tv3c = tv3_shmem->cacheBefore();
    tv3_shmem->setMemoryType(MemoryType::Shared);
    tv3c->setMemoryType(MemoryType::Local);
    tv3_shmem->definition()->as<LoadStoreOp>()->setOpType(
        LoadStoreOpType::StMatrix);
    tv3->definition()->as<LoadStoreOp>()->setOpType(
        LoadStoreOpType::CpAsyncBulkTensorTile);
  } else {
    tv3c = tv3->cacheBefore();
    tv3c->setMemoryType(MemoryType::Local);
  }

  // gmem [K, M, 1] -TMA-> smem [K, M, 1]
  // gmem [K, 1, N] -TMA-> smem [K, 1, N]
  // smem [K, M, 1] x smem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> register [M, N] -set-> gmem [M, N]

  // Create tiles
  tv2->split(-3, cta_m);
  tv2->split(-2, cta_n);
  tv2->split(-1, getK(macro));
  // [Mo, Mi, No, Ni, Ko, Ki] -> [Mo, No, Ko, Mi, Ni, Ki]
  tv2->reorder({{-5, -3}, {-3, -2}});
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(1)->parallelize(ParallelType::BIDx);

  TransformPropagator propagator(tv2);
  MaxLogicalDomainInfoSpanningTree(tv2).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv2);

  // [..., Mi, Ki] -> [..., Ki, Mi]
  tv0c->reorder({{-3, -1}});
  tv0c->applyMmaSwizzleForTMALoad(swizzle);
  tv0c->axis(-6)->parallelize(ParallelType::Unroll);
  // [..., Ni, Ki] -> [..., Ki, Ni]
  tv1c->reorder({{-1, -2}});
  tv1c->applyMmaSwizzleForTMALoad(swizzle);
  tv1c->axis(-6)->parallelize(ParallelType::Unroll);

  // Strip ParallelType::Bulk from the broadcast tensors, since its definition
  // is not a TMA
  for (TensorView* tv : {tv0b, tv1b}) {
    for (IterDomain* id : tv->getLoopDomain()) {
      if (id->isBulk()) {
        id->parallelize(ParallelType::Serial);
      }
    }
  }

  {
    tv2->split(-3, getM(macro));
    tv2->split(-2, getN(macro));
    // [Mo, No, Ko, Mio, Mii, Nio, Nii, Ki]
    // -> [Mo, No, Ko, Mio, Nio, Mii, Nii, Ki]
    tv2->reorder({{-4, -3}});
    tv2->merge(-5);
    tv2->axis(-4)->parallelize(ParallelType::TIDy);
    scheduler_utils::BoundedDirectionalTransformPropagator::forward(
        tv2,
        -1,
        {tv3},
        scheduler_utils::BoundedDirectionalTransformPropagator::Options()
            .propagateParallelType()
            .propagateToBoundary());
  }

  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setAllocationDomain(s.as<IterDomain*>(), true);
    tv2->axis(-1)->parallelize(ParallelType::Mma);
    tv2->axis(-2)->parallelize(ParallelType::Mma);
    tv2->axis(-3)->parallelize(ParallelType::Mma);
  }

  if (!use_smem_epilogue) {
    for (auto tv : {tv3c, tv3}) {
      auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
          tv->getLoopDomain());
      tv->setLoopDomain(s.as<IterDomain*>());
    }
    tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  } else {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3c->getLoopDomain());
    tv3c->setLoopDomain(s.as<IterDomain*>());
    tv3c->setAllocationDomain(s.as<IterDomain*>(), true);

    constexpr int64_t stmatrix_tile_m = 16;
    constexpr int64_t stmatrix_tile_n = 16;
    fusion.manage("st_matrix_m_tile", stmatrix_tile_m);
    fusion.manage("st_matrix_n_tile", stmatrix_tile_n);
    fusion.manage("st_matrix_m", getM(macro));
    fusion.manage("st_matrix_n", getN(macro));

    MmaInputSmemSwizzle store_swizzle =
        mma_utils::tmaSwizzleSharedMemory(tv3_shmem);

    // This internally calls
    // Schedule shared memory cache; Output from StMatrix
    mma_utils::scheduleLdStMatrixForMmaOutput(
        tv3_shmem, stmatrix_tile_m, stmatrix_tile_n);

    // Schedule global memory output; Output from TMA Store
    mma_utils::scheduleTMAStoreForMmaOutput(tv3, store_swizzle);
  }

  inlineMost(ir_utils::allTvsExcept(&fusion, {tv0, tv1, tv0b, tv1b}));

  if (use_warp_specialization) {
    tv0c->circularBuffer(stages, prefetch, WarpSpecialized(ParallelType::TIDy));
    tv1c->circularBuffer(stages, prefetch, WarpSpecialized(ParallelType::TIDy));
  } else {
    tv0c->circularBuffer(stages, prefetch);
    tv1c->circularBuffer(stages, prefetch);
  }

  auto inputs = matmulAtInput3DSS(M, N, K, layout, data_type_to_aten(dtype));
  inputs.first = inputs.first.squeeze();
  inputs.second = inputs.second.squeeze();

  KernelExecutor ke;
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(inputs.first.squeeze(), inputs.second.squeeze(), layout);
  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 1e-5, 1e-5));

  // The following check fails if the BroadcastOps are not removed at lowering
  // time, resulting in two intermediate global allocations.
  const kir::KernelSummary& summary = ke.compiledKernel()->kernel()->summary();
  EXPECT_EQ(summary.global_allocations.size(), 0)
      << "Expected to have no intermediate global allocations";
}

using WarpGroupTestParams = int64_t;
using MLPGemmPersistentBroadcastInputs =
    NVFuserFixtureParamTest<WarpGroupTestParams>;
TEST_P(MLPGemmPersistentBroadcastInputs, NumWarpGroups) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(9, 0, 10, 0);
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 8192, N = 8192, K = 8192;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv4 = fusedMultiplySum(tv0, tv1, {2});
  auto tv3 = castOp(dtype, tv4);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto a_ref = at::randn({M, 1, K}, options);
  auto b_ref = at::randn({1, N, K}, options);
  auto c_ref = at::randn({1, N, K}, options);
  clearL2Cache();

  auto tv3_ref = at::linear(a_ref.squeeze(), b_ref.squeeze());
  clearL2Cache();

  int64_t num_warp_groups = GetParam();
  constexpr int64_t warp_m = 64;
  int64_t cta_m = warp_m * num_warp_groups;

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(cta_m, 256, 64);
  gemm_tile.warp_tile = GemmTile(warp_m, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.grid_traversal_factor = {16, 8};
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  // Legacy launch is faster than Cluster launch when using full 132 SM grid.
  // Cluster launch is better when using 128 SM grid that matches 2d grid
  // traveral.
  mparams.cluster_dims = {1, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);
  std::vector<c10::IValue> inputs = {a_ref, b_ref, c_ref};

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      cg_outputs[0].as<at::Tensor>().allclose(tv3_ref, 1e-6 * K, 1e-6 * K));
}
INSTANTIATE_TEST_SUITE_P(
    HopperMatmulTest,
    MLPGemmPersistentBroadcastInputs,
    ::testing::Range(1L, 3L),
    [](const testing::TestParamInfo<WarpGroupTestParams>& info) {
      std::stringstream ss;
      ss << info.param;
      return sanitizeTestName(ss.str());
    });

TEST_F(HopperMatmulTest, EpilogueBiasPersistentBroadcastInputs) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 8192, N = 8192, K = 8192;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {2});
  auto tv4 = add(tv3, tv2);
  auto tv5 = castOp(DataType::BFloat16, tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({M, N}, options);
  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze(), t2);

  std::vector<c10::IValue> inputs = {t0, t1, t2};

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tv3_ref, 5e-2, 5e-2));
}

TEST_F(HopperMatmulTest, EpilogueSiluPersistentBroadcastInputs) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 8192, N = 8192, K = 8192;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {2});
  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);
  auto tv10 = mul(tv9, tv2);
  auto tv11 = castOp(DataType::BFloat16, tv10);
  fusion.addOutput(tv11);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({M, N}, options);

  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze());
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv11_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * t2).to(at::kBFloat16);

  std::vector<c10::IValue> inputs = {t0, t1, t2};

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(64, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tv11_ref, 5e-2, 1e-1));
}

TEST_F(BlackwellMatmulTest, EpilogueBiasPersistentBroadcastInputs) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 8192, N = 8192, K = 8192;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {2});
  auto tv4 = add(tv3, tv2);
  auto tv5 = castOp(DataType::BFloat16, tv4);
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({M, N}, options);
  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze(), t2);

  std::vector<c10::IValue> inputs = {t0, t1, t2};

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(128, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Blackwell1CTA_128_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tv3_ref, 5e-2, 5e-2));
}

TEST_F(BlackwellMatmulTest, EpilogueSiluPersistentBroadcastInputs) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 8192, N = 8192, K = 8192;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  auto tv2 = makeContigConcreteTensor({-1, -1}, dtype); // M, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = fusedMultiplySum(tv0, tv1, {2});
  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = neg(tv4);
  auto tv6 = exp(tv5);
  auto tv7 = add(fusion.oneVal(DataType::Float), tv6);
  auto tv8 = reciprocal(tv7);
  auto tv9 = mul(tv4, tv8);
  auto tv10 = mul(tv9, tv2);
  auto tv11 = castOp(DataType::BFloat16, tv10);
  fusion.addOutput(tv11);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto t2 = at::randn({M, N}, options);

  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze());
  auto tv4_ref = tv3_ref.to(at::kFloat);
  auto tv11_ref =
      (tv4_ref * (1. / (1.0 + at::exp(-tv4_ref))) * t2).to(at::kBFloat16);

  std::vector<c10::IValue> inputs = {t0, t1, t2};

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 256, 64);
  gemm_tile.warp_tile = GemmTile(128, 256, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Blackwell1CTA_128_256_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {2, 1};
  mparams.promote_prologue_smem_reuse = true;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  // TODO Fix std::get variant error in expression evaluator because of linear
  // index
  // EXPECT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      at::allclose(cg_outputs[0].as<at::Tensor>(), tv11_ref, 5e-2, 1e-1));
}

// Test that when we have a single math warp group we still include a block sync
TEST_F(HopperMatmulTest, HSH_NT_SingleMathGroupSyncCheck) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // K, M
  auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  auto tv3 = castOp(DataType::Half, tv2);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({K, M, 1}, options);
  auto t1 = at::randn({K, 1, N}, options);
  auto out_ref = at::matmul(t0.squeeze().t(), t1.squeeze()).to(at::kHalf);

  MatMulTileOptions gemm_tile;
  // Regardless of the instruction, this should result in 2 warp groups i.e. 256
  // threads
  gemm_tile.cta_tile = GemmTile(64, 64, 64);
  gemm_tile.warp_tile = GemmTile(64, 64, 64);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_64_16;
  mparams.tile_sizes = gemm_tile;
  mparams.tiling_strategy = MatmulParams::TilingStrategy::OneTilePerCTA;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::Pipelined;
  mparams.buffering_loop_level = MatmulParams::BufferingLoopLevel::CTATiles;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::ColumnMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  mparams.circular_buffer_options.smem_circular_buffer_stage = 3;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 1;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = false;
  mparams.cluster_dims = {1, 1};
  mparams.promote_prologue_smem_reuse = false;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;

  ke.registerPostLoweringHook([](kir::Kernel* kernel) {
    class SyncChecker : kir::IrVisitor {
     public:
      static bool check(kir::Kernel* kernel) {
        SyncChecker c(kernel);
        return c.passed_;
      }

     private:
      SyncChecker(kir::Kernel* kernel) {
        kir::IrVisitor::handle(kernel->topLevelExprs());
      }

      kir::ForLoop* getInnerNonTrivialLoop() {
        for (int64_t pos = (int64_t)for_loops_.size() - 1; pos >= 0; --pos) {
          kir::ForLoop* loop = for_loops_.at(pos);
          if (!loop->isTrivial()) {
            return loop;
          }
        }
        return nullptr;
      }

      using OptOutDispatch::dispatch;

      void dispatch(Expr* expr) {
        if (next_expr_must_be_sync_) {
          if (!expr->isA<kir::BlockSync>() ||
              getInnerNonTrivialLoop() != wait_loop_) {
            passed_ = false;
          }
          next_expr_must_be_sync_ = false;
        }
        OptOutDispatch::dispatch(expr);
      }

      using kir::IrVisitor::handle;

      void handle(kir::Asm* aop) {
        if (aop->code().find("wgmma.wait_group") != std::string::npos) {
          kir::ForLoop* loop = getInnerNonTrivialLoop();
          if (loop != nullptr) {
            // We don't need a sync for the wait that is placed at the end of
            // the top scope of the kernel
            wait_loop_ = loop;
            next_expr_must_be_sync_ = true;
          }
        }
      }

     private:
      kir::ForLoop* wait_loop_ = nullptr;
      bool next_expr_must_be_sync_ = false;
      bool passed_ = true;
    };
    EXPECT_TRUE(SyncChecker::check(kernel));
  });

  ke.compile(&fusion, {t0, t1});
  kir::Kernel* kernel = ke.compiledKernel()->kernel();
  ASSERT_TRUE(kernel != nullptr);
  EXPECT_TRUE(getBankConflictInfo(kernel).empty());
  EXPECT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(kernel));

  auto cg_outputs = ke.run({t0, t1});

  // Check number of launched threads matches what we expect
  EXPECT_EQ(ke.lastLaunchParams().bdimx(), 128);
  EXPECT_EQ(ke.lastLaunchParams().bdimy(), 1)
      << " expected 1 warp groups (BIDy==1) but found BIDy=="
      << ke.lastLaunchParams().bdimy();

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

// See https://github.com/NVIDIA/Fuser/issues/4159
TEST_F(HopperMatmulTest, HSS_NT_SplitKTMAStore) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 2048, N = 2048, K = 8192;
  const auto dtype = DataType::Half;

  auto tv0 = makeContigConcreteTensor({-1, -1, 1}, dtype); // K, M
  auto tv1 = makeContigConcreteTensor({-1, 1, -1}, dtype); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = fusedMultiplySum(tv0, tv1, {0});

  // Reorder the accumulator as [M, N, K]
  // [K, M, N] -> [M, N, K]
  tv2->reorder({{-3, -1}});
  tv2->commitLeafToLogical();

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA);
  auto t0 = at::randn({K, M, 1}, options);
  auto t1 = at::randn({K, 1, N}, options);
  auto out_ref =
      at::matmul(t0.squeeze().t().to(at::kFloat), t1.squeeze().to(at::kFloat));

  MatmulParams mparams = defaultHopperParams();
  mparams.use_smem_epilogue = true;
  mparams.splitk_factor = 2;

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  // TODO: Either enable stmatrix for 32-bit outputs or fix current 2-way bank
  // conflict by scheduling the vectorized store properly
  auto bank_conflicts = getBankConflictInfo(ke.compiledKernel()->kernel());
  EXPECT_EQ(bank_conflicts.size(), 1);
  for (const auto& [expr, conflict_ways] : bank_conflicts) {
    int64_t input_ways, output_ways;
    std::tie(input_ways, output_ways) = conflict_ways;
    EXPECT_EQ(input_ways, 0);
    EXPECT_EQ(output_ways, 2);
  }
  auto cg_outputs = ke.run({t0, t1});
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  NVF_CHECK(at::allclose(
      cg_outputs[0].as<at::Tensor>(), out_ref, 1e-6 * K, 1e-6 * K));
}

TEST_F(HopperMatmulTest, PingPongPersistent) {
  EnableOptionsGuard eog;
  EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMultipleMatmuls);

  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 4096, N = 3584, K = 512;
  const auto dtype = DataType::BFloat16;

  auto tv0 = makeContigConcreteTensor({-1, 1, -1}, dtype); // M, 1, K
  auto tv1 = makeContigConcreteTensor({1, -1, -1}, dtype); // 1, N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv3 = fusedMultiplySum(tv0, tv1, {2});
  auto tv4 = castOp(DataType::BFloat16, tv3);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto t0 = at::randn({M, 1, K}, options);
  auto t1 = at::randn({1, N, K}, options);
  auto tv3_ref = at::linear(t0.squeeze(), t1.squeeze());

  std::vector<c10::IValue> inputs = {t0, t1};

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 64);
  gemm_tile.warp_tile = GemmTile(128, 128, 64);

  MatmulParams mparams;
  // Activate Ping-Pong schedule
  mparams.buffering_loop_level = MatmulParams::BufferingLoopLevel::WarpTiles;
  mparams.supported_vec_size = {8, 8, 8};
  mparams.mma_macro = MmaMacro::Hopper_64_128_16;
  mparams.tile_sizes = gemm_tile;
  mparams.cta_order = MatmulParams::TileRasterizationOrder::RowMajor;
  mparams.async_gmem_load_operands = true;
  mparams.circular_buffering_strategy =
      MatmulParams::CircularBufferingStrategy::WarpSpecialized;
  mparams.tiling_strategy =
      MatmulParams::TilingStrategy::DistributeTilesAcrossSMs;
  mparams.circular_buffer_options.circular_buffer_smem_write = true;
  mparams.circular_buffer_options.circular_buffer_smem_read = false;
  // TODO reduced share memory aliasing because of persistent scheduling
  mparams.circular_buffer_options.smem_circular_buffer_stage = 6;
  mparams.circular_buffer_options.smem_circular_buffer_prefetch_gap = 2;
  mparams.splitk_factor = 1;
  mparams.use_smem_epilogue = true;
  mparams.cluster_dims = {1, 2};
  mparams.promote_prologue_smem_reuse = true;
  mparams.grid_traversal_factor = {16, 8};

  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  KernelExecutor ke;
  ke.compile(&fusion, inputs);
  auto cg_outputs = ke.run(inputs);
  ASSERT_FALSE(PredicatedChecker::isCpAsyncMmaPredicatedByIfThenElse(
      ke.compiledKernel()->kernel()));

  // Relax tolerance for larger sum due to large K
  EXPECT_TRUE(
      cg_outputs[0].as<at::Tensor>().allclose(tv3_ref, 1e-6 * K, 1e-6 * K));
}

} // namespace nvfuser
