// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <macros.h>

#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/analysis/bank_conflict.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_profiler.h>
#include <fusion_segmenter.h>
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/pre_segmenter.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <sys_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>
#include <utils.h>

// fuser and IR parser
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <ir/builder.h>
#include <algorithm>
#include <iostream>
#include "c10/core/ScalarType.h"

namespace nvfuser {

using MatmulTest = NVFuserTest;

class MatmulTestWithLayout : public NVFuserTest,
                             public ::testing::WithParamInterface<MmaLayout> {
 protected:
  MmaLayout layout;
  void SetUp() override {
    layout = GetParam();
    NVFuserTest::SetUp();
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmperePrologueFusionBroadcast) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmpereProloguePointwise) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.sin().to(at::kFloat),
      inputs.second.sin().to(at::kFloat),
      layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

TEST_P(MatmulTestWithLayout, AmpereMatmulBFloat16) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Matmul test for Ampere MMA: with pipelined gmem load
TEST_P(MatmulTestWithLayout, AmpereMatmulPipelineGmem) {
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
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

    FusionExecutor fe;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        fe.compileFusion(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
  }
}

// Matmul test for Ampere MMA: checking CTA Swizzles
TEST_P(MatmulTestWithLayout, AmpereSwizzle) {
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

    preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
        &fusion);

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

    MatmulParams mparams;
    mparams.supported_vec_size = {8, 8, 4};
    mparams.mma_macro = MmaMacro::Ampere_16_8_16;
    mparams.tile_sizes = gemm_tile;
    mparams.async_gmem_load_operands = true;
    mparams.circular_buffer_options.circular_buffer_smem_write = true;
    mparams.circular_buffer_options.circular_buffer_smem_read = true;
    mparams.circular_buffer_options.smem_circular_buffer_stage = 3;

    mparams.cta_order = order;
    mparams.grid_swizzle_factor = swizzle;

    SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
        ->schedule(&fusion, &mparams);

    auto inputs = matmulAtInput3DTuring(M, N, K, layout);

    if (!detectComputeSanitizer()) {
      ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
      FusionProfiler::start();
      FusionProfiler::createSegments(1);
    }

    FusionExecutor fe;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        fe.compileFusion(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(cg_outputs[0].allclose(tref, 0.01, 0.01));

    int gdimx = fe.lastLaunchParams().gdimx();
    int gdimy = fe.lastLaunchParams().gdimy();

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

    // Check that mma op is not predicated. This is a regression test for
    // https://github.com/NVIDIA/Fuser/issues/95
    class PredicateChecker : public kir::IrVisitor {
     public:
      using kir::IrVisitor::handle;
      bool found_mma = false;

     private:
      void handle(kir::Asm* asm_) final {
#if IS_CPP20
        if (!asm_->code().starts_with("mma")) {
#else
        if (asm_->code().substr(0, 3) != "mma") {
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
    } pred_checker;

    GpuLower gpulw(&fusion);
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
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

    FusionExecutor fe;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        fe.compileFusion(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
  }
}

// Matmul-Matmul fusion test on Ampere
TEST_F(MatmulTest, MatmulMatmulAmpere) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);
  int M = 512, N = 256, K1 = 128, K2 = 128;

  // Fusion definition (Both gemms are TN)
  // [M,K1]
  auto tv0 = makeContigConcreteTensor({M, K1}, DataType::Half);
  // [K2,K1]
  auto tv1 = makeContigConcreteTensor({K2, K1}, DataType::Half);
  // [N,K2]
  auto tv2 = makeContigConcreteTensor({N, K2}, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  // [M,N,K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});
  auto tv2b = broadcast(tv2, {true, false, false});

  // [M,K2,R]
  auto tv3 = fusedMultiplySum(tv0b, tv1b, {2});

  auto tv3h = castOp(DataType::Half, tv3);
  auto tv3b = broadcast(tv3h, {false, true, false});

  auto tv4 = fusedMultiplySum(tv3b, tv2b, {2});

  fusion.addOutput(tv4);

  // Fusion:
  //  Gemm(M,K2,K1) x Gemm(M,N,K2)

  MatMulTileOptions gemm_tile1, gemm_tile2;

  // cta tile:
  //  To save register, n of cta tile 1
  //  matches k of cta tile2
  gemm_tile1.cta_tile = GemmTile(128, 64, 32);
  gemm_tile2.cta_tile = GemmTile(128, 32, 64);

  // Distribute to 2x2 warps
  gemm_tile1.warp_tile = GemmTile(64, 32, 32);
  gemm_tile2.warp_tile = GemmTile(64, 16, 64);

  // Using Ampere mma macro
  gemm_tile2.instruction_tile = GemmTile(16, 8, 16);
  gemm_tile2.instruction_tile = GemmTile(16, 8, 16);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      2 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 2, got ",
      mma_ops.size());
  mma_ops[0]->setMacro(MmaMacro::Ampere_16_8_16);
  mma_ops[1]->setMacro(MmaMacro::Ampere_16_8_16);

  // Global read for gemm 1
  auto tv0r = tv0->cacheAfter();
  auto tv1r = tv1->cacheAfter();

  // Global read for gemm 2
  auto tv2r = tv2->cacheAfter();

  // Gemm 1 main loop read
  auto tv0cw = tv0r->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1r->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  // Gemm 1 accumulator reg
  auto tv3c = tv3->cacheBefore();

  // Gemm 2 main loop read
  auto tv3cw = tv3h->cacheAfter();
  auto tv3cr = tv3cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2cw = tv2r->cacheAfter();
  auto tv2cr = tv2cw->cacheAfter(LoadStoreOpType::LdMatrix);

  // Gemm 2 accumulator reg
  auto tv4c = tv4->cacheBefore();

  // General idea is inlining gemm1's main loop inside gemm2's

  // Schedule gemm 2:
  // ------------------------------------------------------------------
  tv4->split(-2, gemm_tile2.cta_tile.m);
  tv4->split(-1, gemm_tile2.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv4->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv2->computeAt(tv4, 2);
  tv3->computeAt(tv4, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv4c->split(-1, gemm_tile2.cta_tile.k);
  tv4c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv3->computeAt(tv4c, 3); // Implicitly defines cta tile of gemm1
  tv2r->computeAt(tv4c, 3);

  // Make warp tile
  mma_utils::scheduleWarpTileWithReduction(tv4c, gemm_tile2);
  mma_utils::scheduleWarpTileWithNoReduction(tv4, gemm_tile2);
  //           -8   -7  -6 -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv3cr->computeAt(tv4c, -4);
  tv2cr->computeAt(tv4c, -4);

  // Schedule tv2 gmem read and smem write:
  // ----------------------------------------------------------------
  // [No,Ko,N,K]
  tv2cw->merge(-2);
  tv2r->merge(-2);

  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv2cw, gemm_tile2, 8);
  mma_utils::scheduleContiguousVectorLoad(tv2r, gemm_tile2, 8);
  tv2cw->setMemoryType(MemoryType::Shared);

  // Schedule tv2 gmem read and smem write:
  // ----------------------------------------------------------------

  // Schedule gemm 2 mma input
  // ---------------------------------------------------------------------------
  tv3cr->applyMmaSwizzle(MmaOperand::A);

  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv3b->reorder({{-2, -3}, {-3, -2}});
  tv3b->applyMmaSwizzle(MmaOperand::A);

  tv2cr->applyMmaSwizzle(MmaOperand::B);
  tv2b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv4c->getLoopDomain());
    tv4c->setLoopDomain(s.as<IterDomain*>());
    tv4c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv4->getLoopDomain());
    tv4->setLoopDomain(s.as<IterDomain*>());
  }

  // Schedule gemm 1:
  // ------------------------------------------------------------------

  // CTA tile:
  tv0->computeAt(tv3, 2);
  tv1->computeAt(tv3, 2);

  // Schedule K dim for gemm 1:

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv3c->split(-1, gemm_tile1.cta_tile.k);
  tv3c->reorder({{2, 3}, {3, 4}, {4, 2}});
  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv3c, 3);
  tv1r->computeAt(tv3c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  mma_utils::scheduleWarpTileWithReduction(tv3c, gemm_tile1);
  mma_utils::scheduleWarpTileWithNoReduction(tv3cw, gemm_tile1);

  tv0cr->computeAt(tv3c, -4);
  tv1cr->computeAt(tv3c, -4);

  tv3->computeAt(tv3cw, -3);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,Ko,M,K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(tv0cw, gemm_tile1, 8);
  mma_utils::scheduleContiguousVectorLoad(tv0r, gemm_tile1, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [No,Ko,N,K]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv1cw, gemm_tile1, 8);
  mma_utils::scheduleContiguousVectorLoad(tv1r, gemm_tile1, 8);
  tv1cw->setMemoryType(MemoryType::Shared);

  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(MmaOperand::A);
  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3c->getLoopDomain());
    tv3c->setLoopDomain(s.as<IterDomain*>());
    tv3c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3cw->getLoopDomain());
    tv3cw->setLoopDomain(s.as<IterDomain*>());
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3h->getLoopDomain());
    tv3h->setLoopDomain(s.as<IterDomain*>());
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3->getLoopDomain());
    tv3->setLoopDomain(s.as<IterDomain*>());
  }
  tv3cw->setMemoryType(MemoryType::Shared);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  // Gemm 1
  tv3c->axis(4)->parallelize(ParallelType::TIDz);
  tv3c->axis(5)->parallelize(ParallelType::TIDy);

  tv3->computeAt(tv3cw, -2);
  tv3cw->axis(2)->parallelize(ParallelType::TIDz);
  tv3cw->axis(3)->parallelize(ParallelType::TIDy);

  // Gemm 2
  tv4->axis(2)->parallelize(ParallelType::TIDz);
  tv4->axis(3)->parallelize(ParallelType::TIDy);
  tv4c->axis(4)->parallelize(ParallelType::TIDz);
  tv4c->axis(5)->parallelize(ParallelType::TIDy);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDy);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K1}, options);
  auto t1 = at::randn({K2, K1}, options);
  auto t2 = at::randn({N, K2}, options);

  auto tref = t0.to(at::kFloat)
                  .matmul(t1.t().to(at::kFloat))
                  .matmul(t2.t().to(at::kFloat));

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1, t2}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1, t2});

  // relaxed check for now, err accumulation is significant.
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.1, 0.1));
}

// Simplified Matmul-Softmax-Matmul test on Ampere
//   (To be extended in follow ups)
TEST_F(MatmulTest, MatmulSoftmaxMatmulAmpere) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Omitting outer dimensions and pointwise ops

  const int seql_q = 32;
  const int seql_k = 128;
  const int hidden_size = 1024;
  const int num_heads = 16;
  const int head_dim = hidden_size / num_heads;

  // Gemm 1:
  // (80, 80, 64)
  const int M1 = seql_q, N1 = seql_k, K1 = head_dim;
  // (64, 80)
  const int N2 = head_dim, K2 = seql_k;

  // Fusion definition (Both gemms are TN)
  // [M,K1]
  auto inp = makeContigConcreteTensor({M1, K1}, DataType::Half);
  // Query matrix
  auto qk = makeContigConcreteTensor({N1, K1}, DataType::Half);
  // Second linear matrix
  auto acc = makeContigConcreteTensor({N2, K2}, DataType::Half);

  fusion.addInput(inp);
  fusion.addInput(qk);
  fusion.addInput(acc);

  // [M,N,K]
  auto tv0b = broadcast(inp, {false, true, false});
  auto tv1b = broadcast(qk, {true, false, false});
  auto tv2b = broadcast(acc, {true, false, false});

  // [M,K2,R]
  auto tv3 = fusedMultiplySum(tv0b, tv1b, {2});

  // Inline define softmax for now for scheduling
  auto x = tv3;
  const int kReductionAxis = 1;
  const int kNumberOfDims = 2;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(x, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(x, bcast_max);
  auto exp_val = exp(x_max_sub);
  auto sum_exp = sum(exp_val, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto recip = reciprocal(bcast_sum);
  auto tv3sfm = mul(exp_val, recip);

  auto tv3h = castOp(DataType::Half, tv3sfm);
  auto tv3b = broadcast(tv3h, {false, true, false});
  auto tv4 = fusedMultiplySum(tv3b, tv2b, {2});

  fusion.addOutput(tv4);

  // Fusion:
  //  Gemm(M,K2,K1) x Gemm(M,N,K2)
  MatMulTileOptions gemm_tile;

  // TODO: use very small tiles for now since
  //  alias pass is not re-using smem. Fix later.
  gemm_tile.cta_tile = GemmTile(32, 128, 32);

  // Distribute to 2x2 warps
  gemm_tile.warp_tile = GemmTile(16, 64, 32);

  // Using Ampere mma macro
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      2 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 2, got ",
      mma_ops.size());
  mma_ops[0]->setMacro(MmaMacro::Ampere_16_8_16);
  mma_ops[1]->setMacro(MmaMacro::Ampere_16_8_16);

  // Global read for gemm 1
  auto tv0r = inp->cacheAfter();
  auto tv1r = qk->cacheAfter();

  // Global read for gemm 2
  auto tv2r = acc->cacheAfter();

  // Gemm 1 main loop read
  auto tv0cw = tv0r->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1r->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  // Gemm 1 accumulator reg
  auto tv3c = tv3->cacheBefore();

  // Softmax conversion:
  auto tv3ccr = tv3->cacheAfter();

  // tv3ccr -> tv3h : softmax

  // Gemm 2 main loop read
  auto tv3cr = tv3h->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2cw = tv2r->cacheAfter();
  auto tv2cr = tv2cw->cacheAfter(LoadStoreOpType::LdMatrix);

  // Gemm 2 accumulator reg
  auto tv4c = tv4->cacheBefore();

  // Schedule gemm 2:
  // ------------------------------------------------------------------
  tv4->split(-2, gemm_tile.cta_tile.m);
  tv4->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv4->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  acc->computeAt(tv4, 2);
  tv3->computeAt(tv4, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv4c->split(-1, gemm_tile.cta_tile.k);
  tv4c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv3->computeAt(tv4c, 2);
  tv2r->computeAt(tv4c, 3);

  // Make warp tile
  mma_utils::scheduleWarpTileWithReduction(tv4c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv4, gemm_tile);
  //           -8  -7  -6  -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv3cr->computeAt(tv4c, -4);
  tv2cr->computeAt(tv4c, -4);

  // Schedule tv2 gmem read and smem write:
  // ----------------------------------------------------------------
  // [No,Ko,N,K]
  tv2cw->merge(-2);
  tv2r->merge(-2);

  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv2cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv2r, gemm_tile, 8);
  tv2cw->setMemoryType(MemoryType::Shared);

  // Schedule tv2 gmem read and smem write:
  // ----------------------------------------------------------------

  // Schedule gemm 2 mma input
  // ---------------------------------------------------------------------------
  tv3cr->applyMmaSwizzle(MmaOperand::A);
  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv3b->reorder({{-2, -3}, {-3, -2}});
  tv3b->applyMmaSwizzle(MmaOperand::A);

  tv2cr->applyMmaSwizzle(MmaOperand::B);
  tv2b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv4c->getLoopDomain());
    tv4c->setLoopDomain(s.as<IterDomain*>());
    tv4c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv4->getLoopDomain());
    tv4->setLoopDomain(s.as<IterDomain*>());
  }

  // Schedule gemm 1:
  // ------------------------------------------------------------------

  // CTA tile:
  // [Mo, Mi128, N80]

  tv3->split(-1, gemm_tile.cta_tile.n);
  // [Mo, Mi128, No, Ni128]

  tv3->reorder({{1, 2}, {2, 1}});

  // [Mo, No, Mi128, Ni128]
  inp->computeAt(tv3, 2);
  qk->computeAt(tv3, 2);

  // Schedule K dim for gemm 1:

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv3c->split(-1, gemm_tile.cta_tile.k);
  tv3c->reorder({{2, 3}, {3, 4}, {4, 2}});
  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv3c, 3);
  tv1r->computeAt(tv3c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  mma_utils::scheduleWarpTileWithReduction(tv3c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv3, gemm_tile);

  tv0cr->computeAt(tv3c, -4);
  tv1cr->computeAt(tv3c, -4);

  // tv3->computeAt(tv3cw,-3);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,Ko,M,K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(tv0cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv0r, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [No,Ko,N,K]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv1cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv1r, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);

  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(MmaOperand::A);
  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // // Schedule mma output
  // //
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3c->getLoopDomain());
    tv3c->setLoopDomain(s.as<IterDomain*>());
    tv3c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv3->getLoopDomain());
    tv3->setLoopDomain(s.as<IterDomain*>());
  }

  // Put tv3 result in smem
  tv3->setMemoryType(MemoryType::Shared);

  // schedule a reg persistent softmax: from tv3
  // [Mo, M128, RN]
  max_val->split(-1, 128);
  // [Mo, M128, RN1, RN128]
  max_val->split(-1, 4);
  // Map to warp (2x2)
  max_val->split(-4, 4);
  max_val->split(-4, 2);

  // [Mo, Mo32, My2, Mx2, RN1, RNo32, RNi4]
  auto max_rf = max_val->rFactor({-1});
  // [Mo, Mo32, My2, Mx2, RN1, I32, RNi4]

  // [Mo, M128, RN]
  sum_exp->split(-1, 128);
  // [Mo, M128, RN1, RN128]
  sum_exp->split(-1, 4);
  // Map to warp (2x2)
  sum_exp->split(-4, 4);
  sum_exp->split(-4, 2);

  // [Mo, Mo32, My2, Mx2, RN1, RNo32, RNi4]
  auto sum_exp_rf = sum_exp->rFactor({-1});
  // [Mo, Mo32, My2, Mx2, RN1, I32, RNi4]

  exp_val->computeAt(sum_exp_rf, 4);
  exp_val->split(-1, 128);
  exp_val->split(-1, 4);
  bcast_max->computeAt(exp_val, -2);

  // [Mo, Mo32, My2, Mx2, IN1, I32, INi4]

  // Read from smem
  tv3ccr->computeAt(max_rf, 4);
  // [Mo, Mo32, My2, Mx2, N80]
  tv3ccr->split(-1, 128);
  tv3ccr->split(-1, 4);
  // [Mo, Mo32, My2, Mx2, IN1, I32, INi4]

  // Write to second gemm
  tv3h->split(-1, 128);
  tv3h->split(-1, 4);
  // Map to warp (2x2)
  tv3h->split(-4, 4);
  tv3h->split(-4, 2);

  bcast_sum->computeAt(tv3h, -2);

  tv3h->setMemoryType(MemoryType::Shared);

  // Parallelize
  tv4->axis(0)->parallelize(ParallelType::BIDx);
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  // Gemm 1
  tv3c->axis(4)->parallelize(ParallelType::TIDz);
  tv3c->axis(5)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDz);
  tv3->axis(3)->parallelize(ParallelType::TIDy);

  auto parallelize_non_reduced_val = [](TensorView* tv) {
    tv->axis(-2)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::TIDz);
    tv->axis(3)->parallelize(ParallelType::TIDy);
  };

  auto parallelize_reduced_val = [](TensorView* tv) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
    tv->axis(2)->parallelize(ParallelType::TIDz);
    tv->axis(3)->parallelize(ParallelType::TIDy);
  };

  parallelize_non_reduced_val(tv3h);
  parallelize_non_reduced_val(max_rf);
  parallelize_non_reduced_val(bcast_max);
  parallelize_non_reduced_val(exp_val);
  parallelize_non_reduced_val(sum_exp_rf);
  parallelize_non_reduced_val(bcast_sum);
  parallelize_non_reduced_val(recip);

  parallelize_reduced_val(max_val);
  parallelize_reduced_val(sum_exp);

  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  // Gemm 2
  tv4->axis(2)->parallelize(ParallelType::TIDz);
  tv4->axis(3)->parallelize(ParallelType::TIDy);
  tv4c->axis(4)->parallelize(ParallelType::TIDz);
  tv4c->axis(5)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M1, K1}, options);
  auto t1 = at::randn({N1, K1}, options);
  auto t2 = at::randn({N2, K2}, options);

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1, t2}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1, t2});

  auto g1 = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));
  auto sg1 = at::_softmax(g1, -1, false);
  auto gsg1 = sg1.matmul(t2.t().to(at::kFloat));

  NVF_CHECK(cg_outputs[0].allclose(gsg1, 0.001, 0.001));
}

// Matmul test for Turing MMA: across supported layouts
TEST_P(MatmulTestWithLayout, TuringMatmul) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Turing_16_8_16;
  mparams.tile_sizes = gemm_tile;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      7, 5, fe.compileFusion(&fusion, {inputs.first, inputs.second}));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Matmul test on ampere, using ampere memory ops
TEST_F(MatmulTest, AmpereMatmulTNCpAsync) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 255, N = 511, K = 88;

  // [M,K]
  auto tv0 = makeContigTensor(2, DataType::Half);
  // [N,K]
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,N,K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(MmaMacro::Ampere_16_8_16);

  auto tv0cw = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1->cacheAfter(LoadStoreOpType::CpAsync);
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv2c = tv2->cacheBefore();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0cw->computeAt(tv2c, 3);
  tv1cw->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  mma_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv2, gemm_tile);
  //           -8  -7  -6  -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,Ko,M,K]
  tv0cw->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(tv0cw, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [No,Ko,N,K]
  tv1cw->merge(-2);
  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv1cw, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(MmaOperand::A);
  // [... Mi, Ni, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Parallelize
  //  0   1  2  3   4   5  6  7  8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  tv2c->axis(4)->parallelize(ParallelType::TIDz);
  tv2c->axis(5)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

TEST_F(MatmulTest, AmpereStridedBatchedMatmulTN) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);
  int64_t M = 511, N = 123, K = 88, B0 = 3, B1 = 5;

  // [B0 ,M, B1, K]
  auto tv0 = makeContigTensor(4, DataType::Half);
  // [B0, N, B1, K]
  auto tv1 = makeContigTensor(4, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [B0, M, N, B1, K]
  auto tv0b = broadcast(tv0, {false, false, true, false, false});
  auto tv1b = broadcast(tv1, {false, true, false, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {4});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(MmaMacro::Ampere_16_8_16);

  auto tv0r = tv0->cacheAfter();
  auto tv1r = tv1->cacheAfter();
  auto tv0cw = tv0r->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1r->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv2c = tv2->cacheBefore();

  // Group the BATCHED DIMS:
  //  -4 -3  -2 -1
  // [B0, M, N, B1]
  tv2->reorder({{-3, -2}, {-2, -1}, {-1, -4}});

  //  -4  -3 -2  -1
  // [B0, B1, M, N]

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [B0, B1, M, N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1   2   3     4   5
  // [B0, B1, Mo, M128, No, N128]
  tv2->reorder({{-3, -2}, {-2, -3}});

  //  0   1   2   3   4     5
  // [B0, B1, Mo, No, M128, N128]

  // Merge the outer dims:
  tv2->merge(0);
  tv2->merge(0);

  //  0   1   2     3
  // [Mo, No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1   2     3     4   5
  // [Mo, No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1   2   3     4     5
  // [Mo, No, Ko, M128, N128, K32]
  tv0r->computeAt(tv2c, 3);
  tv1r->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  mma_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv2, gemm_tile);
  //           -8  -7  -6  -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo, Ko, M, K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(tv0cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv0r, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo, Ko, i, wy, wx, v]

  // [No, Ko, N, K]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [No, Ko, i, wy, wx, v]
  mma_utils::scheduleContiguousVectorLoad(tv1cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv1r, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(MmaOperand::A);

  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Parallelize
  //  0   1  2  3   4   5  6  7   8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  tv2c->axis(4)->parallelize(ParallelType::TIDz);
  tv2c->axis(5)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({B0, M, B1, K}, options);
  auto t1 = at::randn({B0, N, B1, K}, options);

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1});

  // ref implementation:
  auto ref_t0 = t0.permute({0, 2, 1, 3})
                    .contiguous()
                    .view({B0 * B1, M, K}); // B0, B1, M, K
  auto ref_t1 = t1.permute({0, 2, 3, 1})
                    .contiguous()
                    .view({B0 * B1, K, N}); // B0, B1, K, N
  auto ref_permuted =
      ref_t0.to(at::kFloat).bmm(ref_t1.to(at::kFloat)); // B0*B1, M,N
  auto ref = ref_permuted.view({B0, B1, M, N})
                 .permute({0, 2, 3, 1})
                 .contiguous(); // B0,M,N,B1
  NVF_CHECK(cg_outputs[0].allclose(ref, 0.0001, 0.0001));
}

// Matmul test on Ampere with a reshape on prolog
TEST_F(MatmulTest, AmpereViewMatmulTN) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);
  int M = 511, N = 257, K = 88;
  int Ko = 11, Ki = 8;

  // [M,Ko,Ki]
  auto tv0 = makeContigTensor(3, DataType::Half);
  // [N,K]
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv0_reshape = reshape(tv0, {M, Ko, Ki}, {M, K});

  // [M,N,K]
  auto tv0b = broadcast(tv0_reshape, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(MmaMacro::Ampere_16_8_16);

  auto tv0r = tv0->cacheAfter();
  auto tv1r = tv1->cacheAfter();
  auto tv0cw = tv0_reshape->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1r->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv2c = tv2->cacheBefore();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0r->computeAt(tv2c, 3);
  tv1r->computeAt(tv2c, 3);

  // Make warp tile:
  // -------------------------------------------------------------------------
  mma_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv2, gemm_tile);
  //           -8  -7  -6  -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  // ---------------------------------------------------------------------------
  // [Mo,Ko,M,K]
  tv0cw->merge(-2);
  tv0r->merge(-2);
  tv0_reshape->merge(-2);
  mma_utils::scheduleContiguousVectorLoad(tv0cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv0r, gemm_tile, 8);
  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [No,Ko,N,K]
  tv1cw->merge(-2);
  tv1r->merge(-2);
  // [No,Ko,i,wy,wx,v]
  mma_utils::scheduleContiguousVectorLoad(tv1cw, gemm_tile, 8);
  mma_utils::scheduleContiguousVectorLoad(tv1r, gemm_tile, 8);
  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  // ---------------------------------------------------------------------------
  tv0cr->applyMmaSwizzle(MmaOperand::A);

  // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  // ---------------------------------------------------------------------------
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Inline the reshape op with the shared mem write minus
  //  the vectorization axes for now.
  tv0_reshape->computeAt(tv0cw, -2);

  // Parallelize
  //  0   1  2  3   4   5  6  7  8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  tv2c->axis(4)->parallelize(ParallelType::TIDz);
  tv2c->axis(5)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, Ko, Ki}, options);
  auto t1 = at::randn({N, K}, options);

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref =
      at::native::view(t0, {M, K}).to(at::kFloat).matmul(t1.t().to(at::kFloat));

  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Test an end-to-end matmul case with swizzled smem
// data layout.
TEST_F(MatmulTest, AmpereMatmulTNSwizzled) {
  NVFUSER_TEST_CUDA_ARCH_GUARD(8, 0);

  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 257, N = 511, K = 136;

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(128, 128, 32);
  gemm_tile.warp_tile = GemmTile(64, 64, 32);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  // [M,K]
  auto tv0 = makeContigTensor(2, DataType::Half);
  // [N,K]
  auto tv1 = makeContigTensor(2, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M,N,K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(MmaMacro::Turing_16_8_16);

  auto tv0cw = tv0->cacheAfter(LoadStoreOpType::CpAsync);
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1->cacheAfter(LoadStoreOpType::CpAsync);
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv2c = tv2->cacheBefore();

  // Make a CTA tile
  // ------------------------------------------------------------------
  // [M,N]
  tv2->split(-2, gemm_tile.cta_tile.m);
  tv2->split(-1, gemm_tile.cta_tile.n);

  //  0   1    2   3
  // [Mo,M128, No, N128]
  tv2->reorder({{1, 2}, {2, 1}});

  //  0   1    2   3
  // [Mo,No, M128, N128]
  tv0->computeAt(tv2, 2);
  tv1->computeAt(tv2, 2);

  // Order K
  //  0   1    2   3     4    5
  // [Mo,No, M128, N128, Ko, K32]
  tv2c->split(-1, gemm_tile.cta_tile.k);
  tv2c->reorder({{2, 3}, {3, 4}, {4, 2}});

  //  0   1  2   3     4    5
  // [Mo,No, Ko M128, N128, K32]
  tv0cw->computeAt(tv2c, 3);
  tv1cw->computeAt(tv2c, 3);

  // Make warp tile:
  //
  mma_utils::scheduleWarpTileWithReduction(tv2c, gemm_tile);
  mma_utils::scheduleWarpTileWithNoReduction(tv2, gemm_tile);
  //           -8   -7 -6 -5 -4 -3 -2 -1
  // [Mo No Ko Kwo Mwo Nwo Mw Nw Mi Ni Ki]
  tv0cr->computeAt(tv2c, -4);
  tv1cr->computeAt(tv2c, -4);

  // Schedule gmem read and smem write:
  //
  // [Mo,Ko,M,K]
  // Swizzle tv0: 128 x 32 tile:
  tv0cw->split(-2, 8);
  tv0cw->split(-2, 2);
  tv0cw->split(-1, 8);
  //        -5   -4 -3 -2 -1
  // [Mo,Ko,Mo16,M4,M2,Ko4,K8]
  tv0cw->swizzle(Swizzle2DType::XOR, -4, -2);
  tv0cw->merge(-4);
  tv0cw->merge(-3);
  //         -3   -2  -1
  // [Mo,Ko,Mo16,warp,K8]
  tv0cw->split(-3, 4);
  tv0cw->split(-3, 2);
  //             -4  -3   -2  -1
  // [Mo,Ko, S4, wz2, wy2, warp,K8]
  tv0cw->axis(-4)->parallelize(ParallelType::TIDz);
  tv0cw->axis(-3)->parallelize(ParallelType::TIDy);
  tv0cw->axis(-2)->parallelize(ParallelType::TIDx);
  tv0cw->axis(-1)->parallelize(ParallelType::Vectorize);

  tv0cw->setMemoryType(MemoryType::Shared);
  // [Mo,Ko,i,wy,wx,v]

  // [No,Ko,N,K]
  // Swizzle tv0: 128 x 32 tile:
  tv1cw->split(-2, 8);
  tv1cw->split(-2, 2);
  tv1cw->split(-1, 8);
  //        -5   -4 -3 -2 -1
  // [No,Ko,No16,N4,N2,Ko4,K8]
  tv1cw->swizzle(Swizzle2DType::XOR, -4, -2);
  tv1cw->merge(-4);
  tv1cw->merge(-3);
  //         -3   -2  -1
  // [No,Ko,No16,warp,K8]
  tv1cw->split(-3, 4);
  tv1cw->split(-3, 2);
  //             -4  -3   -2  -1
  // [No,Ko, S4, wz2, wy2, warp,K8]
  tv1cw->axis(-4)->parallelize(ParallelType::TIDz);
  tv1cw->axis(-3)->parallelize(ParallelType::TIDy);
  tv1cw->axis(-2)->parallelize(ParallelType::TIDx);
  tv1cw->axis(-1)->parallelize(ParallelType::Vectorize);

  tv1cw->setMemoryType(MemoryType::Shared);
  // Schedule mma input
  tv0cr->applyMmaSwizzle(MmaOperand::A);

  // [... Mi, Ni, Ki]
  tv0b->reorder({{-2, -3}, {-3, -2}});
  tv0b->applyMmaSwizzle(MmaOperand::A);

  tv1cr->applyMmaSwizzle(MmaOperand::B);
  tv1b->applyMmaSwizzle(MmaOperand::B);

  // Schedule mma output
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2c->getLoopDomain());
    tv2c->setLoopDomain(s.as<IterDomain*>());
    tv2c->setAllocationDomain(s.as<IterDomain*>(), true);
  }
  {
    auto s = mma_utils::MmaSwizzler::scheduleMmaOutputAllocation(
        tv2->getLoopDomain());
    tv2->setLoopDomain(s.as<IterDomain*>());
  }

  // Parallelize
  //  0   1  2  3   4   5  6   7  8  9  10
  // [Mo No Ko Kwo Mwo Nwo Mw Nw (Mi Ni Ki)]
  tv2c->axis(4)->parallelize(ParallelType::TIDz);
  tv2c->axis(5)->parallelize(ParallelType::TIDy);

  // Parallelize
  //  0  1  2   3   4   5  6  7
  // [Mo No Mwo Nwo Mw Nw (Mi Ni)]
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::BIDy);
  tv2->axis(2)->parallelize(ParallelType::TIDz);
  tv2->axis(3)->parallelize(ParallelType::TIDy);

  tv0cw->circularBuffer(/*number_of_stages=*/2);
  tv1cw->circularBuffer(/*number_of_stages=*/2);
  tv0cr->circularBuffer(/*number_of_stages=*/2);
  tv1cr->circularBuffer(/*number_of_stages=*/2);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({M, K}, options);
  auto t1 = at::randn({N, K}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Matmul test on Ampere using ldmatrix.x4 to load operands
TEST_P(MatmulTestWithLayout, AmpereMatmulLargeLoad) {
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
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);
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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Matmul test for Turing MMA: across supported layouts
TEST_P(MatmulTestWithLayout, TuringMatmulLargeLoad) {
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
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  MatmulParams mparams;
  mparams.supported_vec_size = {8, 8, 4};
  mparams.mma_macro = MmaMacro::Turing_16_16_16;
  mparams.tile_sizes = gemm_tile;
  SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
      ->schedule(&fusion, &mparams);

  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      7,
      5,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Tile layout check for symmetric 4-warp recipes
TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck4warp) {
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
      gemm_tile.instruction_tile = GemmTile(16, 16, 16);

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

      FusionExecutor fe;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          8,
          0,
          fe.compileFusion(
              &fusion,
              {inputs.first, inputs.second},
              LaunchParams(),
              matmul_cparams));
      EXPECT_TRUE(getBankConflictInfo(fe.kernel()).empty());
      auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
      auto tref = atMatmul(
          inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
      NVF_CHECK(
          cg_outputs[0].allclose(tref, 0.0001, 0.0001),
          "error :",
          (cg_outputs[0] - tref).abs().max(),
          "tile dim:",
          mn_size,
          " ",
          k_size);
    }
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck8warp) {
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
        gemm_tile.instruction_tile = GemmTile(16, 16, 16);

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

        FusionExecutor fe;
        NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
            8,
            0,
            fe.compileFusion(
                &fusion,
                {inputs.first, inputs.second},
                LaunchParams(),
                matmul_cparams));
        ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
        auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
        auto tref = atMatmul(
            inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
        NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
      }
    }
  }
}

TEST_P(MatmulTestWithLayout, AmpereMatmulTileCheck6warp) {
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
    gemm_tile.instruction_tile = GemmTile(16, 16, 16);

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

    FusionExecutor fe;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        fe.compileFusion(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
  }
}

// Matmul test on Ampere using ldmatrix.x4 to load operands
TEST_P(MatmulTestWithLayout, AmpereMatmulLargeLoadLargeK) {
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
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.001, 0.001));
}

// Matmul test for Ampere MMA: across supported layouts
TEST_P(MatmulTestWithLayout, AmpereSplitKLikeStridedBatchedMatmul) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({t0, t1});
  auto tref = splitkLikeAtMatmul(t0.to(at::kFloat), t1.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
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
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

    FusionExecutor fe;
    NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
        8,
        0,
        fe.compileFusion(
            &fusion,
            {inputs.first, inputs.second},
            LaunchParams(),
            matmul_cparams));
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

    // check bank conflicts
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    // (0.001, 0.001) passed on local A100 but failed on CI A100
    NVF_CHECK(
        cg_outputs[0].allclose(tref, 0.01, 0.01),
        "Result validation failed. Max diff: ",
        (cg_outputs[0] - tref).abs().max());

    if (!mparams.use_smem_epilogue) {
      GTEST_SKIP()
          << "Test conducted without utilizing shared memory epilogue due to the device's constrained shared memory capacity.";
    }

    // Check that smem is allocated as expected.
    // There are three cases that are determined by the current device in
    // mma_utils::generateSharedMemoryEpilogueHeuristics:
    //   - !use_smem_epilogue : A + B (this test is skipped in this case)
    //   - use_smem_epilogue && !promote_prologue_smem_reuse : A + B + C
    //   - use_smem_epilogue && promote_prologue_smem_reuse : max(A + B, C)
    auto smem_allocs = fe.kernel()->summary().dynamic_smem_allocations;
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
              dataTypeSize(smem_allocs.at(1)->buffer()->dtype()));
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
                  dataTypeSize(smem_allocs.at(1)->buffer()->dtype()) +
              smem_allocs.at(2)->size()->evaluate() *
                  dataTypeSize(smem_allocs.at(2)->buffer()->dtype()));
      EXPECT_EQ(
          smem_allocs.at(1)->address()->evaluate(),
          smem_allocs.at(2)->size()->evaluate() *
              dataTypeSize(smem_allocs.at(2)->buffer()->dtype()));
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  // FusionExecutor::compileFusion would fail otherwise.
  SKIP_IF_INSUFFICIENT_SMEM(&mparams, data_types);

  at::manual_seed(0);
  auto inputs = matmulAtInput3DTuring(M, N, K, layout);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      cg_outputs[0].allclose(tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0] - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP()
        << "Test conducted without utilizing shared memory epilogue due to the device's constrained shared memory capacity.";
  }
  if (!mparams.promote_prologue_smem_reuse) {
    GTEST_SKIP()
        << "Test conducted with shared memory epilogue but without promoting prologue smem re-use.";
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  tref = tref.to(at::kHalf);
  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      cg_outputs[0].allclose(tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0] - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP()
        << "Test conducted without utilizing shared memory epilogue due to the device's constrained shared memory capacity.";
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto t2 = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  auto tref = at::relu(t2).to(at::kFloat);

  // check bank conflicts
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  // (0.001, 0.001) passed on local A100 but failed on CI A100
  NVF_CHECK(
      cg_outputs[0].allclose(tref, 0.01, 0.01),
      "Result validation failed. Max diff: ",
      (cg_outputs[0] - tref).abs().max());

  if (!mparams.use_smem_epilogue) {
    GTEST_SKIP()
        << "Test conducted without utilizing shared memory epilogue due to the device's constrained shared memory capacity.";
  }
}

// Test the matmul scheduler's single-kernel split-K support
TEST_P(MatmulTestWithLayout, FusionAmpereMatmulSplitK_CUDA) {
  // requires Ampere or higher GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

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
      gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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
          std::cout
              << "Skipping smem epilogue due to shared memory constraints on this device"
              << std::endl;
          continue;
        }
        mparams.promote_prologue_smem_reuse = true;
      }

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto inputs = matmulAtInput3DTuring(M, N, K, layout);

      FusionExecutor fe;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, fe.compileFusion(&fusion, {inputs.first, inputs.second}));
      EXPECT_TRUE(getBankConflictInfo(fe.kernel()).empty());
      auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
      auto tref = atMatmul(
          inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);

      // Relax tolerance for larger sum due to large K
      NVF_CHECK(cg_outputs[0].allclose(tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Test splitk with bias epilogue
TEST_P(MatmulTestWithLayout, FusionAmpereMatmulSplitKBias_CUDA) {
  // requires Ampere or higher GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

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
      gemm_tile.instruction_tile = GemmTile(16, 8, 16);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      auto [aten_a, aten_b] = matmulAtInput3DTuring(M, N, K, layout);
      at::Tensor aten_bias = at::randn({M}, aten_a.options());
      std::vector<c10::IValue> inputs = {aten_a, aten_b, aten_bias};

      FusionExecutor fe;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, fe.compileFusion(&fusion, inputs));
      EXPECT_TRUE(getBankConflictInfo(fe.kernel()).empty());
      auto cg_outputs = fe.runFusion(inputs);
      auto tref = atBiasEpilogue(
          atMatmul(aten_a.to(at::kFloat), aten_b.to(at::kFloat), layout),
          aten_bias);

      // Relax tolerance for larger sum due to large K
      EXPECT_TRUE(cg_outputs[0].allclose(tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Same as above but has a batch dimension and splitk
TEST_P(MatmulTestWithLayout, AmpereMatmulBatchSplitK) {
  // requires Ampere or higher GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

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
      gemm_tile.instruction_tile = GemmTile(16, 8, 16);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      at::Tensor aten_a =
          matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
      at::Tensor aten_b =
          matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);

      std::vector<c10::IValue> inputs = {aten_a, aten_b};

      FusionExecutor fe;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, fe.compileFusion(&fusion, inputs));
      ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
      auto cg_outputs = fe.runFusion(inputs);
      auto tref =
          atMatmul(aten_a.to(at::kFloat), aten_b.to(at::kFloat), layout);

      // Relax tolerance for larger sum due to large K
      EXPECT_TRUE(cg_outputs[0].allclose(tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Test batch splitk with bias epilogue
TEST_P(MatmulTestWithLayout, AmpereMatmulBatchSplitKBias) {
  // requires Ampere or higher GPU
  if (!deviceMajorMinorCheck(8)) {
    GTEST_SKIP() << "skipping tests on pre-AMPERE GPUs";
  }

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
      gemm_tile.instruction_tile = GemmTile(16, 8, 16);

      MatmulParams mparams;
      mparams.supported_vec_size = {8, 8, 4};
      mparams.mma_macro = MmaMacro::Ampere_16_8_16;
      mparams.tile_sizes = gemm_tile;
      mparams.splitk_factor = splitk_factor;
      mparams.use_smem_epilogue = use_smem_epilogue;
      mparams.promote_prologue_smem_reuse = true;

      SchedulerEntry::makeSchedulerInstance(SchedulerType::Matmul)
          ->schedule(&fusion, &mparams);

      at::Tensor aten_a =
          matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K, B);
      at::Tensor aten_b =
          matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K, B);
      at::Tensor aten_bias = at::randn({M}, aten_a.options());

      std::vector<c10::IValue> inputs = {aten_a, aten_b, aten_bias};

      FusionExecutor fe;
      NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
          7, 5, fe.compileFusion(&fusion, inputs));
      ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
      auto cg_outputs = fe.runFusion(inputs);
      auto tref = atBiasEpilogue(
          atMatmul(aten_a.to(at::kFloat), aten_b.to(at::kFloat), layout),
          aten_bias);

      // Relax tolerance for larger sum due to large K
      EXPECT_TRUE(cg_outputs[0].allclose(tref, 1e-6 * K, 1e-6 * K));
    }
  }
}

// Avoid lowering error https://github.com/NVIDIA/Fuser/issues/1808
TEST_F(MatmulTest, ReproIssue1808) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(
          &fusion,
          {inputs.first, inputs.second},
          LaunchParams(),
          matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// Test matmul with sizes that are not divisible by 8 and with misaligned inputs
TEST_P(MatmulTestWithLayout, MisalignedVectorization) {
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

        std::vector<c10::IValue> inputs = {t0, t1};

        if (add_2d_bias) {
          const auto options =
              at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
          auto bias = maybeUnalign(at::randn({M, N}, options), alignBias);
          tref = tref + bias;
          inputs.push_back(bias);
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

        FusionExecutor fe;
        NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
            8,
            0,
            fe.compileFusion(
                fusion.get(), inputs, LaunchParams(), matmul_cparams));
        ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
        auto outputs = fe.runFusion(inputs);

        EXPECT_TRUE(outputs[0].allclose(tref, 0.001, 0.001));
      }
    }
  }
}

// Matmul test with multiple M and N dimensions that are consecutive
TEST_F(MatmulTest, MultipleConsecutiveDims) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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
  at::Tensor A = at::randn({M1, M2, K}, options);
  at::Tensor B = at::randn({N1, N2, K}, options);
  std::vector<c10::IValue> inputs{A, B};

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, fe.compileFusion(&fusion, inputs, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion(inputs);
  auto tref = at::reshape(
      at::linear(
          at::reshape(A.to(at::kFloat), {M1 * M2, K}),
          at::reshape(B.to(at::kFloat), {N1 * N2, K})),
      {M1, M2, N1, N2});
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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
  at::Tensor A = at::randn({M1, K, M2}, options);
  at::Tensor B = at::randn({N, K}, options);
  std::vector<c10::IValue> inputs{A, B};

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, fe.compileFusion(&fusion, inputs, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion(inputs);
  auto Apermuted = A.permute({{1, 2}}).reshape({M1 * M2, K});
  auto tref = at::linear(Apermuted.to(at::kFloat), B.to(at::kFloat))
                  .reshape({M1, M2, N})
                  .permute({{1, 2}});
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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
  at::Tensor A = at::randn({M, K}, options);
  at::Tensor B = at::randn({N1, K, N2}, options);
  std::vector<c10::IValue> inputs{A, B};

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, fe.compileFusion(&fusion, inputs, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion(inputs);
  auto Bpermuted = B.permute({{1, 2}}).reshape({N1 * N2, K});
  auto tref = at::linear(A.to(at::kFloat), Bpermuted.to(at::kFloat))
                  .reshape({M, N1, N2});
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

// This is a tougher test where we insert a batch dim between the M dims
// The batch dim is parallelized, so M1 and M2 are consecutive in shared
// memory.
TEST_F(MatmulTest, MultipleMDimsBatch) {
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
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

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
  at::Tensor A = at::randn({M1, Batch, M2, K}, options);
  at::Tensor B = at::randn({Batch, N, K}, options);
  std::vector<c10::IValue> inputs{A, B};

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8, 0, fe.compileFusion(&fusion, inputs, LaunchParams(), matmul_cparams));
  ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
  auto cg_outputs = fe.runFusion(inputs);
  auto tref =
      at::matmul(A.to(at::kFloat), at::permute(B.to(at::kFloat), {0, 2, 1}));
  NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
}

#undef NVFUSER_TEST_CUDA_ARCH_GUARD

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulTestWithLayout,
    kAllSupportedMmaLayout,
    mmaLayoutName);

using HopperMatmulTest = HopperBase;

template <typename data_type>
void compare(
    int64_t tensor_outer_dim,
    int64_t tensor_inner_dim,
    at::Tensor result,
    at::Tensor reference) {
  at::Tensor reference_cpu_data = reference.cpu();
  at::Tensor result_cpu_data = result.cpu();

  auto reference_cpu = reference_cpu_data.accessor<data_type, 2>();
  auto result_cpu = result_cpu_data.accessor<data_type, 2>();

  constexpr double tolerance = 1e-3;
  for (int64_t out_pos = 0; out_pos < tensor_outer_dim; ++out_pos) {
    for (int64_t in_pos = 0; in_pos < tensor_inner_dim; ++in_pos) {
      if (fabs(
              (double)reference_cpu[out_pos][in_pos] -
              (double)result_cpu[out_pos][in_pos]) > tolerance) {
        std::cout << "[" << out_pos << ", " << in_pos
                  << "] - result: " << result_cpu[out_pos][in_pos]
                  << " | ref: " << reference_cpu[out_pos][in_pos] << std::endl;
      }
    }
  }
}

TEST_F(HopperMatmulTest, HSH_NT_128BSwizzle) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t M = 1024 * 16, N = 1024 * 16, K = 1024;
  constexpr auto macro = MmaMacro::Hopper_64_256_16;
  constexpr auto layout = MmaLayout::NT; // [K, M] x [K, N] -> [M, N]
  constexpr auto swizzle = MmaInputSmemSwizzle::B128;
  const auto dtype = DataType::Half;

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

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_ops.front()->setMacro(macro);

  // gmem [K, M, 1] x gmem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> gmem [M, N]

  auto tv0c = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0c->setMemoryType(MemoryType::Shared);
  auto tv1c = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv1c->setMemoryType(MemoryType::Shared);
  auto tv3c = tv3->cacheBefore();

  // gmem [K, M, 1] -TMA-> smem [K, M, 1]
  // gmem [K, 1, N] -TMA-> smem [K, 1, N]
  // smem [K, M, 1] x smem [K, 1, N] -mma-> register [M, N, rK]
  // register [M, N, rK] -cast-> register [M, N] -set-> gmem [M, N]

  // Create tiles
  tv2->split(-3, getM(macro));
  tv2->split(-2, getN(macro));
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

  tv0c->circularBuffer(/*number_of_stages=*/4);
  tv1c->circularBuffer(/*number_of_stages=*/4);

  auto inputs =
      matmulAtInput3DHopperSS(M, N, K, layout, data_type_to_aten(dtype));

  FusionExecutor fe;
  fe.compileFusion(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
  auto tref = atMatmul(inputs.first.squeeze(), inputs.second.squeeze(), layout);
  compare(M, N, cg_outputs[0], tref);
  EXPECT_TRUE(at::allclose(cg_outputs[0], tref, 1e-5, 1e-5));
}

} // namespace nvfuser
