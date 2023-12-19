/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <macros.h>

#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/analysis/bank_conflict.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class CombineMulSumAsMmaTest : public NVFuserTest {};

// Test checks to see that the combiner can correctly replace
// the mul-sum pair with a mma op.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_Pass) {
  // Assumes layout is kAllSupportedMmaLayout::NT;
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv0t = transpose(tv0, 0, 1);
  auto tv1t = transpose(tv1, 0, 1);

  std::vector<bool> bcast_dims(tv0->nDims() + 1, false);
  bcast_dims.at(bcast_dims.size() - 2) = true;
  auto tv0b = broadcast(tv0t, bcast_dims);
  bcast_dims.at(bcast_dims.size() - 2) = false;
  bcast_dims.at(bcast_dims.size() - 3) = true;
  auto tv1b = broadcast(tv1t, bcast_dims);
  auto tv2 = mul(tv0b, tv1b);
  auto tv3 = sum(tv2, {-1});
  fusion.addOutput(tv3);

  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());

  nvfuser::mma_utils::CombineMulSum combiner(&fusion);
  combiner.replaceWithMmaOp();

  ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());
}

// This test checks that the combiner does not incorrectly
// replace this mul-sum pair, and the mul is not fed by broadcasts ops.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_Fail1) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(3, DataType::Half);
  auto tv1 = makeContigTensor(3, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = mul(tv0, tv1);
  auto tv3 = sum(tv2, {-1});
  fusion.addOutput(tv3);

  nvfuser::mma_utils::CombineMulSum combiner(&fusion);
  combiner.replaceWithMmaOp();

  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());
}

// This test checks to see that the mul-sum combiner does not
// combine a mul-sum which does not have appropriate broadcasts.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_Fail2) {
  // Assumes layout is kAllSupportedMmaLayout::NT;
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv0t = transpose(tv0, 0, 1);
  auto tv1t = transpose(tv1, 0, 1);

  // We are broadcating to a tensor that will have too many dims
  // to be valid for a mma op.
  std::vector<bool> bcast_dims(tv0->nDims() + 2, false);
  bcast_dims.at(bcast_dims.size() - 2) = true;
  bcast_dims.at(bcast_dims.size() - 3) = true;
  auto tv0b = broadcast(tv0t, bcast_dims);
  bcast_dims.at(bcast_dims.size() - 2) = false;
  bcast_dims.at(bcast_dims.size() - 3) = true;
  bcast_dims.at(bcast_dims.size() - 4) = true;
  auto tv1b = broadcast(tv1t, bcast_dims);
  auto tv2 = mul(tv0b, tv1b);
  auto tv3 = sum(tv2, {-1});
  fusion.addOutput(tv3);

  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());

  nvfuser::mma_utils::CombineMulSum combiner(&fusion);
  combiner.replaceWithMmaOp();

  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());
}

// As a sanity check we test that after replacing a mul-sum
// pair with a mma op, we are able to schedule it as we did with
// a fusion that had a mma op to begin with.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_Schedule) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;

  for (auto layout : kAllSupportedMmaLayout) {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto tv0 = makeContigTensor(2, DataType::Half);
    auto tv1 = makeContigTensor(2, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    auto tv2 = matmul(tv0, tv1, layout, true, true);
    fusion.addOutput(tv2);
    ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());

    nvfuser::mma_utils::CombineMulSum combiner(&fusion);
    combiner.replaceWithMmaOp();
    ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

    MatmulParams params;
    params.mma_macro = MmaMacro::Ampere_16_8_16;
    params.tile_sizes = gemm_tile;
    params.async_gmem_load_operands = true;
    params.double_buffer_options.double_buffer_smem_write = true;
    params.double_buffer_options.double_buffer_smem_read = true;
    params.double_buffer_options.smem_double_buffer_stage = 4;
    scheduleMatmul(&fusion, params);

    auto inputs = matmulAtInput(M, N, K, layout);

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

} // namespace nvfuser
