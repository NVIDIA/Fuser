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
#include <options.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/matmul.h>
#include <scheduler/mma_utils.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class CombineMulSumAsMmaTest : public NVFuserTest {
  void SetUp() override {
    // These test are enable for Turing and newer. Temporarily
    // we are skipping Hopper since the matmul for it is under development.
    auto lower_major = 8;
    auto lower_minor = 0;
    auto upper_major = 9;
    auto upper_minor = 0;
    if (cudaArchGuardShouldSkip(
            lower_major, lower_minor, upper_major, upper_minor)) {
      GTEST_SKIP() << "CombineMulSumAsMmaTest skipped "
                   << "Requires GPU capability between  " << lower_major << "."
                   << lower_minor << "and " << upper_major << "." << upper_minor
                   << " to run.\n";
    }
    NVFuserTest::SetUp();
  }
};

void performSubstitution(Fusion* fusion, bool should_not_find = false) {
  EXPECT_TRUE(ir_utils::getOpsOfType<MmaOp>(fusion).empty());

  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion);
  if (should_not_find) {
    EXPECT_TRUE(patterns.empty());
    return;
  }

  ASSERT_FALSE(patterns.empty());
  EXPECT_EQ(patterns.size(), 1);

  patterns.front().translateToMmaOp();

  ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(fusion).empty());
}

// Test checks to see that the combiner can correctly replace
// the mul-sum pair with a mma op.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_Pass) {
  for (auto layout : kAllSupportedMmaLayout) {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto tv0 = makeContigTensor(2, DataType::Half);
    auto tv1 = makeContigTensor(2, DataType::Half);

    fusion.addInput(tv0);
    fusion.addInput(tv1);

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = mul(tv0, tv1);
    auto tv3 = sum(tv2, {-1});

    fusion.addOutput(tv3);

    performSubstitution(&fusion);
  }
}

// This test checks that the pattern matcher does not incorrectly identify
// this mul-sum pair, as the mul is not fed by broadcasts ops; i.e. it is
// not a matmul.
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

  performSubstitution(&fusion, /*should_not_find=*/true);
}

// This fusion has Broadcast batch axes in each operand.
TEST_F(CombineMulSumAsMmaTest, AmpereMulSumToMatmul_MultipleBroadcasts) {
  // Assumes layout is kAllSupportedMmaLayout::NT;
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

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
  fusion->addOutput(tv3);

  performSubstitution(fusion, /*should_not_find=*/false);

  // We test running this fusion also to verify that the broadcast batch
  // dimension does not cause unforeseen issues

  int64_t M = 256, N = 128, K = 64;
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({K, M}, options);
  auto t1 = at::randn({K, N}, options);
  auto tref = at::linear(t0.t(), t1.t()).unsqueeze(1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
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

    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = sum(mul(tv0, tv1), {-1});

    fusion.addOutput(tv2);

    performSubstitution(&fusion);

    MatMulTileOptions gemm_tile;
    gemm_tile.cta_tile = GemmTile(128, 128, 32);
    gemm_tile.warp_tile = GemmTile(64, 64, 32);
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

    MatmulParams params;
    params.supported_vec_size = {8, 8, 4};
    params.mma_macro = MmaMacro::Ampere_16_8_16;
    params.tile_sizes = gemm_tile;
    params.async_gmem_load_operands = true;
    params.double_buffer_options.double_buffer_smem_write = true;
    params.double_buffer_options.double_buffer_smem_read = true;
    params.double_buffer_options.smem_double_buffer_stage = 4;
    scheduleMatmul(&fusion, params);

    auto inputs = matmulAtInput2D(M, N, K, layout);

    FusionExecutor fe;
    fe.compileFusion(
        &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
    ASSERT_TRUE(getBankConflictInfo(fe.kernel()).empty());
    auto cg_outputs = fe.runFusion({inputs.first, inputs.second});
    auto tref = atMatmul(
        inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
    NVF_CHECK(cg_outputs[0].allclose(tref, 0.0001, 0.0001));
  }
}

TEST_F(CombineMulSumAsMmaTest, UseMatmulScheduler) {
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
  for (auto layout : kAllSupportedMmaLayout) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0 = makeContigTensor(2, DataType::Half);
    auto tv1 = makeContigTensor(2, DataType::Half);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
    tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
    auto tv2 = sum(mul(tv0, tv1), {-1});
    // setting output alloc_domain to avoid allocation order propagation, which
    // breaks the assumption of matmul scheduler. see issue:
    // https://github.com/NVIDIA/Fuser/issues/2014
    tv2->setAllocationDomain(tv2->getMaybeRFactorDomain(), true);

    fusion->addOutput(tv2);
    ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(fusion.get()).empty());

    auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
    auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
    auto tref = atMatmul(t0, t1, layout);

    FusionExecutorCache executor_cache(std::move(fusion));
    auto outputs = executor_cache.runFusionWithInputs({t0, t1});
    // Ensure there's a mma op.
    // If there's no mma op present, then stop the test.
    ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(
                     executor_cache.getMostRecentKernelRuntime()
                         ->executors()
                         .at(0)
                         .kernel())
                     .empty());
    // Ensure that the matmul scheduler ran.
    EXPECT_TRUE(
        dynamic_cast<MatmulScheduler*>(
            executor_cache.getMostRecentKernelRuntime()
                ->schedulerHeuristics()
                ->heuristicsList()
                .at(0)
                .get()) != nullptr);

    EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

    testValidate(
        executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
  }
}

// Check that we determine A and B properly when they are swapped as inputs to
// mul
TEST_F(CombineMulSumAsMmaTest, SwapAandB) {
  for (auto layout : kAllSupportedMmaLayout) {
    for (bool swap : {false, true}) {
      Fusion fusion;
      FusionGuard fg(&fusion);
      auto tv0 = makeContigTensor(2, DataType::Half);
      auto tv1 = makeContigTensor(2, DataType::Half);

      fusion.addInput(tv0);
      fusion.addInput(tv1);

      tv0 = canonicalizeInputToBMNK(tv0, layout, MmaOperand::A);
      tv1 = canonicalizeInputToBMNK(tv1, layout, MmaOperand::B);
      // We should identify tv0 as A and tv1 as B regardless of the order here
      auto tv2 = swap ? mul(tv1, tv0) : mul(tv0, tv1);
      auto tv3 = sum(tv2, {-1});

      fusion.addOutput(tv3);

      std::vector<mma_utils::MatmulPattern> patterns =
          mma_utils::findMatmulPatterns(&fusion);

      ASSERT_FALSE(patterns.empty());
      EXPECT_EQ(patterns.size(), 1);

      mma_utils::MatmulPattern& pattern = patterns.front();

      EXPECT_EQ(pattern.A, tv0);
      EXPECT_EQ(pattern.B, tv1);
      EXPECT_EQ(pattern.output, tv3);

      pattern.translateToMmaOp();

      // Check that we didn't modify the pattern roles
      EXPECT_EQ(pattern.A, tv0);
      EXPECT_EQ(pattern.B, tv1);
      EXPECT_EQ(pattern.output, tv3);

      // Check that we properly map M and N to their roles even with swap
      IdModel id_model(&fusion);
      std::unordered_map<ValGroup, MatmulDomain> dim_roles =
          pattern.getDimRoles(id_model);
      ValGraph& exact_graph = id_model.idGraph(IdMappingMode::EXACT);
      const ValGroup& m_gp = exact_graph.toGroup(tv0->axis(-3));
      auto m_it = dim_roles.find(m_gp);
      ASSERT_NE(m_it, dim_roles.end());
      EXPECT_EQ(m_it->second, MatmulDomain::M);

      const ValGroup& n_gp = exact_graph.toGroup(tv1->axis(-2));
      auto n_it = dim_roles.find(n_gp);
      ASSERT_NE(n_it, dim_roles.end());
      EXPECT_EQ(n_it->second, MatmulDomain::N);

      ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());
    }
  }
}

} // namespace nvfuser
