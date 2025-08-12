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
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <logical_domain_map.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <options.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
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
    // we are skipping Blackwell since the matmul for it is under development.
    auto lower_major = 8;
    auto lower_minor = 0;
    auto upper_major = 10;
    auto upper_minor = 0;
    if (cudaArchGuardShouldSkip(
            lower_major, lower_minor, upper_major, upper_minor)) {
      GTEST_SKIP() << "CombineMulSumAsMmaTest skipped "
                   << "Requires GPU capability between  " << lower_major << "."
                   << lower_minor << "and " << upper_major << "." << upper_minor
                   << " to run.\n";
    }

    pre_hopper = at::cuda::getCurrentDeviceProperties()->major < 9;

    NVFuserTest::SetUp();
  }

 protected:
  bool pre_hopper;
};

class CombineMulSumAsMmaTestWithLayout
    : public NVFuserTest,
      public ::testing::WithParamInterface<MmaLayout> {
 protected:
  MmaLayout layout;
  void SetUp() override {
    layout = GetParam();
    // These test are enable for Turing and newer.
    // we are skipping Blackwell since the matmul for it is under development.
    auto lower_major = 8;
    auto lower_minor = 0;
    auto upper_major = 10;
    auto upper_minor = 0;
    if (cudaArchGuardShouldSkip(
            lower_major, lower_minor, upper_major, upper_minor)) {
      GTEST_SKIP() << "CombineMulSumAsMmaTestWithLayout skipped "
                   << "Requires GPU capability between  " << lower_major << "."
                   << lower_minor << "and " << upper_major << "." << upper_minor
                   << " to run.\n";
    }
    pre_hopper = at::cuda::getCurrentDeviceProperties()->major < 9;
    NVFuserTest::SetUp();
  }

  bool pre_hopper;
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
TEST_P(CombineMulSumAsMmaTestWithLayout, MulSumToMatmul_Pass) {
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

// This test checks that the pattern matcher does not incorrectly identify
// this mul-sum pair, as the mul is not fed by broadcasts ops; i.e. it is
// not a matmul.
TEST_F(CombineMulSumAsMmaTest, MulSumToMatmul_Fail1) {
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
TEST_F(CombineMulSumAsMmaTest, MulSumToMatmul_MultipleBroadcasts) {
  // This test expicitly broadcasts and transposes, so we cannot avoid
  // intermediates on Hopper (yet).
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
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
TEST_P(CombineMulSumAsMmaTestWithLayout, AmpereMulSumToMatmul_Schedule) {
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
  auto tv2 = sum(mul(tv0, tv1), {-1});

  fusion.addOutput(tv2);

  performSubstitution(&fusion);

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
  ke.compile(
      &fusion, {inputs.first, inputs.second}, LaunchParams(), matmul_cparams);
  ASSERT_TRUE(getBankConflictInfo(ke.compiledKernel()->kernel()).empty());
  auto cg_outputs = ke.run({inputs.first, inputs.second});
  auto tref = atMatmul(
      inputs.first.to(at::kFloat), inputs.second.to(at::kFloat), layout);
  NVF_CHECK(at::allclose(cg_outputs[0].as<at::Tensor>(), tref, 0.0001, 0.0001));
}

TEST_P(CombineMulSumAsMmaTestWithLayout, UseMatmulScheduler) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  // Keep multiples of 8 to keep vectorizable.
  int M = 504, N = 136, K = 248;
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
  tv2->setAllocationDomain(tv2->getLogicalDomain(), true);

  fusion->addOutput(tv2);
  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(fusion.get()).empty());

  auto t0 = matmulAtInput2D(layout, TensorMatmulPos::A, at::kHalf, M, N, K);
  auto t1 = matmulAtInput2D(layout, TensorMatmulPos::B, at::kHalf, M, N, K);
  auto tref = atMatmul(t0, t1, layout);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  // Ensure there's a mma op.
  // If there's no mma op present, then stop the test.
  const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
  ASSERT_FALSE(
      ir_utils::getOpsOfType<MmaOp>(ke->compiledKernel()->kernel()).empty());

  // Ensure that the matmul scheduler ran.
  EXPECT_TRUE(
      executor_cache.getMostRecentKernelRuntime()
          ->schedulerHeuristics()
          ->heuristicsList()
          .at(0)
          ->scheduler_type == SchedulerType::Matmul);

  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// Parameters: [A_dim, B_dim, enable_fusion, transpose_a_alloc,
// expect_segmented, SchedulerType]
using MatmulNodeTranslationTestParams =
    std::tuple<int64_t, int64_t, bool, bool, bool, SchedulerType>;
using MatmulNodeTranslationTest =
    NVFuserFixtureParamTest<MatmulNodeTranslationTestParams>;

// Test that a simple matmul op fusion is picked up by the appropriate scheduler
// and the translation to MmaOp is performed properly.
TEST_P(MatmulNodeTranslationTest, AutomaticSchedulerMatmulNode) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 10, 0);
  const int64_t A_dim = std::get<0>(GetParam());
  const int64_t B_dim = std::get<1>(GetParam());
  const bool enable_fusion = std::get<2>(GetParam());
  const bool transpose_a_alloc = std::get<3>(GetParam());
  const bool expect_segmented = std::get<4>(GetParam());
  const SchedulerType expected_heuristic = std::get<5>(GetParam());

  if (A_dim == 3 && B_dim == 2) {
    // TODO: Fix the failure at checkConcreteStaticDim on Hopper in this case
    NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 9, 0);
  }

  // CombineMulSumAsMmaTest disabled MatmulExprEval, but we need it enabled
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().unset(DisableOption::MatmulExprEval);

  EnableOptionsGuard eog;
  if (enable_fusion) {
    EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMatmul);
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::FuseMatmul);
  }

  // The allocation domain propagation pass sets the output allocation domain,
  // which sometimes causes the matmul scheduler to decline the whole fusion
  // when it could compile it otherwise.
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      alloc_pass_guard(false);

  int batch_size = 3, M = 504, N = 136, K = 248;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(A_dim, DataType::Half);
  auto tv1 = makeContigTensor(B_dim, DataType::Half);

  if (transpose_a_alloc && A_dim > 1) {
    std::vector<IterDomain*> alloc = tv0->getMaybeAllocationDomain();
    alloc[alloc.size() - 1] = tv0->axis(-2);
    alloc[alloc.size() - 2] = tv0->axis(-1);
    tv0->setAllocationDomain(alloc, true);
  }

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = matmul(tv0, tv1);

  // add an epilogue
  auto tv3 = sin(tv2);
  auto tv4 = castOp(DataType::Half, tv3);

  fusion->addOutput(tv4);

  // Verify that we no longer set up MmaOp in matmul()
  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(fusion.get()).empty());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<int64_t> A_shape(A_dim, batch_size);
  A_shape[A_dim - 1] = K;
  if (A_dim > 1) {
    A_shape[A_dim - 2] = M;
  }
  at::Tensor t0 = at::randn(A_shape, options);
  std::vector<int64_t> B_shape(B_dim, batch_size);
  if (B_dim > 1) {
    B_shape[B_dim - 2] = K;
    B_shape[B_dim - 1] = N;
  } else {
    B_shape[B_dim - 1] = K;
  }
  auto t1 = at::randn(B_shape, options);
  if (transpose_a_alloc) {
    t0 = t0.as_strided({M, K}, {1, M});
  }
  auto tref = at::matmul(t0, t1).sin().to(at::kHalf);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});

  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();
  ASSERT_NE(runtime, nullptr);

  if (expect_segmented) {
    EXPECT_TRUE(runtime->isSegmented());
  } else {
    EXPECT_FALSE(runtime->isSegmented());
  }

  SchedulerType scheduler_type =
      runtime->schedulerHeuristics()->heuristicsList().front()->scheduler_type;
  EXPECT_EQ(scheduler_type, expected_heuristic);

  if (scheduler_type == SchedulerType::Matmul) {
    // Ensure there's an MmaOp.

    const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
    ASSERT_FALSE(
        ir_utils::getOpsOfType<MmaOp>(ke->compiledKernel()->kernel()).empty());
  }

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulNodeTranslationTest,
    ::testing::Values(
        // Tests without fusion enabled
        std::make_tuple(2l, 2l, false, false, true, SchedulerType::ExprEval),
        std::make_tuple(2l, 2l, false, true, true, SchedulerType::ExprEval),

        // Tests with fusion enabled

        std::make_tuple(2l, 2l, true, false, false, SchedulerType::Matmul),
        std::make_tuple(2l, 2l, true, true, false, SchedulerType::Matmul),
        // Size-1 input combinations
        std::make_tuple(1l, 2l, true, false, true, SchedulerType::ExprEval),
        std::make_tuple(2l, 1l, true, false, true, SchedulerType::ExprEval),
        std::make_tuple(1l, 1l, true, false, true, SchedulerType::ExprEval),
        // Batch dims

        // mat-vec handled by ExprEval
        std::make_tuple(3l, 1l, true, false, true, SchedulerType::ExprEval),
        std::make_tuple(3l, 3l, true, false, false, SchedulerType::Matmul),

        std::make_tuple(3l, 2l, true, false, false, SchedulerType::Matmul),
        std::make_tuple(4l, 4l, true, false, false, SchedulerType::Matmul),

        // TODO: mixed length inputs via broadcasted batch dims
        // When different numbers of M or N dimensions exist, they must be
        // consecutive. However, these examples lead to [M, B, M, K] and [N, B,
        // N, K] patterns which we don't yet support.
        std::make_tuple(2l, 3l, true, false, true, SchedulerType::ExprEval),
        std::make_tuple(3l, 4l, true, false, true, SchedulerType::ExprEval)),

    [](const testing::TestParamInfo<MatmulNodeTranslationTestParams>& info) {
      std::ostringstream os;
      os << std::get<0>(info.param) << "dA";
      os << "_" << std::get<1>(info.param) << "dB";
      if (!std::get<2>(info.param)) {
        os << "_nofuse";
      }
      if (std::get<3>(info.param)) {
        os << "_transposeA";
      }
      return os.str();
    });

using LinearNodeTranslationTestParams =
    std::tuple<int64_t, int64_t, int64_t, bool, bool, bool>;
using LinearNodeTranslationTest =
    NVFuserFixtureParamTest<LinearNodeTranslationTestParams>;

// Test that a simple linear op fusion is picked up by the appropriate scheduler
// and the translation to MmaOp is performed properly.
TEST_P(LinearNodeTranslationTest, AutomaticSchedulerLinearNode) {
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(7, 5, 10, 0);
  // The allocation domain propagation pass sets the output allocation domain,
  // which sometimes causes the matmul scheduler to decline the whole fusion
  // when it could compile it otherwise.
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      alloc_pass_guard(false);
  const int64_t A_dim = std::get<0>(GetParam());
  const int64_t B_dim = std::get<1>(GetParam());
  const int64_t bias_dim = std::get<2>(GetParam());
  const bool enable_fusion = std::get<3>(GetParam());
  const bool transpose_a_alloc = std::get<4>(GetParam());
  const bool expect_aten_eval = std::get<5>(GetParam());

  // CombineMulSumAsMmaTest disabled MatmulExprEval, but we need it
  // enabled
  DisableOptionsGuard dog;
  DisableOptionsGuard::getCurOptions().unset(DisableOption::MatmulExprEval);

  EnableOptionsGuard eog;
  if (enable_fusion) {
    if (A_dim != 2 && !cudaArchGuardShouldSkip(9, 0)) {
      GTEST_SKIP() << "Translating linear with batch dims is not yet supported "
                      "on Hopper";
    }

    EnableOptionsGuard::getCurOptions().set(EnableOption::FuseMatmul);
  } else {
    EnableOptionsGuard::getCurOptions().unset(EnableOption::FuseMatmul);
  }

  int batch_size = 3, M = 504, N = 136, K = 248;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(A_dim, DataType::Half);
  auto tv1 = makeContigTensor(B_dim, DataType::Half);

  if (transpose_a_alloc && A_dim > 1) {
    std::vector<IterDomain*> alloc = tv0->getMaybeAllocationDomain();
    alloc[alloc.size() - 1] = tv0->axis(-2);
    alloc[alloc.size() - 2] = tv0->axis(-1);
    tv0->setAllocationDomain(alloc, true);
  }

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  TensorView* tv2 = nullptr;
  if (bias_dim >= 0) {
    // bias_dim = -1 indicates we should not use any bias argument
    auto bias = makeContigTensor(bias_dim, DataType::Half);
    fusion->addInput(bias);
    tv2 = linear(tv0, tv1, bias);
  } else {
    tv2 = linear(tv0, tv1);
  }

  // add an epilogue
  auto tv3 = sin(tv2);
  auto tv4 = castOp(DataType::Half, tv3);

  fusion->addOutput(tv4);

  // Verify that we no longer set up MmaOp in matmul()
  ASSERT_TRUE(ir_utils::getOpsOfType<MmaOp>(fusion.get()).empty());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<int64_t> A_shape(A_dim, batch_size);
  A_shape[A_dim - 1] = K;
  if (A_dim > 1) {
    A_shape[A_dim - 2] = M;
  }
  at::Tensor t0 = at::randn(A_shape, options);
  std::vector<int64_t> B_shape(B_dim, batch_size);
  B_shape[B_dim - 1] = K;
  if (B_dim > 1) {
    B_shape[B_dim - 2] = N;
  }
  auto t1 = at::randn(B_shape, options);
  if (transpose_a_alloc) {
    t0 = t0.as_strided({M, K}, {1, M});
  }
  KernelArgumentHolder inputs{t0, t1};
  at::Tensor tref;
  if (bias_dim >= 0) {
    at::Tensor bias;
    if (bias_dim == 0) {
      bias = at::randn({}, options);
    } else if (bias_dim == 1) {
      bias = at::randn({N}, options);
    } else {
      NVF_THROW("Invalid bias dimension given:", bias_dim);
    }
    inputs.push(bias);
    tref = at::linear(t0, t1, bias);
  } else {
    tref = at::linear(t0, t1);
  }
  tref = tref.sin().to(at::kHalf);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

  const FusionKernelRuntime* runtime =
      executor_cache.getMostRecentKernelRuntime();
  ASSERT_NE(runtime, nullptr);

  if (expect_aten_eval) {
    EXPECT_TRUE(runtime->isSegmented());
  } else {
    EXPECT_FALSE(runtime->isSegmented());
  }

  SchedulerType scheduler_type =
      runtime->schedulerHeuristics()->heuristicsList().front()->scheduler_type;
  if (expect_aten_eval) {
    EXPECT_EQ(scheduler_type, SchedulerType::ExprEval);
  } else {
    // Ensure that the Matmul scheduler ran.
    // Assert here since we will inspect the kernel next, which we can't
    // do if ExprEval accepts the segment.
    ASSERT_EQ(scheduler_type, SchedulerType::Matmul);
    // Ensure there's an MmaOp.
    const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
    ASSERT_FALSE(
        ir_utils::getOpsOfType<MmaOp>(ke->compiledKernel()->kernel()).empty());
  }

  testValidate(
      executor_cache.fusion(), outputs, inputs, {tref}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    LinearNodeTranslationTest,
    ::testing::Values(
        // Tests without fusion enabled
        std::make_tuple(2l, 2l, -1l, false, false, true),
        std::make_tuple(2l, 2l, -1l, false, true, true),
        std::make_tuple(1l, 2l, -1l, false, false, true),
        std::make_tuple(2l, 2l, 1l, false, false, true),
        std::make_tuple(3l, 2l, 1l, false, false, true),
        std::make_tuple(4l, 2l, 1l, false, false, true),

        // Enable fusion

        std::make_tuple(2l, 2l, -1l, true, false, false),
        std::make_tuple(2l, 2l, -1l, true, true, false),
        // We don't fuse 1D inputs
        std::make_tuple(1l, 2l, -1l, true, false, true),
        // Batch dims in input
        // mixed length inputs via broadcasted batch dims
        std::make_tuple(3l, 2l, -1l, true, false, false),
        std::make_tuple(4l, 2l, -1l, true, false, false),
        // Bias cases
        std::make_tuple(2l, 2l, 1l, true, false, false),
        std::make_tuple(3l, 2l, 1l, true, false, false),
        std::make_tuple(4l, 2l, 1l, true, false, false)),
    [](const testing::TestParamInfo<LinearNodeTranslationTestParams>& info) {
      std::ostringstream os;
      os << std::get<0>(info.param) << "dA";
      os << "_" << std::get<1>(info.param) << "dB";
      int64_t bias_dim = std::get<2>(info.param);
      if (bias_dim >= 0) {
        os << "_" << bias_dim << "dBias";
      }
      if (!std::get<3>(info.param)) {
        os << "_nofuse";
      }
      if (std::get<4>(info.param)) {
        os << "_transposeA";
      }
      return os.str();
    });

// Check that we determine A and B properly when they are swapped as inputs to
// mul
TEST_P(CombineMulSumAsMmaTestWithLayout, SwapAandB) {
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
    mma_utils::DimRolesMap dim_roles = pattern.getDimRoles(id_model);
    ValGraph& graph = id_model.idGraph(IdMappingMode::BROADCAST);
    const ValGroup& m_gp = graph.toGroup(tv0->axis(-3));
    auto m_it = dim_roles.find(m_gp);
    ASSERT_NE(m_it, dim_roles.end());
    EXPECT_EQ(m_it->second, MatmulDimRole::M);

    const ValGroup& n_gp = graph.toGroup(tv1->axis(-2));
    auto n_it = dim_roles.find(n_gp);
    ASSERT_NE(n_it, dim_roles.end());
    EXPECT_EQ(n_it->second, MatmulDimRole::N);

    ASSERT_FALSE(ir_utils::getOpsOfType<MmaOp>(&fusion).empty());
  }
}

// Check that a fusion with epilogue does not introduce round-trip casts when
// translating MatmulOp to MmaOp
using TranslationCastTestParams = std::tuple<bool, bool, bool>;
using TranslationCastTest = NVFuserFixtureParamTest<TranslationCastTestParams>;
TEST_P(TranslationCastTest, CountCasts) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  bool use_linear = std::get<0>(GetParam());
  bool sin_epilogue = std::get<1>(GetParam());
  bool output_pre_epilogue = std::get<2>(GetParam());

  auto tv0 = makeContigTensor(2, DataType::Half);
  auto tv1 = makeContigTensor(2, DataType::Half);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = use_linear ? linear(tv0, tv1) : matmul(tv0, tv1);

  if (output_pre_epilogue) {
    // Add the Half output before epilogue. If this is true and we have an
    // epilogue, then the MmaOp will have two uses: sin and cast.
    fusion->addOutput(tv2);
  }

  TensorView* tv3 = tv2;
  if (sin_epilogue) {
    tv3 = sin(tv2);
  }
  auto tv4 = castOp(DataType::Half, tv3);

  fusion->addOutput(tv4);

  std::vector<mma_utils::MatmulPattern> patterns =
      mma_utils::findMatmulPatterns(fusion.get());

  ASSERT_EQ(patterns.size(), 1);

  mma_utils::MatmulPattern& pattern = patterns.front();

  EXPECT_EQ(pattern.A, tv0);
  EXPECT_EQ(pattern.B, tv1);
  EXPECT_EQ(pattern.output, tv2);

  pattern.translateToMmaOp();

  // Check that we modified the pattern roles
  EXPECT_NE(pattern.A, tv0);
  EXPECT_NE(pattern.B, tv1);
  EXPECT_EQ(
      pattern.output->dtype(), sin_epilogue ? DataType::Float : DataType::Half);

  // Count cast ops. In any case there should be only a single cast, at the end
  // of the fusion.
  const auto exprs = fusion->exprs();
  size_t num_casts = std::count_if(exprs.begin(), exprs.end(), [](Expr* e) {
    if (auto* uop = dynamic_cast<UnaryOp*>(e)) {
      return uop->getUnaryOpType() == UnaryOpType::Cast;
    }
    return false;
  });
  if (sin_epilogue && output_pre_epilogue) {
    // Fusion looks like
    // Inputs:
    //   A
    //   B
    // Outputs:
    //   C
    //   D
    // C = linear(A, B)
    // D = sin(C)
    // We will need to cast both C and D.
    EXPECT_EQ(num_casts, 2);
  } else {
    // Fusion is either
    //   C = linear(A, B)
    // or
    //   C = linear(A, B)
    //   D = sin(C)
    // but we are not outputting both C and D so there is only one cast.
    EXPECT_EQ(num_casts, 1);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TranslationCastTest,
    testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<TranslationCastTestParams>& info) {
      std::ostringstream os;
      os << (std::get<0>(info.param) ? "linear" : "matmul");
      os << (std::get<1>(info.param) ? "_epilogue" : "");
      os << (std::get<2>(info.param) ? "_twooutputs" : "");
      return os.str();
    });

INSTANTIATE_TEST_SUITE_P(
    ,
    CombineMulSumAsMmaTestWithLayout,
    testing::ValuesIn(kAllSupportedMmaLayout),
    mmaLayoutName);

} // namespace nvfuser
