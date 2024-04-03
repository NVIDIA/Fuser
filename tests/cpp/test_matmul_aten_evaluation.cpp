// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/mma_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class MatmulATenEvaluationTest : public NVFuserTest {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  MatmulATenEvaluationTest() : optimization_guard_(false) {}

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
};

TEST_F(MatmulATenEvaluationTest, MmaOpAndCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::matmul(t0, t1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  // Verify that the io_alias_ set has the correct entry
  kir::Kernel* kernel = executors.front().kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_F(MatmulATenEvaluationTest, TransposeMmaOpAndCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{n, k}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv1_t = transpose(tv1, 0, 1);
  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1_t, {true, false, false}); // [1, K, N]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::matmul(t0, t1.transpose(0, 1));

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

  scheduleMatmul(fusion.get(), params);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {t0, t1}, LaunchParams(), matmul_cparams);
  auto out = fe.runFusion({t0, t1});
  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_F(MatmulATenEvaluationTest, MulSumAndCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv2 = sum(mul(tv0b, tv1b), {1});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::matmul(t0, t1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  // Verify that the io_alias_ set has the correct entry
  kir::Kernel* kernel = executors.front().kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

// Disabled until at::addmm support is add.
// See https://github.com/NVIDIA/Fuser/pull/1874#discussion_r1516991574
TEST_F(MatmulATenEvaluationTest, MatmulWithBias) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});
  auto tv3 = makeConcreteTensor({m}, DataType::Half);
  auto tv4 = castOp(DataType::Float, tv3);
  auto tv5 = biasEpilogue(tv2, tv4);
  auto tv6 = castOp(DataType::Half, tv5);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv3);
  fusion->addOutput(tv6);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn({m}, at::kHalf).cuda();
  at::Tensor out_ref = at::addmm(t2.unsqueeze(-1), t0, t1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1, t2});

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  // Verify that the io_alias_ set has the correct entry
  kir::Kernel* kernel = executors.front().kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

} // namespace nvfuser
