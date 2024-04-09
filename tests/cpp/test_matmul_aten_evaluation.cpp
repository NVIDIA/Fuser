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

// fd.ops.matmul (a, b) where a = [M,K], b = [K,N]
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

// fd.ops.matmul (a, b.t()) where a = [M,K], b = [N,K]
TEST_F(MatmulATenEvaluationTest, TransposeMmaOpAndCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{n, k};

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

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  ASSERT_EQ(executors.size(), 1);
  // Verify that the io_alias_ set has the correct entry
  kir::Kernel* kernel = executors.front().kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

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

// addmm: a [M,K] x b [K,N] + bias [M,1]
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

// addmm: alpha * (a [M,K] x b [K,N]) + beta * bias [M,N]
TEST_F(MatmulATenEvaluationTest, MatmulBiasAlphaBeta) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto s0 = IrBuilder::create<Val>(DataType::Float); // alpha
  auto s1 = IrBuilder::create<Val>(DataType::Float); // beta

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv2 = makeConcreteTensor(out_shape, DataType::Half);

  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv3 = fusedMultiplySum(tv0b, tv1b, {1});
  auto tv4 = mul(s0, tv3); // alpha * (A x B)
  auto tv5 = mul(s1, tv2); // beta * bias
  auto tv6 = add(tv4, tv5);
  auto tv7 = castOp(DataType::Half, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addInput(s1);
  fusion->addOutput(tv7);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn(out_shape, at::kHalf).cuda();
  float alpha = 2.5;
  float beta = 1.5;
  at::Tensor out_ref = at::addmm(t2, t0, t1, beta, alpha);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1, t2, alpha, beta});

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

// addmm: a [M,K] x b [K,N] + beta * bias [1,N]
TEST_F(MatmulATenEvaluationTest, MatmulBiasBeta) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto s1 = IrBuilder::create<Val>(DataType::Float); // beta

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv2 = makeConcreteTensor({n}, DataType::Half);

  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv3 = fusedMultiplySum(tv0b, tv1b, {1});

  auto tv4 = broadcast(tv2, {true, false});
  auto tv5 = mul(tv4, s1); // bias * beta

  auto tv6 = add(tv3, tv5);

  auto tv7 = castOp(DataType::Half, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s1);
  fusion->addOutput(tv7);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn({n}, at::kHalf).cuda();
  float beta = 1.5;
  at::Tensor out_ref = at::addmm(t2.unsqueeze(0), t0, t1, beta);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1, t2, beta});

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

// addmm: alpha * (a [M,K] x b [K,N]) + bias [M,N]
TEST_F(MatmulATenEvaluationTest, MatmulBiasAlpha) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto s0 = IrBuilder::create<Val>(DataType::Float); // alpha
  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv2 = makeConcreteTensor(out_shape, DataType::Half);

  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv3 = fusedMultiplySum(tv0b, tv1b, {1});
  auto tv4 = mul(tv3, s0); // (A x B) * alpha

  auto tv5 = castOp(DataType::Float, tv2);
  auto tv6 = add(tv4, tv5);
  auto tv7 = castOp(DataType::Half, tv6);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(s0);
  fusion->addOutput(tv7);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn(out_shape, at::kHalf).cuda();
  float alpha = 2.5;
  at::Tensor out_ref = at::addmm(t2, t0, t1, 1, alpha);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1, t2, alpha});

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

// fd.ops.linear(a, b) where a = [M,K], b = [N,K]
TEST_F(MatmulATenEvaluationTest, Linear) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{n, k}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, true, false}); // [M, 1, K]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, N, K]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});
  auto tv3 = castOp(DataType::Half, tv2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv3);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::linear(t0, t1);

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

// fd.ops.linear(a, b, bias) where a = [M,K], b = [N,K], bias = [N]
TEST_F(MatmulATenEvaluationTest, LinearWithBias) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t m = 32, n = 64, k = 128;
  std::vector<int64_t> a_shape{m, k}, b_shape{n, k}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, true, false}); // [M, 1, K]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, N, K]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});
  auto tv3 = makeConcreteTensor({n}, DataType::Half);
  auto tv5 = add(tv2, tv3);
  auto tv6 = castOp(DataType::Half, tv5);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv3);
  fusion->addOutput(tv6);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn({n}, at::kHalf).cuda();
  at::Tensor out_ref = at::linear(t0, t1, t2);

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
