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
#include <id_model/id_model.h>
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

using Sizes = std::vector<int64_t>;
using MatmulNodeParamType = std::tuple<Sizes, Sizes>;

class MatmulNodeParametrizedTest
    : public NVFuserFixtureParamTest<MatmulNodeParamType> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  MatmulNodeParametrizedTest() : optimization_guard_(false) {}

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
};

using LinearNodeParamType = std::tuple<Sizes, Sizes, std::optional<Sizes>>;
class LinearNodeParametrizedTest
    : public NVFuserFixtureParamTest<LinearNodeParamType> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  LinearNodeParametrizedTest() : optimization_guard_(false) {}

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

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
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

// Check that ID exact mapping works as expected
void checkMatmulOpIdMapping(
    Fusion* fusion,
    TensorView* A,
    TensorView* B,
    TensorView* output) {
  IdModel id_model(fusion);
  const ValGraph& vg = id_model.idGraph(IdMappingMode::EXACT);
  vg.validateConsistency();

  const auto checkMapped = [&vg](IterDomain* x, IterDomain* y) -> bool {
    if (!vg.hasGroup(x) || !vg.hasGroup(y)) {
      return false;
    }
    const ValGroup& gx = vg.toGroup(x);
    const ValGroup& gy = vg.toGroup(y);
    return gx.get() == gy.get();
  };

  // If K is Broadcast then we will not have a reduction dim
  bool k_bcast = A->axis(-1)->isBroadcast();
  int64_t red_dims = k_bcast ? 0 : 1;

  if (A->nDims() == 1 && B->nDims() == 1) {
    // [K] @ [K] = []
    // Note there is no IterType::Reduction dim ever in this case because we
    // translate to a mul+sum+cast
    EXPECT_EQ(output->nDims(), 0);
    // When K is Broadcast, we squeeze then multiply then cast instead
    if (!k_bcast) {
      EXPECT_TRUE(checkMapped(A->axis(0), B->axis(0))); // K
    }
  } else if (A->nDims() > 1 && B->nDims() == 1) {
    // [..., iM, iK] @ [iK] = [..., iM, rK]
    ASSERT_EQ(output->nDims(), A->nDims() + red_dims - 1);
    EXPECT_TRUE(checkMapped(A->axis(-2), output->axis(-1 - red_dims))); // M
    if (!k_bcast) {
      EXPECT_TRUE(checkMapped(A->axis(-1), B->axis(0))); // K
      EXPECT_TRUE(checkMapped(A->axis(-1), output->axis(-1))); // K
    }
    // Check that batch dims are mapped
    for (int64_t i : c10::irange(output->nDims() - red_dims - 1)) {
      if (!A->axis(i)->isBroadcast()) {
        EXPECT_TRUE(checkMapped(A->axis(i), output->axis(i)));
      }
    }
  } else if (A->nDims() == 1 && B->nDims() > 1) {
    // [iK] @ [..., iK, iN] = [..., iN, rK]
    ASSERT_EQ(output->nDims(), B->nDims() + red_dims - 1);
    EXPECT_TRUE(checkMapped(B->axis(-1), output->axis(-1 - red_dims))); // N
    if (!k_bcast) {
      EXPECT_TRUE(checkMapped(A->axis(0), B->axis(-2))); // K
      EXPECT_TRUE(checkMapped(A->axis(0), output->axis(-1))); // K
    }
    // Check that batch dims are mapped
    for (int64_t i : c10::irange(output->nDims() - red_dims - 1)) {
      if (!B->axis(i)->isBroadcast()) {
        EXPECT_TRUE(checkMapped(B->axis(i), output->axis(i)));
      }
    }
  } else if (A->nDims() > 1 && B->nDims() > 1) {
    // [..., iM, iK] @ [..., iK, iN] = [..., iM, iN, rK]
    ASSERT_EQ(output->nDims(), std::max(A->nDims(), B->nDims()) + red_dims);
    EXPECT_TRUE(checkMapped(A->axis(-2), output->axis(-2 - red_dims))); // M
    EXPECT_TRUE(checkMapped(B->axis(-1), output->axis(-1 - red_dims))); // N
    if (!k_bcast) {
      EXPECT_TRUE(checkMapped(A->axis(-1), B->axis(-2))); // K
      EXPECT_TRUE(checkMapped(A->axis(-1), output->axis(-1))); // K
    }
    // Check that batch dims are mapped
    // Note that A and B can have different dimensions, so here we count
    // backwards from the innermost batch dimension. Then we check that the axis
    // exists (is not negative) and is not Broadcast before checking mapping.
    for (int64_t i : c10::irange(output->nDims() - red_dims - 2)) {
      int64_t i_a = A->nDims() - 3 - i;
      int64_t i_b = B->nDims() - 3 - i;
      int64_t i_out = output->nDims() - red_dims - 3 - i;
      if (i_a >= 0 && !A->axis(i_a)->isBroadcast()) {
        EXPECT_TRUE(checkMapped(A->axis(i_a), output->axis(i_out)));
      }
      if (i_b >= 0 && !B->axis(i_b)->isBroadcast()) {
        EXPECT_TRUE(checkMapped(B->axis(i_b), output->axis(i_out)));
      }
    }
  } else {
    std::cerr << "Unhandled set of input dimensions" << std::endl;
    EXPECT_TRUE(false);
  }
}

TEST_P(ATenNodesParametrizedTest, MatmulNodeConcrete) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto& [a_shape, b_shape] = GetParam();

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv2 = matmul(tv0, tv1);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  checkMatmulOpIdMapping(fusion.get(), tv0, tv1, tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::matmul(t0, t1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_P(MatmulNodeParametrizedTest, MatmulNodeSymbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto& [a_shape, b_shape] = GetParam();

  auto tv0 = makeSymbolicTensor(a_shape, DataType::Half);
  auto tv1 = makeSymbolicTensor(b_shape, DataType::Half);
  auto tv2 = matmul(tv0, tv1);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  checkMatmulOpIdMapping(fusion.get(), tv0, tv1, tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::matmul(t0, t1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_P(LinearNodeParametrizedTest, LinearNodeConcrete) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto& [a_shape, b_shape, bias_shape] = GetParam();

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  TensorView* bias = nullptr;
  if (bias_shape.has_value()) {
    bias = makeConcreteTensor(*bias_shape, DataType::Half);
  }
  auto tv2 = linear(tv0, tv1, bias);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  if (bias_shape.has_value()) {
    fusion->addInput(bias);
  }
  fusion->addOutput(tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias_shape.has_value()) {
    bias_opt = bias_shape.value().empty() ? at::scalar_tensor(3.14).to(at::kHalf).cuda(): at::randn(*bias_shape, at::kHalf).cuda();
  }
  at::Tensor out_ref = at::linear(t0, t1, bias_opt);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> out = {};
  if (bias_shape.has_value()) {
    out = fec.runFusionWithInputs({t0, t1, bias_opt});
  } else {
    out = fec.runFusionWithInputs({t0, t1});
  }

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}
TEST_P(LinearNodeParametrizedTest, LinearNodeSymbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto& [a_shape, b_shape, bias_shape] = GetParam();

  auto tv0 = makeSymbolicTensor(a_shape.size(), DataType::Half);
  auto tv1 = makeSymbolicTensor(b_shape.size(), DataType::Half);

  TensorView* bias = nullptr;
  if (bias_shape.has_value()) {
    bias = makeSymbolicTensor(*bias_shape, DataType::Half);
  }

  auto tv2 = linear(tv0, tv1, bias);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  if (bias_shape.has_value()) {
    fusion->addInput(bias);
  }
  fusion->addOutput(tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias_shape.has_value()) {
    bias_opt = bias_shape.value().empty() ? at::scalar_tensor(3.14).to(at::kHalf).cuda() : at::randn(*bias_shape, at::kHalf).cuda();
  }
  at::Tensor out_ref = at::linear(t0, t1, bias_opt);

  FusionExecutorCache fec(std::move(fusion));

  std::vector<at::Tensor> out = {};
  if (bias_shape.has_value()) {
    out = fec.runFusionWithInputs({t0, t1, bias_opt});
  } else {
    out = fec.runFusionWithInputs({t0, t1});
  }

  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  // Verify that fusion compilation was skipped.
  EXPECT_FALSE(executors.front().hasCompiledKernel());

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

constexpr int64_t b = 128, m = 64, k = 32, n = 16;

// Parametrize a_shape and b_shape
INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulNodeParametrizedTest,
    testing::Combine(
        testing::Values(
            Sizes({k}),
            Sizes({m, k}),
            Sizes({1, k}),
            Sizes({b, m, k}),
            Sizes({b, 1, m, k})),
        testing::Values(
            Sizes({k}),
            Sizes({k, n}),
            Sizes({k, 1}),
            Sizes({b, k, n}))));

// Test case where K=1
INSTANTIATE_TEST_SUITE_P(
    ReductionAxisIsOne,
    ATenNodesParametrizedTest,
    testing::Combine(
        testing::Values(
            Sizes({1}),
            Sizes({m, 1}),
            Sizes({1, 1}),
            Sizes({b, m, 1}),
            Sizes({b, 1, m, 1})),
        testing::Values(
            Sizes({1}),
            Sizes({1, n}),
            Sizes({1, 1}),
            Sizes({b, 1, n}))));

INSTANTIATE_TEST_SUITE_P(
    LinearWithoutBias,
    LinearNodeParametrizedTest,
    testing::Combine(
        testing::Values(Sizes({k}), Sizes({m, k}), Sizes({b, m, k}), Sizes({1, k}), Sizes({b, 1, k})),
        testing::Values(Sizes({k}), Sizes({n, k}), Sizes({1, k})),
        testing::Values(std::nullopt)));

INSTANTIATE_TEST_SUITE_P(
    LinearWithBias,
    LinearNodeParametrizedTest,
    testing::Combine(
        testing::Values(Sizes({k}), Sizes({m, k}), Sizes({b, m, k}), Sizes({1, k}), Sizes({b, 1, k})),
        testing::Values(Sizes({n, k})),
        testing::Values(Sizes({}), Sizes({n}))));

INSTANTIATE_TEST_SUITE_P(
    LinearReductionAxisIsOne,
    LinearNodeParametrizedTest,
    testing::Combine(
        testing::Values(Sizes({m, 1}), Sizes({b, m, 1}))),
        testing::Values(Sizes({n, 1})),
        testing::Values(Sizes({}), Sizes({n})));

} // namespace nvfuser
