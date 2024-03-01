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
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

class MatmulATenEvaluationTest : public NVFuserTest {
 protected:
  void SetUp() override {
    // allocation order set by the pass breaks matmul tests
    // see issue https://github.com/NVIDIA/Fuser/issues/1810
    guard_ = std::make_unique<nvfuser::preseg_passes::OptimizationPassGuard<
        nvfuser::preseg_passes::AllocationDomainPass>>(false);
  }
  std::unique_ptr<nvfuser::preseg_passes::OptimizationPassGuard<
      nvfuser::preseg_passes::AllocationDomainPass>>
      guard_;
};

TEST_F(MatmulATenEvaluationTest, SingleMmaOp) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard enable_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MatmulExprEval);

  int64_t m = 2, k = 3, n = 4;
  std::vector<int64_t> a_shape{m, k}, b_shape{k, n}, out_shape{m, n};

  auto tv0 = makeConcreteTensor(a_shape, DataType::Half);
  auto tv1 = makeConcreteTensor(b_shape, DataType::Half);
  auto tv0b = broadcast(tv0, {false, false, true}); // [M, K, 1]
  auto tv1b = broadcast(tv1, {true, false, false}); // [1, K, N]
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {1});

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  at::Tensor t0 = at::ones(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::ones(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::full(out_shape, k, at::kFloat).cuda();

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  EXPECT_EQ(fec.getMostRecentKernelRuntime()->executors().size(), 1);

  // Verify that the io_alias_ set has the correct entry
  auto kernel = fec.getMostRecentKernelRuntime()->executors().at(0).kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_F(MatmulATenEvaluationTest, MmaOpAndCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard enable_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MatmulExprEval);

  int64_t m = 2, k = 3, n = 4;
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

  at::Tensor t0 = at::ones(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::ones(b_shape, at::kHalf).cuda();
  at::Tensor out_ref = at::full(out_shape, k, at::kHalf).cuda();

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1});

  EXPECT_EQ(fec.getMostRecentKernelRuntime()->executors().size(), 1);

  // Verify that the io_alias_ set has the correct entry
  auto kernel = fec.getMostRecentKernelRuntime()->executors().at(0).kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

TEST_F(MatmulATenEvaluationTest, MatmulWithBias) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  EnableOptionsGuard enable_guard;
  EnableOptionsGuard::getCurOptions().set(EnableOption::MatmulExprEval);

  int64_t m = 2, k = 3, n = 4;
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

  at::Tensor t0 = at::ones(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::ones(b_shape, at::kHalf).cuda();
  at::Tensor t2 = at::randn({m}, at::kHalf).cuda();
  at::Tensor out_ref =
      at::full(out_shape, k, at::kHalf).cuda() + t2.unsqueeze(-1);

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs({t0, t1, t2});

  EXPECT_EQ(fec.getMostRecentKernelRuntime()->executors().size(), 1);

  // Verify that the io_alias_ set has the correct entry
  auto kernel = fec.getMostRecentKernelRuntime()->executors().at(0).kernel();
  EXPECT_EQ(
      kernel->getOutputAlias(kernel->outputs()[0]).type,
      AllocationType::Evaluate);

  EXPECT_TRUE(at::allclose(out[0], out_ref));
}

} // namespace nvfuser
