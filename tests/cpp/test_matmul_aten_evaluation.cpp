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

using Sizes = std::vector<int64_t>;
using MatmulNodeParamType = std::tuple<Sizes, Sizes>;

class MatmulNodeParameterizedTest
    : public NVFuserFixtureParamTest<MatmulNodeParamType> {
 protected:
  // Allocation order set by the pass breaks matmul tests
  // see issue https://github.com/NVIDIA/Fuser/issues/1810
  MatmulNodeParameterizedTest() : optimization_guard_(false) {}

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

// Check that ID exact mapping works as expected
void checkMatmulOpIdMapping(
    Fusion* fusion,
    TensorView* A,
    TensorView* B,
    TensorView* output) {
  IdModel id_model(fusion);
  const ValGraph& vg = id_model.buildExactGraph();
  vg.validateConsistency();

  // If K is Broadcast then we will not have a reduction dim.
  bool k_bcast = A->axis(-1)->isBroadcast();
  int64_t red_dims = k_bcast ? 0 : 1;
  auto out_ndims = std::max(A->nDims(), B->nDims()) + red_dims;
  if (std::min(A->nDims(), B->nDims()) == 1) {
    out_ndims = std::max(A->nDims(), B->nDims()) - 1 + red_dims;
  }
  ASSERT_EQ(output->nDims(), out_ndims);

  if (A->nDims() > 1) {
    int out_mpos = B->nDims() > 1 ? -2 - red_dims : -1 - red_dims;
    EXPECT_TRUE(checkMapped(vg, A->axis(-2), output->axis(out_mpos))); // M
  }

  if (B->nDims() > 1) {
    EXPECT_TRUE(checkMapped(vg, B->axis(-1), output->axis(-1 - red_dims))); // N
  }

  if (!k_bcast) {
    int b_kpos = B->nDims() > 1 ? -2 : -1; // {..iK, iN} or {iK}
    EXPECT_TRUE(checkMapped(vg, A->axis(-1), B->axis(b_kpos))); // K
    EXPECT_TRUE(checkMapped(vg, A->axis(-1), output->axis(-1))); // K
  }

  // Check that batch dims are mapped
  // Note that A and B can have different dimensions, so here we count
  // backwards from the innermost batch dimension. Then we check that the axis
  // exists (is not negative) and is not Broadcast before checking mapping.
  int batch_ndims =
      output->nDims() - (B->nDims() > 1) - (A->nDims() > 1) - red_dims;
  for (int64_t i : arange(batch_ndims)) {
    int64_t i_a = A->nDims() - 3 - i;
    int64_t i_b = B->nDims() - 3 - i;
    int64_t i_out = batch_ndims - 1 - i;
    if (i_a >= 0 && !A->axis(i_a)->isBroadcast()) {
      EXPECT_TRUE(checkMapped(vg, A->axis(i_a), output->axis(i_out)));
    }
    if (i_b >= 0 && !B->axis(i_b)->isBroadcast()) {
      EXPECT_TRUE(checkMapped(vg, B->axis(i_b), output->axis(i_out)));
    }
  }
}

// Check that ID exact mapping works as expected
void checkLinearOpIdMapping(
    Fusion* fusion,
    TensorView* input,
    TensorView* weight,
    TensorView* bias,
    TensorView* output) {
  IdModel id_model(fusion);
  const ValGraph& vg = id_model.buildExactGraph();
  vg.validateConsistency();

  // input: [* , in_features]
  // weight: [out_features, in_features]
  // bias (optional): [out_features]
  // output = [*, (out_features), rK]

  bool k_bcast = input->axis(-1)->isBroadcast();
  int64_t red_dims = k_bcast ? 0 : 1;
  ASSERT_EQ(output->nDims(), input->nDims() + weight->nDims() - 2 + red_dims);

  // Check that the first input_size - 1 dims are mapped for input
  for (auto i : arange(input->nDims() - 1)) {
    if (!input->axis(i)->isBroadcast()) {
      EXPECT_TRUE(checkMapped(vg, input->axis(i), output->axis(i)));
    }
  }
  // Check out_features dim is mapped in weight & bias if present.
  if (weight->nDims() > 1) {
    if (!weight->axis(0)->isBroadcast()) {
      EXPECT_TRUE(
          checkMapped(vg, weight->axis(0), output->axis(-1 - red_dims)));
    }
    if (bias != nullptr && bias->nDims() > 0 && !bias->axis(0)->isBroadcast()) {
      EXPECT_TRUE(checkMapped(vg, bias->axis(0), output->axis(-1 - red_dims)));
    }
  }
  // Check mapping for reduction axis in input and weight
  if (!input->axis(-1)->isBroadcast()) {
    EXPECT_TRUE(checkMapped(vg, input->axis(-1), weight->axis(-1)));
    EXPECT_TRUE(checkMapped(vg, input->axis(-1), output->axis(-1)));
  }
}

TEST_P(MatmulNodeParameterizedTest, MatmulNodeConcrete) {
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs({t0, t1});

  EXPECT_TRUE(at::allclose(out[0].as<at::Tensor>(), out_ref));
}

TEST_P(MatmulNodeParameterizedTest, MatmulNodeSymbolic) {
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out = executor_cache.runFusionWithInputs({t0, t1});

  EXPECT_TRUE(at::allclose(out[0].as<at::Tensor>(), out_ref));
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

  checkLinearOpIdMapping(fusion.get(), tv0, tv1, bias, tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias_shape.has_value()) {
    bias_opt = at::randn(*bias_shape, at::kHalf).cuda();
  }
  at::Tensor out_ref = at::linear(t0, t1, bias_opt);

  FusionExecutorCache executor_cache(std::move(fusion));

  KernelArgumentHolder cg_outputs;
  if (bias_shape.has_value()) {
    cg_outputs = executor_cache.runFusionWithInputs({t0, t1, bias_opt});
  } else {
    cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  }

  const auto& executors =
      executor_cache.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  EXPECT_FALSE(executors.front()->isA<KernelExecutor>());

  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), out_ref));
}
TEST_P(LinearNodeParametrizedTest, LinearNodeSymbolic) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const auto& [a_shape, b_shape, bias_shape] = GetParam();

  auto tv0 = makeSymbolicTensor(a_shape, DataType::Half);
  auto tv1 = makeSymbolicTensor(b_shape, DataType::Half);

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

  checkLinearOpIdMapping(fusion.get(), tv0, tv1, bias, tv2);

  at::Tensor t0 = at::randn(a_shape, at::kHalf).cuda();
  at::Tensor t1 = at::randn(b_shape, at::kHalf).cuda();
  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias_shape.has_value()) {
    bias_opt = at::randn(*bias_shape, at::kHalf).cuda();
  }
  at::Tensor out_ref = at::linear(t0, t1, bias_opt);

  FusionExecutorCache executor_cache(std::move(fusion));

  KernelArgumentHolder cg_outputs;
  if (bias_shape.has_value()) {
    cg_outputs = executor_cache.runFusionWithInputs({t0, t1, bias_opt});
  } else {
    cg_outputs = executor_cache.runFusionWithInputs({t0, t1});
  }

  const auto& executors =
      executor_cache.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  EXPECT_FALSE(executors.front()->isA<KernelExecutor>());

  EXPECT_TRUE(at::allclose(cg_outputs[0].as<at::Tensor>(), out_ref));
}

constexpr int64_t b = 128, m = 64, k = 32, n = 16;

// Parametrize a_shape and b_shape
INSTANTIATE_TEST_SUITE_P(
    ,
    MatmulNodeParameterizedTest,
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
    MatmulNodeParameterizedTest,
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
        testing::Values(
            Sizes({k}),
            Sizes({m, k}),
            Sizes({b, m, k}),
            Sizes({1, k}),
            Sizes({b, 1, k})),
        testing::Values(Sizes({n, k}), Sizes({1, k})),
        testing::Values(std::nullopt)));

INSTANTIATE_TEST_SUITE_P(
    LinearWithBias,
    LinearNodeParametrizedTest,
    testing::Combine(
        testing::Values(
            Sizes({k}),
            Sizes({m, k}),
            Sizes({b, m, k}),
            Sizes({1, k}),
            Sizes({b, 1, k})),
        testing::Values(Sizes({n, k})),
        testing::Values(Sizes({n}))));

INSTANTIATE_TEST_SUITE_P(
    LinearReductionAxisIsOne,
    LinearNodeParametrizedTest,
    testing::Combine(
        testing::Values(
            Sizes({1}),
            Sizes({m, 1}),
            Sizes({b, m, 1}),
            Sizes({1, 1}),
            Sizes({b, 1, 1})),
        testing::Values(Sizes({n, 1})),
        testing::Values(Sizes({n}))));

} // namespace nvfuser
