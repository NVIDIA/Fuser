// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include "fusion.h"
#include "fusion_guard.h"
#include "ops/all_ops.h"
#include "runtime/executor.h"
#include "tests/cpp/utils.h"
#include "validator_utils.h"

namespace nvfuser {

using MathOptTest = NVFuserTest;

TEST_F(MathOptTest, FastMathTanh) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = tanh(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({4, 8}, options);

  KernelExecutor ke;
  {
    DebugDumpOptionsGuard debug_dump_options_guard;
    DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::Ptx);
    EnableOptionsGuard enable_opt_guard;
    EnableOptionsGuard::getCurOptions().set(EnableOption::FastMath);
    ke.compile(fusion.get(), {t0});
  }

  // Verify PTX, result validation is skipped since reference won't use fast
  // math.
  const executor_utils::CudaExecutable* compiled_kernel =
      ke.compiledKernel()->cudaExecutable().get();
  std::string ptx_string(
      compiled_kernel->ptx.begin(), compiled_kernel->ptx.end());
  EXPECT_TRUE(ptx_string.find("tanh.approx.f32") != std::string::npos);
}

using NanReductionTest = NVFuserFixtureParamTest<BinaryOpType>;

TEST_P(NanReductionTest, Test) {
  // Check NAN reduction behavior for several cases:
  // 1. No NAN input -> no NAN output
  // 2. Single NAN input -> NAN output only for min/max (not fmin/fmax)
  // 3. All NAN input -> NAN output

  BinaryOpType opType = GetParam();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  Val* init = ops::binOpIdentity(opType, tv0->dtype());
  auto tv1 = reductionOp(opType, {0}, init, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t0});

  // No-NAN input
  auto cg_outputs = ke.run({t0});
  EXPECT_FALSE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());

  // Single NAN input
  t0[0] = std::numeric_limits<float>::quiet_NaN();
  cg_outputs = ke.run({t0});
  bool any_nan =
      at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>();
  if (opType == BinaryOpType::FMax || opType == BinaryOpType::FMin) {
    EXPECT_FALSE(any_nan);
  } else {
    EXPECT_TRUE(any_nan);
  }

  // All NAN input
  t0 = at::full({32}, std::numeric_limits<float>::quiet_NaN(), options);
  cg_outputs = ke.run({t0});
  EXPECT_TRUE(at::any(at::isnan(cg_outputs[0].as<at::Tensor>())).item<bool>());
}

INSTANTIATE_TEST_SUITE_P(
    MathOptTest,
    NanReductionTest,
    ::testing::Values(
        BinaryOpType::Max,
        BinaryOpType::FMax,
        BinaryOpType::Min,
        BinaryOpType::FMin),
    [](const testing::TestParamInfo<BinaryOpType>& info) -> std::string {
      std::stringstream ss;
      ss << info.param;
      return sanitizeTestName(ss.str());
    });

class FMinFMaxPromotionTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();

    fusion_ = std::make_unique<Fusion>();
    fg_ = std::make_unique<FusionGuard>(fusion_.get());

    in_tv0_ = makeSymbolicTensor(2);
    fusion_->addInput(in_tv0_);
    in_tv1_ = makeSymbolicTensor(2);
    fusion_->addInput(in_tv1_);
    in_tv2_ = makeSymbolicTensor(2);
    fusion_->addInput(in_tv2_);
  }

  void TearDown() override {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({32, 32}, options);
    t0[0][1] = std::numeric_limits<float>::quiet_NaN();

    auto t1 = at::randn({32, 32}, options);
    auto t2 = at::randn({32, 32}, options);

    FusionExecutorCache executor_cache(std::move(fusion_));
    auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

    testValidate(
        executor_cache.fusion(), outputs, {t0, t1, t2}, __LINE__, __FILE__);

    auto kernel_runtime = executor_cache.getMostRecentKernelRuntime();

    bool anyFMax = false;
    for (auto& segment : kernel_runtime->fusionSegments()->groups()) {
      const auto* ke = kernel_runtime->executors()
                           .at(segment->groupId())
                           ->as<KernelExecutor>();
      std::string kernel_code = ke->compiledKernel()->kernelString();
      if (kernel_code.find("fmax(") != std::string::npos) {
        anyFMax = true;
      }
    }

    NVF_CHECK(anyFMax == should_promote_fmax_);
  }

  std::unique_ptr<Fusion> fusion_;
  std::unique_ptr<FusionGuard> fg_;
  TensorView* in_tv0_;
  TensorView* in_tv1_;
  TensorView* in_tv2_;
  bool should_promote_fmax_ = false;
};

// The most basic case of promotion. The sum covers the max reduction.
TEST_F(FMinFMaxPromotionTest, BasicMaxSum) {
  TensorView* tv1 = max(in_tv0_, {0, 1});
  TensorView* tv2 = sum(in_tv0_, {0, 1});
  // At tv3, the damage done by an fmax promotion is repaired by the BinaryOp
  // with tv2.
  TensorView* tv3 = add(tv1, tv2);
  fusion_->addOutput(tv3);
  should_promote_fmax_ = true;
}

// Like BasicMaxSum but reducing over different axes, so sum doesn't cover max.
TEST_F(FMinFMaxPromotionTest, MaxSumDifferentAxes) {
  TensorView* tv1 = max(in_tv0_, {0});
  TensorView* tv2 = sum(in_tv0_, {1});
  TensorView* tv3 = add(tv1, tv2);
  fusion_->addOutput(tv3);
  should_promote_fmax_ = false;
}

// Like BasicMaxSum, but the tensors are different, so sum doesn't cover max.
TEST_F(FMinFMaxPromotionTest, MaxSumDifferentTensorViews) {
  TensorView* tv1 = max(in_tv0_, {0});
  TensorView* tv2 = sum(in_tv1_, {0});
  TensorView* tv3 = add(tv1, tv2);
  fusion_->addOutput(tv3);
  should_promote_fmax_ = false;
}

// Like BasicMaxSum but with unary ops inserted.
// Unary ops should not affect the promotion at all.
TEST_F(FMinFMaxPromotionTest, MaxSumSameAxesUnary) {
  TensorView* tv1 = max(in_tv0_, {0, 1});
  TensorView* tv2 = sum(in_tv0_, {0, 1});
  TensorView* tv3 = abs(tv1);
  TensorView* tv4 = abs(tv2);
  TensorView* tv5 = add(tv3, tv4);
  fusion_->addOutput(tv5);
  should_promote_fmax_ = true;
}

// Like BasicMaxSum but with binary ops connected to unrelated inputs.
// Like unary ops, binary ops with unrelated inputs do not affect the promotion.
TEST_F(FMinFMaxPromotionTest, MaxSumSameAxesBinary) {
  TensorView* tv1 = max(in_tv0_, {0, 1});
  TensorView* tv2 = sum(in_tv0_, {0, 1});
  TensorView* tv3 = broadcast(tv1, {true, true});
  TensorView* tv4 = broadcast(tv2, {true, true});
  TensorView* tv5 = add(tv3, in_tv1_);
  TensorView* tv6 = add(tv4, in_tv2_);
  TensorView* tv7 = add(tv5, tv6);
  fusion_->addOutput(tv7);
  should_promote_fmax_ = true;
}

// The axes are repaired separately by multiple safe reductions
// Although this is safe to promote, the current algorithm cannot verify it.
TEST_F(FMinFMaxPromotionTest, MultiStageRepair) {
  TensorView* tv1 = max(in_tv0_, {0, 1});
  TensorView* tv2 = sum(in_tv0_, {1});
  TensorView* tv3 = sum(tv2, {0});
  TensorView* tv4 = add(tv1, tv3);
  fusion_->addOutput(tv4);
  should_promote_fmax_ = false;
}

// Here the reductions broadcast up to 2D along different axes.
// They are basically transposed with each other, and repair doesn't happen.
TEST_F(FMinFMaxPromotionTest, WrongBroadcast) {
  TensorView* tv1 = max(in_tv0_, {1});
  TensorView* tv2 = sum(in_tv0_, {1});
  TensorView* tv3 = broadcast(tv1, {true, false});
  TensorView* tv4 = broadcast(tv2, {false, true});
  TensorView* tv5 = add(tv3, tv4);
  fusion_->addOutput(tv5);
  should_promote_fmax_ = false;
}

// Normalization pattern requiring a mixed state
TEST_F(FMinFMaxPromotionTest, Normalization) {
  TensorView* tv1 = max(in_tv0_, {1});
  TensorView* tv2 = broadcast(tv1, {false, true});

  // tv2 is in a mixed state. It's not a safe output, but it could be repaired
  // by a safe reduction.
  TensorView* tv3 = add(tv2, in_tv0_);

  TensorView* tv4 = sum(tv3, {1});
  TensorView* tv5 = broadcast(tv4, {false, true});
  TensorView* tv6 = add(tv5, tv4);
  fusion_->addOutput(tv6);
  should_promote_fmax_ = true;
}

// Normalization with unary and binary ops thrown in.
// These should not affect promotion.
TEST_F(FMinFMaxPromotionTest, NormalizationUnaryBinary) {
  TensorView* tv1 = max(in_tv0_, {0});

  // Unary op
  TensorView* tv2 = abs(tv1);
  TensorView* tv3 = broadcast(tv2, {true, false});
  TensorView* tv4 = add(tv3, in_tv0_);
  TensorView* tv5 = sum(tv4, {0});
  TensorView* tv6 = broadcast(tv5, {true, false});

  // Unrelated binary op
  TensorView* tv7 = add(tv6, in_tv1_);
  fusion_->addOutput(tv7);
  should_promote_fmax_ = true;
}

// Normalization style pattern, but with different axes, breaking promotion.
TEST_F(FMinFMaxPromotionTest, NormalizationDifferentAxes) {
  TensorView* tv1 = max(in_tv0_, {0});
  TensorView* tv2 = broadcast(tv1, {true, false});
  TensorView* tv3 = add(tv2, in_tv0_);
  TensorView* tv4 = sum(tv3, {1});
  TensorView* tv5 = broadcast(tv4, {true, false});
  TensorView* tv6 = add(tv5, tv3);
  fusion_->addOutput(tv6);
  should_promote_fmax_ = false;
}

// Two unsafe reductions on the same input. Exactly one should be promoted.
// This tests that the promotion considers its previous promotion decisions.
TEST_F(FMinFMaxPromotionTest, SiblingReduction) {
  TensorView* tv1 = max(in_tv0_, {0, 1});
  TensorView* tv2 = min(in_tv0_, {0, 1});
  // tv1 must be the first argument to add, simply because we check for fmax and
  // not fmin.
  TensorView* tv3 = add(tv1, tv2);
  fusion_->addOutput(tv3);
  should_promote_fmax_ = true;
}

} // namespace nvfuser
