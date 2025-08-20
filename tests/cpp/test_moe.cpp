// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <gtest/gtest.h>

namespace nvfuser {

using MoEConfig = std::tuple<
    int64_t, // num_experts
    int64_t, // num_tokens
    int64_t, // topk
    int64_t, // rounding_factor
    bool // manual_scheduling
    >;

std::ostream& operator<<(std::ostream& os, const MoEConfig& config) {
  os << std::get<0>(config) << "_" << std::get<1>(config) << "_"
     << std::get<2>(config) << "_" << std::get<3>(config) << "_"
     << (std::get<4>(config) ? "manual" : "auto");
  return os;
}

// Reproducing the CUDA kernels used in SGLang MoE gate
// logic.
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/prepare_moe_input.cu
class SgLangMoETest : public NVFuserFixtureParamTest<MoEConfig> {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});

    std::tie(
        num_experts, num_tokens, topk, rounding_factor, manual_scheduling) =
        GetParam();
  }

 protected:
  int64_t num_experts = 16;
  int64_t num_tokens = 32;
  int64_t topk = 4;
  int64_t rounding_factor = 128;
  bool manual_scheduling = false;
};

TEST_P(SgLangMoETest, ComputeProblemSizes) {
  if (manual_scheduling) {
    GTEST_SKIP() << "No manual scheduling implemented";
  }

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // topk_ids
  auto tv0 = makeContigConcreteTensor({num_tokens, topk}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = flatten(tv0);

  // tokens_per_expert
  auto tv2 = zeros({IrBuilder::create<Val>(num_experts)}, DataType::Int);

  auto tv3 = ones({IrBuilder::create<Val>(num_tokens * topk)}, DataType::Int);

  auto tv4 = indexPutAccumulate(tv2, tv1, tv3);

  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, num_experts, {num_tokens, topk}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}

TEST_P(SgLangMoETest, ComputeExpertOffsets) {
  if (manual_scheduling) {
    GTEST_SKIP() << "No manual scheduling implemented";
  }

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // tokens_per_expert
  auto tv0 = makeContigConcreteTensor({num_experts}, DataType::Int);
  fusion.addInput(tv0);

  // Inclusive scan
  auto zero = fusion.zeroVal(DataType::Int);
  auto tv1 = cumsum(tv0, zero);
  // Exclusive scan + total count
  auto tv2 =
      pad(tv1, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});

  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, num_tokens * topk, {num_experts}, options);

  if (manual_scheduling) {
    tv0->cacheAfter();
    // Stage tv1 to shared memory as tv1 and tv2 loop domains are not
    // mapped
    auto tv1_cache = tv1->cacheAfter();

    for (auto tv : fusion.allTvs()) {
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }

    tv1_cache->setMemoryType(MemoryType::Shared);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
  } else {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto outputs = executor_cache.runFusionWithInputs({t0});
    testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_P(SgLangMoETest, ComputeExpertBlockScaleOffsets) {
  if (manual_scheduling) {
    GTEST_SKIP() << "No manual scheduling implemented";
  }

  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // tokens_per_expert
  auto tv0 = makeContigConcreteTensor({num_experts}, DataType::Int);
  fusion.addInput(tv0);

  // The first part is the same as ComputeExpertOffsets
  auto zero = fusion.zeroVal(DataType::Int);
  auto tv1 = cumsum(tv0, zero);
  auto tv2 =
      pad(tv1, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});

  // Does the same cumsum with rounded token counts
  auto tv3 =
      ceilDiv(tv0, IrBuilder::create<Val>(rounding_factor, DataType::Int));
  auto tv4 = mul(tv3, IrBuilder::create<Val>(rounding_factor, DataType::Int));
  auto tv5 = cumsum(tv4, zero);
  auto tv6 =
      pad(tv5, {fusion.oneVal(DataType::Int), fusion.zeroVal(DataType::Int)});

  fusion.addOutput(tv2);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, num_tokens * topk, {num_experts}, options);

  // Manually computing the results as ceilDiv doesn't seem to work
  // with ExprEvaluator. This is the error message:
  //
  // C++ exception with description "std::get: wrong index for variant" thrown
  // in the test body.
  auto t2 = at::pad(at::cumsum(t0, 0), {1, 0});
  auto t0_rounded =
      at::floor((t0 + rounding_factor - 1) / rounding_factor) * rounding_factor;
  auto t6 = at::pad(at::cumsum(t0_rounded, 0), {1, 0});

  if (manual_scheduling) {
    tv0->cacheAfter();
    auto tv1_cache = tv1->cacheAfter();
    auto tv5_cache = tv5->cacheAfter();

    for (auto tv : fusion.allTvs()) {
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }

    tv1_cache->setMemoryType(MemoryType::Shared);
    tv5_cache->setMemoryType(MemoryType::Shared);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    auto outputs = ke.run({t0});
    testValidate(&fusion, outputs, {t0}, {t2, t6}, __LINE__, __FILE__);
  } else {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto outputs = executor_cache.runFusionWithInputs({t0});
    testValidate(
        executor_cache.fusion(), outputs, {t0}, {t2, t6}, __LINE__, __FILE__);
  }
}

TEST_P(SgLangMoETest, ComputeArgSort) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  // topk_ids
  auto tv0 = makeContigConcreteTensor({num_tokens, topk}, DataType::Int);
  fusion.addInput(tv0);

  auto tv1 = flatten(tv0);
  // argsort indices
  auto tv2 = argsort(tv1, 0, /*descending=*/true, /*stable=*/true);
  // input_permutation
  auto tv3 = div(tv2, IrBuilder::create<Val>(topk, DataType::Int));

  // This doesn't need to be initialized
  // output_permutation
  auto tv4 = zeros({IrBuilder::create<Val>(num_tokens * topk)}, DataType::Int);
  // topk_ids_offset
  auto tv5 = arange(
      fusion.zeroVal(DataType::Int),
      IrBuilder::create<Val>(num_tokens * topk, DataType::Int),
      DataType::Int);
  auto tv6 = scatter(tv4, 0, tv2, tv5);

  fusion.addOutput(tv3);
  fusion.addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto t0 = at::randint(0, num_tokens * topk, {num_tokens, topk}, options);

  KernelArgumentHolder outputs;

  if (manual_scheduling) {
    auto tv6_cache = tv6->cacheBefore();

    // Scheduling all tensors as 1D tensors
    for (auto tv : fusion.allTvs()) {
      tv->flatten();
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }

    tv4->setMemoryType(MemoryType::Shared);
    tv4->setAllocationDomain(tv4->getLogicalDomain(), true);
    tv6_cache->setMemoryType(MemoryType::Shared);
    tv6_cache->setAllocationDomain(tv6_cache->getLogicalDomain(), true);

    KernelExecutor ke;
    ke.compile(&fusion, {t0});
    outputs = ke.run({t0});
  } else {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    outputs = executor_cache.runFusionWithInputs({t0});
  }

  auto t2 = at::argsort(t0.flatten(), true, 0, true);
  auto t3 = at::floor(t2 / topk);
  auto t4 = at::zeros({num_tokens * topk}, options);
  auto t5 = at::arange(0, num_tokens * topk, options);
  t4.scatter_(0, t2, t5);

  EXPECT_TRUE(outputs[0].as<at::Tensor>().equal(t3));
  EXPECT_TRUE(outputs[1].as<at::Tensor>().equal(t4));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    SgLangMoETest,
    testing::Combine(
        testing::Values(16), // M
        testing::Values(32, 1), // N
        testing::Values(4, 1), // topk
        testing::Values(128), // rounding factor
        testing::Bool()), // manual_scheduling
    [](const testing::TestParamInfo<MoEConfig>& info) {
      const auto& moe_config = info.param;
      std::ostringstream os;
      os << moe_config;
      return os.str();
    });

} // namespace nvfuser
