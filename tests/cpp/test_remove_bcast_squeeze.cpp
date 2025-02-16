// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ir/all_nodes.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/optimization_pass.h>
#include <preseg_passes/pre_segmenter.h>
#include <preseg_passes/remove_bcast_squeeze.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <torch/torch.h>

namespace nvfuser {

using RemoveBcastSqueezeTest = NVFuserTest;

namespace {

inline TensorView* makeBroadcastTensor(
    const std::vector<bool>& is_broadcast_dim,
    DataType input_dtype = DataType::Float) {
  std::vector<IterDomain*> out_domain;
  out_domain.reserve(is_broadcast_dim.size());
  for (auto is_broadcast : is_broadcast_dim) {
    out_domain.push_back(
        IterDomainBuilder(
            FusionGuard::getCurFusion()->zeroVal(),
            is_broadcast ? FusionGuard::getCurFusion()->oneVal()
                         : IrBuilder::create<Val>(DataType::Index))
            .iter_type(is_broadcast ? IterType::Broadcast : IterType::Iteration)
            .build());
  }
  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      input_dtype);
}

} // namespace

TEST_F(RemoveBcastSqueezeTest, BcastSqueeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = set(tv3);
  fusion->addOutput(tv4);

  // preseg_passes should remove both broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeMultipleUses) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim{false, false, true};
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tvb = makeBroadcastTensor(is_broadcast_dim, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tvb);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, is_broadcast_dim);
  auto tv3 = squeeze(tv2, is_broadcast_dim);
  auto tv4 = add(tv3, tv3);
  auto tv5 = add(tv2, tvb);
  fusion->addOutput(tv4);
  fusion->addOutput(tv5);

  // preseg_passes should remove squeeze only since broadcast output is
  // also used by another op tv5 = add(tv2, tvb).
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_TRUE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));

  // run fusion
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::ones({3, 4}, options);
  at::Tensor t1 = t0.unsqueeze(-1);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(executor_cache.fusion(), outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeUnmatchedDim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, false, true});
  auto tv4 = set(tv3);
  fusion->addOutput(tv4);

  // preseg_passes should remove squeeze and alter broadcast flags to simply not
  // insert the squeezed axis.
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_TRUE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
  for (auto expr : fusion->exprs()) {
    if (auto* bcast = dynamic_cast<BroadcastOp*>(expr)) {
      EXPECT_EQ(
          bcast->getBroadcastDimFlags(),
          (std::vector<bool>{false, false, true}));
    }
  }
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeOutputBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = set(tv3);
  fusion->addOutput(tv2);
  fusion->addOutput(tv4);

  // preseg_passes should remove squeeze but not broadcast scine tv2 is an
  // output
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_TRUE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeOutputSqueeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = set(tv3);
  fusion->addOutput(tv3);
  fusion->addOutput(tv4);

  // preseg_passes should remove both squeeze and broadcast.
  // tv3 is an output and it is replaced with tv1.
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeInputBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = broadcast(tv0, {false, false, true});
  auto tv2 = squeeze(tv1, std::vector<bool>{false, false, true});
  fusion->addOutput(tv2);

  // input to broadcast is also an input to the fusion
  // preseg_passes should remove both broadcast and squeeze
  // This fusion is a no-op
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim{false, true};
  auto tv0 = makeBroadcastTensor(is_broadcast_dim, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, is_broadcast_dim);
  auto tv3 = broadcast(tv2, is_broadcast_dim);
  auto tv4 = set(tv3);
  fusion->addOutput(tv4);

  // preseg_passes should remove both broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcastOutputBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim{false, true};
  auto tv0 = makeBroadcastTensor(is_broadcast_dim, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, is_broadcast_dim);
  auto tv3 = broadcast(tv2, is_broadcast_dim);
  fusion->addOutput(tv3);

  // broadcast output is the fusion output
  // preseg_passes should remove both broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcastOutputSqueeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim{false, true};
  auto tv0 = makeBroadcastTensor(is_broadcast_dim, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, is_broadcast_dim);
  auto tv3 = broadcast(tv2, is_broadcast_dim);
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  // squeeze output is the fusion output
  // preseg_passes should remove broadcast but not squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_TRUE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcastSqueezeBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim{false, true};
  auto tv0 = makeBroadcastTensor(is_broadcast_dim, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, is_broadcast_dim);
  auto tv3 = broadcast(tv2, is_broadcast_dim);
  auto tv4 = squeeze(tv3, is_broadcast_dim);
  auto tv5 = broadcast(tv4, is_broadcast_dim);
  auto tv6 = add(tv3, tv5);
  fusion->addOutput(tv6);

  // squeeze output is the fusion output
  // preseg_passes should remove broadcast but not squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcastBcastSqueeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  const std::vector<bool> is_broadcast_dim_1{false, true};
  const std::vector<bool> is_broadcast_dim_2{false, false, true};
  auto tv0 = makeBroadcastTensor(is_broadcast_dim_1, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = squeeze(tv1, is_broadcast_dim_1);
  auto tv3 = broadcast(tv2, is_broadcast_dim_1);
  auto tv4 = set(tv3);
  auto tv5 = broadcast(tv4, is_broadcast_dim_2);
  auto tv6 = squeeze(tv5, is_broadcast_dim_2);
  auto tv7 = add(tv3, tv6);
  fusion->addOutput(tv7);

  // preseg_passes should remove both broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = broadcast(tv3, {false, false, true});
  auto tv5 = set(tv4);
  fusion->addOutput(tv5);

  // preseg_passes should remove the first broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_TRUE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeBcastSqueeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = broadcast(tv3, {false, false, true});
  auto tv5 = squeeze(tv4, std::vector<bool>{false, false, true});
  auto tv6 = add(tv3, tv5);
  fusion->addOutput(tv6);

  // preseg_passes should remove the first broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, BcastSqueezeSqueezeBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeBroadcastTensor({false, true}, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, std::vector<bool>{false, false, true});
  auto tv3 = squeeze(tv2, std::vector<bool>{false, false, true});
  auto tv4 = squeeze(tv3, std::vector<bool>{false, true});
  auto tv5 = broadcast(tv4, std::vector<bool>{false, true});
  auto tv6 = add(tv3, tv5);
  fusion->addOutput(tv6);

  // preseg_passes should remove the first broadcast and squeeze
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

TEST_F(RemoveBcastSqueezeTest, SqueezeBcastSetBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DataType input_dtype = DataType::Float;
  auto tv0 = makeBroadcastTensor({true, false, true}, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = squeeze(tv0, std::vector<bool>{true, false, true});
  auto tv3 = broadcast(tv1, std::vector<bool>{true, false});
  auto tv4 = set(tv3);
  auto tv5 = broadcast(tv4, std::vector<bool>{false, false, true});
  fusion->addOutput(tv5);

  // preseg_passes should remove all ops between input and output
  preseg_passes::OptimizationPass<preseg_passes::PreSegmenter>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<BroadcastOp>(fusion.get()));
  EXPECT_FALSE(ir_utils::hasOpsOfType<SqueezeOp>(fusion.get()));
}

// Test that reduction axes are ignored in input to LoadStoreOp
// See https://github.com/NVIDIA/Fuser/pull/3189
TEST_F(RemoveBcastSqueezeTest, SumSetBcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto tv0 = makeSymbolicTensor(3);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {2});
  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, std::vector<bool>{false, false, true});
  fusion->addOutput(tv3);

  preseg_passes::OptimizationPass<preseg_passes::RemoveBcastSqueeze>::runPass(
      fusion.get());
  EXPECT_FALSE(ir_utils::hasOpsOfType<LoadStoreOp>(fusion.get()));
}

} // namespace nvfuser
