// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <optimization/alias_analysis.h>
#include <test/utils.h>
#include <test/validator.h>

namespace nvfuser {

using testing::ElementsAre;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

using AliasAnalysisTest = NVFuserTest;

TEST_F(AliasAnalysisTest, View_ContiguousAndSameAllocationOrder) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);
}

TEST_F(AliasAnalysisTest, View_SymbolicTensor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({-1, -1, -1});
  fusion.addInput(in);
  std::vector<Val*> in_shape = shape(in);
  ASSERT_EQ(in_shape.size(), 3);
  TensorView* out = reshape(in, {in_shape[0], mul(in_shape[1], in_shape[2])});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);
}

TEST_F(AliasAnalysisTest, ChainOfViews) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> intermediate_shape({2, 12});
  const std::vector<int64_t> out_shape({24});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* intermediate = reshape(in, in_shape, intermediate_shape);
  TensorView* out = reshape(intermediate, intermediate_shape, out_shape);
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);
}

TEST_F(AliasAnalysisTest, View_DifferentAllocationOrder) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);
  out->setAllocationDomain(
      {out->axis(1), out->axis(0)}, /*new_contiguity=*/true);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
}

TEST_F(AliasAnalysisTest, View_NonContiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);
  out->setAllocationDomain(
      {out->axis(0), out->axis(1)}, /*new_contiguity=*/{true, false});

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
}

TEST_F(AliasAnalysisTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = permute(in, {1, 2, 0});
  fusion.addOutput(out);

  // We haven't handled `Set.Permute` yet.
  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
}

TEST_F(AliasAnalysisTest, View_SplitExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({4, 5});
  fusion.addInput(in);
  TensorView* out = broadcast(in, {false, false, true});
  out = expand(
      out,
      {IrBuilder::create<Val>(4),
       IrBuilder::create<Val>(5),
       IrBuilder::create<Val>(6)});
  // tryStaticReshape used to fail to get the expanded extent, which is 6.
  out = reshape(out, {IrBuilder::create<Val>(40), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
}

TEST_F(AliasAnalysisTest, View_ForwardExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({4, 5});
  fusion.addInput(in);
  TensorView* broadcast_out = broadcast(in, {false, false, true});
  TensorView* expand_out = expand(
      broadcast_out,
      {IrBuilder::create<Val>(4),
       IrBuilder::create<Val>(5),
       IrBuilder::create<Val>(6)});
  TensorView* out = reshape(expand_out, {4, 5, 6}, {20, -1});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), expand_out);

  // Verify the last dimension isn't expanded physically.
  FusionExecutor fe;
  at::Tensor in_tensor =
      at::randn({4, 5}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  fe.compileFusion(&fusion, {in_tensor});
  at::Tensor out_tensor = fe.runFusion({in_tensor})[0];

  EXPECT_THAT(out_tensor.strides(), ElementsAre(1, 0));
}

TEST_F(AliasAnalysisTest, View_MergeExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({4, 5});
  fusion.addInput(in);
  TensorView* out = broadcast(in, {false, false, true});
  out = expand(
      out,
      {IrBuilder::create<Val>(4),
       IrBuilder::create<Val>(5),
       IrBuilder::create<Val>(6)});
  out = reshape(out, {4, 5, 6}, {4, -1});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
}

using AliasTest = NVFuserTest;

TEST_F(AliasTest, ReinterpretCast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0];

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  // Verify output values.
  testValidate(
      fec.fusion(),
      {out_tensor},
      {in_tensor},
      {in_tensor.view({2, 12})},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
