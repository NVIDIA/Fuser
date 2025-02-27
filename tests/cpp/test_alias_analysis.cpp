// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock-more-matchers.h>
#include <gtest/gtest.h>

#include <alias_analysis.h>
#include <fusion.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using AliasAnalysisTest = NVFuserTest;

using testing::Each;
using testing::ElementsAre;
using testing::IsTrue;
using testing::Optional;

TEST_F(AliasAnalysisTest, View_SymbolicTensor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({-1, -1, -1});
  fusion.addInput(in);
  std::vector<Val*> in_shape = shape(in);
  ASSERT_EQ(in_shape.size(), 3);
  TensorView* out = reshape(in, {in_shape[0], mul(in_shape[1], in_shape[2])});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);
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

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);
}

TEST_F(AliasAnalysisTest, View_Contiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);
  Layout preferred_layout = analysis.preferredLayout(out);
  EXPECT_THAT(
      preferred_layout.allocation_domain,
      ElementsAre(out->axis(0), out->axis(1)));
  EXPECT_THAT(preferred_layout.contiguity, Each(Optional(IsTrue())));
}

TEST_F(AliasAnalysisTest, View_MergeNonContiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = TensorViewBuilder()
                       .shape(in_shape)
                       .dtype(DataType::Float)
                       .contiguity({true, false, true})
                       .build();
  fusion.addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), nullptr);
}

TEST_F(AliasAnalysisTest, Set) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* out = set(in);
  fusion.addOutput(out);

  in->setAllocationDomain({in->axis(1), in->axis(2), in->axis(0)}, true);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);

  const std::vector<IterDomain*>& out_logical = out->getLogicalDomain();
  EXPECT_THAT(
      analysis.preferredLayout(out).allocation_domain,
      ElementsAre(out_logical[1], out_logical[2], out_logical[0]));
}

TEST_F(AliasAnalysisTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = permute(in, {1, 2, 0});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);

  const std::vector<IterDomain*>& out_logical = out->getLogicalDomain();
  EXPECT_THAT(
      analysis.preferredLayout(out).allocation_domain,
      ElementsAre(out_logical[2], out_logical[0], out_logical[1]));
}

TEST_F(AliasAnalysisTest, View_SplitExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .ndims(3)
                       .dtype(DataType::Float)
                       .contiguity({true, true, std::nullopt})
                       .shape({4, 5, 6})
                       .expanded({false, false, true})
                       .build();
  fusion.addInput(in);
  // tryStaticReshape used to fail to get the expanded extent, which is 6.
  // Therefore, we use the `vector<Val*>` version of `reshape` as a regression
  // test.
  TensorView* out =
      reshape(in, {IrBuilder::create<Val>(40), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), nullptr);
}

TEST_F(AliasAnalysisTest, View_ForwardExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .ndims(3)
                       .dtype(DataType::Float)
                       .contiguity({true, true, std::nullopt})
                       .shape({4, 5, 6})
                       .expanded({false, false, true})
                       .build();
  fusion.addInput(in);
  TensorView* out = reshape(in, {4, 5, 6}, {20, -1});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);

  // Verify the last dimension isn't expanded physically.
  KernelExecutor ke;
  at::Tensor in_tensor =
      at::randn({4, 5}).cuda().as_strided({4, 5, 6}, {5, 1, 0});
  ke.compile(&fusion, {in_tensor});
  at::Tensor out_tensor = ke.run({in_tensor})[0].as<at::Tensor>();

  EXPECT_THAT(out_tensor.strides(), ElementsAre(1, 0));
}

TEST_F(AliasAnalysisTest, View_MergeExpandedBroadcast) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = TensorViewBuilder()
                       .ndims(3)
                       .dtype(DataType::Float)
                       .contiguity({true, true, std::nullopt})
                       .shape({4, 5, 6})
                       .expanded({false, false, true})
                       .build();
  fusion.addInput(in);
  TensorView* out = reshape(in, {4, 5, 6}, {4, -1});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), nullptr);
}

TEST_F(AliasAnalysisTest, TrivialSlice) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0}, {2, 3});
  out = reshape(out, {2, 3}, {6});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);
}

TEST_F(AliasAnalysisTest, MergeTriviallySlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {2, 2, 5});
  out = reshape(out, {2, 2, 5}, {2, 10});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);
}

TEST_F(AliasAnalysisTest, MergeSlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* slice_out = slice(in, {0, 0, 0}, {2, 2, 5});
  TensorView* out = reshape(slice_out, {2, 2, 5}, {4, 5});
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), nullptr);
}

TEST_F(AliasAnalysisTest, BroadcastExpandDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3});
  fusion.addInput(in);
  TensorView* broadcast_tv = broadcast(in, {false, true, false});
  TensorView* expanded_tv = expand(
      broadcast_tv,
      {broadcast_tv->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       broadcast_tv->axis(2)->extent()});
  fusion.addOutput(expanded_tv);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(expanded_tv), in);
}

TEST_F(AliasAnalysisTest, NoAliasForReshardingExprs) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int kNumDevices = 4;
  auto mesh = DeviceMesh::createForNumDevices(kNumDevices);

  TensorView* in = makeContigTensor(2);
  TensorView* out = set(in);

  in->setDeviceMesh(mesh);
  in->axis(0)->parallelize(ParallelType::DIDx);
  out->setDeviceMesh(mesh);

  fusion.addInput(in);
  fusion.addOutput(out);

  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_TRUE(analysis.getRoot(out) == nullptr);
}

} // namespace nvfuser
