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

using testing::Each;
using testing::ElementsAre;
using testing::IsEmpty;
using testing::IsTrue;
using testing::Optional;
using testing::Pair;

using AliasAnalysisTest = NVFuserTest;

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

TEST_F(AliasAnalysisTest, View_Contiguous) {
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
  optimization::Layout preferred_layout = alias_analysis.preferredLayout(out);
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

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);

  const std::vector<IterDomain*>& out_rfactor = out->getMaybeRFactorDomain();
  EXPECT_THAT(
      alias_analysis.preferredLayout(out).allocation_domain,
      ElementsAre(out_rfactor[2], out_rfactor[0], out_rfactor[1]));
}

TEST_F(AliasAnalysisTest, View_SplitExpandedBroadcast) {
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
  // tryStaticReshape used to fail to get the expanded extent, which is 6.
  TensorView* out = reshape(
      expand_out, {IrBuilder::create<Val>(40), IrBuilder::create<Val>(3)});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), expand_out);
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
  TensorView* broadcast_out = broadcast(in, {false, false, true});
  TensorView* expand_out = expand(
      broadcast_out,
      {IrBuilder::create<Val>(4),
       IrBuilder::create<Val>(5),
       IrBuilder::create<Val>(6)});
  TensorView* out = reshape(expand_out, {4, 5, 6}, {4, -1});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), expand_out);
}

TEST_F(AliasAnalysisTest, TrivialSlice) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0}, {2, 3});
  out = reshape(out, {2, 3}, {6});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);
}

TEST_F(AliasAnalysisTest, MergeTriviallySlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {2, 2, 5});
  out = reshape(out, {2, 2, 5}, {2, 10});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), in);
}

TEST_F(AliasAnalysisTest, MergeSlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* slice_out = slice(in, {0, 0, 0}, {2, 2, 5});
  TensorView* out = reshape(slice_out, {2, 2, 5}, {4, 5});
  fusion.addOutput(out);

  optimization::AliasAnalysisResult alias_analysis =
      optimization::findAliases(&fusion);
  EXPECT_EQ(alias_analysis.findRoot(out), out);
  EXPECT_EQ(alias_analysis.findRoot(slice_out), in);
}

using AliasTest = NVFuserTest;

TEST_F(AliasTest, View) {
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
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, View_NoAliasForIncompatibleLayout) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion->addOutput(out);

  // I intentionally set the allocation order to be column major to break the
  // alias.
  out->setAllocationDomain({out->axis(1), out->axis(0)}, true);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0];

  // Verify `out_tensor` is not an alias of `in_tensor`.
  EXPECT_FALSE(out_tensor.is_alias_of(in_tensor));

  // Verify output values.
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, ViewPermute) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  out = permute(out, {1, 0});
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
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, DuplicateOutputs) {
  // testing a complete fusion
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = add(in, IrBuilder::create<Val>(3.141));
  fusion->addOutput(out);
  fusion->addOutput(out); // duplicated outputs

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 2);
  at::Tensor out_tensor_0 = out_tensors[0];
  at::Tensor out_tensor_1 = out_tensors[1];

  // Verify aliasing among duplicated outputs
  EXPECT_TRUE(out_tensor_0.is_alias_of(out_tensor_1));
  // Verify no segmentation
  NVF_CHECK(
      !fec.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation is not supposed to happen");

  at::Tensor expected_out_tensor = in_tensor.add(3.141);
  // Verify output values.
  testValidate(
      fec.fusion(),
      {expected_out_tensor, expected_out_tensor},
      {in_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, SliceToSizeOne_Issue1353) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 6, 7});
  fusion->addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {4, 6, 1});
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({4, 6, 7}).cuda();
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, SliceViewPermute) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int batches = 16;
  constexpr int seq_length = 128;
  constexpr int features = 1024;
  constexpr int heads = 16;

  // The input tensor is a concatenation of [query, key, value], and therefore
  // has a feature dimension of size `features * 3`.
  TensorView* in =
      makeContigConcreteTensor({batches, seq_length, features * 3});
  fusion->addInput(in);
  std::vector<TensorView*> splits({
      slice(in, {0, 0, 0}, {batches, seq_length, features}),
      slice(in, {0, 0, features}, {batches, seq_length, features * 2}),
      slice(in, {0, 0, features * 2}, {batches, seq_length, features * 3}),
  });
  for (TensorView* split : splits) {
    split = reshape(
        split,
        {batches, seq_length, features},
        {batches, seq_length, heads, features / heads});
    split = permute(split, {0, 2, 1, 3});
    fusion->addOutput(split);
  }

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({batches, seq_length, features * 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  EXPECT_EQ(out_tensors.size(), 3);

  for (const auto& out_tensor : out_tensors) {
    EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
  }

  std::vector<at::Tensor> expected_out_tensors =
      in_tensor.split(/*split_size=*/features, /*dim=*/-1);
  for (auto& expected_out_tensor : expected_out_tensors) {
    expected_out_tensor =
        expected_out_tensor.view({batches, seq_length, heads, -1})
            .permute({0, 2, 1, 3});
  }

  testValidate(
      fec.fusion(),
      out_tensors,
      {in_tensor},
      expected_out_tensors,
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, DuplicateOutputsSegmentedFusion) {
  // testing duplicated output in segmented fusion
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* intermediate_tv = add(in, IrBuilder::create<Val>(3.141));
  TensorView* segment_tv = segment_set(intermediate_tv);
  TensorView* out = mul(segment_tv, IrBuilder::create<Val>(2.0));

  fusion->addOutput(intermediate_tv);
  fusion->addOutput(intermediate_tv);
  fusion->addOutput(out);
  fusion->addOutput(out); // duplicated outputs

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 4);
  at::Tensor out_tensor_0 = out_tensors[0];
  at::Tensor out_tensor_1 = out_tensors[1];
  at::Tensor out_tensor_2 = out_tensors[2];
  at::Tensor out_tensor_3 = out_tensors[3];

  // Verify aliasing among duplicated outputs
  EXPECT_TRUE(out_tensor_0.is_alias_of(out_tensor_1));
  EXPECT_TRUE(out_tensor_2.is_alias_of(out_tensor_3));
  // Verify segmentation
  NVF_CHECK(
      fec.getMostRecentKernelRuntime()->fusionSegments()->groups().size() == 2,
      "segmentation didn't happen as expected");

  at::Tensor intermediate_tensor = in_tensor.add(3.141);
  at::Tensor out_tensor = intermediate_tensor.mul(2.0);
  // Verify output values.
  testValidate(
      fec.fusion(),
      {intermediate_tensor, intermediate_tensor, out_tensor, out_tensor},
      {in_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, NotAllOutputsAlias) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* slice_out = slice(in, {0, 0}, {2, 2});
  TensorView* add_out = add(in, fusion->oneVal());
  fusion->addInput(in);
  fusion->addOutput(slice_out);
  fusion->addOutput(add_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});

  // As a known limitation, nvFuser still generates code to copy data from `in`
  // to `slice_out` despite the fact that `slice_out` is an alias.
  testValidate(
      fec.fusion(),
      out_tensors,
      {in_tensor},
      {in_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/2), in_tensor + 1.f},
      __LINE__,
      __FILE__);

  at::Tensor slice_out_tensor = out_tensors[0];
  EXPECT_TRUE(slice_out_tensor.is_alias_of(in_tensor));
}

} // namespace nvfuser
