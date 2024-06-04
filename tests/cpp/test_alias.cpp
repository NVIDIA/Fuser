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

#include <alias_analysis.h>
#include <fusion.h>
#include <fusion_profiler.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <sys_utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::_;
using testing::Contains;
using testing::ContainsRegex;
using testing::Each;
using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::IsTrue;
using testing::Not;
using testing::Optional;
using testing::Pair;
using testing::UnorderedElementsAre;

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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);
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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);
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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);
  Layout preferred_layout = alias_analysis.preferredLayout(out);
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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), nullptr);
}

TEST_F(AliasAnalysisTest, Set) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* out = set(in);
  fusion.addOutput(out);

  in->setAllocationDomain({in->axis(1), in->axis(2), in->axis(0)}, true);

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);

  const std::vector<IterDomain*>& out_rfactor = out->getRFactorDomain();
  EXPECT_THAT(
      alias_analysis.preferredLayout(out).allocation_domain,
      ElementsAre(out_rfactor[1], out_rfactor[2], out_rfactor[0]));
}

TEST_F(AliasAnalysisTest, Permute) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion.addInput(in);
  TensorView* out = permute(in, {1, 2, 0});
  fusion.addOutput(out);

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);

  const std::vector<IterDomain*>& out_rfactor = out->getRFactorDomain();
  EXPECT_THAT(
      alias_analysis.preferredLayout(out).allocation_domain,
      ElementsAre(out_rfactor[2], out_rfactor[0], out_rfactor[1]));
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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), nullptr);
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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);

  // Verify the last dimension isn't expanded physically.
  FusionExecutor fe;
  at::Tensor in_tensor =
      at::randn({4, 5}).cuda().as_strided({4, 5, 6}, {5, 1, 0});
  fe.compileFusion(&fusion, {in_tensor});
  at::Tensor out_tensor = fe.runFusion({in_tensor})[0];

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

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), nullptr);
}

TEST_F(AliasAnalysisTest, TrivialSlice) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0}, {2, 3});
  out = reshape(out, {2, 3}, {6});
  fusion.addOutput(out);

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);
}

TEST_F(AliasAnalysisTest, MergeTriviallySlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {2, 2, 5});
  out = reshape(out, {2, 2, 5}, {2, 10});
  fusion.addOutput(out);

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), in);
}

TEST_F(AliasAnalysisTest, MergeSlicedDimensions) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion.addInput(in);
  TensorView* slice_out = slice(in, {0, 0, 0}, {2, 2, 5});
  TensorView* out = reshape(slice_out, {2, 2, 5}, {4, 5});
  fusion.addOutput(out);

  AliasAnalysisResult alias_analysis = findAliases(&fusion);
  EXPECT_EQ(alias_analysis.getNearestAliasedIo(out), nullptr);
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

TEST_F(AliasTest, View_AliasForSameLayout) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion->addOutput(out);

  in->setAllocationDomain(
      {in->axis(1), in->axis(2), in->axis(0)}, {true, false, false});
  out->setAllocationDomain({out->axis(1), out->axis(0)}, false);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({60}).cuda().as_strided({2, 3, 4}, {2, 20, 5});
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, View_AliasForCompliantLayout) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({2, 12});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = reshape(in, in_shape, out_shape);
  fusion->addOutput(out);

  out->setAllocationDomain({out->axis(0), out->axis(1)}, {false, false});

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 4}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, View_NoAliasForIncompliantLayout) {
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
  EXPECT_FALSE(fec.getMostRecentKernelRuntime()->isSegmented())
      << "segmentation is not supposed to happen";

  at::Tensor expected_out_tensor = in_tensor.add(3.141);
  // Verify output values.
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
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
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.strides(), ElementsAre(42, 7, _));

  testValidate(
      fec.fusion(),
      {in_tensor.slice(/*dim=*/2, /*start=*/c10::nullopt, /*end=*/1)},
      {in_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, SliceRightOfBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 1, 7});
  fusion->addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {4, 1, 5});
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({4, 1, 7}).cuda();
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.strides(), ElementsAre(7, _, 1));

  testValidate(
      fec.fusion(),
      {in_tensor.slice(/*dim=*/2, /*start=*/c10::nullopt, /*end=*/5)},
      {in_tensor},
      __LINE__,
      __FILE__);
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
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* mid = add(in, IrBuilder::create<Val>(3.141));
  mid = segment_set(mid);
  TensorView* out = mul(mid, IrBuilder::create<Val>(2.0));

  fusion->addOutput(mid);
  fusion->addOutput(mid);
  fusion->addOutput(out);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  // Verify aliasing among duplicated outputs
  EXPECT_TRUE(out_tensors[0].is_alias_of(out_tensors[1]));
  EXPECT_TRUE(out_tensors[2].is_alias_of(out_tensors[3]));

  // Verify segmentation
  EXPECT_EQ(
      fec.getMostRecentKernelRuntime()->fusionSegments()->groups().size(), 2)
      << "segmentation didn't happen as expected";
}

namespace {

// Returns the only executor in the most recent runtime.
const FusionExecutor& onlyExecutorInMostRecentRuntime(
    const FusionExecutorCache& fec) {
  const std::vector<FusionExecutor>& executors =
      fec.getMostRecentKernelRuntime()->executors();
  EXPECT_EQ(executors.size(), 1);
  return executors.front();
}

void expectKernelDoesNotStoreToOutput(
    const FusionExecutor& executor,
    const int64_t out_index) {
  // Get the variable name from the `kir::Kernel` not the input fusion, because
  // they are not always the same.
  std::string var_name =
      ir_utils::varName(executor.kernel()->outputs()[out_index]);
  EXPECT_THAT(
      executor.kernelString(), Not(ContainsRegex(R"(\b)" + var_name + R"(\[)")))
      << "The generated CUDA kernel shouldn't store data to `" << var_name
      << "`:" << executor.kernelString();
}
} // namespace

TEST_F(AliasTest, NotAllOutputsAlias_Pointwise) {
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
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor slice_out_tensor = out_tensors[0];
  EXPECT_TRUE(slice_out_tensor.is_alias_of(in_tensor));

  expectKernelDoesNotStoreToOutput(
      onlyExecutorInMostRecentRuntime(fec), /*out_index=*/0);
}

TEST_F(AliasTest, NotAllOutputsAlias_Reduction) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({16, 12, 128, 192});
  in->setAllocationDomain(
      {in->axis(0), in->axis(2), in->axis(1), in->axis(3)}, true);
  TensorView* dqkv = permute(in, {0, 2, 1, 3});
  dqkv = reshape(dqkv, {16, 128, 12, 192}, {16, 128, 2304});
  TensorView* sum_out = sum(dqkv, {0, 1});
  TensorView* view_out = reshape(dqkv, {16, 128, 2304}, {2048, 2304});
  TensorView* permute_out = permute(view_out, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(sum_out);
  fusion->addOutput(view_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({16 * 12 * 128 * 192})
          .cuda()
          .as_strided({16, 12, 128, 192}, {128 * 12 * 192, 192, 12 * 192, 1});
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[1].is_alias_of(in_tensor));
  EXPECT_TRUE(out_tensors[2].is_alias_of(in_tensor));
}

TEST_F(AliasTest, Issue1452) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Large enough to trigger vectorization.
  TensorView* in = makeContigConcreteTensor({1024, 1024});
  TensorView* set_out = set(in);
  TensorView* add_out = add(in, fusion->oneVal());
  fusion->addInput(in);
  fusion->addOutput(set_out);
  fusion->addOutput(add_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({1024, 1024}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor set_out_tensor = out_tensors[0];
  EXPECT_TRUE(set_out_tensor.is_alias_of(in_tensor));

  expectKernelDoesNotStoreToOutput(
      onlyExecutorInMostRecentRuntime(fec), /*out_index=*/0);
}

TEST_F(AliasTest, AliasOutputBeforeNonAliasOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* slice_out = slice(in, {0, 0}, {2, 2});
  TensorView* add_out = add(slice_out, slice_out);
  fusion->addInput(in);
  fusion->addOutput(slice_out);
  fusion->addOutput(add_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();

  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor slice_out_tensor = out_tensors[0];
  EXPECT_TRUE(slice_out_tensor.is_alias_of(in_tensor));

  expectKernelDoesNotStoreToOutput(
      onlyExecutorInMostRecentRuntime(fec), /*out_index=*/0);
}

TEST_F(AliasTest, Set_NoAliasForIncompatibleLayout) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion->addInput(in);
  TensorView* out = set(in);
  fusion->addOutput(out);

  // I intentionally set the allocation order to be different to block aliasing.
  out->setAllocationDomain({out->axis(1), out->axis(2), out->axis(0)}, true);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0];

  // Verify `out_tensor` is not an alias of `in_tensor`.
  EXPECT_FALSE(out_tensor.is_alias_of(in_tensor));
}

// Verifying that duplicated outputs are properly alised
TEST_F(AliasTest, DuplicateOutputsComplex) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion->addInput(in);
  TensorView* out = add(in, IrBuilder::create<Val>(5.0));
  fusion->addOutput(out);
  // duplicated output
  fusion->addOutput(out);
  TensorView* out1 = add(in, IrBuilder::create<Val>(1.0));
  fusion->addOutput(out1);
  // duplicated output
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();

  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 4);

  // Verify aliases among outputs.
  EXPECT_TRUE(out_tensors[0].is_alias_of(out_tensors[1]));
  EXPECT_FALSE(out_tensors[0].is_alias_of(out_tensors[2]));
  EXPECT_TRUE(out_tensors[0].is_alias_of(out_tensors[3]));

  // Verify output values.
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

// test verifying that duplicated input is not allowed in nvfuser
TEST_F(AliasTest, DuplicateInputs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  fusion->addInput(in);

  // duplicated input is not allowed
  EXPECT_THAT(
      [&]() { fusion->addInput(in); },
      testing::ThrowsMessage<nvfuser::nvfError>(
          testing::HasSubstr("duplicated inputs is not allowed")));
}

TEST_F(AliasTest, AliasInSegment) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // The segment between `permute_in` and `permute_out` is meta-op only and
  // turned into a no-op kernel.
  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* add_out = add(in, in);
  TensorView* permute_in = segment_set(in);
  TensorView* permute_out = permute(permute_in, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(add_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[1].is_alias_of(in_tensor));
}

TEST_F(AliasTest, TrivialInputForwarding) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeConcreteTensor({-1, -1});
  TensorView* tv1 = makeConcreteTensor({-1, -1});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  // Note: output of add is not used. Kept it here since previously there was an
  // assertion from sorting in codegen.
  add(tv1, IrBuilder::create<Val>(3.141));
  fusion->addOutput(tv0);

  at::Tensor t0 = at::randn({10, 4}).cuda();
  at::Tensor t1 = at::randn({10, 4}).cuda();

  FusionExecutorCache fec(std::move(fusion));
  std::vector<at::Tensor> cg_outputs = fec.runFusionWithInputs({t0, t1});

  EXPECT_EQ(cg_outputs[0].data_ptr(), t0.data_ptr());
  testValidate(fec.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  EXPECT_TRUE(fec.isCompiled({t0, t1}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0, t1});
  EXPECT_EQ(cg_outputs2[0].data_ptr(), t0.data_ptr());
  testValidate(fec.fusion(), cg_outputs2, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(AliasTest, TrivialInputForwarding_ScalarTensor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(0);
  fusion->addInput(tv0);
  fusion->addOutput(tv0);

  at::Tensor t0 = at::randn({}).cuda();

  FusionExecutorCache fec(std::move(fusion));
  auto cg_outputs = fec.runFusionWithInputs({t0});
  EXPECT_EQ(cg_outputs[0].data_ptr(), t0.data_ptr());
  testValidate(fec.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  EXPECT_TRUE(fec.isCompiled({t0}));
  auto cg_outputs2 = fec.runFusionWithInputs({t0});
  EXPECT_EQ(cg_outputs2[0].data_ptr(), t0.data_ptr());
  testValidate(fec.fusion(), cg_outputs2, {t0}, __LINE__, __FILE__);
}

TEST_F(AliasTest, OutputAliasesAnotherOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* add_out = add(in, in);
  TensorView* reshape_out = reshape(add_out, {2, 3, 5}, {6, 5});
  TensorView* permute_out = permute(reshape_out, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(reshape_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  ASSERT_EQ(out_tensors.size(), 2);
  at::Tensor reshape_out_tensor = out_tensors[0];
  at::Tensor permute_out_tensor = out_tensors[1];
  EXPECT_TRUE(permute_out_tensor.is_alias_of(reshape_out_tensor));
}

TEST_F(AliasTest, OutputNotAliasedByAnotherOutputShouldNotBeSegmented) {
  // Reproduces #1646.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* add_out = add(in, in);
  TensorView* reshape_out = reshape(add_out, {2, 3, 5}, {6, 5});
  TensorView* permute_out = permute(reshape_out, {1, 0});
  TensorView* mul_out = mul(permute_out, permute_out);

  fusion->addInput(in);
  fusion->addOutput(reshape_out);
  fusion->addOutput(mul_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());
}

TEST_F(AliasTest, ManyAliasesBetweenOutputs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* add_out = add(in, in);
  TensorView* permute_out = permute(add_out, {1, 2, 0});
  TensorView* reshape_out = reshape(permute_out, {3, 5, 2}, {15, 2});
  TensorView* slice_out = slice(permute_out, {0, 0, 0}, {2, 4, 1});

  fusion->addInput(in);
  // I intentionally add the outputs in reverse order to execise sorting in
  // `allocateOutputs`.
  fusion->addOutput(slice_out);
  fusion->addOutput(reshape_out);
  fusion->addOutput(permute_out);
  fusion->addOutput(add_out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
  ASSERT_EQ(out_tensors.size(), 4);
  at::Tensor slice_out_tensor = out_tensors[0];
  at::Tensor reshape_out_tensor = out_tensors[1];
  at::Tensor permute_out_tensor = out_tensors[2];
  at::Tensor add_out_tensor = out_tensors[3];

  EXPECT_EQ(add_out_tensor.data_ptr(), slice_out_tensor.data_ptr());
  EXPECT_EQ(add_out_tensor.data_ptr(), reshape_out_tensor.data_ptr());
  EXPECT_EQ(add_out_tensor.data_ptr(), permute_out_tensor.data_ptr());

  // Segment 1: in -> add_out
  // Segment 2: add_out -> its output aliases
  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);
}

TEST_F(AliasTest, Broadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, -1});
  TensorView* out = broadcast(in, {false, true, false});
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensor.data_ptr(), in_tensor.data_ptr());
}

TEST_F(AliasTest, MergeTwoExpandedBroadcasts) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .ndims(3)
                       .dtype(DataType::Float)
                       .contiguity({std::nullopt, std::nullopt, std::nullopt})
                       .shape({4, 5, 6})
                       .expanded({true, true, true})
                       .build();
  fusion->addInput(in);
  TensorView* out = reshape(in, {4, 5, 6}, {20, -1});
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({1}).cuda().as_strided({4, 5, 6}, {0, 0, 0});
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  // TODO(#1126): This should become an alias when #1126 is fixed.
  // EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, MergeBroadcastsBetweenConcretes) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .ndims(4)
                       .dtype(DataType::Float)
                       .contiguity({true, std::nullopt, std::nullopt, true})
                       .shape({2, 3, 5, 7})
                       .expanded({false, true, true, false})
                       .build();
  fusion->addInput(in);
  TensorView* out = reshape(in, {2, 3, 5, 7}, {2, -1, 7});
  out = reshape(out, {2, 15, 7}, {30, 7});
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2 * 7}).cuda().as_strided({2, 3, 5, 7}, {7, 0, 0, 1});
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, Squeeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, 1, -1});
  TensorView* out = squeeze(in, std::vector<bool>({false, true, false}));
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 1, 3}).cuda();
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensor.data_ptr(), in_tensor.data_ptr());
}

TEST_F(AliasTest, SourceIsBothInputAndOutput) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = permute(in, {1, 0});
  fusion->addInput(in);
  fusion->addOutput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  std::vector<at::Tensor> out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(in_tensor.data_ptr(), out_tensors[0].data_ptr());
  EXPECT_EQ(in_tensor.data_ptr(), out_tensors[1].data_ptr());
}

MATCHER_P(HeuristicIs, heuristic, "") {
  return arg->heuristic() == heuristic;
}

TEST_F(AliasTest, SegmentBoundary) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = permute(in, {1, 0});
  // With the current segmentation algorithm, `slice` has to be the start of a
  // fusion. So we expect `permute` to form a meta-op-only segment and the rest
  // a pointwise segment.
  out = slice(out, {0, 0}, {2, 2});
  out = add(out, out);
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  at::Tensor out_tensor = fec.runFusionWithInputs({in_tensor})[0];
  testValidate(fec.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(ScheduleHeuristic::NoOp),
          HeuristicIs(ScheduleHeuristic::PointWise)));
}

TEST_F(AliasTest, ReuseBufferAliasAcrossSegments) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(1.0)); // Group 0
  TensorView* tv4 =
      max(tv3, {0}); // Group 0 (use max instead to avoid numerical issues)
  TensorView* tv5 = add(tv4, tv1); //  Group 0 (Non Broadcast after reduce,
                                   //  keeps normalization scheduler away)
  TensorView* tv6 = add(tv5, tv2); //  Group 1 (Broadcast after reduce)

  // Note: test alias;
  fusion->aliasOutputToInput(tv6, tv0, AllocationType::ReuseBuffer);
  // TODO: support output on aliased fusion #1488
  // remove tv7 after #1488
  // fusion->addOutput(tv6);
  TensorView* tv7 = add(tv6, IrBuilder::create<Val>(1.0)); // Group 0
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  at::Tensor t1 = at::randn({65}, options);
  at::Tensor t2 = at::randn({128, 65}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  // Make a copy of `t0` because `t0` will be in-place updated.
  at::Tensor original_t0 = t0.clone();
  std::vector<at::Tensor> outputs =
      executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(),
      outputs,
      {original_t0, t1, t2},
      __LINE__,
      __FILE__);

  EXPECT_EQ(
      executor_cache.getMostRecentKernelRuntime()
          ->fusionSegments()
          ->groups()
          .size(),
      2)
      << "segmentation didn't happen as expected";

  auto t3 = original_t0.add(1.0);
  auto t4 = std::get<0>(at::max(t3, 0));
  auto t5 = t4.add(t1);
  auto t6 = t5.add(t2);
  EXPECT_TRUE(t0.allclose(t6))
      << "`t0` should have been in-place updated to the same value as `t6`.";
}

TEST_F(AliasTest, AliasOnlyKernelsAreNotLaunched) {
  if (detectComputeSanitizer()) {
    GTEST_SKIP()
        << "Skipped because compute-sanitizer is detected, which conflicts with FusionProfiler";
  }

  ProfilerOptionsGuard options_guard;
  ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
  FusionProfiler::start();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // The segment between `add_out` and `permute_out` is meta-op only and
  // turned into a no-op kernel.
  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* add_out = add(in, in);
  TensorView* permute_out = permute(add_out, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(add_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  const FusionProfile& profile = FusionProfiler::profile();
  // Expect a kernel launched for one of the two segments but not the
  // other.
  EXPECT_THAT(
      profile.kernel_profiles,
      UnorderedElementsAre(
          Field(&KernelProfile::name, IsEmpty()),
          Field(&KernelProfile::name, Not(IsEmpty()))));

  if (ProfilerState::Running == FusionProfiler::state()) {
    FusionProfiler::stop();
  }
}

TEST_F(AliasTest, PerfDebugVerboseWhenSomeKernelsNotLaunched) {
  // A reproducer for #1943.
  DebugDumpOptionsGuard options_guard;
  DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::PerfDebugVerbose);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // The segment between `add_out` and `permute_out` is meta-op only and
  // turned into a no-op kernel.
  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* add_out = add(in, in);
  TensorView* permute_out = permute(add_out, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(add_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = fec.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(ScheduleHeuristic::NoOp),
          HeuristicIs(ScheduleHeuristic::PointWise)));
}

TEST_F(AliasTest, NoKernelsAreLaunched) {
  if (detectComputeSanitizer()) {
    GTEST_SKIP()
        << "Skipped because compute-sanitizer is detected, which conflicts with FusionProfiler";
  }

  ProfilerOptionsGuard option_guard;
  ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);
  FusionProfiler::start();

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = permute(in, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  fec.runFusionWithInputs({in_tensor});

  const FusionProfile& profile = FusionProfiler::profile();
  // Expect a kernel launched for one of the two segments but not the
  // other.
  EXPECT_THAT(
      profile.kernel_profiles,
      UnorderedElementsAre(Field(&KernelProfile::name, IsEmpty())));

  if (ProfilerState::Running == FusionProfiler::state()) {
    FusionProfiler::stop();
  }
}

} // namespace nvfuser
