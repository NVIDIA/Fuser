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
using testing::Not;
using testing::Pair;
using testing::SizeIs;
using testing::UnorderedElementsAre;

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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  // Verify output values.
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({60}).cuda().as_strided({2, 3, 4}, {2, 20, 5});
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 4}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify `out_tensor` is not an alias of `in_tensor`.
  EXPECT_FALSE(out_tensor.is_alias_of(in_tensor));

  // Verify output values.
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2, 3, 4}, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

  // Verify aliasing.
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());

  // Verify output values.
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 2);
  at::Tensor out_tensor_0 = out_tensors[0].as<at::Tensor>();
  at::Tensor out_tensor_1 = out_tensors[1].as<at::Tensor>();

  // Verify aliasing among duplicated outputs
  EXPECT_TRUE(out_tensor_0.is_alias_of(out_tensor_1));
  // Verify no segmentation
  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented())
      << "segmentation is not supposed to happen";

  at::Tensor expected_out_tensor = in_tensor.add(3.141);
  // Verify output values.
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, SliceToSizeOne_Issue1353) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({4, 6, 7});
  fusion->addInput(in);
  TensorView* out = slice(in, {0, 0, 0}, {4, 6, 1});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({4, 6, 7}).cuda();
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.strides(), ElementsAre(42, 7, _));

  testValidate(
      executor_cache.fusion(),
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({4, 1, 7}).cuda();
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  EXPECT_EQ(in_tensor.data_ptr(), out_tensor.data_ptr());
  EXPECT_THAT(out_tensor.strides(), ElementsAre(7, _, 1));

  testValidate(
      executor_cache.fusion(),
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
  std::vector<TensorView*> splits = chunk(in, /*chunks=*/3, /*dim=*/-1);
  for (TensorView* split : splits) {
    split = reshape(
        split,
        {batches, seq_length, features},
        {batches, seq_length, heads, features / heads});
    split = permute(split, {0, 2, 1, 3});
    fusion->addOutput(split);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({batches, seq_length, features * 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  EXPECT_EQ(out_tensors.size(), 3);

  for (const auto& out_tensor : out_tensors) {
    EXPECT_TRUE(out_tensor.as<at::Tensor>().is_alias_of(in_tensor));
  }

  std::vector<at::Tensor> expected_out_tensors =
      in_tensor.split(/*split_size=*/features, /*dim=*/-1);
  for (auto& expected_out_tensor : expected_out_tensors) {
    expected_out_tensor =
        expected_out_tensor.view({batches, seq_length, heads, -1})
            .permute({0, 2, 1, 3});
  }

  testValidate(
      executor_cache.fusion(),
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  // Verify aliasing among duplicated outputs
  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(
      out_tensors[1].as<at::Tensor>()));
  EXPECT_TRUE(out_tensors[2].as<at::Tensor>().is_alias_of(
      out_tensors[3].as<at::Tensor>()));

  // Verify segmentation
  EXPECT_EQ(
      executor_cache.getMostRecentKernelRuntime()
          ->fusionSegments()
          ->groups()
          .size(),
      2)
      << "segmentation didn't happen as expected";
}

namespace {

bool storesToOutput(const KernelExecutor* ke, const int64_t out_index) {
  // Get the variable name from the `kir::Kernel` not the input fusion, because
  // they are not always the same.
  std::string var_name =
      ir_utils::varName(ke->compiledKernel()->kernel()->outputs()[out_index]);
  std::regex store_to_output(R"(\b)" + var_name + R"(\[)");
  return std::regex_search(
      ke->compiledKernel()->kernelString(), store_to_output);
}

} // namespace

TEST_F(AliasTest, NotAllOutputsAlias_Pointwise) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* t = reshape(in, {2, 3}, {6});
  TensorView* broadcast_out = broadcast(t, {true, false});
  broadcast_out = expand(
      broadcast_out,
      {IrBuilder::create<Val>(5), broadcast_out->axis(1)->extent()});
  TensorView* add_out = add(t, t);

  fusion->addInput(in);
  fusion->addOutput(broadcast_out);
  fusion->addOutput(add_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto cg_outputs = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), cg_outputs, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(cg_outputs[0].as<at::Tensor>().is_alias_of(in_tensor));

  // Expect two segments:
  //
  // - reshape & add. This segment has two outputs, corresponding to the
  // reshape and the add. It's expected to be compiled to a pointwise kernel
  // that stores only to the output of the add.
  //
  // - broadcast & expand. This segment is meta-op only.
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::PointWise)));

  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      const auto* ke =
          runtime->executors().at(group->groupId())->as<KernelExecutor>();
      int num_stores = 0;
      for (auto i : arange(group->outputs().size())) {
        if (storesToOutput(ke, i)) {
          num_stores++;
        }
      }
      EXPECT_EQ(num_stores, 1) << "The generated CUDA kernel is expected to "
                                  "store data to one output:"
                               << std::endl
                               << ke->compiledKernel()->kernelString();
    }
  }
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({16 * 12 * 128 * 192})
          .cuda()
          .as_strided({16, 12, 128, 192}, {128 * 12 * 192, 192, 12 * 192, 1});
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(in_tensor));
  EXPECT_TRUE(out_tensors[2].as<at::Tensor>().is_alias_of(in_tensor));
}

TEST_F(AliasTest, Issue1452) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Large enough to trigger vectorization.
  TensorView* in = makeContigConcreteTensor({1024, 1024});
  TensorView* t = reshape(in, {1024, 1024}, {1024 * 1024});
  TensorView* set_out = set(t);
  TensorView* add_out = add(t, fusion->oneVal());
  fusion->addInput(in);
  fusion->addOutput(set_out);
  fusion->addOutput(add_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({1024, 1024}).cuda();
  auto cg_outputs = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), cg_outputs, {in_tensor}, __LINE__, __FILE__);

  at::Tensor set_out_tensor = cg_outputs[0].as<at::Tensor>();
  EXPECT_TRUE(set_out_tensor.is_alias_of(in_tensor));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::PointWise)));

  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      const auto& ke =
          runtime->executors().at(group->groupId())->as<KernelExecutor>();
      int num_stores = 0;
      for (auto i : arange(group->outputs().size())) {
        if (storesToOutput(ke, i)) {
          num_stores++;
        }
      }
      EXPECT_EQ(num_stores, 1) << "The generated CUDA kernel is expected to "
                                  "store data to one output:"
                               << std::endl
                               << ke->compiledKernel()->kernelString();
    }
  }
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();

  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor slice_out_tensor = out_tensors[0].as<at::Tensor>();
  EXPECT_TRUE(slice_out_tensor.is_alias_of(in_tensor));

  const auto* ke = onlyKernelExecutorInMostRecentRuntime(executor_cache);
  EXPECT_FALSE(storesToOutput(ke, /*out_index=*/0))
      << "The generated CUDA kernel shouldn't store data to output 0:"
      << std::endl
      << ke->compiledKernel()->kernelString();
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);
  at::Tensor out_tensor = out_tensors[0].as<at::Tensor>();

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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();

  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 4);

  // Verify aliases among outputs.
  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(
      out_tensors[1].as<at::Tensor>()));
  EXPECT_FALSE(out_tensors[0].as<at::Tensor>().is_alias_of(
      out_tensors[2].as<at::Tensor>()));
  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(
      out_tensors[3].as<at::Tensor>()));

  // Verify output values.
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(in_tensor));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  EXPECT_EQ(cg_outputs[0].as<at::Tensor>().data_ptr(), t0.data_ptr());
  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  EXPECT_TRUE(executor_cache.isCompiled({t0, t1}));
  auto cg_outputs2 = executor_cache.runFusionWithInputs({t0, t1});
  EXPECT_EQ(cg_outputs2[0].as<at::Tensor>().data_ptr(), t0.data_ptr());
  testValidate(
      executor_cache.fusion(), cg_outputs2, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(AliasTest, TrivialInputForwarding_ScalarTensor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(0);
  fusion->addInput(tv0);
  fusion->addOutput(tv0);

  at::Tensor t0 = at::randn({}).cuda();

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  EXPECT_EQ(cg_outputs[0].as<at::Tensor>().data_ptr(), t0.data_ptr());
  testValidate(executor_cache.fusion(), cg_outputs, {t0}, __LINE__, __FILE__);

  // Second run to ensure cache hit handles trivial forwarding properly
  EXPECT_TRUE(executor_cache.isCompiled({t0}));
  auto cg_outputs2 = executor_cache.runFusionWithInputs({t0});
  EXPECT_EQ(cg_outputs2[0].as<at::Tensor>().data_ptr(), t0.data_ptr());
  testValidate(executor_cache.fusion(), cg_outputs2, {t0}, __LINE__, __FILE__);
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  ASSERT_EQ(out_tensors.size(), 2);
  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(
      out_tensors[0].as<at::Tensor>()));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
  ASSERT_EQ(out_tensors.size(), 4);
  at::Tensor slice_out_tensor = out_tensors[0].as<at::Tensor>();
  at::Tensor reshape_out_tensor = out_tensors[1].as<at::Tensor>();
  at::Tensor permute_out_tensor = out_tensors[2].as<at::Tensor>();
  at::Tensor add_out_tensor = out_tensors[3].as<at::Tensor>();

  EXPECT_EQ(add_out_tensor.data_ptr(), slice_out_tensor.data_ptr());
  EXPECT_EQ(add_out_tensor.data_ptr(), reshape_out_tensor.data_ptr());
  EXPECT_EQ(add_out_tensor.data_ptr(), permute_out_tensor.data_ptr());

  // Segment 1: in -> add_out
  // Segment 2: add_out -> its output aliases
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_EQ(runtime->fusionSegments()->groups().size(), 2);
}

TEST_F(AliasTest, DoNotOverSegment_Straightline) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* add_out = add(in, in);
  TensorView* permute_out = permute(add_out, {1, 0});
  TensorView* mul_out = mul(permute_out, permute_out);

  fusion->addInput(in);
  fusion->addOutput(add_out);
  fusion->addOutput(permute_out);
  fusion->addOutput(mul_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_FALSE(runtime->isSegmented());

  // permute_out should be recognized as an alias of add_out. However, the
  // pointwise scheduler `cacheFork`s outputs
  // (https://github.com/NVIDIA/Fuser/blob/de488172d280d033511dfc0939e2207f87c1b1cb/csrc/scheduler/pointwise.cpp#L540),
  // making permute_out no longer a transitive user of add_out.
  // EXPECT_TRUE(out_tensors[1].is_alias_of(out_tensors[0]));
}

TEST_F(AliasTest, DoNotOverSegment_WithForks) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* t0 = add(in, in);
  TensorView* out0 = permute(t0, {1, 0});
  TensorView* t1 = reshape(t0, {2, 3}, {6});
  TensorView* out1 = reshape(t1, {6}, {3, 2});
  TensorView* out2 = mul(t1, t1);

  fusion->addInput(in);
  fusion->addOutput(out0);
  fusion->addOutput(out1);
  fusion->addOutput(out2);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::PointWise)).Times(1));

  // For a similar reason as DoNotOverSegment_Straightline.
  // EXPECT_TRUE(out_tensors[1].is_alias_of(out_tensors[0]));
}

TEST_F(AliasTest, Broadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, -1});
  TensorView* out = broadcast(in, {false, true, false});
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensor.data_ptr(), in_tensor.data_ptr());
}

TEST_F(AliasTest, Expand) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, -1});
  fusion->addInput(in);
  TensorView* broadcast_tv = broadcast(in, {false, true, false});
  TensorView* expanded_tv = expand(
      broadcast_tv,
      {broadcast_tv->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       broadcast_tv->axis(2)->extent()});
  fusion->addOutput(expanded_tv);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(out_tensor.data_ptr(), in_tensor.data_ptr());
}

TEST_F(AliasTest, MergeTwoExpandedBroadcasts) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({std::nullopt, std::nullopt, std::nullopt})
                       .shape({4, 5, 6})
                       .expanded({true, true, true})
                       .build();
  fusion->addInput(in);
  TensorView* out = reshape(in, {4, 5, 6}, {20, -1});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({1}).cuda().as_strided({4, 5, 6}, {0, 0, 0});
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

  // TODO(#1126): This should become an alias when #1126 is fixed.
  // EXPECT_TRUE(out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, MergeBroadcastsBetweenConcretes) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Float)
                       .contiguity({true, std::nullopt, std::nullopt, true})
                       .shape({2, 3, 5, 7})
                       .expanded({false, true, true, false})
                       .build();
  fusion->addInput(in);
  TensorView* out = reshape(in, {2, 3, 5, 7}, {2, -1, 7});
  out = reshape(out, {2, 15, 7}, {30, 7});
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({2 * 7}).cuda().as_strided({2, 3, 5, 7}, {7, 0, 0, 1});
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

TEST_F(AliasTest, Squeeze) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, 1, -1});
  TensorView* out = squeeze(in, std::vector<bool>({false, true, false}));
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 1, 3}).cuda();
  at::Tensor out_tensor =
      executor_cache.runFusionWithInputs({in_tensor})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);

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

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().is_alias_of(in_tensor));
  EXPECT_TRUE(out_tensors[1].as<at::Tensor>().is_alias_of(in_tensor));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      ElementsAre(HeuristicIs(SchedulerType::ExprEval)));
}

TEST_F(AliasTest, ReuseBuffer) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  TensorView* in = makeContigTensor(1);
  TensorView* out = add(in, IrBuilder::create<Val>(1.0));
  fusion->addInput(in);
  fusion->addOutput(out);
  fusion->aliasOutputToInput(out, in, AllocationType::ReuseBuffer);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto tensor = at::randn({10}, options);
  auto expected_tensor = tensor + 1.0;

  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.runFusionWithInputs({tensor});
  EXPECT_TRUE(tensor.allclose(expected_tensor));
}

TEST_F(AliasTest, ReuseBuffer_KernelExecutor) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* in = makeContigTensor(1);
  TensorView* out = add(in, IrBuilder::create<Val>(1.0));
  fusion.addInput(in);
  fusion.addOutput(out);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto tensor = at::randn({10}, options);
  auto expected_tensor = tensor + 1.0;

  KernelExecutor ke;
  ke.compile(&fusion, {tensor});
  ke.run({tensor}, {tensor});
  EXPECT_TRUE(tensor.allclose(expected_tensor));
}

TEST_F(AliasTest, ReuseBuffer_AliasAcrossSegments) {
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
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});
  testValidate(
      executor_cache.fusion(),
      outputs,
      {original_t0, t1, t2},
      __LINE__,
      __FILE__);

  // https://github.com/NVIDIA/Fuser/pull/2999 will cause 3 segments instead of
  // the optimal 2 segments. Change back to 2 segments once
  // https://github.com/NVIDIA/Fuser/issues/3251 is resolved.
  EXPECT_EQ(
      executor_cache.getMostRecentKernelRuntime()
          ->fusionSegments()
          ->groups()
          .size(),
      3)
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
    GTEST_SKIP() << "Skipped because compute-sanitizer is detected, which "
                    "conflicts with FusionProfiler";
  }

  ProfilerOptionsGuard options_guard;
  ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);

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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  EXPECT_EQ(FusionProfiler::state(), ProfilerState::Processed);
  ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::Enable);

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  const FusionProfile& profile = FusionProfiler::profile();
  // Expect a kernel launched for one of the two segments but not the
  // other.
  EXPECT_THAT(
      profile.kernel_profiles,
      UnorderedElementsAre(
          Field(&KernelProfile::name, IsEmpty()),
          Field(&KernelProfile::name, Not(IsEmpty()))));
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

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(AliasTest, NoKernelsAreLaunched) {
  if (detectComputeSanitizer()) {
    GTEST_SKIP() << "Skipped because compute-sanitizer is detected, which "
                    "conflicts with FusionProfiler";
  }

  ProfilerOptionsGuard option_guard;
  ProfilerOptionsGuard::getCurOptions().set(ProfilerOption::Enable);

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = permute(in, {1, 0});

  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({2, 3}, options);
  executor_cache.runFusionWithInputs({in_tensor});

  EXPECT_EQ(FusionProfiler::state(), ProfilerState::Processed);
  ProfilerOptionsGuard::getCurOptions().unset(ProfilerOption::Enable);

  const FusionProfile& profile = FusionProfiler::profile();
  // Expect a kernel launched for one of the two segments but not the
  // other.
  EXPECT_THAT(
      profile.kernel_profiles,
      UnorderedElementsAre(Field(&KernelProfile::name, IsEmpty())));
}

// While most use cases go through FusionExecutorCache, nvFuser also supports
// evaluating an alias via KernelExecutor.
TEST_F(AliasTest, KernelExecutor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigConcreteTensor({10, 10});
  TensorView* out = slice(in, {0, 0}, {5, 5});
  fusion.addInput(in);
  fusion.addOutput(out);

  // Sanity-check that `out` is a valid alias of `in`.
  AliasAnalysisResult analysis = findAliases(&fusion);
  EXPECT_EQ(analysis.getRoot(out), in);

  // Mark them alias so KernelExecutor::runFusion expression-evaluates the
  // output on the host instead of launching a CUDA kernel.
  fusion.aliasOutputToInput(out, in, AllocationType::Evaluate);

  ExprEvalExecutor ee;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 10}, options);
  ee.compile(&fusion);
  KernelArgumentHolder args({t0});
  at::Tensor out_tensor = ee.run(args)[0].as<at::Tensor>();
  EXPECT_EQ(out_tensor.data_ptr(), t0.data_ptr());
}

TEST_F(AliasTest, InplaceUpdate) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = makeContigConcreteTensor({2, 3});
  fusion->addInput(in);
  fusion->addInput(out);
  fusion->aliasOutputToInput(out, in, AllocationType::ReuseBuffer);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  at::Tensor out_tensor = in_tensor + 1;
  executor_cache.runFusionWithInputs({in_tensor, out_tensor});
  EXPECT_TRUE(out_tensor.equal(in_tensor));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::PointWise)));
}

TEST_F(AliasTest, Bookend_SegmentSetPreservesAllocation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* permute_out = permute(in, {1, 0});
  TensorView* compute_out = mul(in, in);
  fusion->addInput(in);
  fusion->addOutput(permute_out);
  fusion->addOutput(compute_out);

  in->setAllocationDomain({in->axis(1), in->axis(0)}, true);
  permute_out->setAllocationDomain(
      {permute_out->axis(0), permute_out->axis(1)}, true);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({3, 2}).cuda().transpose(0, 1);
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor permute_out_tensor = out_tensors[0].as<at::Tensor>();
  EXPECT_TRUE(permute_out_tensor.is_alias_of(in_tensor));
}

TEST_F(AliasTest, Bookend_InputsAndOutputs) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* permute_out = permute(in, {1, 0});
  TensorView* compute_out = mul(in, in);
  compute_out = add(compute_out, in);
  fusion->addInput(in);
  fusion->addOutput(permute_out);
  fusion->addOutput(compute_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  at::Tensor permute_out_tensor = out_tensors[0].as<at::Tensor>();
  EXPECT_TRUE(permute_out_tensor.is_alias_of(in_tensor));

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  // MarkAliasesPrepare adds a `segment_set` between `in` and `permute`, which
  // leads to three segments:
  // 1. segment_set`, a no-op segment,
  // 2. permute`, a no-op segment,
  // 3. `mul` and `add`, a pointwise segment.
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::PointWise)));
  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      EXPECT_THAT(group->inputs(), SizeIs(1));
      EXPECT_THAT(group->outputs(), SizeIs(1));
    }
  }
}

TEST_F(AliasTest, Bookend_IntermediateTensors) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* out = permute(in, {1, 0});
  out = mul(out, out);
  out = reshape(out, {2, 3}, {6});
  fusion->addInput(in);
  fusion->addOutput(out);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::PointWise)));
  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      EXPECT_THAT(group->inputs(), SizeIs(1));
      EXPECT_THAT(group->outputs(), SizeIs(1));
    }
  }
}

TEST_F(AliasTest, Bookend_AliasesOfSameTensor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3});
  TensorView* t = add(in, in);
  TensorView* out0 = reshape(t, {2, 3}, {6});
  TensorView* out1 = reshape(t, {2, 3}, {3, 2});
  TensorView* out2 = transpose(t);

  fusion->addInput(in);
  fusion->addOutput(out0);
  fusion->addOutput(out1);
  fusion->addOutput(out2);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(
      out_tensors[0].as<at::Tensor>().data_ptr(),
      out_tensors[1].as<at::Tensor>().data_ptr());
  EXPECT_EQ(
      out_tensors[0].as<at::Tensor>().data_ptr(),
      out_tensors[2].as<at::Tensor>().data_ptr());

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::PointWise)).Times(1));
  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      EXPECT_THAT(group->inputs(), SizeIs(1));
      EXPECT_THAT(group->outputs(), SizeIs(1));
    }
  }
}

TEST_F(AliasTest, Bookend_ReuseSegmentSet) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({2, 3, 5});
  TensorView* t = add(in, in);
  // Expect the three reshapes to be placed in the same no-op segment.
  t = reshape(t, {2, 3, 5}, {30});
  TensorView* out0 = reshape(t, {30}, {6, 5});
  TensorView* out1 = reshape(t, {30}, {2, 15});

  fusion->addInput(in);
  fusion->addOutput(out0);
  fusion->addOutput(out1);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor = at::randn({2, 3, 5}).cuda();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_EQ(
      out_tensors[0].as<at::Tensor>().data_ptr(),
      out_tensors[1].as<at::Tensor>().data_ptr());

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::PointWise),
          HeuristicIs(SchedulerType::ExprEval)));
  for (SegmentedGroup* group : runtime->fusionSegments()->groups()) {
    if (group->schedulerType() == SchedulerType::PointWise) {
      EXPECT_THAT(group->inputs(), SizeIs(1));
      EXPECT_THAT(group->outputs(), SizeIs(1));
    }
  }
}

TEST_F(AliasTest, QKVSplitBackprop) {
  // A subgraph of MoveSplitCatTest.Cancellable_Issue1768.
  constexpr int b = 16;
  constexpr int h = 12;
  constexpr int s = 128;
  constexpr int f = 64;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* dq = makeContigConcreteTensor({b, s, h * f});
  TensorView* dk = makeContigConcreteTensor({b, s, h * f});
  TensorView* dv = makeContigConcreteTensor({b, s, h * f});
  TensorView* catted = cat({dq, dk, dv}, /*dim=*/-1);
  TensorView* sum_out = sum(catted, {0, 1});
  TensorView* view_out = reshape(catted, {b, s, h * f * 3}, {b * s, h * f * 3});
  TensorView* permute_out = permute(view_out, {1, 0});

  fusion->addInput(dq);
  fusion->addInput(dk);
  fusion->addInput(dv);
  fusion->addOutput(sum_out);
  fusion->addOutput(view_out);
  fusion->addOutput(permute_out);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args;
  for (int i = 0; i < 3; i++) {
    args.push(at::randn({b, s, h * f}).cuda());
  }
  auto out_tensors = executor_cache.runFusionWithInputs(args);
  testValidate(executor_cache.fusion(), out_tensors, args, __LINE__, __FILE__);

  EXPECT_TRUE(out_tensors[2].as<at::Tensor>().is_alias_of(
      out_tensors[1].as<at::Tensor>()));
}

TEST_F(AliasTest, Bookend_Issue2375) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t n = 2, c = 128, h = 16, w = 16, g = 32;
  const DataType dtype = DataType::Half;
  const std::vector<int64_t> input_shape = {n, g, c / g, h, w};
  const std::vector<int64_t> output_shape = {n, c, h, w};

  auto tv0 = makeContigTensor(input_shape.size(), dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {-1, -2, -3});
  auto tv4 = broadcast(tv3, {false, false, true, true, true});
  auto tv5 = div(tv2, tv4);
  auto tv6 = castOp(dtype, tv5);
  auto tv7 = reshape(tv6, input_shape, output_shape);
  fusion->addOutput(tv7);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({t0});
  testValidate(executor_cache.fusion(), out_tensors, {t0}, __LINE__, __FILE__);

  EXPECT_THAT(
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::ExprEval),
          HeuristicIs(SchedulerType::InnerPersistent)));
}

// Repro for https://github.com/NVIDIA/Fuser/issues/2664
TEST_F(AliasTest, Issue2664) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t n = 4194304;
  const DataType dtype = DataType::Float;
  const std::vector<int64_t> input_shape = {n};

  auto tv1 = makeContigTensor(1, dtype);
  auto tv2 = makeContigTensor(0, dtype);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto s3 = IrBuilder::create<Val>(1.0);
  auto tv4 = add(tv2, s3);
  auto tv5 = broadcast(tv4, {true});
  auto tv7 = expand(tv5, {tv1->axis(0)->extent()});
  auto tv8 = mul(tv1, tv7);
  fusion->aliasOutputToInput(tv4, tv2, AllocationType::ReuseBuffer);
  fusion->addOutput(tv8);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t1 = at::randn(input_shape, options);
  auto t2 = at::randn({}, options);
  auto aten_out = (t2 + 1.0) * t1;

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({t1, t2});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {t1, t2},
      {aten_out},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, TrivialInplaceUpdateNoSegmentation) {
  // testing a complete fusion
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* out = add(in, IrBuilder::create<Val>(3.141));
  fusion->addOutput(out);
  // this is an inplace update and shouldn't be segmented into its own kernel by
  // alias analysis
  TensorView* update_input = set(out);
  fusion->aliasOutputToInput(update_input, in, AllocationType::ReuseBuffer);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor original_tensor = in_tensor.clone();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);

  // Verify inplace update
  EXPECT_TRUE(out_tensors[0].as<at::Tensor>().equal(in_tensor));
  // Verify no segmentation
  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

  // Verify output values.
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {original_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, ReshapeInplaceUpdateNoSegmentation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const std::vector<int64_t> in_shape({2, 3, 4});
  const std::vector<int64_t> out_shape({24});

  TensorView* in = makeContigConcreteTensor(in_shape);
  fusion->addInput(in);
  TensorView* r_in = reshape(in, in_shape, out_shape);
  TensorView* out = add(r_in, IrBuilder::create<Val>(3.141));
  fusion->addOutput(out);
  // this is an inplace update and shouldn't be segmented into its own kernel by
  // alias analysis
  TensorView* r_out = reshape(out, out_shape, in_shape);
  TensorView* update_input = set(r_out);
  fusion->aliasOutputToInput(update_input, in, AllocationType::ReuseBuffer);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn(in_shape, at::dtype(at::kFloat).device(at::kCUDA, 0));
  at::Tensor original_tensor = in_tensor.clone();
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  ASSERT_EQ(out_tensors.size(), 1);

  // Verify inplace update
  EXPECT_TRUE(out_tensors[0]
                  .as<at::Tensor>()
                  .as_strided({2, 3, 4}, {12, 4, 1})
                  .equal(in_tensor));

  // Verify no segmentation
  EXPECT_FALSE(executor_cache.getMostRecentKernelRuntime()->isSegmented());

  // Verify output values.
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {original_tensor},
      __LINE__,
      __FILE__);
}

TEST_F(AliasTest, FusionEmpty) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({10, 10, 10});
  auto tv1 = makeConcreteTensor({10, 10, 10});
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 10, 10}, options);
  at::Tensor t1 = at::randn({10, 10, 10}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  testValidate(
      executor_cache.fusion(), cg_outputs, {t0, t1}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(HeuristicIs(SchedulerType::ExprEval)));
}

TEST_F(AliasTest, IntermediateTensorWithAllocation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({15, 2});
  TensorView* x = transpose(in);
  x = segment_set(x);
  TensorView* out = reshape(x, {2, 15}, {2, 3, 5});
  fusion->addInput(in);
  fusion->addOutput(out);

  x->setAllocationDomain(x->getLogicalDomain(), true);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor in_tensor =
      at::randn({15, 2}, at::dtype(at::kFloat).device(at::kCUDA));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});

  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      runtime->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::PointWise),
          HeuristicIs(SchedulerType::ExprEval)));
}

TEST_F(AliasTest, SliceOfExpandedBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = TensorViewBuilder()
                       .ndims(2)
                       .dtype(DataType::Float)
                       .contiguity({true, std::nullopt})
                       .shape({2, 3})
                       .expanded({false, true})
                       .build();
  fusion->addInput(in);
  TensorView* out = slice(in, {0, 1}, {2, 3});
  fusion->addOutput(out);

  FusionExecutorCache fec(std::move(fusion));
  at::Tensor in_tensor = at::randn({2}).cuda().as_strided({2, 3}, {1, 0});
  auto out_tensors = fec.runFusionWithInputs({in_tensor});
  testValidate(fec.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);
}

} // namespace nvfuser
