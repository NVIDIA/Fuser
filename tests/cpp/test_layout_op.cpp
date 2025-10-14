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
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {
bool validateGroupedLayout(
    BlockScalingFactorLayout layout,
    at::Tensor out,
    at::Tensor ref,
    at::Tensor expert_offsets,
    at::Tensor sf_offsets) {
  NVF_ERROR(BlockScalingFactorLayout::Block128x4 == layout);
  int num_group = expert_offsets.size(0);

  // validate output logical shape
  EXPECT_EQ(out.sizes(), ref.sizes());

  // take length of reference for un-padded k size.
  int m = ref.size(0);
  int k = ref.size(1);
  int padded_k = (k + 4 - 1) / 4 * 4;
  int m_offset_last = sf_offsets[num_group - 1].item().to<int>();
  int m_last = m - expert_offsets[num_group - 1].item().to<int>();
  int padded_m = m_offset_last + (m_last + 127) / 128 * 128;

  out.as_strided_({padded_m, padded_k}, {padded_k, 1});

  // We validate each group individually
  for (int i = 0; i < num_group; ++i) {
    int start_idx = sf_offsets[i].item().to<int>();

    int m_g = (i + 1 < num_group ? expert_offsets[i + 1].item().to<int>() : m) -
        expert_offsets[i].item().to<int>();

    int padded_m_g = std::ceil(m_g / 128.0) * 128;

    auto out_g = out.slice(0, start_idx, start_idx + padded_m_g);

    int mn_tile = padded_m_g / 128;
    // ceil div in order to get padded k_tile size.
    int k_tile = std::ceil(k / 4.0);

    // view as {mn_tile, k_tile, m_4, mn_32, k_4}
    // restore the swizzle/padding on output.
    auto restored_out_g = out_g.view({mn_tile, k_tile, 32, 4, 4})
                              .transpose(1, 3)
                              .reshape({mn_tile * 4 * 32, k_tile * 4})
                              .slice(0, 0, m_g)
                              .slice(1, 0, k);
    auto ref_g = ref.slice(
        0,
        expert_offsets[i].item().to<int>(),
        expert_offsets[i].item().to<int>() + m_g);
    if (!at::allclose(restored_out_g, ref_g)) {
      std::cout << "failed at group: " << i << std::endl;
      std::cout << "out_g:\n" << out_g << std::endl;
      std::cout << "ref_g:\n" << ref_g << std::endl;
      std::cout << "restored_out_g:\n" << restored_out_g << std::endl;
      return false;
    }
  }
  return true;
}

} // namespace

using testing::UnorderedElementsAre;

class LayoutOpTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::IdModel, {"all"});
  }
};

TEST_F(LayoutOpTest, CppApi) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto out = preprocessGroupedMatmulInputSf(
      inp, offsets, rounded_offsets, BlockScalingFactorLayout::Block128x4);
  fusion.addOutput(out);
}

TEST_F(LayoutOpTest, ManualKernel) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto inp_tv = set(inp);
  auto out_tv = preprocessGroupedMatmulInputSf(
      inp_tv, offsets, rounded_offsets, BlockScalingFactorLayout::Block128x4);
  // NOTE: output of preprocessGroupedMatmulInputSf needs to be on global
  // memory, because we do indexing on output inside the runtime function.
  fusion.addOutput(out_tv);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 512;
  int k = 9; // note: padded column size would be 12
  // auto t0 = at::randn({m, k}, options);
  auto t0 = at::arange(m * k, options).reshape({m, k});
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384}, options.dtype(at::kInt));

  // naive scheduling.
  for (auto tv : {inp, inp_tv, out_tv}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2});
  auto outputs = ke.run({t0, t1, t2});

  ASSERT_TRUE(validateGroupedLayout(
      BlockScalingFactorLayout::Block128x4,
      outputs[0].as<at::Tensor>(),
      t0,
      t1,
      t2));
}

TEST_F(LayoutOpTest, SchedulerKernel) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto out_tv = preprocessGroupedMatmulInputSf(
      inp, offsets, rounded_offsets, BlockScalingFactorLayout::Block128x4);
  fusion.addOutput(out_tv);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 512;
  int k = 9; // note: padded column size would be 12
  auto t0 = at::randn({m, k}, options);
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384}, options.dtype(at::kInt));

  // running through automatic scheduler.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  ASSERT_TRUE(validateGroupedLayout(
      BlockScalingFactorLayout::Block128x4,
      outputs[0].as<at::Tensor>(),
      t0,
      t1,
      t2));
}

TEST_F(LayoutOpTest, SchedulerKernelWithConsumer) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto out_tv = preprocessGroupedMatmulInputSf(
      inp, offsets, rounded_offsets, BlockScalingFactorLayout::Block128x4);
  fusion.addOutput(out_tv);

  // This is not allowed and we should error out since layout op output should
  // only be consumed by grouped_matmul op
  auto relu_tv = relu(out_tv);
  fusion.addOutput(relu_tv);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 512;
  int k = 9; // note: padded column size would be 12
  auto t0 = at::randn({m, k}, options);
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384}, options.dtype(at::kInt));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  EXPECT_ANY_THROW(executor_cache.runFusionWithInputs({t0, t1, t2}));
}

TEST_F(LayoutOpTest, SchedulerKernelWithOffsetsProducer) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  // fusion should segment here, because layout op requires offsets to be in
  // global memory
  auto offsets_add = add(offsets, fusion.oneVal());
  auto rounded_offsets_add = add(rounded_offsets, fusion.oneVal());

  auto out_tv = preprocessGroupedMatmulInputSf(
      inp,
      offsets_add,
      rounded_offsets_add,
      BlockScalingFactorLayout::Block128x4);
  fusion.addOutput(out_tv);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 512;
  int k = 9; // note: padded column size would be 12
  auto t0 = at::randn({m, k}, options);
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250, 512}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384, 768}, options.dtype(at::kInt));

  // naive scheduling.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1.sub(1), t2.sub(1)});

  ASSERT_TRUE(validateGroupedLayout(
      BlockScalingFactorLayout::Block128x4,
      outputs[0].as<at::Tensor>(),
      t0,
      t1,
      t2));
}

TEST_F(LayoutOpTest, SchedulerKernelWithExplicitQuantizationPattern) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto block_size = IrBuilder::create<Val>(16, DataType::Int);
  auto remainder = ceilDiv(inp->axis(1)->extent(), block_size);

  auto reshaped_inp =
      reshape(inp, {inp->axis(0)->extent(), remainder, block_size});
  auto blocked_sf = max(reshaped_inp, {2});
  auto scaled_output = reshape(
      div(reshaped_inp, broadcast(blocked_sf, {false, false, true})),
      {inp->axis(0)->extent(), inp->axis(1)->extent()});
  // NOTE: output needs to be casted to DataType::Float4_e2m1fn, skipping that
  // for simplicity for validation
  fusion.addOutput(scaled_output);

  auto out_blocked_sf_fp8 = preprocessGroupedMatmulInputSf(
      blocked_sf,
      offsets,
      rounded_offsets,
      BlockScalingFactorLayout::Block128x4);
  // NOTE: output needs to be casted to DataType::Float8_e4m3fn, skipping that
  // for simplicity for validation
  fusion.addOutput(out_blocked_sf_fp8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 512;
  int k = 9 * 16; // note: padded column size needs to be a multiple of 16
  auto t0 = at::randn({m, k}, options);
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384}, options.dtype(at::kInt));

  // automatic scheduling.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  // producing reference
  auto ref_reshaped_inp = t0.view({m, k / 16, 16});
  auto ref_block_sf = ref_reshaped_inp.amax(-1);
  auto ref_scaled_out =
      (ref_reshaped_inp / ref_block_sf.unsqueeze(-1)).view({m, k});

  // check scaled output
  EXPECT_TRUE(at::allclose(ref_scaled_out, outputs[0].as<at::Tensor>()));
  // check block scaling factor
  ASSERT_TRUE(validateGroupedLayout(
      BlockScalingFactorLayout::Block128x4,
      outputs[1].as<at::Tensor>(),
      ref_block_sf,
      t1,
      t2));

  EXPECT_THAT(
      executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups(),
      UnorderedElementsAre(
          HeuristicIs(SchedulerType::InnerPersistent),
          HeuristicIs(SchedulerType::ExprEval)));
}

TEST_F(LayoutOpTest, InferenceBenchmarkLoopPromotionIssue) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeContigConcreteTensor({2048, 320, 16}, DataType::BFloat16);
  auto sf = makeContigConcreteTensor({2048, 320}, DataType::BFloat16);
  // FIXME: this should be 128. i.e. number of groups, fix Masaki's scripts
  // later.
  auto in_offset = makeContigConcreteTensor({3}, DataType::Int32);
  auto out_offset = makeContigConcreteTensor({3}, DataType::Int32);

  fusion.addInput(inp);
  fusion.addInput(sf);
  fusion.addInput(in_offset);
  fusion.addInput(out_offset);

  auto max_fp4 = IrBuilder::create<Val>(6, DataType::Double);
  auto max_fp8 = IrBuilder::create<Val>(448, DataType::Double);
  auto min_fp8 = IrBuilder::create<Val>(-448, DataType::Double);
  auto eps = IrBuilder::create<Val>(0.015625, DataType::Double);

  auto T81 = castOp(DataType::Float, inp);
  auto T77 = castOp(DataType::Float, sf);
  auto T78 = div(T77, max_fp4);
  auto T79 = clamp(T78, eps, max_fp8);
  auto T80 = broadcast(T79, {false, false, true});
  auto T82 = div(T81, T80);
  auto T83 = clamp(T82, min_fp8, max_fp8);
  auto T146 = castOp(DataType::Float4_e2m1fn, T83);
  auto T155 = reshape(T146, {2048, 320, 16}, {2048, 320 * 16});

  auto T86 = castOp(DataType::Float8_e4m3fn, T79);
  auto T87 = preprocessGroupedMatmulInputSf(
      T86, in_offset, out_offset, BlockScalingFactorLayout::Block128x4);

  fusion.addOutput(T155);
  fusion.addOutput(T87);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2048, 320, 16}, options);
  at::Tensor t1 = t0.amax({2});
  at::Tensor in_offsets = at::tensor({0, 600, 1200}, options.dtype(at::kInt));
  at::Tensor out_offsets = at::tensor({0, 640, 1280}, options.dtype(at::kInt));

  // automatic scheduling.
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs =
      executor_cache.runFusionWithInputs({t0, t1, in_offsets, out_offsets});

  // TODO: add validation
  // // check block scaling factor
  // ASSERT_TRUE(validateGroupedLayout(
  //     BlockScalingFactorLayout::Block128x4,
  //     outputs[1].as<at::Tensor>(),
  //     ref_block_sf,
  //     t1,
  //     t2));
  // EXPECT_THAT(
  //     executor_cache.getMostRecentKernelRuntime()->fusionSegments()->groups(),
  //     UnorderedElementsAre(
  //         HeuristicIs(SchedulerType::InnerPersistent),
  //         HeuristicIs(SchedulerType::ExprEval)));
}

} // namespace nvfuser
