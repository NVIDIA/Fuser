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
  int num_group = expert_offsets.size(0) - 1;

  // take length of reference for un-padded k size.
  int k = ref.size(1);

  // We validate each group individually
  for (int i = 0; i < num_group; ++i) {
    int start_idx = sf_offsets[i].item().to<int>();
    int padded_m_g = sf_offsets[i + 1].item().to<int>() - start_idx;
    int m_g = expert_offsets[i + 1].item().to<int>() -
        expert_offsets[i].item().to<int>();
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
        expert_offsets[i + 1].item().to<int>());
    if (!at::allclose(restored_out_g, ref_g)) {
      return false;
    }
  }
  return true;
}

} // namespace

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
  auto t0 = at::randn({m, k}, options);
  // tokens per group are [100, 150, 262] respectively, so each group would be
  // padded to multiple of 128. Hence the total output row span would cover a
  // length of 128 + 256 + 384 = 768.
  auto t1 = at::tensor({0, 100, 250, 512}, options.dtype(at::kInt));
  auto t2 = at::tensor({0, 128, 384, 768}, options.dtype(at::kInt));

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

  auto inp_tv = set(inp);
  auto out_tv = preprocessGroupedMatmulInputSf(
      inp_tv, offsets, rounded_offsets, BlockScalingFactorLayout::Block128x4);
  // NOTE: output of preprocessGroupedMatmulInputSf needs to be on global
  // memory, because we do indexing on output inside the runtime function.
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
  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  ASSERT_TRUE(validateGroupedLayout(
      BlockScalingFactorLayout::Block128x4,
      outputs[0].as<at::Tensor>(),
      t0,
      t1,
      t2));
}

} // namespace nvfuser
