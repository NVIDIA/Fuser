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
bool validateGroupedLayout(BlockScalingFactorLayout layout,
  at::Tensor inp,
  at::Tensor ref,
  at::Tensor expert_offsets,
  at::Tensor sf_offsets) {
  NVF_ERROR(BlockScalingFactorLayout::Block128x4 == layout);
  int num_group = expert_offsets.size(0) - 1;

  int k = ref.size(1);

  for (int i = 0; i < num_group; ++i) {
    int start_idx = sf_offsets[i].item().to<int>();
    int m_g = expert_offsets[i+1].item().to<int>()- expert_offsets[i].item().to<int>();
    auto inp_g = inp.slice(0, start_idx, start_idx + m_g);

    int mn_tile = std::ceil(m_g / 128);
    int k_tile = std::ceil(k / 4);

    // view as {mn_tile, k_tile, m_4, mn_32, k_4}

    inp_g = inp_g.view({mn_tile, k_tile, 4, 32, 4}).transpose(1, 3).reshape({mn_tile*32*4, k_tile*4}).slice(0,0,m_g).slice(1,0,k);
    if (!at::allclose(inp_g, ref.slice(0, expert_offsets[i].item().to<int>(), expert_offsets[i+1].item().to<int>()))) {
      return false;
    }
  }
  return true;
}

}

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
  auto buffer = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(buffer);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto out = groupedBlockSfLayout(
      inp,
      buffer,
      offsets,
      rounded_offsets,
      BlockScalingFactorLayout::Block128x4);
  fusion.addOutput(out);

  fusion.printMath();
}

TEST_F(LayoutOpTest, ManaulKernel) {
  auto fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto inp = makeSymbolicTensor(2);
  auto buffer = makeSymbolicTensor(2);
  auto offsets = makeSymbolicTensor(1, DataType::Int32);
  auto rounded_offsets = makeSymbolicTensor(1, DataType::Int32);
  fusion.addInput(inp);
  fusion.addInput(buffer);
  fusion.addInput(offsets);
  fusion.addInput(rounded_offsets);

  auto inp_tv = set(inp);
  auto out_tv = groupedBlockSfLayout(
      inp_tv,
      buffer,
      offsets,
      rounded_offsets,
      BlockScalingFactorLayout::Block128x4);
  auto out = set(out_tv);
  fusion.addOutput(out);

  // naive scheduling.
  for (auto tv : {inp, inp_tv, out_tv, out}) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  fusion.printMath();
  fusion.printTransforms();

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int m = 256;
  int k = 16;
  int g = 2;

  auto t0 = at::randn({m, k}, options);
  // buffer size is computed by assuming maximum padding per group.
  int pad_size = 128;
  int buffer_m = m + (pad_size - 1) * g;
  auto t1 = at::randn({buffer_m, k}, options);

  auto t2 = at::tensor({0, 128, 256}, options.dtype(at::kInt));
  auto t3 = at::tensor({0, 128, 256}, options.dtype(at::kInt));

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2, t3});
  auto outputs = ke.run({t0, t1, t2, t3});

  // assert that outputs[0] is t1;
  ASSERT_TRUE(validateGroupedLayout(BlockScalingFactorLayout::Block128x4, t1, t0, t2, t3));
}

} // namespace nvfuser 
