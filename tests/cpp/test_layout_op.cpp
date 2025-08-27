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

  auto t2 = at::tensor({0, 128}, options.dtype(at::kInt));
  auto t3 = at::tensor({0, 128}, options.dtype(at::kInt));

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2, t3});
  auto outputs = ke.run({t0, t1, t2, t3});
}

} // namespace nvfuser
