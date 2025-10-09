// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <ATen/ops/randn.h>
#include <ATen/ops/zeros_like.h>

#include <fusion.h>
#include <fusion_guard.h>
#include <ir/interface_nodes.h>
#include <multidevice/utils.h>
#include <ops/alias.h>
#include <ops/all_ops.h>
#include <preseg_passes/propagate_shardings.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using StreamTest = NVFuserTest;

TEST_F(StreamTest, AddPerStream) {
  constexpr int64_t c = 3;
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(2);
  TensorView* out = add(in, in);
  fusion.addInput(in);
  fusion.addOutput(out);

  in->outer_split(1, c);
  in->axis(1)->parallelize(ParallelType::Stream);
  out->outer_split(1, c);
  out->axis(1)->parallelize(ParallelType::Stream);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5, c * 2}, options);
  at::Tensor out_tensor = at::zeros_like(in_tensor);

  KernelExecutor ke;
  ke.compile(&fusion, {in_tensor});
  constexpr int64_t kStreamIndex = 1;
  ke.run({in_tensor, kStreamIndex}, {out_tensor});

  at::Tensor expected_out_tensor = in_tensor + in_tensor;
  std::vector<at::Tensor> chunks = expected_out_tensor.chunk(c, 1);
  for (auto [i, chunk] : enumerate(chunks)) {
    if (i != kStreamIndex) {
      chunk.zero_();
    }
  }
  EXPECT_TRUE(at::allclose(out_tensor, expected_out_tensor))
      << out_tensor << " vs " << expected_out_tensor;
}

TEST_F(StreamTest, haveDifferentShardings) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int64_t s = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = set(tv2);
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  // tv1: [s, i0/s, i1]
  tv1->outer_split(0, s);
  tv1->axis(0)->parallelize(ParallelType::Stream);

  // tv2: [s, i0/s, i1]
  tv2->outer_split(0, s);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  // tv3: [s, i0, i1/s]
  tv3->outer_split(1, s);
  tv3->axis(1)->parallelize(ParallelType::Stream);

  EXPECT_FALSE(haveDifferentShardings(tv1, tv2, {ParallelType::Stream}));
  EXPECT_TRUE(haveDifferentShardings(tv2, tv3, {ParallelType::Stream}));
}

TEST_F(StreamTest, ForwardPropagation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* in = makeContigTensor(2);
  TensorView* w = makeContigTensor(2);
  TensorView* out = matmul(in, w);
  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  w->outer_split(1, s);
  w->axis(1)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  EXPECT_TRUE(out->axis(1)->isStream()) << out;
}

TEST_F(StreamTest, BackwardPropagation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int64_t s = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  tv2->outer_split(0, s);
  tv2->axis(0)->parallelize(ParallelType::Stream);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto* tv : {tv0, tv1, tv2}) {
    EXPECT_TRUE(tv->axis(0)->isStream()) << tv;
  }
}

} // namespace nvfuser
