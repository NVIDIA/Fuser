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
#include <ops/arith.h>
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
      << out_tensor << " != " << expected_out_tensor;
}

} // namespace nvfuser
