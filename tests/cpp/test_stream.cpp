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
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class StreamTest : public NVFuserTest {
 protected:
  StreamTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

TEST_F(StreamTest, AddPerStream) {
  constexpr int64_t c = 3;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigTensor(2);
  TensorView* out = add(in, in);
  fusion->addInput(in);
  fusion->addOutput(out);

  in->outer_split(1, c);
  in->axis(1)->parallelize(ParallelType::Stream);
  out->outer_split(1, c);
  out->axis(1)->parallelize(ParallelType::Stream);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  at::Tensor in_tensor = at::randn({5, c * 2}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder out_tensors =
      executor_cache.runFusionWithInputs({in_tensor});
  auto out_tensor = out_tensors[0].as<at::Tensor>();

  testValidate(
      executor_cache.fusion(), {out_tensor}, {in_tensor}, __LINE__, __FILE__);
}

} // namespace nvfuser
