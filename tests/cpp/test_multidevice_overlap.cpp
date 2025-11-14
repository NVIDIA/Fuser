// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <options.h>
#include <tests/cpp/multidevice.h>

namespace nvfuser {

class StreamTest : public MultiDeviceTest {
 public:
  StreamTest() {
    EnableOptionsGuard::getCurOptions().set(EnableOption::HostIrLowering);
  }
};

// FIXME: make a python test
// Unlike Linear+ReduceScatter, this Linear+Allreduce pattern doesn't benefit
// from ring-based overlapping. However we decompose the compute and the
// communication, a trailing allreduce has to be exposed on the critical path to
// combine the partial results from all participating devices.
TEST_F(StreamTest, RowParallelLinear_Forward) {
  constexpr int64_t h = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h, h * 4}, DataType::BFloat16);
  TensorView* out = linear(in, w, nullptr);

  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  const int64_t d = communicator_->size();
  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, w}) {
    tv->setDeviceMesh(mesh);
  }

  constexpr int64_t s = 3;
  in->outer_split(0, s);
  in->axis(0)->parallelize(ParallelType::Stream);
  in->outer_split(2, d);
  in->axis(2)->parallelize(ParallelType::DIDx);
  w->outer_split(1, d);
  w->axis(1)->parallelize(ParallelType::DIDx);

  // Fusion IR before segmentation will look like this:
  //
  //   [t, 4h]                                 [h, 4h]
  //   /\  /\                                      /\.
  //  s*  d                                       d
  //                      |
  //                      | linear
  //                      |
  //                          r{4h}
  //                          /  \.
  //                 [t, h, d, r{4h/d}]
  //                 /\.
  //                s
  //                     |
  //                     | sum
  //                     |
  //                  [t, h, r{d}]
  //                  /\.
  //                 s*

  constexpr int64_t t = 6;
  static_assert(t % s == 0);
  at::Tensor in_tensor =
      at::randn({t, h * 4}, tensor_options_.dtype(at::kBFloat16));
  at::Tensor w_tensor =
      at::randn({h, h * 4}, tensor_options_.dtype(at::kBFloat16));
  at::Tensor out_tensor = at::linear(in_tensor, w_tensor);

  at::Tensor sharded_in_tensor = shardTensor(in_tensor, in);
  at::Tensor sharded_w_tensor = shardTensor(w_tensor, w);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor sharded_out_tensor =
      executor_cache
          .runFusionWithInputs({sharded_in_tensor, sharded_w_tensor})[0]
          .as<at::Tensor>();

  EXPECT_TRUE(at::allclose(sharded_out_tensor, out_tensor));
}

} // namespace nvfuser
