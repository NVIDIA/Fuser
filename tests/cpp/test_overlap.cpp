// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <preseg_passes/decompose_reshardings.h>
#include <preseg_passes/propagate_shardings.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::_;
using testing::ElementsAre;

using RingBasedOverlapTest = NVFuserTest;

TEST_F(RingBasedOverlapTest, ColumnAndSequenceParallelLinear_Forward) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h * 4, h}, DataType::BFloat16);
  TensorView* out = linear(in, w, nullptr);

  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, w}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);

  out->outer_split(0, d);
  // A Swizzle is needed to represent a cyclic shift needed for ring-based
  // overlapping: http://nv/eNL. This is not implemented yet and therefore
  // omitted in all `RingBasedOverlapTest`s.
  out->axis(0)->parallelize(ParallelType::Stream);

  // Fusion IR before segmentation will look like this:
  //
  //   [t, h]
  //   /\.
  //  d
  // (deviceIdx.x)
  //    |
  //    | set
  //    |
  //   [t, h]                                  [4h,  h]
  //   /\                                      /\.
  //  d                                       d
  // (streamIdx + deviceIdx.x) % deviceDim.x
  //  | swizzle
  //  s
  // (streamIdx)
  //                      |
  //                      | linear
  //                      |
  //                   [t, 4h, r{h}]
  //                   /\  /\.
  //                  d   d
  //          swizzle |
  //                  s*
  //
  // Notes for this test and many other RingBasedOverlapTest below:
  // - A `set` from `/d` to `/s` (or vice versa) will be lowered to a
  // cyclic-shift communication like XLA's CollectivePermute. In the example
  // above, in the first iteration (i.e. streamIdx = 0), `set` does nothing
  // because the input `deviceIdx.x` equals the output `deviceIdx.x`. In the
  // second iteration (i.e. streamIdx = 1), `set` sends data from device i to
  // device (i-1)%deviceDim.x. This is captured by the math of the swizzle.
  // According to the logical domain mapping, the input deviceIdx.x equals
  // (streamIdx + the output deviceIdx.x) % deviceDim.x. Equivalently, the
  // output deviceIdx.x equals (the input deviceIdx.x - streamIdx) %
  // deviceDim.x.
  // - All leaf `s`s are parallelized on `Stream` and all leaf `d`s are
  // parallelized on `DIDx`.
  // - `s*`s are parallelized on `Stream` in loop but replicated in allocation.
  // Fusion inputs/outputs can't be allocated per stream because the
  // user of a FusionDefinition can't inline external ops into a loop inside.

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  EXPECT_THAT(
      out->getLoopDomain(),
      ElementsAre(
          IsParallelized(ParallelType::Stream),
          _,
          IsParallelized(ParallelType::DIDx),
          _,
          _));
}

TEST_F(RingBasedOverlapTest, ColumnAndSequenceParallelLinear_WeightGrad) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h}, DataType::BFloat16);
  TensorView* out = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* w = matmul(transpose(out), in);

  fusion->addInput(in);
  fusion->addInput(out);
  fusion->addOutput(w);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);

  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);

  w->outer_split(-1, d);
  w->axis(-2)->parallelize(ParallelType::Stream);

  // Fusion IR before segmentation will look like this:
  //
  //   [t, h]                                  [t,  4h]
  //   /\                                      /\   /\.
  //  d                                       s*   d
  //    |                                        |
  //    | set                                    | permute
  //    |                                        |
  //   [t, h]                                  [4h,  t]
  //   /\                                      /\   /\.
  //  s                                       d    s
  //           \                          /
  //          (operand B)         (operand A)
  //                      | matmul
  //                      |
  //                         r{t}
  //                         /  \.
  //                [4h, h, s*, r{t/s}]
  //                 /\.
  //                d
  //                      |
  //                      | sum
  //                      |
  //                   [4h, h, r{s}]
  //                    /\.
  //                   d
  //
  // Notes:
  // - The matmul output is parallelized on `Stream` in loop but not in
  // allocation. This dimension should eventually be parallelized on `Stream`
  // in allocation as well. This way, host IR lowering will fuse `sum` into the
  // same loop as `matmul` as an add, saving time and memory. It's not yet
  // clear to me how to implement this in host IR lowering, so I recommend we
  // go with `s*` for now for simplicity.

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  EXPECT_THAT(
      out->getLoopDomain(),
      ElementsAre(
          IsParallelized(ParallelType::Stream),
          _,
          IsParallelized(ParallelType::DIDx),
          _));
}

TEST_F(RingBasedOverlapTest, ColumnAndSequenceParallelLinear_InputGrad) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* out = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h * 4, h}, DataType::BFloat16);
  TensorView* in = matmul(out, w);

  fusion->addInput(out);
  fusion->addInput(w);
  fusion->addOutput(in);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);
  // This is debatable. On the one hand, we want to express the intention to
  // stream-parallelize the sequence dimension. On the other hand, we want to
  // avoid parallelizing a fusion input because a fusion input doesn't have a
  // producer.
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::Stream);

  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);

  // Fusion IR before segmentation will look like this:
  //
  //   [t, 4h]                                 [4h,  h]
  //   /\  /\                                   /\.
  //  d   d                                    d
  //  |
  //  s*
  //                      |
  //                      | matmul
  //                      |
  //                          r{4h}
  //                          /  \.
  //                 [t, h, d, r{4h/d}]
  //                 /\.
  //                d
  //                |
  //                s
  //                     |
  //                     | set
  //                     |
  //                  [t, h, d]
  //                  /\.    |
  //                 d       s*
  //                     |
  //                     | sum
  //                     |
  //                  [t, h, r{d}]
  //                  /\.
  //                 d
}

TEST_F(RingBasedOverlapTest, RowAndSequenceParallelLinear_Forward) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h, h * 4}, DataType::BFloat16);
  TensorView* out = linear(in, w, nullptr);

  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);
  w->outer_split(1, d);
  w->axis(1)->parallelize(ParallelType::DIDx);
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);
  out->outer_split(-1, d);
  out->axis(-2)->parallelize(ParallelType::DIDx);

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::Stream);

  // Fusion IR before segmentation will look similar to
  // `ColumnAndSequenceParallelLinear_InputGrad`.
}

TEST_F(RingBasedOverlapTest, RowAndSequenceParallelLinear_WeightGrad) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* out = makeContigConcreteTensor({-1, h}, DataType::BFloat16);
  TensorView* w = matmul(transpose(out), in);

  fusion->addInput(in);
  fusion->addInput(out);
  fusion->addOutput(w);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);
  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::Stream);

  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);

  // Fusion IR before segmentation will be slightly different from
  // `ColumnAndSequenceParallelLinear_WeightGrad`:
  //
  //                                           [t, h]
  //                                           /\.
  //                                          d
  //                                             |
  //                                             | permute
  //                                             |
  //                                          [h, t]
  //                                             /\.
  //                                            d
  //                                             |
  //                                             | set
  //                                             |
  //   [t, 4h]                                [h, t]
  //   /\  /\                                    /\.
  //  s*  d                                     s
  //           \                          /
  //          (operand B)         (operand A)
  //                      | matmul
  //                      |
  //                          r{t}
  //                          /  \.
  //                 [h, 4h, s*, r{t/s}]
  //                     /\.
  //                    d
  //                      |
  //                      | sum
  //                      |
  //                   [h, 4h, r{s}]
  //                       /\.
  //                      d

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  // Due to lack of DecomposeReshardingsPass, `w` looks like the following:
  //
  //                   [h, 4h, r{t}]
  //                       /\  /\.
  //                      d   s
  EXPECT_THAT(
      w->getLoopDomain(),
      ElementsAre(
          _,
          IsParallelized(ParallelType::DIDx),
          _,
          IsParallelized(ParallelType::Stream),
          _));
}

TEST_F(RingBasedOverlapTest, RowAndSequenceParallelLinear_InputGrad) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* out = makeContigConcreteTensor({-1, h}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h, h * 4}, DataType::BFloat16);
  TensorView* in = matmul(out, w);

  fusion->addInput(out);
  fusion->addInput(w);
  fusion->addOutput(in);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {w, out}) {
    tv->setDeviceMesh(mesh);
  }

  w->outer_split(1, d);
  w->axis(1)->parallelize(ParallelType::DIDx);
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::Stream);

  // Fusion IR before segmentation will be similar to
  // `ColumnAndSequenceParallelLinear_Forward`.
  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  EXPECT_THAT(
      in->getLoopDomain(),
      ElementsAre(
          IsParallelized(ParallelType::Stream),
          _,
          IsParallelized(ParallelType::DIDx),
          _,
          _));
}

// We can apply collective-based overlapping to the above patterns as well. The
// tradeoffs are:
// - Collective-based overlapping will expose a fine-grained communication to
// the critical path.
// - Collective-based overlapping requires extra data copy due to NCCL/UCC's
// layout constraints.
// + Collective-based overlapping is more flexible with respect to the number of
// steps. Ring-based decomposition requires that to be a multiple of the number
// of devices, whereas collective-based decomposition doesn't.
//
// Therefore, we leave it for future work.
using CollectiveBasedOverlapTest = NVFuserTest;

// Unlike Linear+ReduceScatter, this Linear+Allreduce pattern doesn't benefit
// from ring-based overlapping. However we decompose the compute and the
// communication, a trailing allreduce has to be exposed on the critical path to
// combine the partial results from all participating devices.
TEST_F(CollectiveBasedOverlapTest, RowParallelLinear_Forward) {
  constexpr int64_t h = 12288;
  constexpr int64_t d = 2;
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* in = makeContigConcreteTensor({-1, h * 4}, DataType::BFloat16);
  TensorView* w = makeContigConcreteTensor({h, h * 4}, DataType::BFloat16);
  TensorView* out = linear(in, w, nullptr);

  fusion->addInput(in);
  fusion->addInput(w);
  fusion->addOutput(out);

  const auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto* tv : {in, w}) {
    tv->setDeviceMesh(mesh);
  }
  in->outer_split(0, d);
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

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  preseg_passes::OptimizationPass<
      preseg_passes::DecomposeReshardingsPass>::runPass(fusion.get());

  EXPECT_THAT(
      fusion->outputs().at(0)->as<TensorView>()->getLoopDomain(),
      ElementsAre(
          IsParallelized(ParallelType::Stream),
          _,
          _,
          IsParallelized(ParallelType::DIDx)));
}

} // namespace nvfuser
