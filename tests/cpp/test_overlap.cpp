// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

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
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);
  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);

  out->outer_split(0, d);
  // A Swizzle is needed to represent a cyclic shift needed for ring-based
  // overlapping: http://nv/eNL. This is not implemented yet and therefore
  // omitted in all `RingBasedOverlapTest`s.
  out->axis(0)->parallelize(ParallelType::Stream);
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
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);
  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);

  w->outer_split(-1, d);
  w->axis(-2)->parallelize(ParallelType::Stream);
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

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);
  // For computing input gradients, `in` is the output and its reduction
  // dimension needs to be sharded.
  in->outer_split(-1, d);
  in->axis(-2)->parallelize(ParallelType::DIDx);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);
  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);

  // This is debatable. On the one hand, we want to express the intention to
  // stream-parallelize the sequence dimension. On the other hand, we want to
  // avoid parallelizing a fusion input because a fusion input doesn't have a
  // producer.
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::Stream);
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
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);
  w->outer_split(1, d);
  w->axis(1)->parallelize(ParallelType::DIDx);
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);

  w->outer_split(-1, d);
  w->axis(-2)->parallelize(ParallelType::Stream);
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
  for (auto* tv : {in, w, out}) {
    tv->setDeviceMesh(mesh);
  }

  in->outer_split(1, d);
  in->axis(1)->parallelize(ParallelType::DIDx);
  w->outer_split(1, d);
  w->axis(1)->parallelize(ParallelType::DIDx);
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::DIDx);

  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::Stream);
}

// TODO: add tests for row-wise parallel linear without sequence parallelism

// TODO: add tests for collective-based overlapping for which layouts are in
// favor

} // namespace nvfuser
