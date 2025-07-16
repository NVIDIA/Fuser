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

using OverlapTest = NVFuserTest;

TEST_F(OverlapTest, ColumnAndSequenceParallelLinear_Forward) {
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
  out->axis(0)->parallelize(ParallelType::Stream);

  fusion->print();
}

TEST_F(OverlapTest, ColumnAndSequenceParallelLinear_WeightGrad) {
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
  out->outer_split(1, d);
  out->axis(1)->parallelize(ParallelType::DIDx);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);
  w->outer_split(-1, d);
  w->axis(-2)->parallelize(ParallelType::Stream);

  fusion->print();
}

TEST_F(OverlapTest, ColumnAndSequenceParallelLinear_InputGrad) {
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
  out->outer_split(0, d);
  out->axis(0)->parallelize(ParallelType::Stream);
  w->outer_split(0, d);
  w->axis(0)->parallelize(ParallelType::DIDx);
  in->outer_split(0, d);
  in->axis(0)->parallelize(ParallelType::DIDx);

  fusion->print();
}

} // namespace nvfuser