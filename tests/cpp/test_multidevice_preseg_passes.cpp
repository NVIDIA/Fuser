// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/optimization_pass.h>
#include <preseg_passes/propagate_shardings.h>
#include <runtime/fusion_executor_cache.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>
#include "multidevice/utils.h"

namespace nvfuser {

constexpr int64_t b = 2, s = 3, h = 128, a = 8;
constexpr double dropout_p = 0.0;
constexpr bool is_causal = false;

using testing::ElementsAre;
using MultiDevicePresegPassesTest = MultiDeviceTest;

TEST_F(MultiDevicePresegPassesTest, ResidualAdd) {
  // This is similar to the residual add after MHA dropout in the transformer.
  // The output of linear following MHA is all-gathered and sharded on the
  // sequence dim. This sharding can be propagated to the linear output through
  // backpropagating the shardings from residual add. This information is not
  // present during forward propagation.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();

  TensorView* tv0 = makeContigConcreteTensor({d, 4});
  TensorView* tv1 = uniform(
      shape(tv0),
      fusion->zeroVal(DataType::Float),
      fusion->oneVal(DataType::Float),
      DataType::Float);
  TensorView* tv2 = add(tv0, tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv0->split(0, d, /*inner_split=*/false);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  NVF_CHECK(tv1->hasDeviceMesh());
  NVF_CHECK(getShardedLogicalAxis(tv1, ParallelType::DIDx) == getShardedLogicalAxis(tv0, ParallelType::DIDx), "Expected tv1 to be sharded like tv0 due to backpropagation of shardings.");
}

namespace {
at::Tensor reference_mha(at::Tensor inp) {
  auto qkv =
      inp.view({b, s, a, 3 * h / a}).transpose(1, 2).split(h / a, -1);
  double scale = 1.0 / std::sqrt(h / a);
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv[0],
      qkv[1],
      qkv[2],
      /*dropout_p=*/dropout_p,
      /*is_causal=*/is_causal,
      /*return_debug_mask=*/false,
      scale);
  auto attn = std::get<0>(sdpa_out);
  return attn.transpose(1, 2);
}
} // namespace

TEST_F(MultiDevicePresegPassesTest, MHAFwd) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 128, a = 8;

  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* qkv = makeConcreteTensor({b, s, d * a, 3 * h / a}, DataType::Half);
  TensorView* q = slice(qkv, {0, 0, 0, 0}, {b, s, d * a, h / a});
  TensorView* k = slice(qkv, {0, 0, 0, h / a}, {b, s, d * a, 2 * h / a});
  TensorView* v = slice(qkv, {0, 0, 0, 2 * h / a}, {b, s, d * a, 3 * h / a});

  TensorView* q_permuted = permute(q, {0, 2, 1, 3});
  TensorView* k_permuted = permute(k, {0, 2, 1, 3});
  TensorView* v_permuted = permute(v, {0, 2, 1, 3});

  SdpfaFwdResult sdpa_out = sdpfa_fwd(
      q_permuted,
      k_permuted,
      v_permuted,
      /*dropout_p=*/IrBuilder::create<Val>(dropout_p),
      /*is_causal=*/IrBuilder::create<Val>(is_causal),
      /*scale=*/nullptr);

  TensorView* attn = sdpa_out.output;
  TensorView* attn_permute = permute(attn, {0, 2, 1, 3});

  fusion->addInput(qkv);
  fusion->addOutput(attn_permute);

  qkv->setDeviceMesh(mesh);
  qkv->outer_split(2, d);
  qkv->axis(2)->parallelize(ParallelType::DIDx);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor unsharded_inp_tensor = at::randn({b, s, d * a, 3 * h / a}, tensor_options.dtype(at::kHalf));
  at::Tensor inp_tensor = shardTensor(unsharded_inp_tensor, 2, mesh);

  KernelArgumentHolder args = {inp_tensor};
  auto outputs = executor_cache.runFusionWithInputs(args);
  at::Tensor nvf_out = outputs[0].as<at::Tensor>();

  at::Tensor ref_out = reference_mha(inp_tensor);

  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {inp_tensor},
      {ref_out},
      __LINE__,
      __FILE__);
}
  
} // namespace nvfuser
