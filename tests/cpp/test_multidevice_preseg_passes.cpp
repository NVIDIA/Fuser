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

using MultiDevicePresegPassesTest = MultiDeviceTest;

TEST_F(MultiDevicePresegPassesTest, ResidualAdd) {
  // This is similar to the residual add after MHA dropout in the transformer.
  // The output of linear following MHA is all-gathered and sharded on the sequence dim.
  // This sharding can be propagated to the linear output through backpropagating the shardings
  // from residual add. This information is not present during forward propagation.
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 8;

  TensorView* tv0 = makeContigConcreteTensor({b, d*s, h});
  TensorView* tv1 = makeContigConcreteTensor({b, d*s, h});
  TensorView* tv2 = add(tv0, tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv1->split(1, d, /*inner_split=*/false);
  tv1->axis(1)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  preseg_passes::OptimizationPass<preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  // Set the allocation domain explicitly until the preseg pass is fixed.
  for (auto* tv : {tv0, tv1, tv2}) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  NVF_CHECK(getShardedLogicalAxis(tv0, ParallelType::DIDx) == 1);
  at::Tensor inp0 = at::randn({b, d*s, h}, tensor_options);
  at::Tensor inp1 = at::randn({b, d*s, h}, tensor_options);
  at::Tensor sharded_inp0 = shardTensor(inp0, 1, mesh);
  at::Tensor sharded_inp1 = shardTensor(inp1, 1, mesh);
  
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out = executor_cache.runFusionWithInputs({sharded_inp0, sharded_inp1})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp0, sharded_inp1},
      {sharded_inp0 + sharded_inp1},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDevicePresegPassesTest, MultipleTransformReshape) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 8, e = 4;

  TensorView* tv0 = makeContigConcreteTensor({d*b, s, h*e});
  TensorView* tv1 = reshape(tv0, {d*b, s, h*e}, {d*b*s*h, e});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv0->split(0, d, /*inner_split=*/false);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto* tv : {tv0, tv1}) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  NVF_CHECK(getShardedLogicalAxis(tv1, ParallelType::DIDx) == 0);
  at::Tensor inp = at::randn({d*b, s, h*e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, 0, mesh);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out = executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {sharded_inp.view({b*s*h, e})},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDevicePresegPassesTest, TransformerFwd) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, a = 8, h=128;
  // const double kDropoutProb = 0.1;

  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* inp = makeConcreteTensor({b, d*s, h}, DataType::Half);
  TensorView* mha_linear0_weight = makeConcreteTensor({3*d*h, h}, DataType::Half);
  TensorView* mha_linear1_weight = makeConcreteTensor({h, d*h}, DataType::Half);

  // Layernorm
  TensorView* ln_in = maybeCastOp(DataType::Float, inp);
  TensorView* ln_out = layer_norm(ln_in, {h}, /*weight=*/nullptr, /*bias=*/nullptr, /*eps=*/IrBuilder::create<Val>(1e-5)).output;
  TensorView* mha_in = maybeCastOp(inp->dtype(), ln_out);
  // MHA Linear0
  TensorView* mha_linear0_out = linear(mha_in, mha_linear0_weight);

  // reshape -> slice -> permute
  TensorView* qkv = reshape(mha_linear0_out, {b, s, 3*d*h}, {b, s, d*a, 3*h/a});
  TensorView* q = slice(qkv, {0, 0, 0, 0}, {b, s, d*a, h/a});
  TensorView* k = slice(qkv, {0, 0, 0, h/a}, {b, s, d*a, 2*h/a});
  TensorView* v = slice(qkv, {0, 0, 0, 2*h/a}, {b, s, d*a, 3*h/a});
  
  TensorView* q_permuted = permute(q, {0, 2, 1, 3});
  TensorView* k_permuted = permute(k, {0, 2, 1, 3});
  TensorView* v_permuted = permute(v, {0, 2, 1, 3});

  SdpfaFwdResult sdpa_out = sdpfa_fwd(
      q_permuted,
      k_permuted,
      v_permuted,
      /*dropout_p=*/IrBuilder::create<Val>(0.0),
      /*is_causal=*/IrBuilder::create<Val>(false),
      /*scale=*/nullptr);

  TensorView* attn = sdpa_out.output;
  TensorView* attn_permute = permute(attn, {0, 2, 1, 3});
  TensorView* attn_reshaped = reshape(attn_permute, {b, s, d*a, h/a}, {b, s, d*h});

  // MHA Linear1: The reduction dimension is sharded and requires communication.
  TensorView* mha_linear1_out = linear(attn_reshaped, mha_linear1_weight);
  // Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  // Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  // TensorView* dropout_out = dropout(mha_linear1_out, prob, scale).output;
  // TensorView* residual_add_out = add(dropout_out, inp);

  fusion->addInput(inp);
  fusion->addInput(mha_linear0_weight);
  fusion->addInput(mha_linear1_weight);

  fusion->addOutput(ln_out);

  // Rfactor mha_linear1_out for communication.
  mha_linear1_out->split(-1, d, /*inner_split=*/false);
  TensorView* local_mha_linear1_out = mha_linear1_out->rFactor({-1});

  // Shard input tensors
  for (auto* tv : {inp, mha_linear0_weight, mha_linear1_weight, mha_linear1_out, local_mha_linear1_out}) {
    tv->setDeviceMesh(mesh);
  }
  inp->split(1, d, /*inner_split=*/false);
  inp->axis(1)->parallelize(ParallelType::DIDx);
  
  mha_linear0_weight->split(0, d, /*inner_split=*/false);
  mha_linear0_weight->axis(0)->parallelize(ParallelType::DIDx);
  
  mha_linear1_weight->split(1, d, /*inner_split=*/false);
  mha_linear1_weight->axis(1)->parallelize(ParallelType::DIDx);

  // Parallelize MHA linear out: This will be done in insert_reshardings preseg pass.
  local_mha_linear1_out->axis(-2)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto tv : fusion->allTvs()) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp_tensor = at::randn({b, d*s, h}, tensor_options.dtype(at::kHalf));
  at::Tensor mha_linear0_weight_tensor = at::randn({3*d*h, h}, tensor_options.dtype(at::kHalf));
  at::Tensor mha_linear1_weight_tensor = at::randn({h, d*h}, tensor_options.dtype(at::kHalf));

  at::Tensor sharded_inp = shardTensor(inp_tensor, 1, mesh);
  at::Tensor sharded_mha_linear0_weight = shardTensor(mha_linear0_weight_tensor, 0, mesh);
  at::Tensor sharded_mha_linear1_weight = shardTensor(mha_linear1_weight_tensor, 1, mesh);

  at::Tensor nvf_out =
      executor_cache
          .runFusionWithInputs({sharded_inp, sharded_mha_linear0_weight, sharded_mha_linear1_weight})[0]
          .as<at::Tensor>();
  debug() << "nvf_out: " << nvf_out.sizes() << std::endl;

  // double scale = 1.0 / std::sqrt(e);
  // auto reference_out = at::_scaled_dot_product_flash_attention(
  //     hq_tensor.view(out_shape).transpose(1, 2),
  //     hk_tensor.view(out_shape).transpose(1, 2),
  //     hv_tensor.view(out_shape).transpose(1, 2),
  //     /*dropout_p=*/0.0,
  //     /*is_causal=*/false,
  //     /*return_debug_mask=*/false,
  //     /*scale=*/scale);
  // at::Tensor ref_attn = shardTensor(
  //     std::get<0>(reference_out).transpose(1, 2).view(in_shape), -1, mesh);

  // testValidate(
  //     executor_cache.fusion(),
  //     {nvf_out},
  //     {sharded_hq, sharded_hk, sharded_hv},
  //     {ref_attn},
  //     __LINE__,
  //     __FILE__);
}


} // namespace nvfuser