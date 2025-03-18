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
  const int64_t b = 2, s = 3, h = 8;

  TensorView* tv0 = makeContigConcreteTensor({b, d * s, h});
  TensorView* tv1 = makeContigConcreteTensor({b, d * s, h});
  TensorView* tv2 = add(tv0, tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv1->setDeviceMesh(mesh);
  tv1->split(1, d, /*inner_split=*/false);
  tv1->axis(1)->parallelize(ParallelType::DIDx);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addOutput(tv2);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  // Set the allocation domain explicitly until the preseg pass is fixed.
  for (auto* tv : {tv0, tv1, tv2}) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  NVF_CHECK(getShardedLogicalAxis(tv0, ParallelType::DIDx) == 1);
  at::Tensor inp0 = at::randn({b, d * s, h}, tensor_options);
  at::Tensor inp1 = at::randn({b, d * s, h}, tensor_options);
  at::Tensor sharded_inp0 = shardTensor(inp0, 1, mesh);
  at::Tensor sharded_inp1 = shardTensor(inp1, 1, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp0, sharded_inp1})[0]
          .as<at::Tensor>();
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

  TensorView* tv0 = makeContigConcreteTensor({d * b, s, h * e});
  TensorView* tv1 = reshape(tv0, {d * b, s, h * e}, {d * b * s * h, e});
  fusion->addInput(tv0);
  fusion->addOutput(tv1);

  auto mesh = DeviceMesh::createForNumDevices(d);
  tv0->setDeviceMesh(mesh);
  tv0->split(0, d, /*inner_split=*/false);
  tv0->axis(0)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto* tv : {tv0, tv1}) {
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  NVF_CHECK(getShardedLogicalAxis(tv1, ParallelType::DIDx) == 0);
  at::Tensor inp = at::randn({d * b, s, h * e}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, 0, mesh);
  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();
  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {sharded_inp.view({b * s * h, e})},
      __LINE__,
      __FILE__);
}

TEST_F(MultiDevicePresegPassesTest, SliceReshapePermute) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 128, a = 8;

  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* tv0 = makeConcreteTensor({b, s, 3 * d * h});
  TensorView* tv1 = reshape(tv0, {b, s, 3 * d * h}, {b, s, d * a, 3 * h / a});
  TensorView* tv2 = slice(tv1, {0, 0, 0, 0}, {b, s, d * a, h / a});
  TensorView* tv3 = permute(tv2, {0, 2, 1, 3});

  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  tv0->setDeviceMesh(mesh);
  tv0->split(-1, d, /*inner_split=*/false);
  tv0->axis(-2)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());

  for (auto* tv : fusion->allTvs()) {
    reorderDIDToFront(tv);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp = at::randn({b, s, 3 * d * h}, tensor_options);
  at::Tensor sharded_inp = shardTensor(inp, -1, mesh);
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp})[0].as<at::Tensor>();

  at::Tensor reference_out = sharded_inp.view({b, s, a, 3 * h / a})
                                 .index(
                                     {at::indexing::Slice(0),
                                      at::indexing::Slice(0),
                                      at::indexing::Slice(0),
                                      at::indexing::Slice(0, h / a)})
                                 .transpose(1, 2);

  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {sharded_inp},
      {reference_out},
      __LINE__,
      __FILE__);
}

// TODO: Enable this test once the insert_reshardings preseg pass is fixed.
TEST_F(MultiDevicePresegPassesTest, DISABLED_MHALinear) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  auto mesh = DeviceMesh::createForNumDevices(d);
  const int64_t b = 2, s = 3, h = 128; //,a=8;

  TensorView* inp = makeConcreteTensor({b, d * s, h}, DataType::Half);
  TensorView* weight = makeConcreteTensor({3 * d * h, h}, DataType::Half);
  TensorView* out = linear(inp, weight);

  fusion->addInput(inp);
  fusion->addInput(weight);
  fusion->addOutput(out);

  inp->setDeviceMesh(mesh);
  weight->setDeviceMesh(mesh);
  inp->split(1, d, /*inner_split=*/false);
  inp->axis(1)->parallelize(ParallelType::DIDx);
  weight->split(0, d, /*inner_split=*/false);
  weight->axis(0)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto* tv : fusion->allTvs()) {
    reorderDIDToFront(tv);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }
  NVF_CHECK(getShardedLogicalAxis(out, ParallelType::DIDx) == 2);
  at::Tensor inp_tensor =
      at::randn({b, d * s, h}, tensor_options.dtype(at::kHalf));
  at::Tensor sharded_inp = shardTensor(inp_tensor, 1, mesh);

  at::Tensor weight_tensor =
      at::randn({3 * d * h, h}, tensor_options.dtype(at::kHalf));
  at::Tensor sharded_weight = shardTensor(weight_tensor, 0, mesh);

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor nvf_out =
      executor_cache.runFusionWithInputs({sharded_inp, sharded_weight})[0]
          .as<at::Tensor>();
}

namespace {
at::Tensor reference_mha(at::Tensor inp, at::Tensor weight) {
  at::Tensor linear0_out = at::linear(inp, weight);
  auto qkv =
      linear0_out.view({b, s, a, 3 * h / a}).transpose(1, 2).split(h / a, -1);
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
  return attn.transpose(1, 2).reshape({b, s, h});
}
} // namespace

TEST_F(MultiDevicePresegPassesTest, MHAFwd) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  const int d = communicator_->size();
  const int64_t b = 2, s = 3, h = 128, a = 8;

  auto mesh = DeviceMesh::createForNumDevices(d);

  TensorView* inp = makeConcreteTensor({b, s, h}, DataType::Half);
  TensorView* mha_w0 = makeConcreteTensor({3 * d * h, h}, DataType::Half);

  // MHA Linear0
  TensorView* mha_linear0_out = linear(inp, mha_w0);

  // reshape -> slice -> permute
  TensorView* qkv =
      reshape(mha_linear0_out, {b, s, 3 * d * h}, {b, s, d * a, 3 * h / a});
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
  TensorView* attn_reshaped =
      reshape(attn_permute, {b, s, d * a, h / a}, {b, s, d * h});

  fusion->addInput(inp);
  fusion->addInput(mha_w0);
  fusion->addOutput(attn_reshaped);

  // Shard input tensors
  for (auto* tv : {inp, mha_w0}) {
    tv->setDeviceMesh(mesh);
  }

  mha_w0->split(0, d, /*inner_split=*/false);
  mha_w0->axis(0)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::PropagateShardingsPass>::runPass(fusion.get());
  for (auto tv : fusion->allTvs()) {
    reorderDIDToFront(tv);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  at::Tensor inp_tensor = at::randn({b, s, h}, tensor_options.dtype(at::kHalf));

  at::Tensor mha_w0_tensor =
      at::randn({3 * d * h, h}, tensor_options.dtype(at::kHalf));
  at::Tensor sharded_mha_w0 = shardTensor(mha_w0_tensor, 0, mesh);

  KernelArgumentHolder args = {inp_tensor, sharded_mha_w0};
  auto outputs = executor_cache.runFusionWithInputs(args);
  at::Tensor nvf_out = outputs[0].as<at::Tensor>();

  at::Tensor ref_out = reference_mha(inp_tensor, sharded_mha_w0);

  testValidate(
      executor_cache.fusion(),
      {nvf_out},
      {inp_tensor, sharded_mha_w0},
      {ref_out},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
