// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <executor.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<bool, DataType>> {
 protected:
  DistributedTransformerTest()
      : D(communicator_->size()),
        optimization_guard_(false),
        allocation_order_guard_(false) {
    NVF_CHECK(E % H == 0);
    NVF_CHECK(H % D == 0);
    NVF_CHECK(E % D == 0);
  }

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  hir::HostIrExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};
  const int D;
  const int B = 5;
  const int E = 32;
  const int H = 4;
  const int S = 32;

 private:
  // Note: `MoveSplitCat` and `AllocationDomain` preseg passes use ID model.
  // `SdpaFwdOp` currently does not work with ID model since it requires all
  // sibling outputs to have the same root domain.
  //  This will be modified in a future PR.
  preseg_passes::OptimizationPassGuard<preseg_passes::MoveSplitCatPass>
      optimization_guard_;
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      allocation_order_guard_;
};

namespace {
TensorView* replicated_dropout(
    TensorView* x,
    const double kProb,
    Fusion* fusion,
    DeviceMesh mesh) {
  // Need to modify two things before we can use the existing dropout function
  // in composite.cpp (1) Sharding propagation breaks at rand_like because it
  // creates a fresh TV. (2) The philox seed and offset must be set to ensure
  // the masks are identical across processes.
  TensorView* x_float = castOp(DataType::Float, x);
  const double kScale = 1.0 / (1.0 - kProb);
  Val* philox_seed = fusion->zeroVal();
  Val* philox_offset = fusion->zeroVal();
  TensorView* rand_vals = rand_like(x_float, philox_seed, philox_offset);
  TensorView* mask = lt(rand_vals, IrBuilder::create<Val>(1.0 - kProb));
  TensorView* apply_mask = mul(x_float, mask);
  TensorView* dropout = mul(apply_mask, IrBuilder::create<Val>(kScale));
  rand_vals->setDeviceMesh(mesh);
  return dropout;
}

void validate_with_prints(
    std::vector<at::Tensor> expected_out,
    std::vector<at::Tensor> out) {
  EXPECT_EQ(expected_out.size(), out.size());
  for (auto i : c10::irange(out.size())) {
    auto all_close = expected_out[i].allclose(
        out[i].to(expected_out[i].dtype()),
        1e-3 * (i + 1),
        1e-3 * (i + 1),
        /*equal_nan=*/true);

    if (!all_close) {
      auto max_error =
          (expected_out[i].sub(out[i])).abs().max().item().to<double>();
      auto max_relative_error = max_error / expected_out[i].abs().max();
      auto error_count = at::sum((expected_out[i].sub(out[i])).abs() > 1e-3 * (i + 1));
      std::cout << "output[" << i << "] max error: " << max_error << std::endl;
      std::cout << "          max relative error: " << max_relative_error
                << std::endl;
      std::cout << error_count << " elements failing " << std::endl;// << float(error_count) / at::numel(out[i]) << "\% of tensor" << std::endl;
      EXPECT_TRUE(all_close);
    }
  }
}
} // namespace

TEST_P(DistributedTransformerTest, MLP_Layer) {
  auto [use_aten_matmul, dtype] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* x = makeContigTensor(2, dtype);
  TensorView* w0 = makeContigTensor(3, dtype);
  TensorView* b0 = makeContigTensor(2, dtype);
  TensorView* w1 = makeContigTensor(3, dtype);
  TensorView* b1 = makeContigTensor(1, dtype);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  // Linear #1
  TensorView* matmul1;
  if (use_aten_matmul) {
    // TODO: use linear op instead
    TensorView* w0_t = transpose(w0, 2, 1);
    matmul1 = matmul(x, w0_t);
  } else {
    TensorView* linear_int0 = broadcast(x, {true, false, true, false});
    TensorView* linear_int1 = broadcast(w0, {false, true, false, false});
    TensorView* linear_int2 = mul(linear_int0, linear_int1);
    matmul1 = sum(linear_int2, {-1});
    // TODO: linear_int0 has a bcast device axis that the sharding propagation
    // pass misses.
    linear_int0->setDeviceMesh(mesh);
    linear_int0->axis(0)->parallelize(ParallelType::DIDx);
  }
  TensorView* b0_bcast = broadcast(b0, {false, true, false});
  TensorView* linear1 = add(matmul1, b0_bcast);

  TensorView* linear1_ = castOp(DataType::Float, linear1);
  TensorView* gelu = tanh_gelu(linear1_);
  TensorView* gelu_ = castOp(dtype, gelu);

  // Linear #2
  TensorView* local_matmul2;
  if (use_aten_matmul) {
    TensorView* w1_t = transpose(w1, 1, 2);
    local_matmul2 = matmul(gelu_, w1_t);
  } else {
    // segment_set required to ensure the matmul scheduler is called
    gelu_ = segment_set(gelu_);
    TensorView* linear2_int0 = broadcast(gelu_, {false, false, true, false});
    TensorView* linear2_int1 = broadcast(w1, {false, true, false, false});
    TensorView* linear2_int2 = mul(linear2_int0, linear2_int1);
    local_matmul2 = sum(linear2_int2, {-1});
  }

  TensorView* matmul2 = sum(local_matmul2, {0}); // Allreduce
  TensorView* bcast_bias = broadcast(b1, {true, false});
  TensorView* linear2 = add(matmul2, bcast_bias);

  // Dropout
  const double kProb = 0.1;
  TensorView* dropout = replicated_dropout(linear2, kProb, fusion.get(), mesh);

  fusion->addOutput(linear1);
  fusion->addOutput(gelu);
  fusion->addOutput(linear2);
  fusion->addOutput(dropout);

  // Manually shard inputs: x, w0, b0, w1, b1
  // outputs: linear1, gelu, linear2, dropout
  // TVs where sharding changes: matmul2
  // (TODO) TVs where sharding propagation breaks down:
  // linear_int0: broadcasts where a device dim axis is broadcasted.
  // rand_vals: rand_like creates a fresh new TV.

  // TVs replicated on each device.
  auto tv_inputs = {x, b1, matmul2, linear2, dropout};
  for (auto tv : tv_inputs) {
    tv->setDeviceMesh(mesh);
  }

  // TVs sharded on the outermost dimension.
  auto tvs = {w0, b0, w1, linear1, gelu, gelu_};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions().dtype(at_dtype).device(
      at::kCUDA, communicator_->local_rank());
  auto x_ = at::randn({B * S, H}, options);
  auto w0_ = at::randn({4 * H, H}, options);
  auto b0_ = at::randn({4 * H}, options);
  auto w1_ = at::randn({H, 4 * H}, options);
  auto b1_ = at::randn({H}, options);

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(w0_.view({D, 4 * H / D, H}), w0, communicator_->deviceId()),
      shardTensor(b0_.view({D, 4 * H / D}), b0, communicator_->deviceId()),
      shardTensor(
          w1_.view({H, D, 4 * H / D}).transpose(1, 0),
          w1,
          communicator_->deviceId()),
      b1_};
  at::manual_seed(0);
  auto linear1_aten = at::matmul(x_, w0_.transpose(1, 0)).add(b0_);
  auto gelu_aten = at::gelu(linear1_aten.to(at::kFloat), "tanh");
  auto linear2_aten =
      at::matmul(gelu_aten.to(at_dtype), w1_.transpose(1, 0)).add(b1_);
  auto dropout_aten = at::dropout(linear2_aten.to(at::kFloat), kProb, true);
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(
          at::transpose(linear1_aten.view({B * S, D, 4 * H / D}), 1, 0),
          linear1,
          communicator_->deviceId()),
      shardTensor(
          at::transpose(gelu_aten.view({B * S, D, 4 * H / D}), 1, 0),
          gelu,
          communicator_->deviceId()),
      linear2_aten,
      dropout_aten};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate_with_prints(expected_outputs, outputs);
}

TEST_F(DistributedTransformerTest, Split) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  TensorView* x = makeContigConcreteTensor({D, 8, 8});
  TensorView* x_slice0 = slice(x, {0, 0, 0}, {D, 8, 4});
  TensorView* x_slice1 = slice(x, {0, 0, 4}, {D, 8, 8});

  fusion->addInput(x);
  fusion->addOutput(x_slice0);
  fusion->addOutput(x_slice1);

  for (auto tv : {x, x_slice0, x_slice1}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions().device(
      at::kCUDA, communicator_->local_rank());
  auto x_ = at::randn({D, 8, 8}, options);
  auto expected_out = x_.split(4, 2);
  std::vector<c10::IValue> inputs = {{shardTensor(x_, x, communicator_->deviceId())}};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto out = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      out,
      inputs,
      {shardTensor(expected_out[0], x, communicator_->deviceId()),
       shardTensor(expected_out[1], x, communicator_->deviceId())},
      __LINE__,
      __FILE__);

}

TEST_P(DistributedTransformerTest, Multiheaded_Attention) {
  auto [_, dtype] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* x = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* w0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* b0 = makeContigConcreteTensor({D, 3 * E/D}, dtype);
  TensorView* w1 = makeContigConcreteTensor({D, E/D, E}, dtype);
  TensorView* b1 = makeContigConcreteTensor({E}, dtype);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  // linear #1 weight/bias is sharded along the heads, which is concatenated
  // into the column dimension (3*E). Non-reduction axis.
  // linear #2 is also sharded along the heads which is the row (reduction E). 
  // Self attention linear
  TensorView* mm = matmul(x, w0);
  TensorView* proj_bias_bcast = broadcast(b0, {false, true, false});
  TensorView* qkv1 = add(mm, proj_bias_bcast);
  // Forming the q,k,v vectors:
  TensorView* qkv = reshape(qkv1, {D, B * S, 3 * E/D}, {D, B, S, 3 * E/D});
  std::vector<TensorView*> qkv_reshaped = {};
  for (auto i : c10::irange(3)) {
    TensorView* tv_slice = slice(qkv, {0, 0, 0, E/D * i}, {D, B, S, E/D * (i + 1)});
    // Reshape all the vectors into (B,S,E) -> (B,S,H,E/H) -> (B,H,S,E/H)
    TensorView* tv_reshape = reshape(tv_slice, {D, B, S, E/D}, {D, B, S, H/D, E/H});
    TensorView* tv_trans = transpose(tv_reshape, 2, 3); // D, B, H/D, S, E/H
    TensorView* tv_cast = castOp(dtype, tv_trans);
    qkv_reshaped.push_back(tv_cast);
    // Explicitly shard qkv before calling SDPA node
    // so that we can filter out the device dimensions logical domain when
    // checking sizes. This is temporary until we allow DID parallelization on the loop domain.
    for (auto tv : {tv_slice, tv_reshape, tv_trans, tv_cast}) {
      tv->setDeviceMesh(mesh);
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }

  // SDPA
  constexpr double kProb = 0.0;
  constexpr double kScale = 1.0 / (1.0 - kProb);
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv_reshaped[0],
      qkv_reshaped[1],
      qkv_reshaped[2],
      IrBuilder::create<Val>(kProb),
      IrBuilder::create<Val>(true),
      IrBuilder::create<Val>(kScale));
  TensorView* sdpa_output = sdpa.output; // D, B, H/D, S, E/H

  // Linear projection
  TensorView* sdpa_transpose = transpose(sdpa_output, 2, 3); // D, B, S, H/D, E/H
  // Note: We have to reshape into a 2D tensor instead of 3D
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {D, B, S, H/D, E/H}, {D, B * S, E/D});
  TensorView* mm2 = matmul(sdpa_reshape, w1); // D, B*S, E
  TensorView* mm2_ar = sum(mm2, {0}); // allreduce B*S, E
  TensorView* b1_bcast = broadcast(b1, {true, false});
  TensorView* linear2 = add(mm2_ar, b1_bcast);

  // // Dropout
  const double kDropoutProb = 0.1;
  TensorView* dropout =
      replicated_dropout(linear2, kDropoutProb, fusion.get(), mesh);

  fusion->addOutput(qkv);
  fusion->addOutput(qkv_reshaped[0]);
  fusion->addOutput(qkv_reshaped[1]);
  fusion->addOutput(qkv_reshaped[2]);
  fusion->addOutput(sdpa_output);
  fusion->addOutput(linear2);
  fusion->addOutput(dropout);

  for (auto tv : {x, b1, mm2_ar, linear2, dropout}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv : {w0, b0, w1, mm, mm2, proj_bias_bcast, qkv, sdpa_output}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions().dtype(at_dtype).device(
      at::kCUDA, communicator_->local_rank());
  at::manual_seed(0);
  // auto x_ = at::arange(0, B*S*E, 1, options).view({B * S, E}) / 100;
  auto x_ = at::randn({B*S, E}, options);
  auto w0_ = at::randn({E, 3 * E}, options) / 10.0;
  auto b0_ = at::randn({3 * E}, options) / 10.0;
  auto w1_ = at::ones({E, E}, options);
  auto b1_ = at::ones({E}, options);
  auto m_ = at::matmul(x_, w0_).add(b0_).view({B, S, 3 * E});
  std::cout << m_.sizes() << std::endl;
  auto qkv_vec = m_.split(E, 2);
  // move vectors from (B, S, E) to (B, S, H, E/H) to (B, H, S, E/H)
  for (auto i = 0; i < 3; i++) {
    qkv_vec[i] =
        qkv_vec[i].reshape({B, S, H, E / H}).transpose(2, 1).to(at_dtype);
  }

  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv_vec[0], qkv_vec[1], qkv_vec[2], kProb, true, false, kScale);
  auto sdpa_ = std::get<0>(sdpa_out);
  // Reassemble heads (B, H, S, E/H) to (B, S, H, E/H) to (B, S, E)
  auto y = sdpa_.transpose(1, 2).reshape({B * S, E});
  auto y_proj = at::matmul(y, w1_).add(b1_);
  at::manual_seed(0);
  auto y_dropout = at::dropout(y_proj.to(at::kFloat), kDropoutProb, true);

  std::vector<c10::IValue> inputs = {x_, 
    shardTensor(w0_.view({E, D, 3*E/D}).transpose(0,1), w0, communicator_->deviceId()),
    shardTensor(b0_.view({D, 3*E/D}), b0, communicator_->deviceId()), 
    shardTensor(w1_.view({D, E/D, E}), w1, communicator_->deviceId()),
    b1_};
  // qkv_vec {B, H, S, E/H} ---> {H, B, S, E/H} ---> {D, H/D, B, S, E/H} -shard--> {D, B, H/D, S, E/H}
  auto q_aten = shardTensor(qkv_vec[0].transpose(0,1).view({D, H/D, B, S, E/H}), qkv_reshaped[0], communicator_->deviceId()).transpose(1,2);
  std::cout << "True DeviceId " << communicator_->deviceId() << " " << q_aten[0][0][0] << std::endl;
  std::vector<at::Tensor> expected_outputs = {
    shardTensor(m_.view({B, S, D, 3*E/D}).permute({2, 0, 1, 3}), qkv, communicator_->deviceId()),    
    shardTensor(qkv_vec[0].transpose(0,1).view({D, H/D, B, S, E/H}), qkv_reshaped[0], communicator_->deviceId()).transpose(1,2),
    shardTensor(qkv_vec[1].transpose(0,1).view({D, H/D, B, S, E/H}), qkv_reshaped[1], communicator_->deviceId()).transpose(1,2),
    shardTensor(qkv_vec[2].transpose(0,1).view({D, H/D, B, S, E/H}), qkv_reshaped[2], communicator_->deviceId()).transpose(1,2),
    // sdpa_ = {B, H, S, E/H} --> {H, B, S, E/H} --> {D, H/D, B, S, E/H} --> {1, H/D, B, S, E/H} --> {1, B, H/D, S, E/H}
    shardTensor(sdpa_.transpose(0,1).view({D, H/D, B, S, E/H}), sdpa_output, communicator_->deviceId()).transpose(1,2),
    y_proj,
    y_dropout};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto out = runtime.runWithInput(inputs);
  std::cout << "NV DeviceId " << communicator_->deviceId() << " " << out[1][0][0][0] << std::endl;
  validate_with_prints(expected_outputs, out);
};

// INSTANTIATE_TEST_SUITE_P(
//     ,
//     DistributedTransformerTest,
//     testing::Bool(),
//     testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
    aten,
    DistributedTransformerTest,
    testing::Combine(
        testing::Values(true),
        testing::Values(
            DataType::Double,
            DataType::Float,
            DataType::Half,
            DataType::BFloat16)));
INSTANTIATE_TEST_SUITE_P(
    nvfuser,
    DistributedTransformerTest,
    testing::Combine(
        testing::Values(false),
        testing::Values(DataType::Half, DataType::BFloat16)));
} // namespace nvfuser
