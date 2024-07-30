// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cmath>

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
#include <preseg_passes/mark_aliases_prepare.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {
int64_t D = 1;
}
constexpr int64_t B = 2, E = 768, H = 12, S = 128;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
constexpr double kDropoutProb = 0.1, kParamScale = 0.02, kSdpaProb = 0.1,
                 kSdpaScale = 1e-3;

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest()
      : optimization_guard_(false),
        allocation_order_guard_(false),
        alias_guard_(false) {
    D = communicator_->size();
  }

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (H % D != 0) {
      GTEST_SKIP()
          << "Distributed transformer tests require number of devices evenly divide E ";
    }
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  hir::HostIrExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};

 private:
  // Note: `MoveSplitCat` and `AllocationDomain` preseg passes use ID model.
  // `SdpaFwdOp` currently does not work with ID model since it requires all
  // sibling outputs to have the same root domain.
  //  This will be modified in a future PR.
  preseg_passes::OptimizationPassGuard<preseg_passes::MoveSplitCatPass>
      optimization_guard_;
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      allocation_order_guard_;
  preseg_passes::OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      alias_guard_;
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

void validate(
    std::vector<at::Tensor> expected_out,
    std::vector<at::Tensor> out) {
  EXPECT_EQ(expected_out.size(), out.size());
  for (auto i : c10::irange(out.size())) {
    // Note: Scaling tolerance up since the error accumulates across ops
    // BFloat16 error is quite high, but the program has been verified with
    // double precision to be logically correct.
    double atol = 0.5 * (i + 1);
    double rtol = 1.6e-2;
    auto all_close = out[i]
                         .to(expected_out[i].dtype())
                         .allclose(
                             expected_out[i],
                             rtol,
                             atol,
                             /*equal_nan=*/true);

    if (!all_close) {
      auto error = (out[i].to(expected_out[i].dtype()) - expected_out[i]).abs();
      auto max_error = error.max().item().to<double>();
      auto max_relative_error =
          (max_error / expected_out[i].abs().max()).item();
      auto error_count =
          at::sum(error >= (atol + expected_out[i].abs() * rtol)).item();
      std::cout << "output[" << i << "] max error: " << max_error << std::endl;
      std::cout << "          max relative error: " << max_relative_error
                << std::endl;
      std::cout << "          failing elements: " << error_count << ", "
                << error_count.to<float>() / at::numel(out[i]) * 100.0
                << "\% of tensor" << std::endl;
    }
    EXPECT_TRUE(all_close);
  }
}

std::vector<at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  auto linear0 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear0, "tanh");
  auto linear1 = at::matmul(gelu.to(at_dtype), w1).add(b1).to(at::kFloat);
  auto dropout = at::dropout(linear1, kDropoutProb, true);
  return {linear0, gelu, linear1, dropout};
}

std::vector<at::Tensor> reference_mha(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  auto m = at::matmul(x, w0).add(b0).view({B, S, 3 * E});
  auto qkv_vec = m.split(E, 2);
  for (auto i = 0; i < 3; i++) {
    qkv_vec[i] =
        qkv_vec[i].reshape({B, S, H, E / H}).transpose(2, 1).to(at_dtype);
  }
  at::manual_seed(0);
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv_vec[0], qkv_vec[1], qkv_vec[2], kSdpaProb, true, false, kSdpaScale);
  auto sdpa = std::get<0>(sdpa_out);
  // Reassemble heads (B, H, S, E/H) to (B, S, H, E/H) to (B, S, E)
  auto y = sdpa.transpose(1, 2).reshape({B * S, E});
  auto y_proj = at::matmul(y, w1).add(b1);
  at::manual_seed(0);
  auto y_dropout = at::dropout(y_proj.to(at::kFloat), kDropoutProb, true);
  return {m, sdpa, y_proj, y_dropout};
}

std::vector<at::Tensor> reference_mlp_backwards(
    at::Tensor grad,
    at::Tensor x,
    at::Tensor mask,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::ScalarType at_dtype) {
  // recompute activations
  auto linear0 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear0, "tanh");

  // backwards pass
  auto dropout_grad =
      at::native_dropout_backward(grad, mask, 1.0 / (1.0 - kDropoutProb));
  auto dropout_grad_q = dropout_grad.to(at_dtype);
  auto matmul1_grad = at::matmul(dropout_grad_q, w1.transpose(0, 1));
  auto matmul1_grad_w =
      at::matmul(dropout_grad_q.transpose(0, 1), gelu.to(at_dtype))
          .transpose(0, 1);
  auto matmul1_grad_b = at::sum(dropout_grad, {0});
  auto gelu_grad =
      at::gelu_backward(matmul1_grad.to(at::kFloat), linear0, "tanh");
  auto gelu_grad_q = gelu_grad.to(at_dtype);
  auto matmul0_grad_b = at::sum(gelu_grad, {0});
  auto matmul0_grad = at::matmul(gelu_grad_q, w0.transpose(0, 1));
  auto matmul0_grad_w =
      at::matmul(gelu_grad_q.transpose(0, 1), x).transpose(0, 1);

  std::vector<at::Tensor> grads = {
      dropout_grad,
      matmul1_grad_w,
      matmul1_grad_b,
      gelu_grad,
      matmul0_grad_w,
      matmul0_grad_b,
      matmul0_grad};
  return grads;
}

std::vector<TensorView*> mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    Fusion* fusion,
    DeviceMesh& mesh,
    DataType dtype) {
  // Linear #1
  TensorView* matmul1 = matmul(x, w0);
  TensorView* b0_bcast = broadcast(b0, {false, true, false});
  TensorView* linear1 = add(matmul1, b0_bcast);
  // GeLU
  TensorView* linear1_ = castOp(DataType::Float, linear1);
  TensorView* gelu = tanh_gelu(linear1_);
  TensorView* gelu_ = castOp(dtype, gelu);
  // Linear #2
  TensorView* local_matmul2 = matmul(gelu_, w1);
  TensorView* matmul2 = sum(local_matmul2, {0}); // Allreduce
  TensorView* bcast_bias = broadcast(b1, {true, false});
  TensorView* linear2 = add(matmul2, bcast_bias);
  // Dropout
  TensorView* dropout = replicated_dropout(linear2, kDropoutProb, fusion, mesh);

  // Sharding
  // (TODO) TVs where sharding propagation breaks down:
  // linear_int0: broadcasts where a device dim axis is broadcasted.
  // rand_vals: rand_like creates a fresh new TV.
  // TVs replicated on each device.
  for (auto tv : {x, b1, matmul2, linear2, dropout}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear1, gelu}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  return {linear1, gelu, linear2, dropout};
}

std::vector<TensorView*> mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    Fusion* fusion,
    DeviceMesh& mesh,
    DataType dtype) {
  // Linear 1
  TensorView* mm = matmul(x, w0);
  TensorView* proj_bias_bcast = broadcast(b0, {false, true, false});
  TensorView* qkv1 = add(mm, proj_bias_bcast);
  // Forming the q,k,v vectors:
  TensorView* qkv = reshape(qkv1, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv_reshaped = {};
  for (auto i : c10::irange(3)) {
    TensorView* tv_slice =
        slice(qkv, {0, 0, 0, E / D * i}, {D, B, S, E / D * (i + 1)});
    TensorView* tv_reshape =
        reshape(tv_slice, {D, B, S, E / D}, {D, B, S, H / D, E / H});
    TensorView* tv_trans = transpose(tv_reshape, 2, 3); // D, B, H/D, S, E/H
    TensorView* tv_cast = castOp(dtype, tv_trans);
    qkv_reshaped.push_back(tv_cast);
    // Explicitly shard qkv before calling SDPA node
    for (auto tv : {tv_slice, tv_reshape, tv_trans, tv_cast}) {
      tv->setDeviceMesh(mesh);
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }
  // SDPA
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv_reshaped[0],
      qkv_reshaped[1],
      qkv_reshaped[2],
      IrBuilder::create<Val>(kSdpaProb),
      IrBuilder::create<Val>(true),
      IrBuilder::create<Val>(kSdpaScale));
  TensorView* sdpa_output = sdpa.output; // D, B, H/D, S, E/H
  // Linear projection
  TensorView* sdpa_transpose =
      transpose(sdpa_output, 2, 3); // D, B, S, H/D, E/H
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* mm2 = matmul(sdpa_reshape, w1); // D, B*S, E/D * D, E/D, E
  TensorView* mm2_ar = sum(mm2, {0}); // allreduce rD, B*S, E
  TensorView* b1_bcast = broadcast(b1, {true, false});
  TensorView* linear2 = add(mm2_ar, b1_bcast);
  // Dropout
  TensorView* dropout = replicated_dropout(linear2, kDropoutProb, fusion, mesh);

  for (auto tv : {x, b1, mm2_ar, linear2, dropout}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, proj_bias_bcast, mm, mm2, qkv, sdpa_output}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {qkv, sdpa_output, linear2, dropout};
}

std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    Fusion* fusion,
    DeviceMesh& mesh,
    DataType dtype) {
  // Activation recomputation
  TensorView* matmul0 = matmul(x, w0);
  TensorView* b0_bcast = broadcast(b0, {false, true, false});
  TensorView* linear0 = add(matmul0, b0_bcast);
  linear0 = castOp(DataType::Float, linear0);
  TensorView* gelu = tanh_gelu(linear0);
  gelu = castOp(dtype, gelu);

  // Backwards pass
  constexpr double kScale = 1.0 / (1.0 - kDropoutProb);
  Val* dscale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dscale);
  TensorView* dropout_grad_q = castOp(dtype, dropout_grad);

  TensorView* w1_t = transpose(w1, 1, 2);
  TensorView* matmul1_grad_x = matmul(dropout_grad_q, w1_t);
  TensorView* grad_t = transpose(dropout_grad_q, 0, 1);
  TensorView* matmul1_grad_w_t = matmul(grad_t, gelu);
  TensorView* matmul1_grad_w = transpose(matmul1_grad_w_t, 1, 2);
  TensorView* matmul1_grad_b = sum(dropout_grad, {0});

  TensorView* matmul1_grad_x_ = castOp(DataType::Float, matmul1_grad_x);
  TensorView* gelu_grad = tanh_gelu_backward(matmul1_grad_x_, linear0);
  TensorView* gelu_grad_f = castOp(dtype, gelu_grad);

  TensorView* w0_t = transpose(w0, 1, 2);
  TensorView* matmul0_grad_x_partial = matmul(gelu_grad_f, w0_t);
  TensorView* matmul0_grad_x = sum(matmul0_grad_x_partial, {0}); // allreduce
  TensorView* grad_gelu_t = transpose(gelu_grad_f, 1, 2);
  TensorView* matmul0_grad_w_t = matmul(grad_gelu_t, x);
  TensorView* matmul0_grad_w = transpose(matmul0_grad_w_t, 1, 2);
  TensorView* matmul0_grad_b = sum(gelu_grad, {1});

  for (auto tv :
       {x,
        grad,
        mask,
        dropout_grad,
        matmul1_grad_x,
        matmul1_grad_b,
        matmul0_grad_x}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv :
       {w0,
        b0,
        w1,
        matmul0,
        matmul1_grad_x,
        matmul1_grad_w,
        matmul1_grad_w_t,
        gelu_grad,
        matmul0_grad_w_t,
        matmul0_grad_w,
        matmul0_grad_x_partial,
        matmul0_grad_b}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  std::vector<TensorView*> outputs = {
      dropout_grad,
      matmul1_grad_w,
      matmul1_grad_b,
      gelu_grad,
      matmul0_grad_w,
      matmul0_grad_b,
      matmul0_grad_x};
  return outputs;
}
} // namespace

TEST_P(DistributedTransformerTest, MLP_Layer) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* tvx = makeContigTensor(2, dtype);
  TensorView* tvw0 = makeContigTensor(3, dtype);
  TensorView* tvb0 = makeContigTensor(2, dtype);
  TensorView* tvw1 = makeContigTensor(3, dtype);
  TensorView* tvb1 = makeContigTensor(1, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  std::vector<TensorView*> tvsout =
      mlp(tvx, tvw0, tvb0, tvw1, tvb1, fusion.get(), mesh, dtype);

  for (TensorView* tv : tvsout) {
    fusion->addOutput(tv);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 4 * E}, options) * kParamScale;
  auto b0 = at::randn({4 * E}, options) * kParamScale;
  auto w1 = at::randn({4 * E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  std::vector<at::Tensor> reference_outs =
      reference_mlp(x, w0, b0, w1, b1, at_dtype);

  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0, 1, mesh),
      shardTensor(b0, 0, mesh),
      shardTensor(w1, 0, mesh),
      b1};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  at::manual_seed(getATenRandomSeed());
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

TEST_P(DistributedTransformerTest, Multiheaded_Attention) {
  auto dtype = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* tvx = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* tvb0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* tvb1 = makeContigConcreteTensor({E}, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  auto tv_outs = mha(tvx, tvw0, tvb0, tvw1, tvb1, fusion.get(), mesh, dtype);

  for (auto tv : tv_outs) {
    fusion->addOutput(tv);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options) * .05;
  auto w0 = at::randn({E, 3 * E}, options) * kParamScale;
  auto b0 = at::randn({3 * E}, options) * kParamScale;
  auto w1 = at::randn({E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  auto reference_outs = reference_mha(x, w0, b0, w1, b1, at_dtype);

  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0.view({E, 3, E}), 2, mesh)
          .view({1, E, 3 * E / D})
          .contiguous(),
      shardTensor(b0.view({3, E}), 1, mesh).view({1, 3 * E / D}).contiguous(),
      shardTensor(w1, 0, mesh),
      b1};
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0].view({B, S, 3, E}), 3, mesh)
          .view({1, B, S, 3 * E / D})
          .contiguous(),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  at::manual_seed(0);
  auto out = runtime.runWithInput(inputs);
  validate(expected_outputs, out);
}

TEST_P(DistributedTransformerTest, MLP_Backward) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* grad = makeContigTensor(2, DataType::Float);
  TensorView* x = makeContigTensor(2, dtype);
  TensorView* mask = makeContigTensor(2, DataType::Bool);
  TensorView* w0 = makeContigTensor(3, dtype);
  TensorView* b0 = makeContigTensor(2, dtype);
  TensorView* w1 = makeContigTensor(3, dtype);

  fusion->addInput(grad);
  fusion->addInput(x);
  fusion->addInput(mask);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);

  std::vector<TensorView*> tv_outs =
      mlp_backwards(grad, x, mask, w0, b0, w1, fusion.get(), mesh, dtype);

  for (TensorView* tv : tv_outs) {
    fusion->addOutput(tv);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto grad_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto x_ = at::randn({B * S, E}, options);
  auto mask_ = at::randn({B * S, E}, options).lt(1.0 - kDropoutProb);
  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;

  std::vector<at::Tensor> outs = reference_mlp_backwards(
      grad_, x_, mask_, mlp_w0_, mlp_b0_, mlp_w1_, at_dtype);

  std::vector<c10::IValue> inputs = {
      grad_,
      x_,
      mask_,
      shardTensor(mlp_w0_, 1, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 0, mesh)};
  std::vector<at::Tensor> expected_outputs = {
      outs[0], // dropout grad
      shardTensor(outs[1], 0, mesh), // linear1 weight grad
      outs[2], // linear1 bias grad
      shardTensor(outs[3], 1, mesh), // gelu grad
      shardTensor(outs[4], 1, mesh), // linear0 weight grad
      shardTensor(outs[5], 0, mesh), // linear0 bias grad
      outs[6]}; // linear0 grad

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

TEST_P(DistributedTransformerTest, Forward) {
  auto dtype = DataType::Half;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = makeContigConcreteTensor({B * S, E}, DataType::Float);
  TensorView* mha_w0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* mha_b1 = makeContigConcreteTensor({E}, dtype);
  TensorView* mlp_w0 = makeContigTensor(3, dtype);
  TensorView* mlp_b0 = makeContigTensor(2, dtype);
  TensorView* mlp_w1 = makeContigTensor(3, dtype);
  TensorView* mlp_b1 = makeContigTensor(1, dtype);

  fusion->addInput(x);
  fusion->addInput(mha_w0);
  fusion->addInput(mha_b0);
  fusion->addInput(mha_w1);
  fusion->addInput(mha_b1);
  fusion->addInput(mlp_w0);
  fusion->addInput(mlp_b0);
  fusion->addInput(mlp_w1);
  fusion->addInput(mlp_b1);

  constexpr float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);
  std::vector<int64_t> norm_shape{E};

  auto ln_1 = layer_norm(x, norm_shape, nullptr, nullptr, eps_ptr);
  auto mha_in = castOp(dtype, ln_1.output);
  auto mha_out =
      mha(mha_in, mha_w0, mha_b0, mha_w1, mha_b1, fusion.get(), mesh, dtype);
  auto resid_1 = add(x, mha_out[3]);
  auto ln_2 = layer_norm(resid_1, norm_shape, nullptr, nullptr, eps_ptr);
  auto mlp_in = castOp(dtype, ln_2.output);
  auto mlp_out =
      mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, fusion.get(), mesh, dtype);
  auto resid_2 = add(mha_out[3], mlp_out[3]);

  fusion->addOutput(ln_1.output);
  fusion->addOutput(mha_out[3]);
  fusion->addOutput(ln_2.output);
  fusion->addOutput(mlp_out[3]);
  fusion->addOutput(resid_2);

  for (auto tv : {x, ln_1.output, ln_2.output, resid_2}) {
    tv->setDeviceMesh(mesh);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto mha_w0_ = at::randn({E, 3 * E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;

  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  auto at_weight = c10::optional<at::Tensor>();
  auto at_bias = c10::optional<at::Tensor>();
  auto ln_1_ = at::native_layer_norm(x_, norm_shape, at_weight, at_bias, kEps);
  auto ln_1_out_ = std::get<0>(ln_1_).to(at_dtype);

  auto mha_out_ =
      reference_mha(ln_1_out_, mha_w0_, mha_b0_, mha_w1_, mha_b1_, at_dtype);
  auto resid1_ = mha_out_[3] + x_;
  auto ln_2_ =
      at::native_layer_norm(resid1_, norm_shape, at_weight, at_bias, kEps);
  auto ln_2_out_ = std::get<0>(ln_2_).to(at_dtype);

  auto mlp_out_ =
      reference_mlp(ln_2_out_, mlp_w0_, mlp_b0_, mlp_w1_, mlp_b1_, at_dtype);
  auto at_out = mha_out_[3] + mlp_out_[3];

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(mha_w0_.view({E, 3, E}), 2, mesh)
          .view({1, E, 3 * E / D})
          .contiguous(),
      shardTensor(mha_b0_.view({3, E}), 1, mesh)
          .view({1, 3 * E / D})
          .contiguous(),
      shardTensor(mha_w1_, 0, mesh),
      mha_b1_,
      shardTensor(mlp_w0_, 1, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 0, mesh),
      mlp_b1_};

  std::vector<at::Tensor> expected_outputs = {
      ln_1_out_, mha_out_[3], ln_2_out_, mlp_out_[3], at_out};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedTransformerTest,
    testing::Values(DataType::Half, DataType::BFloat16));
} // namespace nvfuser
