// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

constexpr int64_t B = 2, E = 768, H = 16, S = 128;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
// Note: Sdpa probability is set to 0. Since the dropout mask is sharded it
// throws off the seed offset between the sharded nvFuser program and the
// unsharded reference.
constexpr double kDropoutProb = 0.1, kParamScale = 0.02, kSdpaProb = 0.0,
                 kSdpaScale = 1e-3;

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest() : D(communicator_->size()) {}

  void SetUp() override {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  const int64_t D; // number of devices
};

namespace {
// testValidate doesn't work out of the box due to #2906, so I had to manually
// specify the absolute tolerances. The atols passed in are tuned for bfloat,
// the least precise dtype. They can probably be made stricter for other
// dtypes.
void validate(
    const std::vector<at::Tensor>& expected_outputs,
    const std::vector<at::Tensor>& outputs,
    const std::vector<double>& atols) {
  using testing::SizeIs;
  const auto num_outputs = outputs.size();
  ASSERT_THAT(expected_outputs, SizeIs(num_outputs));
  ASSERT_THAT(atols, SizeIs(num_outputs));

  for (const auto i : c10::irange(num_outputs)) {
    // allclose can catch this as well. However, it would throw an exception,
    // not showing which output was problematic.
    ASSERT_EQ(outputs[i].dtype(), expected_outputs[i].dtype())
        << "Output " << i << " has a mismatching data type.";

    const double atol = atols[i];
    // These default rtols are copied from
    // https://github.com/pytorch/pytorch/blob/951c21d6790334d57862e94a3f582ac724147a53/torch/testing/_comparison.py#L65-L73.
    double rtol;
    switch (outputs[i].scalar_type()) {
      case at::kBFloat16:
        rtol = 1.6e-2;
        break;
      case at::kHalf:
        rtol = 1e-3;
        break;
      case at::kFloat:
        rtol = 1.3e-6;
        break;
      default:
        rtol = 0.0;
        break;
    }

    auto generate_comparison_details = [](at::Tensor expected_out,
                                          at::Tensor out,
                                          double atol,
                                          double rtol) -> std::string {
      std::ostringstream oss;
      auto error = (out - expected_out).abs();
      auto max_relative_error =
          (error.max() / expected_out.abs().max()).item().to<double>();
      auto error_count =
          at::sum(error >= atol + expected_out.abs() * rtol).item();
      indent(oss, 1)
          << "max absolute error under rtol: "
          << (error - expected_out.abs() * rtol).max().item().to<double>()
          << std::endl;
      indent(oss, 1) << "max relative error: " << max_relative_error
                     << std::endl;
      indent(oss, 1) << "failing elements: " << error_count << ", "
                     << error_count.to<float>() / at::numel(out) * 100.0
                     << "\% of tensor";
      return oss.str();
    };

    EXPECT_TRUE(outputs[i].allclose(expected_outputs[i], rtol, atol))
        << "Output " << i << " mismatches with atol " << atol << ":"
        << std::endl
        << generate_comparison_details(
               expected_outputs[i], outputs[i], atol, rtol);
  }
}

std::vector<at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1) {
  auto at_dtype = w0.dtype();
  auto linear0 = at::linear(x, w0, b0);
  auto gelu = at::gelu(linear0.to(at::kFloat), "tanh").to(at_dtype);
  auto linear1 = at::linear(gelu, w1, b1).to(at::kFloat);
  auto [dropout, mask] = at::native_dropout(linear1, kDropoutProb, true);
  return {linear0, gelu, linear1, dropout, mask};
}

std::vector<at::Tensor> reference_mha(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1) {
  auto linear0 = at::linear(x, w0, b0);
  auto qkv = linear0.view({B, S, 3 * E}).split(E, 2);
  for (auto i = 0; i < 3; i++) {
    qkv[i] = qkv[i].reshape({B, S, H, E / H}).transpose(1, 2);
  }
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv[0], qkv[1], qkv[2], kSdpaProb, true, false, kSdpaScale);
  auto sdpa = std::get<0>(sdpa_out);
  // Reassemble heads (B, H, S, E/H) to (B, S, H, E/H) to (B, S, E)
  auto y = sdpa.transpose(1, 2).reshape({B * S, E});
  auto linear1 = at::linear(y, w1, b1).to(at::kFloat);
  auto [dropout, mask] = at::native_dropout(linear1, kDropoutProb, true);
  return {linear0, sdpa, linear1, dropout, mask};
}

std::vector<at::Tensor> reference_mlp_backwards(
    at::Tensor grad,
    at::Tensor x,
    at::Tensor mask,
    at::Tensor w0,
    at::Tensor w1,
    at::Tensor linear0) {
  auto at_dtype = w0.dtype();
  auto gelu = at::gelu(linear0.to(at::kFloat), "tanh");

  // backwards pass
  auto dropout_grad =
      at::native_dropout_backward(grad, mask, 1.0 / (1.0 - kDropoutProb));
  auto dropout_grad_q = dropout_grad.to(at_dtype);
  auto matmul1_grad = at::matmul(dropout_grad_q, w1);
  auto matmul1_grad_w =
      at::matmul(dropout_grad_q.transpose(0, 1), gelu.to(at_dtype));
  auto matmul1_grad_b = at::sum(dropout_grad, {0}).to(at_dtype);
  auto gelu_grad =
      at::gelu_backward(matmul1_grad.to(at::kFloat), linear0, "tanh");
  auto gelu_grad_q = gelu_grad.to(at_dtype);
  auto matmul0_grad_b = at::sum(gelu_grad, {0}).to(at_dtype);
  auto matmul0_grad = at::matmul(gelu_grad_q, w0);
  auto matmul0_grad_w = at::matmul(gelu_grad_q.transpose(0, 1), x);

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

std::vector<at::Tensor> reference_mha_backwards(
    at::Tensor y_grad,
    at::Tensor x,
    at::Tensor mask,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1) {
  auto at_dtype = w0.dtype();
  // recompute up to sdpa
  auto linear0 = at::linear(x, w0, b0);
  auto qkv = linear0.split(E, /*dim=*/-1);
  for (auto i = 0; i < 3; i++) {
    qkv[i] = qkv[i].reshape({B, S, H, E / H}).transpose(1, 2).to(at_dtype);
  }
  auto
      [sdpa_output,
       log_sumexp,
       cum_seq_q,
       cum_seq_k,
       query_seq_len,
       key_seq_len,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
          at::_scaled_dot_product_flash_attention(
              qkv[0],
              qkv[1],
              qkv[2],
              /*dropout_p=*/kSdpaProb,
              /*is_causal=*/true,
              /*return_debug_mask=*/false,
              /*scale=*/kSdpaScale);

  // backwards pass
  auto dropout_grad =
      at::native_dropout_backward(y_grad, mask, 1.0 / (1.0 - kDropoutProb));
  auto dropout_grad_q = dropout_grad.to(at_dtype);
  auto linear1_x_grad = at::matmul(dropout_grad_q, w1);
  auto sdpa_output_reshape = sdpa_output.transpose(1, 2).view({B * S, E});
  auto linear1_w_grad =
      at::matmul(dropout_grad_q.transpose(0, 1), sdpa_output_reshape);
  auto linear1_b_grad = at::sum(dropout_grad, {0}).to(at_dtype);

  auto [q_grad, k_grad, v_grad] =
      at::_scaled_dot_product_flash_attention_backward(
          linear1_x_grad.view({B, S, H, E / H}).transpose(1, 2),
          qkv[0],
          qkv[1],
          qkv[2],
          sdpa_output,
          log_sumexp,
          cum_seq_q,
          cum_seq_k,
          /*max_q=*/*query_seq_len.maybe_as_int(),
          /*max_k=*/*key_seq_len.maybe_as_int(),
          /*dropout_p=*/kSdpaProb,
          /*is_causal=*/true,
          philox_seed,
          philox_offset,
          /*scale=*/kSdpaScale);
  auto qkv_grad = at::cat(
      {q_grad.transpose(1, 2).view({B * S, E}),
       k_grad.transpose(1, 2).view({B * S, E}),
       v_grad.transpose(1, 2).view({B * S, E})},
      -1);
  auto linear0_b_grad = at::sum(qkv_grad.to(at::kFloat), {0}).to(at_dtype);
  auto linear0_x_grad = at::matmul(qkv_grad, w0);
  auto linear0_w_grad = at::matmul(qkv_grad.transpose(0, 1), x);

  // Note: sdpa_output, sdpa_logsumexp are saved for the backwards pass
  // and become inputs to the nvfuser mha backwards pass
  std::vector<at::Tensor> tensors = {
      sdpa_output,
      log_sumexp,
      philox_seed,
      philox_offset,
      dropout_grad,
      linear1_w_grad,
      linear1_b_grad,
      q_grad,
      k_grad,
      v_grad,
      linear0_w_grad,
      linear0_b_grad,
      linear0_x_grad,
      linear0};
  return tensors;
}

std::vector<TensorView*> mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh,
    bool sequence_parallel = false,
    TensorView* mask = nullptr) {
  const DataType dtype = w0->dtype();

  if (sequence_parallel) {
    // Input arrives sharded and must be allgathered back
    x->setDeviceMesh(mesh);
    x->axis(0)->parallelize(ParallelType::DIDx);
    x = set(x); // allgather
    x->setDeviceMesh(mesh);
    x->axis(0)->parallelize(ParallelType::Serial);
    // Reshape back to 2D for linearOp
    auto D = w0->axis(0)->extent()->value().as<int64_t>();
    x = reshape(x, {D, B * S / D, E}, {B * S, E});
  }
  // Linear 0
  TensorView* linear0 = linear(x, w0, b0);
  // GeLU
  TensorView* gelu = tanh_gelu(castOp(DataType::Float, linear0));
  gelu = castOp(dtype, gelu);
  // Linear 1
  TensorView* local_matmul1 = matmul(gelu, transpose(w1, 1, 2));
  if (sequence_parallel) {
    // Remove after https://github.com/NVIDIA/Fuser/issues/2563
    // Reshape to explicitly pull the sharded axis into the logical domain
    auto D = w0->axis(0)->extent()->value().as<int64_t>();
    local_matmul1 = reshape(local_matmul1, {D, B * S, E}, {D, D, B * S / D, E});
  }
  TensorView* matmul1 = sum(local_matmul1, {0}); // Allreduce or Reduce scatter
  TensorView* linear1 = add(matmul1, broadcast(b1, {true, true, false}));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  if (mask == nullptr) {
    auto rand_vals = rand_like(linear1);
    mask = lt(rand_vals, prob);
  }
  auto apply_mask = mul(linear1, mask);
  auto dropout_result = mul(apply_mask, scale);

  // Tensor parallel shardings
  for (auto* tv : {w0, b0, w1, linear0, gelu}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  for (auto* tv : {x, b1, linear1, dropout_result}) {
    tv->setDeviceMesh(mesh);
  }

  // Sequence parallel shardings
  if (sequence_parallel) {
    for (auto* tv : {linear1, dropout_result}) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
    matmul1->setDeviceMesh(mesh);
    matmul1->axis(1)->parallelize(ParallelType::DIDx);
    mask->setDeviceMesh(mesh);
    mask->axis(0)->parallelize(ParallelType::DIDx);
  }

  return {linear0, gelu, linear1, dropout_result};
}

std::vector<TensorView*> mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh) {
  const auto D = w0->axis(0)->extent()->value().as<int64_t>();
  auto dtype = w0->dtype();

  TensorView* linear0 = linear(x, w0, b0);
  // Note: Remove linear0 and qkv[i] casts
  // when https://github.com/NVIDIA/Fuser/issues/3194 is resolved
  TensorView* linear0_temp = castOp(DataType::Float, linear0);
  // Forming the q,k,v vectors:
  TensorView* qkv_cat =
      reshape(linear0_temp, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : c10::irange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = transpose(qkv[i], 2, 3);
    qkv[i] = castOp(dtype, qkv[i]);
  }
  // SDPA
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv[0],
      qkv[1],
      qkv[2],
      IrBuilder::create<Val>(kSdpaProb),
      IrBuilder::create<Val>(true),
      IrBuilder::create<Val>(kSdpaScale));
  TensorView* sdpa_output = sdpa.output;
  // Linear 1
  TensorView* sdpa_transpose = transpose(sdpa_output, 2, 3);
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* local_matmul1 = matmul(sdpa_reshape, transpose(w1, 1, 2));
  TensorView* matmul1 = sum(local_matmul1, {0}); // allreduce
  TensorView* linear1 = add(matmul1, broadcast(b1, {true, false}));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  auto dropout_result = dropout(linear1, prob, scale).output;

  for (auto tv : {x, b1, matmul1, linear1, dropout_result}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear0, sdpa_output}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {linear0, sdpa_output, linear1, dropout_result};
}

// TODO: These linear_backwards helper functions can be merged once
// we do not have logically split rfactor domain.
struct LinearBackwardsResult {
  TensorView* grad_x;
  TensorView* grad_w;
  TensorView* grad_b;
};

// x format: [i0, i1] dtype
// weight format: [DID(D), i2/D, i1] dtype
// grad format: [DID(D) i0, i2/D] float or dtype
// outputs: grad_x [i0, i1] dtype
// grad_w [DID i2/D, i1] dtype
// grad_b [DID i2/2] dtype
LinearBackwardsResult linear_backwards(
    TensorView* x,
    TensorView* w,
    TensorView* grad) {
  DataType dtype = w->dtype();
  TensorView* grad_f = maybeCastOp(DataType::Float, grad);
  TensorView* grad_q = maybeCastOp(dtype, grad);
  TensorView* grad_x_partials = matmul(grad_q, w);
  TensorView* grad_x = sum(grad_x_partials, {0}); // allreduce
  TensorView* grad_q_t = transpose(grad_q, 1, 2);
  TensorView* grad_w = matmul(grad_q_t, x);
  TensorView* grad_b = sum(grad_f, {1});
  grad_b = castOp(dtype, grad_b);

  return {grad_x, grad_w, grad_b};
}

// x format: [DID, i0, i1/D] dtype
// weight format: [DID, i2, i1/D] dtype
// grad format: [i0, i2] float
// outputs: grad_x [DID i0, i1/D] dtype
// grad_w [DID, i2,  i1/D] dtype
// grad_b [i2] dtype
LinearBackwardsResult sharded_linear_backwards(
    TensorView* x,
    TensorView* w,
    TensorView* grad) {
  DataType dtype = w->dtype();
  TensorView* grad_q = castOp(dtype, grad);
  TensorView* grad_x = matmul(grad_q, w);
  TensorView* grad_t = transpose(grad_q, 0, 1);
  TensorView* grad_w = matmul(grad_t, x);
  TensorView* grad_b = sum(grad, {0});
  grad_b = castOp(dtype, grad_b);

  return {grad_x, grad_w, grad_b};
}

// Forward layer_norm with cached mean_bcast and invstd tensors to avoid
// recomputing Welford. For use in backwards pass.
TensorView* layer_norm_with_cached_statistics(
    TensorView* x,
    TensorView* mean_bcast,
    TensorView* invstd,
    const std::vector<int64_t>& norm_shape,
    TensorView* weight,
    TensorView* bias) {
  const int64_t kNumberOfDims =
      (int64_t)TensorDomain::noReductions(x->getLogicalDomain()).size();
  const int64_t kOuterNumDims = kNumberOfDims - norm_shape.size();
  std::vector<bool> outer_broadcast_mask(kNumberOfDims, false);
  for (const auto idx : c10::irange(kOuterNumDims)) {
    outer_broadcast_mask[idx] = true;
  }

  auto x_sub_mean = sub(x, mean_bcast);
  auto y = mul(x_sub_mean, invstd);

  auto weight_bcast = broadcast(weight, outer_broadcast_mask);
  y = mul(y, weight_bcast);
  auto bias_bcast = broadcast(bias, outer_broadcast_mask);
  return add(y, bias_bcast);
}

// Backwards MLP block.
std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* w1,
    TensorView* linear0,
    const DeviceMesh& mesh) {
  DataType dtype = w0->dtype();

  // Activation recomputation: Always recompute gelu
  TensorView* gelu = castOp(dtype, tanh_gelu(castOp(DataType::Float, linear0)));

  // Backwards pass
  constexpr double kScale = 1.0 / (1.0 - kDropoutProb);
  Val* dropout_scale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dropout_scale);
  auto linear1_grads = sharded_linear_backwards(gelu, w1, dropout_grad);
  TensorView* matmul1_grad_x_ = castOp(DataType::Float, linear1_grads.grad_x);
  TensorView* gelu_grad = tanh_gelu_backward(matmul1_grad_x_, linear0);
  auto linear0_grads = linear_backwards(x, w0, gelu_grad);

  // Manaul sharding annotations
  for (auto tv :
       {x,
        grad,
        mask,
        dropout_grad,
        linear1_grads.grad_b,
        linear0_grads.grad_x}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv :
       {w0,
        w1,
        linear0,
        linear1_grads.grad_x,
        linear1_grads.grad_w,
        gelu_grad,
        linear0_grads.grad_w,
        linear0_grads.grad_b}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  std::vector<TensorView*> outputs = {
      dropout_grad,
      linear1_grads.grad_w,
      linear1_grads.grad_b,
      gelu_grad,
      linear0_grads.grad_w,
      linear0_grads.grad_b,
      linear0_grads.grad_x};
  return outputs;
}

std::vector<TensorView*> mha_backwards(
    TensorView* x,
    TensorView* w0,
    TensorView* w1,
    TensorView* mask,
    TensorView* sdpa_output,
    TensorView* sdpa_log_sumexp,
    TensorView* sdpa_seed,
    TensorView* sdpa_offset,
    TensorView* grad,
    TensorView* linear0,
    const DeviceMesh& mesh) {
  DataType dtype = w0->dtype();
  const auto D = w0->axis(0)->extent()->value().as<int64_t>();
  // Reform qkv from linear0 output
  TensorView* qkv_cat = reshape(
      castOp(DataType::Float, linear0),
      {D, B * S, 3 * E / D},
      {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : c10::irange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = transpose(qkv[i], 2, 3);
    qkv[i] = castOp(dtype, qkv[i]);
    qkv[i]->setDeviceMesh(mesh);
    qkv[i]->axis(0)->parallelize(ParallelType::DIDx);
  }

  // dropout backwards
  constexpr double kScale = 1.0 / (1.0 - kDropoutProb);
  auto dropout_scale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dropout_scale);

  // linear1 backwards
  TensorView* sdpa_output_reshape =
      transpose(sdpa_output, 2, 3); // D, B, S, H/D, E/H
  sdpa_output_reshape =
      reshape(sdpa_output_reshape, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  auto linear1_grads =
      sharded_linear_backwards(sdpa_output_reshape, w1, dropout_grad);

  // SDPA backwards
  TensorView* linear1_x_grad =
      reshape(linear1_grads.grad_x, {D, B * S, E / D}, {D, B, S, H / D, E / H});
  linear1_x_grad = transpose(linear1_x_grad, 2, 3); // D, B, H/D, S, E/H
  // Explicitly shard inputs before SDPA backward node
  for (auto tv : {linear1_x_grad, sdpa_output, sdpa_log_sumexp}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  auto sdpa_grad = sdpfa_bwd(
      linear1_x_grad,
      qkv[0],
      qkv[1],
      qkv[2],
      sdpa_output,
      sdpa_log_sumexp,
      /*dropout_p=*/IrBuilder::create<Val>(kSdpaProb),
      /*is_causal=*/IrBuilder::create<Val>(true),
      sdpa_seed,
      sdpa_offset,
      /*scale=*/IrBuilder::create<Val>(kSdpaScale));

  TensorView* q_grad = transpose(sdpa_grad.grad_query, 2, 3);
  q_grad = reshape(q_grad, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* v_grad = transpose(sdpa_grad.grad_value, 2, 3);
  v_grad = reshape(v_grad, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* k_grad = transpose(sdpa_grad.grad_key, 2, 3);
  k_grad = reshape(k_grad, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* kqv_grad = cat({k_grad, q_grad, v_grad}, -1);
  auto linear0_grads = linear_backwards(x, w0, kqv_grad);

  for (auto tv :
       {x,
        mask,
        grad,
        dropout_grad,
        linear1_grads.grad_b,
        linear0_grads.grad_x}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv :
       {w0,
        w1,
        sdpa_output,
        sdpa_log_sumexp,
        linear0,
        linear1_grads.grad_x,
        linear1_grads.grad_w,
        linear0_grads.grad_w,
        linear0_grads.grad_b,
        sdpa_grad.grad_query,
        sdpa_grad.grad_key,
        sdpa_grad.grad_value}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {
      dropout_grad,
      linear1_grads.grad_w,
      linear1_grads.grad_b,
      sdpa_grad.grad_query,
      sdpa_grad.grad_key,
      sdpa_grad.grad_value,
      linear0_grads.grad_w,
      linear0_grads.grad_b,
      linear0_grads.grad_x};
}
} // namespace

TEST_P(DistributedTransformerTest, MLP_Layer) {
  if ((4 * E) % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide 4*E=" << 4 * E;
  }
  DataType dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

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

  std::vector<TensorView*> tvsout = mlp(tvx, tvw0, tvb0, tvw1, tvb1, mesh);

  for (auto* tv : tvsout) {
    fusion->addOutput(tv);
  }
  shardBetween({tvw0, tvb0, tvw1}, {tvsout[3]}, tvw0);
  shardBetween({tvx, tvb1}, {tvsout[3]}, tvx);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({4 * E, E}, options) * kParamScale;
  auto b0 = at::randn({4 * E}, options) * kParamScale;
  auto w1 = at::randn({E, 4 * E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  // Note: resetting the seed before reference and nvFuser
  // execution so that random vals are the same.
  at::manual_seed(getATenRandomSeed());
  std::vector<at::Tensor> reference_outs = reference_mlp(x, w0, b0, w1, b1);

  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0, 0, mesh),
      shardTensor(b0, 0, mesh),
      shardTensor(w1, 1, mesh),
      b1};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec.runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {0.01, 0.01, 0.02, 0.02});
}

TEST_P(DistributedTransformerTest, Sequence_Parallel_MLP_Layer) {
  // TODO: Reshapes that form device axes when D=1 get optimized away causing
  // failures. This won't be a problem after
  // https://github.com/NVIDIA/Fuser/issues/2563.
  if (D == 1) {
    GTEST_SKIP() << "Requires >1 devices, D=" << D;
  }
  if ((4 * E) % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide 4*E=" << 4 * E;
  }
  if ((B * S) % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide B*S=" << B * S;
  }
  DataType dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = makeContigConcreteTensor({D, B * S / D, E}, dtype);
  TensorView* w0 = makeContigConcreteTensor({D, 4 * E / D, E}, dtype);
  TensorView* b0 = makeContigConcreteTensor({D, 4 * E / D}, dtype);
  TensorView* w1 = makeContigConcreteTensor({D, E, 4 * E / D}, dtype);
  TensorView* b1 = makeContigConcreteTensor({E}, dtype);
  TensorView* mask =
      makeContigConcreteTensor({D, B * S / D, E}, DataType::Bool);

  // Input x is sharded on B*S dimension.
  // Note only the sequence (S) dimension that is sharded
  // but to avoid DID parallelizations of inner logical axes
  // B*S is sharded.
  auto tvsout = mlp(x, w0, b0, w1, b1, mesh, true, mask);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);
  fusion->addInput(mask);

  for (auto* tv : tvsout) {
    fusion->addOutput(tv);
  }

  // Ensure broadcasts of bias are sharded.
  shardBetween({b1}, {tvsout[2]}, tvsout[2]);
  // Needed to ensure that rand_like is sharded initially.
  // sharding from linear1 to dropout like dropout
  shardBetween({tvsout[2]}, {tvsout[3]}, tvsout[3]);

  auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options);
  auto w0_ = at::randn({4 * E, E}, options) * kParamScale;
  auto b0_ = at::randn({4 * E}, options) * kParamScale;
  auto w1_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto b1_ = at::randn({E}, options) * kParamScale;

  // Dropout is sharded among devices.
  // For validation against ATen the sharded reference dropout mask is an input
  // to the Fusion, but in regular setting it would be generated.
  std::vector<at::Tensor> reference_outs =
      reference_mlp(x_, w0_, b0_, w1_, b1_);
  auto mask_ = reference_outs[4];

  std::vector<c10::IValue> inputs = {
      shardTensor(x_, 0, mesh),
      shardTensor(w0_, 0, mesh),
      shardTensor(b0_, 0, mesh),
      shardTensor(w1_, 1, mesh),
      b1_,
      shardTensor(mask_, 0, mesh)};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh),
      shardTensor(reference_outs[1], 1, mesh),
      shardTensor(reference_outs[2], 0, mesh),
      shardTensor(reference_outs[3], 0, mesh)};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec.runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {0.01, 0.01, 0.02, 0.02});
}

TEST_P(DistributedTransformerTest, MultiheadAttention) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  auto dtype = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* tvx = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* tvb0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* tvb1 = makeContigConcreteTensor({E}, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  auto tv_outs = mha(tvx, tvw0, tvb0, tvw1, tvb1, mesh);

  for (auto tv : tv_outs) {
    fusion->addOutput(tv);
  }

  shardBetween({tvw0, tvb0, tvw1}, {tv_outs[2]}, tvw0);
  shardBetween({tvx, tvb1}, {tv_outs[3]}, tvx);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({3 * E, E}, options) * kParamScale;
  auto b0 = at::randn({3 * E}, options) * kParamScale;
  auto w1 = at::randn({E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  auto reference_outs = reference_mha(x, w0, b0, w1, b1);
  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(b0.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(w1, 1, mesh),
      b1};
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0].view({B * S, 3, E}), 2, mesh)
          .view({1, B * S, 3 * E / D}),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec.runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {0.02, 0.02, 0.02, 0.02});
}

TEST_P(DistributedTransformerTest, MLP_Backward) {
  if ((4 * E) % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide 4*E=" << 4 * E;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* grad = makeContigTensor(2);
  TensorView* x = makeContigTensor(2, dtype);
  TensorView* mask = makeContigTensor(2, DataType::Bool);
  TensorView* w0 = makeContigTensor(3, dtype);
  TensorView* b0 = makeContigTensor(2, dtype);
  TensorView* w1 = makeContigTensor(3, dtype);
  TensorView* linear0 = makeContigTensor(3, dtype);

  fusion->addInput(grad);
  fusion->addInput(x);
  fusion->addInput(mask);
  fusion->addInput(w0);
  fusion->addInput(w1);
  fusion->addInput(linear0);

  std::vector<TensorView*> tv_outs =
      mlp_backwards(grad, x, mask, w0, w1, linear0, mesh);

  for (TensorView* tv : tv_outs) {
    fusion->addOutput(tv);
  }

  // Sharded: matmul1_grad_w, gelu_grad, matmul0_grad_w, matmul0_grad_b
  shardBetween(
      {w0, b0, w1}, {tv_outs[1], tv_outs[3], tv_outs[4], tv_outs[5]}, w0);
  // Unsharded: dropout_grad, matmul1_grad_b, matmul0_grad_x
  shardBetween({grad, x}, {tv_outs[0], tv_outs[2], tv_outs[6]}, grad);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto grad_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto x_ = at::randn({B * S, E}, options);
  auto mask_ = at::rand({B * S, E}, options).lt(1.0 - kDropoutProb);
  auto mlp_w0_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({E, 4 * E}, options) * kParamScale;

  auto linear0_ = at::linear(x_, mlp_w0_, mlp_b0_);
  std::vector<at::Tensor> outs =
      reference_mlp_backwards(grad_, x_, mask_, mlp_w0_, mlp_w1_, linear0_);

  std::vector<c10::IValue> inputs = {
      grad_,
      x_,
      mask_,
      shardTensor(mlp_w0_, 0, mesh),
      shardTensor(mlp_w1_, 1, mesh),
      shardTensor(linear0_, 1, mesh)};
  std::vector<at::Tensor> expected_outputs = {
      outs[0], // dropout grad
      shardTensor(outs[1], 1, mesh), // linear1 weight grad
      outs[2], // linear1 bias grad
      shardTensor(outs[3], 1, mesh), // gelu grad
      shardTensor(outs[4], 0, mesh), // linear0 weight grad
      shardTensor(outs[5], 0, mesh), // linear0 bias grad
      outs[6]}; // linear0 grad x

  FusionExecutorCache fec(std::move(fusion));
  auto outputs = fec.runFusionWithInputs(inputs);

  validate(expected_outputs, outputs, {1e-5, 0.2, 1e-5, 0.01, 0.2, 0.01, 0.02});
}

TEST_P(DistributedTransformerTest, MHA_Backward) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* tvx = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* tvgrad = makeContigConcreteTensor({B * S, E});
  TensorView* tvmask = makeContigConcreteTensor({B * S, E}, DataType::Bool);
  TensorView* tvsdpa_out =
      makeContigConcreteTensor({D, B, H / D, S, E / H}, dtype);
  TensorView* tvsdpa_log_sumexp =
      makeContigConcreteTensor({D, B, H / D, S}, DataType::Float);
  TensorView* tvsdpa_seed = makeSymbolicTensor({}, DataType::Int);
  TensorView* tvsdpa_offset = makeSymbolicTensor({}, DataType::Int);
  TensorView* linear0 = makeSymbolicTensor(3, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvw1);
  fusion->addInput(tvgrad);
  fusion->addInput(tvmask);
  fusion->addInput(tvsdpa_out);
  fusion->addInput(tvsdpa_log_sumexp);
  fusion->addInput(tvsdpa_seed);
  fusion->addInput(tvsdpa_offset);
  fusion->addInput(linear0);

  auto tvouts = mha_backwards(
      tvx,
      tvw0,
      tvw1,
      tvmask,
      tvsdpa_out,
      tvsdpa_log_sumexp,
      tvsdpa_seed,
      tvsdpa_offset,
      tvgrad,
      linear0,
      mesh);

  for (auto tv : tvouts) {
    fusion->addOutput(tv);
  }

  // propagate shardings (mesh + DIDx) from sharded roots to all sharded leafs
  // (grads for linear0 bias and weight, linear1 weight)
  shardBetween({tvw1, tvw0}, {tvouts[1], tvouts[2], tvouts[6]}, tvw0);
  // propagate DeviceMesh from unsharded roots to unsharded leafs (grads for
  // dropout, linear1 bias, input)
  shardBetween({tvx, tvmask, tvgrad}, {tvouts[0], tvouts[7], tvouts[8]}, tvx);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({3 * E, E}, options) * kParamScale;
  auto b0 = at::randn({3 * E}, options) * kParamScale;
  auto w1 = at::randn({E, E}, options) * kParamScale;
  auto grad = at::randn({B * S, E}, options).to(at::kFloat);
  auto mask = at::rand({B * S, E}, options).lt(1.0 - kDropoutProb);

  at::manual_seed(getATenRandomSeed());
  auto reference_outs = reference_mha_backwards(grad, x, mask, w0, b0, w1);
  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(w1, 1, mesh),
      grad,
      mask,
      shardTensor(reference_outs[0], 1, mesh), // sdpa.output
      shardTensor(reference_outs[1], 1, mesh), // sdpa.log_sumexp
      reference_outs[2], // sdpa.seed
      reference_outs[3], // sdpa.offset
      shardTensor(reference_outs[13], 1, mesh) // linear0
  };
  std::vector<at::Tensor> expected_outputs = {
      reference_outs[4], // dropout grad
      shardTensor(reference_outs[5], 1, mesh), // linear1 weight grad
      reference_outs[6], // linear1 bias grad
      shardTensor(reference_outs[7], 1, mesh), // q grad
      shardTensor(reference_outs[8], 1, mesh), // k grad
      shardTensor(reference_outs[9], 1, mesh), // v grad
      shardTensor(reference_outs[10].view({3, E, E}), 1, mesh)
          .view({1, 3 * E / D, E}), // linear0 weight grad
      shardTensor(reference_outs[11].view({3, E}), 1, mesh)
          .view({1, 3 * E / D}), // linear0 bias grad
      reference_outs[12]};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto out = fec.runFusionWithInputs(inputs);
  validate(
      expected_outputs, out, {1e-5, 0.02, 1e-5, .01, .02, 0.2, 0.2, 0.2, 0.02});
}

TEST_P(DistributedTransformerTest, Forward) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* ln0_w = makeContigTensor(1);
  TensorView* ln0_b = makeContigTensor(1);
  TensorView* mha_w0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* mha_b1 = makeContigConcreteTensor({E}, dtype);
  TensorView* ln1_w = makeContigTensor(1);
  TensorView* ln1_b = makeContigTensor(1);
  TensorView* mlp_w0 = makeContigTensor(3, dtype);
  TensorView* mlp_b0 = makeContigTensor(2, dtype);
  TensorView* mlp_w1 = makeContigTensor(3, dtype);
  TensorView* mlp_b1 = makeContigTensor(1, dtype);

  fusion->addInput(x);
  fusion->addInput(ln0_w);
  fusion->addInput(ln0_b);
  fusion->addInput(mha_w0);
  fusion->addInput(mha_b0);
  fusion->addInput(mha_w1);
  fusion->addInput(mha_b1);
  fusion->addInput(ln1_w);
  fusion->addInput(ln1_b);
  fusion->addInput(mlp_w0);
  fusion->addInput(mlp_b0);
  fusion->addInput(mlp_w1);
  fusion->addInput(mlp_b1);

  constexpr float kEps = 1e-5;
  auto eps = IrBuilder::create<Val>(kEps);
  std::vector<int64_t> norm_shape{E};

  auto ln_input = castOp(DataType::Float, x);
  auto ln0 = layer_norm(ln_input, norm_shape, ln0_w, ln0_b, eps);
  auto mha_in = castOp(dtype, ln0.output);
  auto mha_out = mha(mha_in, mha_w0, mha_b0, mha_w1, mha_b1, mesh)[3];
  auto resid0 = add(ln_input, mha_out);
  auto ln1 = layer_norm(resid0, norm_shape, ln1_w, ln1_b, eps);
  auto mlp_in = castOp(dtype, ln1.output);
  auto mlp_out = mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, mesh)[3];
  auto resid1 = add(resid0, mlp_out);
  resid1 = castOp(dtype, resid1);

  fusion->addOutput(ln0.output);
  fusion->addOutput(mha_out);
  fusion->addOutput(ln1.output);
  fusion->addOutput(mlp_out);
  fusion->addOutput(resid1);

  for (auto tv : {x, ln0.output, ln1.output, resid1}) {
    tv->setDeviceMesh(mesh);
  }

  shardBetween({mha_in->definition()}, {mha_out->definition()}, mha_w0);
  shardBetween({mlp_in->definition()}, {mlp_out->definition()}, mlp_w0);
  shardBetween({x}, {mha_in}, x);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options);
  auto ln0_w_ = at::randn(E, options).to(at::kFloat);
  auto ln0_b_ = at::randn(E, options).to(at::kFloat);
  auto mha_w0_ = at::randn({3 * E, E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;
  auto ln1_w_ = at::randn(E, options).to(at::kFloat);
  auto ln1_b_ = at::randn(E, options).to(at::kFloat);
  auto mlp_w0_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  auto x_float_ = x_.to(at::kFloat);
  auto ln0_ = at::native_layer_norm(x_float_, norm_shape, ln0_w_, ln0_b_, kEps);
  auto ln0_out_ = std::get<0>(ln0_);

  auto mha_out_ = reference_mha(
      ln0_out_.to(at_dtype), mha_w0_, mha_b0_, mha_w1_, mha_b1_)[3];

  auto resid0_ = mha_out_ + x_float_;
  auto ln1_ = at::native_layer_norm(resid0_, norm_shape, ln1_w_, ln1_b_, kEps);
  auto ln1_out_ = std::get<0>(ln1_);

  auto mlp_out_ = reference_mlp(
      ln1_out_.to(at_dtype), mlp_w0_, mlp_b0_, mlp_w1_, mlp_b1_)[3];
  auto at_out = (resid0_ + mlp_out_).to(at_dtype);

  std::vector<c10::IValue> inputs = {
      x_,
      ln0_w_,
      ln0_b_,
      shardTensor(mha_w0_.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(mha_w1_, 1, mesh),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      shardTensor(mlp_w0_, 0, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 1, mesh),
      mlp_b1_};

  std::vector<at::Tensor> expected_outputs = {
      ln0_out_, mha_out_, ln1_out_, mlp_out_, at_out};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec.runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {1e-4, 0.02, 0.04, 0.04, 0.04});
}

TEST_P(DistributedTransformerTest, Backward) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
  constexpr float kEps = 1e-5;
  std::vector<int64_t> norm_shape{E};

  TensorView* x = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* grad = makeContigTensor(2, dtype);
  TensorView* mha_w0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* mlp_w0 = makeContigTensor(3, dtype);
  TensorView* mlp_w1 = makeContigTensor(3, dtype);
  TensorView* mha_mask = makeContigTensor(2, DataType::Bool);
  TensorView* mlp_mask = makeContigTensor(2, DataType::Bool);
  TensorView* mha_sdpa_out = makeConcreteTensor({D, B, H / D, S, E / H}, dtype);
  TensorView* mha_sdpa_log_sumexp =
      makeContigConcreteTensor({D, B, H / D, S}, DataType::Float);
  TensorView* mha_sdpa_seed = makeSymbolicTensor({}, DataType::Int);
  TensorView* mha_sdpa_offset = makeSymbolicTensor({}, DataType::Int);
  TensorView* ln1_w = makeContigTensor(1);
  TensorView* ln1_b = makeContigTensor(1);
  TensorView* ln1_mean = makeConcreteTensor({B * S, 1});
  TensorView* ln1_rstd = makeConcreteTensor({B * S, 1});
  TensorView* ln0_w = makeContigTensor(1);
  TensorView* ln0_b = makeContigTensor(1);
  TensorView* ln0_mean = makeConcreteTensor({B * S, 1});
  TensorView* ln0_rstd = makeConcreteTensor({B * S, 1});
  TensorView* mha_linear0 = makeContigTensor(3, dtype);
  TensorView* mha_linear1 = makeContigTensor(2);
  TensorView* mlp_linear0 = makeContigTensor(3, dtype);

  fusion->addInput(x);
  fusion->addInput(grad);
  fusion->addInput(mha_w0);
  fusion->addInput(mha_w1);
  fusion->addInput(mlp_w0);
  fusion->addInput(mlp_w1);
  fusion->addInput(mlp_mask);
  fusion->addInput(mha_mask);
  fusion->addInput(mha_sdpa_out);
  fusion->addInput(mha_sdpa_log_sumexp);
  fusion->addInput(mha_sdpa_seed);
  fusion->addInput(mha_sdpa_offset);
  fusion->addInput(ln1_w);
  fusion->addInput(ln1_b);
  fusion->addInput(ln1_mean);
  fusion->addInput(ln1_rstd);
  fusion->addInput(ln0_w);
  fusion->addInput(ln0_b);
  fusion->addInput(ln0_mean);
  fusion->addInput(ln0_rstd);
  fusion->addInput(mha_linear0);
  fusion->addInput(mha_linear1);
  fusion->addInput(mlp_linear0);

  // Activation recomputation: mlp gelu, dropouts, and
  // partially recompute layer norms using cached statistics.
  auto ln0_in = castOp(DataType::Float, x);
  auto ln0 = layer_norm_with_cached_statistics(
      ln0_in, ln0_mean, ln0_rstd, norm_shape, ln0_w, ln0_b);
  auto mha_in = castOp(dtype, ln0);

  Val* dropout_scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  // Use input mha_mask to implement dropout
  auto mha_out = mul(mha_linear1, mha_mask);
  mha_out = mul(mha_out, dropout_scale);
  auto resid0 = add(ln0_in, mha_out);
  auto ln1 = layer_norm_with_cached_statistics(
      resid0, ln1_mean, ln1_rstd, norm_shape, ln1_w, ln1_b);
  auto mlp_in = castOp(dtype, ln1);

  // Backwards
  auto grad_float = castOp(DataType::Float, grad);
  auto mlp_grads = mlp_backwards(
      grad_float, mlp_in, mlp_mask, mlp_w0, mlp_w1, mlp_linear0, mesh);
  auto ln1_grads = layer_norm_backward(
      castOp(DataType::Float, mlp_grads[6]),
      resid0,
      norm_shape,
      ln1_mean,
      ln1_rstd,
      ln1_w,
      ln1_b,
      {true, true, true});
  auto resid1_grad = add(ln1_grads.grad_input, grad_float);
  auto mha_grads = mha_backwards(
      mha_in,
      mha_w0,
      mha_w1,
      mha_mask,
      mha_sdpa_out,
      mha_sdpa_log_sumexp,
      mha_sdpa_seed,
      mha_sdpa_offset,
      resid1_grad,
      mha_linear0,
      mesh);
  auto ln0_grads = layer_norm_backward(
      castOp(DataType::Float, mha_grads[8]),
      ln0_in,
      norm_shape,
      ln0_mean,
      ln0_rstd,
      ln0_w,
      ln0_b,
      {true, true, true});
  auto dx = add(ln0_grads.grad_input, resid1_grad);
  dx = castOp(dtype, dx);

  fusion->addOutput(mlp_grads[1]); // mlp linear1 weight grad
  fusion->addOutput(mlp_grads[2]); // mlp linear1 bias grad
  fusion->addOutput(mlp_grads[4]); // mlp linear0 weight grad
  fusion->addOutput(mlp_grads[5]); // mlp linear0 bias grad
  fusion->addOutput(ln1_grads.grad_weight);
  fusion->addOutput(ln1_grads.grad_bias);
  fusion->addOutput(mha_grads[1]); // mha linear1 weight grad
  fusion->addOutput(mha_grads[2]); // mha linear1 bias grad
  fusion->addOutput(mha_grads[6]); // mha linear0 weight grad
  fusion->addOutput(mha_grads[7]); // mha linear0 bias grad
  fusion->addOutput(ln0_grads.grad_weight);
  fusion->addOutput(ln0_grads.grad_bias);
  fusion->addOutput(dx); // transformer grad input

  // Sharding annotations for input and output TVs not sharded
  // by mlp_backward or mha_backward
  for (auto* tv :
       {ln0_w,
        ln0_b,
        ln0_mean,
        ln0_rstd,
        ln1_w,
        ln1_b,
        ln1_mean,
        ln1_rstd,
        ln1_grads.grad_weight,
        ln1_grads.grad_bias,
        ln0_grads.grad_weight,
        ln0_grads.grad_bias,
        ln0_grads.grad_input}) {
    tv->setDeviceMesh(mesh);
  }

  // Sharded inputs to outputs
  shardBetween(
      {mha_w0, mha_w1, mha_sdpa_out},
      {mha_grads[1], mha_grads[6], mha_grads[7]},
      mha_w0);
  shardBetween(
      {mlp_w0, mlp_w1}, {mlp_grads[1], mlp_grads[4], mlp_grads[5]}, mlp_w0);

  // Unsharded inputs to outputs
  shardBetween(
      {x,
       grad,
       mha_mask,
       mlp_mask,
       mha_linear1,
       ln0_mean,
       ln0_w,
       ln0_b,
       ln1_mean,
       ln1_w,
       ln1_b},
      {mlp_grads[2],
       ln1_grads.grad_weight,
       ln1_grads.grad_bias,
       mha_grads[2],
       ln0_grads.grad_weight,
       ln0_grads.grad_bias,
       dx},
      x);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options);
  auto ln0_w_ = at::randn(E, options).to(at::kFloat);
  auto ln0_b_ = at::randn(E, options).to(at::kFloat);
  auto mha_w0_ = at::randn({3 * E, E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;
  auto ln1_w_ = at::randn(E, options).to(at::kFloat);
  auto ln1_b_ = at::randn(E, options).to(at::kFloat);
  auto mlp_w0_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto grad_ = at::randn({B * S, E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  // Run forward pass up to MLP to generate cached inputs
  auto [ln0_, ln0_mean_, ln0_rstd_] = at::native_layer_norm(
      x_.to(at::kFloat), norm_shape, ln0_w_, ln0_b_, kEps);
  auto mha_in_ = ln0_.to(at_dtype);
  auto mha_out_ = reference_mha(mha_in_, mha_w0_, mha_b0_, mha_w1_, mha_b1_);
  auto resid0_ = mha_out_[3] + x_.to(at::kFloat);
  auto [ln1_, ln1_mean_, ln1_rstd_] =
      at::native_layer_norm(resid0_, norm_shape, ln1_w_, ln1_b_, kEps);
  auto mlp_in_ = ln1_.to(at_dtype);
  auto mlp_out_ = reference_mlp(mlp_in_, mlp_w0_, mlp_b0_, mlp_w1_, mlp_b1_);

  // Backwards pass
  auto mlp_grads_ = reference_mlp_backwards(
      grad_, mlp_in_, mlp_out_[4], mlp_w0_, mlp_w1_, mlp_out_[0]);
  auto [ln1_x_grad_, ln1_w_grad_, ln1_b_grad_] = at::native_layer_norm_backward(
      mlp_grads_[6].to(at::kFloat),
      resid0_,
      norm_shape,
      ln1_mean_,
      ln1_rstd_,
      ln1_w_,
      ln1_b_,
      {true, true, true});
  auto resid1_grad_ = ln1_x_grad_ + grad_.to(at::kFloat);
  auto mha_grads_ = reference_mha_backwards(
      resid1_grad_, mha_in_, mha_out_[4], mha_w0_, mha_b0_, mha_w1_);
  auto [ln0_x_grad_, ln0_w_grad_, ln0_b_grad_] = at::native_layer_norm_backward(
      mha_grads_[12].to(at::kFloat),
      x_.to(at::kFloat),
      norm_shape,
      ln0_mean_,
      ln0_rstd_,
      ln0_w_,
      ln0_b_,
      {true, true, true});
  auto dx_ = (ln0_x_grad_ + resid1_grad_).to(at_dtype);

  auto expected_outputs = {
      shardTensor(mlp_grads_[1], 1, mesh), // mlp_linear1_weight_grad
      mlp_grads_[2], // mlp_linear1_bias_grad
      shardTensor(mlp_grads_[4], 0, mesh), // mlp_linear0_weight_grad
      shardTensor(mlp_grads_[5], 0, mesh), // mlp_linear0_bias_grad
      ln1_w_grad_,
      ln1_b_grad_,
      shardTensor(mha_grads_[5], 1, mesh), // mha linear1 weight grad
      mha_grads_[6], // mha linear1 bias grad
      shardTensor(
          mha_grads_[10].view({3, E, E}), 1, mesh) // failing starting here
          .view({1, 3 * E / D, E}), // mha linear0 bias grad
      shardTensor(mha_grads_[11].view({3, E}), 1, mesh)
          .view({1, 3 * E / D}), // mha linear0 bias grad
      ln0_w_grad_,
      ln0_b_grad_,
      dx_};

  std::vector<c10::IValue> inputs = {
      x_,
      grad_,
      shardTensor(mha_w0_.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(mha_w1_, 1, mesh),
      shardTensor(mlp_w0_, 0, mesh),
      shardTensor(mlp_w1_, 1, mesh),
      mlp_out_[4], // mlp dropout mask
      mha_out_[4], // mha dropout mask
      shardTensor(mha_grads_[0], 1, mesh), // sdpa output
      shardTensor(mha_grads_[1], 1, mesh), // sdpa logsum_exp
      mha_grads_[2], // sdpa seed
      mha_grads_[3], // sdpa offset
      ln1_w_,
      ln1_b_,
      ln1_mean_,
      ln1_rstd_,
      ln0_w_,
      ln0_b_,
      ln0_mean_,
      ln0_rstd_,
      shardTensor(mha_out_[0], 1, mesh), // mha linear0
      mha_out_[2].to(at::kFloat), // mha linear1
      shardTensor(mlp_out_[0], 1, mesh) // mlp linear1
  };

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec.runFusionWithInputs(inputs);
  validate(
      expected_outputs,
      outputs,
      {1e-3,
       5e-3,
       4e-3,
       4e-3,
       4e-3,
       5e-3,
       0.01,
       4e-3,
       0.04,
       0.02,
       0.02,
       0.02,
       0.02});
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedTransformerTest,
    testing::Values(DataType::Half, DataType::BFloat16),
    testing::PrintToStringParamName());

} // namespace nvfuser
