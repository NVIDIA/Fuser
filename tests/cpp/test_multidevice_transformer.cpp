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
#include <multidevice/communicator.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/multidevice_transformer.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

namespace {
// Note: We test on smaller model and input sizes to avoid high error
// accumulation for validation.
static constexpr int64_t B = 2, E = 768, H = 16, S = 128;
// Note: Dropout probabilities are set to 0. Since the dropout mask is sharded
// it throws off the seed offset between the sharded nvFuser program and the
// unsharded reference.
static constexpr double kDropoutProb = 0.0, kSdpaProb = 0.0, kSdpaScale = 1e-3;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
static constexpr double kParamScale = 0.02;
} // namespace

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest() : D(communicator_->size()) {
    model = std::make_unique<DistributedTransformer>(
        D, B, E, H, S, kDropoutProb, kSdpaProb);
  }

  void SetUp() override {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  const int64_t D; // number of devices
  std::unique_ptr<DistributedTransformer> model;
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

  auto tvsout = model->mlp(tvx, tvw0, tvb0, tvw1, tvb1, mesh);

  fusion->addOutput(tvsout.linear0);
  fusion->addOutput(tvsout.gelu);
  fusion->addOutput(tvsout.linear1);
  fusion->addOutput(tvsout.output);

  shardBetween({tvw0}, {tvsout.output}, tvw0);
  shardBetween({tvw1}, {tvsout.output}, tvw1);

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
      shardTensor(w0, 0, mesh).unsqueeze(0),
      shardTensor(b0, 0, mesh).unsqueeze(0),
      shardTensor(w1, 1, mesh).unsqueeze(0),
      b1};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh).unsqueeze(0),
      shardTensor(reference_outs[1], 1, mesh).unsqueeze(0),
      reference_outs[2],
      reference_outs[3]};

  FusionExecutorCache executor_cache(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = executor_cache.runFusionWithInputs(inputs);
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

  // Input x is sharded on B*S dimension.
  // Note only the sequence (S) dimension that is sharded
  // but to avoid DID parallelizations of inner logical axes
  // B*S is sharded.
  auto tvsout = model->mlp(x, w0, b0, w1, b1, mesh, true);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  fusion->addOutput(tvsout.linear0);
  fusion->addOutput(tvsout.gelu);
  fusion->addOutput(tvsout.linear1);
  fusion->addOutput(tvsout.output);

  shardBetween({w0}, {tvsout.matmul1}, w0);
  shardBetween({w1}, {tvsout.matmul1}, w1);
  shardBetween({tvsout.matmul1}, {tvsout.output}, tvsout.matmul1);

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
      shardTensor(x_, 0, mesh).unsqueeze(0),
      shardTensor(w0_, 0, mesh).unsqueeze(0),
      shardTensor(b0_, 0, mesh).unsqueeze(0),
      shardTensor(w1_, 1, mesh).unsqueeze(0),
      b1_};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh).unsqueeze(0),
      shardTensor(reference_outs[1], 1, mesh).unsqueeze(0),
      shardTensor(reference_outs[2], 0, mesh).unsqueeze(0),
      shardTensor(reference_outs[3], 0, mesh).unsqueeze(0)};

  FusionExecutorCache executor_cache(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = executor_cache.runFusionWithInputs(inputs);
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

  auto tv_outs = model->mha(tvx, tvw0, tvb0, tvw1, tvb1, mesh);

  fusion->addOutput(tv_outs.linear0);
  fusion->addOutput(tv_outs.sdpa);
  fusion->addOutput(tv_outs.linear1);
  fusion->addOutput(tv_outs.output);

  shardBetween({tvw0}, {tv_outs.output}, tvw0);
  shardBetween({tvw1}, {tv_outs.output}, tvw1);

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
      shardTensor(w1, 1, mesh).unsqueeze(0),
      b1};
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0].view({B * S, 3, E}), 2, mesh)
          .view({1, B * S, 3 * E / D}),
      shardTensor(reference_outs[1], 1, mesh).unsqueeze(0),
      reference_outs[2],
      reference_outs[3]};

  FusionExecutorCache executor_cache(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto outputs = executor_cache.runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {0.02, 0.02, 0.02, 0.02});
}

TEST_P(DistributedTransformerTest, MultiheadAttention_SP) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  if (D == 1) {
    GTEST_SKIP() << "Requires >1 devices, D=" << D;
  }
  auto dtype = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* tvx = makeContigConcreteTensor({D, B * S / D, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* tvb0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* tvb1 = makeContigConcreteTensor({E}, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  auto tv_outs = model->mha(tvx, tvw0, tvb0, tvw1, tvb1, mesh, true);

  fusion->addOutput(tv_outs.linear0);
  fusion->addOutput(tv_outs.sdpa);
  fusion->addOutput(tv_outs.linear1);
  fusion->addOutput(tv_outs.output);

  shardBetween({tvw0}, {tv_outs.matmul1}, tvw0);
  shardBetween({tvw1}, {tv_outs.matmul1}, tvw1);
  shardBetween({tv_outs.matmul1}, {tv_outs.output}, tv_outs.matmul1);

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
      shardTensor(x, 0, mesh).unsqueeze(0),
      shardTensor(w0.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(b0.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(w1, 1, mesh).unsqueeze(0),
      b1};
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0].view({B * S, 3, E}), 2, mesh)
          .view({1, B * S, 3 * E / D}),
      shardTensor(reference_outs[1], 1, mesh).unsqueeze(0),
      shardTensor(reference_outs[2], 0, mesh).unsqueeze(0),
      shardTensor(reference_outs[3], 0, mesh).unsqueeze(0)};

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
      model->mlp_backwards(grad, x, mask, w0, w1, linear0, mesh);

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
      shardTensor(mlp_w0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_w1_, 1, mesh).unsqueeze(0),
      shardTensor(linear0_, 1, mesh).unsqueeze(0)};
  std::vector<at::Tensor> expected_outputs = {
      outs[0], // dropout grad
      shardTensor(outs[1], 1, mesh).unsqueeze(0), // linear1 weight grad
      outs[2], // linear1 bias grad
      shardTensor(outs[3], 1, mesh).unsqueeze(0), // gelu grad
      shardTensor(outs[4], 0, mesh).unsqueeze(0), // linear0 weight grad
      shardTensor(outs[5], 0, mesh).unsqueeze(0), // linear0 bias grad
      outs[6]}; // linear0 grad x

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(inputs);

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

  auto tvouts = model->mha_backwards(
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
      shardTensor(w1, 1, mesh).unsqueeze(0),
      grad,
      mask,
      shardTensor(reference_outs[0], 1, mesh).unsqueeze(0), // sdpa.output
      shardTensor(reference_outs[1], 1, mesh).unsqueeze(0), // sdpa.log_sumexp
      reference_outs[2], // sdpa.seed
      reference_outs[3], // sdpa.offset
      shardTensor(reference_outs[13], 1, mesh).unsqueeze(0) // linear0
  };
  std::vector<at::Tensor> expected_outputs = {
      reference_outs[4], // dropout grad
      shardTensor(reference_outs[5], 1, mesh)
          .unsqueeze(0), // linear1 weight grad
      reference_outs[6], // linear1 bias grad
      shardTensor(reference_outs[7], 1, mesh).unsqueeze(0), // q grad
      shardTensor(reference_outs[8], 1, mesh).unsqueeze(0), // k grad
      shardTensor(reference_outs[9], 1, mesh).unsqueeze(0), // v grad
      shardTensor(reference_outs[10].view({3, E, E}), 1, mesh)
          .view({1, 3 * E / D, E}), // linear0 weight grad
      shardTensor(reference_outs[11].view({3, E}), 1, mesh)
          .view({1, 3 * E / D}), // linear0 bias grad
      reference_outs[12]};

  FusionExecutorCache executor_cache(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  auto out = executor_cache.runFusionWithInputs(inputs);
  validate(
      expected_outputs, out, {1e-5, 0.02, 1e-5, .01, .02, 0.2, 0.2, 0.2, 0.02});
}

TEST_P(DistributedTransformerTest, Forward_SP) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  if (D == 1) {
    GTEST_SKIP() << "Requires >1 devices, D=" << D;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  constexpr float kEps = 1e-5;
  std::vector<int64_t> norm_shape{E};

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
      shardTensor(x_, 0, mesh).unsqueeze(0),
      ln0_w_,
      ln0_b_,
      shardTensor(mha_w0_.view({3, E, E}), 1, mesh).view({1, 3 * E / D, E}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(mha_w1_, 1, mesh).unsqueeze(0),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      shardTensor(mlp_w0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_b0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_w1_, 1, mesh).unsqueeze(0),
      mlp_b1_};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(ln0_out_, 0, mesh).unsqueeze(0),
      shardTensor(mha_out_, 0, mesh).unsqueeze(0),
      shardTensor(ln1_out_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_out_, 0, mesh).unsqueeze(0),
      shardTensor(at_out, 0, mesh).unsqueeze(0)};

  auto fec = model->forward(dtype, true);
  at::manual_seed(getATenRandomSeed());
  auto outputs = fec->runFusionWithInputs(inputs);
  validate(expected_outputs, outputs, {1e-4, 0.02, 0.04, 0.04, 0.04});
}

TEST_P(DistributedTransformerTest, Forward) {
  if (H % D != 0) {
    GTEST_SKIP() << "Requires number of devices=" << D
                 << " evenly divide H=" << H;
  }
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  const auto mesh = DeviceMesh::createForNumDevices(D);
  constexpr float kEps = 1e-5;
  std::vector<int64_t> norm_shape{E};

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
      shardTensor(mha_w1_, 1, mesh).unsqueeze(0),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      shardTensor(mlp_w0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_b0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_w1_, 1, mesh).unsqueeze(0),
      mlp_b1_};

  std::vector<at::Tensor> expected_outputs = {
      ln0_out_, mha_out_, ln1_out_, mlp_out_, at_out};

  auto executor_cache = model->forward(dtype);
  at::manual_seed(getATenRandomSeed());
  auto outputs = executor_cache->runFusionWithInputs(inputs);
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
      shardTensor(mlp_grads_[1], 1, mesh)
          .unsqueeze(0), // mlp_linear1_weight_grad
      mlp_grads_[2], // mlp_linear1_bias_grad
      shardTensor(mlp_grads_[4], 0, mesh)
          .unsqueeze(0), // mlp_linear0_weight_grad
      shardTensor(mlp_grads_[5], 0, mesh).unsqueeze(0), // mlp_linear0_bias_grad
      ln1_w_grad_,
      ln1_b_grad_,
      shardTensor(mha_grads_[5], 1, mesh)
          .unsqueeze(0), // mha linear1 weight grad
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
      shardTensor(mha_w1_, 1, mesh).unsqueeze(0),
      shardTensor(mlp_w0_, 0, mesh).unsqueeze(0),
      shardTensor(mlp_w1_, 1, mesh).unsqueeze(0),
      mlp_out_[4], // mlp dropout mask
      mha_out_[4], // mha dropout mask
      shardTensor(mha_grads_[0], 1, mesh).unsqueeze(0), // sdpa output
      shardTensor(mha_grads_[1], 1, mesh).unsqueeze(0), // sdpa logsum_exp
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
      shardTensor(mha_out_[0], 1, mesh).unsqueeze(0), // mha linear0
      mha_out_[2].to(at::kFloat), // mha linear1
      shardTensor(mlp_out_[0], 1, mesh).unsqueeze(0) // mlp linear1
  };

  auto executor_cache = model->backward(dtype);
  at::manual_seed(getATenRandomSeed());
  auto outputs = executor_cache->runFusionWithInputs(inputs);
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
