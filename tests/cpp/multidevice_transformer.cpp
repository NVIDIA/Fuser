// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <vector>

#include <fusion.h>
#include <ops/all_ops.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/multidevice_transformer.h>

namespace nvfuser {
namespace {
// TODO: These linearBackwards helper functions can be merged once
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
LinearBackwardsResult linearBackwards(
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
LinearBackwardsResult shardedLinearBackwards(
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
TensorView* layerNormWithCachedStats(
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
  for (const auto idx : arange(kOuterNumDims)) {
    outer_broadcast_mask[idx] = true;
  }

  auto x_sub_mean = sub(x, mean_bcast);
  auto y = mul(x_sub_mean, invstd);

  auto weight_bcast = broadcast(weight, outer_broadcast_mask);
  y = mul(y, weight_bcast);
  auto bias_bcast = broadcast(bias, outer_broadcast_mask);
  return add(y, bias_bcast);
}
} // namespace

MlpResult DistributedTransformer::mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh,
    bool sequence_parallel) {
  const DataType dtype = w0->dtype();

  if (sequence_parallel) {
    // Input arrives sharded and must be allgathered back
    x->setDeviceMesh(mesh);
    x->axis(0)->parallelize(ParallelType::DIDx);
    x = set(x); // allgather
    x->axis(0)->parallelize(ParallelType::Serial);
    // Reshape back to 2D. This is uncessary except to keep
    // the shapes of linear0 the same for TP and TP+SP.
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
    local_matmul1 = reshape(local_matmul1, {D, B * S, E}, {D, D, B * S / D, E});
  }
  TensorView* matmul1 = sum(local_matmul1, {0}); // Allreduce or Reduce scatter
  std::vector<bool> bcast_mask(matmul1->nDims() - 1, true);
  bcast_mask[matmul1->nDims() - 2] = false;
  TensorView* linear1 = add(matmul1, broadcast(b1, bcast_mask));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  TensorView* dropout_result = dropout(linear1, prob, scale).output;

  // Tensor parallel shardings
  for (auto* tv : {w0, b0, w1}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  for (auto* tv : {x, b1}) {
    tv->setDeviceMesh(mesh);
  }

  // Sequence parallel shardings
  if (sequence_parallel) {
    matmul1->setDeviceMesh(mesh);
    matmul1->axis(1)->parallelize(ParallelType::DIDx);
  }

  return {linear0, gelu, matmul1, linear1, dropout_result};
}

MhaResult DistributedTransformer::mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh,
    bool sequence_parallel) {
  auto dtype = w0->dtype();

  if (sequence_parallel) {
    // Input arrives sharded and must be allgathered back
    x->setDeviceMesh(mesh);
    x->axis(0)->parallelize(ParallelType::DIDx);
    x = set(x); // allgather
    x->axis(0)->parallelize(ParallelType::Serial);
    // Reshape is uncessary, it is here to keep shapes with TP and TP+SP the
    // same for validation.
    x = reshape(x, {D, B * S / D, E}, {B * S, E});
  }

  TensorView* linear0 = linear(x, w0, b0);
  // Forming the q,k,v vectors:
  TensorView* qkv_cat =
      reshape(linear0, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : arange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = transpose(qkv[i], 2, 3);
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
  if (sequence_parallel) {
    // Remove after https://github.com/NVIDIA/Fuser/issues/2563
    // Reshape to explicitly pull the sharded axis into the logical domain
    local_matmul1 = reshape(local_matmul1, {D, B * S, E}, {D, D, B * S / D, E});
  }
  TensorView* matmul1 = sum(local_matmul1, {0}); // allreduce
  std::vector<bool> bcast_mask(matmul1->nDims() - 1, true);
  bcast_mask[matmul1->nDims() - 2] = false;
  TensorView* linear1 = add(matmul1, broadcast(b1, bcast_mask));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  TensorView* dropout_result = dropout(linear1, prob, scale).output;

  // Tensor parallel shardings
  for (auto tv : {x, b1}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  // Sequence parallel sharding.
  if (sequence_parallel) {
    matmul1->setDeviceMesh(mesh);
    matmul1->axis(1)->parallelize(ParallelType::DIDx);
  }

  return {linear0, sdpa_output, matmul1, linear1, dropout_result};
}

std::vector<TensorView*> DistributedTransformer::mlp_backwards(
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
  const double kScale = 1.0 / (1.0 - kDropoutProb);
  Val* dropout_scale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dropout_scale);
  auto linear1_grads = shardedLinearBackwards(gelu, w1, dropout_grad);
  TensorView* matmul1_grad_x_ = castOp(DataType::Float, linear1_grads.grad_x);
  TensorView* gelu_grad = tanh_gelu_backward(matmul1_grad_x_, linear0);
  auto linear0_grads = linearBackwards(x, w0, gelu_grad);

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

std::vector<TensorView*> DistributedTransformer::mha_backwards(
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
  // Reform qkv from linear0 output
  TensorView* qkv_cat = reshape(
      castOp(DataType::Float, linear0),
      {D, B * S, 3 * E / D},
      {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : arange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = transpose(qkv[i], 2, 3);
    qkv[i] = castOp(dtype, qkv[i]);
    qkv[i]->setDeviceMesh(mesh);
    qkv[i]->axis(0)->parallelize(ParallelType::DIDx);
  }

  // dropout backwards
  const double kScale = 1.0 / (1.0 - kDropoutProb);
  auto dropout_scale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dropout_scale);

  // linear1 backwards
  TensorView* sdpa_output_reshape =
      transpose(sdpa_output, 2, 3); // D, B, S, H/D, E/H
  sdpa_output_reshape =
      reshape(sdpa_output_reshape, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  auto linear1_grads =
      shardedLinearBackwards(sdpa_output_reshape, w1, dropout_grad);

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
  auto linear0_grads = linearBackwards(x, w0, kqv_grad);

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

std::unique_ptr<FusionExecutorCache> DistributedTransformer::forward(
    DataType dtype,
    bool sequence_parallel) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = sequence_parallel
      ? makeContigConcreteTensor({D, B * S / D, E}, dtype)
      : makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* ln0_w = makeContigTensor(1);
  TensorView* ln0_b = makeContigTensor(1);
  TensorView* mha_w0 = makeContigConcreteTensor({D, 3 * E / D, E}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E, E / D}, dtype);
  TensorView* mha_b1 = makeContigConcreteTensor({E}, dtype);
  TensorView* ln1_w = makeContigTensor(1);
  TensorView* ln1_b = makeContigTensor(1);
  TensorView* mlp_w0 = makeContigConcreteTensor({D, 4 * E / D, E}, dtype);
  TensorView* mlp_b0 = makeContigConcreteTensor({D, 4 * E / D}, dtype);
  TensorView* mlp_w1 = makeContigConcreteTensor({D, E, 4 * E / D}, dtype);
  TensorView* mlp_b1 = makeContigConcreteTensor({E}, dtype);

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
  auto mha_tvs =
      mha(mha_in, mha_w0, mha_b0, mha_w1, mha_b1, mesh, sequence_parallel);
  auto resid0 = add(ln_input, mha_tvs.output);
  auto ln1 = layer_norm(resid0, norm_shape, ln1_w, ln1_b, eps);
  auto mlp_in = castOp(dtype, ln1.output);
  auto mlp_tvs =
      mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, mesh, sequence_parallel);
  auto resid1 = add(resid0, mlp_tvs.output);
  resid1 = castOp(dtype, resid1);

  fusion->addOutput(ln0.output);
  fusion->addOutput(mha_tvs.output);
  fusion->addOutput(ln1.output);
  fusion->addOutput(mlp_tvs.output);
  fusion->addOutput(resid1);

  x->setDeviceMesh(mesh);
  if (sequence_parallel) {
    // Input arrives sharded
    x->axis(0)->parallelize(ParallelType::DIDx);
    // Propagate SP shardings from x through layernorms, dropouts, residual
    // adds. Even though mha_in is part of the boundary set, residuals allow the
    // shardings to propagate up the graph so we must cut off the propagation at
    // the outputs of reduce scatters (mha and mlp matmul1)
    shardBetween({x}, {mha_in, mlp_in, mha_tvs.matmul1, mlp_tvs.matmul1}, x);
    // Propagate TP sharding for MLP and MHA from sharded weights. We do not
    // need to shard from mha_b0 or mlp_b0 because they are only consumed by
    // their respective linear0 expression which is sharded from *_w0.
    shardBetween({mha_w0}, {mha_tvs.matmul1}, mha_w0);
    shardBetween({mha_w1}, {mha_tvs.matmul1}, mha_w1);
    shardBetween({mlp_w0}, {mlp_tvs.matmul1}, mlp_w0);
    shardBetween({mlp_w1}, {mlp_tvs.matmul1}, mlp_w1);
  } else {
    // TP only shardings
    // Layernorm, residuals, are all replicated like x. shardBetween
    // shards all tvs reachable from x, so the input and output tvs must
    // be in the boundary set.
    shardBetween({x}, {mha_in, mha_tvs.output, mlp_in, mlp_tvs.output}, x);
    // TP sharded regions within mha and mlp
    shardBetween({mha_in}, {mha_tvs.output}, mha_w0);
    shardBetween({mlp_in}, {mlp_tvs.output}, mlp_w0);
  }

  return std::make_unique<FusionExecutorCache>(std::move(fusion));
}

std::unique_ptr<FusionExecutorCache> DistributedTransformer::backward(
    DataType dtype) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
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
  auto [mha_sdpa_seed, mha_sdpa_offset] = createSdpaRngTvs();
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
  auto ln0 = layerNormWithCachedStats(
      ln0_in, ln0_mean, ln0_rstd, norm_shape, ln0_w, ln0_b);
  auto mha_in = castOp(dtype, ln0);

  Val* dropout_scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  // Use input mha_mask to implement dropout
  auto mha_out = mul(mha_linear1, mha_mask);
  mha_out = mul(mha_out, dropout_scale);
  auto resid0 = add(ln0_in, mha_out);
  auto ln1 = layerNormWithCachedStats(
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

  return std::make_unique<FusionExecutorCache>(std::move(fusion));
}
} // namespace nvfuser
