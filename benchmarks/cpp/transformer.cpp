// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fusion.h>
#include <ops/all_ops.h>
#include <utils.h>
#include <multidevice/utils.h>
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/multidevice.h>

using namespace nvfuser;

constexpr int64_t B = 1, E = 12288, H = 96, S = 2048;
constexpr double kDropoutProb = 0.1, kParamScale = 0.02, kSdpaProb = 0.1;
constexpr int64_t warmup_itrs = 10, num_itrs = 10;

struct MLP_Results {
  TensorView* linear0;
  TensorView* gelu;
  TensorView* linear1;
  TensorView* output;
  TensorView* dropout_mask;
};

MLP_Results mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh) {
  const DataType dtype = w0->dtype();
  // Linear 0
  TensorView* matmul0 = matmul(x, w0);
  TensorView* linear0 = add(matmul0, broadcast(b0, {false, true, false}));
  // GeLU
  TensorView* gelu = tanh_gelu(linear0);
  gelu = castOp(dtype, gelu);
  // Linear 1
  TensorView* local_matmul1 = matmul(gelu, w1);
  TensorView* matmul1 = sum(local_matmul1, {0}); // Allreduce
  TensorView* linear1 = add(matmul1, broadcast(b1, {true, false}));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  auto dropout_tvs = dropout(linear1, prob, scale);

  // Manual sharding annotations
  for (auto tv : {x, b1, linear1, dropout_tvs.output}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear0, gelu}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  return {linear0, gelu, linear1, dropout_tvs.output, dropout_tvs.mask};
}

std::vector<TensorView*> mha_qkv(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    const DeviceMesh& mesh) {
  DataType dtype = w0->dtype();
  const auto D = w0->axis(0)->extent()->value().as<int64_t>();
  // compute linear 0, q, k, and v
  TensorView* matmul0 = matmul(x, w0);
  TensorView* linear0 = add(matmul0, broadcast(b0, {false, true, false}));
  TensorView* qkv_cat =
      reshape(linear0, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : c10::irange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = castOp(dtype, transpose(qkv[i], 2, 3));
    // Explicitly shard q, k, and v before calling SDPA node
    qkv[i]->setDeviceMesh(mesh);
    qkv[i]->axis(0)->parallelize(ParallelType::DIDx);
  }
  return qkv;
}
struct MHA_Results {
  TensorView* linear0;
  TensorView* sdpa_output;
  TensorView* linear1;
  TensorView* output;
  TensorView* sdpa_log_sumexp;
  TensorView* sdpa_philox_seed;
  TensorView* sdpa_philox_offset;
  TensorView* dropout_mask;
};

MHA_Results mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh) {
  const auto D = w0->axis(0)->extent()->value().as<int64_t>();
  auto dtype = w0->dtype();
  // Linear 0
  TensorView* matmul0 = matmul(x, w0);
  TensorView* linear0 = add(matmul0, broadcast(b0, {false, true, false}));
  // Forming the q,k,v vectors:
  TensorView* qkv_cat =
      reshape(linear0, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv = chunk(qkv_cat, 3, -1);
  for (auto i : c10::irange(3)) {
    qkv[i] = reshape(qkv[i], {D, B, S, E / D}, {D, B, S, H / D, E / H});
    qkv[i] = castOp(dtype, transpose(qkv[i], 2, 3));
    // Explicitly shard q, k, and v before calling SDPA node
    qkv[i]->setDeviceMesh(mesh);
    qkv[i]->axis(0)->parallelize(ParallelType::DIDx);
  }
  // SDPA
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv[0],
      qkv[1],
      qkv[2],
      IrBuilder::create<Val>(kSdpaProb),
      IrBuilder::create<Val>(true),
      nullptr);
  TensorView* sdpa_output = sdpa.output;
  // Linear 1
  TensorView* sdpa_transpose = transpose(sdpa_output, 2, 3);
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* local_matmul1 = matmul(sdpa_reshape, w1);
  TensorView* matmul1 = sum(local_matmul1, {0}); // allreduce
  TensorView* linear1 = add(matmul1, broadcast(b1, {true, false}));
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  auto dropout_tvs = dropout(linear1, prob, scale);

  for (auto tv : {x, b1, matmul1, linear1, dropout_tvs.output}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear0, sdpa_output}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {linear0, sdpa_output, linear1, dropout_tvs.output, 
    sdpa.log_sumexp, sdpa.philox_seed, sdpa.philox_offset, dropout_tvs.mask};
}

// TODO: These linear_backwards helper functions can be merged once
// we do not have logically split rfactor domain.
struct LinearBackwardsResult {
  TensorView* grad_x;
  TensorView* grad_w;
  TensorView* grad_b;
};

// x format: [i0, i1] dtype
// weight format: [DID(D), i1, i2/D] dtype
// grad format: [DID(D) i0, i2/D] float or dtype
// outputs: grad_x [i0, i1] dtype
// grad_w [DID i1, i2/D] dtype
// grad_b [DID i2/2] float
LinearBackwardsResult linear_backwards(
    TensorView* x,
    TensorView* w,
    TensorView* grad) {
  DataType dtype = w->dtype();
  TensorView* grad_f = maybeCastOp(DataType::Float, grad);
  TensorView* grad_q = maybeCastOp(dtype, grad);
  TensorView* w_t = transpose(w, 1, 2);
  TensorView* grad_x_partials = matmul(grad_q, w_t);
  TensorView* grad_x = sum(grad_x_partials, {0}); // allreduce
  TensorView* grad_q_t = transpose(grad_q, 1, 2);
  TensorView* grad_w_t = matmul(grad_q_t, x);
  TensorView* grad_w = transpose(grad_w_t, 1, 2);
  TensorView* grad_b = sum(grad_f, {1});

  return {grad_x, grad_w, grad_b};
}

// x format: [DID, i0, i1/D] dtype
// weight format: [DID, i1/D, i2] dtype
// grad format: [i0, i2] float
// outputs: grad_x [DID i0, i1/D] dtype
// grad_w [DID i1/D, i2] dtype
// grad_b [i2] float
LinearBackwardsResult sharded_linear_backwards(
    TensorView* x,
    TensorView* w,
    TensorView* grad) {
  DataType dtype = w->dtype();
  TensorView* grad_q = castOp(dtype, grad);
  TensorView* w_t = transpose(w, 1, 2);
  TensorView* grad_x = matmul(grad_q, w_t);
  TensorView* grad_t = transpose(grad_q, 0, 1);
  TensorView* grad_w_t = matmul(grad_t, x);
  TensorView* grad_w = transpose(grad_w_t, 1, 2);
  TensorView* grad_b = sum(grad, {0});

  return {grad_x, grad_w, grad_b};
}

// forward_transformer layer_norm with cached mean_bcast and invstd tensors to avoid
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

// Backwards MLP block. Recomputes linear0 and gelu
// if either isn't provided as input.
std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    const DeviceMesh& mesh,
    TensorView* linear0 = nullptr,
    TensorView* gelu = nullptr) {
  DataType dtype = w0->dtype();
  if (linear0 == nullptr) {
    TensorView* matmul0 = matmul(x, w0);
    linear0 = add(
        matmul0, broadcast(b0, {false, true, false})); // add generates float.
    linear0 = castOp(DataType::Float, linear0);
  }
  if (gelu == nullptr) {
    gelu = castOp(dtype, tanh_gelu(linear0));
  }

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
        b0,
        w1,
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
    TensorView* b0,
    TensorView* w1,
    TensorView* mask,
    TensorView* sdpa_output,
    TensorView* sdpa_log_sumexp,
    TensorView* sdpa_seed,
    TensorView* sdpa_offset,
    TensorView* grad,
    const std::vector<TensorView*>& qkv,
    const DeviceMesh& mesh) {
  DataType dtype = w0->dtype();
  const auto D = w0->axis(0)->extent()->value().as<int64_t>();
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
      /*scale=*/nullptr);

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
        b0,
        w1,
        sdpa_output,
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

at::Tensor shardTensor(
    at::Tensor tensor,
    int64_t axis,
    const DeviceMesh& mesh,
    Communicator* communicator_) {
  const auto device_id = communicator_->deviceId();
  auto i = mesh.idxOf(device_id);
  auto extent = tensor.size(axis);
  auto nslices = mesh.size();
  NVF_CHECK(
      extent % nslices == 0, "Sharded axis must be evenly divisble by mesh");
  auto stride = extent / nslices;
  // TODO: returning slice 0 temporarily when device is not in the mesh.
  i = (i < 0) ? 0 : i;
  auto slice = tensor.slice(axis, i * stride, (i + 1) * stride).contiguous();
  // Temporary until https://github.com/NVIDIA/Fuser/issues/2563. Adds DIDx
  // axis in front representing the sharded extent of the tensor.
  if (stride > 1) {
    slice = slice.unsqueeze(0);
  }
  return slice;
}

std::vector<at::Tensor> forward_transformer(Communicator* communicator_, bool profile) {
  int64_t D = communicator_->size();
  auto dtype = DataType::BFloat16;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = makeContigConcreteTensor({B * S, E}, DataType::Float);
  TensorView* ln0_w = makeContigTensor(1);
  TensorView* ln0_b = makeContigTensor(1);
  TensorView* mha_w0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E / D, E}, dtype);
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

  auto ln_1 = layer_norm(x, norm_shape, ln0_w, ln0_b, eps);
  auto mha_in = castOp(dtype, ln_1.output);
  auto mha_tvs = mha(mha_in, mha_w0, mha_b0, mha_w1, mha_b1, mesh);
  auto resid_1 = add(x, mha_tvs.output);
  auto ln_2 = layer_norm(resid_1, norm_shape, ln1_w, ln1_b, eps);
  auto mlp_in = castOp(dtype, ln_2.output);
  auto mlp_tvs = mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, mesh);
  auto resid_2 = add(resid_1, mlp_tvs.output);

  fusion->addOutput(resid_2); // output of the layer
  // Cached TVs for backward pass
  fusion->addOutput(ln_1.mean);
  fusion->addOutput(ln_1.invstd);
  fusion->addOutput(mha_tvs.sdpa_output);
  fusion->addOutput(mha_tvs.sdpa_log_sumexp);
  fusion->addOutput(mha_tvs.sdpa_philox_seed);
  fusion->addOutput(mha_tvs.sdpa_philox_offset);
  fusion->addOutput(mha_tvs.linear1);
  fusion->addOutput(mha_tvs.dropout_mask);
  fusion->addOutput(ln_2.mean);
  fusion->addOutput(ln_2.invstd);
  fusion->addOutput(mlp_tvs.dropout_mask);

  for (auto tv : {x, ln_1.output, ln_2.output, resid_2, ln0_w, ln0_b, ln1_w, ln1_b}) {
    tv->setDeviceMesh(mesh);
  }

  shardBetween({mha_in->definition()}, {mha_tvs.output->definition()}, mha_w0);
  shardBetween({mlp_in->definition()}, {mlp_tvs.output->definition()}, mlp_w0);
  shardBetween({x}, {mha_in}, x);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);

  auto ln0_w_ = at::randn(E, options).to(at::kFloat);
  auto ln0_b_ = at::randn(E, options).to(at::kFloat);

  auto mha_w0_ = at::randn({E, 3 * E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;

  auto ln1_w_ = at::randn(E, options).to(at::kFloat);
  auto ln1_b_ = at::randn(E, options).to(at::kFloat);

  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  std::vector<c10::IValue> inputs = {
      x_,
      ln0_w_,
      ln0_b_,
      shardTensor(mha_w0_.view({E, 3, E}), 2, mesh, communicator_).view({1, E, 3 * E / D}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh, communicator_).view({1, 3 * E / D}),
      shardTensor(mha_w1_, 0, mesh, communicator_),
      mha_b1_,
      ln1_w_,
      ln1_b_,
      shardTensor(mlp_w0_, 1, mesh, communicator_),
      shardTensor(mlp_b0_, 0, mesh, communicator_),
      shardTensor(mlp_w1_, 0, mesh, communicator_),
      mlp_b1_};

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  std::vector<at::Tensor> outputs;

  cudaSetDevice(communicator_->deviceId());
  std::cout << "PROFILE? " << profile << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : c10::irange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      start = std::chrono::high_resolution_clock::now();
      if (profile) {
        cudaProfilerStart();
      }
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("Iteration" + std::to_string(i)).c_str());
    }
    outputs = fec.runFusionWithInputs(inputs);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize is not blocking until kernels are finished on all devices except 0
    // TODO: are we not waiting until all kernels are appended to the stream?
    std::cout << outputs[0][0][0] << std::endl; 

    if (i > warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (profile) {
    cudaProfilerStop();
  }

  double foward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double) num_itrs / 1000.0;
  std::cout << communicator_->deviceId() << ": Average forward time " << foward_time << "ms" << std::endl;
  return outputs;
}


void backward_transformer(Communicator* communicator_, bool profile, std::vector<at::Tensor> fwd_tensors) {
  auto dtype = DataType::BFloat16;
  int64_t D = communicator_->size();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
  constexpr float kEps = 1e-5;
  std::vector<int64_t> norm_shape{E};

  TensorView* x = makeContigConcreteTensor({B * S, E});
  TensorView* grad = makeContigTensor(2);
  TensorView* mha_w0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* mlp_w0 = makeContigTensor(3, dtype);
  TensorView* mlp_b0 = makeContigTensor(2, dtype);
  TensorView* mlp_w1 = makeContigTensor(3, dtype);
  TensorView* mlp_b1 = makeContigTensor(1, dtype);
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
  TensorView* mha_linear1 = makeContigTensor(2);

  fusion->addInput(x);
  fusion->addInput(grad);
  fusion->addInput(mha_w0);
  fusion->addInput(mha_b0);
  fusion->addInput(mha_w1);
  fusion->addInput(mlp_w0);
  fusion->addInput(mlp_b0);
  fusion->addInput(mlp_w1);
  fusion->addInput(mlp_b1);
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
  fusion->addInput(mha_linear1);

  // Recomputation: Recompute, mha linear0, qkv, mlp linear0, and mlp gelu.
  // Partially recompute layer norms using cached statistics.
  // Note: The thunder trace recompute mha linear1, but this would result in 3
  // AllReduces in the backwards pass.
  auto ln_0 = layer_norm_with_cached_statistics(
      x, ln0_mean, ln0_rstd, norm_shape, ln0_w, ln0_b);
  auto mha_in = castOp(dtype, ln_0);
  auto qkv = mha_qkv(mha_in, mha_w0, mha_b0, mesh);

  Val* dropout_scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  // Use input mha_mask to implement dropout
  auto mha_out = mul(mha_linear1, mha_mask);
  mha_out = mul(mha_out, dropout_scale);
  auto resid_0 = add(x, mha_out);
  auto ln_1 = layer_norm_with_cached_statistics(
      resid_0, ln1_mean, ln1_rstd, norm_shape, ln1_w, ln1_b);
  auto mlp_in = castOp(dtype, ln_1);
  // Note: We only use linear0 and gelu outputs from the mlp forward pass.
  auto mlp_tvs = mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, mesh);

  // Backwards
  auto mlp_grads = mlp_backwards(
      grad,
      mlp_in,
      mlp_mask,
      mlp_w0,
      mlp_b0,
      mlp_w1,
      mesh,
      mlp_tvs.linear0,
      mlp_tvs.gelu);
  auto ln1_grads = layer_norm_backward(
      castOp(DataType::Float, mlp_grads[6]),
      resid_0,
      norm_shape,
      ln1_mean,
      ln1_rstd,
      ln1_w,
      ln1_b,
      {true, true, true});
  auto resid1_grad = add(ln1_grads.grad_input, grad);
  auto mha_grads = mha_backwards(
      mha_in,
      mha_w0,
      mha_b0,
      mha_w1,
      mha_mask,
      mha_sdpa_out,
      mha_sdpa_log_sumexp,
      mha_sdpa_seed,
      mha_sdpa_offset,
      resid1_grad,
      qkv,
      mesh);
  auto ln0_grads = layer_norm_backward(
      castOp(DataType::Float, mha_grads[8]),
      x,
      norm_shape,
      ln0_mean,
      ln0_rstd,
      ln0_w,
      ln0_b,
      {true, true, true});
  auto dx = add(ln0_grads.grad_input, resid1_grad);

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
  // by mlp_backward, mha_backward, or mlp.
  for (auto* tv :
       {mha_linear1,
        ln0_w,
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
  for (auto* tv : {mha_w0, mha_b0, mha_w1, mha_sdpa_out, mha_sdpa_log_sumexp}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  // Sharded inputs to outputs
  shardBetween(
      {mha_w0, mha_b0, mha_w1, mlp_w0, mlp_w1, mlp_b0, mha_sdpa_out},
      {mlp_grads[1],
       mlp_grads[4],
       mlp_grads[5],
       mha_grads[1],
       mha_grads[6],
       mha_grads[7]},
      mha_w0);
    shardBetween(
      {mha_w0, mha_b0, mha_w1, mlp_w0, mlp_w1, mlp_b0, mha_sdpa_out},
      {mlp_grads[1],
       mlp_grads[4],
       mlp_grads[5],
       mha_grads[1],
       mha_grads[6],
       mha_grads[7]},
      mlp_w0);


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
       ln1_b,
       mlp_b1},
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
  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto ln0_w_ = at::randn(E, options).to(at::kFloat);
  auto ln0_b_ = at::randn(E, options).to(at::kFloat);
  auto mha_w0_ = at::randn({E, 3 * E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;
  auto ln1_w_ = at::randn(E, options).to(at::kFloat);
  auto ln1_b_ = at::randn(E, options).to(at::kFloat);
  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto grad_ = at::randn({B * S, E}, options).to(at::kFloat) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  std::vector<c10::IValue> inputs = {
      x_,
      grad_,
      shardTensor(mha_w0_.view({E, 3, E}), 2, mesh, communicator_).view({1, E, 3 * E / D}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh, communicator_).view({1, 3 * E / D}),
      shardTensor(mha_w1_, 0, mesh, communicator_),
      shardTensor(mlp_w0_, 1, mesh, communicator_),
      shardTensor(mlp_b0_, 0, mesh, communicator_),
      shardTensor(mlp_w1_, 0, mesh, communicator_),
      mlp_b1_,
      fwd_tensors[8], // mlp dropout mask
      fwd_tensors[11], // mha dropout mask
      fwd_tensors[3], // sdpa output
      fwd_tensors[4], // sdpa logsum_exp
      fwd_tensors[5], // sdpa seed
      fwd_tensors[6], // sdpa offset
      ln1_w_,
      ln1_b_,
      fwd_tensors[9],
      fwd_tensors[10],
      ln0_w_,
      ln0_b_,
      fwd_tensors[1],
      fwd_tensors[2],
      fwd_tensors[7]
  };

  FusionExecutorCache fec(std::move(fusion));
  at::manual_seed(getATenRandomSeed());
  std::vector<at::Tensor> outputs;

  cudaSetDevice(communicator_->deviceId());

  auto start = std::chrono::high_resolution_clock::now();
  for (auto i : c10::irange(num_itrs + warmup_itrs)) {
    if (i == warmup_itrs) {
      start = std::chrono::high_resolution_clock::now();
      if (profile) {
        cudaProfilerStart();
      }
    }
    if (i >= warmup_itrs && profile) {
      nvtxRangePush(("Iteration" + std::to_string(i)).c_str());
    }
    outputs = fec.runFusionWithInputs(inputs);
    cudaDeviceSynchronize();
    // cudaDeviceSynchronize is not blocking until kernels are finished on all devices except 0
    // TODO: are we not waiting until all kernels are appended to the stream?
    std::cout << outputs[0][0][0][0] << std::endl; 

    if (i > warmup_itrs && profile) {
      nvtxRangePop();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  if (profile) {
    cudaProfilerStop();
  }

  double backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (double) num_itrs / 1000.0;
  std::cout << communicator_->deviceId() << ": Average backward time " << backward_time << "ms" << std::endl;

}

int main(int argc, char** argv) {
  // using this is as a flag for when to profile
  bool profile = argc > 1;
  auto communicator_ = &Communicator::getInstance();
  auto outputs = forward_transformer(communicator_, false);
  backward_transformer(communicator_, profile, outputs);
}
