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
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

int64_t D = 1;
constexpr int64_t B = 2, E = 512, H = 4, S = 128;
constexpr double kDropoutProb = 0.1, kSdpaProb = 0.1, kSdpaScale = 1e-3;
// Note:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
constexpr double scale = 0.02;

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest()
      : optimization_guard_(false), allocation_order_guard_(false) {
    D = communicator_->size();
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

void validate(
    std::vector<at::Tensor> expected_out,
    std::vector<at::Tensor> out) {
  EXPECT_EQ(expected_out.size(), out.size());
  for (auto i : c10::irange(out.size())) {
    std::cout << "Out size " << out[i].sizes() << " expected " << expected_out[i].sizes() << std::endl;
    // Note: Scaling tolerance up since the error accumulates across ops
    // BFloat16 error is quite high, but the program has been verified with
    // double precision to be logically correct.
    double atol = 1.0 * (i + 1);
    double rtol = 1.6e-2; // Default for pytorch bfloat16
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
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  at::manual_seed(0);
  auto linear1 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear1, "tanh");
  auto linear2 = at::matmul(gelu.to(at_dtype), w1).add(b1).to(at::kFloat);
  auto dropout = at::dropout(linear2, kDropoutProb, true);
  return std::make_tuple(linear1, gelu, linear2, dropout);
}

std::tuple<TensorView*, TensorView*, TensorView*, TensorView*> mlp(
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

  return std::make_tuple(linear1, gelu, linear2, dropout);
}

TEST_P(DistributedTransformerTest, MLP_Layer) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
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

  auto [tvlinear1, tvgelu, tvlinear2, tvdropout] =
      mlp(tvx, tvw0, tvb0, tvw1, tvb1, fusion.get(), mesh, dtype);

  fusion->addOutput(tvlinear1);
  fusion->addOutput(tvgelu);
  fusion->addOutput(tvlinear2);
  fusion->addOutput(tvdropout);

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 4 * E}, options) * scale;
  auto b0 = at::randn({4 * E}, options) * scale;
  auto w1 = at::randn({4 * E, E}, options) * scale;
  auto b1 = at::randn({E}, options) * scale;

  auto [linear1, gelu, linear2, dropout] =
      reference_mlp(x, w0, b0, w1, b1, at_dtype);

  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0, 1, mesh),
      shardTensor(b0, 0, mesh),
      shardTensor(w1, 0, mesh),
      b1};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(linear1, 1, mesh),
      shardTensor(gelu, 1, mesh),
      linear2,
      dropout};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

TEST_F(DistributedTransformerTest, MLP_Backward) {
  auto dtype = DataType::Half;
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* grad = makeContigTensor(2, DataType::Float);
  TensorView* x = makeContigTensor(2, dtype);
  TensorView* mask = makeContigTensor(2, DataType::Bool);
  TensorView* w0 = makeContigTensor(3, dtype); // sharded
  TensorView* b0 = makeContigTensor(2, dtype); // sharded
  TensorView* w1 = makeContigTensor(3, dtype); // sharded

  fusion->addInput(grad);
  fusion->addInput(x);
  fusion->addInput(mask);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);

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

  // reshape
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
  TensorView* matmul0_grad_x = sum(matmul0_grad_x_partial, {0}); //allreduce
  TensorView* grad_gelu_t = transpose(gelu_grad_f, 1, 2);
  TensorView* matmul0_grad_w_t = matmul(grad_gelu_t, x);
  TensorView* matmul0_grad_w = transpose(matmul0_grad_w_t, 1, 2);
  TensorView* matmul0_grad_b = sum(gelu_grad, {1});

  fusion->addOutput(dropout_grad);
  fusion->addOutput(matmul1_grad_w);
  fusion->addOutput(matmul1_grad_b);
  fusion->addOutput(gelu_grad);
  fusion->addOutput(matmul0_grad_w);
  fusion->addOutput(matmul0_grad_b);
  fusion->addOutput(matmul0_grad_x);


  for (auto tv : {x, grad, mask, dropout_grad, matmul1_grad_x, matmul1_grad_b, matmul0_grad_x}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv : {w0, b0, w1, matmul1_grad_x, matmul1_grad_w, matmul1_grad_w_t, gelu_grad,
    matmul0_grad_w_t, matmul0_grad_w, matmul0_grad_x_partial, matmul0_grad_b}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B*S, E}, options);
  auto grad_ = at::randn({B*S, E}, options).to(at::kFloat);
  auto mask_ = at::randn({B*S, E}, options).lt(1.0 - kDropoutProb);
  auto mlp_w0_ = at::randn({E, 4 * E}, options) * scale;
  auto mlp_b0_ = at::randn({4 * E}, options) * scale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * scale;

  // recompute activations:
  auto linear1_ = at::matmul(x_, mlp_w0_).add(mlp_b0_).to(at::kFloat);
  auto gelu_ = at::gelu(linear1_, "tanh");

  auto dropout_grad_ = at::native_dropout_backward(grad_, mask_, kScale);
  auto dropout_grad_h_ = dropout_grad_.to(at_dtype);
  auto matmul_grad_ = at::matmul(dropout_grad_h_, mlp_w1_.transpose(0,1));
  auto matmul_grad_w_ = at::matmul(dropout_grad_h_.transpose(0,1), gelu_.to(at_dtype)).transpose(0,1);
  auto matmul_grad_bias_ = at::sum(dropout_grad_, {0});
  auto gelu_grad_ = at::gelu_backward(matmul_grad_.to(at::kFloat), gelu_, "tanh");
  auto gelu_grad_h_ = gelu_grad_.to(at_dtype);
  auto matmul1_grad_bias_ = at::sum(gelu_grad_, {0});
  auto matmul1_grad_ = at::matmul(gelu_grad_h_, mlp_w0_.transpose(0, 1));
  auto matmul1_grad_w_ = at::matmul(gelu_grad_h_.transpose(0,1), x_).transpose(0,1);

  std::vector<c10::IValue> inputs = {grad_, x_, mask_,
      shardTensor(mlp_w0_, 1, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 0, mesh)};
  std::vector<at::Tensor> expected_outputs = {dropout_grad_, 
    shardTensor(matmul_grad_w_, 0, mesh),
    matmul_grad_bias_,
    shardTensor(gelu_grad_, 1, mesh),
    shardTensor(matmul1_grad_w_, 1, mesh),
    shardTensor(matmul1_grad_bias_, 0, mesh),
    matmul1_grad_};

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
