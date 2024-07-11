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
  const int B = 4;
  const int E = 128;
  const int H = 4;
  const int S = 32;

  // Temporary until https://github.com/NVIDIA/Fuser/issues/2563
  at::Tensor shardTensor(at::Tensor tensor, int64_t axis, DeviceMesh& mesh) {
    auto i = mesh.idxOf(communicator_->deviceId());
    auto extent = tensor.size(axis);
    auto nslices = mesh.size();
    NVF_CHECK(
        extent % nslices == 0,
        "Sharded axis must be evenly divisble by mesh");
    auto stride = extent / nslices;
    // TODO: returning slice 0 temporarily when device is not in the mesh.
    i = (i < 0) ? 0 : i;
    return tensor.slice(axis, i * stride, (i + 1) * stride)
                 .contiguous()
                 .unsqueeze(0);
  }

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
    // Note: Scale the tolerance up since the error accumulates across ops
    double tolerance = 0.5 * (i + 1);
    auto all_close = expected_out[i].allclose(
        out[i].to(expected_out[i].dtype()),
        tolerance,
        tolerance,
        /*equal_nan=*/true);

    if (!all_close) {
      auto max_error =
          (expected_out[i].sub(out[i])).abs().max().item().to<double>();
      auto max_relative_error = (max_error / expected_out[i].abs().max()).item();
      auto error_count =
          at::sum((expected_out[i].sub(out[i])).abs() > tolerance).item();
      std::cout << "output[" << i << "] max error: " << max_error << std::endl;
      std::cout << "          max relative error: " << max_relative_error
                << std::endl;
       std::cout << error_count << " elements failing "
                << error_count.to<float>() / at::numel(out[i]) * 100.0 << "\% of tensor" << std::endl;
    }
    EXPECT_TRUE(all_close);
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
  auto x_ = at::randn({B * S, E}, options) / 10.0;
  auto w0_ = at::randn({4 * E, E}, options);
  auto b0_ = at::randn({4 * E}, options);
  auto w1_ = at::randn({E, 4 * E}, options);
  auto b1_ = at::randn({E}, options);

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(w0_, 0, mesh),
      shardTensor(b0_, 0, mesh),
      shardTensor(w1_, 1, mesh),
      b1_};
  at::manual_seed(0);
  auto linear1_aten = at::matmul(x_, w0_.transpose(1, 0)).add(b0_).to(at::kFloat);
  auto gelu_aten = at::gelu(linear1_aten, "tanh");
  auto linear2_aten =
      at::matmul(gelu_aten.to(at_dtype), w1_.transpose(1, 0)).add(b1_).to(at::kFloat);
  auto dropout_aten = at::dropout(linear2_aten, kProb, true);
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(linear1_aten, 1, mesh),
      shardTensor(gelu_aten, 1, mesh),
      linear2_aten,
      dropout_aten};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedTransformerTest,
    testing::Combine(
        testing::Bool(),
        testing::Values(DataType::Half, DataType::BFloat16)));
} // namespace nvfuser
