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
#include <ir/iostream.h>
#include <ir/printer.h>
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

class DistributedMatmulTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<std::tuple<bool, DataType>> {
 protected:
  DistributedMatmulTest()
      : num_devices_(communicator_->size()),
        optimization_guard_(false),
        allocation_order_guard_(false) {}

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed matmul tests require Ampere or newer";
    }
  }

  MultiDeviceExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};
  const int num_devices_;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> getInputsAndReferenceOutputs(
      MmaLayout layout,
      int M,
      int N,
      int K,
      c10::ScalarType dtype) {
    int local_rank = communicator_->local_rank();
    c10::ScalarType type = c10::ScalarType::Half;
    auto a = matmulAtInput2D(
        layout, TensorMatmulPos::A, type, M, N, K, 0, local_rank);
    auto b = matmulAtInput2D(
        layout, TensorMatmulPos::B, type, M, N, K, 0, local_rank);
    auto c = atMatmul(a.to(at::kDouble), b.to(at::kDouble), layout).to(dtype);
    return std::make_tuple(a, b, c);
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

TEST_F(DistributedMatmulTest, MulSum_LayoutTN_NoComms) {
  // MmaLayout::TN A(T), B(N), C(T)
  // A and C are sharded on dimension M
  // Tests local matmul with no communication
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());
  int M = 256, N = 64, K = 64;
  int Mo = num_devices_;
  int Mi = M / Mo;
  std::vector<int> a_shape = {Mo, Mi, K};
  std::vector<int> b_shape = {N, K};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
  TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
  TensorView* a_b = broadcast(a, {false, false, true, false}); // (Mo,Mi,b,K)
  TensorView* b_b = broadcast(b, {true, true, false, false}); // (b,b,N,K)
  TensorView* ab = mul(a_b, b_b); // (Mo,Mi,N,K)
  TensorView* c = sum(ab, {-1}); // (Mo,Mi,N,r)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, a_b, b_b, ab, c};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);
  // TODO: If c's allocation domain isn't set, it will fail validation at
  // csrc/device_lower/validation.cpp:419, Vectorized dim for consumer has to be
  // from a contiguous inner most position.
  c->setAllocationDomain(c->getLoopDomain(), true);
  auto [in0, in1, out] = getInputsAndReferenceOutputs(
      MmaLayout::TN, M, N, K, /*dtype=*/at::kFloat);
  in0 = in0.view({Mo, Mi, K});
  out = out.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()), in1};
  auto expected_output = shardTensor(out, c, communicator_->deviceId());
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);

  std::vector<FusionExecutorCache*> fecs = runtime.getFusionExecutorCaches();
  EXPECT_EQ(fecs.size(), 1);

  const FusionKernelRuntime* kernel_runtime =
      fecs.front()->getMostRecentKernelRuntime();
  EXPECT_FALSE(kernel_runtime->isSegmented());

  ScheduleHeuristic heuristic = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .front()
                                    ->heuristic();
  EXPECT_EQ(heuristic, ScheduleHeuristic::Matmul);
}

TEST_F(DistributedMatmulTest, Matmul_LayoutTN_NoComms) {
  // MmaLayout::TN A(T), B(N), C(T)
  // A and C are sharded on dimension M
  // Tests local matmul with no communication
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int M = 256, N = 64, K = 64;
  int Mo = num_devices_;
  int Mi = M / Mo;
  std::vector<int> a_shape = {Mo, Mi, K};
  std::vector<int> b_shape = {N, K};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
  TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
  TensorView* b_t = transpose(b, 0, 1); // (K,N)
  TensorView* c = matmul(a, b_t); //(Mo,Mi,N,r)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, c};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);

  // TODO: If c's allocation domain isn't set, it will fail validation at
  // csrc/device_lower/validation.cpp:419, Vectorized dim for consumer has to be
  // from a contiguous inner most position.
  c->setAllocationDomain(c->getLoopDomain(), true);

  auto [in0, in1, out] =
      getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K, /*dtype=*/at::kHalf);
  in0 = in0.view({Mo, Mi, K});
  out = out.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()), in1};
  auto expected_output = shardTensor(out, c, communicator_->deviceId());

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);

  std::vector<FusionExecutorCache*> fecs = runtime.getFusionExecutorCaches();
  EXPECT_EQ(fecs.size(), 1);

  const FusionKernelRuntime* kernel_runtime =
      fecs.front()->getMostRecentKernelRuntime();
  EXPECT_TRUE(kernel_runtime->isSegmented());

  ScheduleHeuristic heuristic = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .at(1)
                                    ->heuristic();
  EXPECT_EQ(heuristic, ScheduleHeuristic::ExprEval);
}

TEST_F(DistributedMatmulTest, Matmul_LayoutTN_Allgather) {
  // MmaLayout::TN matmul A(T), B(N), C(T)
  // A is sharded on dimension M
  // Tests local matmul + allgather
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int M = 256, N = 64, K = 64;
  int Mo = num_devices_;
  int Mi = M / Mo;
  std::vector<int> a_shape = {Mo, Mi, K};
  std::vector<int> b_shape = {N, K};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
  TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
  TensorView* b_t = transpose(b, 0, 1); // (K,N)
  TensorView* c0 = matmul(a, b_t); //(Mo,Mi,N,r)
  TensorView* c = set(c0);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);

  auto [in0, in1, out] =
      getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K, /*dtype=*/at::kHalf);
  in0 = in0.view({Mo, Mi, K});
  out = out.view({Mo, Mi, N});

  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()), in1};
  auto expected_output = shardTensor(out, c, communicator_->deviceId());
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);

  std::vector<FusionExecutorCache*> fecs = runtime.getFusionExecutorCaches();
  EXPECT_EQ(fecs.size(), 1);

  const FusionKernelRuntime* kernel_runtime =
      fecs.front()->getMostRecentKernelRuntime();
  EXPECT_TRUE(kernel_runtime->isSegmented());

  ScheduleHeuristic heuristic = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .at(1)
                                    ->heuristic();
  EXPECT_EQ(heuristic, ScheduleHeuristic::ExprEval);
}

TEST_F(DistributedMatmulTest, Matmul_LayoutNT_AllReduce) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // Sharding: A, B are sharded along K. C is replicated.
  // Tests local matmul + allreduce
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  // Transpose into TT layout, keep Ko (device axis) as the outermost.
  TensorView* a_t = transpose(a, 1, 2); // (Ko,M,Ki)
  TensorView* c0 = matmul(a_t, b); // (Ko,M,N,r)
  TensorView* c = sum(c0, {0}); // (r,M,N)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Parallelize K on all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  c->setDeviceMesh(mesh);

  auto [in0, in1, out] =
      getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K, /*dtype=*/at::kHalf);
  in0 = in0.view({Ko, Ki, M});
  in1 = in1.view({Ko, Ki, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()),
      shardTensor(in1, b, communicator_->deviceId())};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(), outputs, inputs, {out}, __LINE__, __FILE__);

  std::vector<FusionExecutorCache*> fecs = runtime.getFusionExecutorCaches();
  EXPECT_EQ(fecs.size(), 1);

  const FusionKernelRuntime* kernel_runtime =
      fecs.front()->getMostRecentKernelRuntime();
  EXPECT_TRUE(kernel_runtime->isSegmented());

  ScheduleHeuristic heuristic = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .at(1)
                                    ->heuristic();
  EXPECT_EQ(heuristic, ScheduleHeuristic::ExprEval);
}

TEST_F(DistributedMatmulTest, Matmul_LayoutNT_ReduceScatter) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // A, B are sharded on K. C is sharded on M
  // Tests local matmul + reduce scatter
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  int Mo = num_devices_, Mi = M / Mo;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
  TensorView* c0 = matmul(a_t, b); // (Ko,M,N,r)
  c0 = segment_set(c0);
  std::vector<int64_t> orig_size = {Ko, M, N};
  std::vector<int64_t> new_size = {Ko, Mo, Mi, N};
  TensorView* c1 = reshape(c0, orig_size, new_size);
  TensorView* c = sum(c1, {0});

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding K dimension of all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, c0, c1};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  // Sharding M on output
  c->setDeviceMesh(mesh);
  c->axis(1)->parallelize(ParallelType::DIDx);

  auto [in0, in1, out] =
      getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K, /*dtype=*/at::kHalf);
  in0 = in0.view({Ko, Ki, M});
  in1 = in1.view({Ko, Ki, N});
  out = out.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()),
      shardTensor(in1, b, communicator_->deviceId())};
  auto expected_output =
      shardTensor(out, c, communicator_->deviceId()).view({1, Mi, N});

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);

  std::vector<FusionExecutorCache*> fecs = runtime.getFusionExecutorCaches();
  EXPECT_EQ(fecs.size(), 1);

  const FusionKernelRuntime* kernel_runtime =
      fecs.front()->getMostRecentKernelRuntime();
  EXPECT_TRUE(kernel_runtime->isSegmented());

  ScheduleHeuristic heuristic = kernel_runtime->schedulerHeuristics()
                                    ->heuristicsList()
                                    .at(1)
                                    ->heuristic();
  EXPECT_EQ(heuristic, ScheduleHeuristic::ExprEval);
}

TEST_F(DistributedMatmulTest, Segmentation) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  TensorView* bias = makeContigTensor(1, DataType::Half);
  // Transpose into TT layout, keep Ko (device axis) as the outermost.
  TensorView* a_t = transpose(a, 1, 2); // (Ko,M,Ki)
  TensorView* c0 = matmul(a_t, b); // (Ko,M,N,r)
  TensorView* c = sum(c0, {0}); // (r,M,N)
  TensorView* bias_bcast = broadcast(bias, {true, false});
  TensorView* linear = add(c, bias_bcast);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(linear);

  // Parallelize K on all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  c->setDeviceMesh(mesh);
  bias->setDeviceMesh(mesh);
  linear->setDeviceMesh(mesh);

  auto [in0, in1, out] =
      getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K, /*dtype=*/at::kHalf);
  in0 = in0.view({Ko, Ki, M});
  in1 = in1.view({Ko, Ki, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator_->deviceId()),
      shardTensor(in1, b, communicator_->deviceId())};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
}

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
        1e-4,
        1e-4,
        /*equal_nan=*/true);

    auto max_error =
        expected_out[i].sub(out[i]).abs().max().item().to<double>();
    auto max_relative_error = ((expected_out[i].sub(out[i])) / expected_out[i])
                                  .abs()
                                  .max()
                                  .item()
                                  .to<double>();
    std::cout << "Max error " << i << " " << max_error << " max rel. error "
              << max_relative_error << std::endl;
    EXPECT_TRUE(all_close);
  }
}
} // namespace

TEST_P(DistributedMatmulTest, MLP_Layer) {
  auto [use_aten_matmul, dtype] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int64_t sb = 64; // sequence * batch
  int64_t h = 128;
  int64_t h4 = 4 * h;

  at::ScalarType at_dtype = at::kDouble;
  if (dtype == DataType::Double) {
    at_dtype = at::kDouble;
  } else if (dtype == DataType::Float) {
    at_dtype = at::kFloat;
  } else if (dtype == DataType::Half) {
    at_dtype = at::kHalf;
  } else if (dtype == DataType::BFloat16) {
    at_dtype = at::kBFloat16;
  }

  TensorView* x = makeContigConcreteTensor({sb, h}, dtype);
  TensorView* w0 =
      makeContigConcreteTensor({num_devices_, h4 / num_devices_, h}, dtype);
  TensorView* b0 =
      makeContigConcreteTensor({num_devices_, h4 / num_devices_}, dtype);
  TensorView* w1 =
      makeContigConcreteTensor({num_devices_, h, h4 / num_devices_}, dtype);
  TensorView* b1 = makeContigConcreteTensor({h}, dtype);

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
  auto x_ = at::randn({sb, h}, options);
  auto w0_ = at::randn({h4, h}, options);
  auto b0_ = at::randn({h4}, options);
  auto w1_ = at::randn({h, h4}, options);
  auto b1_ = at::randn({h}, options);

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(
          w0_.view({num_devices_, h4 / num_devices_, h}),
          w0,
          communicator_->deviceId()),
      shardTensor(
          b0_.view({num_devices_, h4 / num_devices_}),
          b0,
          communicator_->deviceId()),
      shardTensor(
          w1_.view({h, num_devices_, h4 / num_devices_}).transpose(1, 0),
          w1,
          communicator_->deviceId()),
      b1_};
  at::manual_seed(0);
  auto linear1_aten =
      at::linear(x_.to(at::kDouble), w0_.to(at::kDouble), b0_.to(at::kDouble))
          .to(at::kFloat);
  auto gelu_aten = at::gelu(linear1_aten, "tanh");
  auto linear2_aten = at::linear(
                          gelu_aten.to(at_dtype).to(at::kDouble),
                          w1_.to(at::kDouble),
                          b1_.to(at::kDouble))
                          .to(at::kFloat);
  auto dropout_aten = at::dropout(linear2_aten, kProb, true);
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(
          at::transpose(
              linear1_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
          linear1,
          communicator_->deviceId()),
      shardTensor(
          at::transpose(
              gelu_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
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

TEST_F(DistributedMatmulTest, Multiheaded_Attention) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  // input would be 3D (sequence, batch, embedding)
  // to simplify we make it 2D immediately (sequence * batch, embedding)
  int B = 5; // batch size
  int E = 16; // embedding size
  int H = 2; // number of heads
  int S = 32; // sequence length
  // int D = num_devices_;
  NVF_CHECK(E % H == 0);

  TensorView* x = makeContigConcreteTensor({B * S, E}, DataType::BFloat16);
  // Note: 3*E, the non-reduction axis will be sharded.
  TensorView* w0 = makeContigConcreteTensor({E, 3 * E}, DataType::BFloat16);
  TensorView* b0 = makeContigConcreteTensor({3 * E}, DataType::BFloat16);
  TensorView* w1 = makeContigConcreteTensor({E, E}, DataType::BFloat16);
  TensorView* b1 = makeContigConcreteTensor({E}, DataType::BFloat16);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  // TODO:
  // linear #1 is sharded along the columns (heads)
  // linear #2 is sharded along the rows + allreduce to reform x
  // Self attention linear
  TensorView* mm = matmul(x, w0);
  TensorView* proj_bias_bcast = broadcast(b0, {true, false});
  TensorView* qkv = add(mm, proj_bias_bcast);
  // Forming the q,k,v vectors:
  qkv = reshape(qkv, {B * S, 3 * E}, {B, S, 3 * E});
  std::vector<TensorView*> qkv_reshaped = {};
  for (auto i : c10::irange(3)) {
    TensorView* tv_slice = slice(qkv, {0, 0, E * i}, {B, S, E * (i + 1)});
    // Reshape all the vectors into (B,S,E) -> (B,S,H,E/H) -> (B,H,S,E/H)
    TensorView* tv_reshape = reshape(tv_slice, {B, S, E}, {B, S, H, E / H});
    TensorView* tv_trans = transpose(tv_reshape, 2, 1);
    TensorView* tv_cast = castOp(DataType::BFloat16, tv_trans);
    // TODO: We need this segment to force SDPA op into an isolated segment or
    // else when the segmenter tries different segments it tries to build a
    // compute at graph with SDPA which is not supported yet due to different
    // sized outputs.
    tv_cast = segment_set(tv_cast);
    qkv_reshaped.push_back(tv_cast);
  }

  // SDPA
  constexpr double kProb = 0.0;
  constexpr double kScale = 1.0 / (1.0 - kProb);
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv_reshaped[0],
      qkv_reshaped[1],
      qkv_reshaped[2],
      IrBuilder::create<Val>(kProb),
      IrBuilder::create<Val>(false),
      IrBuilder::create<Val>(kScale));
  TensorView* sdpa_output = segment_set(sdpa.output);
  
  // Linear projection
  TensorView* sdpa_transpose = transpose(sdpa_output, 1, 2); // B, S, H, E/H
  // Note: We have to reshape into a 2D tensor instead of 3D
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {B, S, H, E / H}, {B * S, E});
  TensorView* mm2 = matmul(sdpa_reshape, w1);
  TensorView* b1_bcast = broadcast(b1, {true, false});
  TensorView* linear2 = add(mm2, b1_bcast);

  // Dropout
  const double kDropoutProb = 0.1;
  TensorView* dropout =
      replicated_dropout(linear2, kDropoutProb, fusion.get(), mesh);

  fusion->addOutput(qkv);
  fusion->addOutput(sdpa_output);
  fusion->addOutput(linear2);
  fusion->addOutput(dropout);

  //  for (TensorView* tv : {x, w0, b0, mm, q, k, v, sdpa.output}) {
  //     tv->setDeviceMesh(mesh);
  //  }

  const auto options = at::TensorOptions()
                           .dtype(at::kBFloat16)
                           .device(at::kCUDA, communicator_->local_rank());
  auto x_ = at::randn({B * S, E}, options);
  auto w0_ = at::randn({E, 3 * E}, options);
  auto b0_ = at::randn({3 * E}, options);
  auto w1_ = at::randn({E, E}, options);
  auto b1_ = at::randn({E}, options);
  auto m_ = at::linear(
                x_.to(at::kDouble),
                w0_.transpose(1, 0).to(at::kDouble),
                b0_.to(at::kDouble))
                .view({B, S, 3 * E})
                .to(at::kBFloat16);
  auto qkv_vec = m_.split(E, 2);
  // move vectors from (B, T, S) to (B, T, H, S/H) to (B, H, T, S/H)
  for (auto i = 0; i < 3; i++) {
    qkv_vec[i] = qkv_vec[i].reshape({B, S, H, E / H}).transpose(2, 1);
  }
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv_vec[0], qkv_vec[1], qkv_vec[2], kProb, true, false, kScale);
  auto sdpa_ = std::get<0>(sdpa_out);
  // Reassemble heads
  std::cout << "ATen sdpa size " << sdpa_.sizes() << std::endl;
  auto y = sdpa_.transpose(1, 2).reshape({B, S, E});
  auto y_proj = at::linear(y, w1_, b1_);
  auto y_dropout = at::dropout(y_proj, kDropoutProb, true);

  std::vector<c10::IValue> inputs = {x_, w0_, b0_, w1_, b1_};
  std::vector<at::Tensor> expected_outputs = {m_, sdpa_, y_proj, y_dropout};

  FusionExecutorCache fec(std::move(fusion));
  auto out = fec.runFusionWithInputs(inputs);
  // MultiDeviceExecutor runtime(
  //     std::move(fusion), *communicator_, executor_params_);
  // auto out = runtime.runWithInput(inputs);
  validate_with_prints(expected_outputs, out);
};

// INSTANTIATE_TEST_SUITE_P(
//     ,
//     DistributedMatmulTest,
//     testing::Bool(),
//     testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
    aten,
    DistributedMatmulTest,
    testing::Combine(
        testing::Values(true),
        testing::Values(
            DataType::Double,
            DataType::Float,
            DataType::Half,
            DataType::BFloat16)));
INSTANTIATE_TEST_SUITE_P(
    nvfuser,
    DistributedMatmulTest,
    testing::Combine(
        testing::Values(false),
        testing::Values(DataType::Half, DataType::BFloat16)));
} // namespace nvfuser
