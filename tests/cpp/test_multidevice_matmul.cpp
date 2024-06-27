// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <executor.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

namespace nvfuser {

class DistributedMatmulTest : public MultiDeviceTest,
                              public testing::WithParamInterface<bool> {
 protected:
  DistributedMatmulTest() : num_devices_(communicator_->size()) {}

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

TEST_P(DistributedMatmulTest, MLP_Layer) {
  bool use_aten_matmul = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  int64_t sb = 64; // sequence * batch
  int64_t h = 128;
  int64_t h4 = 4 * h;

  // TODO: error with dynamic shape
  // C++ exception with description "ext_opt.hasValue() INTERNAL ASSERT FAILED
  // at "csrc/dynamic_transform.cpp":276, Could not evaluate dynamic extent: i3
  // Exception raised from DynamicTransformConcretizationInfo at
  // csrc/dynamic_transform.cpp:276
//   TensorView* x = makeContigConcreteTensor({sb, h}, DataType::BFloat16);
//   TensorView* w0 = makeContigConcreteTensor(
//       {num_devices_, h4 / num_devices_, h}, DataType::BFloat16);
//   TensorView* b0 = makeContigConcreteTensor(
//       {num_devices_, h4 / num_devices_}, DataType::BFloat16);
//   TensorView* w1 = makeContigConcreteTensor(
//       {num_devices_, h, h4 / num_devices_}, DataType::BFloat16);
//   TensorView* b1 = makeContigConcreteTensor({h}, DataType::BFloat16);

  TensorView* x = makeContigTensor(2, DataType::BFloat16); // Unsharded (sb, h)
  TensorView* w0 = makeContigTensor(3, DataType::BFloat16); // (h4, h) -> sharded: (D, h4/D, h)
  TensorView* b0 = makeContigTensor(2, DataType::BFloat16); // (h4) -> (D, h4/D)
  TensorView* w1 = makeContigTensor(3, DataType::BFloat16); // (h, h4) -> (D, h, h4/D)
  TensorView* b1 = makeContigTensor(1, DataType::BFloat16); // Unsharded (h)

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
  TensorView* gelu_ = castOp(DataType::BFloat16, gelu);

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
  // Note: Propagation breaks at rand_like because it creates a fresh TV.
  // Temporarily this prevents us from using dropout composite node.
  TensorView* linear2_ = castOp(DataType::Float, linear2);
  constexpr double kProb = 0.1;
  constexpr double kScale = 1.0 / (1.0 - kProb);
  Val* philox_seed = fusion->zeroVal();
  Val* philox_offset = fusion->zeroVal();
  TensorView* rand_vals = rand_like(linear2_, philox_seed, philox_offset);
  TensorView* mask = lt(rand_vals, IrBuilder::create<Val>(1.0 - kProb));
  TensorView* apply_mask = mul(linear2_, mask);
  TensorView* dropout = mul(apply_mask, IrBuilder::create<Val>(kScale));

  fusion->addOutput(linear1);
  fusion->addOutput(gelu);
  fusion->addOutput(linear2);
  // fusion->addOutput(dropout);

  // Manually shard inputs: x, w0, b0, w1, b1
  // outputs: linear1, gelu, linear2, dropout
  // TVs where sharding changes: matmul2
  // (TODO) TVs where sharding propagation breaks down:
  // linear_int0 = broadcasts where a device dim axis is broadcasted.
  // rand_vals => rand_like creates a fresh new TV.

  // TVs replicated on each device.
  auto tv_inputs = {x, b1, matmul2, linear2, rand_vals, dropout};
  for (auto tv : tv_inputs) {
    tv->setDeviceMesh(mesh);
  }

  // TVs sharded on the outermost dimension.
  auto tvs = {w0, b0, w1, linear1, gelu, gelu_};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  std::cout << x->toString() << std::endl;
  std::cout << w0->toString() << std::endl;

  const auto options = at::TensorOptions()
                           .dtype(c10::ScalarType::BFloat16)
                           .device(at::kCUDA, communicator_->local_rank());
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
      at::linear(x_.to(at::kDouble), w0_.to(at::kDouble), b0_.to(at::kDouble));
  auto gelu_aten = at::gelu(linear1_aten.to(at::kFloat), "tanh");
  auto linear2_aten = at::linear(
      gelu_aten.to(at::kBFloat16).to(at::kDouble),
      w1_.to(at::kDouble),
      b1_.to(at::kDouble));
  auto dropout_aten = at::dropout(linear2_aten.to(at::kFloat), kProb, true);
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
  DynamicTransform::concretizeFusion(fusion.get(), inputs);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  // Bump up the tolerance - the second matmul carries
  // the numerical error from the prior matmul
  auto tolerance_overwrite = ValidationConstants();
  std::array<std::array<double, 2>, 20> relaxed_sum_tol;
  for (auto& arr : relaxed_sum_tol) {
    arr = {128, 3.0};
  }
  tolerance_overwrite.sum_tolerances_float = relaxed_sum_tol;

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      expected_outputs,
      __LINE__,
      __FILE__,
      "",
      LaunchParams(),
      tolerance_overwrite);
}

TEST_P(DistributedMatmulTest, Multiheaded_Attention) {
  bool use_aten_matmul = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  // input would be 3D (sequence, batch, embedding)
  // to simplify we make it 2D immediately (sequence * batch, embedding)
  int s = 16;
  int b = 4;
  int sb = s * b;
  int e = 64;

  TensorView* x = makeContigConcreteTensor({sb, e}, DataType::BFloat16);
  TensorView* w0 = makeContigConcreteTensor(
      {num_devices_, e, ??/num_devices_}, DataType::BFloat16);
  TensorView* b0 = makeContigConcreteTensor(
      {num_devices_, ?? / num_devices_}, DataType::BFloat16);

  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);

  // self attention (linear, sdpa), linear (sharded compute), dropout
  // linear #1 is sharded along the columns (heads)
  // linear #2 is sharded along the rows + allreduce to reform x
  TensorView* qkv_ = matmul(x, proj_w);
  TensorView* proj_bias_bcast = broadcast(proj, bias);
  TensorView* qkv = add(qkv_, proj_bias_bcast);
  // reshape from (sequence*batch, embedding) to (batch, sequence, embedding)
  // note: this should be (sequence, batch, embedding), but linear layer requires 
  // 2D input for now. 
  TensorView* q_ = select(qkv, {0, 0, 0}, {16, 128, 1600});
  TensorView* k_ = select(qkv, {0, 0, 1600}, {16, 128, 3200});
  TensorView* v_ = select(qkv, {0, 0, 3200}, {16, 128, 4800});
  TensorView* q = reshape(q_, {16, 128, 35, 64});
  q = transpose(q, 2, 1);
  TensorView* k = reshape(k_, {16, 128, 35, 64});
  k = transpose(k, 2, 1);
  TensorView* v = reshape(q_, {16, 128, 35, 64});
  v = transpose(v, 2, 1);

  constexpr double kProb = 0.1;
  constexpr double kScale = 1.0 / (1.0 - kProb);
  SdpfaFwdResult sdpa = sdpfa_fwd(q, k, v, 
    IrBuilder::create<Val>(kProb), 
    IrBuilder::create<Val>(true), 
    IrBuilder::create<Val>(kScale));

 fusion->addOutput(sdpa.output);

 const auto options = at::TensorOptions()
                           .dtype(c10::ScalarType::BFloat16)
                           .device(at::kCUDA, communicator_->local_rank());
  auto x_ = at::randn({sb, h}, options);
  auto w0_ = at::randn({h4, h}, options);
  auto b0_ = at::randn({h4}, options);

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(
          w0_.view({num_devices_, h4 / num_devices_, h}),
          w0,
          communicator_->deviceId()),
      shardTensor(
          b0_.view({num_devices_, h4 / num_devices_}),
          b0,
          communicator_->deviceId())};

  auto expected_output =
      shardTensor(out, sdpa_, communicator_->deviceId()).view({1, Mi, N});

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
};

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedMatmulTest,
    testing::Bool(),
    testing::PrintToStringParamName());
} // namespace nvfuser
