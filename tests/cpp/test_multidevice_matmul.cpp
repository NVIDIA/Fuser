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
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

namespace nvfuser {

class DistributedMatmulTest : public MultiDeviceTest {
 protected:
  DistributedMatmulTest() : optimization_guard_(false) {
    DisableOptionsGuard::getCurOptions().set(DisableOption::MatmulExprEval);
    num_devices_ = communicator->size();
  }

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
  int num_devices_;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> getInputsAndReferenceOutputs(
      MmaLayout layout,
      int M,
      int N,
      int K) {
    int local_rank = communicator->local_rank();
    c10::ScalarType type = c10::ScalarType::Half;
    auto a = matmulAtInput2D(
        layout, TensorMatmulPos::A, type, M, N, K, 0, local_rank);
    auto b = matmulAtInput2D(
        layout, TensorMatmulPos::B, type, M, N, K, 0, local_rank);
    auto c =
        atMatmul(a.to(at::kDouble), b.to(at::kDouble), layout).to(at::kFloat);
    return std::make_tuple(a, b, c);
  }

 private:
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      optimization_guard_;
  DisableOptionsGuard option_guard_;
};

TEST_F(DistributedMatmulTest, LayoutTN_NoComms) {
  // MmaLayout::TN A(T), B(N), C(T)
  // A and C are sharded on dimension M
  // Tests local matmul with no communication
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh = createDeviceMesh(communicator->size());

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

  auto [a_, b_, c_] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
  a_ = a_.view({Mo, Mi, K});
  c_ = c_.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(a_, a, communicator->deviceId()), b_};
  auto expected_output = shardTensor(c_, c, communicator->deviceId());

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);
}

TEST_F(DistributedMatmulTest, LayoutTN_Allgather) {
  // MmaLayout::TN matmul A(T), B(N), C(T)
  // A is sharded on dimension M
  // Tests local matmul + allgather
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh = createDeviceMesh(communicator->size());

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
  TensorView* c0 = sum(ab, {-1}); // (Mo,Mi,N,r)
  TensorView* c = set(c0);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding M dimension
  auto all_sharded_tvs = {a, a_b, b_b, ab, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);

  auto [a_, b_, c_] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
  a_ = a_.view({Mo, Mi, K});
  c_ = c_.view({Mo, Mi, N});

  std::vector<c10::IValue> inputs = {
      shardTensor(a_, a, communicator->deviceId()), b_};
  auto expected_output = shardTensor(c_, c, communicator->deviceId());
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);
}

TEST_F(DistributedMatmulTest, LayoutNT_AllReduce) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // Sharding: A, B are sharded along K. C is replicated.
  // Tests local matmul + allreduce
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh = createDeviceMesh(communicator->size());

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  // Transpose into TN layout, keep Ko (device axis) as the outermost.
  TensorView* a_t = transpose(a, 1, 2); // (Ko,M,Ki)
  TensorView* b_t = transpose(b, 1, 2); // (Ko,N,Ki)
  TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
  TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
  TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
  TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
  TensorView* c = sum(c0, {0}); // (r,M,N)

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Parallelize K on all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  c->setDeviceMesh(mesh);

  auto [a_, b_, c_] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
  a_ = a_.view({Ko, Ki, M});
  b_ = b_.view({Ko, Ki, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(a_, a, communicator->deviceId()),
      shardTensor(b_, b, communicator->deviceId())};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(), outputs, inputs, {c_}, __LINE__, __FILE__);
}

TEST_F(DistributedMatmulTest, LayoutNT_ReduceScatter) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // A, B are sharded on K. C is sharded on M
  // Tests local matmul + reduce scatter
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  DeviceMesh mesh = createDeviceMesh(communicator->size());

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  int Mo = num_devices_, Mi = M / Mo;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
  TensorView* b_t = transpose(b, 1, 2); // (Ko, N, Ki)
  TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
  TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
  TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
  TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
  c0 = segment_set(c0);
  std::vector<int64_t> orig_size = {K, M, N};
  std::vector<int64_t> new_size = {K, Mo, Mi, N};
  TensorView* c1 = reshape(c0, orig_size, new_size);
  TensorView* c = sum(c1, {0});

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  // Sharding K dimension of all inputs and intermediates.
  auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0, c1};
  for (auto tv : all_sharded_tvs) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }
  // Sharding M on output
  c->setDeviceMesh(mesh);
  c->axis(1)->parallelize(ParallelType::DIDx);

  auto [a_, b_, c_] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
  a_ = a_.view({Ko, Ki, M});
  b_ = b_.view({Ko, Ki, N});
  c_ = c_.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(a_, a, communicator->deviceId()),
      shardTensor(b_, b, communicator->deviceId())};
  auto expected_output =
      shardTensor(c_, c, communicator->deviceId()).view({1, Mi, N});

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      {expected_output},
      __LINE__,
      __FILE__);
}
} // namespace nvfuser
