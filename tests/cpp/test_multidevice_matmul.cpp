// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

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
#include <runtime/executor.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

using testing::AnyOf;
using testing::Contains;
using testing::Eq;
using testing::IsNull;

class DistributedMatmulTest : public MultiDeviceTest {
 protected:
  DistributedMatmulTest() : num_devices_(communicator_->size()) {}

  void SetUp() override {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed matmul tests require Ampere or newer";
    }
  }

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
  NVFUSER_TEST_CUDA_ARCH_RANGE_GUARD(8, 0, 9, 0);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(num_devices_);
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
  KernelArgumentHolder args = {shardTensor(in0, a), in1};
  auto expected_output = shardTensor(out, c);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);
  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {expected_output},
      __LINE__,
      __FILE__);

  const FusionKernelRuntime* kernel_runtime =
      executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      kernel_runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::Matmul)).Times(1));
}

TEST_F(DistributedMatmulTest, Matmul_LayoutTN_NoComms) {
  // MmaLayout::TN A(T), B(N), C(T)
  // A and C are sharded on dimension M
  // Tests local matmul with no communication
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(num_devices_);

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
  KernelArgumentHolder args = {shardTensor(in0, a), in1};
  auto expected_output = shardTensor(out, c);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);

  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {expected_output},
      __LINE__,
      __FILE__);

  const FusionKernelRuntime* kernel_runtime =
      executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      kernel_runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::ExprEval)).Times(2));
}

TEST_F(DistributedMatmulTest, Matmul_LayoutTN_Allgather) {
  // MmaLayout::TN matmul A(T), B(N), C(T)
  // A is sharded on dimension M
  // Tests local matmul + allgather
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(num_devices_);

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

  KernelArgumentHolder args = {shardTensor(in0, a), in1};
  auto expected_output = shardTensor(out, c);
  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);

  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {expected_output},
      __LINE__,
      __FILE__);

  const FusionKernelRuntime* kernel_runtime =
      executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      kernel_runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::ExprEval)).Times(3));
}

TEST_F(DistributedMatmulTest, Matmul_LayoutNT_AllReduce) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // Sharding: A, B are sharded along K. C is replicated.
  // Tests local matmul + allreduce
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(num_devices_);

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
  KernelArgumentHolder args = {shardTensor(in0, a), shardTensor(in1, b)};

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);

  testValidate(
      executor_cache.fusion(), outputs, args, {out}, __LINE__, __FILE__);

  const FusionKernelRuntime* kernel_runtime =
      executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      kernel_runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::ExprEval)).Times(2));
}

TEST_F(DistributedMatmulTest, Matmul_LayoutNT_ReduceScatter) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // A, B are sharded on K. C is sharded on M
  // Tests local matmul + reduce scatter
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(num_devices_);

  int M = 128, N = 64, K = 128;
  int Ko = num_devices_, Ki = K / Ko;
  int Mo = num_devices_, Mi = M / Mo;
  std::vector<int> a_shape = {Ko, Ki, M};
  std::vector<int> b_shape = {Ko, Ki, N};

  TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
  TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
  TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
  TensorView* c0 = matmul(a_t, b); // (Ko,M,N,r)
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
  KernelArgumentHolder args = {shardTensor(in0, a), shardTensor(in1, b)};
  auto expected_output = shardTensor(out, c).view({1, Mi, N});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs(args);
  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {expected_output},
      __LINE__,
      __FILE__);

  const FusionKernelRuntime* kernel_runtime =
      executor_cache.getMostRecentKernelRuntime();
  EXPECT_THAT(
      kernel_runtime->fusionSegments()->groups(),
      Contains(HeuristicIs(SchedulerType::ExprEval)));
}

// Reproduces #2721.
TEST_F(DistributedMatmulTest, PresegPreservesSharding) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());

  TensorView* x = makeContigTensor(2);
  TensorView* w = makeContigTensor(3);
  fusion->addInput(x);
  fusion->addInput(w);

  TensorView* w_t = transpose(w, 1, 2);
  TensorView* mm = matmul(x, w_t);
  TensorView* mm_t = transpose(mm, 1, 2);
  fusion->addOutput(mm_t);

  for (auto tv : {x}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w, w_t, mm, mm_t}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  auto x_tensor = at::randn({12, 48}, tensor_options);
  auto w_tensor = at::randn({mesh.size(), 36, 48}, tensor_options);
  auto sharded_w_tensor = shardTensor(w_tensor, w);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {x_tensor, sharded_w_tensor};
  auto outputs = executor_cache.runFusionWithInputs(args);

  at::Tensor expected_mm_t_tensor =
      atMatmul(x_tensor, w_tensor.view({mesh.size() * 36, 48}), MmaLayout::TN)
          .transpose(0, 1)
          .view({mesh.size(), 36, 12});
  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {shardTensor(expected_mm_t_tensor, mm_t)},
      __LINE__,
      __FILE__);
}

TEST_F(DistributedMatmulTest, AnnotateWeightOnly) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigTensor(2);
  TensorView* w = makeContigTensor(3);
  TensorView* y = matmul(x, w);
  fusion->addInput(x);
  fusion->addInput(w);
  fusion->addOutput(y);

  auto mesh = DeviceMesh::createForNumDevices(communicator_->size());
  x->setDeviceMesh(mesh);
  w->setDeviceMesh(mesh);
  w->axis(0)->parallelize(ParallelType::DIDx);

  // x is of shape [2, 3] and replicated.
  // w is of shape [3, D*5] and column-wise sharded.
  // y is expected to have shape [2, D*5] and to be also column-wise sharded.
  constexpr int64_t kLowerBound = 0;
  constexpr int64_t kUpperBound = 10;
  auto x_tensor = at::randint(kLowerBound, kUpperBound, {2, 3}, tensor_options);
  auto w_tensor = at::randint(
      kLowerBound, kUpperBound, {mesh.size(), 3, 5}, tensor_options);
  auto sharded_w_tensor = shardTensor(w_tensor, w);

  FusionExecutorCache executor_cache(std::move(fusion));
  KernelArgumentHolder args = {x_tensor, sharded_w_tensor};
  auto outputs = executor_cache.runFusionWithInputs(args);

  at::Tensor expected_y_tensor = at::matmul(x_tensor, w_tensor);
  testValidate(
      executor_cache.fusion(),
      outputs,
      args,
      {shardTensor(expected_y_tensor, 0, mesh)},
      __LINE__,
      __FILE__);
}

// linear([M, K], [N, K]) -> [M, N]
//
// K, the row dimension of the weight, is sharded on DIDx. Note that LinearOp's
// weight is of shape [column, row]. This LinearOp is decomposed into a local
// LinearOp followed by an Allreduce.
TEST_F(DistributedMatmulTest, RowParallelLinear) {
  const auto d = communicator_->size();
  constexpr int64_t e = 12;
  if (e % d != 0) {
    GTEST_SKIP() << "The test requires e (" << e << ") to be divisible by d ("
                 << d << ").";
  }

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* x = makeContigConcreteTensor({-1, -1, e});
  TensorView* w = makeContigConcreteTensor({e, e});
  TensorView* y = linear(x, w);
  fusion->addInput(x);
  fusion->addInput(w);
  fusion->addOutput(y);

  x->split(-1, d, /*inner_split=*/false);
  x->axis(-2)->parallelize(ParallelType::DIDx);

  w->split(-1, d, /*inner_split=*/false);
  w->axis(-2)->parallelize(ParallelType::DIDx);

  y->split(-1, d, /*inner_split=*/false);
  TensorView* local_y = y->rFactor({-1});

  local_y->axis(-2)->parallelize(ParallelType::DIDx);

  auto mesh = DeviceMesh::createForNumDevices(d);
  for (auto tv : {x, w, y, local_y}) {
    tv->setDeviceMesh(mesh);
    tv->setAllocationDomain(tv->getLoopDomain(), true);
  }

  FusionExecutorCache executor_cache(std::move(fusion));
  FusionKernelRuntime* previous_runtime = nullptr;
  constexpr int64_t b = 1;
  for (int64_t s : {4, 8, 16}) {
    // Use randint instead of randn to avoid floating point accumulation errors.
    auto x_tensor = at::randint(/*high=*/5, {b, s, e}, tensor_options);
    auto w_tensor = at::randint(/*high=*/5, {e, e}, tensor_options);
    auto sharded_x = shardTensor(x_tensor, x);
    auto sharded_w = shardTensor(w_tensor, w);

    KernelArgumentHolder args = {sharded_x, sharded_w};
    auto out_tensors = executor_cache.runFusionWithInputs(args);

    at::Tensor expected_y_tensor = at::linear(x_tensor, w_tensor);
    testValidate(
        executor_cache.fusion(),
        out_tensors,
        args,
        {expected_y_tensor},
        __LINE__,
        __FILE__);

    FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
    EXPECT_THAT(previous_runtime, AnyOf(IsNull(), Eq(runtime)))
        << "The same runtime should be reused for different sequence lengths.";
    previous_runtime = runtime;
  }
}

} // namespace nvfuser
