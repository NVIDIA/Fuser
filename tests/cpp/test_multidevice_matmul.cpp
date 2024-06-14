
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>
#include <math.h>

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

class DistributedMatmulTest : public MultiDeviceTest {
 protected:
  DistributedMatmulTest() : num_devices_(communicator->size()) {}

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
};

TEST_F(DistributedMatmulTest, LayoutTN_NoComms) {
  // MmaLayout::TN A(T), B(N), C(T)
  // A and C are sharded on dimension M
  // Tests local matmul with no communication
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

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

  auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
  in0 = in0.view({Mo, Mi, K});
  out = out.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator->deviceId()), in1};
  auto expected_output = shardTensor(out, c, communicator->deviceId());

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
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

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

  auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
  in0 = in0.view({Mo, Mi, K});
  out = out.view({Mo, Mi, N});

  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator->deviceId()), in1};
  auto expected_output = shardTensor(out, c, communicator->deviceId());
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
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

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

  auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
  in0 = in0.view({Ko, Ki, M});
  in1 = in1.view({Ko, Ki, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator->deviceId()),
      shardTensor(in1, b, communicator->deviceId())};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(), outputs, inputs, {out}, __LINE__, __FILE__);
}

TEST_F(DistributedMatmulTest, LayoutNT_ReduceScatter) {
  // MmaLayout::NT matmul A(N), B(T), C(T)
  // A, B are sharded on K. C is sharded on M
  // Tests local matmul + reduce scatter
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

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

  auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
  in0 = in0.view({Ko, Ki, M});
  in1 = in1.view({Ko, Ki, N});
  out = out.view({Mo, Mi, N});
  std::vector<c10::IValue> inputs = {
      shardTensor(in0, a, communicator->deviceId()),
      shardTensor(in1, b, communicator->deviceId())};
  auto expected_output =
      shardTensor(out, c, communicator->deviceId()).view({1, Mi, N});

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
TEST_F(DistributedMatmulTest, MLP_Layer) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

  int64_t sb = 64; // sequence * batch
  int64_t h = 128;
  int64_t h4 = 4 * h;

  TensorView* x = makeContigConcreteTensor(
      {sb, h}, DataType::BFloat16); // Unsharded (sb, h)
  TensorView* w0 = makeContigConcreteTensor(
      {num_devices_, h4 / num_devices_, h},
      DataType::BFloat16); // (h4, h) -> sharded: (D, h4/D, h)
  TensorView* b0 = makeContigConcreteTensor(
      {num_devices_, h4 / num_devices_},
      DataType::BFloat16); // (h4) -> (D, h4/D)
  // TODO: initially had this as {h, num_devices_, h4/num_devices} with a
  // reorder in the compute graph, but that caused some issues. Look into what
  // went wrong.
  TensorView* w1 = makeContigConcreteTensor(
      {num_devices_, h, h4 / num_devices_},
      DataType::BFloat16); // (h, h4) -> (D, h, h4/D)
  TensorView* b1 =
      makeContigConcreteTensor({h}, DataType::BFloat16); // Unsharded (h)
  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  // Linear #1
  // Notes: Manually breaking down the linear layer into nvfuser primitives
  TensorView* linear_int0 =
      broadcast(x, {true, false, true, false}); // b, sb, b, h
  TensorView* linear_int1 =
      broadcast(w0, {false, true, false, false}); // D, b, h4/D, h
  TensorView* linear_int2 = mul(linear_int0, linear_int1); // D, sb, h4/D, h
  TensorView* linear_int3 = sum(linear_int2, {-1}); // D, sb, h4/D, r
  TensorView* linear_int4 = broadcast(b0, {false, true, false});
  TensorView* linear1 = add(linear_int3, linear_int4); // D, sb, h4/D
  linear1 = segment_set(linear1);

  // GeLU (taken from tanh_gelu composite.cpp)
  // TODO: use the tanh_gelu node when we are confident with sharding
  // propagation
  const double kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
  const double kKappa = 0.044715;
  TensorView* gelu0 = castOp(DataType::Float, linear1); // x
  TensorView* gelu1 = mul(gelu0, gelu0); // x^2
  TensorView* gelu2 = mul(gelu1, gelu0); // x^3
  TensorView* gelu3 = mul(gelu0, IrBuilder::create<Val>(0.5)); // x * 0.5
  TensorView* gelu4 =
      mul(gelu2, IrBuilder::create<Val>(kKappa)); // x^3 * 0.0447150
  TensorView* gelu5 = add(gelu0, gelu4); // x + x^3 * 0.0447150
  TensorView* gelu6 =
      mul(gelu5,
          IrBuilder::create<Val>(kBeta)); // (x + x^3 * 0.0447150) * .0.797885
  TensorView* gelu7 = tanh(gelu6); // tanh((x + x^3 * 0.0447150) * .0.797885)
  TensorView* gelu8 =
      add(gelu7,
          IrBuilder::create<Val>(
              1.0)); // 1 + tanh((x + x^3 * 0.0447150) * .0.797885)
  TensorView* gelu = mul(
      gelu3, gelu8); // x * 0.5 * (1 + tanh((x + x^3 * 0.0447150) * .0.797885))
  TensorView* gelu_ = castOp(DataType::BFloat16, gelu); // D, sb, h4/D
  gelu_ = segment_set(gelu_);

  // Linear #2
  // TODO: canonicalize inputs to push DID axis to front. Technically this
  // doesn't need a reorder since the D axis isn't materialized.
  TensorView* linear2_int0 =
      broadcast(gelu_, {false, false, true, false}); // D, sb, b, h4/D
  TensorView* linear2_int1 =
      broadcast(w1, {false, true, false, false}); // D, b, h, h4/D
  TensorView* linear2_int2 = mul(linear2_int0, linear2_int1); // D, sb, h, h4/D
  TensorView* linear2_int3 = sum(linear2_int2, {-1}); // D, sb, h, r
  TensorView* linear2_int4 = sum(linear2_int3, {0}); // Allreduce sum // r sb, h
  TensorView* linear2_int5 = broadcast(b1, {true, false}); // b, h
  TensorView* linear2 = add(linear2_int4, linear2_int5); // sb, h

  TensorView* linear2_ = castOp(DataType::Float, linear2);
  // // Dropout (taken from composite.cpp)
  const double kProb = 0.1;
  const double kScale = 1.0 / (1.0 - kProb);
  Val* philox_seed = fusion->zeroVal();
  Val* philox_offset = fusion->zeroVal();
  TensorView* rand_vals = rand_like(linear2_, philox_seed, philox_offset);
  TensorView* mask = lt(rand_vals, IrBuilder::create<Val>(1.0 - kProb));
  TensorView* apply_mask = mul(linear2_, mask);
  TensorView* dropout = mul(apply_mask, IrBuilder::create<Val>(kScale));

  fusion->addOutput(linear1);
  fusion->addOutput(gelu);
  fusion->addOutput(linear2);
  fusion->addOutput(dropout);

  auto tv_inputs = {
      x,
      b1,
      linear2_int4,
      linear2_int5,
      linear2,
      linear2_,
      rand_vals,
      mask,
      apply_mask,
      dropout};
  for (auto tv : tv_inputs) {
    tv->setDeviceMesh(mesh);
  }

  auto tvs = {w0,           b0,           w1,           linear_int0,
              linear_int1,  linear_int2,  linear_int3,  linear_int4,
              linear1,      gelu0,        gelu1,        gelu2,
              gelu3,        gelu4,        gelu5,        gelu6,
              gelu7,        gelu8,        gelu,         gelu_,
              linear2_int0, linear2_int1, linear2_int2, linear2_int3};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions()
                           .dtype(c10::ScalarType::BFloat16)
                           .device(at::kCUDA, communicator->local_rank());
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
          communicator->deviceId()),
      shardTensor(
          b0_.view({num_devices_, h4 / num_devices_}),
          b0,
          communicator->deviceId()),
      shardTensor(
          w1_.view({h, num_devices_, h4 / num_devices_}).transpose(1, 0),
          w1,
          communicator->deviceId()),
      b1_};
  at::manual_seed(0);
  auto linear1_aten =
      at::linear(x_.to(at::kFloat), w0_.to(at::kFloat), b0_.to(at::kFloat));
  auto gelu_aten = at::gelu(linear1_aten, "tanh");
  auto linear2_aten = at::linear(
      gelu_aten.to(at::kBFloat16).to(at::kFloat),
      w1_.to(at::kFloat),
      b1_.to(at::kFloat));
  auto dropout_aten = at::dropout(linear2_aten.to(at::kFloat), kProb, true);
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(
          at::transpose(
              linear1_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
          linear1,
          communicator->deviceId()),
      shardTensor(
          at::transpose(
              gelu_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
          gelu,
          communicator->deviceId()),
      linear2_aten,
      dropout_aten};
  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);

  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      expected_outputs,
      __LINE__,
      __FILE__);
}

TEST_F(DistributedMatmulTest, MLP_Layer) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(communicator->size());

  int64_t sb = 64; // sequence * batch
  int64_t h = 128;
  int64_t h4 = 4 * h;

  TensorView* x = makeContigConcreteTensor(
      {sb, h}, DataType::BFloat16); // Unsharded (sb, h)
  TensorView* w0 = makeContigConcreteTensor(
      {num_devices_, h4 / num_devices_, h},
      DataType::BFloat16); // (h4, h) -> sharded: (D, h4/D, h)
  TensorView* b0 = makeContigConcreteTensor(
      {num_devices_, h4 / num_devices_},
      DataType::BFloat16); // (h4) -> (D, h4/D)
  TensorView* w1 = makeContigConcreteTensor(
      {num_devices_, h, h4 / num_devices_},
      DataType::BFloat16); // (h, h4) -> (D, h, h4/D)
  TensorView* b1 =
      makeContigConcreteTensor({h}, DataType::BFloat16); // Unsharded (h)
  fusion->addInput(x);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);
  fusion->addInput(b1);

  // Linear #1
  // Notes: Manually breaking down the linear layer into nvfuser primitives
  TensorView* linear_int0 =
      broadcast(x, {true, false, true, false}); // b, sb, b, h
  TensorView* linear_int1 =
      broadcast(w0, {false, true, false, false}); // D, b, h4/D, h
  TensorView* linear_int2 = mul(linear_int0, linear_int1); // D, sb, h4/D, h
  TensorView* linear_int3 = sum(linear_int2, {-1}); // D, sb, h4/D, r
  TensorView* linear_int4 = broadcast(b0, {false, true, false});
  TensorView* linear1 = add(linear_int3, linear_int4); // D, sb, h4/D
  linear1 = segment_set(linear1);

  TensorView* gelu = tanh_gelu(linear1);
  TensorView* gelu_ = castOp(DataType::BFloat16, gelu); // D, sb, h4/D
  gelu_ = segment_set(gelu_);

  // Linear #2
  TensorView* linear2_int0 =
      broadcast(gelu_, {false, false, true, false}); // D, sb, b, h4/D
  TensorView* linear2_int1 =
      broadcast(w1, {false, true, false, false}); // D, b, h, h4/D
  TensorView* linear2_int2 = mul(linear2_int0, linear2_int1); // D, sb, h, h4/D
  TensorView* linear2_int3 = sum(linear2_int2, {-1}); // D, sb, h, r
  TensorView* linear2_int4 = sum(linear2_int3, {0}); // Allreduce sum // r sb, h
  TensorView* linear2_int5 = broadcast(b1, {true, false}); // b, h
  TensorView* linear2 = add(linear2_int4, linear2_int5); // sb, h

  TensorView* linear2_ = castOp(DataType::Float, linear2);
  // Note: Propagation breaks at rand_like because it creates a fresh TV.
  // Temporarily this prevents us from using dropout composite node.
  const double kProb = 0.1;
  const double kScale = 1.0 / (1.0 - kProb);
  Val* philox_seed = fusion->zeroVal();
  Val* philox_offset = fusion->zeroVal();
  TensorView* rand_vals = rand_like(linear2_, philox_seed, philox_offset);
  TensorView* mask = lt(rand_vals, IrBuilder::create<Val>(1.0 - kProb));
  TensorView* apply_mask = mul(linear2_, mask);
  TensorView* dropout = mul(apply_mask, IrBuilder::create<Val>(kScale));

  fusion->addOutput(linear1);
  fusion->addOutput(gelu);
  fusion->addOutput(linear2);
  fusion->addOutput(dropout);

  // Manually shard inputs: x, w0, b0, w1, b1
  // outputs: linear1, gelu, linear2, dropout
  // TVs where sharding changes: linear2_int4
  // (TODO) TVs where sharding propagation breaks down:
  //  linear_int0 = broadcasts where a device dim axis is broadcasted.
  //  rand_vals => rand_like creates a fresh new TV.

  // TVs replicated on each device.
  auto tv_inputs = {x, b1, linear2_int4, linear2, rand_vals, dropout};
  for (auto tv : tv_inputs) {
    tv->setDeviceMesh(mesh);
  }

  // TVs sharded on the outermost dimension.
  auto tvs = {w0, b0, w1, linear_int0, linear1, gelu};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  const auto options = at::TensorOptions()
                           .dtype(c10::ScalarType::BFloat16)
                           .device(at::kCUDA, communicator->local_rank());
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
          communicator->deviceId()),
      shardTensor(
          b0_.view({num_devices_, h4 / num_devices_}),
          b0,
          communicator->deviceId()),
      shardTensor(
          w1_.view({h, num_devices_, h4 / num_devices_}).transpose(1, 0),
          w1,
          communicator->deviceId()),
      b1_};
  at::manual_seed(0);
  auto linear1_aten =
      at::linear(x_.to(at::kFloat), w0_.to(at::kFloat), b0_.to(at::kFloat));
  auto gelu_aten = at::gelu(linear1_aten, "tanh");
  auto linear2_aten = at::linear(
      gelu_aten.to(at::kBFloat16).to(at::kFloat),
      w1_.to(at::kFloat),
      b1_.to(at::kFloat));
  auto dropout_aten = at::dropout(linear2_aten.to(at::kFloat), kProb, true);
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(
          at::transpose(
              linear1_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
          linear1,
          communicator->deviceId()),
      shardTensor(
          at::transpose(
              gelu_aten.view({sb, num_devices_, h4 / num_devices_}), 1, 0),
          gelu,
          communicator->deviceId()),
      linear2_aten,
      dropout_aten};
  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      expected_outputs,
      __LINE__,
      __FILE__);
}
} // namespace nvfuser
