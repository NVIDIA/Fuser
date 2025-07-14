// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <device_lower/pass/magic_zero.h>
#include <disjoint_set.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <grouped_reduction.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_ir.h>
#include <kernel_ir_dispatch.h>
#include <logical_domain_map.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/executor_params.h>
#include <runtime/fusion_executor_cache.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>
#include <transform_replay.h>
#include <transform_rfactor.h>
#include <utils.h>

#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, FusionGlobalIntermediate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Val>(0.0), tv0);
  fusion.addInput(tv0);
  fusion.addOutput(tv1);
  // tv1[I0, R1] = tv0[I0, I1]

  // Interface should just be a direct split with a Parallel type. We can
  // include the parallelize call if we do this.
  tv1->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
  // tv1[I0, R1o, R1i{BIDx}] = tv0[I0, I1]

  TensorView* tv2 = tv1->rFactor({2});
  tv2->setMemoryType(MemoryType::Global);
  // tv2[I0, R1oo, Ir1i{BIDx}] = tv0[I0, I1]
  // tv1[I0,        R1i{BIDx}] = tv2[I0, R1oo, Ir1i{BIDx}]

  tv0->computeAt(tv1, 1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  constexpr int numel_x = 65000, numel_y = 1024;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  // How many threads to use for the block reduction
  constexpr int runtime_threadIdx_dim = 128;

  auto lparams = LaunchParams(-1, -1, -1, runtime_threadIdx_dim, -1, -1);

  KernelExecutor ke;
  ke.compile(&fusion, {input}, lparams);
  auto cg_outputs = ke.run({input}, {}, lparams);

  auto aten_output = input.to(at::kDouble).sum({1});
  testValidate(
      &fusion,
      cg_outputs,
      {input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      lparams);
}

TEST_F(NVFuserTest, FusionGlobalIntermediateDefaultSchedule_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);
  TensorView* tv2 = makeSymbolicTensor(2);
  TensorView* tv3 = makeSymbolicTensor(2);
  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);
  fusion.addInput(tv3);
  fusion.addOutput(tv6);
  // t6 = ((t1 + (t2 - t3)) - t0)

  tv4->setMemoryType(MemoryType::Global);
  tv5->setMemoryType(MemoryType::Global);
  tv6->setMemoryType(MemoryType::Global);

  constexpr int M = 32, N = 810;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t1 = at::randn({M, N}, options);
  at::Tensor t2 = at::randn({M, N}, options);
  at::Tensor t3 = at::randn({M, N}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2, t3});
  auto cg_outputs = ke.run({t0, t1, t2, t3});

  testValidate(&fusion, cg_outputs, {t0, t1, t2, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionConstCheck_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto one = IrBuilder::create<Val>(1L);
  NVF_CHECK(one->isConstScalar());

  auto one_x2 = mul(one, one);
  NVF_CHECK(one_x2->isConstScalar());

  auto one_x3 = mul(one_x2, one);
  NVF_CHECK(one_x3->isConstScalar());

  auto one_x4 = mul(one_x3, one);
  NVF_CHECK(one_x4->isConstScalar());
}

TEST_F(NVFuserTest, FusionUnrollWithAlloc_CUDA) {
  const std::vector<int64_t> tensor_dims_in = {128, 128};
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(tensor_dims_in.size());
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(0.0));
  TensorView* tv2 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Val>(0.0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(tensor_dims_in, options);
  at::Tensor cg_output = at::empty({tensor_dims_in[0]}, options);

  // Schedule
  tv2->split(1, 32);
  tv2->split(1, 4); // unroll

  auto tv2_rf = tv2->rFactor({-3, -2});

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv2_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv2_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv2_rf->axis(-2)->parallelize(ParallelType::Unroll);

  tv1->computeAt(tv2_rf, -1);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  auto aten_output = (input + 0).to(at::kDouble).sum(1);

  testValidate(&fusion, cg_outputs, {input}, {aten_output}, __LINE__, __FILE__);
}

// Test isZeroInt
TEST_F(NVFuserTest, FusionIsZeroInt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Val* x = IrBuilder::create<Val>(0L);
  Val* y = IrBuilder::create<Val>(1L);
  Val* z = mul(x, y);
  NVF_CHECK(x->isZeroInt());
  NVF_CHECK(!y->isZeroInt());
  NVF_CHECK(!z->isZeroInt());
}

// Test isOneInt
TEST_F(NVFuserTest, FusionIsOneInt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Val* x = IrBuilder::create<Val>(1L);
  Val* y = IrBuilder::create<Val>(1L);
  Val* z = mul(x, y);
  NVF_CHECK(x->isOneInt());
  NVF_CHECK(y->isOneInt());
  NVF_CHECK(!z->isOneInt());
}

// This is to verify no cycle of computeAt is created. A more complex
// variation of this pattern appears in one of the Python tests
// (test_random_topo).
TEST_F(NVFuserTest, FusionComputeAtNonterminatingOutput_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Common intermediate tensor
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  // tv1 -> tv2
  auto tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  // tv1 -> tv3 -> tv4
  auto tv3 = add(tv1, IrBuilder::create<Val>(3.0));
  auto tv4 = add(tv3, IrBuilder::create<Val>(4.0));

  // NOTE: This should no longer occur as of PR #201.
  // The order of adding outputs matters. If tv3 is added before tv4,
  // it should be fine. However, if tv4 is added before tv3, there
  // will be a cycle of tv3->tv4 and tv4->tv3. tv3->tv4 is created
  // first, and then tv4->tv3 is created at the final phase of
  // computeAt (ComputeAt::setupOutputs).
  fusion.addOutput(tv2);
  fusion.addOutput(tv4);
  fusion.addOutput(tv3);

  tv0->computeAt(tv2, -1);

  NVF_CHECK(tv3->hasComputeAt());
  NVF_CHECK(!tv4->hasComputeAt());

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(100, options);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = t1 + 3;
  auto t4 = t3 + 4;

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = add(tv1, IrBuilder::create<Val>(4.0));

  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);

  tv1->computeAt(tv3, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 10}, options);

  auto cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, cg_outputs);
  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));

  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = add(tv3, IrBuilder::create<Val>(4.0));

  TensorView* tv5 = add(tv1, tv3);

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv1->computeAt(tv5, -1);
  tv3->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 10}, options);

  auto cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, cg_outputs);

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder3_CUDA) {
  for (const auto i : arange(2)) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    TensorView* tv0 = makeSymbolicTensor(1);
    fusion.addInput(tv0);

    TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
    TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));

    TensorView* tv3 = add(tv0, IrBuilder::create<Val>(3.0));
    TensorView* tv4 = add(tv3, IrBuilder::create<Val>(4.0));

    TensorView* tv5 = add(tv1, tv3);

    fusion.addOutput(tv2);
    fusion.addOutput(tv4);
    fusion.addOutput(tv5);

    const int tile = 32;

    tv1->split(-1, tile);
    tv2->split(-1, tile);
    tv3->split(-1, tile);
    tv4->split(-1, tile);
    tv5->split(-1, tile);

    auto compute_at_outer = tv1;
    auto compute_at_inner = tv3;
    if (i == 1) {
      std::swap(compute_at_inner, compute_at_outer);
    }

    compute_at_outer->computeAt(tv5, -2);
    compute_at_inner->computeAt(tv5, -1);

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor aten_input = at::randn({100}, options);

    auto cg_outputs = {
        at::empty_like(aten_input, options),
        at::empty_like(aten_input, options),
        at::empty_like(aten_input, options)};

    KernelExecutor ke;
    ke.compile(&fusion, {aten_input});
    ke.run({aten_input}, cg_outputs);

    testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
  }
}

TEST_F(NVFuserTest, FusionTraversalOrder4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // First tree
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Val>(3.0));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  // Second tree
  TensorView* tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);
  TensorView* tv5 = add(tv4, IrBuilder::create<Val>(5.0));
  TensorView* tv6 = add(tv5, IrBuilder::create<Val>(6.0));
  TensorView* tv7 = add(tv5, IrBuilder::create<Val>(7.0));
  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  tv1->computeAt(tv2, -1);
  tv5->computeAt(tv6, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({100}, options);
  at::Tensor t4 = at::rand_like(t0, options);

  auto cg_outputs = {
      at::empty_like(t0, options),
      at::empty_like(t0, options),
      at::empty_like(t0, options),
      at::empty_like(t0, options)};

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t4});
  ke.run({t0, t4}, cg_outputs);

  testValidate(&fusion, cg_outputs, {t0, t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = add(tv3, IrBuilder::create<Val>(4.0));
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv1);
  fusion.addOutput(tv3);
  fusion.addOutput(tv5);

  tv2->computeAt(tv5, -1);
  tv4->computeAt(tv5, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);
  auto cg_outputs = {
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options),
      at::empty_like(aten_input, options)};

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, cg_outputs);

  auto t1 = aten_input + 1;
  auto t2 = t1 + 2;
  auto t3 = aten_input + 3;
  auto t4 = t3 + 4;
  auto t5 = t2 + t4;

  std::vector<at::Tensor> aten_outputs = {t1, t3, t5};

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv1, tv2);
  TensorView* tv4 = add(tv3, IrBuilder::create<Val>(4.0));

  fusion.addOutput(tv4);

  tv1->split(0, 32);
  tv2->split(0, 32);
  tv3->split(0, 32);
  tv4->split(0, 32);

  tv3->computeAt(tv4, -2);
  tv1->computeAt(tv3, -1);
  tv2->computeAt(tv3, -2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);

  at::Tensor cg_output = at::empty_like(aten_input, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, {cg_output});

  testValidate(&fusion, {cg_output}, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTraversalOrder7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = add(tv3, IrBuilder::create<Val>(4.0));
  TensorView* tv5 = add(tv2, tv4);

  fusion.addOutput(tv5);

  TensorView* tvs[] = {tv1, tv2, tv3, tv4, tv5};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->split(0, 4);
    tv->split(0, 8);
  }

  // computeAt into inner loop nests
  tv1->computeAt(tv2, -1);
  tv3->computeAt(tv4, -2);

  tv2->computeAt(tv5, -4);
  tv4->computeAt(tv5, -3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100}, options);

  at::Tensor cg_output = at::empty_like(aten_input, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, {cg_output});

  testValidate(&fusion, {cg_output}, {aten_input}, __LINE__, __FILE__);
}

// Test predication of grid reduction
TEST_F(NVFuserTest, FusionThreadPredicate_CUDA) {
  const int gdimx = 4;
  const int bdimx = 128;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {1}, IrBuilder::create<Val>(0.0), tv0);
  TensorView* tv2 = unaryOp(UnaryOpType::Neg, tv1);
  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(2.0));

  fusion.addOutput(tv3);
  fusion.addOutput(tv2);

  tv1->split(1, bdimx);
  tv1->split(1, gdimx);
  tv3->split(1, bdimx);
  tv3->split(1, gdimx);

  TensorView* tv1_rf = tv1->rFactor({1});

  tv1->computeAt(tv2, -1);

  tv1->axis(0)->parallelize(ParallelType::BIDy);
  tv1_rf->axis(0)->parallelize(ParallelType::BIDy);
  tv2->axis(0)->parallelize(ParallelType::BIDy);
  tv1->axis(-2)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-2)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(2)->parallelize(ParallelType::BIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDy);

  int numel_x = 100;
  int numel_y = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);

  auto t2 = -aten_input.to(at::kDouble).sum({1});
  auto t3 = aten_input + 2.0;

  std::vector<at::Tensor> aten_outputs = {t3, t2};

  auto cg_outputs = {
      at::empty_like(aten_input, options), at::empty({numel_x}, options)};

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, cg_outputs);

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLSTMCell_CUDA) {
  const int hidden_features = 512;
  const int batch_size = 64;

  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tvs[16];
  for (const auto i : arange(16)) {
    tvs[i] = makeSymbolicTensor(2);
    fusion.addInput(tvs[i]);
  }

  auto ingate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[0], tvs[1]), tvs[2]), tvs[3]));

  auto forgetgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[4], tvs[5]), tvs[6]), tvs[7]));

  auto cellgate = unaryOp(
      UnaryOpType::Tanh, add(add(add(tvs[8], tvs[9]), tvs[10]), tvs[11]));

  auto outgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[12], tvs[13]), tvs[14]), tvs[15]));

  auto cx = makeContigTensor(2);
  fusion.addInput(cx);

  auto cy = add(mul(forgetgate, cx), mul(ingate, cellgate));

  auto hy = mul(outgate, unaryOp(UnaryOpType::Tanh, cy));

  fusion.addOutput(cy);
  fusion.addOutput(hy);

  KernelArgumentHolder inputs;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor large_tensor0 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor1 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor2 =
      at::randn({batch_size, hidden_features * 4}, options);
  at::Tensor large_tensor3 =
      at::randn({batch_size, hidden_features * 4}, options);

  auto chunked0 = large_tensor0.chunk(4, 1);
  auto chunked1 = large_tensor1.chunk(4, 1);
  auto chunked2 = large_tensor2.chunk(4, 1);
  auto chunked3 = large_tensor3.chunk(4, 1);

  inputs.push(chunked0);
  inputs.push(chunked1);
  inputs.push(chunked2);
  inputs.push(chunked3);

  auto at_cx = at::randn({batch_size, hidden_features}, options);
  inputs.push(at_cx);

  auto cg_outputs = scheduleAndRun(&fusion, SchedulerType::PointWise, inputs);
  testValidate(&fusion, cg_outputs.outputs, inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionHalf_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = sum(tv2, {2});
  auto tv4 = castOp(DataType::Half, tv3);

  fusion.addOutput(tv4);

  const auto options =
      at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({8, 8, 16}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::Reduction, {aten_input});
  auto aten_output = aten_input.add(1.0).to(at::kDouble).sum({2});

  testValidate(
      &fusion,
      cg_results.outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionReduceSingle_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({100, 1});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({100, 1}, options);

  // Grab only tensor views, though there shouldn't be any other type
  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  // no broadcasting needed, omitting the last optional argument;
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 = reductionOp(
      BinaryOpType::Add, {red_dim, 2}, IrBuilder::create<Val>(0.0), tv0);
  fusion.addOutput(tv1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::Reduction, {aten_input});
  auto aten_output = aten_input.to(at::kDouble).sum({red_dim, 2});

  testValidate(
      &fusion,
      cg_results.outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast2_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {2}, IrBuilder::create<Val>(0.0), tv0);

  TensorView* tv2 = reductionOp(
      BinaryOpType::Add, {red_dim}, IrBuilder::create<Val>(0.0), tv1);
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::Reduction, {aten_input});
  auto aten_output = aten_input.to(at::kDouble).sum({1, 2});

  testValidate(
      &fusion,
      cg_results.outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionReduceImplicitBroadcast3_CUDA) {
  constexpr int bid_x = 80;
  constexpr int tid_x = 4096;
  constexpr int red_dim = 1;

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({bid_x, tid_x, 1});
  fusion.addInput(tv0);

  TensorView* tv1 = sum(tv0, {red_dim});
  TensorView* tv2 = squeeze(tv1, std::vector<bool>{false, true});
  fusion.addOutput(tv2);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({bid_x, tid_x, 1}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::Reduction, {aten_input});
  auto aten_output = aten_input.to(at::kDouble).sum({2, 1});

  testValidate(
      &fusion,
      cg_results.outputs,
      {aten_input},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionTrivialReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({10, 20, 1});
  fusion.addInput(tv0);
  TensorView* tv1 =
      reductionOp(BinaryOpType::Add, {2}, IrBuilder::create<Val>(0.0), tv0);
  fusion.addOutput(tv1);

  NVF_CHECK(
      !ir_utils::hasOpsOfType<ReductionOp>(&fusion),
      "Trivial reduction not converted to squeeze.");

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 20, 1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 1, x = 1, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeConcreteTensor({w, x, y, z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = sum(tv1, {0});
  auto tv3 = sum(tv2, {0});
  auto tv4 = add(tv3, tv0);

  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto cg_outputs = scheduleAndRun(&fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(&fusion, cg_outputs.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialReduction3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int v = 1, w = 1, x = 1, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeConcreteTensor({v, w, x, y, z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = sum(tv1, {0, 1, 2});
  auto tv3 = add(tv2, tv0);

  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({y, z}, options);
  at::Tensor t1 = at::randn({v, w, x, y, z}, options);

  auto cg_outputs = scheduleAndRun(&fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(&fusion, cg_outputs.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionInputsIdLookup_CUDA) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 8, 8}, options);
  at::Tensor t1 = at::randn({8, 8}, options);
  at::Tensor t2 = at::randn({6, 4}, options);

  // create a cache with max size 2;
  nvfuser::InputsIdLookup inputs_id_lookup(2);

  // testing basic function, same encoding for identical inputs
  auto id_0 = inputs_id_lookup.lookupId({t0, t1, 5.0});
  auto id_0_lookup = inputs_id_lookup.lookupId({t0, t1, 2.5});
  NVF_CHECK(id_0.id == id_0_lookup.id);
  NVF_CHECK(inputs_id_lookup.size() == 1);
  NVF_CHECK(id_0.eviction == false);

  // new input (even tho same shape, but we have different signature because of
  // missing scalar input
  auto id_1 = inputs_id_lookup.lookupId({t0, t1});
  auto id_1_lookup = inputs_id_lookup.lookupId({t0, t1});
  NVF_CHECK(id_1.id == id_1_lookup.id);
  NVF_CHECK(inputs_id_lookup.size() == 2);
  NVF_CHECK(id_1.eviction == false);

  // eviction should happen at this point
  auto id_2 = inputs_id_lookup.lookupId({t2, t1});
  NVF_CHECK(id_2.id != id_0.id);
  NVF_CHECK(id_2.id != id_1.id);
  NVF_CHECK(inputs_id_lookup.size() == 2);
  NVF_CHECK(id_2.eviction == true);
  NVF_CHECK(id_2.evict_id == id_0.id);

  // look at input 1 again
  auto id_1_relook = inputs_id_lookup.lookupId({t0, t1});
  NVF_CHECK(id_1_relook.id == id_1.id);
  NVF_CHECK(id_1_relook.eviction == false);

  // test scalars don't affect ID unless we ask them to
  auto id_3 = inputs_id_lookup.lookupId(
      {t0, t1, 5.0, 1, true}, /*scalar_inputs_to_record*/ {2, 3, 4});
  auto id_3_lookup = inputs_id_lookup.lookupId(
      {t0, t1, 2.5, 2, false}, /*scalar_inputs_to_record*/ {2, 3, 4});
  auto id_3_norecord = inputs_id_lookup.lookupId(
      {t0, t1, 5.0, 1, true}, /*scalar_inputs_to_record*/ {});
  auto id_3_lookup_norecord = inputs_id_lookup.lookupId(
      {t0, t1, 2.5, 2, false}, /*scalar_inputs_to_record*/ {});
  NVF_CHECK(id_3.id != id_3_lookup.id);
  NVF_CHECK(id_3_norecord.id == id_3_lookup_norecord.id);
}

TEST_F(NVFuserTest, FusionDisjointSet_CUDA) {
  DisjointSets<int> set;

  const std::set<int> group_x({0, 1, 2});
  const std::set<int> group_y({3, 4, 5});
  const std::set<int> group_z({6, 7, 8});
  const std::vector<std::set<int>> groups({group_x, group_y, group_z});
  std::set<int> group_all;
  std::for_each(groups.begin(), groups.end(), [&](const auto& g) {
    group_all.insert(g.begin(), g.end());
  });

  // Initially, nothing should be considered equivalent
  for (auto i : group_all) {
    for (auto j : group_all) {
      NVF_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_x are equivalent
  for (auto i : group_x) {
    for (auto j : group_x) {
      set.mapEntries(i, j);
      NVF_CHECK(set.mappingExists(i));
      NVF_CHECK(set.mappingExists(j));
    }
  }

  // All values in group_x shoudl be equivalent with each other
  for (auto i : group_x) {
    for (auto j : group_x) {
      NVF_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // But nothing else should be equivalent
  for (auto i : group_all) {
    for (auto j : group_y) {
      NVF_CHECK(!set.permissiveAreMapped(i, j));
    }
    for (auto j : group_z) {
      NVF_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_y are equivalent
  for (auto i : group_y) {
    for (auto j : group_y) {
      set.mapEntries(i, j);
      NVF_CHECK(set.mappingExists(i));
      NVF_CHECK(set.mappingExists(j));
    }
  }

  // group_x should be still equivalent
  for (auto i : group_x) {
    for (auto j : group_x) {
      NVF_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // group_y should be now equivalent
  for (auto i : group_y) {
    for (auto j : group_y) {
      NVF_CHECK(set.permissiveAreMapped(i, j));
    }
  }
  // But group_z should not be equivalent with anything yet
  for (auto i : group_all) {
    for (auto j : group_z) {
      NVF_CHECK(!set.permissiveAreMapped(i, j));
    }
  }

  // Sets values in group_z are equivalent
  for (auto i : group_z) {
    for (auto j : group_z) {
      set.mapEntries(i, j);
      NVF_CHECK(set.mappingExists(i));
      NVF_CHECK(set.mappingExists(j));
    }
  }

  // Now each of the three groups should be equivalent within each
  // group
  for (const auto gi : arange(groups.size())) {
    for (const auto gj : arange(groups.size())) {
      for (auto i : groups[gi]) {
        for (auto j : groups[gj]) {
          NVF_CHECK(
              (gi == gj && set.permissiveAreMapped(i, j)) ||
              (gi != gj && !set.permissiveAreMapped(i, j)));
        }
      }
    }
  }

  std::vector<int> all_elements = set.getAllElements().vector();
  std::sort(all_elements.begin(), all_elements.end());
  std::vector<int> group_all_vec(group_all.begin(), group_all.end());
  std::sort(group_all_vec.begin(), group_all_vec.end());
  NVF_CHECK(all_elements == group_all_vec);

  set.clear();
  NVF_CHECK(set.getAllElements().vector().size() == 0);

  // All cleared. Nothing should be considered equivalent.
  for (auto i : group_all) {
    for (auto j : group_all) {
      NVF_CHECK(!set.permissiveAreMapped(i, j));
    }
  }
}

TEST_F(NVFuserTest, FusionNonUniqueBroadcastSize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv0, {true, false});
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv3, tv2);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  // In order to do this, tv1->axis(1) and tv2->axis(1) must have the
  // same size, but we can't prove it, so this should throw an error.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(tv3->computeAt(tv4, -1));
}

TEST_F(NVFuserTest, FusionBiasGeluFwd_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;

  // bias vector
  auto t0 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(t0);
  auto t1 = castOp(DataType::Float, t0);
  // input tensor
  auto t2 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t2);
  auto t3 = castOp(DataType::Float, t2);
  auto t4 = broadcast(t1, {true, true, false});
  auto t5 = add(t4, t3);
  auto t6 = mul(t5, IrBuilder::create<Val>(0.5));
  auto t7 = mul(t5, IrBuilder::create<Val>(k_079));
  auto t8 = mul(t5, IrBuilder::create<Val>(k_004));
  auto t9 = mul(t8, t5);
  auto t10 = add(t9, IrBuilder::create<Val>(1L));
  auto t11 = mul(t7, t10);
  auto t12 = unaryOp(UnaryOpType::Tanh, t11);
  auto t13 = add(t12, IrBuilder::create<Val>(1L));
  auto t14 = mul(t6, t13);
  auto t15 = castOp(DataType::Half, t14);
  fusion.addOutput(t15);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<int64_t> input_shape{6, 512, 4096};
  std::vector<int64_t> bias_shape{4096};

  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_bias, at_input});
  testValidate(
      &fusion, cg_outputs.outputs, {at_bias, at_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBiasGeluBwd_CUDA) {
  if (at::cuda::getDeviceProperties(0)->major < 6) {
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  // disable fma to avoid numerical issue for reference implementation
  DisableOptionsGuard opt_guard;
  opt_guard.getCurOptions().set(DisableOption::Fma);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;
  const float k_010 = 0.1070322243;

  // gradient tensor
  auto t0 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t0);
  auto t1 = castOp(DataType::Float, t0);
  // bias tensor
  auto t2 = makeSymbolicTensor(1, DataType::Half);
  fusion.addInput(t2);
  auto t3 = castOp(DataType::Float, t2);
  // input tensor
  auto t4 = makeSymbolicTensor(3, DataType::Half);
  fusion.addInput(t4);
  auto t5 = castOp(DataType::Float, t4);
  auto t6 = broadcast(t3, {true, true, false});
  auto t7 = add(t6, t5);
  auto t8 = mul(t7, IrBuilder::create<Val>(k_079));
  auto t9 = mul(t7, IrBuilder::create<Val>(k_004));
  auto t10 = mul(t9, t7);
  auto t11 = add(t10, IrBuilder::create<Val>(1L));
  auto t12 = mul(t8, t11);
  auto t13 = unaryOp(UnaryOpType::Tanh, t12);
  auto t14 = mul(t7, IrBuilder::create<Val>(0.5));
  auto t15 = mul(t13, t13);
  auto t16 = unaryOp(UnaryOpType::Neg, t15);
  auto t17 = add(t16, IrBuilder::create<Val>(1L));
  auto t18 = mul(t7, IrBuilder::create<Val>(k_010));
  auto t19 = mul(t18, t7);
  auto t20 = add(t19, IrBuilder::create<Val>(k_079));
  auto t21 = mul(t17, t20);
  auto t22 = mul(t14, t21);
  auto t23 = add(t13, IrBuilder::create<Val>(1L));
  auto t24 = mul(t23, IrBuilder::create<Val>(0.5));
  auto t25 = add(t22, t24);
  auto t26 = mul(t25, t1);
  // Save float output for validation
  fusion.addOutput(t26);
  auto t27 = castOp(DataType::Half, t26);
  fusion.addOutput(t27);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<int64_t> input_shape{6, 512, 4096};
  std::vector<int64_t> bias_shape{4096};
  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);
  auto at_grad = at::randn(input_shape, options);

  auto cg_outputs = scheduleAndRun(
      &fusion, SchedulerType::PointWise, {at_grad, at_bias, at_input});
  auto tolerance_overwrite = ValidationConstants();
  // bump tolerance
  tolerance_overwrite.base_float_abs_tol = 3e-6;
  tolerance_overwrite.base_float_rel_tol = 4e-3;
  testValidate(
      &fusion,
      cg_outputs.outputs,
      {at_grad, at_bias, at_input},
      __LINE__,
      __FILE__,
      "",
      LaunchParams(),
      tolerance_overwrite);
}

// Reproducer of issue #459
TEST_F(NVFuserTest, FusionIssue459_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv1, tv3);

  // Create two outputs from the final arithmetic result
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);
  auto tv6 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv6);

  // Scheduling
  for (auto output : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    output->merge(-2, -1);
  }
  for (auto output : ir_utils::filterByType<TensorView>(fusion.outputs())) {
    output->split(0, 128);
  }

  tv0->computeAt(tv5, -1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int numel_x = 10;
  const int numel_y = 20;
  auto t0 = at::randn({numel_x}, options);
  auto t1 = at::randn({numel_y, numel_x}, options);

  nvfuser::KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSmemIndexingSimple_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto aten_input = at::randn({12, 34}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSmemIndexing_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Val* symbolic_m_tile_dim = IrBuilder::create<Val>(DataType::Int);
  Val* symbolic_split_k_tile_dim = IrBuilder::create<Val>(DataType::Int);
  Val* symbolic_block_k_tile_dim = IrBuilder::create<Val>(DataType::Int);
  // Compile-time integer for tiling
  int n_smem_tile = 32;

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
  TensorView* tv5 = sum(tv4, {1});

  // Register inputs and outputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  // Register runtime tile dims as inputs
  fusion.addInput(symbolic_m_tile_dim);
  fusion.addInput(symbolic_split_k_tile_dim);
  fusion.addInput(symbolic_block_k_tile_dim);

  // Make a 3D tile, mix of symbolic and constant, do in reverse order because
  // dims are inserted
  // [M, rK, N]
  tv5->split(2, n_smem_tile);
  // [M, rK, No, Ni{32}]
  tv5->split(1, symbolic_block_k_tile_dim);
  // [M, rKo, rKi{i2}, No, Ni{32}]
  tv5->split(1, symbolic_split_k_tile_dim);
  // [M, rKoo, rKoi{i1}, rKi{i2}, No, Ni{32}]
  tv5->split(0, symbolic_m_tile_dim);
  // [Mo, Mi{i0}, rKoo, rKoi{i1}, rKi{i2}, No, Ni{32}]

  // Reorder so all outer tiles are in the leftmost 3 positions
  // [Mo, Mi{i0}, rKoo, rKoi{i1}, rKi{i2},     No, Ni{32}]
  // [Mo,     No, rKoo, rKoi{i1}, rKi{i2}, Mi{i0}, Ni{32}]
  tv5->reorder({{1, 5}, {5, 1}});

  // Factor out the outer reduction IterDomain, then run the inter-cta
  // reduction, and intra-cta reduction
  // [Mo, No, rKoo,  Koi{i1},  Ki{i2}, Mi{i0}, Ni{32}]
  // [Mo, No,       rKoi{i1}, rKi{i2}, Mi{i0}, Ni{32}]
  auto tv6 = tv5->rFactor({2});

  // Scope computations
  tv6->computeAt(tv5, 2);

  // [Mo, No, rKoo, Koi{i1},  Ki{i2}, Mi{i0}, Ni{32}]
  // [Mo, No, Ki{i2}, Mi{i0}, Ni{32}, rKoo, Koi{i1}]
  tv6->reorder({
      {5, -2},
      {6, -1},
      {2, 2},
      {3, 3},
      {4, 4},
  });

  // Setup compute at schedule
  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);
  tv4->computeAt(tv6, -1);

  // Cache smem tiles
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Shared);
  tv6->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  std::vector<TensorView*> tv_list = {tv2, tv3, tv4, tv5, tv6};
  for (auto tv : tv_list) {
    tv->axis(-2)->parallelize(ParallelType::TIDz);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }

  constexpr int M = 31, K = 65, N = 32;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  // A, B, m_tile_dim, split_k, intra_cta_tile
  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, 3, 4, 5});
  auto cg_outputs = ke.run({t0, t1, 3, 4, 5});

  testValidate(
      &fusion,
      cg_outputs,
      {t0, t1, 3, 4, 5},
      {aten_output},
      __LINE__,
      __FILE__);
}

// Reproducer of issue 408
TEST_F(NVFuserTest, FusionCacheBeforeReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);

  tv2->split(0, 4);

  auto tv3 = tv2->cacheBefore();

  tv0->computeAt(tv3, -1);
  tv3->computeAt(tv2, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_x}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  ke.run({aten_input}, {cg_output});

  testValidate(&fusion, {cg_output}, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionCacheBeforeReduction2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  auto tv4 = tv2->cacheBefore();

  tv4->computeAt(tv3, 1);
  tv0->computeAt(tv4, -1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  const int numel_x = 10;
  const int numel_y = 20;
  const int numel_z = 30;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({numel_x, numel_y, numel_z}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue367_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic integers we will use for runtime tiling
  Val* symbolic_m_tile_dim = IrBuilder::create<Val>(DataType::Int);
  Val* symbolic_split_k_tile_dim = IrBuilder::create<Val>(DataType::Int);
  Val* symbolic_block_k_tile_dim = IrBuilder::create<Val>(DataType::Int);
  // Compile-time integer for tiling
  int n_smem_tile = 32;

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
  TensorView* tv5 = sum(tv4, {1});

  // Register inputs and outputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  // Register runtime tile dims as inputs
  fusion.addInput(symbolic_m_tile_dim);
  fusion.addInput(symbolic_split_k_tile_dim);
  fusion.addInput(symbolic_block_k_tile_dim);

  // Make a 3D tile, mix of symbolic and constant, do in reverse order because
  // dims are inserted
  // [M, K, N]
  tv5->split(2, n_smem_tile);
  tv5->split(1, symbolic_block_k_tile_dim);
  tv5->split(1, symbolic_split_k_tile_dim);
  tv5->split(0, symbolic_m_tile_dim);
  // [Mo, Mi, Koo, Koi, Ki, No, Ni]
  tv5->reorder({{1, 5}, {5, 1}});
  // [Mo, No, Koo, Koi, Ki, Mi, Ni]

  auto tv6 = tv5->rFactor({2});
  auto tv7 = tv5->rFactor({2});
  // [Mo, No, rKoo,  Koi,  Ki, Mi, Ni]
  // [Mo, No,       rKoi, rKi, Mi, Ni]

  // Scope computations
  tv6->computeAt(tv5, 2);

  tv0->computeAt(tv6, 3);
  tv1->computeAt(tv6, 3);
  tv4->computeAt(tv6, -1);

  // Cache smem tiles
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);
  tv4->setMemoryType(MemoryType::Local);
  tv6->setMemoryType(MemoryType::Local);
  tv7->setMemoryType(MemoryType::Local);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  std::vector<TensorView*> tv_list = {tv2, tv3, tv4, tv5, tv6, tv7};
  for (auto tv : tv_list) {
    tv->axis(-2)->parallelize(ParallelType::TIDz);
    tv->axis(-1)->parallelize(ParallelType::TIDy);
  }
  tv2->axis(3)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::TIDx);
  tv6->axis(3)->parallelize(ParallelType::TIDx);
  tv7->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(4)->parallelize(ParallelType::BIDx);
  tv3->axis(4)->parallelize(ParallelType::BIDx);
  tv4->axis(4)->parallelize(ParallelType::BIDx);
  tv6->axis(4)->parallelize(ParallelType::BIDx);
  tv7->axis(3)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::BIDx);

  constexpr int M = 3, K = 6, N = 16;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  at::Tensor aten_output =
      mul(t0.unsqueeze(2), t1.unsqueeze(0)).to(at::kDouble).sum(1);

  // A, B, m, split_k, block_k
  nvfuser::KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, 2, 2, 3});
  auto cg_outputs = ke.run({t0, t1, 2, 2, 3});

  testValidate(
      &fusion,
      cg_outputs,
      {t0, t1, 2, 2, 3},
      {aten_output},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionIssue468_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = sum(tv1, {0});
  fusion.addOutput(tv2);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({10, 100}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue363_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Symbolic 2D tensors TV0[M, K], TV1[K, N]
  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(2);

  // Broadcast tv0 to [M, K, *]
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  // Broadcast tv1 to [*, K, N]
  TensorView* tv3 = broadcast(tv1, {true, false, false});

  // Pointwise multiplication resulting in tv3[M, K, N]
  TensorView* tv4 = mul(tv2, tv3);

  // Sum the K-dim
  TensorView* tv5 = sum(tv4, {1});

  // Register inputs and outputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv5);

  tv2->setMemoryType(MemoryType::Global);
  tv3->setMemoryType(MemoryType::Global);
  tv4->setMemoryType(MemoryType::Global);

  tv0->computeAt(tv5, -1);
  tv1->computeAt(tv5, -1);

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::BIDy);

  tv5->axis(2)->parallelize(ParallelType::BIDx);

  constexpr int M = 3, K = 6, N = 16;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  nvfuser::KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue484_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Val>(0.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Global);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  constexpr int M = 100;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({M, M}, options);

  nvfuser::KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue329_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  fusion.addOutput(tv2);
  auto tv3 = sum(tv1, {1});
  fusion.addOutput(tv3);

  tv1->computeAt(tv2, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> t0_shape{17, 19};
  auto aten_input = at::randn(t0_shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue382_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv2->merge(1);
  tv4->merge(1);

  tv1->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);

  tv1->setMemoryType(MemoryType::Global);
  tv2->setMemoryType(MemoryType::Global);

  const int numel_x = 12;
  const int numel_y = 34;
  const int numel_z = 56;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({numel_x, numel_y}, options);
  auto t3 = at::randn({numel_x, numel_y, numel_z}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto cg_outputs = ke.run({t0, t3});

  testValidate(&fusion, cg_outputs, {t0, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue507_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<int64_t> t0_shape{17, 19};
  auto aten_input = at::randn(t0_shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  testValidate(&fusion, cg_outputs, {aten_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue532_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  const int M_BLOCK = 64;
  const int M_THREAD = 4;

  tv2->split(0, M_BLOCK);
  // tv2: [M/M_BLOCK, M_BLOCK]
  tv1->computeAt(tv2, 1);
  // tv1: [M/M_BLOCK, M_BLOCK]

  tv1->split(-1, M_BLOCK / M_THREAD);
  // tv1: [M/M_BLOCK, M_THREAD, M_BLOCK / M_THREAD]

  tv2->split(-1, M_THREAD);
  // tv2: [M/M_BLOCK, M_BLOCK / M_THREAD, M_THREAD]

  constexpr int M = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionLoopUnswitch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Algorithm
  TensorView* tv0 = makeSymbolicTensor(1);
  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addInput(tv0);
  fusion.addOutput(tv2);

  tv2->split(0, 32);
  tv1->computeAt(tv2, -1);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  constexpr int M = 1000;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue549_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2); // M, K
  TensorView* tv1 = makeSymbolicTensor(2); // K, N
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));

  TensorView* tv3 = broadcast(tv2, {false, false, true});
  // tv3[I0, I1, B] = tv0[I0, I1]

  TensorView* tv4 = broadcast(tv1, {true, false, false});
  // tv4[B, I1, I2] = tv1[I1, I2]

  // tv5[I0, I1, I2] = tv3[I0, I1, B] * tv4[B, I1, I2]
  TensorView* tv5 = mul(tv3, tv4);
  // tv6[I0, R1, I2] = tv5[I0, I1, I2]
  TensorView* tv6 = sum(tv5, {1});
  fusion.addOutput(tv6);

  tv6->split(1, 32);
  // tv6[I0, R1o, R1i{32}, I2]

  auto tv7 = tv6->rFactor({1});
  // tv7[I0, R1o, I1i{32}, I2] = tv5[I0, I1, I2]
  // tv6[I0,    , R1i{32}, I2] = tv7[I0, R1o, I1i{32}, I2]

  tv6->split(0, 4);
  tv6->split(-1, 4);
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]

  tv0->computeAt(tv6, -1);
  tv1->computeAt(tv6, -1);

  // tv7[I0o, I0i{4}, R1o, I1i{32}, I2o, I2i{4}]
  // tv6[I0o, I0i{4},    , R1i{32}, I2o, I2i{4}]
  //--> (line symbolizes compute at location)
  // tv5[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, I1o]
  // tv7[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, R1o]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv0->computeAt(tv7, -1);
  tv1->computeAt(tv7, -1);
  // tv5[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, I1o |]
  // tv7[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, R1o |]
  // tv6[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv6->axis(0)->parallelize(ParallelType::BIDz);
  tv6->axis(1)->parallelize(ParallelType::TIDz);

  tv6->axis(-2)->parallelize(ParallelType::BIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDy);

  tv6->axis(2)->parallelize(ParallelType::TIDx);
  tv7->axis(2)->parallelize(ParallelType::TIDx);

  constexpr int M = 65, K = 33, N = 17;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({M, K}, options);
  at::Tensor t1 = at::randn({K, N}, options);

  // Lets specify a few bounds in launch params to make sure it works
  LaunchParams lparams(1, -1, -1, 32, 4, 4);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1}, lparams);
  ke.run({t0, t1}, {}, lparams);

  // Make sure bad launch params throws
  // TODO: Re-enable once we have parallelization validation in.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  // ASSERT_ANY_THROW(ke.run({t0, t1}, LaunchParams(1, 2, 3, 4, 5, 6)));

  // Don't specify any launch params
  auto cg_outputs = ke.run({t0, t1});

  auto aten_output = (t0 + 1).to(at::kDouble).matmul(t1.to(at::kDouble));

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleCompileRtc_CUDA) {
  RtcKernel rk;
  std::string kernel = R"(
__global__ void kernel1(Tensor<float, 1> T0, Tensor<float, 1> T1) {
  if(threadIdx.x==0){
    for(size_t ki28 = 0; ki28 < T0.logical_size[0]; ++ki28) {
      T1[ki28*T1.alloc_stride[0]] = T0[ki28*T0.alloc_stride[0]]*2;
    }
  }
}
    )";
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      256, // gdimx
      1, // gdimy
      1, // gdimz
      1, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {8};
  auto in0 = at::randn(tensor_dims, options);
  auto out0 = at::empty_like(in0);
  rk.run(lp, {in0, out0}, PrimDataType::Int);

  auto out_ref = in0 * 2;
  NVF_CHECK(out_ref.allclose(out0));
}

TEST_F(NVFuserTest, FusionSerialWelford_CUDA) {
  int x = 128, y = 64, z = 64;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_var,
    Tensor<float,1> out_avg
){
    for(int i0=0;i0<inp.logical_size[0];i0++){
        float tmp_M2=0;
        float tmp_avg=0;
        long tmp_N=0;
        for(int i1=0;i1<inp.logical_size[1];i1++){
            for(int i2=0;i2<inp.logical_size[2];i2++){
                welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    inp[i0*inp.alloc_stride[0]+
                        i1*inp.alloc_stride[1]+
                        i2*inp.alloc_stride[2]],
                    0.f,
                    (long)1
                );
            }
        }
        out_var[i0*out_var.alloc_stride[0]]=
            tmp_M2/(tmp_N);
        out_avg[i0*out_avg.alloc_stride[0]]=
            tmp_avg;
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      1, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  rk.run(lp, {in0, out_var, out_avg}, PrimDataType::Int);

  NVF_CHECK(in0.var({1, 2}, false).allclose(out_var));
  NVF_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionBlockWelford_CUDA) {
  int x = 7, y = 8, z = 9;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,2> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> init_avg,
    Tensor<float,1> init_var,
    Tensor<long,0> init_N
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.alloc_stride[0]+
                        threadIdx.y*inp.alloc_stride[1]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    blockWelford<false,true,false, true>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long)1,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.logical_size[0]),
        0.f,
        blockDim);
    __syncthreads();
    if(threadIdx.x<out_var.logical_size[0] && threadIdx.y==0){
        welfordCombine(
                    tmp_avg,
                    tmp_M2,
                    tmp_N,
                    init_avg[threadIdx.x*init_avg.alloc_stride[0]],
                    init_var[threadIdx.x*init_var.alloc_stride[0]]*init_N[0],
                    init_N[0]
                );
        out_avg[threadIdx.x*out_avg.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y};
  const std::vector<int64_t> init_dims = {x, z};

  // generate initial values
  auto init_in = at::randn(init_dims, options);
  auto init_var = init_in.var({1}, false);
  auto init_avg = init_in.mean({1});
  auto init_N =
      at::tensor(z, at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));

  auto in0 = at::randn(tensor_dims, options);

  // run kernel
  auto out_var = at::zeros({x}, options);
  auto out_avg = at::zeros({x}, options);
  rk.run(
      lp,
      {in0, out_avg, out_var, init_avg, init_var, init_N},
      PrimDataType::Int);

  // compare with reference output
  auto cat_tensor = at::cat({init_in, in0}, 1);
  NVF_CHECK(cat_tensor.var({1}, false).allclose(out_var));
  NVF_CHECK(
      cat_tensor.mean({1}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionBlockWelfordNoInit_CUDA) {
  int x = 7, y = 8, z = 9;

  // need support IValue for integer input as initial count
  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var
){
    //actual generated kernel will use dynamic shared mem,
    // here is just for prototype
    __shared__ float mem_avg[512];
    __shared__ float mem_M2[512];
    __shared__ long mem_N[512];
    float in=inp[threadIdx.x*inp.alloc_stride[0]+
                        threadIdx.y*inp.alloc_stride[1]+
                        threadIdx.z*inp.alloc_stride[2]];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    block_sync::init();
    blockWelford<false,true,true, true>(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        (float*)mem_avg,
        (float*)mem_M2,
        (long*)mem_N,
        (bool)(threadIdx.x<inp.logical_size[0]),
        0.f,
        blockDim);
    __syncthreads();
    if(threadIdx.x<out_var.logical_size[0] && threadIdx.y==0 && threadIdx.z==0){
        out_avg[threadIdx.x*out_var.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/(tmp_N);
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      1, // gdimx
      1, // gdimy
      1, // gdimz
      x, // bdimx
      y, // bdimy
      z // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);
  auto out_var = at::empty({x}, options);
  auto out_avg = at::empty({x}, options);
  rk.run(lp, {in0, out_avg, out_var}, PrimDataType::Int);

  NVF_CHECK(in0.var({1, 2}, false).allclose(out_var));
  NVF_CHECK(in0.mean({1, 2}).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
}

TEST_F(NVFuserTest, FusionGridWelfordNoInit_CUDA) {
  KernelExecutor ke;
  int x = 128, y = 64, z = 128;

  std::string kernel = R"(
__global__ void kernel1(
    Tensor<float,3> inp,
    Tensor<float,1> out_avg,
    Tensor<float,1> out_var,
    Tensor<float,1> work_buf_avg,
    Tensor<float,1> work_buf_M2,
    Tensor<long,1> work_buf_N,
    Tensor<int64_t,1> sync_flag
){
    __shared__ float shared_buf_avg[512];
    __shared__ float shared_buf_M2[512];
    __shared__ long shared_buf_N[512];
    float tmp_avg=0;
    float tmp_M2=0;
    long tmp_N=0;
    float in = inp[ blockIdx.x  * inp.alloc_stride[0]+
                    blockIdx.y  * inp.alloc_stride[1]+
                    threadIdx.x * inp.alloc_stride[2]];
    block_sync::init();
    welford::gridWelford<
        true,true,false,
        true,false,false,
        false, true
    >(
        tmp_avg,
        tmp_M2,
        tmp_N,
        in,
        0.f,
        (long) 1,
        &work_buf_avg[0],
        &work_buf_M2[0],
        &work_buf_N[0],
        sync_flag,
        (float*)shared_buf_avg,
        (float*)shared_buf_M2,
        (long*)shared_buf_N,
        threadIdx.x<out_var.logical_size[0],
        threadIdx.x<out_var.logical_size[0],
        0.f,
        0,
        1,
        blockDim);
    if(blockIdx.x == gridDim.x - 1 && blockIdx.y == gridDim.y - 1){
        out_avg[threadIdx.x*out_avg.alloc_stride[0]]=tmp_avg;
        out_var[threadIdx.x*out_var.alloc_stride[0]]=tmp_M2/tmp_N;
    }
}
    )";
  RtcKernel rk;
  rk.compile(kernel, "kernel1", false, PrimDataType::Int);
  LaunchParams lp(
      x, // gdimx
      y, // gdimy
      1, // gdimz
      z, // bdimx
      1, // bdimy
      1 // bdimz
  );
  lp.setSmem(0);
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const auto options_int =
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);

  const std::vector<int64_t> tensor_dims = {x, y, z};
  auto in0 = at::randn(tensor_dims, options);

  auto out_avg = at::empty({z}, options);
  auto out_var = at::empty({z}, options);
  auto work_buf_avg = at::empty({x * y * z}, options);
  auto work_buf_var = at::empty({x * y * z}, options);
  auto work_buf_N = at::empty({x * y * z}, options_int);
  auto sync_flag = at::zeros({1}, options_int);
  rk.run(
      lp,
      {in0,
       out_avg,
       out_var,
       work_buf_avg,
       work_buf_var,
       work_buf_N,
       sync_flag},
      PrimDataType::Int);
  std::vector<int64_t> dims{0, 1};

  NVF_CHECK(in0.mean(dims).allclose(out_avg, /*rtol*/ 1e-5, /*atol*/ 1e-6));
  NVF_CHECK(in0.var(dims, false).allclose(out_var));
}

TEST_F(NVFuserTest, FusionWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 32);
  tv_avg->split(0, 32);
  tv_avg->split(0, 4);
  tv_avg->reorder({{-1, -3}, {-3, -1}});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionBlockWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->computeAt(tv_avg, -1);

  //
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionGridWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->axis(0)->parallelize(ParallelType::TIDx);
  tv_avg->axis(-1)->parallelize(ParallelType::BIDx);

  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorWelfordOp_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  tv_avg->split(1, 4);
  ir_utils::rFactorHelper(tvs.avg, {2});
  tv1->computeAt(tv_avg, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  at::Tensor t_avg = at::empty({M}, options);
  at::Tensor t_var = at::empty({M}, options);
  at::Tensor t_N = at::empty({M}, options_int);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  outputs[1] = outputs[1].as<at::Tensor>() / N;

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {t0},
      {t0.mean({1}), t0.var({1}, false), at::ones({M}, options_int) * N},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordSchedule_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = mul(tv0, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {1});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_int = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);
  auto cg_results = scheduleAndRun(&fusion, SchedulerType::Reduction, {t0});

  // by default Welford outputs sum of square diff so need to divide to get var
  cg_results.outputs[1] = cg_results.outputs[1].as<at::Tensor>() / N;

  auto at_avg = t0.mean({1});
  auto at_var = t0.var({1}, false);
  auto at_n = at::ones({M}, options_int) * N;

  testValidate(
      &fusion,
      cg_results.outputs,
      {t0},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      cg_results.heuristic_params->lparams);
}

using WelfordReductionParams = std::tuple<DataType, int64_t, int64_t, int64_t>;
using WelfordReduction = NVFuserFixtureParamTest<WelfordReductionParams>;
TEST_P(WelfordReduction, Test) {
  auto [dtype, rdim, odim, axis] = GetParam();

  // TODO: original welford algorithm actually keeps a running sum of
  // squares, i.e. M_{2n} in the
  //       cf:
  //       https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  //       algorithm notation, and it can reach inf for large numbers
  //       with half precision. skipping too large volumes for half for
  //       nwo might need further numerical experiments to re-design
  //       this.
  if (rdim > 32768 &&
      (dtype == DataType::Half || dtype == DataType::BFloat16)) {
    GTEST_SKIP() << "Skipping large reduction dims (" << rdim
                 << ") for half and bfloat16";
  }

  maybeClearAllocator();

  at::ScalarType aten_dtype = data_type_to_aten(dtype);

  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* tv0 = makeSymbolicTensor(2, dtype);
  bool is_fp16 = dtype == DataType::Half;
  bool is_bf16 = dtype == DataType::BFloat16;
  TensorView* tv0_cast = tv0;
  if (is_fp16 || is_bf16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }
  fusion.addInput(tv0);
  auto tv1 = mul(tv0_cast, IrBuilder::create<Val>(1.0));
  auto tvs = Welford(tv1, {axis});
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  auto tv_N = tvs.n;

  TensorView* avg_cast = tv_avg;
  TensorView* M2_cast = tv_M2;

  if (is_fp16) {
    avg_cast = castOp(DataType::Half, tv_avg);
    M2_cast = castOp(DataType::Half, tv_M2);
  }
  if (is_bf16) {
    avg_cast = castOp(DataType::BFloat16, tv_avg);
    M2_cast = castOp(DataType::BFloat16, tv_M2);
  }

  fusion.addOutput(avg_cast);
  fusion.addOutput(M2_cast);
  fusion.addOutput(tv_N);

  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  std::vector<TensorView*> outputs_of_red;
  at::Tensor aten_input =
      (axis ? at::randn({odim, rdim}, options)
            : at::randn({rdim, odim}, options));

  if (is_fp16 || is_bf16) {
    outputs_of_red.push_back(avg_cast);
    outputs_of_red.push_back(M2_cast);
  }

  auto heuristic_params = SchedulerEntry::scheduleWith(
      &fusion, SchedulerType::Reduction, {aten_input});
  auto reduction_params = heuristic_params->as<ReductionParams>();

  auto lparams = reduction_params->lparams;
  auto cparams = reduction_params->cparams;

  KernelExecutor ke;
  // Needs to pass compile para to use the correct index type, otherwise the
  // lowering pass will use int64 as the index tpye, since this test saves
  // `tv_N` as index type, it may cause vectorization size validation error. For
  // example, the heuristics set index type to int32 and the max vectorization
  // factor is 4, if compile para is not passed to compile, the lowering
  // pass uses int64 as index type, so the max vectorization factor is 16 bytes
  // sizeof(int64) = 2, which is wrong since the actual index type is int32
  // and the max vectorization factor is 4.
  ke.compile(&fusion, {aten_input}, lparams, cparams);
  auto outputs = ke.run({aten_input}, {}, lparams);

  // by default Welford outputs sum of square diff so need to divide to
  // get var

  outputs[1] = outputs[1].as<at::Tensor>() / rdim;

  auto at_avg = aten_input.mean({axis});
  auto at_var = aten_input.var({axis}, false);
  auto at_n =
      (axis ? at::ones({odim, rdim}, options)
            : at::ones({rdim, odim}, options));
  at_n = at_n.sum({axis});

  testValidate(
      ke.compiledKernel()->kernel(),
      outputs,
      {aten_input},
      {at_avg, at_var, at_n},
      __LINE__,
      __FILE__,
      "validate welford",
      reduction_params->lparams);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    WelfordReduction,
    ::testing::Combine(
        testing::ValuesIn(getFloatingDataTypes()), // data type
        testing::ValuesIn(Pow2Vals1to1Million), // reduction dimension size
        testing::Values(160, 320), // iteration dimension size
        testing::Values(0, 1)), // reduction axis
    // when using structured bindings within TestParamInfo,
    // parentheses are required to avoid compile errors,
    // see https://github.com/google/googletest/issues/3848
    ([](const testing::TestParamInfo<WelfordReductionParams>& info)
         -> std::string {
      std::stringstream ss;
      auto [dtype, rdim, odim, axis] = info.param;
      ss << "dtype_" << dtype;
      ss << "_redu_" << rdim;
      ss << "_iter_" << odim;
      ss << "_axis_" << axis;
      return sanitizeTestName(ss.str());
    }));

namespace {
void testVarMean(at::ScalarType dtype, int correction, bool keepdim) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int M = 64, N = 128;

  auto tv0 = makeSymbolicTensor(2, aten_to_data_type(dtype));
  fusion->addInput(tv0);
  auto tvs = variance_mean(tv0, {1}, correction, keepdim);
  auto tv_mean = tvs.mean;
  auto tv_var = tvs.var;
  fusion->addOutput(tv_var);
  fusion->addOutput(tv_mean);

  auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto outputs = executor_cache.runFusionWithInputs({t0});

  testValidate(executor_cache.fusion(), outputs, {t0}, __LINE__, __FILE__);
}
} // namespace

TEST_F(NVFuserTest, FusionVarMean_CUDA) {
  std::vector<at::ScalarType> dtypes = {
      at::kFloat, at::kDouble, at::kComplexFloat, at::kComplexDouble};
  std::vector<int> corrections = {0, 1};
  std::vector<bool> keepdims = {false, true};
  for (auto correction : corrections) {
    for (auto keepdim : keepdims) {
      for (auto dtype : dtypes) {
        testVarMean(dtype, correction, keepdim);
      }
    }
  }
}

TEST_F(NVFuserTest, FusionSimpleGemmTransposed_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views

  TensorView* tv0 = makeSymbolicTensor(2); // K, M
  TensorView* tv1 = makeSymbolicTensor(2); // N, K
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  TensorView* tv0_t = transpose(tv0);
  TensorView* tv1_t = transpose(tv1);

  TensorView* tv2 = broadcast(tv0_t, {false, false, true});
  // tv2[I0, I1, B] = tv0[I0, I1]

  TensorView* tv3 = broadcast(tv1_t, {true, false, false});
  // tv3[B, I1, I2] = tv1[I1, I2]

  // tv4[I0, I1, I2] = tv2[I0, I1, B] * tv3[B, I1, I2]
  TensorView* tv4 = mul(tv2, tv3);
  // tv5[I0, R1, I2] = tv4[I0, I1, I2]
  TensorView* tv5 = sum(tv4, {1});
  fusion.addOutput(tv5);

  tv5->split(1, 32);
  // tv5[I0, R1o, R1i{32}, I2]

  auto tv6 = tv5->rFactor({1});
  // tv6[I0, R1o, I1i{32}, I2] = tv4[I0, I1, I2]
  // tv5[I0,    , R1i{32}, I2] = tv6[I0, R1o, I1i{32}, I2]

  tv5->split(0, 4);
  tv5->split(-1, 4);
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}]

  tv0_t->computeAt(tv5, -1);
  tv1_t->computeAt(tv5, -1);

  // tv6[I0o, I0i{4}, R1o, I1i{32}, I2o, I2i{4}]
  // tv5[I0o, I0i{4},    , R1i{32}, I2o, I2i{4}]
  //--> (line symbolizes compute at location)
  // tv4[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, I1o]
  // tv6[I0o, I0i{4}, I1i{32}, I2o, I2i{4}|, R1o]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv0_t->computeAt(tv6, -1);
  tv1_t->computeAt(tv6, -1);
  // tv4[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, I1o |]
  // tv6[I0o, I0i{4}, I1i{32}, I2o, I2i{4}, R1o |]
  // tv5[I0o, I0i{4}, R1i{32}, I2o, I2i{4}|]

  tv5->axis(0)->parallelize(ParallelType::BIDz);
  tv5->axis(1)->parallelize(ParallelType::TIDz);

  tv5->axis(-2)->parallelize(ParallelType::BIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);

  tv5->axis(2)->parallelize(ParallelType::TIDx);
  tv6->axis(2)->parallelize(ParallelType::TIDx);

  constexpr int M = 65, K = 33, N = 17;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({K, M}, options);
  at::Tensor t1 = at::randn({N, K}, options);

  // Lets specify a few bounds in launch params to make sure it works
  LaunchParams lparams(1, -1, -1, 32, 4, 4);
  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1}, lparams);
  ke.run({t0, t1}, {}, lparams);

  // Don't specify any launch params
  auto cg_outputs = ke.run({t0, t1});

  auto aten_output = t0.t().to(at::kDouble).matmul(t1.t().to(at::kDouble));

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSoftmax3DTransposed_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int tidx = 32;
  const int dimx = 32;
  const int dimy = 16;
  const int dimz = 130;

  // Set up your input tensor views
  TensorView* input_tv0 = makeSymbolicTensor(3);
  fusion.addInput(input_tv0);

  TensorView* input_t = transpose(input_tv0, 1, 2);

  TensorView* exp_tv1 = unaryOp(UnaryOpType::Exp, input_t);
  TensorView* sum_exp_tv2 = sum(exp_tv1, {-1});
  TensorView* bcast_sum_tv3 = broadcast(sum_exp_tv2, {false, false, true});

  // Replicate exp_tv4 as exp_tv4_copy because exp_tv4 is going to be
  // computed at sum_exp_rf_tv8.
  TensorView* input_t_copy = transpose(input_tv0, 1, 2);
  TensorView* exp_tv1_copy = unaryOp(UnaryOpType::Exp, input_t_copy);

  TensorView* output_tv4 = div(exp_tv1_copy, bcast_sum_tv3);

  fusion.addOutput(output_tv4);

  bcast_sum_tv3->split(-1, tidx);

  sum_exp_tv2->split(-1, tidx);
  TensorView* sum_exp_rf_tv5 = sum_exp_tv2->rFactor({-2});

  output_tv4->split(-1, tidx);

  input_t->computeAt(sum_exp_rf_tv5, -1);
  input_t_copy->computeAt(output_tv4, -1);

  TensorView* tensors_to_parallelize[] = {
      sum_exp_tv2, bcast_sum_tv3, output_tv4, sum_exp_rf_tv5};

  for (auto tv : tensors_to_parallelize) {
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({dimx, dimz, dimy}, options);

  at::Tensor cg_output = at::empty({dimx, dimy, dimz}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  ke.run({input}, {cg_output});

  auto aten_input_t = at::transpose(input, 1, 2);
  auto aten_output = at::_softmax(aten_input_t.to(at::kDouble), -1, false);

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed1_CUDA) {
  // Case 1
  // tv1 = tv0 * 0.5
  // tv2 = tv1 * -1
  // tv3 = tv1 + 3
  // tv4 = tv1 * 2
  // tv5 = tv3 + tv2
  // tv6 = tv5 + tv4
  // tv7 = tv1 + tv4
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  tv0 = transpose(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Val>(0.5));
  TensorView* tv2 = mul(tv1, IrBuilder::create<Val>(-1.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = mul(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv5 = add(tv3, tv2);

  TensorView* tv6 = add(tv5, tv4);
  TensorView* tv7 = add(tv1, tv4);

  fusion.addOutput(tv6);
  fusion.addOutput(tv7);

  // Lets setup to actually run
  tv7->merge(0);
  tv7->split(0, 128);
  tv7->split(0, 4);

  tv7->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv7, 1);

  // The this-position of the last tensor should be zero.
  NVF_CHECK(
      tv7->nDims() == 3 && tv7->getComputeAtPosition() == 0 &&
      tv7->getMaxProducerPosition() == 1);
  NVF_CHECK(
      tv6->nDims() == 3 && tv6->getComputeAtPosition() == 0 &&
      tv6->getMaxProducerPosition() == 1);
  // The position of every other tensor should be 1.
  for (auto tv : {tv1, tv2, tv3, tv4, tv5}) {
    NVF_CHECK(tv->nDims() == 3 && tv->getComputeAtPosition() == 1);
  }

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);
      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor aten_input = at::randn({129, 127}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {aten_input});
  auto cg_outputs = ke.run({aten_input});

  at::Tensor aten_input_t = aten_input.t();

  auto t1 = aten_input_t.mul({0.5});
  auto t2 = t1.mul({-1.0});
  auto t3 = t1.add({3.0});
  auto t4 = t1.mul({2.0});
  auto t5 = t3.add(t2);
  auto t6 = t5.add(t4);
  auto t7 = t1.add(t4);

  std::vector<at::Tensor> aten_outputs = {t6, t7};

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed2_CUDA) {
  // Case 2
  // tv1 = tv0 * -1
  // tv2 = tv0 + 3
  // tv3 = tv0 * 2
  // tv4 = tv2 + tv1
  // tv5 = tv4 + tv3
  // tv6 = tv5 + tv3
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  tv0 = transpose(tv0);

  TensorView* tv1 = mul(tv0, IrBuilder::create<Val>(-1.0));
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(3.0));
  TensorView* tv3 = mul(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv4 = add(tv2, tv1);

  TensorView* tv5 = add(tv4, tv3);
  TensorView* tv6 = add(tv5, tv3);

  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  // Lets setup to actually run
  tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  tv0->computeAt(tv6, 1);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({129, 127}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto cg_outputs = ke.run({input});

  auto input_t = input.t();
  auto t1 = input_t.mul({-1.0});
  auto t2 = input_t.add({3.0});
  auto t3 = input_t.mul({2.0});
  auto t4 = t2.add(t1);
  auto t5 = t4.add(t3);
  auto t6 = t5.add(t3);

  std::vector<at::Tensor> aten_outputs = {t5, t6};

  testValidate(&fusion, cg_outputs, {input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed3_CUDA) {
  // Case 3
  // T2 = T1 * 0.979361
  // T3 = T2 * T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  tv0 = permute(tv0, {3, 0, 1, 2});

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  tv1 = permute(tv1, {3, 0, 1, 2});

  TensorView* tv2 = mul(tv1, IrBuilder::create<Val>(.979361));
  TensorView* tv3 = mul(tv2, tv0);

  fusion.addOutput(tv3);

  // Lets setup to actually run
  while (tv3->nDims() > 1)
    tv3->merge(0);
  tv3->split(0, 128);
  tv3->split(0, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto t0_t = t0.permute({3, 0, 1, 2});
  auto t1_t = t1.permute({3, 0, 1, 2});
  auto t2 = t1_t.mul({0.979361});
  auto aten_output = t2.mul(t0_t);

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed4_CUDA) {
  // Case 4
  // T4 = T2 - T3
  // T5 = T1 + T4
  // T6 = T5 - T0
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(4);
  fusion.addInput(tv0);

  tv0 = permute(tv0, {3, 0, 1, 2});

  TensorView* tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv1);

  tv1 = permute(tv1, {3, 0, 1, 2});

  TensorView* tv2 = makeSymbolicTensor(4);
  fusion.addInput(tv2);

  tv2 = permute(tv2, {3, 0, 1, 2});

  TensorView* tv3 = makeSymbolicTensor(4);
  fusion.addInput(tv3);

  tv3 = permute(tv3, {3, 0, 1, 2});

  TensorView* tv4 = sub(tv2, tv3);
  TensorView* tv5 = add(tv1, tv4);
  TensorView* tv6 = sub(tv5, tv0);

  fusion.addOutput(tv6);

  // Lets setup to actually run
  while (tv6->nDims() > 1)
    tv6->merge(0);
  tv6->split(0, 128);
  tv6->split(0, 4);

  tv0->computeAt(tv6, 1);
  tv1->computeAt(tv6, 1);
  tv2->computeAt(tv6, 1);
  tv3->computeAt(tv6, 1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);

  for (Val* val : fusion.vals()) {
    if (!val->isFusionInput() &&
        val->getValType().value() == ValType::TensorView) {
      TensorView* tv = static_cast<TensorView*>(val);

      tv->axis(1)->parallelize(ParallelType::Unroll);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({129, 127, 63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);
  at::Tensor t2 = at::rand_like(t0, options);
  at::Tensor t3 = at::rand_like(t0, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2, t3});
  auto cg_outputs = ke.run({t0, t1, t2, t3});

  auto t0_t = t0.permute({3, 0, 1, 2});
  auto t1_t = t1.permute({3, 0, 1, 2});
  auto t2_t = t2.permute({3, 0, 1, 2});
  auto t3_t = t3.permute({3, 0, 1, 2});
  auto t4 = t2_t.sub(t3_t);
  auto t5 = t1_t.add(t4);
  auto aten_output = t5.sub(t0_t);

  testValidate(
      &fusion, cg_outputs, {t0, t1, t2, t3}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed5_CUDA) {
  // Case 5
  // tv2 = tv0 + 2.0
  // tv3 = tv1 * tv2
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  tv0 = transpose(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  tv1 = transpose(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->merge(0);
  tv3->split(-1, 8);
  tv3->split(-1, 4);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto t2 = t0.t().add(2.0);
  auto aten_output = t1.t().mul(t2);

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionAdvancedComputeAtTransposed6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  tv0 = transpose(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  tv1 = transpose(tv1);
  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = mul(tv1, tv2);
  fusion.addOutput(tv3);

  tv2->merge(0);
  tv2->split(-1, 8);
  tv2->split(-1, 4);
  tv3->merge(0);
  tv3->split(-1, 8);

  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({63, 65}, options);
  at::Tensor t1 = at::rand_like(t0, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  auto t2 = t0.t().add(2.0);
  auto aten_output = t1.t().mul(t2);

  testValidate(
      &fusion, cg_outputs, {t0, t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmentReducePointwise_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  TensorView* tv0 = makeSymbolicTensor(2);
  TensorView* tv1 = makeSymbolicTensor(1);
  TensorView* tv2 = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(1.0)); // Group 0
  TensorView* tv4 =
      max(tv3, {0}); // Group 0 (use max instead to avoid numerical issues)
  TensorView* tv5 = add(tv4, tv1); //  Group 0 (Non Broadcast after reduce,
                                   //  keeps normalization scheduler away)
  TensorView* tv6 = add(tv5, tv2); //  Group 1 (Broadcast after reduce)

  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  at::Tensor t1 = at::randn({65}, options);
  at::Tensor t2 = at::randn({128, 65}, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation didn't happen");
  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()
              ->fusionSegments()
              ->groups()
              .size() == 2,
      "segmentation didn't happen as expected");

  testValidate(
      executor_cache.fusion(), outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmentReduceSoftmax_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  std::vector<int64_t> input_shape{32, 64, 8};
  const int kReductionAxis = 1;

  auto tv0 = TensorViewBuilder()
                 .ndims(input_shape.size())
                 .dtype(DataType::Double)
                 .build();

  fusion->addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {2}); // Group 0

  auto output = softmax(tv2, kReductionAxis); // Group 1
  fusion->addOutput(output);

  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto outputs = executor_cache.runFusionWithInputs({at_x});

  auto optimized_fusion = executor_cache.getMostRecentKernelRuntime();
  ASSERT_TRUE(optimized_fusion->isSegmented()) << "segmentation didn't happen";
  ASSERT_EQ(optimized_fusion->fusionSegments()->groups().size(), 2)
      << "segmentation didn't happen as expected";

  // Make sure the second kernel is vectorized. See issue #658
  auto& heuristic_params = executor_cache.getMostRecentKernelRuntime()
                               ->schedulerHeuristics()
                               ->heuristicsList()
                               .at(1);
  ASSERT_TRUE(heuristic_params->isA<ReductionParams>());
  auto rparams = heuristic_params->as<ReductionParams>();
  ASSERT_TRUE(rparams->vectorize_inner_reduction) << "Failed to vectorize";
  ASSERT_EQ(rparams->unroll_factor_inner_reduction, 2)
      << "Unexpected vectorization factor";

  testValidate(executor_cache.fusion(), outputs, {at_x}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridPersistence_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  std::vector<TensorView*> tvs = {tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
  }

  const int numel_x = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  testValidate(&fusion, out, {input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridPersistence2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  fusion.addOutput(tv3);

  std::vector<TensorView*> tvs = {tv1, tv2, tv3};
  for (auto tv : tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }

  const int numel_x = 10;
  const int numel_y = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  testValidate(&fusion, out, {input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordPersistence_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};

  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::BIDy);
  }

  const int numel_x = 10;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(-1)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWelfordPersistence2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0});
  auto tv4 = add(tvs.avg, tvs.var_sum);
  auto tv5 = broadcast(tv4, {true, false});
  auto tv6 = add(tv0, tv5);
  fusion.addOutput(tv6);

  std::vector<TensorView*> schedule_tvs = {
      tvs.avg, tvs.var_sum, tvs.n, tv5, tv6};
  for (auto tv : schedule_tvs) {
    tv->split(0, 2);
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDx);
  }
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 10;
  const int numel_y = 3;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  auto out = ke.run({input});

  auto aten_output = (input.mean({0}) + (input.var({0}, false) * numel_x))
                         .unsqueeze(0)
                         .add(input);

  testValidate(&fusion, out, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue633_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int dx = 10;
  const int dy = 11;
  const int dz = 12;

  auto tv0 = makeConcreteTensor({dx, dy, dz});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({dx, dy, 1});
  fusion.addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion.addOutput(tv2);

  tv2->merge(1);
  tv2->merge(0);
  tv2->split(-1, 128);

  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({dx, dy, dz}, options);
  at::Tensor t1 = at::randn({dx, dy, 1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBroadcastAcrossComputeAt_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape{17, 19};

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->split(1, 128);
  tv0->computeAt(tv3, 2);

  for (auto tv : {tv2, tv3}) {
    tv->axis(-1)->parallelize(ParallelType::TIDx);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({shape[0]}, options);
  at::Tensor t1 = at::randn(shape, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto cg_outputs = ke.run({t0, t1});

  testValidate(&fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

// Unswitched loops with extent one may omit else clause.
TEST_F(NVFuserTest, FusionSizeOneLoop1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Progressively broadcast tensors
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = broadcast(tv0, {false, true});
  TensorView* tv4 = add(tv3, tv1);
  TensorView* tv5 = add(tv4, tv2);

  fusion.addOutput(tv5);

  // Split inner dimension
  tv5->split(1, 8);
  // Merge middle dims with outer dimensions
  tv5->merge(2);
  tv5->merge(0);

  // tv5[I0*I1o, I1i*I2]
  // Get a dim of size 1 to unswitch
  tv5->split(0, 1, false);

  // Compute everything inline
  tv0->computeAt(tv5, -1);

  tv5->axis(0)->parallelize(ParallelType::Unswitch);
  tv5->axis(1)->parallelize(ParallelType::BIDx);
  tv5->axis(2)->parallelize(ParallelType::TIDx);

  // Make sure the unswitched loop does not have an else clause.
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(!UnswitchInElseChecker::check(gpulw));

  const int x = 11;
  const int y = 12;
  const int z = 13;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t1 = at::randn({x, y}, options);
  at::Tensor t2 = at::randn({z, x, y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1, t2});
  auto cg_outputs = ke.run({t0, t1, t2});

  testValidate(&fusion, cg_outputs, {t0, t1, t2}, __LINE__, __FILE__);
}

// The unswitched loop has extent one but inner loops don't. The else
// part should not be omitted.
TEST_F(NVFuserTest, FusionSizeOneLoop2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int x = 15;
  auto tv0 = makeConcreteTensor({x});
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  tv1->split(-1, 4);
  tv1->split(-2, 1);

  tv1->axis(-2)->parallelize(ParallelType::Unswitch);

  // Make sure the size-one unswitched loop does not omit the else clause.
  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(UnswitchInElseChecker::check(gpulw));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto cg_outputs = ke.run({t0});

  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionValidateParallelize1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);

  // Invalid as tv1 and tv2 do have the same ParallelType
  KernelExecutor ke;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(ke.compile(&fusion));
}

TEST_F(NVFuserTest, FusionValidateParallelize2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv1->setMemoryType(MemoryType::Shared);

  // tv1 and tv2 do have the same ParallelType, but tv1 is on shared
  // memory, so it is valid
  KernelExecutor ke;
  ke.compile(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->split(-1, 4);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Global);

  // tv1 and tv2 have the same shape and ParallelType
  KernelExecutor ke;
  ke.compile(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->split(-1, 8);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Global);

  // tv1 and tv2 do not have the same shape but global memory comm is supported.
  KernelExecutor ke;
  ke.compile(&fusion);
}

TEST_F(NVFuserTest, FusionValidateParallelize5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv1->split(-1, 4);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 8);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // tv1 and tv2 do not have the same shape, but tv1 is on shared
  // memory, so it is valid
  KernelExecutor ke;
  ke.compile(&fusion);
}

// See issue #995
TEST_F(NVFuserTest, FusionValidateParallelize6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int64_t W = 5, X = 6, Y = 7, Z = 8;

  auto tv0 = makeConcreteTensor({X, Y, Z});
  auto tv1 = makeConcreteTensor({W, X, Y, Z});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);
  tv4->split(0, 4);
  tv4->split(0, 3);
  tv4->split(0, 2);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv0->computeAt(tv2, 2);
  tv3->computeAt(tv4, 2);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  // Validation should throw an exception saying the first axes of tv2
  // and tv3 have incompatible parallelization. See also issue #995.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

// Repro of https://github.com/csarofeen/pytorch/issues/2046
TEST_F(NVFuserTest, FusionValidateParallelize7_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // The original repro used set instead of exp. However, that is now aliased as
  // a tensor producer alias so it gets skipped when tv1 is scheduled on Global
  // memory.
  auto tv1 = exp(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);

  tv1->setMemoryType(MemoryType::Global);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  // tv2 uses tv1 but is not parallelized with BIDx, so a grid sync is
  // required. It should be placed as a top-level expression.

  GpuLower gpulw(&fusion);
  gpulw.run();
  NVF_CHECK(
      std::any_of(
          gpulw.kernel()->topLevelExprs().begin(),
          gpulw.kernel()->topLevelExprs().end(),
          [](Expr* expr) { return expr->isA<kir::GridSync>(); }),
      "Grid sync not found");
}

// From issue #1880
TEST_F(NVFuserTest, FusionValidateParallelize8_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv1);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv3, tv0);
  fusion.addOutput(tv4);

  tv3->split(0, 32);
  tv3->reorder({{1, -1}});
  tv3->split(1, 32);
  tv3->reorder({{2, -1}});
  tv3->merge(2);
  tv3->split(2, 16);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);

  MaxLogicalDomainInfoSpanningTree tree(tv3);
  TransformPropagator tp(tv3);
  tree.traverse(&tp);
  scheduler_utils::parallelizeAllLike(tv3);

  // Fully inline tv3, but inline tv2 only at position 1. This makes
  // tv3 indices are resolved based on the whole tv4 IDs, but tv2 uses
  // only the outer-most ID, resulting in different indeices and thus
  // requiring tv2 to be placed on shared memory as they are
  // parallelized with TIDx.
  tv2->computeAt(tv4, 1);
  tv3->computeAt(tv4, -1);

  // Since tv2 is not on shared memory, the fusion should be flagged
  // as invalid by the parallel validation
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(SyncMap sync_map_fail(&fusion));

  // The fusion should work if tv2 is also fully inlined
  tv2->computeAt(tv4, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::arange(64, options).view({32, 2});
  at::Tensor t1 = at::arange(32, options) * 0.01;

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionValidateParallelize9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> t0_shape({10});
  std::vector<int64_t> t1_shape({10, 5});

  auto tv0 = makeConcreteTensor(t0_shape);
  auto tv1 = makeConcreteTensor(t1_shape);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv4);

  tv4->merge(0)->split(0, 4);

  MaxLogicalDomainInfoSpanningTree tree(tv4);
  TransformPropagator tp(tv4);
  tree.traverse(&tp);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  // No CA. Although a broadcast is merged, it's never concretized and
  // the merge is done with just a broadcast of extent 1, thus it can
  // be ignored. While this fusion should be valid, the validation
  // doesn't consider whether concretized or not, so the validation
  // should result in a failure at this point. Note that even if it's
  // not concretized, it may be split by some factor, which creates
  // non-size-1 broadcast domains. If they are merged, it's not valid
  // to ignore, so we need to check each of forwarded broadcast merges
  // and make sure the forwarded broadcast domains have extent 1.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(SyncMap sync_map_fail(&fusion));
}

TEST_F(NVFuserTest, FusionValidateParallelize10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t s0 = 10;
  const int64_t s1 = 5;

  auto tv0 = makeConcreteTensor({s0});
  auto tv1 = makeConcreteTensor({s0, s1});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv2);
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  tv5->merge(0)->split(0, 4);

  TransformPropagatorWithCheck propagator(tv5);
  MaxLogicalDomainInfoSpanningTree(tv5).traverse(&propagator);

  // tv2 has no CA
  tv3->computeAt(tv5, 1);

  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5);

  // Since tv2 has no CA, it's indexed differently from tv3.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(SyncMap sync_map_fail(&fusion));

  // If tv2 is also computed at, all tensors should be indexed
  // uniformly
  tv2->computeAt(tv5, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({s0}, options);
  at::Tensor t1 = at::randn({s0, s1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Similar to ValidateParallelize10, tv2 has a shared loop axis
TEST_F(NVFuserTest, FusionValidateParallelize11_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int64_t s0 = 10;
  const int64_t s1 = 5;

  auto tv0 = makeConcreteTensor({s0});
  auto tv1 = makeConcreteTensor({s0, s1});
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = set(tv2);
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv4, tv1);
  fusion.addOutput(tv5);

  tv5->merge(0)->split(0, 4)->split(0, 2);

  TransformPropagatorWithCheck propagator(tv5);
  MaxLogicalDomainInfoSpanningTree(tv5).traverse(&propagator);

  tv2->computeAt(tv5, 1);

  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv5);

  // Although tv3->axis(1) is a consumer-only loop ID permissively
  // mapped with its consumer, tv3->axis(0) and tv2->axis(0) are
  // shared, so all the tensors are indexed consistently. No sync is
  // required.

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({s0}, options);
  at::Tensor t1 = at::randn({s0, s1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionDAGMerging_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(5);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Branch 0
  auto tv2 = sum(tv0, {0}); // 0
  auto tv3 = sum(tv2, {0}); // 1
  auto tv4 = sum(tv3, {0}); // 2
  auto tv5 = sum(tv4, {0}); // 3

  // Branch 1
  auto tv6 = add(tv1, IrBuilder::create<Val>(1.0)); // 4

  // Merge
  auto tv7 = add(tv6, tv5); // 5

  // Maximum expected output groups (can improve overtime):
  //  {0}, {1}, {2}, {3,4,5}
  //  without final merge would have been {0}, {1}, {2}, {3,4}, {5}

  fusion.addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2, 2, 2}, options);
  at::Tensor t1 = at::randn({2}, options);

  auto fusion_segments = fusion.segment({t0, t1});
  NVF_CHECK(fusion_segments->groups().size() <= 4);
}

TEST_F(NVFuserTest, FusionDAGScalarMerging_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto i0 = IrBuilder::create<Val>(DataType::Double);

  fusion->addInput(tv0);
  fusion->addInput(i0);

  auto i1 = add(i0, IrBuilder::create<Val>(1.0));
  auto i2 = mul(i1, i1);
  auto i3 = add(i2, i1);

  // Branch 0
  auto tv1 = sum(tv0, {0}); // 0
  auto tv2 = add(tv1, i2);
  // Branch 1
  auto tv3 = sum(tv2, {0}); // 1
  auto tv4 = add(tv3, i3);

  auto tv5 = add(tv4, i0);

  fusion->addOutput(tv5);

  FusionExecutorCache executor_cache(std::move(fusion));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16, 16}, options);
  double s0 = 0.5;

  auto outputs = executor_cache.runFusionWithInputs({t0, s0});

  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()->isSegmented(),
      "segmentation didn't happen");
  NVF_CHECK(
      executor_cache.getMostRecentKernelRuntime()
              ->fusionSegments()
              ->groups()
              .size() == 2,
      "segmentation didn't happen as expected");

  testValidate(executor_cache.fusion(), outputs, {t0, s0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBlockReduceInSerialLoop_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;
  constexpr int K = 20;

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = sum(tv0, {{1, 2}});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N, K}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBlockWelfordInSerialLoop_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  constexpr int M = 10;
  constexpr int N = 20;
  constexpr int K = 20;

  auto tv0 = makeSymbolicTensor(3);
  auto tvs = Welford(tv0, {{1, 2}});
  fusion.addInput(tv0);
  auto tv_avg = tvs.avg;
  auto tv_M2 = tvs.var_sum;
  fusion.addOutput(tv_avg);
  fusion.addOutput(tv_M2);

  tv_avg->axis(-1)->parallelize(ParallelType::TIDx);
  tv_avg->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({M, N, K}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  at::Tensor aten_avg = t0.mean({1, 2});
  at::Tensor aten_M2 = t0.var({1, 2}, false) * N * K;
  testValidate(&fusion, outputs, {t0}, {aten_avg, aten_M2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionReductionPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto tv2 = tv0->cacheAfter();

  const int bdimx = 128;
  tv1->split(1, bdimx);
  tv1->split(1, 4);
  tv1->split(1, 1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(2)->parallelize(ParallelType::Unroll);
  tv1->split(0, 10);
  tv0->computeAt(tv1, 4);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 650;
  int numel_y = 102;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn({numel_x, numel_y}, options);
  at::Tensor cg_output = at::empty({numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {input});
  ke.run({input}, {cg_output});

  auto aten_output = input.to(at::kDouble).sum({0});

  testValidate(
      &fusion, {cg_output}, {input}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue728_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addOutput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addOutput(tv1);
  auto tv2 = makeSymbolicTensor(1);
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv3, tv1);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  auto tv6 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  // tv0 -> tv3 -+
  // tv1 --------+-> tv4 -> tv5
  //
  // tv2 -> tv6

  auto all_vals_under_tv3 =
      DependencyCheck::getAllValsBetween({tv3}, fusion.outputs());
  std::unordered_set<Val*> included_tensors({tv3, tv4, tv5});
  for (auto tv : included_tensors) {
    NVF_CHECK(
        std::find(all_vals_under_tv3.begin(), all_vals_under_tv3.end(), tv) !=
            all_vals_under_tv3.end(),
        "TV",
        tv->name(),
        " not found");
  }
  for (auto tv : ir_utils::filterByType<TensorView>(fusion.vals())) {
    if (included_tensors.find(tv) == included_tensors.end()) {
      NVF_CHECK(
          std::find(all_vals_under_tv3.begin(), all_vals_under_tv3.end(), tv) ==
              all_vals_under_tv3.end(),
          "TV",
          tv->name(),
          " should not be found");
    }
  }

  auto no_dependency = DependencyCheck::getAllValsBetween({}, fusion.outputs());
  NVF_CHECK(no_dependency.empty(), "No val should be returned");

  auto no_dep_path = DependencyCheck::getAllValsBetween({tv0, tv1}, {tv6});
  NVF_CHECK(no_dep_path.empty(), "No val should be returned");

  auto no_dep_path2 = DependencyCheck::getAllValsBetween({tv2}, {tv5});
  NVF_CHECK(no_dep_path2.empty(), "No val should be returned");

  auto just_tv3 = DependencyCheck::getAllValsBetween({tv3}, {tv3});
  NVF_CHECK(
      just_tv3.size() == 1 && *(just_tv3.begin()) == tv3,
      "Only tv3 should be included");
}

TEST_F(NVFuserTest, FusionIssue757_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv1->computeAt(tv4, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 650;
  int numel_y = 102;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t3 = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto outputs = ke.run({t0, t3});

  testValidate(&fusion, outputs, {t0, t3}, __LINE__, __FILE__);
}

// See issue #759
TEST_F(NVFuserTest, FusionPredicatedBlockBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = makeSymbolicTensor(2);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->split(0, 4);
  tv1->computeAt(tv4, -1);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  int numel_x = 100;
  int numel_y = 101;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);
  at::Tensor t3 = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto outputs = ke.run({t0, t3});

  testValidate(&fusion, outputs, {t0, t3}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmentVerticalMerge_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);
  // {first kernel}
  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv1, tv0);
  auto tv3 = sum(tv2, {0});
  auto tv4 = add(tv3, tv0);
  auto tv5 = sum(tv4, {0});
  auto tv6 = sum(tv5, {0});
  // {second kernel}
  auto tv7 = add(tv6, tv5);
  auto tv8 = add(tv7, tv5);
  auto tv9 = sum(tv8, {0});

  fusion->addOutput(tv9);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), {t0}, segment_options);

  NVF_CHECK(segmented_fusion->groups().size() == 2);
}

TEST_F(NVFuserTest, FusionSegmentHorizontalMerge_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);
  auto i0 = IrBuilder::create<Val>(DataType::Double);

  fusion->addInput(tv0);
  fusion->addInput(i0);

  // Branch 0 {first kernel}
  auto tv1 = sum(tv0, {0});
  auto tv2 = add(tv0, i0);
  auto tv3 = unaryOp(UnaryOpType::Rsqrt, tv2);
  auto tv4 = sum(tv3, {0});

  // Branch 1 {first kernel}
  auto tv5 = unaryOp(UnaryOpType::Rsqrt, tv3);
  auto tv6 = sum(tv5, {0});

  // Incompatible {second kernel}
  auto tv7 = sum(tv6, {0});

  fusion->addOutput(tv1);
  fusion->addOutput(tv4);
  fusion->addOutput(tv7);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), {t0, 1.0}, segment_options);

  NVF_CHECK(segmented_fusion->groups().size() == 2);
}

TEST_F(NVFuserTest, FusionSegmentMixReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  // def of tv1 in kernel 1 through horizontal
  auto tv1 = sum(tv0, {0, 1});
  // kernel 2
  auto tv2 = sum(tv0, {2});
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv0, tv3);
  auto tv5 = sum(tv4, {2});
  // end of kernel 2
  // kernel 1
  auto tv6 = unaryOp(UnaryOpType::Rsqrt, tv0);
  auto tv7 = sum(tv6, {0, 1});
  auto tv8 = sum(tv6, {0, 1});

  fusion->addOutput(tv1);
  fusion->addOutput(tv5);
  fusion->addOutput(tv7);
  fusion->addOutput(tv8);

  SegmentCandidateFinderOptions segment_options;
  segment_options.run_herrmann_merge = false;
  segment_options.run_final_merge = false;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 2}, options);

  auto segmented_fusion =
      SegmentCandidateFinder::segment(fusion.get(), {t0}, segment_options);

  NVF_CHECK(segmented_fusion->groups().size() <= 2);
}

TEST_F(NVFuserTest, FusionSBAR_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // N, H, W, C format
  std::vector<int64_t> input_shape{656, 7, 7, 64};

  auto x = makeContigTensor(4);
  auto y = makeContigTensor(4);
  auto weight = makeContigTensor(1);
  auto bias = makeContigTensor(1);

  fusion.addInput(x);
  fusion.addInput(y);
  fusion.addInput(weight);
  fusion.addInput(bias);

  const size_t kNumberOfDims = x->nDims();
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  for (const auto axis : arange(kNumberOfDims - 1)) {
    broadcast_mask[axis] = true;
  }

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto scale = mul(x, weight_bcast);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto scale_bias = add(scale, bias_bcast);
  auto scale_bias_add = add(scale_bias, y);
  auto scale_bias_add_relu = unaryOp(UnaryOpType::Relu, scale_bias_add);

  fusion.addOutput(scale_bias_add_relu);

  // inputs
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[3]}, options);
  at::Tensor at_bias = at::zeros({input_shape[3]}, options);

  // inputs
  KernelArgumentHolder inputs = {at_x, at_y, at_weight, at_bias};

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, inputs).outputs;
  testValidate(&fusion, cg_outputs, inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSingleElement_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(0);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(2.5));

  auto tv2 = add(tv1, IrBuilder::create<Val>(3.5));
  fusion.addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({}, options);
  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {t0}).outputs;
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBNBackwardRepro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 4;
  int c = 4;
  int h = 4;
  int w = 4;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);
  auto save_mean = makeSymbolicTensor(1);
  fusion.addInput(save_mean);
  auto save_invstd = makeSymbolicTensor(1);
  fusion.addInput(save_invstd);

  auto grad_out_prev = makeSymbolicTensor(numDims);
  fusion.addInput(grad_out_prev);
  auto gt_0 =
      makeSymbolicTensor(numDims); // single tensor broadcasted is dangerous.
  fusion.addInput(gt_0);

  auto gt_bool = binaryOp(BinaryOpType::GT, gt_0, IrBuilder::create<Val>(1L));
  auto gt_float = castOp(DataType::Float, gt_bool);

  auto grad_out = mul(grad_out_prev, gt_float);

  Val* eps_ptr = IrBuilder::create<Val>(1e-5);

  auto grads = batch_norm_backward(
      input,
      grad_out,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      true,
      eps_ptr,
      {true, true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);
  fusion.addOutput(grads.grad_bias);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({batch, c, h, w}, options);
  at::Tensor t1 = at::randn({c}, options);
  at::Tensor t2 = at::randn_like(t1);
  at::Tensor t3 = at::randn_like(t1);
  at::Tensor t4 = at::randn_like(t1);
  at::Tensor t5 = at::randn_like(t1);
  at::Tensor t6 = at::randn_like(t0);
  at::Tensor t7 = at::randn_like(t0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  KernelArgumentHolder args = {t0, t1, t2, t3, t4, t5, t6, t7};
  auto outputs = executor_cache.runFusionWithInputs(args);
}

// TODO: We only changed inputs, merge this with the test above.
TEST_F(NVFuserTest, FusionBNBackwardRepro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 2;
  int c = 81;
  int h = 1;
  int w = 1;

  auto input = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);
  auto save_mean = makeSymbolicTensor(1);
  fusion.addInput(save_mean);
  auto save_invstd = makeSymbolicTensor(1);
  fusion.addInput(save_invstd);

  auto grad_out_prev = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(grad_out_prev);
  auto gt_0 = makeConcreteTensor({-1, -1, 1, 1});
  fusion.addInput(gt_0);

  auto gt_bool = binaryOp(BinaryOpType::GT, gt_0, IrBuilder::create<Val>(1L));
  auto gt_float = castOp(DataType::Float, gt_bool);

  auto grad_out = mul(grad_out_prev, gt_float);

  Val* eps_ptr = IrBuilder::create<Val>(1e-5);

  auto grads = batch_norm_backward(
      input,
      grad_out,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      true,
      eps_ptr,
      {true, true, true});

  fusion.addOutput(grads.grad_input);
  fusion.addOutput(grads.grad_weight);
  fusion.addOutput(grads.grad_bias);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({batch, c, h, w}, options);
  at::Tensor t1 = at::randn({c}, options);
  at::Tensor t2 = at::randn_like(t1);
  at::Tensor t3 = at::randn_like(t1);
  at::Tensor t4 = at::randn_like(t1);
  at::Tensor t5 = at::randn_like(t1);
  at::Tensor t6 = at::randn_like(t0);
  at::Tensor t7 = at::randn_like(t0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  KernelArgumentHolder args = {t0, t1, t2, t3, t4, t5, t6, t7};
  auto outputs = executor_cache.runFusionWithInputs(args);
}

TEST_F(NVFuserTest, FusionBNRepro_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  int batch = 14;
  int c = 65;
  int h = 7;
  int w = 7;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);
  auto weight = makeSymbolicTensor(1);
  fusion.addInput(weight);
  auto bias = makeSymbolicTensor(1);
  fusion.addInput(bias);
  auto running_mean = makeSymbolicTensor(1);
  fusion.addInput(running_mean);
  auto running_var = makeSymbolicTensor(1);
  fusion.addInput(running_var);

  auto momentum_ptr = IrBuilder::create<Val>(kMomentum);
  auto eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({batch, c, h, w}, options);
  at::Tensor t2 = at::randn({c}, options);
  at::Tensor t3 = at::randn_like(t2);
  at::Tensor t4 = at::randn_like(t2);
  at::Tensor t5 = at::randn_like(t2);

  auto t1_ref = t1.clone();
  auto t2_ref = t2.clone();
  auto t3_ref = t3.clone();
  auto t4_ref = t4.clone();
  auto t5_ref = t5.clone();

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1, t2, t3, t4, t5});

  auto at_results = at::native_batch_norm(
      t1_ref, t2_ref, t3_ref, t4_ref, t5_ref, kTraining, kMomentum, kEps);

  auto at_output = std::get<0>(at_results);
  auto at_mean = std::get<1>(at_results);
  auto at_invstd = std::get<2>(at_results);

  std::vector<at::Tensor> aten_outputs = {at_output, at_mean, at_invstd};

  testValidate(
      &fusion,
      cg_outputs,
      {t1, t2, t3, t4, t5},
      aten_outputs,
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionBNRepro2_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  int batch = 2;
  int c = 4;
  int h = 17;
  int w = 17;
  int numDims = 4;

  auto input = makeSymbolicTensor(numDims);
  fusion.addInput(input);

  Val* momentum_ptr = IrBuilder::create<Val>(kMomentum);
  Val* eps_ptr = IrBuilder::create<Val>(kEps);

  auto result = batch_norm(
      input,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      kTraining,
      momentum_ptr,
      eps_ptr);

  fusion.addOutput(result.output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({batch, c, h, w}, options);

  auto t1_ref = t1.clone();
  at::Tensor r_m;
  at::Tensor r_v;
  at::Tensor weight;
  at::Tensor bias;

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto cg_outputs = executor_cache.runFusionWithInputs({t1});

  testValidate(&fusion, cg_outputs, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionZeroSizeTensorPW_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(2.5));
  fusion.addOutput(tv2);

  // This test used to just have:
  // auto tv3 = makeConcreteTensor({0});
  // and somehow that was running through our system fine, but size-0 tensors
  // are not supported, so making sure this fails.
  auto tv3 = set(tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2}, options);
  at::Tensor t1 = at::randn({0}, options);

  // Fails at schedule pointwise because our (maybe only) size-0 check is in
  // binding input sizes which the scheduler ends up calling.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(scheduleAndRun(&fusion, SchedulerType::PointWise, {t0, t1}));
}

TEST_F(NVFuserTest, FusionZeroSizeTensorReduction_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {1});
  fusion.addOutput(tv2);

  auto tv3 = makeConcreteTensor({0});
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({0}, options);
  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::Reduction, {t0, t1}, false);
  auto t2 = t0.sum({1});
  at::Tensor t3 = at::empty({0}, options);

  testValidate(
      &fusion,
      cg_results.outputs,
      {t0, t1},
      {t2, t3},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionZeroSizeTensorNormalization_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = makeConcreteTensor({0});
  fusion.addInput(tv1);

  auto tv2 = sum(tv0, {0});
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv0, tv3);
  fusion.addOutput(tv4);

  auto tv5 = makeConcreteTensor({0});
  fusion.addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({0}, options);

  auto cg_results =
      scheduleAndRun(&fusion, SchedulerType::OuterPersistent, {t0, t1}, false);
  auto t2 = t0.sum({0}).add(t0);
  at::Tensor t3 = at::empty({0}, options);

  testValidate(
      &fusion,
      cg_results.outputs,
      {t0, t1},
      {t2, t3},
      __LINE__,
      __FILE__,
      "",
      cg_results.heuristic_params->lparams);
}

TEST_F(NVFuserTest, FusionWelford1Output_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  fusion->addOutput(tvs.var_sum);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({128, 65}, options);
  auto outputs = executor_cache.runFusionWithInputs({t0});

  auto t1 = t0.var({1}, false) * 65;
  testValidate(fusion, outputs, {t0}, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTranslate1Welford_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs = Welford(tv0, {1});
  auto tv_out = add(tv0, broadcast(tvs.avg, {false, true}));
  fusion->addOutput(tv_out);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});
    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    testValidate(
        fusion,
        outputs,
        {t0},
        {t0.add(t0.mean({1}).unsqueeze(1))},
        __LINE__,
        __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  NVF_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 2);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);

  bool found_welford = false;
  for (auto group : runtime2->fusionSegments()->groups()) {
    for (auto expr : group->exprs()) {
      if (expr->isA<WelfordOp>()) {
        found_welford = true;
      }
    }
  }
  NVF_CHECK(found_welford);
}

TEST_F(NVFuserTest, FusionTranslate2Welford_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto tv_out1 = add(tv0, broadcast(tvs1.avg, {false, true}));
  fusion->addOutput(tv_out1);

  auto tvs2 = Welford(tv0, {1});
  auto tv_out2 = add(tv0, broadcast(tvs2.avg, {false, true}));
  fusion->addOutput(tv_out2);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    // Square sums does not fit well in the testValidate assumptions,
    //  so we just compare the divided output here.
    auto out = t0.add(t0.mean({1}).unsqueeze(1));
    testValidate(fusion, outputs, {t0}, {out, out}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  // Run a translated welford
  auto runtime1 = run_test(64);
  // Check it was translated
  NVF_CHECK(
      runtime1->fusionSegments()->groups().size() == 1 &&
      runtime1->fusionSegments()->groups()[0]->exprs().size() > 4);

  // Run an un-translated welford
  auto runtime2 = run_test(65536);
  // // Check it was not translated
  bool found_welford = false;
  for (auto group : runtime2->fusionSegments()->groups()) {
    for (auto expr : group->exprs()) {
      if (expr->isA<WelfordOp>()) {
        found_welford = true;
      }
    }
  }
  NVF_CHECK(found_welford);
}

TEST_F(NVFuserTest, FusionLargeWelfordNormalization_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});

  fusion->addOutput(tvs1.var_sum);
  fusion->addOutput(sum_of_tv0);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.var({1}, false) * inner_size;
    auto t2 = t0.sum({1});
    testValidate(fusion, outputs, {t0}, {t1, t2}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  auto runtime = run_test(65536);
  NVF_CHECK(!runtime->isSegmented());
}

TEST_F(NVFuserTest, FusionWelfordOuterPersistence_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tvs1 = Welford(tv0, {1});
  auto sum_of_tv0 = sum(tv0, {1});
  auto sum_bcasted = broadcast(sum_of_tv0, {false, true});
  auto avg_bcasted = broadcast(tvs1.avg, {false, true});
  auto tv0_plus_sum = add(tv0, sum_bcasted);
  auto tv0_plus_avg = add(tv0, avg_bcasted);

  fusion->addOutput(tv0_plus_sum);
  fusion->addOutput(tv0_plus_avg);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto run_test = [&executor_cache,
                   fusion](auto inner_size) -> FusionKernelRuntime* {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({128, inner_size}, options);
    auto outputs = executor_cache.runFusionWithInputs({t0});

    auto t1 = t0.to(c10::kDouble).mean({1}).unsqueeze(1) + t0;
    auto t2 = t0.to(c10::kDouble).sum({1}).unsqueeze(1) + t0;
    testValidate(fusion, outputs, {t0}, {t2, t1}, __LINE__, __FILE__);

    return executor_cache.getMostRecentKernelRuntime();
  };

  for (auto inner_size : {4096, 8192, 32768}) {
    auto runtime = run_test(inner_size);
    NVF_CHECK(!runtime->isSegmented());
  }
}

TEST_F(NVFuserTest, FusionSegmentIslands_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = sum(tv0, {0});
  auto tv3 = sum(tv1, {1});
  fusion->addOutput(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16}, options);
  at::Tensor t1 = at::randn({16, 16}, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  executor_cache.runFusionWithInputs({t0, t1});
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  auto tv1 = makeSymbolicTensor(2);
  auto tv2 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv3 = broadcast(tv0, {false, true, true, true});
  auto tv4 = broadcast(tv1, {false, false, true, true});
  auto tv5 = unaryOp(UnaryOpType::Rsqrt, tv2);

  auto tv6 = add(tv3, tv5);
  auto tv7 = add(tv4, tv5);
  auto tv8 = add(tv3, tv4);

  auto tv9 = add(tv6, tv7);
  auto tv10 = add(tv9, tv8);

  fusion->addOutput(tv10);

  tv0->computeAt(tv10, -2);
  tv1->computeAt(tv10, -2);
  tv2->computeAt(tv10, -2);

  NVF_CHECK(tv3->getComputeAtPosition() == 1);
  NVF_CHECK(tv4->getComputeAtPosition() == 2);
  NVF_CHECK(tv5->getComputeAtPosition() == 3);

  NVF_CHECK(tv6->getMaxProducerPosition() == 3);
  NVF_CHECK(tv7->getMaxProducerPosition() == 3);
  NVF_CHECK(tv8->getMaxProducerPosition() == 2);
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(3);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, false, true});
  auto tv3 = add(tv2, tv1);

  fusion->addOutput(tv3);
  tv3->split(-2, 4);
  tv3->reorder({{-1, -2}});
  tv0->computeAt(tv3, -2);
  tv1->computeAt(tv3, -2);
  NVF_CHECK(tv2->getComputeAtPosition() == 2);
  NVF_CHECK(tv3->getMaxProducerPosition() == 2);
}

TEST_F(NVFuserTest, FusionBackOffInnerBroadcast3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = makeSymbolicTensor(4);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, false, true});
  auto tv3 = broadcast(tv2, {false, true, false, false});
  auto tv4 = add(tv3, tv1);

  fusion->addOutput(tv4);
  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);
  NVF_CHECK(tv2->getComputeAtPosition() == 2);
  NVF_CHECK(tv3->getMaxProducerPosition() == 3);
}

TEST_F(NVFuserTest, FusionSimpleWarp_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  tv1->split(1, 32);
  auto tv1_rf = tv1->rFactor({1});
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 128}, options);

  auto at_output = t1.sum({1}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});

  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSimpleWarpPad_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->padToMultipleOfWarp(32);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp(32);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(-1)->padToMultipleOfWarp(32);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->padToMultipleOfWarp(32);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->padToMultipleOfWarp(32);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->padToMultipleOfWarp(32);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 127}, options);

  auto at_output = t1.sum({1}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpPadMergeSplit_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  auto tv2 = broadcast(tv1, {false, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 17, 128}, options);

  auto at_output = t1.sum({1, 2}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSerialWarpReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(3);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  auto tv2 = broadcast(tv1, {false, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 17, 128}, options);

  auto at_output = t1.sum({1, 2}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTrivialWarpReduction_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeConcreteTensor({17, 18, 128, 1});

  fusion->addInput(tv0);
  auto tv1 = reductionOpRaw(
      BinaryOpType::Add, {1, 2, 3}, IrBuilder::create<Val>(0.0), tv0);
  auto tv2 = broadcast(tv1, {false, true, true, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->merge(1);
  tv1->split(1, 8, false);

  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);
  tv1->axis(-2)->padToMultipleOfWarp();
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-2)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-2)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDx);
  tv3->axis(-2)->parallelize(ParallelType::TIDx);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({17, 18, 128, 1}, options);

  auto at_output = t1.sum({1, 2, 3}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionMultipleDimBinding_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  auto tv_add = makeSymbolicTensor(2);

  fusion->addInput(tv0);
  fusion->addInput(tv_add);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  auto tv4 = add(tv0, tv_add);

  fusion->addOutput(tv3);
  fusion->addOutput(tv4);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  auto tv1_rf = tv1->rFactor({1});
  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1_rf->axis(-1)->padToMultipleOfWarp(32);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp(32);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(-1)->padToMultipleOfWarp(32);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(-1)->padToMultipleOfWarp(32);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->padToMultipleOfWarp(32);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->padToMultipleOfWarp(32);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-1)->padToMultipleOfWarp(64);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 128}, options);
  at::Tensor t2 = at::randn({16, 128}, options);

  auto at_output = t1.sum({1}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1, t2});
  auto outputs = ke.run({t1, t2});
  testValidate(
      fusion.get(),
      outputs,
      {t1, t2},
      {at_output, t1 + t2},
      __LINE__,
      __FILE__);
}

TEST_F(NVFuserTest, FusionPadNoWarpReduce_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv1->axis(0)->parallelize(ParallelType::TIDy);
  tv2->axis(0)->parallelize(ParallelType::TIDy);
  tv3->axis(0)->parallelize(ParallelType::TIDy);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 31}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpMutipleThreadDim_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  fusion->addOutput(tv2);

  tv2->split(1, 8);
  auto tv2_rf = tv2->rFactor({-1});
  tv2_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv2_rf->axis(-1)->padToMultipleOfWarp();

  TransformPropagatorWithCheck propagator(tv2_rf);
  MaxLogicalDomainInfoSpanningTree(tv2_rf).traverse(&propagator);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::BIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDy);
  tv0->computeAt(tv2, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 31}, options);

  auto at_output = (t1 + 1).sum({1});

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionWarpReduceUnrollOuterLoop_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(2);

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);

  fusion->addOutput(tv3);

  // Schedule a persistent kernel
  auto tv0_cache = tv0->cacheAfter();
  tv1->split(1, 8, false);
  tv1->split(0, 4);
  auto tv1_rf = tv1->rFactor({2});

  tv1_rf->axis(0)->parallelize(ParallelType::BIDx);
  tv1_rf->axis(1)->parallelize(ParallelType::Unroll);
  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->padToMultipleOfWarp();
  tv1->axis(1)->parallelize(ParallelType::Unroll);
  TransformPropagatorWithCheck propagator(tv1_rf);
  MaxLogicalDomainInfoSpanningTree(tv1_rf).traverse(&propagator);
  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv0->axis(1)->parallelize(ParallelType::Unroll);
  tv0_cache->axis(-1)->parallelize(ParallelType::TIDx);
  tv0_cache->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::Unroll);

  tv0->computeAt(tv3, -1, ComputeAtMode::MostInlined);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({16, 128}, options);

  auto at_output = t1.sum({1}, true).add(t1);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(fusion.get(), outputs, {t1}, {at_output}, __LINE__, __FILE__);
}

// Repro of issue #1579
TEST_F(NVFuserTest, FusionWarpReducePredication_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> shape1 = {1024};
  std::vector<int64_t> shape2 = {50};

  auto tv0 = makeConcreteTensor(shape1);
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion.addOutput(tv1);

  auto tv2 = makeConcreteTensor(shape2);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  auto tv4 = sum(tv3, {0});
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  // Just to fill the smem buffer by a thread block of 1024 threads
  // with some values
  tv1->axis(-1)->parallelize(ParallelType::TIDx);

  // Make the tv4_rf reduction a warp reduction to trigger the
  // bug. Since the smem buffer is filled with some values due to the
  // reduction of tv1, those values would be used by predicated-out
  // threads.
  tv4->split(-1, 10);
  auto tv4_rf = tv4->rFactor({-1});
  tv4_rf->axis(-1)->parallelize(ParallelType::TIDx);
  tv4_rf->axis(-1)->padToMultipleOfWarp();

  tv4_rf->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t2});
  auto cg_outputs = ke.run({t0, t2});

  auto t1 = t0.sum({0});
  auto t4 = (t2 + 1).sum({0}) + 1;

  testValidate(&fusion, cg_outputs, {t0, t2}, {t1, t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegfaultReduction_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  int batch = 2;
  int c = 1;
  int h = 1;
  int w = 1;
  int numDims = 4;

  auto input = makeConcreteTensor({-1, 1, 1, 1});
  fusion.addInput(input);
  auto bcast_bias = makeConcreteTensor({-1, 1, 1, 1});
  fusion.addInput(bcast_bias);

  std::vector<int64_t> at_sum_axes;
  std::vector<int64_t> outer_reduction_axes;
  std::vector<bool> outer_broadcast_mask(numDims, false);
  Val* N = IrBuilder::create<Val>(1.0);
  for (const auto axis : arange(numDims)) {
    if (axis != 1) {
      outer_reduction_axes.push_back(axis);
      at_sum_axes.push_back(axis);
      outer_broadcast_mask[axis] = true;
      N = mul(N, input->getLoopDomain()[axis]->extent());
    }
  }

  auto output0 = mul(input, bcast_bias);
  fusion.addOutput(output0);
  auto output1 = sum(output0, outer_reduction_axes);
  fusion.addOutput(output1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({batch, c, h, w}, options);
  at::Tensor t1 = at::randn({batch, c, h, w}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto outputs = executor_cache.runFusionWithInputs({t0, t1});
  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseBroadCastMultiVisit_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Val>(2.0));
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv3, tv1);
  auto tv5 = mul(tv4, IrBuilder::create<Val>(3.0));
  fusion->addOutput(tv5);

  // t4 cannot inner re-use t2, because there's a broadcast
  //  between them.
  tv0->computeAt(tv5, 1, ComputeAtMode::BestEffort);
  tv3->computeAt(tv5, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);

  KernelExecutor ke;
  ke.compile(fusion, {in0, in1});
  auto outputs = ke.run({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseStressTest_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Val>(2.0));
  auto tv3 = mul(tv0, IrBuilder::create<Val>(3.0));
  auto tv4 = mul(tv2, tv3);
  // Broadcast buffer can be reused through outer sharing
  auto tv5 = broadcast(tv4, {true, false, false});
  auto tv6 = mul(tv5, IrBuilder::create<Val>(5.0));
  auto tv7 = mul(tv6, tv1);
  mul(tv7, IrBuilder::create<Val>(7.0));
  // tv9 shouldn't alias to avoid buffer over-subscription
  auto tv9 = broadcast(tv4, {true, false, false});
  mul(tv9, IrBuilder::create<Val>(9.0));
  auto tv11 = add(tv5, tv9);
  fusion->addOutput(tv7);
  fusion->addOutput(tv11);

  tv0->computeAt(tv5, 1, ComputeAtMode::BestEffort);
  tv0->computeAt(tv9, 1, ComputeAtMode::BestEffort);

  tv5->computeAt(tv7, 1, ComputeAtMode::BestEffort);
  tv5->computeAt(tv11, 1, ComputeAtMode::BestEffort);
  tv9->computeAt(tv11, 1, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);

  KernelExecutor ke;
  ke.compile(fusion, {in0, in1});

  auto outputs = ke.run({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseLargeBuffer_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({256, 512});

  fusion->addInput(tv0);

  auto tv1 = mul(tv0, IrBuilder::create<Val>(2.0));
  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = mul(tv2, IrBuilder::create<Val>(2.0));
  auto tv4 = mul(tv3, IrBuilder::create<Val>(2.0));
  auto tv5 = mul(tv4, IrBuilder::create<Val>(2.0));
  auto tv6 = mul(tv5, IrBuilder::create<Val>(2.0));

  fusion->addOutput(tv6);

  tv0->computeAt(tv6, 1, ComputeAtMode::BestEffort);
  tv6->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({256, 512}, options);

  KernelExecutor ke;
  ke.compile(fusion, {in0});
  auto outputs = ke.run({in0});

  testValidate(fusion, outputs, {in0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseNo2hop_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Val>(2.0));
  auto tv3 = broadcast(tv2, {false, false, true});
  auto tv4 = add(tv3, tv1); // T4 to be inner aliased first, and
                            //  shouldn't outer alias on top
  auto tv5 = mul(tv4, IrBuilder::create<Val>(3.0));
  auto tv6 = mul(tv5, IrBuilder::create<Val>(3.0));
  fusion->addOutput(tv6);

  tv0->computeAt(tv6, 1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv6, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);
  KernelExecutor ke;
  ke.compile(fusion, {in0, in1});
  auto outputs = ke.run({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseAllocationOrder_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({3, 3, 3});

  fusion->addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = mul(tv2, IrBuilder::create<Val>(2.0));

  fusion->addOutput(tv3);

  // In this case tv1 "reuses" allocation of tv2
  //  due to the switched allocation order
  tv1->computeAt(tv2, 1, ComputeAtMode::BestEffort);

  tv0->axis(0)->parallelize(ParallelType::TIDx);
  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({3, 3, 3}, options);

  KernelExecutor ke;
  ke.compile(fusion, {in0});
  auto outputs = ke.run({in0});

  testValidate(fusion, outputs, {in0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseLiveInterval_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({16, 16});

  fusion->addInput(tv0);

  auto tv1 = mul(tv0, IrBuilder::create<Val>(3.0));
  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  auto tv3 = mul(tv2, IrBuilder::create<Val>(2.0));
  // tv1 used till here, cannot be reused by tv2 or tv3
  auto tv4 = mul(tv3, tv1);

  fusion->addOutput(tv4);

  tv0->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({16, 16}, options);

  KernelExecutor ke;
  ke.compile(fusion, {in0});
  auto cg_outputs = ke.run({in0});

  testValidate(fusion, cg_outputs, {in0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionBufferReuseNoAcrossBroadcast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeConcreteTensor({2, 2});
  auto tv1 = makeConcreteTensor({2, 2, 2});

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  auto tv2 = mul(tv0, IrBuilder::create<Val>(2.0));
  auto tv3 = mul(tv0, IrBuilder::create<Val>(3.0));
  auto tv4 = mul(tv2, tv3);
  auto tv5 = broadcast(tv4, {false, false, true});
  auto tv6 = mul(tv5, tv1);
  auto tv7 = mul(tv6, IrBuilder::create<Val>(7.0));
  fusion->addOutput(tv7);

  // tv6 shouldn't re-use t2 or t3 because of
  //  the broadcast in between
  tv0->computeAt(tv4, 1, ComputeAtMode::BestEffort);
  tv4->computeAt(tv7, 2, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto in0 = at::randn({2, 2}, options);
  auto in1 = at::randn({2, 2, 2}, options);
  KernelExecutor ke;
  ke.compile(fusion, {in0, in1});
  auto outputs = ke.run({in0, in1});

  testValidate(fusion, outputs, {in0, in1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue970_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int nelm = 10;

  // tv3 = tv0 + sum(tv0)
  auto tv0 = makeConcreteTensor({nelm, nelm});
  fusion.addInput(tv0);
  auto tv1 = sum(tv0, {1});
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv2, tv0);
  fusion.addOutput(tv3);

  tv1->split(1, 4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nelm, nelm}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Reproducer of #1016
TEST_F(NVFuserTest, FusionIssue1016_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(2.0));

  fusion.addOutput(tv2);

  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 8);

  int numel_x = 10;
  int numel_y = 11;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({numel_x, numel_y}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Reproducer of #1021
TEST_F(NVFuserTest, FusionIssue1021_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = broadcast(tv1, {false, true});
  fusion.addOutput(tv2);

  tv2->cacheBefore();

  tv2->split(0, 2);

  tv1->computeAt(tv2, 1);

  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Reproducer of issue #1053
TEST_F(NVFuserTest, FusionNonUniqueThreadDim_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = sum(tv0, {0});
  fusion->addOutput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv2);

  tv1->split(0, 8);
  auto tv1_rf = tv1->rFactor({-1});

  tv1_rf->computeAt(tv1, 1);

  tv1_rf->axis(-1)->parallelize(ParallelType::TIDx);

  tv2->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({32}, options);

  auto at_tv1 = (t1).sum({0});
  auto at_tv2 = t1 + 1;

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});
  testValidate(
      fusion.get(), outputs, {t1}, {at_tv1, at_tv2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap1_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv1);
  fusion->addOutput(tv2);

  tv1->split(0, 8, false);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->split(0, 8, false);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  // The extents of tv1 and tv2 axes are equal even though their
  // actual values are not statically known
  GpuLower gpulw(fusion.get());
  gpulw.run();
  const auto& pdmap = gpulw.parallelDimensionMap();

  NVF_CHECK(pdmap.isExact(ParallelType::TIDx));
  NVF_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({32}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});

  testValidate(fusion.get(), outputs, {t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap2_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion->addOutput(tv3);

  tv3->split(-1, 8, false);
  tv2->computeAt(tv3, -1);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  GpuLower gpulw(fusion.get());
  gpulw.run();
  const auto& pdmap = gpulw.parallelDimensionMap();
  NVF_CHECK(pdmap.isExact(ParallelType::TIDx));
  NVF_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({11}, options);
  at::Tensor t2 = at::randn({11, 13}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1, t2});
  auto outputs = ke.run({t1, t2});

  testValidate(fusion.get(), outputs, {t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap3_CUDA) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv2);
  auto tv3 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv3);

  tv2->split(0, 10);
  tv3->split(0, 20);

  auto tv4 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv4);
  auto tv5 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion->addOutput(tv5);

  // Not mapped but equal extent
  tv4->split(0, 10);
  tv5->split(0, 10);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  tv4->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);

  GpuLower gpulw(fusion.get());
  gpulw.run();
  const auto& pdmap = gpulw.parallelDimensionMap();
  ASSERT_FALSE(pdmap.isExact(ParallelType::TIDx));
  ASSERT_EQ(pdmap.get(ParallelType::TIDx)->value(), 20);
  ASSERT_TRUE(pdmap.isExact(ParallelType::TIDy));
  ASSERT_EQ(pdmap.get(ParallelType::TIDy)->value(), 10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({13}, options);

  KernelExecutor ke;
  ke.compile(fusion.get(), {t1});
  auto outputs = ke.run({t1});

  testValidate(fusion.get(), outputs, {t1}, __LINE__, __FILE__);
}

// Parallelizing merged broadcast domains
TEST_F(NVFuserTest, FusionParallelDimensionMap4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  tv4->reorder({{1, 2}, {2, 1}});
  tv4->merge(0);
  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  // TIDx is mapped to tv4.axis(0) as well as tv2.axis(0), so it's not
  // exact.
  tv4->axis(0)->parallelize(ParallelType::TIDx);

  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  GpuLower gpulw(&fusion);
  gpulw.run();
  const auto& pdmap = gpulw.parallelDimensionMap();
  NVF_CHECK(!pdmap.isExact(ParallelType::TIDx));
  NVF_CHECK(
      pdmap.get(ParallelType::TIDx)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDx)->as<NamedScalar>()->name() == "blockDim.x");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({13}, options);
  at::Tensor t2 = at::randn({15, 13}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t1, t2});
  auto outputs = ke.run({t1, t2});

  testValidate(&fusion, outputs, {t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionParallelDimensionMap5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  auto tv3 = broadcast(tv0, {false, true});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  tv0->computeAt(tv4, -1);
  tv1->computeAt(tv4, -1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);

  GpuLower gpulw(&fusion);
  gpulw.run();
  const auto& pdmap = gpulw.parallelDimensionMap();
  NVF_CHECK(pdmap.isExact(ParallelType::TIDx));
  NVF_CHECK(pdmap.isExact(ParallelType::TIDy));
  NVF_CHECK(
      pdmap.get(ParallelType::TIDx)->isConst() &&
      pdmap.get(ParallelType::TIDx)->value() == 4);
  NVF_CHECK(
      pdmap.get(ParallelType::TIDy)->isA<NamedScalar>() &&
      pdmap.get(ParallelType::TIDy)->as<NamedScalar>()->name() == "blockDim.y");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t1 = at::randn({13}, options);
  at::Tensor t2 = at::randn({13, 15}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t1, t2});
  auto outputs = ke.run({t1, t2});

  testValidate(&fusion, outputs, {t1, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionSegmenterCombineReductionsCycleRepro_CUDA) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto t0 = makeSymbolicTensor(3, DataType::Float);
  auto t1 = makeSymbolicTensor(3, DataType::Half);
  auto t3 = makeSymbolicTensor(3, DataType::Half);
  auto t5 = makeSymbolicTensor(3, DataType::Half);
  auto t7 = makeSymbolicTensor(1, DataType::Half);
  auto t11 = makeSymbolicTensor(3, DataType::Half);
  auto t13 = makeSymbolicTensor(3, DataType::Half);
  auto t15 = makeSymbolicTensor(3, DataType::Half);
  auto t17 = makeSymbolicTensor(3, DataType::Half);
  auto d56 = IrBuilder::create<Val>(DataType::Double);

  fusion.addInput(t0);
  fusion.addInput(t1);
  fusion.addInput(t3);
  fusion.addInput(t5);
  fusion.addInput(t7);
  fusion.addInput(t11);
  fusion.addInput(t13);
  fusion.addInput(t15);
  fusion.addInput(t17);
  fusion.addInput(d56);

  auto t2 = castOp(DataType::Float, t1);
  auto t4 = castOp(DataType::Float, t3);
  auto t22 = sub(t2, t4);
  auto t6 = castOp(DataType::Float, t5);
  auto t23 = mul(t22, t6);
  auto t16 = castOp(DataType::Float, t15);
  auto t18 = castOp(DataType::Float, t17);
  auto t19 = add(t16, t18);
  auto t14 = castOp(DataType::Float, t13);
  auto t20 = add(t19, t14);
  auto t12 = castOp(DataType::Float, t11);
  auto t21 = add(t20, t12);
  auto t8 = castOp(DataType::Float, t7);
  auto t24 = broadcast(t8, {true, true, false});
  auto t25 = mul(t21, t24);
  auto t27 = sum(t25, {2});
  auto t28 = broadcast(t27, {false, false, true});
  auto t29 = mul(t25, t23);
  auto t30 = sum(t29, {2});
  auto t31 = broadcast(t30, {false, false, true});
  auto d59 =
      mul(t1->getLogicalDomain()[2]->extent(), IrBuilder::create<Val>(1.0));
  auto t26 = mul(d59, t25);
  auto txx = mul(t26, IrBuilder::create<Val>(1.0));
  auto t33 = sub(txx, t28);
  auto d70 = unaryOp(UnaryOpType::Reciprocal, d59);
  auto t35 = mul(d70, t6);
  auto t39 = sum(t21, {0, 1});
  auto t47 = castOp(DataType::Half, t39);
  auto t37 = mul(t21, t23);
  auto t38 = sum(t37, {0, 1});
  auto t46 = castOp(DataType::Half, t38);
  auto t32 = mul(t23, t31);
  auto t34 = sub(t33, t32);
  auto t36 = mul(t35, t34);
  auto t45 = castOp(DataType::Half, t36);
  auto t40 = mul(t36, t0);
  auto t41 = mul(t40, d56);
  auto t44 = castOp(DataType::Half, t41);
  auto t42 = sum(t41, {0, 1});
  auto t43 = castOp(DataType::Half, t42);

  fusion.addOutput(t43);
  fusion.addOutput(t44);
  fusion.addOutput(t45);
  fusion.addOutput(t46);
  fusion.addOutput(t47);

  auto options_half = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::randn({128, 64, 1024}, options_float);
  at::Tensor at_t1 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t3 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t5 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t7 = at::randn({1024}, options_half);
  at::Tensor at_t11 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t13 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t15 = at::randn({128, 64, 1024}, options_half);
  at::Tensor at_t17 = at::randn({128, 64, 1024}, options_half);
  double at_d56 = 1.1111;

  std::vector<at::Tensor> aten_inputs = {
      at_t0, at_t1, at_t3, at_t5, at_t7, at_t11, at_t13, at_t15, at_t17};

  KernelArgumentHolder args;
  args.push(aten_inputs);
  args.push(PolymorphicValue(at_d56));

  for (auto i : arange(5)) {
    (void)i; // Suppress unused variable warning
    auto segmented_fusion =
        SegmentCandidateFinder::segment(fusion_ptr.get(), args);
  }
}

TEST_F(NVFuserTest, FusionSerialAndParallelIndexing_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv4);

  auto tv5 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv6 = add(tv5, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv6);

  // Case 1: local memory tensor computed serially and used by
  // parallel threads
  tv2->split(-1, 4);
  tv1->computeAt(tv2, -2);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  // Case 2: shared memory tensor computed serially and used by BID
  tv4->split(-1, 4);
  tv3->computeAt(tv4, -2);
  tv4->axis(-1)->parallelize(ParallelType::BIDx);
  tv3->setMemoryType(MemoryType::Shared);

  // Case 3: shared memory tensor computed by TID and used by BID
  tv6->split(-1, 4);
  tv5->computeAt(tv6, -2);
  tv6->axis(-1)->parallelize(ParallelType::BIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->setMemoryType(MemoryType::Shared);

  const int nx = 11;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nx}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Repro of issue #1105
TEST_F(NVFuserTest, FusionWARSyncAliasedSmem_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv4);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);
  tv3->setMemoryType(MemoryType::Shared);

  tv4->split(0, 4);
  tv0->computeAt(tv4, 1);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  // Make sure a WAR sync is inserted at the end of the outer loop
  GpuLower gpulw(&fusion);
  for (const auto& kir_node : gpulw.run()->topLevelExprs()) {
    if (auto loop = dynamic_cast<ForLoop*>(kir_node)) {
      const auto& body = loop->body().exprs();
      NVF_CHECK(!body.empty());
      auto last_expr = dynamic_cast<kir::BlockSync*>(body.back());
      NVF_CHECK(
          last_expr != nullptr,
          "Invalid expr found: ",
          body.back()->toString());
      NVF_CHECK(last_expr->isWarHazardSync(), "Not a sync for WAR hazard");
    }
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1099_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(1);
  fusion.addInput(tv3);

  // Just to make TIDx/y/z non-exact
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  auto tv6 = add(tv5, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv6);

  tv2->split(0, 4);
  tv0->computeAt(tv2, 1);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->axis(-1)->parallelize(ParallelType::TIDy);
  tv2->axis(-1)->parallelize(ParallelType::TIDz);
  tv2->axis(0)->parallelize(ParallelType::BIDx);

  tv1->setMemoryType(MemoryType::Shared);

  tv4->split(0, 5);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv4->setMemoryType(MemoryType::Shared);
  tv5->split(0, 6);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->setMemoryType(MemoryType::Shared);
  tv6->split(0, 7);
  tv6->axis(-1)->parallelize(ParallelType::TIDz);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t3 = at::randn({19}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t3});
  auto outputs = ke.run({t0, t3});

  testValidate(&fusion, outputs, {t0, t3}, __LINE__, __FILE__);
}

// Repro of issue #1080
TEST_F(NVFuserTest, FusionUnswitchPredicate_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  tv2->split(0, 4);
  tv0->computeAt(tv2, 2);

  tv2->split(-1, 8);
  tv1->split(-1, 8);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-2)->parallelize(ParallelType::TIDy);

  // swap TIDx and TIDy
  tv1->axis(-1)->parallelize(ParallelType::TIDy);
  tv1->axis(-2)->parallelize(ParallelType::TIDx);

  tv1->setMemoryType(MemoryType::Shared);

  const int nx = 4;
  const int ny = 10;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({nx, ny}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1189_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({16, 16});
  auto tv1 = makeConcreteTensor({16, 16});

  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, false, true});

  fusion.addInput(tv0b);
  fusion.addInput(tv1b);

  auto tv2 = add(tv0b, tv1b);
  auto tv3 = sum(tv2, {1});
  fusion.addOutput(tv3);

  auto parallelize = [](auto tv) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
    tv->axis(1)->parallelize(ParallelType::BIDx);
    tv->axis(2)->parallelize(ParallelType::BIDy);
  };

  parallelize(tv0b);
  parallelize(tv1b);
  parallelize(tv2);
  parallelize(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({16, 16, 1}, options);
  at::Tensor t1 = at::randn({16, 16, 1}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1052_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv2);

  auto tv3 = add(tv1, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv2->axis(-1)->parallelize(ParallelType::TIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(tv2, {tv0});
  scheduler_utils::parallelizeAllLike(tv3, {tv1});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);
  at::Tensor t1 = at::randn({100}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Repro of issue #1115
TEST_F(NVFuserTest, FusionPointwiseBroadcast_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{3, 17, 80};
  std::vector<int64_t> output_shape{3, 17, 1, 80};

  TensorView* x = makeSymbolicTensor(input_shape.size());
  TensorView* bias = makeSymbolicTensor(input_shape.size());
  fusion.addInput(x);
  fusion.addInput(bias);

  auto x_add_bias = add(x, bias);
  auto x_bcast = broadcast(x_add_bias, {false, false, true, false});
  auto y = gelu(x_bcast);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_bias = at::randn(input_shape, options);

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {at_x, at_bias});
  testValidate(
      &fusion, cg_outputs.outputs, {at_x, at_bias}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionPointwiseVectorize_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int size = 1024 * 64;

  TensorView* x = makeContigTensor(1);
  fusion.addInput(x);
  auto y = sin(x);
  fusion.addOutput(y);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  // PyTorch's CUDA caching allocator should always return aligned pointer for
  // freshly allocated tensor
  at::Tensor at_x = at::randn({size}, options);

  SchedulerEntry::scheduleWith(&fusion, SchedulerType::PointWise, {at_x});

  for (auto x_consumer : ir_utils::consumerTvsOf(x)) {
    bool found_vec_in_input = false;
    for (auto id : x_consumer->getLoopDomain()) {
      if (isParallelTypeVectorize(id->getParallelType())) {
        found_vec_in_input = true;
        break;
      }
    }
    NVF_CHECK(found_vec_in_input, "Expect input to be vectorized");
  }

  for (auto id : y->getLoopDomain()) {
    if (isParallelTypeVectorize(id->getParallelType())) {
      return;
    }
  }
  NVF_CHECK(false, "Expect output to be vectorized");
}

TEST_F(NVFuserTest, FusionSmemAliasSerial_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  // Just set the dimension of TIDx
  auto tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // tv1 and tv2 are on shared memory and are not parallelized with
  // TIDx. They should be predicated as they are redundant and can
  // interfere with smem aliasing (issue #1100).

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10}, options);
  at::Tensor t4 = at::randn({1024}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t4});
  auto outputs = ke.run({t0, t4});

  testValidate(&fusion, outputs, {t0, t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReductionWithNonExactParallelDimensions_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = sum(tv2, {0});
  fusion.addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t2 = at::randn({19}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t2});
  auto outputs = ke.run({t0, t2});

  testValidate(&fusion, outputs, {t0, t2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridWelfordWithNonExactParallelDimensions_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(1);
  fusion.addInput(tv2);
  auto tv3 = Welford(tv2, {0}).avg;
  fusion.addOutput(tv3);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(0)->parallelize(ParallelType::BIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t2 = at::randn({19}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t2});
  auto outputs = ke.run({t0, t2});

  auto ref1 = t0 + 1;
  auto ref2 = mean(t2, {0});

  testValidate(&fusion, outputs, {t0, t2}, {ref1, ref2}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridReductionWithNonExactParallelDimensions2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0, 1});
  fusion.addOutput(tv1);

  auto tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(3);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDz);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(2)->parallelize(ParallelType::BIDz);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 3}, options);
  at::Tensor t2 = at::randn({5, 6, 7}, options);
  at::Tensor t4 = at::randn({8, 9, 10}, options);
  auto outputs = ke.run({t0, t2, t4});

  testValidate(&fusion, outputs, {t0, t2, t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionGridWelfordWithNonExactParallelDimensions2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tvs = Welford(tv0, {0, 1});
  fusion.addOutput(tvs.avg);

  auto tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(3);
  fusion.addInput(tv4);
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  tvs.avg->axis(0)->parallelize(ParallelType::BIDx);
  tvs.avg->axis(1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::TIDx);
  tv3->axis(1)->parallelize(ParallelType::TIDy);
  tv3->axis(2)->parallelize(ParallelType::TIDz);

  tv5->axis(0)->parallelize(ParallelType::BIDx);
  tv5->axis(1)->parallelize(ParallelType::BIDy);
  tv5->axis(2)->parallelize(ParallelType::BIDz);

  KernelExecutor ke;
  ke.compile(&fusion);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 3}, options);
  at::Tensor t2 = at::randn({5, 6, 7}, options);
  at::Tensor t4 = at::randn({8, 9, 10}, options);
  auto outputs = ke.run({t0, t2, t4});

  auto ref1 = t0.mean(at::IntArrayRef{0, 1});
  auto ref2 = t2 + 1;
  auto ref3 = t4 + 1;

  testValidate(
      &fusion, outputs, {t0, t2, t4}, {ref1, ref2, ref3}, __LINE__, __FILE__);
}

// Repro of issue #1102
TEST_F(NVFuserTest, FusionPredicateParallelizedDomains_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // Just to make TIDx/y/z non-exact
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  auto tv4 = makeSymbolicTensor(1);
  fusion.addInput(tv4);

  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  auto tv6 = add(tv5, IrBuilder::create<Val>(1.0));
  auto tv7 = add(tv6, IrBuilder::create<Val>(1.0));
  auto tv8 = add(tv7, IrBuilder::create<Val>(1.0));
  auto tv9 = sum(tv8, {0});
  fusion.addOutput(tv9);

  tv1->split(0, 5);
  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv1->setMemoryType(MemoryType::Shared);
  tv2->split(0, 6);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv2->setMemoryType(MemoryType::Shared);
  tv3->split(0, 7);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);

  tv9->split(0, 4);
  tv4->computeAt(tv9, 1);

  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv6->axis(-1)->parallelize(ParallelType::TIDz);
  tv7->axis(-1)->parallelize(ParallelType::TIDz);
  tv8->axis(-1)->parallelize(ParallelType::TIDz);
  tv9->axis(-1)->parallelize(ParallelType::TIDz);
  tv9->axis(0)->parallelize(ParallelType::BIDx);

  tv5->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t4 = at::randn({19}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t4});
  auto outputs = ke.run({t0, t4});

  auto ref1 = t0 + 3;
  auto ref2 = sum(t4 + 4);

  testValidate(&fusion, outputs, {t0, t4}, {ref1, ref2}, __LINE__, __FILE__);
}

// Repro of #1102 and #1129
TEST_F(NVFuserTest, FusionSmemPredicateUnswitch_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  auto tv5 = add(tv4, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv5);

  // Just to make TIDx/y/z non-exact
  auto tvx = add(tv1, IrBuilder::create<Val>(1.0));
  auto tvy = add(tvx, IrBuilder::create<Val>(1.0));
  auto tvz = add(tvy, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tvz);

  tv5->split(0, 4);
  tv0->computeAt(tv5, 1);

  tv0->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDz);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);
  tv5->axis(-1)->parallelize(ParallelType::TIDy);
  tv5->axis(0)->parallelize(ParallelType::Unswitch);

  tvx->split(0, 5);
  tvx->axis(-1)->parallelize(ParallelType::TIDx);
  tvy->split(0, 6);
  tvy->axis(-1)->parallelize(ParallelType::TIDy);
  tvz->split(0, 7);
  tvz->axis(-1)->parallelize(ParallelType::TIDz);

  for (auto tv : {tv2, tv3, tv4, tvx, tvy}) {
    tv->setMemoryType(MemoryType::Shared);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({17}, options);
  at::Tensor t1 = at::randn({19}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0, t1});
  auto outputs = ke.run({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Repro of issue #1136
TEST_F(NVFuserTest, FusionFloatPow_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Val>(4L));
  // To check if pow(tv0, 2) is replaced with tv0 * tv0
  auto tv2 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Val>(2L));
  // To check if pow(tv0, 2.0) is replaced with tv0 * tv0
  auto tv3 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Val>(2.0));
  auto tv4 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Val>(3L));
  auto tv5 = binaryOp(BinaryOpType::Pow, tv0, IrBuilder::create<Val>(3.0));
  auto s = binaryOp(
      BinaryOpType::Pow,
      IrBuilder::create<Val>(3.0),
      IrBuilder::create<Val>(3.0));
  auto tv6 = add(tv0, s);

  fusion.addOutput(tv1);
  fusion.addOutput(tv2);
  fusion.addOutput(tv3);
  fusion.addOutput(tv4);
  fusion.addOutput(tv5);
  fusion.addOutput(tv6);

  tv1->split(0, 32);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);

  TransformPropagatorWithCheck propagator(tv1);
  MaxLogicalDomainInfoSpanningTree(tv1).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv1, {tv2, tv3, tv4, tv5, tv6});

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1000}, options);
  // Negative inputs cause nan in Fuesr as use_fast_math is enabled
  t0 = abs(t0);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  auto p4 = at::pow(t0, 4);
  auto p2 = at::pow(t0, 2);
  auto p3 = at::pow(t0, 3);
  auto t6 = t0 + std::pow(3, 3);

  testValidate(
      &fusion, outputs, {t0}, {p4, p2, p2, p3, p3, t6}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1127_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const int numel = 4;

  auto tv0 = makeConcreteTensor({numel});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {0});
  auto tv2 = broadcast(tv1, {true});

  auto tv3 = makeConcreteTensor({numel, numel});
  fusion.addInput(tv3);

  auto tv4 = sum(tv3, {1});

  auto tv5 = add(tv2, tv4);
  fusion.addOutput(tv5);

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  tv2->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv5->axis(0)->parallelize(ParallelType::TIDx);

  // Lowering should fail since tv5 is predicated and paralellized with TIDx.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(fusion.printKernel());
}

TEST_F(NVFuserTest, FusionThreadPredicateUnswitch_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({10, 1024});
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->computeAt(tv3, -1);
  tv3->axis(0)->parallelize(ParallelType::Unswitch);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({10, 1024}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  testValidate(&fusion, outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionNonContigOutputs_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  tv1->setContiguity(false);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_input = at::randn({10}, options);
  at::Tensor at_output = at::empty_strided({10}, {2}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_input});
  auto cg_outputs = ke.run({at_input}, {at_output});

  // Returned outputs should only contain one tensor that is the same
  // as the output tensor given to runFusion
  NVF_CHECK(cg_outputs.size() == 1);
  NVF_CHECK(cg_outputs[0].as<at::Tensor>().is_same(at_output));
  NVF_CHECK(!cg_outputs[0].as<at::Tensor>().is_contiguous());

  testValidate(&fusion, cg_outputs, {at_input}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionTestWarpSoftMax_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Setup softmax fusion
  auto input = makeContigTensor(2);
  fusion.addInput(input);
  auto output = softmax(input, 1);
  fusion.addOutput(output);

  // Setup runtime input
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({8, 16 * 197}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {t0});
  NVF_CHECK(Schedule::canSchedule(
      SchedulerType::InnerPersistent, &fusion, runtime_info));
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::InnerPersistent);
  auto heuristic_params = scheduler->computeHeuristics(&fusion, runtime_info);
  scheduler->schedule(&fusion, heuristic_params.get());

  // Modify the schedule to use warp reduction
  auto used_vals = fusion.usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (IterDomain* id : tv->getLoopDomain()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        id->padToMultipleOfWarp();
      }
    }
  }

  // Test result
  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});
  auto ref_output = at::_softmax(t0, 1, false);
  testValidate(&fusion, outputs, {t0}, {ref_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1133_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {1});
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));

  fusion.addOutput(tv3);

  tv0->computeAt(tv3, 1);

  const int split_factor = 32;

  tv2->split(-1, split_factor);
  tv1->computeAt(tv2, -2);

  tv1->axis(-1)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(0)->parallelize(ParallelType::Unswitch);

  tv1->setMemoryType(MemoryType::Shared);
  tv2->setMemoryType(MemoryType::Shared);

  // Both tv1 and tv2 should be allocated at the top-level scope
  GpuLower gpulw(&fusion);
  bool tv1_validated = false;
  bool tv2_validated = false;
  for (const auto& kir_node : gpulw.run()->topLevelExprs()) {
    if (auto alloc = dynamic_cast<kir::Allocate*>(kir_node)) {
      // Only interested in the allocations of tv1 and tv2
      if (alloc->buffer()->name() != 1 && alloc->buffer()->name() != 2) {
        continue;
      }
      auto size = alloc->size();
      NVF_CHECK(size->isConst(), "Allocation not constant");
      auto size_int = size->value();
      if (alloc->buffer()->name() == 1) {
        NVF_CHECK(
            size_int == split_factor,
            "Invalid allocation size: ",
            size->value());
        tv1_validated = true;
      } else {
        NVF_CHECK(size_int == 1, "Invalid allocation size: ", size->value());
        tv2_validated = true;
      }
    }
  }

  NVF_CHECK(tv1_validated, "Failed to validate tv1 allocation");
  NVF_CHECK(tv2_validated, "Failed to validate tv2 allocation");

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99, 101}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  auto ref = (t0 + 1).sum({1}) + 1;

  GTEST_SKIP() << "NOTE: disabling 1133 test for cuda12.2 failure. "
               << "See issue: https://github.com/NVIDIA/Fuser/issues/615";
  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorContigIDs_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1});
  fusion.addOutput(tv1);

  tv1->split(1, 32);

  auto tv2 = tv1->rFactor({1});

  // This merged domain is not contiguous.
  tv2->merge(0, 2);

  tv2->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({99, 101}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto outputs = ke.run({t0});

  auto ref = t0.sum({1});

  testValidate(&fusion, outputs, {t0}, {ref}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIssue1223_CUDA) {
  if (!deviceMajorMinorCheck(7)) {
    GTEST_SKIP() << "skipping tests on pre-Volta GPUs";
    return;
  }
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = sum(tv1, {0, 1});
  fusion.addOutput(tv2);

  auto tv3 = add(tv0, IrBuilder::create<Val>(0.0));
  fusion.addOutput(tv3);

  tv2->split(0, 4);
  tv2->split(1, 1, false);
  tv2->split(-1, 4);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);
  tv2->axis(-3)->parallelize(ParallelType::TIDx);
  tv2->axis(-1)->parallelize(ParallelType::TIDy);

  tv1->computeAt(tv2, -1);

  // Make TIDx and TIDy non-exact
  tv3->split(0, 32);
  tv3->split(-1, 32);
  tv3->axis(1)->parallelize(ParallelType::TIDx);
  tv3->axis(3)->parallelize(ParallelType::TIDy);

  // The second axis of both tv1 and tv2 are fully unswitched, so they
  // don't need to predicate the parallel type usage of TIDy, whereas
  // the first axis is only partially unswitched, i.e., part of its
  // split output domains is outside the unswitched axis, so the first
  // axis, which uses TIDx, needs to predicate the parallel
  // dimension. Previously, as reported in issue #1223, unswitched
  // expressions didn't predicate parallel dimensions. It should be
  // fixed by PR #1222.

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::ones({11, 10}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_t0});
  auto cg_outputs = ke.run({at_t0});

  auto at_t1 = (at_t0 + 1).sum();

  testValidate(
      &fusion, cg_outputs, {at_t0}, {at_t1, at_t0}, __LINE__, __FILE__);
}

// See #1247 and #1250
TEST_F(NVFuserTest, FusionRfactorPredication1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = min(tv1, {0});

  fusion.addOutput(tv2);

  // Make TIDx non-exact
  auto tv3 = makeContigTensor(1);
  fusion.addInput(tv3);

  auto tv4 = add(tv3, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv4);

  tv2->split(0, 4);
  tv2->rFactor({1});

  tv0->computeAt(tv2, 1);

  tv2->axis(0)->parallelize(ParallelType::TIDx);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::randn({9}, options);
  at_t0 = at::abs(at_t0);
  at::Tensor at_t3 = at::randn({128}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_t0, at_t3});
  auto cg_outputs = ke.run({at_t0, at_t3});

  auto at_t2 = (at_t0 + 1).min();
  auto at_t4 = at_t3 + 1;

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t3}, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorPredication2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = min(tv0, {0});
  fusion.addOutput(tv1);

  // Make TIDx non-exact
  auto tv2 = makeContigTensor(1);
  fusion.addInput(tv2);

  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv1->split(0, 4);
  auto tv4 = tv1->rFactor({0});

  tv1->split(0, 3);

  // tv0->computeAt(tv1, 3);
  tv4->reorder({{0, 1}});
  tv4->split(0, 3);
  tv4->setMemoryType(MemoryType::Shared);

  // tv0: [I]
  // tv4: [4/3, 3, I/4]
  // tv1: [4/3, 3]

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1, {tv4});

  tv3->axis(0)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_t0 = at::randn({9}, options);
  at_t0 = at::abs(at_t0);
  at::Tensor at_t3 = at::randn({128}, options);

  KernelExecutor ke;
  ke.compile(&fusion, {at_t0, at_t3});
  auto cg_outputs = ke.run({at_t0, at_t3});

  auto at_t2 = std::get<0>(at_t0.min(0));
  auto at_t4 = at_t3 + 1;

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t3}, {at_t2, at_t4}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionRfactorIndirectRoot_CUDA) {
  // https://github.com/csarofeen/pytorch/issues/1692
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(3);
  fusion.addInput(tv0);

  auto tv1 = sum(tv0, {1, 2});
  fusion.addOutput(tv1);

  tv1->split(2, 4);
  tv1->split(1, 3);
  tv1->merge(2, 3);
  auto rf = tv1->rFactor({-1});

  tv1->split(0, 256);
  tv1->axis(0)->parallelize(ParallelType::BIDx);
  tv1->axis(1)->parallelize(ParallelType::TIDx);
  rf->computeAt(tv1, -1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto at_in = at::randn({6, 6, 6}, options);
  auto at_out = at_in.sum({1, 2});

  KernelExecutor ke;
  ke.compile(&fusion, {at_in});
  auto cg_outputs = ke.run({at_in});

  testValidate(&fusion, cg_outputs, {at_in}, {at_out}, __LINE__, __FILE__);
}

} // namespace nvfuser
