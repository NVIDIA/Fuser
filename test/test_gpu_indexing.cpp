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

#include <compute_at_map.h>
#include <executor.h>
#include <id_model/id_model.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <scheduler/all_schedulers.h>

#include <test/utils.h>
#include <test/validator.h>

#include <torch/torch.h>

namespace nvfuser {

TEST_F(NVFuserTest, FusionIndexing1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(0);
  tv4->merge(0);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Same as 1 but merge starting from inner most dimension
TEST_F(NVFuserTest, FusionIndexing2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false, false, false});
  auto tv4 = add(tv3, tv1);

  fusion.addOutput(tv4);

  tv4->merge(-2);
  tv4->merge(-2);
  tv4->merge(-2);

  tv4->split(0, 128);
  tv4->split(0, 4);

  tv2->computeAt(tv4, 1);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::Unroll);
  tv4->axis(2)->parallelize(ParallelType::TIDx);

  tv3->axis(1)->parallelize(ParallelType::Unroll);
  tv3->axis(2)->parallelize(ParallelType::TIDx);

  tv2->axis(1)->parallelize(ParallelType::Unroll);
  tv2->axis(2)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Same compute as 1 and 2 but use a scheduler.
TEST_F(NVFuserTest, FusionIndexing3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(3);
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, tv1);
  fusion.addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Same as 3 but use 3 dimensions and concrete sizes
TEST_F(NVFuserTest, FusionIndexing4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeConcreteTensor({4, 8});
  fusion.addInput(tv0);
  TensorView* tv1 = makeConcreteTensor({4, 4, 8});
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv3 = broadcast(tv2, {true, false, false});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 8}, options);
  at::Tensor t1 = at::randn({4, 4, 8}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing5_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(3);
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv3 = broadcast(tv2, {true, false, true});
  TensorView* tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv3->merge(0)->merge(0)->split(0, 2)->split(0, 3);
  tv4->merge(0)->merge(0)->split(0, 2)->split(0, 3);

  tv0->computeAt(tv4, 1);
  tv1->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({7}, options);
  at::Tensor t1 = at::randn({5, 7, 11}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing6_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> tensor0_shape{7, 4, 7};
  std::vector<int64_t> tensor1_shape{4, 7};

  TensorView* tv0 = makeSymbolicTensor(tensor0_shape.size());
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(tensor1_shape.size());
  fusion.addInput(tv1);

  TensorView* tv2 = add(tv0, tv1);
  TensorView* tv3 = sum(tv2, {0, 1});
  fusion.addOutput(tv3);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input0 = at::randn(tensor0_shape, options);
  at::Tensor input1 = at::randn(tensor1_shape, options);

  std::vector<int64_t> reduction_axes{0, 1};
  auto reduction_params = getReductionHeuristics(&fusion, {input0, input1});
  NVF_CHECK(reduction_params, "Reduction schedule was not generated!");
  scheduleReduction(&fusion, *reduction_params);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input0, input1}, reduction_params->lparams);
  auto cg_outputs = fe.runFusion({input0, input1}, reduction_params->lparams);

  auto aten_output = input0.add(input1).to(at::kDouble).sum(reduction_axes);

  testValidate(
      &fusion,
      cg_outputs,
      {input0, input1},
      {aten_output},
      __LINE__,
      __FILE__,
      "",
      reduction_params->lparams);
}

TEST_F(NVFuserTest, FusionIndexing7_CUDA) {
  // Might be able to use this one without 6 as the heuristics in 6 may change
  // and this test is to cover the same issue.
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128);
  tv4->split(0, 4);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing8_CUDA) {
  // Same as 7 but with outer splits instead of inner
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = sum(tv3, {0, 1});
  fusion.addOutput(tv4);

  tv4->merge(0, 1);
  tv4->split(0, 128, false);
  tv4->split(0, 4, false);

  auto tv5 = tv4->rFactor({0, 1});

  tv5->computeAt(tv4, -1);
  tv0->computeAt(tv5, -1);

  tv4->axis(0)->parallelize(ParallelType::TIDx);

  const int numel_x = 100;
  const int numel_y = 200;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_x}, options);
  auto at_t1 = at::randn({numel_x, numel_y}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {at_t0, at_t1});
  auto cg_outputs = fe.runFusion({at_t0, at_t1});

  auto aten_output = (at_t0.unsqueeze(-1).expand({numel_x, numel_y}) + at_t1)
                         .to(at::kDouble)
                         .sum();

  testValidate(
      &fusion, cg_outputs, {at_t0, at_t1}, {aten_output}, __LINE__, __FILE__);
}

// Same as 5 but using implicit broadcast
TEST_F(NVFuserTest, FusionIndexing9_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = broadcast(tv0, {false, true});

  auto tv2 = mul(tv1, IrBuilder::create<Val>(2.0));
  fusion.addOutput(tv2);

  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);

  auto tv4 = add(tv3, tv2);
  fusion.addOutput(tv4);

  const int numel_x = 200;
  const int numel_y = 300;
  const int numel_z = 400;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto at_t0 = at::randn({numel_y}, options);
  auto at_t3 = at::randn({numel_x, numel_y, numel_z}, options);
  std::vector<c10::IValue> aten_inputs = {at_t0, at_t3};

  auto lparams = schedulePointwise(&fusion, aten_inputs);

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs, lparams);
  auto cg_outputs = fe.runFusion(aten_inputs, lparams);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing10_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Set up your input tensor views
  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv1 = makeContigTensor(2);

  // Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // Do math with it, it returns a `Val*` but can be static_casted back to
  // TensorView
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv0, tv2);

  // Register your outputs
  fusion.addOutput(tv3);

  auto tv0_cache = tv0->cacheAfter();
  auto tv1_cache = tv1->cacheAfter();

  std::vector<TensorView*> tvs = {tv0_cache, tv1_cache, tv2, tv3};

  for (auto tv : tvs) {
    tv->split(1, 2, false);
    tv->split(1, 1);
    tv->split(-1, 4);
    // [I0, 2, 1, I1/2/4, 4]
    tv->reorder({{1, 2}, {2, 3}, {3, 1}});
    tv->axis(0)->parallelize(ParallelType::BIDx);
    tv->axis(1)->parallelize(ParallelType::TIDx);
  }

  // For all inputs, computeAt the output inline, temporaries should be squeezed
  // between them
  tv0->computeAt(tv3, 1);
  tv1->computeAt(tv3, 1);

  tv0_cache->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1_cache->axis(-1)->parallelize(ParallelType::Vectorize);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64, 128}, options);
  at::Tensor input2 = at::rand_like(input1);
  at::Tensor output = at::empty_like(input1);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {input1, input2});
  fe.runFusion({input1, input2}, {output});

  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  NVF_CHECK(output_ref.equal(output));
}

TEST_F(NVFuserTest, FusionIndexing11_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  auto tv0 = makeSymbolicTensor(4);
  auto tv1 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  auto tv2 = add(tv1, IrBuilder::create<Val>(1.0));
  auto tv3 = broadcast(tv2, {true, false, true, true});
  auto tv4 = add(tv3, tv0);

  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->merge(1);

  tv4->split(1, 32);
  tv4->split(0, 1);

  tv4->reorder({{2, 1}});

  tv2->computeAt(tv4, 3);

  tv2->setMemoryType(MemoryType::Global);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDy);
  tv4->axis(2)->parallelize(ParallelType::Unswitch);
  tv4->axis(-1)->parallelize(ParallelType::TIDx);

  tv3->axis(-1)->parallelize(ParallelType::TIDx);

  FusionExecutor fe;

  at::Tensor t0 = at::randn({w, x, y, z}, options);
  at::Tensor t1 = at::randn({x}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing12_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({9, 5});
  fusion.addInput(tv0);

  TensorView* tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv2 = add(tv1, IrBuilder::create<Val>(2.0));
  TensorView* tv3 = add(tv1, IrBuilder::create<Val>(3.0));
  TensorView* tv4 = sum(tv3, {1});

  fusion.addOutput(tv2);
  fusion.addOutput(tv4);

  tv4->split(1, 4);
  auto tv5 = tv4->rFactor({2});

  tv1->computeAt(tv5, 2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn({9, 5}, options);

  auto t1 = aten_input.add(1.0);
  auto t2 = t1.add(2.0);
  auto t3 = t1.add(3.0);
  auto t4 = t3.sum(1);

  std::vector<at::Tensor> aten_outputs = {t2, t4};

  FusionExecutor fe;
  fe.compileFusion(&fusion, {aten_input});
  auto cg_outputs = fe.runFusion({aten_input});

  testValidate(
      &fusion, cg_outputs, {aten_input}, aten_outputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing13_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Progressively broadcast tensors
  TensorView* tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  TensorView* tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);
  TensorView* tv2 = makeSymbolicTensor(3);
  fusion.addInput(tv2);

  TensorView* tv3 = add(tv0, IrBuilder::create<Val>(1.0));
  TensorView* tv4 = broadcast(tv3, {false, true});
  TensorView* tv5 = add(tv4, tv1);
  TensorView* tv6 = add(tv5, tv2);

  fusion.addOutput(tv6);

  // Split inner dimension
  tv6->split(1, 4);
  // Merge middle dims with outer dimensions
  tv6->merge(2);
  tv6->merge(0);

  // tv6[I0*I1o, I1i*I2]

  // Compute everything inline
  tv0->computeAt(tv6, -1);

  tv6->axis(0)->parallelize(ParallelType::BIDx);
  tv6->axis(1)->parallelize(ParallelType::TIDx);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9, z = 5;
  at::Tensor t0 = at::randn({y}, options);
  at::Tensor t1 = at::randn({y, z}, options);
  at::Tensor t2 = at::randn({x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1, t2};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing14_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({1, -1});
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [b0, i1]
  auto tv2 = add(tv0, IrBuilder::create<Val>(2.0));

  // [i0, i1]
  auto tv3 = add(tv1, IrBuilder::create<Val>(3.0));

  // [b0, i1]
  auto tv4 = add(tv2, IrBuilder::create<Val>(4.0));

  // [io, i1]
  auto tv5 = add(tv2, tv3);

  fusion.addOutput(tv4);
  fusion.addOutput(tv5);

  tv0->computeAt(tv4, -1);

  tv3->setMemoryType(MemoryType::Global);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int x = 13, y = 9;
  at::Tensor t0 = at::randn({1, y}, options);
  at::Tensor t1 = at::randn({x, y}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// This excercises indexing with broadcast root axes. Non-broadcast
// axes need to be preferred when propagating index exprs to root
// axes. See, e.g., Index::getConsumerIndex_impl.
TEST_F(NVFuserTest, FusionIndexing15_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);
  auto tv1 = broadcast(tv0, {false, true});
  auto tv2 = broadcast(tv1, {false, false, true});
  auto tv3 = makeSymbolicTensor(3);
  fusion.addInput(tv3);
  auto tv4 = add(tv2, tv3);
  fusion.addOutput(tv4);

  tv4->merge(1)->merge(0);
  tv4->split(0, 8);
  tv0->computeAt(tv4, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const int bx = 10;
  const int by = 20;
  const int bz = 30;
  at::Tensor t0 = at::randn({bx}, options);
  at::Tensor t3 = at::randn({bx, by, bz}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t3};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing16_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);

  TensorView* tv1 = makeConcreteTensor({5, 3});
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv1, {false, true, false});

  auto tv3 = add(tv0, tv2);

  fusion.addOutput(tv3);

  tv2->merge(0);
  tv1->computeAt(tv2, 1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({5, 3}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, FusionIndexing17_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 4, 3});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({4});
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = set(tv1);

  auto tv4 = sum(tv2, {0, 2});
  auto tv5 = add(tv4, tv3);
  fusion.addOutput(tv5);

  auto tv6 = broadcast(tv3, {true, false, true});
  auto tv7 = add(tv2, tv6);
  fusion.addOutput(tv7);

  tv2->computeAt(tv4, -1, ComputeAtMode::BestEffort);
  tv3->computeAt(tv7, -1, ComputeAtMode::BestEffort);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({5, 4, 3}, options);
  at::Tensor t1 = at::randn({4}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// TODO: Finish and enable test
TEST_F(NVFuserTest, FusionIndexing18_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv0 = makeConcreteTensor({5, 7, 11, 13});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 = makeConcreteTensor({5, 11});
  fusion.addInput(tv2);

  auto tv3 = broadcast(tv2, {false, true, false, true});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  // // tv4[5, 7, 11, 13] = tv3[5, b1, 11, b3] + tv1[5, 7, 11, 13]
  tv4->merge(0, 3);
  // tv4[5*13, 7, 11]
  tv4->split(0, 3);
  // tv4[5*13//3, 3, 7, 11]
  tv4->merge(2, 3)->split(2, 2);
  // tv4[5*13//3, 3, 7*11//2, 2]
  // tv4->merge(0, 2);
  // // tv4[(5*13//3)*(7*11//2), 3, 2]

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);
  inlineAllAt(tv4, 1, false);
  fusion.printKernel();
  // std::cout<<tv4->definition()->toString()<<std::endl;
  // fusion.print();
  // ComputeAtMap ca_map(&fusion);
  // std::cout << ca_map.idGraph().loopNodes().toString() << std::endl;
}

// TODO: Finish and enable test
//
// Create a case where we're missing a valid concrete id so the compute at map
// processing will fail. We need to be able to create the concrete ID not just
// look for one.
TEST_F(NVFuserTest, FusionIndexing19_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({7});
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  auto tv2 = broadcast(tv1, {false, true});

  auto tv3 = makeConcreteTensor({7, 11});
  fusion.addInput(tv3);

  auto tv4 = add(tv3, tv2);
  auto tv5 = broadcast(tv4, {false, false, true});
  // tv4[7, 11, 1]

  auto tv6 = broadcast(tv1, {false, true});

  auto tv7 = makeConcreteTensor({7, 13});
  fusion.addInput(tv7);
  auto tv8 = add(tv7, tv6);
  auto tv9 = broadcast(tv8, {false, true, false});
  // tv9[7, 1, 13]

  auto tv10 = add(tv5, tv9);
  fusion.addOutput(tv10);

  // tv10[7, 11, 13]
  tv10->merge(0)->merge(0);
  // tv10[7*11*13]
  tv10->split(0, 5)->split(0, 3);
  // tv10[7*11*13//5//3, 3, 5]

  TransformPropagatorWithCheck propagator(tv10);
  MaxRootDomainInfoSpanningTree(tv10).traverse(&propagator);

  std::vector<TensorView*> tensors_to_inline{tv1, tv2, tv4, tv6, tv8};
  for (auto tensor : tensors_to_inline) {
    tensor->inlineAt(1);
  }

  IdModel id_model(&fusion);

  // All of the IDs that are generated with merge operations from the
  // root domains should be mapped to the single group.
  const ValGroup& merge_loop_group =
      id_model.idGraph(IdMappingMode::LOOP).toGroup(tv1->getRootDomain().at(0));
  for (auto tv : {tv1, tv2, tv4, tv5, tv6, tv8, tv9}) {
    for (auto id : ir_utils::allIDsOf(tv)) {
      if (dynamic_cast<Split*>(id->definition()) == nullptr) {
        const ValGroup& loop_group =
            id_model.idGraph(IdMappingMode::LOOP).toGroup(id);
        ASSERT_EQ(loop_group, merge_loop_group)
            << "Unexpected loop group: " << nvfuser::toString(loop_group);
      }
    }
  }

  const auto& promotion_map = id_model.loopPromotionMap();

  // The merge loop group should be promoted to the output of the
  // final merge in tv10
  auto ref_merge_out = tv10->axis(0)
                           ->definition()
                           ->input(0)
                           ->definition()
                           ->input(0)
                           ->as<IterDomain>();

  auto promotion_map_it = promotion_map.find(merge_loop_group);
  ASSERT_TRUE(promotion_map_it != promotion_map.end())
      << "Loop promotion not found for merge loop group";
  auto merge_out_promotion_id = promotion_map_it->second;
  ASSERT_EQ(
      id_model.idGraph(IdMappingMode::EXACT).toGroup(merge_out_promotion_id),
      id_model.idGraph(IdMappingMode::EXACT).toGroup(ref_merge_out))
      << "Merge loop group should be promoted to " << ref_merge_out->toString();
  ASSERT_NE(
      id_model.idGraph(IdMappingMode::LOOP).toGroup(merge_out_promotion_id),
      id_model.idGraph(IdMappingMode::LOOP).toGroup(ref_merge_out))
      << "Should not be loop-mapped with ref: "
      << merge_out_promotion_id->toString();

  // Get the corresponding reference ID in tv10
  auto getRefId = [&](TensorView* tv, IterDomain* id) -> IterDomain* {
    if (dynamic_cast<Split*>(id->definition()) != nullptr) {
      if (id->uses().empty()) {
        auto it = std::find(
            tv->getLeafDomain().begin(), tv->getLeafDomain().end(), id);
        NVF_ERROR(it != tv->getLeafDomain().end());
        int leaf_pos =
            static_cast<int>(std::distance(tv->getLeafDomain().begin(), it));
        return tv10->axis(leaf_pos);
      } else {
        return tv10->axis(0)->definition()->input(0)->as<IterDomain>();
      }
    } else {
      return ref_merge_out;
    }
  };

  // Check if id is a leaf of a consumer tensor of tv
  auto isIdOfConsumerTensor = [&](IterDomain* id, TensorView* tv) -> bool {
    auto consumer_tvs = ir_utils::consumerTvsOf(tv);
    return std::any_of(
        consumer_tvs.begin(), consumer_tvs.end(), [&](auto consumer_tv) {
          auto all_ids = ir_utils::allIDsOf(consumer_tv);
          return std::find(all_ids.begin(), all_ids.end(), id) != all_ids.end();
        });
  };

  // At this point, all of the IDs from the root until split are
  // validated. Validating the remaining IDs
  for (auto tv : {tv1, tv2, tv4, tv5, tv6, tv8, tv9}) {
    for (auto id : ir_utils::allIDsOf(tv)) {
      const auto& loop_group =
          id_model.idGraph(IdMappingMode::LOOP).toGroup(id);
      if (loop_group == merge_loop_group) {
        // already validated
        continue;
      }

      auto promotion_map_it = promotion_map.find(loop_group);
      ASSERT_TRUE(promotion_map_it != promotion_map.end())
          << "Loop promotion not found for " << id->toString() << " of "
          << tv->toString()
          << ". Loop group: " << nvfuser::toString(loop_group);

      auto promotion_id = promotion_map_it->second;

      // Promotion ID should be loop-mapped
      ASSERT_TRUE(loop_group->has(promotion_id))
          << "Loop promotion for " << id->toString() << " of " << tv->toString()
          << " is promoted to an ID that isn't loop mapped: "
          << promotion_id->toString() << std::endl;

      auto promotion_exact_group =
          id_model.idGraph(IdMappingMode::EXACT).toGroup(promotion_id);

      auto ref_id = getRefId(tv, id);
      auto ref_exact_group =
          id_model.idGraph(IdMappingMode::EXACT).toGroup(ref_id);

      ASSERT_EQ(promotion_exact_group, ref_exact_group)
          << "Invalid promotion: " << id->toString() << " of " << tv->toString()
          << ". Promotion group: " << nvfuser::toString(promotion_exact_group);

      auto ref_loop_group =
          id_model.idGraph(IdMappingMode::LOOP).toGroup(ref_id);
      ASSERT_NE(loop_group, ref_loop_group)
          << "Invalid promotion: " << id->toString() << " of " << tv->toString()
          << ". Should not be loop-mapped with ref: "
          << nvfuser::toString(loop_group);

      // If id is a leaf, make sure it isn't mapped with
      auto leaf_id_it =
          std::find(tv->getLeafDomain().begin(), tv->getLeafDomain().end(), id);
      if (leaf_id_it != tv->getLeafDomain().end() &&
          std::distance(tv->getLeafDomain().begin(), leaf_id_it) >=
              tv->getComputeAtPosition()) {
        for (auto loop_mapped_id : *loop_group) {
          if (loop_mapped_id == id) {
            continue;
          }
          ASSERT_FALSE(
              isIdOfConsumerTensor(loop_mapped_id->as<IterDomain>(), tv))
              << "Invalid promotion: " << id->toString() << " of "
              << tv->toString() << ". Found to mapped a consumer tensor: "
              << loop_mapped_id->name();
        }
      }
    }
  }

  // The current ComputeAtMap fails with this fusion
  // fusion.printKernel();
}

// TODO: Finish and enable test
//
// Progressive loop promotion. producer gets promoted in consumer, consumer is
// promoted in a different way to its consumer.
TEST_F(NVFuserTest, FusionIndexing20_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeConcreteTensor({5});
  fusion.addInput(tv0);

  // [5]
  auto tv1 = set(tv0);
  auto tv2 = broadcast(tv1, {true, false});
  // [1, 5]
  auto tv3 = makeConcreteTensor({3, 5});
  fusion.addInput(tv3);
  auto tv4 = add(tv3, tv2);
  // [3, 5]

  auto tv5 = broadcast(tv4, {false, false, true});
  // [3, 5, 1]
  auto tv6 = makeConcreteTensor({3, 5, 7});
  fusion.addInput(tv6);
  auto tv7 = add(tv5, tv6);
  // [3, 5, 7]
  fusion.addOutput(tv7);

  tv4->merge(0)->split(0, 2, false);
  // [3, 5]
  // [3, 3*5//2]

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  // tv0->tv1->tv2(b)->tv4->tv5(b)->tv7

  tv1->inlineAt(1);
  tv2->inlineAt(1);
  tv4->inlineAt(1);

  // [2, 3*5//2]
  tv5->merge(1)->split(1, 4, false);
  // [2, 4, (3*5//2)*1//4]
  tv7->merge(1)->split(1, 4, false);
  // [2, 4, (3*5//2)*7//4]
  tv5->inlineAt(2);

  fusion.printKernel();
}

// Repro for issue #1873
TEST_F(NVFuserTest, FusionInlineBroadcastIndexing0_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);
  tv4->split(0, 32);

  tv0->computeAt(tv4, 1);

  tv2->split(-1, 8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({123}, options);
  at::Tensor t1 = at::randn({3, 123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  auto outputs = fe.runFusion({t0, t1});

  auto tv_ref = t0 + t1;

  testValidate(&fusion, outputs, {t0, t1}, {tv_ref}, __LINE__, __FILE__);
}

// Broadcast inline 3 times and merge all domains
TEST_F(NVFuserTest, FusionMultiPromotion_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [y]
  auto tv0 = makeSymbolicTensor(1);
  // [w, x, y, z]
  auto tv1 = makeSymbolicTensor(4);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // y
  auto tv2 = broadcast(tv0, {true, false});
  // w, y, z
  auto tv3 = broadcast(tv2, {false, false, true});
  // w, y, z
  auto tv4 = broadcast(tv3, {false, true, false, false});
  // w, x, y, z
  auto tv5 = add(tv4, tv1);

  fusion.addOutput(tv5);

  tv5->merge(1)->merge(1)->merge(0)->split(0, 11);

  tv0->computeAt(tv5, 1);
  tv1->computeAt(tv5, 1);

  FusionExecutor fe;

  int w = 3, x = 4, y = 7, z = 8;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({y}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  auto t4 = t0.unsqueeze(0).unsqueeze(0).unsqueeze(-1);
  auto aten_output = t4.add(t1);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

// TODO: Finish and enable test.
// Broadcast and concretize same domain in two different ways and try to merge
// their loops remains unsupported.
TEST_F(NVFuserTest, FusionMultiPromotion2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // [w]
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // [w, x]
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  // [w, y]
  auto tv2 = makeSymbolicTensor(2);
  fusion.addInput(tv2);

  auto tv3 = set(tv0);
  // [w]
  auto tv4 = broadcast(tv3, {false, true});
  // [w, 1]
  auto tv5 = add(tv4, tv1);
  // [w, x]
  fusion.addOutput(tv5);

  // [w]
  auto tv6 = broadcast(tv3, {false, true});
  // [w, 1]
  auto tv7 = add(tv6, tv2);
  // [y]
  fusion.addOutput(tv7);

  for (auto tv : std::vector<TensorView*>{tv4, tv5, tv6, tv7}) {
    tv->merge(0);
  }

  for (auto tv : std::vector<TensorView*>{tv3, tv4, tv6}) {
    tv->inlineAt(1);
  }

  ASSERT_ANY_THROW(fusion.printKernel());
}

// TODO: All the above tests are merges followed by splits, we should make some
// more complex examples even though merging then spliting is the most likely
// use case. In multi-gpu it may be the exact opposite where we split out the
// outer most iter domain to the multi-gpu dimension, then schedule.

TEST_F(NVFuserTest, FusionIndexSplitMerge_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // [w]
  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  // [w, x]
  auto tv1 = makeSymbolicTensor(2);
  fusion.addInput(tv1);

  auto tv2 = broadcast(tv0, {false, true});
  auto tv3 = add(tv1, tv2);
  fusion.addOutput(tv3);

  tv3->split(0, 3);
  tv3->split(2, 4);
  tv3->merge(1);
  tv3->split(1, 5);

  MaxRootDomainInfoSpanningTree tree(tv3);
  TransformPropagator tp(tv3);
  tree.traverse(&tp);

  inlineAllAt(tv3, 1, true);
  FusionExecutor fe;

  int x = 4, y = 7;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor t0 = at::randn({x}, options);
  at::Tensor t1 = at::randn({x, y}, options);

  auto t2 = t0.unsqueeze(-1);
  auto aten_output = t1.add(t2);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(
      &fusion, cg_outputs, aten_inputs, {aten_output}, __LINE__, __FILE__);
}

} // namespace nvfuser
