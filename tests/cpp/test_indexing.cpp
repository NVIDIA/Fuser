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

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <fusion.h>
#include <id_model/id_model.h>
#include <id_model/indexing.h>
#include <id_model/to_string.h>
#include <inlining.h>
#include <ops/all_ops.h>
#include <transform_iter.h>
#include <val_graph_visitor.h>

namespace nvfuser {

using IndexingTest = NVFuserTest;

// Copied from NvFuserTest.FusionIndexing1_CUDA
TEST_F(IndexingTest, Test1) {
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

  fusion.print();
  fusion.printKernel();

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, TMP) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // int w = 3, x = 4, y = 7, z = 8;

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv3);

  tv2->inlineAt(1);
  tv2->setMemoryType(MemoryType::Global);

  fusion.print();
  fusion.printKernel();

  FusionExecutor fe;

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3, 4}, options);
  std::vector<c10::IValue> aten_inputs = {t0};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  // testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Moved from NvFuserTest.FusionIndexing1
TEST_F(IndexingTest, Indexing1) {
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

  fusion.printKernel();

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

// Same as 1 but merge starting from inner most dimension
// Moved from NvFuserTest.FusionIndexing2
TEST_F(IndexingTest, Indexing2) {
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
// Moved from NvFuserTest.FusionIndexing2
TEST_F(IndexingTest, Indexing3) {
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

TEST_F(IndexingTest, Indexing4) {
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

TEST_F(IndexingTest, Indexing5) {
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

TEST_F(IndexingTest, Indexing6) {
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

TEST_F(IndexingTest, Indexing7) {
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

TEST_F(IndexingTest, Indexing8) {
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
TEST_F(IndexingTest, IndexingTest9) {
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

TEST_F(IndexingTest, Indexing10) {
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

TEST_F(IndexingTest, Indexing11) {
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

TEST_F(IndexingTest, Indexing12) {
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

TEST_F(IndexingTest, Indexing13) {
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

TEST_F(IndexingTest, Indexing14) {
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
TEST_F(IndexingTest, Indexing15) {
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

TEST_F(IndexingTest, Indexing16) {
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

TEST_F(IndexingTest, Indexing17) {
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
TEST_F(IndexingTest, Indexing18) {
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
TEST_F(IndexingTest, Indexing19) {
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

  // The current ComputeAtMap fails with this fusion
  // fusion.printKernel();
}

// Progressive loop promotion. producer gets promoted in consumer, consumer is
// promoted in a different way to its consumer.
TEST_F(IndexingTest, Indexing20) {
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

  IdModel id_model(&fusion);
  const auto& promotion_map = id_model.loopPromotionMap();

  // For tv1, tv2, tv4, their first leaf domains should all be
  // loop-mapped and promoted to a domain that is exaclty mapped with
  // the first leaf domain of tv7. The second leaf domains should also
  // be promoted to the domain at the same position in tv7 but since
  // they are not inlined, they should not be loop-mapped
  for (auto tv : {tv1, tv2, tv4}) {
    // Validating the first leaf ID
    {
      auto leaf_id0 = tv->axis(0);
      auto ref_id0 = tv7->axis(0);
      ASSERT_TRUE(id_model.idGraph(IdMappingMode::LOOP)
                      .disjointValSets()
                      .strictAreMapped(leaf_id0, ref_id0));
      auto promotion_map_it = promotion_map.find(
          id_model.idGraph(IdMappingMode::LOOP).toGroup(leaf_id0));
      ASSERT_NE(promotion_map_it, promotion_map.end());
      auto promoted_id = promotion_map_it->second;
      ASSERT_TRUE(id_model.idGraph(IdMappingMode::EXACT)
                      .disjointValSets()
                      .strictAreMapped(promoted_id, ref_id0))
          << "Expected exact mapping: " << promoted_id->toString() << " with "
          << ref_id0->toString() << " of " << tv7->toString();
    }

    // Validating the second leaf ID
    {
      auto leaf_id1 = tv->axis(1);
      // Should be promoted to a domain that is exactly mapped with iS31
      auto ref_id1 = tv7->axis(1)
                         ->definition()
                         ->as<Split>()
                         ->in()
                         ->definition()
                         ->as<Merge>()
                         ->outer();
      auto promotion_map_it = promotion_map.find(
          id_model.idGraph(IdMappingMode::LOOP).toGroup(leaf_id1));
      ASSERT_NE(promotion_map_it, promotion_map.end());
      auto promoted_id = promotion_map_it->second;
      ASSERT_TRUE(id_model.idGraph(IdMappingMode::EXACT)
                      .disjointValSets()
                      .strictAreMapped(promoted_id, ref_id1))
          << "Expected exact mapping: " << promoted_id->toString() << " with "
          << ref_id1->toString() << " of " << tv7->toString();
      // While promoted ID should be exact-mapped with the reference ID, they
      // should not be loop-mapped
      ASSERT_FALSE(id_model.idGraph(IdMappingMode::LOOP)
                       .disjointValSets()
                       .strictAreMapped(promoted_id, ref_id1))
          << "Expected no loop mapping: " << promoted_id->toString() << " with "
          << ref_id1->toString() << " of " << tv7->toString();

      // In the case of tv1 and tv2, the promoted id is a newly replayed
      // domain, whereas for the tv4, there should be no replay as
      // there's no broadcast. So, the size of the loop group should be
      // 2 for the former and 1 for the latter.
      const auto& leaf_id1_loop_group =
          id_model.idGraph(IdMappingMode::LOOP).toGroup(leaf_id1);
      ASSERT_EQ(leaf_id1_loop_group->size(), tv == tv4 ? 1 : 2)
          << "Unexpected loop group: "
          << nvfuser::toString(leaf_id1_loop_group);
    }
  }

  // Validate tv5. The last leaf domain should be promoted to a domain
  // that is exactly mapped with the last domain of tv7
  {
    auto last_leaf = tv5->axis(-1);
    auto promotion_map_it = promotion_map.find(
        id_model.idGraph(IdMappingMode::LOOP).toGroup(last_leaf));
    ASSERT_NE(promotion_map_it, promotion_map.end());
    auto promoted_id = promotion_map_it->second;
    ASSERT_TRUE(id_model.idGraph(IdMappingMode::EXACT)
                    .disjointValSets()
                    .strictAreMapped(promoted_id, tv7->axis(-1)))
        << "Expected exact mapping: " << promoted_id->toString() << " with "
        << tv7->axis(-1)->toString() << " of " << tv7->toString();

    // While promoted ID should be exact-mapped with the last ID, they
    // should not be loop-mapped
    ASSERT_FALSE(id_model.idGraph(IdMappingMode::LOOP)
                     .disjointValSets()
                     .strictAreMapped(promoted_id, tv7->axis(-1)))
        << "Expected no loop maping: " << promoted_id->toString() << " with "
        << tv7->axis(-1)->toString() << " of " << tv7->toString();
  }

  // Validation not enabled yet as incorrect code is generated. Need
  // to use the loop promotion info to generate correct loop-nests
#if 0
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({5}, options);
  auto t3 = at::randn({3, 5}, options);
  auto t6 = at::randn({3, 5, 7}, options);
  std::vector<c10::IValue> aten_inputs = {t0, t3, t6};

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
#endif
}

// Repro for issue #1873
TEST_F(IndexingTest, InlineBroadcastIndexing0) {
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

  TransformPropagatorWithCheck propagator(tv4);
  MaxRootDomainInfoSpanningTree(tv4).traverse(&propagator);

  tv2->inlineAt(1);
  tv3->inlineAt(1);

  tv2->split(-1, 8);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({123}, options);
  at::Tensor t1 = at::randn({3, 123}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1});

  auto outputs = fe.runFusion({t0, t1});

  testValidate(&fusion, outputs, {t0, t1}, __LINE__, __FILE__);
}

// Broadcast inline 3 times and merge all domains
TEST_F(IndexingTest, MultiPromotion) {
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

// Broadcast and concretize same domain in two different ways and try to merge
// their loops. The inlining pattern is invalid but the current
// inlining check is not capable of flagging the inlining position as
// invalid. The loop promotion analysis should not find any promotion
// of the loop group where all the leaf domains are merged into.
TEST_F(IndexingTest, MultiPromotion2) {
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

  // Since x and y are not proven to be the same, this inling position
  // should not be allowed.
  for (auto tv : std::vector<TensorView*>{tv3, tv4, tv6}) {
    tv->inlineAt(1);
  }

  // For now, just make sure there's no loop promotion for the merged
  // loop group.
  IdModel id_model(&fusion);
  const auto& leaf_loop_group =
      id_model.idGraph(IdMappingMode::LOOP).toGroup(tv7->axis(0));
  auto promotion_map_it = id_model.loopPromotionMap().find(leaf_loop_group);
  ASSERT_EQ(promotion_map_it, id_model.loopPromotionMap().end());
}

// TODO: All the above tests are merges followed by splits, we should make some
// more complex examples even though merging then spliting is the most likely
// use case. In multi-gpu it may be the exact opposite where we split out the
// outer most iter domain to the multi-gpu dimension, then schedule.

TEST_F(IndexingTest, IndexSplitMerge) {
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

TEST_F(IndexingTest, Normalization) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  constexpr int ND = 3;
  auto tv0 = makeSymbolicTensor(ND);
  fusion.addInput(tv0);

  TensorView* tv2 = nullptr;
  tv2 = sum(tv0, {0, 2});
  auto tv3 = broadcast(tv2, {true, false, true});
  auto tv4 = sub(tv0, tv3);
  fusion.addOutput(tv4);

  Fusion fusion_copy = fusion;

  std::vector<int64_t> input_shape{20, 10, 35};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  if (getenv("CACHE")) {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto cg_outputs = executor_cache.runFusionWithInputs(aten_inputs);

  } else {

    auto persistent_params =
        getInnerPersistentHeuristics(&fusion, aten_inputs);
    NVF_CHECK(persistent_params, "Persistent schedule was not generated!");
    scheduleInnerPersistentKernel(&fusion, *persistent_params);

    fusion.printKernel();

    FusionExecutor fe;
    fe.compileFusion(&fusion, aten_inputs);
    auto cg_outputs = fe.runFusion(aten_inputs);
    testValidate(&fusion_copy, cg_outputs, aten_inputs, __LINE__, __FILE__);
  }
}

TEST_F(IndexingTest, Reshape1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape1({100});
  const std::vector<int64_t> shape2({4, 25});
  const std::vector<int64_t> shape3({5, 2, 10});

  // [i0]
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // [i2, i3]
  auto tv2 = reshape(tv1, shape1, shape2);

  // [i2, i3]
  auto tv3 = add(tv2, fusion.oneVal());

  // [i4, i5, i6]
  auto tv4 = reshape(tv3, shape2, shape3);

  // [i4, i5, i6]
  auto tv5 = add(tv4, fusion.oneVal());

  fusion.addOutput(tv5);


  TransformPropagator propagator(tv5);
  MaxRootDomainInfoSpanningTree(tv5).traverse(&propagator);

  inlineMost();

  fusion.print();

  //IdModel id_model(&fusion, true, false, false);

}

// Not a DAG due to a residual-like reshape path
TEST_F(IndexingTest, Reshape2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const std::vector<int64_t> shape1({100});
  const std::vector<int64_t> shape2({4, 25});
  const std::vector<int64_t> shape3({5, 2, 10});

  // [i0]
  auto tv0 = makeContigConcreteTensor(shape1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);

  // First path

  // [i2, i3]
  auto tv2 = reshape(tv1, shape1, shape2);

  // [i2, i3]
  auto tv3 = add(tv2, fusion.oneVal());

  // Second path

  // [i4, i5, i6]
  auto tv4 = reshape(tv1, shape1, shape3);

  // [i4, i5, i6]
  auto tv5 = add(tv4, fusion.oneVal());

  // [i2, i3]
  auto tv6 = reshape(tv5, shape3, shape2);

  // [i2, i3]
  auto tv7 = add(tv3, tv6);

  fusion.addOutput(tv7);

  // Schedule two tensor groups differently
  // {tv2, tv3, tv7}
  // {tv4, tv5, tv6}

  for (auto tv: {tv2, tv3, tv7}) {
    tv->merge(0);
    tv->split(0, 4);
    tv->split(0, 32);
  }

  for (auto tv: {tv4, tv5}) {
    tv->merge(0);
    tv->merge(0);
    tv->split(0, shape2[0], false);
  }
  for (auto tv: {tv4, tv5, tv6}) {
    tv->split(1, 4);
    tv->merge(0, 1);
    tv->split(0, 32);
  }

  for (auto tv : {tv2, tv3, tv4, tv5}) {
    tv->inlineAt(-1);
  }

  fusion.printMath();
  fusion.print();

  IdModel id_model(&fusion, true, false, false);

}

TEST_F(IndexingTest, Simple1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  tv1->split(0, 4);

  fusion.printKernel();

  std::vector<int64_t> input_shape{17};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, Simple2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  auto tv2 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv2->split(0, 8);

  tv1->axis(1)->parallelize(ParallelType::TIDx);
  tv2->axis(1)->parallelize(ParallelType::TIDx);

  fusion.printKernel();

  std::vector<int64_t> input_shape{17};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, Vectorize1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv2->split(0, 4);

  tv1->axis(1)->parallelize(ParallelType::Vectorize);
  tv2->axis(1)->parallelize(ParallelType::Vectorize);

  inlineMost();

  fusion.printKernel();

  std::vector<int64_t> input_shape{32};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, Unswitch1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  fusion.addOutput(tv2);

  tv1->split(0, 4);
  tv2->split(0, 4);

  tv2->axis(1)->parallelize(ParallelType::Unswitch);

  inlineMost();

  fusion.printKernel();

  std::vector<int64_t> input_shape{32};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, Unswitch2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->split(0, 4);
  tv1->split(0, 5);

  tv1->axis(1)->parallelize(ParallelType::Unswitch);
  tv1->axis(2)->parallelize(ParallelType::Unswitch);  

  inlineMost();

  fusion.printKernel();

  std::vector<int64_t> input_shape{32};

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({t0});

  FusionExecutor fe;
  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
}

TEST_F(IndexingTest, ContigIndexing1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(0);

  fusion.printMath();
  fusion.printKernel();
}

TEST_F(IndexingTest, ContigIndexing2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = TensorViewBuilder().ndims(3).dtype(DataType::Float).contiguity({true, true, false}).build();
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  fusion.addOutput(tv1);

  tv1->merge(1);

  fusion.printMath();
  fusion.printKernel();
}

TEST_F(IndexingTest, ContigIndexing3) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = makeContigTensor(2);
  fusion.addInput(tv1);

  auto tv2 = set(tv0);
  auto tv3 = broadcast(tv2, {true, false});
  auto tv4 = add(tv3, tv1);
  fusion.addOutput(tv4);

  tv4->merge(0);

  fusion.printMath();
  fusion.printKernel();
}

} // namespace nvfuser
