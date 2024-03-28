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

  // int w = 3, x = 4, y = 7, z = 8;
  // auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

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

  IdModel id_model(&fusion);
  TensorIndexer indexing(id_model);

  for (auto expr : fusion.exprs()) {
    std::cerr << expr->toString();

    for (auto tv_out : ir_utils::filterByType<TensorView>(expr->outputs())) {
      std::cerr << "Consumer indexing of " << tv_out->toString() << std::endl;
      indexing.getIndex(tv_out, expr);
    }
    for (auto tv_inp : ir_utils::filterByType<TensorView>(expr->inputs())) {
      std::cerr << "Producer indexing of " << tv_inp->toString() << std::endl;
      indexing.getIndex(tv_inp, expr);
    }
  }

#if 0

  FusionExecutor fe;

  at::Tensor t0 = at::randn({x, y, z}, options);
  at::Tensor t1 = at::randn({w, x, y, z}, options);

  std::vector<c10::IValue> aten_inputs = {t0, t1};

  fe.compileFusion(&fusion, aten_inputs);
  auto cg_outputs = fe.runFusion(aten_inputs);

  testValidate(&fusion, cg_outputs, aten_inputs, __LINE__, __FILE__);
#endif
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

} // namespace nvfuser
