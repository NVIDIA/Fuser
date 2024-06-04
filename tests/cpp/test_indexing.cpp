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

TEST_F(IndexingTest, Simple1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0));
  fusion.addOutput(tv1);

  tv1->merge(0);
  tv1->split(0, 4);

  IdModel id_model(&fusion);
  TensorIndexer indexer(id_model);

  std::cerr << indexer.getLinearIndex(tv1, tv1->definition())
                   ->toInlineString()
            << std::endl;
  std::cerr << indexer.getLinearIndex(tv0, tv1->definition())
                   ->toInlineString()
            << std::endl;
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

TEST_F(IndexingTest, Reshape) {
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

  IdModel id_model(&fusion, true, false, false);
}

} // namespace nvfuser
