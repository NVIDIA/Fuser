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

#include <dispatch.h>
#include <ir/builder.h>
#include <ir/interface_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/tools/inlining.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <functional>

namespace nvfuser {

// Test that OptOutMutator mutates expressions in a predictable way
// See https://github.com/NVIDIA/Fuser/issues/852
TEST_F(NVFuserTest, OptOutMutatorMutatedOutput) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = neg(tv0);

  auto tv2 = set(tv1);
  fusion->addOutput(tv2);

  auto tv3 = set(tv0);

  OptOutMutator mut;
  mut.registerMutation(tv1, tv3);

  for (auto stmt : StmtSort::getStmts(fusion)) {
    mut.dispatchMutate(stmt);
  }

  EXPECT_NE(tv3->definition(), nullptr);
  EXPECT_TRUE(tv3->definition()->isA<LoadStoreOp>());
  EXPECT_NE(tv2->definition(), nullptr);
  EXPECT_TRUE(tv2->definition()->isA<LoadStoreOp>());
  EXPECT_EQ(tv2->definition()->input(0), tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({3}, options);

  inlineMost();

  KernelExecutor ke;
  ke.compile(fusion);

  auto outputs = ke.run({t0});

  testValidate(fusion, outputs, {t0}, __LINE__, __FILE__);
}

// Another test related to https://github.com/NVIDIA/Fuser/issues/852
TEST_F(NVFuserTest, OptOutMutatorRedefinedConstant) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto s0 = IrBuilder::create<Val>(DataType::Int);
  fusion->addInput(s0);
  auto s1 = neg(s0);

  auto tv0 = full({IrBuilder::create<Val>(2L)}, s1, DataType::Int);
  fusion->addOutput(tv0);

  // After the following mutation, it's reasonable to expect the input scalar s0
  // to be ignored, and the output to just be ones.
  OptOutMutator mut;
  auto c = fusion->oneVal(DataType::Int);
  mut.registerMutation(s1, c);

  for (auto stmt : StmtSort::getStmts(fusion)) {
    mut.dispatchMutate(stmt);
  }

  EXPECT_EQ(
      c->definition(), nullptr); // Replacement value should not be redefined
  EXPECT_EQ(tv0->definition()->as<FullOp>()->getFillValue(), c);

  inlineMost();

  KernelExecutor ke;
  ke.compile(fusion);

  auto outputs = ke.run({3L});

  testValidate(fusion, outputs, {3L}, __LINE__, __FILE__);
}

// Test that additional IDs are preserved when mutating a TensorView
TEST_F(NVFuserTest, OptOutMutatorAdditionalBroadcastID) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion* fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  auto tv0 = makeSymbolicTensor(1);
  fusion->addInput(tv0);

  auto tv1 = exp(tv0);

  fusion->addOutput(tv1);

  // We add a broadcast domain bS2{1}. This adds the new Broadcast ID to
  // tv1->domain()->additionalIDs() logical: [ iS1{i0} ] loop: [ iS1{i0}, bS2{1}
  // ] additional IDs: [ bS2{1} ]
  tv1->broadcast(1);
  EXPECT_FALSE(tv1->domain()->additionalIDs().empty());

  // After this split we have
  // logical: [ iS1{i0} ]
  // loop: [ iS1{i0}, bS3{1}, bS4{2} ]
  // additional IDs: [ bS2{1} ]
  tv1->split(1, 2);
  EXPECT_FALSE(tv1->domain()->additionalIDs().empty());

  // Now register a mutation that will alter some IDs in the domain
  OptOutMutator mut;
  mut.registerMutation(
      tv1->axis(0)->extent(), IrBuilder::create<Val>(DataType::Index));
  TensorDomain* old_tensor_domain = tv1->domain();
  auto all_stmts = StmtSort::getStmts(
      fusion,
      /*traverse_members*/ true,
      /*traverse_attributes*/ true,
      /*traverse_siblings*/ true);
  for (auto stmt : all_stmts) {
    mut.dispatchMutate(stmt);
  }
  EXPECT_TRUE(tv1->domain() != old_tensor_domain)
      << "Mutation did not change the TensorDomain";

  EXPECT_FALSE(tv1->domain()->additionalIDs().empty())
      << "Mutation did not preserve additional IDs";
}

} // namespace nvfuser
