// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ops/all_ops.h>
#include <tests/cpp/utils.h>

namespace nvfuser {

using RaggedIterDomainTest = NVFuserTest;

// Basic construction of RaggedIterDomain
TEST_F(RaggedIterDomainTest, Construction) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a TensorView to use as extents
  // This represents component sizes such as [3, 5, 2]
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Create RaggedIterDomain
  auto ragged_id = IrBuilder::create<RaggedIterDomain>(
      extents, IterType::Iteration, ParallelType::Serial);

  // Verify properties
  EXPECT_NE(ragged_id, nullptr);
  EXPECT_TRUE(ragged_id->isA<RaggedIterDomain>());
  EXPECT_EQ(ragged_id->getIterType(), IterType::Iteration);
  EXPECT_EQ(ragged_id->getParallelType(), ParallelType::Serial);
  EXPECT_EQ(ragged_id->extents(), extents);
  EXPECT_FALSE(ragged_id->isRFactorProduct());

  // Verify extent is not null (it's the sum of extents)
  EXPECT_NE(ragged_id->extent(), nullptr);

  // Verify ValType is RaggedIterDomain, not IterDomain
  EXPECT_EQ(ragged_id->vtype(), ValType::RaggedIterDomain);
  EXPECT_NE(ragged_id->vtype(), ValType::IterDomain);
  EXPECT_TRUE(ragged_id->getValType().has_value());
  EXPECT_EQ(ragged_id->getValType().value(), ValType::RaggedIterDomain);

  // Compare with a regular IterDomain to ensure different types
  auto regular_id =
      IterDomainBuilder(
          fusion.zeroVal(), IrBuilder::create<Val>(10L, DataType::Index))
          .build();
  EXPECT_EQ(regular_id->vtype(), ValType::IterDomain);
  EXPECT_NE(ragged_id->vtype(), regular_id->vtype());
}

// RaggedIterDomain with parallelization
TEST_F(RaggedIterDomainTest, Parallelization) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Create with TIDx parallelization
  auto ragged_parallel = IrBuilder::create<RaggedIterDomain>(
      extents, IterType::Iteration, ParallelType::TIDx);

  EXPECT_EQ(ragged_parallel->getParallelType(), ParallelType::TIDx);
  EXPECT_TRUE(ragged_parallel->isThreadDim());

  // Test that parallelize method works (inherited from IterDomain)
  ragged_parallel->parallelize(ParallelType::TIDy);
  EXPECT_EQ(ragged_parallel->getParallelType(), ParallelType::TIDy);
}

// sameAs comparison
TEST_F(RaggedIterDomainTest, SameAsComparison) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto extents1 = makeSymbolicTensor(1, DataType::Index);
  auto extents2 = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents1);
  fusion.addInput(extents2);

  auto ragged1 = IrBuilder::create<RaggedIterDomain>(
      extents1, IterType::Iteration, ParallelType::Serial);

  auto ragged3 = IrBuilder::create<RaggedIterDomain>(
      extents2, // Different extents
      IterType::Iteration,
      ParallelType::Serial);

  // Same object
  EXPECT_TRUE(ragged1->sameAs(ragged1));

  // Different extents
  EXPECT_FALSE(ragged1->sameAs(ragged3));

  // RaggedIterDomain vs regular IterDomain
  auto regular_id =
      IterDomainBuilder(
          fusion.zeroVal(), IrBuilder::create<Val>(10L, DataType::Index))
          .build();
  EXPECT_FALSE(ragged1->sameAs(regular_id));
}

// Printing/toString
TEST_F(RaggedIterDomainTest, Printing) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  auto ragged_id = IrBuilder::create<RaggedIterDomain>(
      extents, IterType::Iteration, ParallelType::TIDx);

  // Print it
  std::string str = ragged_id->toString();

  // Verify output contains expected elements
  EXPECT_NE(str.find("Ragged"), std::string::npos);
  EXPECT_NE(str.find("extents"), std::string::npos);

  // Also test toInlineString
  std::string inline_str = ragged_id->toInlineString();
  EXPECT_FALSE(inline_str.empty());
}

// Multi-dimensional extents tensor
TEST_F(RaggedIterDomainTest, MultiDimensionalExtents) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create 2D extents tensor for nested ragged structure
  auto extents_2d = makeSymbolicTensor(2, DataType::Index);
  fusion.addInput(extents_2d);

  auto ragged_nested = IrBuilder::create<RaggedIterDomain>(
      extents_2d, IterType::Iteration, ParallelType::Serial);

  EXPECT_TRUE(ragged_nested != nullptr);
  EXPECT_EQ(ragged_nested->extents(), extents_2d);
}

// Validation - null extents should fail
TEST_F(RaggedIterDomainTest, ValidationNullExtents) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Attempt to create with null extents should throw
  EXPECT_THROW(
      IrBuilder::create<RaggedIterDomain>(
          nullptr, // null extents
          IterType::Iteration,
          ParallelType::Serial),
      nvfuser::nvfError);
}

// Validation - non-integer extents should fail
TEST_F(RaggedIterDomainTest, ValidationNonIntegerExtents) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create float extents (should fail)
  auto float_extents = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(float_extents);

  EXPECT_THROW(
      IrBuilder::create<RaggedIterDomain>(
          float_extents, IterType::Iteration, ParallelType::Serial),
      nvfuser::nvfError);
}

// IterVisitor test - ensure graph traversal visits extents field
TEST_F(RaggedIterDomainTest, IterVisitor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create extents TensorView
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Create RaggedIterDomain
  auto ragged_id = IrBuilder::create<RaggedIterDomain>(
      extents, IterType::Iteration, ParallelType::Serial);

  // Collect all statements reachable from the RaggedIterDomain
  std::vector<Val*> from_vals = {ragged_id};
  auto all_stmts = StmtSort::getStmtsTo(from_vals, /*traverse_members=*/true);

  // Verify the extents TensorView is visited
  EXPECT_TRUE(
      std::find(all_stmts.begin(), all_stmts.end(), extents) != all_stmts.end())
      << "IterVisitor should traverse the extents_ field of RaggedIterDomain";
}

// Partition operation - basic test
TEST_F(RaggedIterDomainTest, PartitionBasic) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create input IterDomain
  auto input_id =
      IterDomainBuilder(
          fusion.zeroVal(), IrBuilder::create<Val>(-1, DataType::Index))
          .build();

  // Create a symbolic offset tensor
  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Partition the IterDomain
  auto [component_id, ragged_id] =
      RaggedIterDomain::partition(input_id, offsets);

  // Verify component IterDomain
  EXPECT_TRUE(component_id != nullptr);
  EXPECT_TRUE(component_id->isA<IterDomain>());
  EXPECT_FALSE(component_id->isA<RaggedIterDomain>());

  // Verify RaggedIterDomain
  EXPECT_TRUE(ragged_id != nullptr);
  EXPECT_TRUE(ragged_id->isA<RaggedIterDomain>());
  EXPECT_TRUE(ragged_id->extents() != nullptr);

  // Verify that a Partition expr was created
  EXPECT_TRUE(component_id->definition() != nullptr);
  EXPECT_TRUE(component_id->definition()->isA<Partition>());

  // Both outputs should have the same definition (the Partition expr)
  EXPECT_EQ(component_id->definition(), ragged_id->definition());

  // Verify the Partition expr structure
  auto partition_expr = component_id->definition()->as<Partition>();
  EXPECT_EQ(partition_expr->component(), component_id);
  EXPECT_EQ(partition_expr->ragged(), ragged_id);
  EXPECT_EQ(partition_expr->in(), input_id);
  EXPECT_EQ(partition_expr->extents(), ragged_id->extents());

  // Verify the expr has correct inputs and outputs
  EXPECT_EQ(partition_expr->inputs().size(), 1);
  EXPECT_EQ(partition_expr->outputs().size(), 2);
  EXPECT_EQ(partition_expr->input(0), input_id);
  EXPECT_EQ(partition_expr->output(0), component_id);
  EXPECT_EQ(partition_expr->output(1), ragged_id);

  // Test toString
  std::string str = partition_expr->toString();
  EXPECT_TRUE(str.find("Partition") != std::string::npos);
}

// Partition operation - validation tests
TEST_F(RaggedIterDomainTest, PartitionValidation) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto input_id =
      IterDomainBuilder(
          fusion.zeroVal(), IrBuilder::create<Val>(10L, DataType::Index))
          .build();

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Test 1: Null input should fail
  EXPECT_THROW(
      RaggedIterDomain::partition(nullptr, offsets), nvfuser::nvfError);

  // Test 2: Null offsets should fail
  EXPECT_THROW(
      RaggedIterDomain::partition(input_id, nullptr), nvfuser::nvfError);

  // Test 3: Non-Index offsets should fail
  auto float_offsets = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(float_offsets);
  EXPECT_THROW(
      RaggedIterDomain::partition(input_id, float_offsets), nvfuser::nvfError);

  // Test 4: Multi-dimensional offsets should fail
  auto offsets_2d = makeSymbolicTensor(2, DataType::Index);
  fusion.addInput(offsets_2d);
  EXPECT_THROW(
      RaggedIterDomain::partition(input_id, offsets_2d), nvfuser::nvfError);

  // Test 5: Cannot partition RaggedIterDomain
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);
  auto ragged_id = IrBuilder::create<RaggedIterDomain>(
      extents, IterType::Iteration, ParallelType::Serial);
  EXPECT_THROW(
      RaggedIterDomain::partition(ragged_id, offsets), nvfuser::nvfError);
}

// TensorView::partition operation
TEST_F(RaggedIterDomainTest, TensorViewPartition) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D TensorView
  auto tv0 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(tv0);

  // Create offsets tensor
  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Partition the first axis
  tv0->partition(0, offsets);

  // Verify the tensor now has 3 dimensions: [component, ragged, original_dim1]
  EXPECT_EQ(tv0->nDims(), 3);

  // First axis should be a regular IterDomain (component)
  EXPECT_TRUE(tv0->axis(0)->isA<IterDomain>());

  // Second axis should be a RaggedIterDomain
  EXPECT_TRUE(tv0->axis(1)->isA<RaggedIterDomain>());

  // Third axis should be the original second dimension
  EXPECT_TRUE(tv0->axis(2)->isA<IterDomain>());

  // Verify both partition outputs have the same definition
  EXPECT_TRUE(tv0->axis(0)->definition() != nullptr);
  EXPECT_TRUE(tv0->axis(0)->definition()->isA<Partition>());
  EXPECT_EQ(tv0->axis(0)->definition(), tv0->axis(1)->definition());
}

} // namespace nvfuser
