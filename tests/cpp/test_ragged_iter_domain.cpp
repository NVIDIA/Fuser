// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "fusion.h"
#include "ir/all_nodes.h"
#include "ir/builder.h"
#include "ops/all_ops.h"
#include "tests/cpp/utils.h"
#include "tests/cpp/validator.h"

namespace nvfuser {

using testing::ElementsAre;
using testing::NotNull;

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
      nvfError);
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
      nvfError);
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

  // Create a symbolic extents tensor
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Partition the IterDomain
  auto [component_id, ragged_id] =
      RaggedIterDomain::partition(input_id, extents);

  EXPECT_THAT(component_id, IsStrictlyA<IterDomain>());
  EXPECT_THAT(ragged_id, IsStrictlyA<RaggedIterDomain>());
  EXPECT_THAT(ragged_id->extents(), NotNull());

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

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Test 1: Null input should fail
  EXPECT_THROW(RaggedIterDomain::partition(nullptr, extents), nvfError);

  // Test 2: Null extents should fail
  EXPECT_THROW(RaggedIterDomain::partition(input_id, nullptr), nvfError);

  // Test 3: Non-Index extents should fail
  auto float_extents = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(float_extents);
  EXPECT_THROW(RaggedIterDomain::partition(input_id, float_extents), nvfError);

  // Test 4: Multi-dimensional extents should fail
  auto extents_2d = makeSymbolicTensor(2, DataType::Index);
  fusion.addInput(extents_2d);
  EXPECT_THROW(RaggedIterDomain::partition(input_id, extents_2d), nvfError);

  // Test 5: Non-Iteration IterType should fail
  auto reduction_id =
      IterDomainBuilder(
          fusion.zeroVal(), IrBuilder::create<Val>(10L, DataType::Index))
          .iter_type(IterType::Reduction)
          .build();
  EXPECT_THROW(RaggedIterDomain::partition(reduction_id, extents), nvfError);

  // Test 6: Cannot partition RaggedIterDomain
  auto extents2 = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents2);
  auto ragged_id = IrBuilder::create<RaggedIterDomain>(
      extents2, IterType::Iteration, ParallelType::Serial);
  EXPECT_THROW(RaggedIterDomain::partition(ragged_id, extents), nvfError);
}

// TensorView::partition operation
TEST_F(RaggedIterDomainTest, TensorViewPartition) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 2D TensorView
  auto tv0 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(tv0);

  // Create extents tensor
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Partition the first axis
  tv0->partition(0, extents);

  EXPECT_THAT(
      tv0->getLoopDomain(),
      ElementsAre(
          IsStrictlyA<IterDomain>(),
          IsStrictlyA<RaggedIterDomain>(),
          IsStrictlyA<IterDomain>()));

  // Verify both partition outputs have the same definition
  EXPECT_TRUE(tv0->axis(0)->definition() != nullptr);
  EXPECT_TRUE(tv0->axis(0)->definition()->isA<Partition>());
  EXPECT_EQ(tv0->axis(0)->definition(), tv0->axis(1)->definition());
}

// asNested basic functionality
TEST_F(RaggedIterDomainTest, AsNestedBasic) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Create nested tensor from dimension 0
  auto nested = asNested(data, extents, 0);

  fusion.addOutput(nested);

  // Verify the output is a new TensorView
  EXPECT_TRUE(nested != nullptr);
  EXPECT_NE(nested, data);
  EXPECT_TRUE(nested->isA<TensorView>());

  // Verify nested tensor has 3 dimensions: [component, ragged, original_dim1]
  EXPECT_EQ(nested->nDims(), 3);

  // First axis should be a regular IterDomain (component)
  EXPECT_TRUE(nested->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_FALSE(nested->axis(0)->isA<RaggedIterDomain>());

  // Second axis should be a RaggedIterDomain
  EXPECT_TRUE(nested->axis(1)->isA<RaggedIterDomain>());

  // Third axis should be the original second dimension
  EXPECT_TRUE(nested->axis(2)->isStrictlyA<IterDomain>());

  // Verify the definition exists (LoadStoreOp for aliasing)
  EXPECT_TRUE(nested->definition() != nullptr);
  EXPECT_TRUE(nested->definition()->isA<LoadStoreOp>());

  // Verify the component and ragged IterDomains have Partition as their
  // definition
  EXPECT_TRUE(nested->axis(0)->definition() != nullptr);
  EXPECT_TRUE(nested->axis(0)->definition()->isA<Partition>());
  EXPECT_EQ(nested->axis(0)->definition(), nested->axis(1)->definition());
}

// asNested on different dimensions
TEST_F(RaggedIterDomainTest, AsNestedDifferentDimension) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(3, DataType::Float);
  fusion.addInput(data);

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Partition dimension 1 (middle dimension)
  auto nested = asNested(data, extents, 1);

  EXPECT_THAT(
      nested->getLoopDomain(),
      ElementsAre(
          IsStrictlyA<IterDomain>(),
          IsStrictlyA<IterDomain>(),
          IsStrictlyA<RaggedIterDomain>(),
          IsStrictlyA<IterDomain>()));
}

// asNested with 1D tensor
TEST_F(RaggedIterDomainTest, AsNested1DTensor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create a 1D TensorView [10]
  auto data = makeSymbolicTensor(1, DataType::Float);
  fusion.addInput(data);

  // Create extents tensor
  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Create nested tensor from the only dimension
  auto nested = asNested(data, extents, 0);

  fusion.addOutput(nested);

  EXPECT_THAT(
      nested->getLoopDomain(),
      ElementsAre(IsStrictlyA<IterDomain>(), IsStrictlyA<RaggedIterDomain>()));
}

// asNested validation - null data
TEST_F(RaggedIterDomainTest, AsNestedValidationNullData) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto extents = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(extents);

  // Null data should throw
  EXPECT_THROW(asNested(nullptr, extents, 0), nvfError);
}

// asNested validation - null extents
TEST_F(RaggedIterDomainTest, AsNestedValidationNullExtents) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  // Null extents should throw
  EXPECT_THROW(asNested(data, nullptr, 0), nvfError);
}

// asNested validation - multi-dimensional extents (not yet supported)
TEST_F(RaggedIterDomainTest, AsNestedValidationMultiDimExtents) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  // 2D extents should fail (only 1D supported currently)
  auto extents_2d = makeSymbolicTensor(2, DataType::Index);
  fusion.addInput(extents_2d);

  EXPECT_THROW(asNested(data, extents_2d, 0), nvfError);
}

TEST_F(RaggedIterDomainTest, LoadStoreWithNestedTensor) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor from dimension 0
  auto nested = asNested(data, offsets, 0);

  // This should still be a nested tensor
  auto copy_of_nested = set(nested);

  fusion.addOutput(copy_of_nested);

  // Verify the output is a new TensorView
  EXPECT_TRUE(copy_of_nested != nullptr);
  EXPECT_NE(copy_of_nested, data);
  EXPECT_TRUE(copy_of_nested->isA<TensorView>());

  // Verify copy_of_nested tensor has 3 dimensions: [component, ragged,
  // original_dim1]
  EXPECT_EQ(copy_of_nested->nDims(), 3);

  // First axis should be a regular IterDomain (component)
  EXPECT_TRUE(copy_of_nested->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_FALSE(copy_of_nested->axis(0)->isA<RaggedIterDomain>());

  // Second axis should be a RaggedIterDomain
  EXPECT_TRUE(copy_of_nested->axis(1)->isA<RaggedIterDomain>());

  // Third axis should be the original second dimension
  EXPECT_TRUE(copy_of_nested->axis(2)->isStrictlyA<IterDomain>());

  // The copy of the original copy_of_nested tensor does not inherit the
  // Partition op
  EXPECT_TRUE(copy_of_nested->axis(0)->definition() == nullptr);
}

// Test binary operations with nested tensors
TEST_F(RaggedIterDomainTest, BinaryOpWithNestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create two 2D input tensors
  auto data1 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data1);

  auto data2 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data2);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensors from both inputs
  auto nested1 = asNested(data1, offsets, 0);
  auto nested2 = asNested(data2, offsets, 0);

  // Perform binary operation. The result should be a nested tensor
  auto result = add(nested1, nested2);

  fusion.addOutput(result);

  // Verify the result has 3 dimensions: [component, ragged, original_dim1]
  EXPECT_EQ(result->nDims(), 3);

  // First axis should be a regular IterDomain (component)
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_FALSE(result->axis(0)->isA<RaggedIterDomain>());

  // Second axis should be a RaggedIterDomain
  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());

  // Third axis should be the original second dimension
  EXPECT_TRUE(result->axis(2)->isStrictlyA<IterDomain>());
}

// Test binary operation with mixed inputs (one ragged, one not) - should error
TEST_F(RaggedIterDomainTest, BinaryOpMixedInputsError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data1 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data1);

  auto data2 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data2);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor from first input only
  auto nested1 = asNested(data1, offsets, 0);

  // Try to add nested tensor with non-nested tensor
  // This should fail because one is ragged and one is not
  EXPECT_THROW(add(nested1, data2), nvfuser::nvfError);
}

// Test binary operation with different offsets
TEST_F(RaggedIterDomainTest, BinaryOpDifferentRaggedStructures) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data1 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data1);

  auto data2 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data2);

  auto offsets1 = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets1);

  auto offsets2 = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets2);

  // Create nested tensors with different offset tensors
  auto nested1 = asNested(data1, offsets1, 0);
  auto nested2 = asNested(data2, offsets2, 0);

  // This would be an error if, for example, the values of the offset
  // tensors are not equivalent, but, like binary ops with normal
  // tensors, we assume their shapes are indeed compatible
  auto result = add(nested1, nested2);
  fusion.addOutput(result);

  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());
}

// Test unary operations with nested tensors
TEST_F(RaggedIterDomainTest, UnaryOpWithNestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor
  auto nested = asNested(data, offsets, 0);

  // Perform unary operation: neg
  auto result = neg(nested);

  fusion.addOutput(result);

  // Verify the result preserves RaggedIterDomain structure
  EXPECT_EQ(result->nDims(), 3);
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());
  EXPECT_TRUE(result->axis(2)->isStrictlyA<IterDomain>());
}

// Test broadcast with nested tensors
TEST_F(RaggedIterDomainTest, BroadcastWithNestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  auto result = broadcast(nested, {false, false, false, true});

  fusion.addOutput(result);

  // Result should be: [component, ragged, dim1, broadcast_dim]
  EXPECT_EQ(result->nDims(), 4);
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());
  EXPECT_TRUE(result->axis(2)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(3)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(3)->isBroadcast());
}

// Test squeeze on non-ragged dimension
TEST_F(RaggedIterDomainTest, SqueezeNonRaggedDim) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // First broadcast to add a dimension: [component, ragged, dim1, 1]
  auto broadcasted = broadcast(nested, {false, false, false, true});

  // Then squeeze the broadcast dimension (dimension index 3)
  auto result = squeeze(broadcasted, {3});

  fusion.addOutput(result);

  // Result should be: [component, ragged, dim1]
  EXPECT_EQ(result->nDims(), 3);
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());
  EXPECT_TRUE(result->axis(2)->isStrictlyA<IterDomain>());
}

// Test unsqueeze with nested tensors
TEST_F(RaggedIterDomainTest, UnsqueezeWithNestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Unsqueeze to add dimension at the end
  auto result = unsqueeze(nested, -1);

  fusion.addOutput(result);

  // Result should be: [component, ragged, dim1, 1]
  EXPECT_EQ(result->nDims(), 4);
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(1)->isA<RaggedIterDomain>());
  EXPECT_TRUE(result->axis(2)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(3)->isStrictlyA<IterDomain>());
}

// Test permute/transpose with nested tensors
TEST_F(RaggedIterDomainTest, PermuteWithNestedTensors) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Permute dimensions: swap ragged and dim1
  auto result = permute(nested, {0, 2, 1});

  fusion.addOutput(result);

  // Result should be: [component, dim1, ragged]
  EXPECT_EQ(result->nDims(), 3);
  EXPECT_TRUE(result->axis(0)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(1)->isStrictlyA<IterDomain>());
  EXPECT_TRUE(result->axis(2)->isA<RaggedIterDomain>());
}

// Test reduction on non-ragged dimension
TEST_F(RaggedIterDomainTest, ReductionOnNonRaggedDim) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Reduce along the last dimension (non-ragged)
  auto result = sum(nested, {2});

  fusion.addOutput(result);

  // Result should be: [component, ragged]
  // Get non-reduction dimensions
  auto non_reduction_domain =
      TensorDomain::noReductions(result->getLogicalDomain());

  EXPECT_EQ(non_reduction_domain.size(), 2);
  EXPECT_TRUE(non_reduction_domain[0]->isStrictlyA<IterDomain>());
  EXPECT_TRUE(non_reduction_domain[1]->isA<RaggedIterDomain>());
}

// Test reduction on ragged dimension - should error
TEST_F(RaggedIterDomainTest, ReductionOnRaggedDimError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to reduce along the ragged dimension (axis 1)
  // This should throw an error because reducing RaggedIterDomain is not allowed
  EXPECT_THROW(sum(nested, {1}), nvfuser::nvfError);
}

// Test reduction on component dimension - should error (TODO)
TEST_F(RaggedIterDomainTest, ReductionOnComponentDimError) {
  GTEST_SKIP() << "TODO: Implement validation to prevent reduction of "
                  "component dimension. "
               << "Currently there is no explicit marking of which IterDomains "
                  "are component dimensions, "
               << "so this validation cannot be implemented yet.";

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to reduce along the component dimension (axis 0)
  // This should throw an error because reducing component dimensions is not
  // allowed The component dimension defines the batch structure of the ragged
  // tensor, and reducing it would destroy the ragged structure
  EXPECT_THROW(sum(nested, {0}), nvfuser::nvfError);
}

// Test reshape with nested tensors - should error
TEST_F(RaggedIterDomainTest, ReshapeWithNestedTensorsError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to reshape - this should throw an error because reshape is not
  // supported for tensors with RaggedIterDomain
  std::vector<Val*> new_shape = {
      IrBuilder::create<Val>(-1L, DataType::Index), nested->axis(2)->extent()};
  EXPECT_THROW(reshape(nested, new_shape), nvfuser::nvfError);
}

// Test flatten with nested tensors - should error
TEST_F(RaggedIterDomainTest, FlattenWithNestedTensorsError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to flatten - this should throw an error because flatten is not
  // supported for tensors with RaggedIterDomain
  EXPECT_THROW(flatten(nested, 0, 2), nvfuser::nvfError);
}

// Test slice on ragged dimension - should error
TEST_F(RaggedIterDomainTest, SliceRaggedDimensionError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to slice the ragged dimension (axis 1)
  // This should error because resize on RaggedIterDomain is not allowed
  EXPECT_THROW(
      slice(
          nested,
          {{fusion.zeroVal(), fusion.oneVal()},
           {fusion.zeroVal(), fusion.oneVal()},
           {fusion.zeroVal(), nested->axis(2)->extent()}}),
      nvfuser::nvfError);
}

// Test cat on ragged dimension - should error
TEST_F(RaggedIterDomainTest, CatRaggedDimensionError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data1 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data1);

  auto data2 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data2);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensors with same structure
  auto nested1 = asNested(data1, offsets, 0);
  auto nested2 = asNested(data2, offsets, 0);

  // Try to concatenate along ragged dimension (axis 1)
  // This should error because cat would need to resize RaggedIterDomain
  EXPECT_THROW(cat({nested1, nested2}, 1), nvfuser::nvfError);
}

// Test cat on non-ragged dimension - currently also errors
TEST_F(RaggedIterDomainTest, CatNonRaggedDimensionError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data1 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data1);

  auto data2 = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data2);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensors with same structure
  auto nested1 = asNested(data1, offsets, 0);
  auto nested2 = asNested(data2, offsets, 0);

  // Try to concatenate along non-ragged dimension (axis 2)
  // Currently cat rejects all tensors with RaggedIterDomain for safety
  // In the future, this could be supported if concatenating along non-ragged
  // dimensions
  EXPECT_THROW(cat({nested1, nested2}, 2), nvfuser::nvfError);
}

// Test pad on ragged dimension - should error
TEST_F(RaggedIterDomainTest, PadRaggedDimensionError) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto data = makeSymbolicTensor(2, DataType::Float);
  fusion.addInput(data);

  auto offsets = makeSymbolicTensor(1, DataType::Index);
  fusion.addInput(offsets);

  // Create nested tensor: [component, ragged, dim1]
  auto nested = asNested(data, offsets, 0);

  // Try to pad the ragged dimension (axis 1)
  // This should error because pad uses resize on RaggedIterDomain
  std::vector<Val*> pad_widths = {
      fusion.zeroVal(),
      fusion.zeroVal(), // component: no padding
      fusion.oneVal(),
      fusion.oneVal(), // ragged: PADDING - should error
      fusion.zeroVal(),
      fusion.zeroVal() // dim1: no padding
  };

  EXPECT_THROW(pad(nested, pad_widths), nvfuser::nvfError);
}

} // namespace nvfuser
