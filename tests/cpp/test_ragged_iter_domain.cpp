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

  EXPECT_NE(ragged_nested, nullptr);
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

} // namespace nvfuser
