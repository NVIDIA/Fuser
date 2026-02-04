// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "fusion.h"
#include "ir/container.h"
#include "ops/all_ops.h"
#include "statement_guard.h"
#include "tests/cpp/utils.h"

namespace nvfuser {

// Test class for Phase 2 container sharing tests
class Phase2ContainerTest : public NVFuserTest {
 protected:
  void SetUp() override {
    NVFuserTest::SetUp();
  }
  void TearDown() override {
    NVFuserTest::TearDown();
  }
};

// =============================================================================
// Task 1 Tests: Locking Infrastructure
// =============================================================================

TEST_F(Phase2ContainerTest, LockingBasic) {
  // Verify basic operations still work with locking in place
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Verify container has expected contents
  // Use vals() and unordered_exprs() which return references to container data
  EXPECT_GT(fusion.vals().size(), 0);
  EXPECT_GT(fusion.unordered_exprs().size(), 0);
}

TEST_F(Phase2ContainerTest, ConcurrentReads) {
  // Multiple threads can read simultaneously without data races
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  std::vector<std::thread> threads;
  std::atomic<int> read_count{0};

  // Spawn multiple reader threads
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < 100; ++j) {
        // Access vals and unordered_exprs through fusion's forwarding methods
        // These return const references to the underlying container data
        const auto& vals = fusion.vals();
        const auto& exprs = fusion.unordered_exprs();
        // Just access sizes to verify no crashes under concurrent access
        (void)vals.size();
        (void)exprs.size();
        read_count++;
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(read_count.load(), 400);
}

// =============================================================================
// Task 2 Tests: Fusion Tracking Infrastructure
// =============================================================================

TEST_F(Phase2ContainerTest, FusionRegistration) {
  // Test that addFusion increments count, removeFusion decrements
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Get the IrContainer through Fusion
  auto& container = *fusion.ir_container();

  // Initially no Fusions registered (Phase 1 doesn't use registration yet)
  EXPECT_EQ(container.sharingCount(), 0);

  // Register the Fusion
  container.addFusion(&fusion);
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_FALSE(container.hasMultipleFusions());

  // Create another Fusion and register it with the same container
  // (simulating shared_ptr sharing that will happen in later tasks)
  Fusion fusion2;
  container.addFusion(&fusion2);
  EXPECT_EQ(container.sharingCount(), 2);
  EXPECT_TRUE(container.hasMultipleFusions());

  // Remove one
  container.removeFusion(&fusion2);
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_FALSE(container.hasMultipleFusions());

  // Remove the other
  container.removeFusion(&fusion);
  EXPECT_EQ(container.sharingCount(), 0);
}

TEST_F(Phase2ContainerTest, FusionTransfer) {
  // Test transferFusion correctly updates tracking
  Fusion fusion1;
  Fusion fusion2;

  auto& container = *fusion1.ir_container();

  // Register fusion1
  container.addFusion(&fusion1);
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_TRUE(container.sharingFusions().count(&fusion1) > 0);
  EXPECT_TRUE(container.sharingFusions().count(&fusion2) == 0);

  // Transfer from fusion1 to fusion2
  container.transferFusion(&fusion1, &fusion2);
  EXPECT_EQ(container.sharingCount(), 1);
  EXPECT_TRUE(container.sharingFusions().count(&fusion1) == 0);
  EXPECT_TRUE(container.sharingFusions().count(&fusion2) > 0);
}

TEST_F(Phase2ContainerTest, MultipleRegistration) {
  // Test multiple Fusions can register with same container
  Fusion fusion1;
  Fusion fusion2;
  Fusion fusion3;

  auto& container = *fusion1.ir_container();

  container.addFusion(&fusion1);
  container.addFusion(&fusion2);
  container.addFusion(&fusion3);

  EXPECT_EQ(container.sharingCount(), 3);
  EXPECT_TRUE(container.hasMultipleFusions());

  // Verify all are registered
  const auto& fusions = container.sharingFusions();
  EXPECT_TRUE(fusions.count(&fusion1) > 0);
  EXPECT_TRUE(fusions.count(&fusion2) > 0);
  EXPECT_TRUE(fusions.count(&fusion3) > 0);
}

TEST_F(Phase2ContainerTest, StatementCleanup) {
  // Test that removeFusion removes only Statements owned by that Fusion
  // This is tricky to test directly because Statements are tied to their
  // container at construction. We test the basic mechanism works.

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();
  size_t initial_vals = container.vals().size();
  size_t initial_exprs = container.unordered_exprs().size();

  EXPECT_GT(initial_vals, 0);
  EXPECT_GT(initial_exprs, 0);

  // Register fusion
  container.addFusion(&fusion);

  // When we remove fusion, its Statements should be cleaned up
  // (all Statements in this test are owned by fusion)
  container.removeFusion(&fusion);

  // After removal, the Statements owned by fusion should be removed
  EXPECT_EQ(container.vals().size(), 0);
  EXPECT_EQ(container.unordered_exprs().size(), 0);
}

// =============================================================================
// Task 4 Tests: Per-Fusion Statement Tracking
// =============================================================================

TEST_F(Phase2ContainerTest, PerFusionValsTracking) {
  // Test that ownedVals() returns only this Fusion's vals
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // ownedVals() should return only this Fusion's vals
  const auto& owned_vals = fusion.ownedVals();
  EXPECT_GT(owned_vals.size(), 0);

  // All vals in ownedVals() should have container() == &fusion
  for (auto* val : owned_vals) {
    EXPECT_EQ(val->container(), &fusion);
  }

  // vals() and ownedVals() should be the same with a single Fusion (Phase 1
  // equivalence)
  EXPECT_EQ(fusion.vals().size(), fusion.ownedVals().size());
}

TEST_F(Phase2ContainerTest, PerFusionExprsTracking) {
  // Test that ownedExprs() returns only this Fusion's exprs
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // ownedExprs() should return only this Fusion's exprs
  const auto& owned_exprs = fusion.ownedExprs();
  EXPECT_GT(owned_exprs.size(), 0);

  // All exprs in ownedExprs() should have container() == &fusion
  for (auto* expr : owned_exprs) {
    EXPECT_EQ(expr->container(), &fusion);
  }

  // unordered_exprs() and ownedExprs() should be the same with a single Fusion
  EXPECT_EQ(fusion.unordered_exprs().size(), fusion.ownedExprs().size());
}

TEST_F(Phase2ContainerTest, ValsOwnedByAPI) {
  // Test IrContainer::valsOwnedBy() API directly
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // valsOwnedBy should return same set as ownedVals()
  const auto& vals_by_container = container.valsOwnedBy(&fusion);
  const auto& vals_by_fusion = fusion.ownedVals();
  EXPECT_EQ(vals_by_container.size(), vals_by_fusion.size());

  // valsOwnedBy for a non-registered Fusion should return empty set
  Fusion other_fusion;
  const auto& other_vals = container.valsOwnedBy(&other_fusion);
  EXPECT_EQ(other_vals.size(), 0);
}

TEST_F(Phase2ContainerTest, ExprsOwnedByAPI) {
  // Test IrContainer::exprsOwnedBy() API directly
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // exprsOwnedBy should return same set as ownedExprs()
  const auto& exprs_by_container = container.exprsOwnedBy(&fusion);
  const auto& exprs_by_fusion = fusion.ownedExprs();
  EXPECT_EQ(exprs_by_container.size(), exprs_by_fusion.size());

  // exprsOwnedBy for a non-registered Fusion should return empty set
  Fusion other_fusion;
  const auto& other_exprs = container.exprsOwnedBy(&other_fusion);
  EXPECT_EQ(other_exprs.size(), 0);
}

TEST_F(Phase2ContainerTest, RegisterUpdatesPerFusionTracking) {
  // Test that registering new vals/exprs updates per-Fusion tracking
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Initially no vals
  EXPECT_EQ(fusion.ownedVals().size(), 0);
  EXPECT_EQ(fusion.ownedExprs().size(), 0);

  // Add an input - this creates vals
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  // Now we should have vals tracked for this fusion
  size_t vals_after_input = fusion.ownedVals().size();
  EXPECT_GT(vals_after_input, 0);

  // Add an expression - this creates more vals and exprs
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Both should have grown
  EXPECT_GT(fusion.ownedVals().size(), vals_after_input);
  EXPECT_GT(fusion.ownedExprs().size(), 0);
}

TEST_F(Phase2ContainerTest, TransferStatementOwnership) {
  // Test IrContainer::transferStatementOwnership
  auto container = std::make_shared<IrContainer>();

  // Create dummy Fusions for testing
  Fusion fusion1;
  Fusion fusion2;

  // We can't easily create vals owned by fusion1 in a standalone container,
  // but we can test the tracking data structure directly
  container->addFusion(&fusion1);
  container->addFusion(&fusion2);

  // Transfer ownership - should not crash even with empty tracking
  container->transferStatementOwnership(&fusion1, &fusion2);

  // Verify fusion1 no longer has tracking entries (empty case)
  EXPECT_EQ(container->valsOwnedBy(&fusion1).size(), 0);
  EXPECT_EQ(container->exprsOwnedBy(&fusion1).size(), 0);

  // Cleanup
  container->removeFusion(&fusion1);
  container->removeFusion(&fusion2);
}

TEST_F(Phase2ContainerTest, ClearOnlyAffectsOwnedStatements) {
  // Test that Fusion::clear() only clears THIS Fusion's statements
  // This is critical for shared container correctness

  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  // Get container reference
  auto container_ptr = fusion.ir_container_ptr();

  // Record counts before clear
  size_t vals_before = fusion.ownedVals().size();
  size_t exprs_before = fusion.ownedExprs().size();
  EXPECT_GT(vals_before, 0);
  EXPECT_GT(exprs_before, 0);

  // Clear the fusion
  fusion.clear();

  // After clear, ownedVals/ownedExprs should be empty for this fusion
  EXPECT_EQ(fusion.ownedVals().size(), 0);
  EXPECT_EQ(fusion.ownedExprs().size(), 0);

  // Container-level accessors should also reflect the removal
  EXPECT_EQ(container_ptr->vals().size(), 0);
  EXPECT_EQ(container_ptr->unordered_exprs().size(), 0);
}

TEST_F(Phase2ContainerTest, RemoveStatementsOwnedByAPI) {
  // Test public IrContainer::removeStatementsOwnedBy API
  Fusion fusion;
  FusionGuard fg(&fusion);

  // Create some IR
  auto* tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto& container = *fusion.ir_container();

  // Verify we have statements
  EXPECT_GT(container.vals().size(), 0);
  EXPECT_GT(container.unordered_exprs().size(), 0);
  EXPECT_GT(container.valsOwnedBy(&fusion).size(), 0);
  EXPECT_GT(container.exprsOwnedBy(&fusion).size(), 0);

  // Clear fusion-level state first (inputs_, outputs_, etc.)
  // Note: We're testing the container API directly, not through Fusion::clear()
  // In practice, Fusion::clear() does both
  container.removeStatementsOwnedBy(&fusion);

  // After removal, tracking should be empty
  EXPECT_EQ(container.valsOwnedBy(&fusion).size(), 0);
  EXPECT_EQ(container.exprsOwnedBy(&fusion).size(), 0);

  // Container-level sets should also be empty (single fusion case)
  EXPECT_EQ(container.vals().size(), 0);
  EXPECT_EQ(container.unordered_exprs().size(), 0);
}

// =============================================================================
// Task 7 Tests: Per-Fusion Special Values
// =============================================================================

TEST_F(Phase2ContainerTest, PerFusionSpecialValuesBasic) {
  // Test that special values are created per-Fusion
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();
  Val* one_a = a.oneVal();

  EXPECT_NE(zero_a, nullptr);
  EXPECT_NE(one_a, nullptr);
  EXPECT_EQ(zero_a->container(), &a);
  EXPECT_EQ(one_a->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesOwnedByFusion) {
  // Test that special values are tracked in ownedVals
  Fusion a;
  FusionGuard fg_a(&a);

  Val* zero_a = a.zeroVal();

  // Special values should be in ownedVals
  EXPECT_TRUE(a.ownedVals().count(zero_a) > 0);
}

TEST_F(Phase2ContainerTest, SeparateFusionsHaveOwnSpecialValues) {
  // Two independent Fusions should have different special values
  Fusion a;
  Fusion b;

  {
    FusionGuard fg_a(&a);
    Val* zero_a = a.zeroVal();
    EXPECT_EQ(zero_a->container(), &a);
  }

  {
    FusionGuard fg_b(&b);
    Val* zero_b = b.zeroVal();
    EXPECT_EQ(zero_b->container(), &b);
  }

  // Each has its own zero (different objects)
  EXPECT_NE(a.zeroVal(), b.zeroVal());
}

TEST_F(Phase2ContainerTest, DestroyFusionDoesNotAffectOther) {
  // Destroying one Fusion should not affect another's special values
  Fusion a;
  FusionGuard fg_a(&a);

  // Create special values in a
  Val* zero_a = a.zeroVal();
  EXPECT_NE(zero_a, nullptr);

  {
    Fusion b;
    FusionGuard fg_b(&b);
    Val* zero_b = b.zeroVal();
    EXPECT_NE(zero_b, nullptr);
    // b destroyed here
  }

  // a should still work fine - its special values should still be valid
  Val* zero_a_again = a.zeroVal();
  EXPECT_EQ(zero_a_again, zero_a);
  EXPECT_EQ(zero_a_again->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesLazyCreation) {
  // Special values should be created lazily
  Fusion a;
  FusionGuard fg_a(&a);

  // Before calling zeroVal(), it shouldn't exist
  // (Can't directly test this, but we can verify it works after call)
  Val* zero1 = a.zeroVal();
  Val* zero2 = a.zeroVal();

  // Same value returned on repeated calls
  EXPECT_EQ(zero1, zero2);
}

TEST_F(Phase2ContainerTest, AllSpecialValuesPerFusion) {
  // Test all special value accessors
  Fusion a;
  FusionGuard fg_a(&a);

  Val* zero = a.zeroVal();
  Val* one = a.oneVal();
  Val* true_val = a.trueVal();
  Val* false_val = a.falseVal();
  NamedScalar* magic_zero = a.magicZeroVal();

  // All should be non-null
  EXPECT_NE(zero, nullptr);
  EXPECT_NE(one, nullptr);
  EXPECT_NE(true_val, nullptr);
  EXPECT_NE(false_val, nullptr);
  EXPECT_NE(magic_zero, nullptr);

  // All should have container() == &a
  EXPECT_EQ(zero->container(), &a);
  EXPECT_EQ(one->container(), &a);
  EXPECT_EQ(true_val->container(), &a);
  EXPECT_EQ(false_val->container(), &a);
  EXPECT_EQ(magic_zero->container(), &a);

  // All should be tracked in ownedVals
  EXPECT_TRUE(a.ownedVals().count(zero) > 0);
  EXPECT_TRUE(a.ownedVals().count(one) > 0);
  EXPECT_TRUE(a.ownedVals().count(true_val) > 0);
  EXPECT_TRUE(a.ownedVals().count(false_val) > 0);
  EXPECT_TRUE(a.ownedVals().count(magic_zero) > 0);
}

TEST_F(Phase2ContainerTest, SpecialValuesClearedOnFusionClear) {
  // Test that Fusion::clear() resets special values
  Fusion a;
  FusionGuard fg_a(&a);

  // Create special values
  Val* zero_before = a.zeroVal();
  Val* one_before = a.oneVal();
  EXPECT_NE(zero_before, nullptr);
  EXPECT_NE(one_before, nullptr);

  // Clear the fusion
  a.clear();

  // Special values should be recreated lazily (new objects)
  Val* zero_after = a.zeroVal();
  Val* one_after = a.oneVal();

  // The new objects should be different from the old ones
  // (old ones were removed by removeStatementsOwnedBy)
  EXPECT_NE(zero_after, zero_before);
  EXPECT_NE(one_after, one_before);

  // New objects should be valid and owned by the fusion
  EXPECT_EQ(zero_after->container(), &a);
  EXPECT_EQ(one_after->container(), &a);
}

TEST_F(Phase2ContainerTest, SpecialValuesWithDtype) {
  // Test zeroVal(dtype) and oneVal(dtype) accessors
  Fusion a;
  FusionGuard fg_a(&a);

  // Index type should return the cached value
  Val* zero_index = a.zeroVal(DataType::Index);
  Val* zero_cached = a.zeroVal();
  EXPECT_EQ(zero_index, zero_cached);

  Val* one_index = a.oneVal(DataType::Index);
  Val* one_cached = a.oneVal();
  EXPECT_EQ(one_index, one_cached);

  // Bool type should return true/false val
  Val* zero_bool = a.zeroVal(DataType::Bool);
  Val* false_cached = a.falseVal();
  EXPECT_EQ(zero_bool, false_cached);

  Val* one_bool = a.oneVal(DataType::Bool);
  Val* true_cached = a.trueVal();
  EXPECT_EQ(one_bool, true_cached);

  // Other types should create new values (not cached)
  Val* zero_float = a.zeroVal(DataType::Float);
  Val* zero_float2 = a.zeroVal(DataType::Float);
  // These are not cached, so they're different objects
  EXPECT_NE(zero_float, zero_float2);
}

// =============================================================================
// Task 5 Tests: Copy Semantics with Shared Containers
// =============================================================================

TEST_F(Phase2ContainerTest, CopySharesContainer) {
  // After copy, both Fusions point to the same container
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy

  // Both should share the same container
  EXPECT_EQ(a.ir_container_ptr().get(), b.ir_container_ptr().get());
}

TEST_F(Phase2ContainerTest, CopyRegistersWithContainer) {
  // sharingCount should increment after copy
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  EXPECT_EQ(a.ir_container()->sharingCount(), 1);

  Fusion b(a);

  EXPECT_EQ(a.ir_container()->sharingCount(), 2);
  EXPECT_EQ(b.ir_container()->sharingCount(), 2);
}

TEST_F(Phase2ContainerTest, CopiedNodesOwnedByNewFusion) {
  // Cloned nodes should have container() == &copy
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a);

  // b should have inputs
  EXPECT_EQ(b.inputs().size(), 1);

  // b's input should be owned by b (not a)
  EXPECT_EQ(b.inputs()[0]->container(), &b);

  // b's input should be different from a's input (cloned)
  EXPECT_NE(b.inputs()[0], a.inputs()[0]);
}

TEST_F(Phase2ContainerTest, CopyOwnedValsAreIndependent) {
  // a's ownedVals and b's ownedVals should be disjoint
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a);

  // All of a's ownedVals should have container() == &a
  for (auto* v : a.ownedVals()) {
    EXPECT_EQ(v->container(), &a);
  }

  // All of b's ownedVals should have container() == &b
  for (auto* v : b.ownedVals()) {
    EXPECT_EQ(v->container(), &b);
  }

  // The sets should be disjoint
  for (auto* v : a.ownedVals()) {
    EXPECT_EQ(b.ownedVals().count(v), 0);
  }
}

TEST_F(Phase2ContainerTest, DestructorOnlyRemovesOwnedStatements) {
  // Destroying copy should not affect original's statements
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  size_t a_vals_before = a.ownedVals().size();

  {
    Fusion b(a); // Copy
    // b gets its own cloned nodes
    EXPECT_GT(b.ownedVals().size(), 0);
    // b destroyed here
  }

  // a's vals should still exist and be unchanged
  EXPECT_EQ(a.ownedVals().size(), a_vals_before);

  // a's vals should still have correct container
  for (auto* v : a.ownedVals()) {
    EXPECT_EQ(v->container(), &a);
  }
}

TEST_F(Phase2ContainerTest, CopyHasOwnSpecialValues) {
  // Each Fusion (original and copy) should have its own special values
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();
  Val* one_a = a.oneVal();

  Fusion b(a); // Copy

  // Copy should have its own special values
  Val* zero_b = b.zeroVal();
  Val* one_b = b.oneVal();

  // Different objects
  EXPECT_NE(zero_a, zero_b);
  EXPECT_NE(one_a, one_b);

  // Correct ownership
  EXPECT_EQ(zero_a->container(), &a);
  EXPECT_EQ(zero_b->container(), &b);
}

TEST_F(Phase2ContainerTest, CopySpecialValuesIndependent) {
  // Destroying copy should not affect original's special values
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();

  {
    Fusion b(a); // Copy
    Val* zero_b = b.zeroVal();
    EXPECT_NE(zero_a, zero_b);
    // b destroyed here
  }

  // a's special values should still be valid
  EXPECT_EQ(a.zeroVal(), zero_a);
  EXPECT_EQ(zero_a->container(), &a);
}

TEST_F(Phase2ContainerTest, CopySharingCountDecrementsOnDestruction) {
  // When copy is destroyed, sharingCount should decrement
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  auto container_ptr = a.ir_container_ptr();
  EXPECT_EQ(container_ptr->sharingCount(), 1);

  {
    Fusion b(a);
    EXPECT_EQ(container_ptr->sharingCount(), 2);
    // b destroyed here
  }

  EXPECT_EQ(container_ptr->sharingCount(), 1);
}

TEST_F(Phase2ContainerTest, CopyReturnsIrCloner) {
  // Fusion::copy should return IrCloner for node mapping
  // We test this indirectly via the copy constructor which uses Fusion::copy
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  // Copy constructor uses Fusion::copy internally
  Fusion b(a);

  // Verify the copy worked - b has cloned inputs/outputs
  EXPECT_EQ(b.inputs().size(), a.inputs().size());
  EXPECT_EQ(b.outputs().size(), a.outputs().size());

  // Cloned nodes should belong to b
  EXPECT_EQ(b.inputs()[0]->container(), &b);
  EXPECT_EQ(b.outputs()[0]->container(), &b);

  // They should be different objects from a's nodes
  EXPECT_NE(b.inputs()[0], a.inputs()[0]);
  EXPECT_NE(b.outputs()[0], a.outputs()[0]);
}

// =============================================================================
// Task 6 Tests: Move Semantics with Shared Containers
// =============================================================================

TEST_F(Phase2ContainerTest, MoveConstructorTransfersOwnership) {
  // Move constructor should transfer container ownership
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  auto* container = a.ir_container_ptr().get();
  size_t a_vals_count = a.ownedVals().size();

  Fusion b(std::move(a));

  // b should have a's old container
  EXPECT_EQ(b.ir_container_ptr().get(), container);

  // b should have a's statements
  EXPECT_EQ(b.ownedVals().size(), a_vals_count);
}

TEST_F(Phase2ContainerTest, MoveConstructorSourceIsValid) {
  // After move, source should be valid with new empty container
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  Fusion b(std::move(a));

  // Source has new empty container (not nullptr)
  EXPECT_NE(a.ir_container_ptr().get(), nullptr);
  EXPECT_NE(a.ir_container_ptr().get(), b.ir_container_ptr().get());

  // Source is empty
  EXPECT_EQ(a.ownedVals().size(), 0);
  EXPECT_EQ(a.inputs().size(), 0);
  EXPECT_EQ(a.outputs().size(), 0);

  // Source can still be used safely
  a.clear(); // Should not crash
}

TEST_F(Phase2ContainerTest, MoveUpdatesStatementOwnership) {
  // Moved statements should have container() pointing to destination
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  // Capture original vals
  std::vector<Val*> orig_vals(a.ownedVals().begin(), a.ownedVals().end());
  EXPECT_GT(orig_vals.size(), 0);

  Fusion b(std::move(a));

  // All original vals now belong to b
  for (auto* val : orig_vals) {
    EXPECT_EQ(val->container(), &b);
  }

  // b's ownedVals should contain them
  for (auto* val : orig_vals) {
    EXPECT_TRUE(b.ownedVals().count(val) > 0);
  }
}

TEST_F(Phase2ContainerTest, MoveTransfersSpecialValues) {
  // Move should transfer special value pointers to destination
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();
  Val* one_a = a.oneVal();

  Fusion b(std::move(a));

  // b should have a's special values
  EXPECT_EQ(b.zeroVal(), zero_a);
  EXPECT_EQ(b.oneVal(), one_a);

  // Ownership updated to b
  EXPECT_EQ(zero_a->container(), &b);
  EXPECT_EQ(one_a->container(), &b);
}

TEST_F(Phase2ContainerTest, MoveSourceCanCreateNewSpecialValues) {
  // After move, source can create new special values
  Fusion a;
  FusionGuard fg_a(&a);
  Val* zero_a = a.zeroVal();

  Fusion b(std::move(a));

  // a is now empty but valid - can create new special values
  Val* zero_a_new = a.zeroVal();

  // Different from the moved one
  EXPECT_NE(zero_a_new, zero_a);
  EXPECT_EQ(zero_a_new->container(), &a);
}

TEST_F(Phase2ContainerTest, MoveAssignmentWorks) {
  // Move assignment should transfer ownership
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  auto* container = a.ir_container_ptr().get();

  Fusion b;
  b = std::move(a);

  // b has a's container
  EXPECT_EQ(b.ir_container_ptr().get(), container);

  // a is valid but empty
  EXPECT_NE(a.ir_container_ptr().get(), nullptr);
  EXPECT_EQ(a.ownedVals().size(), 0);
}

TEST_F(Phase2ContainerTest, MoveAssignmentSelfAssignment) {
  // Self-assignment should be a no-op
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  auto* container = a.ir_container_ptr().get();
  size_t vals_count = a.ownedVals().size();

  // Use a reference to avoid -Wself-move warning
  Fusion& a_ref = a;
  a = std::move(a_ref);

  // Should be unchanged
  EXPECT_EQ(a.ir_container_ptr().get(), container);
  EXPECT_EQ(a.ownedVals().size(), vals_count);
}

TEST_F(Phase2ContainerTest, SwapExchangesContainers) {
  // Swap should exchange container pointers
  Fusion a, b;

  auto* container_a = a.ir_container_ptr().get();
  auto* container_b = b.ir_container_ptr().get();

  Fusion::swap(a, b);

  EXPECT_EQ(a.ir_container_ptr().get(), container_b);
  EXPECT_EQ(b.ir_container_ptr().get(), container_a);
}

TEST_F(Phase2ContainerTest, SwapUpdatesStatementOwnership) {
  // Swap should exchange statement ownership
  Fusion a, b;

  {
    FusionGuard fg_a(&a);
    auto* tv0 = makeSymbolicTensor(2);
    a.addInput(tv0);
  }

  {
    FusionGuard fg_b(&b);
    auto* tv0 = makeSymbolicTensor(3);
    b.addInput(tv0);
  }

  // Capture original vals
  std::vector<Val*> a_vals(a.ownedVals().begin(), a.ownedVals().end());
  std::vector<Val*> b_vals(b.ownedVals().begin(), b.ownedVals().end());

  Fusion::swap(a, b);

  // a's old vals now belong to b
  for (auto* val : a_vals) {
    EXPECT_EQ(val->container(), &b);
  }

  // b's old vals now belong to a
  for (auto* val : b_vals) {
    EXPECT_EQ(val->container(), &a);
  }
}

TEST_F(Phase2ContainerTest, SwapExchangesSpecialValues) {
  // Swap should exchange special values
  Fusion a, b;

  Val* zero_a = nullptr;
  Val* zero_b = nullptr;

  {
    FusionGuard fg_a(&a);
    zero_a = a.zeroVal();
  }

  {
    FusionGuard fg_b(&b);
    zero_b = b.zeroVal();
  }

  Fusion::swap(a, b);

  // Special values exchanged
  EXPECT_EQ(a.zeroVal(), zero_b);
  EXPECT_EQ(b.zeroVal(), zero_a);

  // Ownership updated
  EXPECT_EQ(zero_a->container(), &b);
  EXPECT_EQ(zero_b->container(), &a);
}

TEST_F(Phase2ContainerTest, SwapSelfSwapIsNoop) {
  // Swapping with self should be a no-op
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  auto* container = a.ir_container_ptr().get();
  size_t vals_count = a.ownedVals().size();

  Fusion::swap(a, a);

  EXPECT_EQ(a.ir_container_ptr().get(), container);
  EXPECT_EQ(a.ownedVals().size(), vals_count);
}

TEST_F(Phase2ContainerTest, MoveFromCopyPreservesOther) {
  // If we copy A to B (sharing container), then move A to C,
  // B should be unaffected
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  // Capture b's state
  size_t b_vals_before = b.ownedVals().size();
  std::vector<Val*> b_vals(b.ownedVals().begin(), b.ownedVals().end());

  Fusion c(std::move(a)); // Move a to c

  // b should be completely unaffected
  EXPECT_EQ(b.ownedVals().size(), b_vals_before);
  for (auto* val : b_vals) {
    EXPECT_EQ(val->container(), &b);
    EXPECT_TRUE(b.ownedVals().count(val) > 0);
  }
}

TEST_F(Phase2ContainerTest, MoveFromCopyTransfersCorrectly) {
  // If we copy A to B, then move A to C,
  // C should have A's original statements
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);

  // Capture a's vals before copy
  std::vector<Val*> a_vals(a.ownedVals().begin(), a.ownedVals().end());

  Fusion b(a); // Copy
  Fusion c(std::move(a)); // Move a to c

  // c should have a's original vals
  for (auto* val : a_vals) {
    EXPECT_EQ(val->container(), &c);
    EXPECT_TRUE(c.ownedVals().count(val) > 0);
  }
}

TEST_F(Phase2ContainerTest, MovePreservesInputsOutputs) {
  // Move should transfer inputs/outputs vectors
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Val* orig_input = a.inputs()[0];
  Val* orig_output = a.outputs()[0];

  Fusion b(std::move(a));

  // b has the inputs/outputs
  EXPECT_EQ(b.inputs().size(), 1);
  EXPECT_EQ(b.outputs().size(), 1);
  EXPECT_EQ(b.inputs()[0], orig_input);
  EXPECT_EQ(b.outputs()[0], orig_output);

  // a is empty
  EXPECT_EQ(a.inputs().size(), 0);
  EXPECT_EQ(a.outputs().size(), 0);
}

// =============================================================================
// Deterministic Accessor Tests: Per-Fusion Filtering
// =============================================================================

TEST_F(Phase2ContainerTest, DeterministicValsReturnsOnlyOwned) {
  // With a single Fusion, deterministic_vals() should return all vals
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  // deterministic_vals() should return same count as ownedVals()
  auto det_vals = a.deterministic_vals();
  EXPECT_EQ(det_vals.size(), a.ownedVals().size());

  // All vals in deterministic_vals should be owned by a
  for (auto* val : det_vals) {
    EXPECT_EQ(val->container(), &a);
    EXPECT_TRUE(a.ownedVals().count(val) > 0);
  }
}

TEST_F(Phase2ContainerTest, DeterministicExprsReturnsOnlyOwned) {
  // With a single Fusion, deterministic_exprs() should return all exprs
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  // deterministic_exprs() should return same count as ownedExprs()
  auto det_exprs = a.deterministic_exprs();
  EXPECT_EQ(det_exprs.size(), a.ownedExprs().size());

  // All exprs in deterministic_exprs should be owned by a
  for (auto* expr : det_exprs) {
    EXPECT_EQ(expr->container(), &a);
    EXPECT_TRUE(a.ownedExprs().count(expr) > 0);
  }
}

TEST_F(
    Phase2ContainerTest,
    DeterministicValsFiltersByOwnershipInSharedContainer) {
  // After copy, each Fusion's deterministic_vals() returns only ITS vals
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  // Both share the same container
  EXPECT_EQ(a.ir_container_ptr().get(), b.ir_container_ptr().get());

  // But each has its own deterministic vals
  auto a_det_vals = a.deterministic_vals();
  auto b_det_vals = b.deterministic_vals();

  // Sizes should match ownedVals
  EXPECT_EQ(a_det_vals.size(), a.ownedVals().size());
  EXPECT_EQ(b_det_vals.size(), b.ownedVals().size());

  // a's deterministic_vals should all be owned by a
  for (auto* val : a_det_vals) {
    EXPECT_EQ(val->container(), &a);
  }

  // b's deterministic_vals should all be owned by b
  for (auto* val : b_det_vals) {
    EXPECT_EQ(val->container(), &b);
  }

  // The sets should be disjoint
  std::unordered_set<Val*> a_set(a_det_vals.begin(), a_det_vals.end());
  for (auto* val : b_det_vals) {
    EXPECT_EQ(a_set.count(val), 0);
  }
}

TEST_F(
    Phase2ContainerTest,
    DeterministicExprsFiltersByOwnershipInSharedContainer) {
  // After copy, each Fusion's deterministic_exprs() returns only ITS exprs
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  // Both share the same container
  EXPECT_EQ(a.ir_container_ptr().get(), b.ir_container_ptr().get());

  // But each has its own deterministic exprs
  auto a_det_exprs = a.deterministic_exprs();
  auto b_det_exprs = b.deterministic_exprs();

  // Sizes should match ownedExprs
  EXPECT_EQ(a_det_exprs.size(), a.ownedExprs().size());
  EXPECT_EQ(b_det_exprs.size(), b.ownedExprs().size());

  // a's deterministic_exprs should all be owned by a
  for (auto* expr : a_det_exprs) {
    EXPECT_EQ(expr->container(), &a);
  }

  // b's deterministic_exprs should all be owned by b
  for (auto* expr : b_det_exprs) {
    EXPECT_EQ(expr->container(), &b);
  }

  // The sets should be disjoint
  std::unordered_set<Expr*> a_set(a_det_exprs.begin(), a_det_exprs.end());
  for (auto* expr : b_det_exprs) {
    EXPECT_EQ(a_set.count(expr), 0);
  }
}

TEST_F(Phase2ContainerTest, DeterministicValsMapFiltersByOwnership) {
  // deterministic_vals_map should only include owned vals with local indices
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  auto a_map = a.deterministic_vals_map();
  auto b_map = b.deterministic_vals_map();

  // Maps should have same size as ownedVals
  EXPECT_EQ(a_map.size(), a.ownedVals().size());
  EXPECT_EQ(b_map.size(), b.ownedVals().size());

  // All keys in a_map should be owned by a
  for (const auto& [val, idx] : a_map) {
    EXPECT_EQ(val->container(), &a);
  }

  // All keys in b_map should be owned by b
  for (const auto& [val, idx] : b_map) {
    EXPECT_EQ(val->container(), &b);
  }

  // Indices should be sequential starting from 0 (local to each Fusion)
  std::vector<int64_t> a_indices, b_indices;
  for (const auto& [val, idx] : a_map) {
    a_indices.push_back(idx);
  }
  for (const auto& [val, idx] : b_map) {
    b_indices.push_back(idx);
  }

  std::sort(a_indices.begin(), a_indices.end());
  std::sort(b_indices.begin(), b_indices.end());

  // Should be 0, 1, 2, ... for each
  for (size_t i = 0; i < a_indices.size(); ++i) {
    EXPECT_EQ(a_indices[i], static_cast<int64_t>(i));
  }
  for (size_t i = 0; i < b_indices.size(); ++i) {
    EXPECT_EQ(b_indices[i], static_cast<int64_t>(i));
  }
}

TEST_F(Phase2ContainerTest, DeterministicExprsMapFiltersByOwnership) {
  // deterministic_exprs_map should only include owned exprs with local indices
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  auto a_map = a.deterministic_exprs_map();
  auto b_map = b.deterministic_exprs_map();

  // Maps should have same size as ownedExprs
  EXPECT_EQ(a_map.size(), a.ownedExprs().size());
  EXPECT_EQ(b_map.size(), b.ownedExprs().size());

  // All keys in a_map should be owned by a
  for (const auto& [expr, idx] : a_map) {
    EXPECT_EQ(expr->container(), &a);
  }

  // All keys in b_map should be owned by b
  for (const auto& [expr, idx] : b_map) {
    EXPECT_EQ(expr->container(), &b);
  }

  // Indices should be sequential starting from 0
  std::vector<int64_t> a_indices, b_indices;
  for (const auto& [expr, idx] : a_map) {
    a_indices.push_back(idx);
  }
  for (const auto& [expr, idx] : b_map) {
    b_indices.push_back(idx);
  }

  std::sort(a_indices.begin(), a_indices.end());
  std::sort(b_indices.begin(), b_indices.end());

  for (size_t i = 0; i < a_indices.size(); ++i) {
    EXPECT_EQ(a_indices[i], static_cast<int64_t>(i));
  }
  for (size_t i = 0; i < b_indices.size(); ++i) {
    EXPECT_EQ(b_indices[i], static_cast<int64_t>(i));
  }
}

TEST_F(Phase2ContainerTest, DeterministicValsMaintainsInsertionOrder) {
  // deterministic_vals should maintain insertion order
  Fusion a;
  FusionGuard fg_a(&a);

  // Create multiple tensors in specific order
  auto* tv0 = makeSymbolicTensor(1);
  a.addInput(tv0);
  auto* tv1 = makeSymbolicTensor(2);
  a.addInput(tv1);
  auto* tv2 = add(tv0, tv0);
  auto* tv3 = add(tv1, tv1);
  a.addOutput(tv2);
  a.addOutput(tv3);

  auto det_vals = a.deterministic_vals();
  auto det_map = a.deterministic_vals_map();

  // Verify deque order matches map indices
  for (size_t i = 0; i < det_vals.size(); ++i) {
    Val* val = det_vals[i];
    EXPECT_EQ(det_map.at(val), static_cast<int64_t>(i));
  }
}

TEST_F(Phase2ContainerTest, DeterministicExprsMaintainsInsertionOrder) {
  // deterministic_exprs should maintain insertion order
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  auto* tv2 = mul(tv1, tv0);
  auto* tv3 = sub(tv2, tv1);
  a.addOutput(tv3);

  auto det_exprs = a.deterministic_exprs();
  auto det_map = a.deterministic_exprs_map();

  // Verify deque order matches map indices
  for (size_t i = 0; i < det_exprs.size(); ++i) {
    Expr* expr = det_exprs[i];
    EXPECT_EQ(det_map.at(expr), static_cast<int64_t>(i));
  }
}

TEST_F(Phase2ContainerTest, DeterministicAccessorsAfterCopyPreservesOrder) {
  // After copy, deterministic order for each Fusion should be correct
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  auto* tv2 = mul(tv1, tv1);
  a.addOutput(tv2);

  // Capture a's deterministic vals
  auto a_det_vals_before = a.deterministic_vals();
  auto a_det_map_before = a.deterministic_vals_map();

  Fusion b(a); // Copy

  // a's deterministic_vals should be unchanged
  auto a_det_vals_after = a.deterministic_vals();
  EXPECT_EQ(a_det_vals_before.size(), a_det_vals_after.size());
  for (size_t i = 0; i < a_det_vals_before.size(); ++i) {
    EXPECT_EQ(a_det_vals_before[i], a_det_vals_after[i]);
  }

  // b's deterministic_vals should have same structure (but different objects)
  auto b_det_vals = b.deterministic_vals();
  EXPECT_EQ(b_det_vals.size(), a_det_vals_before.size());
}

TEST_F(Phase2ContainerTest, DeterministicAccessorsAfterDestroyingCopy) {
  // After destroying a copy, original's deterministic accessors still work
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  auto a_det_vals_before = a.deterministic_vals();
  auto a_det_map_before = a.deterministic_vals_map();

  {
    Fusion b(a); // Copy
    // b destroyed here
  }

  // a's deterministic accessors should still work correctly
  auto a_det_vals_after = a.deterministic_vals();
  auto a_det_map_after = a.deterministic_vals_map();

  EXPECT_EQ(a_det_vals_before.size(), a_det_vals_after.size());
  EXPECT_EQ(a_det_map_before.size(), a_det_map_after.size());

  // Same values, same order
  for (size_t i = 0; i < a_det_vals_before.size(); ++i) {
    EXPECT_EQ(a_det_vals_before[i], a_det_vals_after[i]);
  }
}

TEST_F(Phase2ContainerTest, DeterministicValsEmptyForNewFusion) {
  // New empty Fusion should have empty deterministic vals
  Fusion a;

  auto det_vals = a.deterministic_vals();
  auto det_exprs = a.deterministic_exprs();
  auto det_vals_map = a.deterministic_vals_map();
  auto det_exprs_map = a.deterministic_exprs_map();

  EXPECT_EQ(det_vals.size(), 0);
  EXPECT_EQ(det_exprs.size(), 0);
  EXPECT_EQ(det_vals_map.size(), 0);
  EXPECT_EQ(det_exprs_map.size(), 0);
}

TEST_F(Phase2ContainerTest, DeterministicValsAfterClear) {
  // After clear, deterministic vals should be empty
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  EXPECT_GT(a.deterministic_vals().size(), 0);
  EXPECT_GT(a.deterministic_exprs().size(), 0);

  a.clear();

  EXPECT_EQ(a.deterministic_vals().size(), 0);
  EXPECT_EQ(a.deterministic_exprs().size(), 0);
  EXPECT_EQ(a.deterministic_vals_map().size(), 0);
  EXPECT_EQ(a.deterministic_exprs_map().size(), 0);
}

// =============================================================================
// StatementGuard Tests with Shared Containers
// =============================================================================

TEST_F(Phase2ContainerTest, StatementGuardWithSharedContainer) {
  // Test that StatementGuard works correctly with shared containers
  // Bug: StatementGuard uses per-Fusion counts but removes from container
  // deques
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  Fusion b(a); // Copy - shares container

  // Capture b's state before StatementGuard on a
  size_t b_vals_before = b.ownedVals().size();
  size_t b_exprs_before = b.ownedExprs().size();
  std::vector<Val*> b_vals(b.ownedVals().begin(), b.ownedVals().end());

  {
    FusionGuard fg_inner(&a);
    StatementGuard sg(&a);

    // Create temporary vals in a
    auto* temp = add(tv1, tv1);
    (void)temp;

    // a has more vals now
    EXPECT_GT(a.ownedVals().size(), b_vals_before);
  }
  // StatementGuard destructor should only remove a's new vals, not b's

  // b should be completely unaffected
  EXPECT_EQ(b.ownedVals().size(), b_vals_before);
  EXPECT_EQ(b.ownedExprs().size(), b_exprs_before);

  // b's vals should still have correct container
  for (auto* val : b_vals) {
    EXPECT_EQ(val->container(), &b);
    EXPECT_TRUE(b.ownedVals().count(val) > 0);
  }
}

TEST_F(Phase2ContainerTest, StatementGuardDoesNotAffectOtherFusion) {
  // StatementGuard on one Fusion should not affect another sharing the
  // container
  Fusion a;
  FusionGuard fg_a(&a);

  auto* tv0 = makeSymbolicTensor(2);
  a.addInput(tv0);
  auto* tv1 = add(tv0, tv0);
  a.addOutput(tv1);

  size_t a_vals_before_copy = a.ownedVals().size();

  Fusion b(a); // Copy - shares container

  // Both should have same number of vals (cloned)
  EXPECT_EQ(a_vals_before_copy, b.ownedVals().size());

  size_t b_vals_at_guard_start = 0;
  size_t b_vals_in_guard = 0;

  // Use StatementGuard on b to create and remove temp statements
  {
    FusionGuard fg_b(&b);
    StatementGuard sg(&b);

    // Note: StatementGuard constructor calls axioms() which may create
    // additional vals. The snapshot is taken AFTER axioms initialization.
    b_vals_at_guard_start = b.ownedVals().size();

    // Create temp vals in b
    auto* b_input = b.inputs()[0]->as<TensorView>();
    auto* temp = mul(b_input, b_input);
    (void)temp;

    b_vals_in_guard = b.ownedVals().size();

    // b should have more vals now (from the temp operation)
    EXPECT_GT(b_vals_in_guard, b_vals_at_guard_start);
  }

  size_t a_vals_after = a.ownedVals().size();
  size_t b_vals_after = b.ownedVals().size();

  // After guard, a should be unchanged
  EXPECT_EQ(a_vals_after, a_vals_before_copy);

  // b should be back to its state at guard construction time
  // (which includes axioms but not the temp vals created inside the guard)
  EXPECT_EQ(b_vals_after, b_vals_at_guard_start);
}

} // namespace nvfuser
