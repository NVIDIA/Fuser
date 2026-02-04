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

} // namespace nvfuser
