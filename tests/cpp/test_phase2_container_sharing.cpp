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

} // namespace nvfuser
