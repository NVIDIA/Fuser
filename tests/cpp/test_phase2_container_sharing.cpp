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

} // namespace nvfuser
