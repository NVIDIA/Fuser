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

#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <test/utils.h>
#include <test/validator.h>
#include <utils.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

class SmemReuseTest : public NVFuserTest {};

int64_t alignInt(int64_t unaligned, int64_t alignment = 16L) {
  return (unaligned + (alignment - 1)) & (-alignment);
}

// Test that we re-use different-size smem allocations
//
//             +-----+
//             |  B  |
//   +-----+   +-----+
//   |  A  |
//   +-----+
//   a     b * c     d
//
// where * indicates an expression that synchronizes each thread block
//
// Should become:
//
//   +-----+   +-----+
//   |  A  |   |  B  |
//   +-----+   +-----+
//   a     b * c     d
//
TEST_F(SmemReuseTest, SimpleCase) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H_int = 5, W_int = 6;
  auto H = IrBuilder::create<Val>(H_int);
  auto W = IrBuilder::create<Val>(W_int);

  auto tv0 = full({H}, fusion->oneVal(), DataType::Float);
  auto tv1 = set(tv0); // pos = a. A = tv1
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = set(tv1); // pos = b
  fusion->addOutput(tv2);

  auto tv3 = sum(tv2, {0}); // gap between b and c
  fusion->addOutput(tv3);

  auto tv4 = full({W}, fusion->oneVal(), DataType::Float);

  auto tv5 = mul(tv3, tv4); // pos = c. B = tv5
  tv5->setMemoryType(MemoryType::Shared);

  auto tv6 = set(tv5); // pos = d
  fusion->addOutput(tv6);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      NVF_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    // tv1{H} comes before tv5{W}, and the last uses follow the same order. When
    // we reorder pushed allocations, we sort them by last read in descending
    // order, so tv5 goes on the bottom.
    EXPECT_EQ(smem_usage, alignInt(W_int * 4) + H_int * 4);
  }

  { // Now introduce a block reduction and check that we re-use memory

    tv3->axis(0)->parallelize(ParallelType::TIDx);

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    // (Aligned size of H) plus W
    EXPECT_EQ(smem_usage, W_int * 4);
  }
}

// Test that we re-use different-size smem allocations
//
//             +-----+
//             |  C  |
//       +-----+-+---+
//       |   B   |
//   +---+-+-----+
//   |  A  |
//   +-----+
//   a   b c * d e   f
//
// where * indicates an expression that synchronizes each thread block
//
// Should become:
//
//   +-----+   +-----+
//   |  A  |   |  C  |
//   +---+-----+-+---+
//       |   B   |
//       +-+-----+
//   a   b c * d e   f
//
std::tuple<TensorView*, TensorView*> needsReorderedPushDefinition(int64_t H) {
  auto fusion = FusionGuard::getCurFusion();

  auto tv0 = full(
      {IrBuilder::create<Val>(H)},
      fusion->oneVal(),
      DataType::Float); // pos = a. A = tv0
  tv0->setMemoryType(MemoryType::Shared);

  auto tv1 =
      pad(tv0, {fusion->zeroVal(), fusion->oneVal()}); // pos = b. B = tv1
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = mul(tv1, tv1); // pos = c

  auto tv3 = sum(tv2, {0}); // gap between b and c. Can parallelize to sync

  auto tv4 = broadcast(tv3, {true});
  auto tv5 = mul(tv4, tv1); // pos = d. C = tv5
  tv5->setMemoryType(MemoryType::Shared);

  auto tv6 = add(tv1, tv1); // pos = e
  fusion->addOutput(tv6);

  auto tv7 = neg(tv5); // pos = f
  fusion->addOutput(tv7);

  return {tv0, tv3};
}

TEST_F(SmemReuseTest, NeedsReorderedPush) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H = 5;
  auto [tv0, tv3] = needsReorderedPushDefinition(H);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      NVF_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(
        smem_usage, alignInt(alignInt((H + 1) * 4) + (H + 1) * 4) + H * 4);
  }

  { // Now introduce a block reduction and check that we re-use memory
    tv3->axis(0)->parallelize(ParallelType::TIDx);

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(smem_usage, alignInt((H + 1) * 4) + (H + 1) * 4);
  }
}

// Same as NeedsReorderedPush but C requests to reuse A instead of pre-existing
// sync
TEST_F(SmemReuseTest, PromoteReuse) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H = 5;
  auto [tv0, tv3] = needsReorderedPushDefinition(H);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      NVF_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(
        smem_usage, alignInt(alignInt((H + 1) * 4) + (H + 1) * 4) + H * 4);
  }

  { // Request that we re-use the allocation for tv0. This should place a
    // __syncthreads() just before tv5 is written.
    tv0->promoteReuse();

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(smem_usage, alignInt((H + 1) * 4) + (H + 1) * 4);
  }
}

// In this example, we promote a single tensor for re-use in a Fusion with two
// downstream tensors that could use its memory.
TEST_F(SmemReuseTest, PromoteReuseMultipleDownstream) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H = 7;

  // The outer live intervals of tv0, tv2, and tv4 will be non-overlapping, but
  // adjacent. tv0->promoteReuse() should be able to insert a sync before tv2,
  // so that tv4 re-uses the memory from tv0, then tv2 is stacked above tv4.

  auto tv0 =
      full({IrBuilder::create<Val>(H)}, fusion->oneVal(), DataType::Float);
  tv0->setMemoryType(MemoryType::Shared);

  auto tv1 = neg(tv0);

  auto tv2 = pad(tv1, {fusion->zeroVal(), fusion->oneVal()});
  tv2->setMemoryType(MemoryType::Shared);

  auto tv3 = neg(tv2);

  auto tv4 = pad(tv3, {fusion->zeroVal(), fusion->oneVal()});
  tv4->setMemoryType(MemoryType::Shared);

  auto tv5 = neg(tv4);

  fusion->addOutput(tv5);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      NVF_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(
        smem_usage, alignInt(alignInt((H + 2) * 4) + (H + 1) * 4) + H * 4);
  }

  { // Request that we re-use the allocation for tv0. This should place a
    // __syncthreads() just before tv2 is written.
    tv0->promoteReuse();

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(smem_usage, alignInt((H + 2) * 4) + (H + 1) * 4);
  }
}

// In this example, multiple smem tensors are promoted for re-use. We have
// non-overlapping smem allocations A B C D, and A and C are promoted for reuse.
// Because of that, B re-uses A, then C does not reuse B but stacks on top of
// it. Then D reuses C, and B is reclaimed in the process. Ultimately this means
// the assigned addresses are:
//
//   A: 0. Assigned then reclaimed before assignment of B.
//   B: alignInt((H + 2) * 4). Stacked on top of C
//   C: 0. Assigned along with B in reverse order of last use
//   D: 0. B and C are reclaimed before this assignment.
//
// Note that although B was not explicitly requested for re-use, since its
// lifetime ends before D is defined, we try and reclaim it at the same time C
// is reclaimed. They are also ordered on the stack at that point, in descending
// order of last use, meaning B is placed higher on the stack than C.
TEST_F(SmemReuseTest, MultiplePromoteReuse) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H = 7;

  auto tv0 =
      full({IrBuilder::create<Val>(H)}, fusion->oneVal(), DataType::Float);
  tv0->setMemoryType(MemoryType::Shared);

  auto tv1 = neg(neg(tv0));

  auto tv2 = pad(tv1, {fusion->zeroVal(), fusion->oneVal()});
  tv2->setMemoryType(MemoryType::Shared);

  auto tv3 = neg(neg(tv2));

  auto tv4 = pad(tv3, {fusion->zeroVal(), fusion->oneVal()});
  tv4->setMemoryType(MemoryType::Shared);

  auto tv5 = neg(neg(tv4));

  auto tv6 = pad(tv5, {fusion->zeroVal(), fusion->oneVal()});
  tv6->setMemoryType(MemoryType::Shared);

  auto tv7 = neg(tv6);

  fusion->addOutput(tv7);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      NVF_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(
        smem_usage,
        alignInt(alignInt(alignInt((H + 3) * 4) + (H + 2) * 4) + (H + 1) * 4) +
            H * 4);
  }

  { // Request that we re-use A and C
    tv0->promoteReuse();
    tv4->promoteReuse();

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->address(), nullptr);
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    // High water mark has C stacked on top of B
    EXPECT_EQ(smem_usage, alignInt((H + 2) * 4) + (H + 1) * 4);
  }
}

} // namespace nvfuser
