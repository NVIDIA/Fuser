// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
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
    for (auto alloc : gpulw.kernel()->summary().dynamic_smem_allocations) {
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      TORCH_CHECK(
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
    for (auto alloc : gpulw.kernel()->summary().dynamic_smem_allocations) {
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
TEST_F(SmemReuseTest, NeedsReorderedPush) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t H_int = 5, W_int = 6;
  auto H = IrBuilder::create<Val>(H_int);
  auto W = IrBuilder::create<Val>(W_int);

  auto tv0 = full({H}, fusion->oneVal(), DataType::Float);
  auto tv1 = set(tv0); // pos = a. A = tv1
  tv1->setMemoryType(MemoryType::Shared);

  auto tv2 = full({W}, fusion->oneVal(), DataType::Float); // pos = b. B = tv2
  tv2->setMemoryType(MemoryType::Shared);

  auto tv3 = add(tv1, tv1); // pos = c

  auto tv4 = sum(tv3, {0}); // gap between b and c
  fusion->addOutput(tv4);

  auto tv5 = broadcast(tv4, {true});
  auto tv6 = mul(tv5, tv3);

  auto tv7 = broadcast(tv6, {true, false});
  auto tv8 = broadcast(tv2, {false, true});
  auto tv9 = mul(tv7, tv8); // pos = d. C = tv9
  tv9->setMemoryType(MemoryType::Shared);

  auto tv10 = add(tv2, tv2); // pos = e
  fusion->addOutput(tv10);

  auto tv11 = neg(tv9); // pos = f
  fusion->addOutput(tv11);

  { // This should not re-use memory
    GpuLower gpulw(fusion.get());

    ExpressionEvaluator ee;
    std::unordered_set<int64_t> addresses;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.kernel()->summary().dynamic_smem_allocations) {
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      TORCH_CHECK(
          addresses.insert(addr).second,
          "Smem addresses should not be re-used");
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(
        smem_usage,
        alignInt(alignInt(H_int * W_int * 4) + W_int * 4) + H_int * 4);
  }

  { // Now introduce a block reduction and check that we re-use memory

    tv4->axis(0)->parallelize(ParallelType::TIDx);

    GpuLower gpulw(fusion.get());
    ExpressionEvaluator ee;
    int64_t smem_usage = 0;
    for (auto alloc : gpulw.kernel()->summary().dynamic_smem_allocations) {
      auto addr = ee.evaluate(alloc->address()).as<int64_t>();
      auto size = ee.evaluate(alloc->size()).as<int64_t>() *
          dataTypeSize(alloc->buffer()->dtype());
      smem_usage = std::max(smem_usage, addr + size);
    }
    EXPECT_EQ(smem_usage, alignInt(H_int * 4) + W_int * H_int * 4);
  }
}

} // namespace nvfuser
