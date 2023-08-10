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
//
//            +-----+
//            |  B  |
//   +-----+  +-----+
//   |  A  |
//   +-----+
//   a     b  c     d
//
// Should become:
//
//   +-----+  +-----+
//   |  A  |  |  B  |
//   +-----+  +-----+
//   a     b  c     d
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

  auto tv2 = add(tv1, tv1); // pos = b
  fusion->addOutput(tv2);

  auto tv3 = neg(tv2); // gap between b and c
  fusion->addOutput(tv3);

  auto tv4 = full({W}, fusion->oneVal(), DataType::Float); // pos = c. B = tv4
  tv4->setMemoryType(MemoryType::Shared);

  auto tv5 = mul(tv4, tv4); // pos = d
  fusion->addOutput(tv5);

  { // Run without reusing
    EnableOptionsGuard eog;
    eog.getCurOptions().unset(EnableOption::ReuseSharedMemory);

    FusionExecutor fe;
    fe.compileFusion(fusion.get(), {});

    fe.runFusion({});

    const auto& lparams = fe.lastLaunchParams();

    // (Aligned size of W) plus H
    EXPECT_EQ(lparams.smem(), alignInt(W_int * 4) + H_int * 4);
  }

  { // Now reuse
    EnableOptionsGuard eog;
    eog.getCurOptions().set(EnableOption::ReuseSharedMemory);

    FusionExecutor fe;
    fe.compileFusion(fusion.get(), {});

    fe.runFusion({});

    const auto& lparams = fe.lastLaunchParams();

    EXPECT_EQ(lparams.smem(), W_int * 4);
  }
}

// Test that we re-use different-size smem allocations
//
//           +-----+
//           |  C  |
//       +---+-+---+
//       |  B  |
//   +---+-+---+
//   |  A  |
//   +-----+
//   a   b c d e   f
//
// Should become:
//
//   +-----+ +-----+
//   |  A  | |  C  |
//   +---+---+-+---+
//       |  B  |
//       +-+---+
//   a   b c d e   f
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

  auto tv2 = add(tv1, tv1); // pos = b. B = tv2
  tv2->setMemoryType(MemoryType::Shared);

  auto tv3 = add(tv1, tv1); // pos = c
  fusion->addOutput(tv3);

  auto tv4 = full({W}, fusion->oneVal(), DataType::Float);

  auto tv5 = broadcast(tv2, {false, true});
  auto tv6 = broadcast(tv4, {true, false});
  auto tv7 = mul(tv5, tv6); // pos = d. C = tv7
  tv7->setMemoryType(MemoryType::Shared);

  auto tv8 = add(tv2, tv2); // pos = e
  fusion->addOutput(tv8);

  auto tv9 = neg(tv7); // pos = f
  fusion->addOutput(tv9);

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), {});

  auto args = KernelArgumentHolder::createKernelArgumentHolder({});

  auto outputs = fe.runFusion(args);

  EXPECT_EQ(outputs.size(), 3);

  { // Run without reusing
    EnableOptionsGuard eog;
    eog.getCurOptions().unset(EnableOption::ReuseSharedMemory);

    FusionExecutor fe;
    fe.compileFusion(fusion.get(), {});

    fe.runFusion({});

    const auto& lparams = fe.lastLaunchParams();

    // (Aligned size of W) plus H
    EXPECT_EQ(
        lparams.smem(),
        alignInt(alignInt(H_int * 4) + W_int * 4) + H_int * W_int * 4);
  }

  { // Now reuse
    EnableOptionsGuard eog;
    eog.getCurOptions().set(EnableOption::ReuseSharedMemory);

    FusionExecutor fe;
    fe.compileFusion(fusion.get(), {});

    fe.runFusion({});

    const auto& lparams = fe.lastLaunchParams();

    EXPECT_EQ(lparams.smem(), alignInt(H_int * 4) + W_int * H_int * 4);
  }
}

} // namespace nvfuser
