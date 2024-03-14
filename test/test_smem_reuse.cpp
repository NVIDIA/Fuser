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
#include <inlining.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
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

// Test that intervals with "skipped" positions in alias analysis are not
// ignored This can happen when we require a sync immediately between two
// https://github.com/NVIDIA/Fuser/pull/1381
TEST_F(SmemReuseTest, SkippedSyncInterval) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  int64_t D = 5;
  int64_t H = 6;
  int64_t W = 7;

  auto tv0 = full(
      {
          IrBuilder::create<Val>(D),
          IrBuilder::create<Val>(H),
          IrBuilder::create<Val>(W),
      },
      fusion->oneVal(),
      DataType::Float);
  tv0->setMemoryType(MemoryType::Shared);

  auto tv1 = neg(tv0); // generate a loop nest, which is the last use of tv0

  // Write to tv2 in adjacent loop nest. Pad so we can't do inner aliasing.
  auto tv2 = pad(tv1, {fusion->zeroVal(), fusion->oneVal()});
  tv2->setMemoryType(MemoryType::Shared);

  auto tv3 = set(tv2); // third loop nest writes to global output

  fusion->addOutput(tv3);

  // By parallelizing the first loop, we remove its appearance in the kernel.
  //
  //   6   FOR threadIdx.x in ithreadIdx.x3{5}:
  //   7     FOR i85 in iS4{6}:
  //   8       FOR i86 in iS5{7}:
  //   9         T0_s[ ithreadIdx.x0{5}, iS1{6}, iS2{7} ] ca_pos( 3 )
  //               = full({5, 6, 7}, ( (float)(1) ));
  //   10        T1_l[ ithreadIdx.x3{5}, iS4{6}, iS5{7} ] produce_pos( 3 )
  //               = -T0_s[ ithreadIdx.x0{5}, iS1{6}, iS2{7} ] ca_pos( 3 );
  //   11      ENDFOR i86
  //   12    ENDFOR i85
  //   13  ENDFOR threadIdx.x
  //   14  FOR threadIdx.x in ithreadIdx.x6{5}:
  //   15    FOR i81 in iS7{6}:
  //   16      FOR i82 in iS9{8}rf:
  //   17        T2_s[ ithreadIdx.x6{5}, iS7{6}, iS9{8}rf ]
  //      = pad( T1_l[ ithreadIdx.x3{5}, iS4{6}, iS5{7} ] produce_pos( 3 ), {0,
  //      0, 0, 0, 0, 1} )
  //   18      ENDFOR i82
  //   19    ENDFOR i81
  //   20  ENDFOR threadIdx.x
  //   21  FOR threadIdx.x in ithreadIdx.x10{5}:
  //   22    FOR i83 in iS11{6}:
  //   23      FOR i84 in iS12{8}:
  //   24        T3_g[ ithreadIdx.x10{5}, iS11{6}, iS12{8} ]
  //      = Set( T2_s[ ithreadIdx.x6{5}, iS7{6}, iS9{8}rf ], cache_op=Streaming)
  //   25      ENDFOR i84
  //   26    ENDFOR i83
  //   27  ENDFOR threadIdx.x
  //
  // Parallelized loops still occupy a "position" in the expression list; i.e.
  // the ENDFOR of the ithreadIdx.x3{5} loop is at position 13.
  // Lifetimes are computed with respect to for loops that appear in the
  // kernel, so we will wind up with a gap between the end of outer lifetime of
  // tv0, which now ends at the end of the H loop, and the beginning of the H
  // loop for tv2, since there are separate parallelized D loops for the two.
  // That is, the outer lifetimes of smem allocations here are
  //
  //   T0_s:  [7, 12]
  //   T2_s:  [15, 26]
  //
  // Promoting reuse means we will insert a sync in (12, 15) which is partially
  // hidden since 12 is an inner ENDFOR position.

  tv1->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv1);

  std::vector<TensorView*> inlinedtvs = {tv0};
  inlineMost(inlinedtvs);

  tv0->promoteReuse();

  GpuLower gpulw(fusion.get());

  ExpressionEvaluator ee;
  for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
    EXPECT_NE(alloc->address(), nullptr);
    auto addr = ee.evaluate(alloc->address()).as<int64_t>();
    // Both smem allocs should be placed at 0, indicating successful reuse
    EXPECT_EQ(addr, 0);
  }
}

TEST_F(SmemReuseTest, SmemReuseWithDifferentVectorizationFactor) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int n_element = 16;
  auto tv0 = makeContigConcreteTensor({n_element});
  fusion->addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion->addOutput(tv4);

  tv1->split(-1, 2);
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);
  tv1->setMemoryType(MemoryType::Shared);

  tv2->split(-1, 4);

  tv3->split(-1, 4);
  tv3->axis(-1)->parallelize(ParallelType::Vectorize);
  tv3->setMemoryType(MemoryType::Shared);

  inlineMost();

  // T1 and T3 are vectorized by 2 and 4, respectively.
  // However, T3 is still able to reuse T1's memory since the shared memory is
  // aligned to 16 bytes which is the largest vectorization width.
  // check T3 alias T1.
  {
    bool t3_alias_t1 = false;
    GpuLower gpulw(fusion.get());
    for (auto alloc : gpulw.run()->summary().dynamic_smem_allocations) {
      EXPECT_NE(alloc->buffer(), nullptr);
      if (alloc->buffer()->name() == 3) {
        EXPECT_NE(alloc->alias(), nullptr);
        EXPECT_EQ(alloc->alias()->buffer()->name(), 1);
        t3_alias_t1 = true;
      }
    }
    EXPECT_TRUE(t3_alias_t1);
  }
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({n_element}, options);
  FusionExecutor fe;
  fe.compileFusion(fusion.get());
  auto cg_outputs = fe.runFusion({t0});
  testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(SmemReuseTest, RegisterReuseWithDifferentVectorizationFactor) {
  auto testRegisterReuse = [](int vect_factor_1, int vect_factor_2) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    constexpr int n_element = 16;
    auto tv0 = makeContigConcreteTensor({n_element});
    fusion->addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    auto tv3 = set(tv2);
    auto tv4 = set(tv3);
    fusion->addOutput(tv4);

    tv1->split(-1, vect_factor_1);
    if (vect_factor_1 > 1) {
      tv1->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    tv2->split(-1, vect_factor_2);

    tv3->split(-1, vect_factor_2);
    if (vect_factor_2 > 1) {
      tv3->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    // T1 and T3 are vectorized by factor_1 and factor_2, respectively.
    // T3 alias T1 when vect_factor_2 <= vect_factor_1.
    {
      bool t3_alias_t1 = false;
      GpuLower gpulw(fusion.get());
      for (auto expr : gpulw.run()->topLevelExprs()) {
        if (expr->isA<kir::Allocate>()) {
          auto alloc = expr->as<kir::Allocate>();
          if (alloc->buffer()->name() == 3) {
            if (alloc->alias() && alloc->alias()->buffer()->name() == 1) {
              t3_alias_t1 = true;
            } else {
              t3_alias_t1 = false;
            }
          }
        }
      }
      if (vect_factor_2 <= vect_factor_1) {
        EXPECT_TRUE(t3_alias_t1);
      } else {
        EXPECT_FALSE(t3_alias_t1);
      }
    }

    // run the fusion
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    auto t0 = at::randn({n_element}, options);
    FusionExecutor fe;
    fe.compileFusion(fusion.get());
    auto cg_outputs = fe.runFusion({t0});
    testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
  };

  // test different vectorization factors, max is 4 since float is used.
  for (int vect_factor_1 : {1, 2, 4}) {
    for (int vect_factor_2 : {1, 2, 4}) {
      testRegisterReuse(vect_factor_1, vect_factor_2);
    }
  }
}

TEST_F(SmemReuseTest, ExpandInterferes) {
  auto testExpand = [](bool is_concrete) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto x = 3L;
    auto y = 4L;
    auto tv0 =
        is_concrete ? makeContigConcreteTensor({y}) : makeSymbolicTensor(1);
    fusion->addInput(tv0);

    auto tv1 = set(tv0);
    auto tv2 = broadcast(tv1, {true, false});
    auto tv3 =
        expand(tv2, {IrBuilder::create<Val>(x), IrBuilder::create<Val>(y)});
    auto tv4 = set(tv3);
    fusion->addOutput(tv4);

    for (auto tv : {tv1, tv2, tv3}) {
      tv->setMemoryType(MemoryType::Shared);
    }

    // tv3 is trying to reuse tv1's memory. however it has a concrete size.
    // The reuse only happens when tv1 is also concrete.
    {
      bool t3_alias_t1 = false;
      GpuLower gpulw(fusion.get());
      for (auto expr : gpulw.run()->topLevelExprs()) {
        if (expr->isA<kir::Allocate>()) {
          auto alloc = expr->as<kir::Allocate>();
          if (alloc->buffer()->name() == 3) {
            if (alloc->alias() && alloc->alias()->buffer()->name() == 1) {
              t3_alias_t1 = true;
            } else {
              t3_alias_t1 = false;
            }
          }
        }
      }
      if (is_concrete) {
        EXPECT_TRUE(t3_alias_t1);
      } else {
        EXPECT_FALSE(t3_alias_t1);
      }
    }

    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({y}, options);
    FusionExecutor fe;
    fe.compileFusion(fusion.get());
    auto cg_outputs = fe.runFusion({t0});
    testValidate(fusion.get(), cg_outputs, {t0}, __LINE__, __FILE__);
  };

  testExpand(true);
  testExpand(false);
}

} // namespace nvfuser
