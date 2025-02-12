/*
> [!NOTE]
> This file is both a [cpp](../../tests/cpp/tutorial_tmem.cpp) and a Markdown.
> You may see some strange symbols in the rendered Markdown.
> It is difficult to avoid them. But they should not affect reading.
> All the unit tests displayed here are executable from the `tutorial` binary

<!--*/
#pragma GCC diagnostic ignored "-Wcomment"
// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
/*-->

To see prints in the test, change below to `true`:<!-- */ //-->\
```cpp
constexpr static bool verbose = true; /*
```

# Tensor Memory Support in NVFuser
<!--*/
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <ops/alias.h>

namespace nvfuser {

using ReviewInliningParallelization = NVFuserTest;
using TMemTutorial = BlackwellBase;

/* -->

Tensor memory is new a memory type added in the Blackwell architecture.
Similar to shared memory, it is a memory in the SM that is accessible by threads
in the CTA. Although there are many differences between tensor memory and shared
memory, the fact that they are shared by threads and distributed across
different CTAs makes the behavior of tensor memory similar to shared memory when
talking about allocation and how it is affected by inlining and parallelization.

Before diving deep into tensor memory, let's first do a quick review of inlining
and parallelization, and how they impact allocation and indexing. This review
will give us a rough idea of how tensor memory should behave.

## Review of inlining and parallelization

Let's consider a simple gmem->shared->gmem copy kernel. Let's look at the kernel
with different inlining and parallelization strategy:<!-- */ //-->\
```cpp
TEST_F(ReviewInliningParallelization, GSGCopy1) {
  at::Tensor t0 = at::rand({2, 4}, at::kCUDA);

  // Naive copy kernel, no inlining, no parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // T1 is allocated in full size
    EXPECT_EQ(ke.lastLaunchParams().smem(), 8 * sizeof(float));
  }

  // No inlining, has BIDx parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->axis(1)->parallelize(ParallelType::BIDx);
    tv2->axis(1)->parallelize(ParallelType::BIDx);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // Because smem is distributed across different CTAs, only the first
    // dimension of T1 is allocated.
    EXPECT_EQ(ke.lastLaunchParams().smem(), 2 * sizeof(float));
  }

  // Inline at 1, no parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->inlineAt(1);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // Because the first dimension of T1 is inlined, inside the outer loop, T1
    // is consumed right after it is produced. So T1 the first dimension of T1
    // is not allocated.
    EXPECT_EQ(ke.lastLaunchParams().smem(), 4 * sizeof(float));
  }

  // Inline at 1, with BIDx parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->inlineAt(1);
    tv1->axis(1)->parallelize(ParallelType::BIDx);
    tv2->axis(1)->parallelize(ParallelType::BIDx);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // Due to inlining, the first dimension of T1 is not allocated. Due to
    // BIDx parallelization, the second dimension of T1 is not allocated. So T1
    // is only allocated in size 1.
    EXPECT_EQ(ke.lastLaunchParams().smem(), 1 * sizeof(float));
  }

  // Inline at 1, with TIDx parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->inlineAt(1);
    tv1->axis(0)->parallelize(ParallelType::TIDx);
    tv2->axis(0)->parallelize(ParallelType::TIDx);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // Although the first dimension of T1 is inlined, because shared memory is
    // shared by threads, the TIDx parallelization will override the inlining,
    // and make the first dimension of T1 allocated. The second dimension of T1
    // is allocated normally. So T1 is allocated in full size.
    EXPECT_EQ(ke.lastLaunchParams().smem(), 8 * sizeof(float));
  }

  // Inline at 1, with TIDx and BIDx parallelization
  {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({2, 4});
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    fusion.addOutput(tv2);

    tv1->setMemoryType(MemoryType::Shared);
    tv1->inlineAt(1);
    tv1->axis(0)->parallelize(ParallelType::TIDx);
    tv2->axis(0)->parallelize(ParallelType::TIDx);
    tv1->axis(1)->parallelize(ParallelType::BIDx);
    tv2->axis(1)->parallelize(ParallelType::BIDx);

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;
    ke.compile(&fusion);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0], t0));
    // Although the first dimension of T1 is inlined, because shared memory is
    // shared by threads, the TIDx parallelization will override the inlining,
    // and make the first dimension of T1 allocated. The second dimension of T1
    // is not allocated due to BIDx parallelization. So T1 is allocated in
    // size 2.
    EXPECT_EQ(ke.lastLaunchParams().smem(), 2 * sizeof(float));
  }
} /*
```

In this kernel, because the computation of `T1` is not inlined, `T1` is
allocated in the top scope of the kernel in full size.

<!-- */
} // namespace nvfuser
// \-->
