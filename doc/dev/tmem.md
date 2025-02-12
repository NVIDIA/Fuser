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
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

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

From the above example, we can see that:
1. If the memory type of a tensor is distributed across a parallel type, then
   IterDomains with this parallel type are not allocated, regardless of
   the tensor's compute-at position.
2. If the memory type of a tensor is shared among a parallel type, then
   IterDomains with this parallel type are always allocated, regardless of
   the tensor's compute-at position.
3. Except for the above two rules, the allocation of a tensor is determined by
   its compute-at position. IterDomains on the right of the compute-at position
   are allocated, while IterDomains on the left of the compute-at position are
   not allocated.

Tensor memory is similar to shared memory, that is, it is distributed across
`BIDx`, `BIDy`, `BIDz`, `DIDx`, `DIDy`, and `DIDz`, and shared across `TIDx`,
`TIDy`, and `TIDz`. So the above rules applies to tensor memory the same way
as they do to shared memory.

## Tensor memory

NVIDIA's official document for tensor memory can be found
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory).

Unlike any other memory type that we commonly see in computer architecture, the
addresses of tensor memory do not form a linear space. Instead, the addresses
of tensor memory are two-dimensional, and the two dimensions are called `row`
(or `lane`) and `column`.

![Tensor-Memory-Layout](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tensor-memory-layout.png)

In NVFuser, the allocation domain is the domain that specifies how tensor is
allocated in memory. For all other memory types, because these memory types are
linear spaces, the allocation domain is just a vector of `IterDomain`s. But for
tensor memory, because it is 2D, the allocation domain is a vector of
`IterDomain`s plus a position that splits the vector into two parts: the left
part specifies the lane, and the right part specifies the column. The position
is called *Tensor Memory Dimension Separator Position*, or *TMem DimSep
Position* in short.

In practice, to support some MMA shapes, we might want to allocate a tensor in
tensor memory like the figure below:

![TMEM-Allocation-Example](tmem/alloc-example.svg)

That is, the tensor is not a contiguous rectangle in tensor memory, but instead,
is strided in the row dimension. Because NVFuser already supports strided
allocation of global memory tensors, the concepts in that space easily extend to
tensor memory:

The TMem dimsep position does not only apply to the allocation domain, but also
applies to contiguity and stride. The contiguity and stride on the left of the
TMem dimsep position are for the lane, and the contiguity and stride on the
right of are for the column.

In the above example, the allocation domain, contiguity, and stride of the
tensor could be:

```python
allocation domain: [ BIDx,  4, 16, | , BIDy, 8 ]
       contiguity: [    ?,  F,  T, | ,    ?, T ]
           stride: [    ?, 32,  1, | ,    ?, 1 ]
```

It worth noting that, because the tensor memory is distributed across `BIDx` and
`BIDy`, IterDomains with these parallel types are not allocated. So whether
these IterDomains are on the left or right of the TMem dimsep position does not
matter. The value of their contiguity and stride does not matter either (The `?`
in the above example).

Also, please note that the term "row", "lane" and "column" when referring to the
memory layout of tensor memory are not related to the "row" and "column" of the
tensor itself. We can store an arbitrary part of the tensor in an arbitrary form
in tensor memory. In the language of NVFuser, the logical domain of a TMem
TensorView and the allocation domain of the TensorView can have arbitrary
transformations between them. The important thing is not the transformation
between the logical domain and the allocation domain, but the transformation
between the allocation domain and the loop domain, which specifies how we access
tensor memory in the kernel.

To demonstrate the above point, we always do an XOR swizzle on the logical domain
when scheduling the allocation domain. This is not because in real world we want
this swizzle, but just to show that the logical domain and the allocation domain
can have arbitrary transformations between them:<!-- */ //-->\
```cpp
void xorSwizzleLogicalDomain(TensorView* tv) {
  tv->swizzle(SwizzleType::XOR, 0, 1);
} /*
```

Now let's take a look at a few code examples of invalid tensor memory
allocation. Valid examples requires knowledge of indexing, which will be
discussed in the next section. For all valid and invalid examples, we will
be looking at gmem->register->tmem->register->gmem copy kernels. Note that
there is no data path between gmem and tmem, so we have to use register as
a transfer station.<!-- */ //-->\
```cpp
TEST_F(TMemTutorial, TooManyLanes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 5, 7, 11, 13, 17});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::BIDy);
  tv4->axis(4)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv4);
  inlineAllAt(tv4, 3);

  tv2->setTMemDimSepPos(-2);

  // Tries to allocate (429, 17) for tv2.
  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Not enough tensor memory lanes: tried to allocate 429, but only 128 available.")));
} /*
```

In the above example, the fusion is scheduled as:
```python
[BIDx{2}, TIDx{3}, 5, (CA), BIDy{7}, TIDy{11}, 13, (DimSep), 17]
```

Because 2 and 7 are parallelized on `BID`s, they are not allocated. Because 3
and 11 are parallelized on `TID`s, they are allocated. Because 5 is on the left
of the compute-at position, it is not allocated. Because 13 and 17 is on the right
of the compute-at position, it is allocated. So the total number of lanes required
for this tensor is: `3 * 11 * 13 = 429`, which is larger than the total available
lanes, `128`.

Now let's take a look at another example:<!-- */ //-->\
```cpp
TEST_F(TMemTutorial, TooManyCols) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 3, 5, 7, 11, 13, 17});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);

  tv4->axis(0)->parallelize(ParallelType::BIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::BIDy);
  tv4->axis(4)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv4);
  inlineAllAt(tv4, 3);

  tv2->setTMemDimSepPos(-2);

  // Tries to allocate (429, 17) for tv2.
  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Not enough tensor memory lanes: tried to allocate 429, but only 128 available.")));
} /*
```

In the above example, the fusion is scheduled as:
```python
[TIDx{32}, (DimSep), BIDx{3}, TIDy{5}, 7, (CA), BIDy{11}, TIDz{13}, 17]
```

Because 3 and 11 are parallelized on `BID`s, they are not allocated. Because
32, 5, and 13 are parallelized on `TID`s, they are allocated. Because 7 is on
the left of the compute-at position, it is not allocated. Because 17 is on the
right of the compute-at position, it is allocated. So the total number of
columns required for this tensor is: `5 * 13 * 17 = 1105`, which is larger than
the total available lanes, `512`.

<!-- */
} // namespace nvfuser
// \-->
