/*
> [!NOTE]
> This file is both a [cpp](../../tests/cpp/tutorial_tmem.cpp) and a Markdown.
> You may see some strange symbols in the rendered Markdown.
> It is difficult to avoid them. But they should not affect reading.
> All the unit tests displayed here are executable from the `test_tutorial`
> binary

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
constexpr static bool verbose = false; /*
```

# Tensor Memory Support in NVFuser
<!--*/
#include <sstream>
#include <string>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

#include <codegen.h>
#include <ops/alias.h>
#include <scheduler/tools/inlining.h>
#include <scheduler/utils.h>

namespace nvfuser {

using ReviewInliningParallelization = NVFuserTest;
using TMemTutorialC = NVFuserTest;
using TMemTutorialR = BlackwellBase;

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

Let's consider a simple gmem->smem->gmem copy kernel. Let's look at the kernels
with different inlining and parallelization strategies:<!-- */ //-->\
```cpp
TEST_F(ReviewInliningParallelization, GSGCopy) {
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
    // Because the first dimension of T1 is inlined, inside the outer loop, T1
    // is consumed right after it is produced. So the first dimension of T1 is
    // not allocated.
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
    // Although the first dimension of T1 is inlined, because shared memory is
    // shared by threads, the TIDx parallelization will override the inlining,
    // and make the first dimension of T1 allocated. The second dimension of T1
    // is allocated normally because it is on the right of the compute-at
    // position. So T1 is allocated in full size.
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
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
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
right are for the column.

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
in tensor memory. That is, in the language of NVFuser, the logical domain of a
TMem TensorView and the allocation domain of that TensorView can have arbitrary
transformations between them. The important thing is not the transformation
between the logical domain and the allocation domain, but the transformation
between the allocation domain and the loop domain, which specifies how we access
tensor memory in the kernel.

Now let's take a look at a few code examples of invalid tensor memory
allocation. Valid examples will be discussed in the next section. For all
valid and invalid examples, we will be looking at
gmem->register->tmem->register->gmem copy kernels. Note that there is no
data path between gmem and tmem, so we have to use registers as transfer
station.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, TooManyLanes) {
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
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

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
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Not enough tensor memory lanes: tried to allocate 429, but only 128 available.")));
} /*
```

In the above example, the fusion is scheduled as:
```python
[BIDx{2}, TIDx{3}, 5, (CA), BIDy{7}, TIDy{11}, 13, (DimSep), 17]
```
(Note that, when the allocation domain is not set, we assume it is the
`getMaybeAllocationDomain`)

Because 2 and 7 are parallelized on `BID`s, they are not allocated. Because 3
and 11 are parallelized on `TID`s, they are allocated. Because 5 is on the left
of the compute-at position, it is not allocated. Because 13 and 17 is on the right
of the compute-at position, it is allocated. So the total number of lanes required
for this tensor is: `3 * 11 * 13 = 429`, which is larger than the total available
lanes `128`.

Now let's take a look at another example:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, TooManyCols) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 3, 5, 7, 11, 13, 17});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::BIDx);
  tv4->axis(2)->parallelize(ParallelType::TIDy);
  tv4->axis(4)->parallelize(ParallelType::BIDy);
  tv4->axis(5)->parallelize(ParallelType::TIDz);
  scheduler_utils::parallelizeAllLike(tv4);
  inlineAllAt(tv4, 4);

  tv2->setTMemDimSepPos(1);

  // Tries to allocate (32, 1105) for tv2.
  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Not enough tensor memory columns: tried to allocate 1105, but only 512 available.")));
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
the total available columns `512`.

## The loop domain of TMem load and store

In NVFuser, the loop structure, parallelization and compute-at strategy of an
expression is determined by the loop domain of the output of the expression.
Unlike shared memory, which allows threads to access it arbitrarily, tensor
memory must be accessed in a specific way. That is, the transformations between
the allocation domain of the TMem TensorView and the loop domain of the consumer
of the TMem accessing expression must satisfy specific patterns. That is, for
the case of a TMem load `T0_r -> T1_t`, `T1_t`'s loop domain and allocation
domain must satisfy specific patterns. For the case of a TMem store `T0_t ->
T1_r`, the loop domain of `T1_r` and allocation domain of `T0_t` must satisfy
specific patterns.

The TMem<->register transfer are warp-collective operations, and the threads in
a warp must access the tensor memory in a specific way.
The specific patterns of TMem<->register transfer is specified in the
[PTX-documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-layout).
These patterns are:

<details>
<summary>32x32b:</summary>

![32x32b](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-3232b.png)
</details>

<details>
<summary>16x64b:</summary>

![16x64b](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-1664b.png)
</details>

<details>
<summary>16x128b:</summary>

![16x128b](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-16128b.png)
</details>


<details>
<summary>16x256b:</summary>

![16x256b](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-16256b.png)
</details>


<details>
<summary>16x32bx2:</summary>

![16x32bx2](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/tcgen05-mma-fragment-1632b2.png)
</details>

Besides threads in a warp must satisfy specific pattern, another restriction of
the TMem<->register transfer is: not all warps can access all lanes of the tensor
memory. The entire 128 lanes of the tensor memory is divided into 4
subpartitions, each has 32 lanes. The warp `i` can only access the subpartition
`i % 4`.

With the above restrictions in mind, let's take a look at a few examples of how
NOT to schedule TMem load and store:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, NotWarpCollective) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({16, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(1);

  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store: "
          "TMem load/store must be warp-collective, "
          "but the innermost extent is not a multiple of 32.")));
} /*
```

The above example is invalid because there are only 16 threads in the kernel.
Warp collective operations require at least a whole warp to run.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, NotContiguous) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({64, 2, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(2);

  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store: "
          "Warp linearly accessing lanes, but not with stride 1.")));
} /*
```

The above example is invalid because the tensor memory is not accessed in any of
the specified pattern. In the above example, because the lane allocation domain is
`[TIDx{64}, TIDy{2}]`, where `TIDx` is not the innermost, threads in a warp access
lanes of the tensor memory in a stride-2 manner, while all the specified
patterns requires the warp to access a contiguous 32 or 16 lanes of data
.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, OneLane) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 32});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDy);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(1);

  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store:")));
} /*
```

The above example is invalid because the tensor memory is not accessed in any of
the specified pattern. In the above example, each warp access one lane and 32
columns of the tensor memory, while all the specified patterns requires the warp
to access a contiguous 32 or 16 lanes of data.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, WrongSubpartition) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 2, 32, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(3);

  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store: "
          "Warps are not accessing the correct sub-partition.")));
} /*
```

The above example is invalid because the warp accesses the wrong subpartition of
the tensor memory. In the above example, there are two warps, where warp 0
accesses the subpartition 0 and 1, and warp 1 accesses the subpartition 2 and 3.
However, warp 0 can only access subpartition 0, and warp 1 can only access
subpartition 1.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialC, WrongSubpartition2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({32, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(1);

  EXPECT_THAT(
      [&]() { KernelExecutor().compile(&fusion); },
      ::testing::ThrowsMessage<nvfError>(::testing::HasSubstr(
          "Invalid data access pattern in TMem load/store: "
          "Warps are not accessing the correct sub-partition.")));
} /*
```

The above example is also invalid because the warp accesses the wrong subpartition.
In the above example, there are two warps, both accessing subpartition 0, which
is not allowed.

Now, let's take a look at some valid examples:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, WarpXYZ) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 4, 4, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDz);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(3);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({2, 4, 4, 2}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

In the above example, each CTA only has one warp, and this warp is split into
TIDz, TIDy, and TIDx. The above kernel uses a loop of 2, where each iteration
accesses a 32x1 box of the tensor memory. This is a valid 32x32b pattern
.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, WarpGroupXYZ) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({2, 8, 8, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDz);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDx);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(3);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({2, 8, 8, 2}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

In the above example, each CTA has one warp group (a group of 4 consecutive warps).
This entire warp group is accessing a whole column, with each warp accessing its
subpartition of 32 lanes. This is a valid 32x32b pattern.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, WarpGroupXYColZ) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({8, 16, 8});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDy);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv4->axis(2)->parallelize(ParallelType::TIDz);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(2);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({8, 16, 8}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

In the above example, each CTA has 8 warp groups, each warp group accesses a
whole column. Warp group `i` is accessing column `i`.
This is a valid 32x32b pattern.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, WarpGroupXColYZ) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({128, 2, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDz);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(1);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({128, 2, 2}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

In the above example, each CTA has 4 warp groups, each warp group accesses a
whole column. The warp group id and the column each warp group accesses are
shown in the table below:

| Warp Group | 0 | 1 | 2 | 3 |
|------------|---|---|---|---|
| Column     | 0 | 2 | 1 | 3 |

This is a valid 32x32b pattern.<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, X1WarpGroupYColZ) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({1, 128, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->axis(0)->parallelize(ParallelType::TIDx);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->axis(2)->parallelize(ParallelType::TIDz);
  scheduler_utils::parallelizeAllLike(tv4);

  tv2->setTMemDimSepPos(2);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({1, 128, 2}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

In the above example, each CTA has 2 warp groups, each warp group accesses a
whole column. Warp group `i` is accessing column `i`. This is a valid 32x32b
pattern. Note that although the order of `TIDx` and `TIDy` seems wrong, it
does not matter because the size of `TIDx` is just 1.

Now, let's take a look at a few more complicated examples that puts a lot
of what we have learned so far into practice.

First, to show that the logical domain and the allocation domain are independent,
we will XOR swizzle the logical domain in a very complicated fashion. What we
want to show is, the row and column of the tensor memory are not related to the
row and column of the tensor itself, and we have the freedom to choose to place
which items of the tensor to where of the tensor memory. In real applications,
it is unlikely that we will XOR swizzle the logical domain, but here we are just
showing that it is possible:<!-- */ //-->\
```cpp
// Apply a fancy transformation to transform a [4096, 4096] tensor back to its
// original shape.
void fancyTransformations(TensorView* tv) {
  tv->swizzle(SwizzleType::XOR, 0, 1);
  tv->split(1, 64);
  tv->split(0, 64);
  tv->reorder({3, 0, 2, 1});
  tv->swizzle(SwizzleType::XOR, 0, 1);
  tv->swizzle(SwizzleType::XOR, 2, 3);
  tv->swizzle(SwizzleType::XOR, 1, 2);
  tv->reorder({3, 0, 2, 1});
  tv->merge(2);
  tv->merge(0);
} /*
```

Second, let's use the following function to check the allocation size of tensor
memory:<!-- */ //-->\
```cpp
void checkAllocationSize(KernelExecutor& ke, int64_t expected_ncols) {
  ke.registerLoweringHook([expected_ncols](GpuLower* lower) {
    auto check_pass = [expected_ncols](const std::vector<Expr*>& exprs) {
      const auto& regions = GpuLower::current()->tmemInfo().allocation.regions;
      [&] { ASSERT_EQ(regions.size(), 1); }();
      const auto& region = regions[0];
      [&] { EXPECT_EQ(region.num_columns->evaluate(), expected_ncols); }();
      return exprs;
    };
    lower->passes().push_back({"Check result", check_pass});
  });
} /*
```

Here comes the example 1:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, Complicated1) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4096, 4096});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  // apply fancy transformations, shape is still [4096, 4096]
  fancyTransformations(tv4);

  // We want the first 4096 to go into lanes, and the second 4096 to go into
  // columns.

  // We want to split the second 4096 into [2, 4, 8, 64], where these dimensions
  // will eventually have the following properties:
  // -  2: serial,  left of CA (not allocated)
  // -  4:   BIDy,  left of CA (not allocated)
  // -  8:   BIDz, right of CA (not allocated)
  // - 64: serial, right of CA (allocated)
  tv4->split(1, 64);
  tv4->split(1, 8);
  tv4->split(1, 4);

  // We want to split the first 4096 into [2, 16, 8, 2, 8, 1], where these
  // dimensions will eventually have the following properties:
  // -  2:   TIDz,  left of CA (allocated)
  // - 16: serial,  left of CA (not allocated)
  // -  8:   TIDy, right of CA (allocated)
  // -  2:   BIDx, right of CA (not allocated)
  // -  8:   TIDx, right of CA (allocated)
  // -  1: serial, right of CA (trivial allocated)
  tv4->split(0, 1);
  tv4->split(0, 8);
  tv4->split(0, 2);
  tv4->split(0, 8);
  tv4->split(0, 16);

  // Parallelize:
  tv4->axis(0)->parallelize(ParallelType::TIDz);
  tv4->axis(2)->parallelize(ParallelType::TIDy);
  tv4->axis(3)->parallelize(ParallelType::BIDx);
  tv4->axis(4)->parallelize(ParallelType::TIDx);
  tv4->axis(7)->parallelize(ParallelType::BIDy);
  tv4->axis(8)->parallelize(ParallelType::BIDz);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv4);

  // Set the allocation domain of TMem tensor
  tv2->setAllocationDomain(tv2->getLoopDomain(), true);
  tv2->setTMemDimSepPos(6);

  // Reorder and inlining
  tv4->reorder({{6, 2}, {7, 3}});
  TransformPropagatorWithCheck propagator2(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator2);
  inlineAllAt(tv4, 4);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;

  // Check that tv2 is allocated 64 columns.
  checkAllocationSize(ke, 64);

  ke.compile(&fusion);

  at::Tensor t0 = at::rand({4096, 4096}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

Here comes the example 2:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, Complicated2) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({4096, 4096});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  // apply fancy transformations, shape is still [4096, 4096]
  fancyTransformations(tv4);

  // We want the first 4096 to go into lanes, and the second 4096 to go into
  // columns.

  // We want to split the second 4096 into [4, 8, 2, 16, 2, 2], where these
  // dimensions will eventually have the following properties:
  // -  4: serial,  left of CA (not allocated)
  // -  8:   BIDy,  left of CA (not allocated)
  // -  2:   TIDy,  left of CA (allocated)
  // - 16: serial, right of CA (allocated)
  // -  2:   BIDz, right of CA (not allocated)
  // -  2:   TIDz, right of CA (allocated)
  tv4->split(1, 2);
  tv4->split(1, 2);
  tv4->split(1, 16);
  tv4->split(1, 2);
  tv4->split(1, 8);

  // We want to split the first 4096 into [32, 128], where these
  // dimensions will eventually have the following properties:
  // -  32: serial,   left of CA (not allocated)
  // - 128:   TIDx,  right of CA (allocated)
  tv4->split(0, 128);

  // Parallelize:
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv4->axis(3)->parallelize(ParallelType::BIDy);
  tv4->axis(4)->parallelize(ParallelType::TIDy);
  tv4->axis(6)->parallelize(ParallelType::BIDz);
  tv4->axis(7)->parallelize(ParallelType::TIDz);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(tv4);

  // Set the allocation domain of TMem tensor
  tv2->setAllocationDomain(tv2->getLoopDomain(), true);
  tv2->setTMemDimSepPos(2);

  // Reorder and inlining
  tv4->reorder({{1, 4}});
  TransformPropagatorWithCheck propagator2(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator2);
  inlineAllAt(tv4, 4);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;

  // Check that tv2 is allocated 64 columns.
  checkAllocationSize(ke, 64);

  ke.compile(&fusion);

  at::Tensor t0 = at::rand({4096, 4096}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
} /*
```

It also worth mentioning that the storing and loading of tensor memory is not
required to be scheduled in the same way. As long as both matches an allowed
pattern. The following example shows how to use tensor memory to do a transpose
:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, Transpose) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigConcreteTensor({128, 2, 2});
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = transpose(tv2, 1, 2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  for (auto tv : {tv1, tv2, tv3, tv4}) {
    tv->axis(0)->parallelize(ParallelType::TIDx);
    tv->axis(1)->parallelize(ParallelType::TIDy);
    tv->axis(2)->parallelize(ParallelType::TIDz);
  }

  tv2->setTMemDimSepPos(1);

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;
  ke.compile(&fusion);

  at::Tensor t0 = at::rand({128, 2, 2}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0.transpose(1, 2)));
} /*
```

In the above example, we store and load the tensor memory like the table below:

| Warp Group   | 0 | 1 | 2 | 3 |
|--------------|---|---|---|---|
| Store Column | 0 | 2 | 1 | 3 |
| Load Column  | 0 | 1 | 2 | 3 |

## Vectorization of TMem load and store

Tensor memory load and store can be vectorized as a power of 2, from 4 bytes all the
way to 512 bytes:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, Vectorization) {
  const std::vector<int64_t> vec_factors = {1, 2, 4, 8, 16, 32, 64, 128};
  for (int64_t st_vec : vec_factors) {
    for (int64_t ld_vec : vec_factors) {
      Fusion fusion;
      FusionGuard fg(&fusion);

      auto tv0 = makeContigConcreteTensor({128, 256});
      fusion.addInput(tv0);
      auto tv1 = set(tv0);
      auto tv2 = set(tv1);
      auto tv3 = set(tv2);
      auto tv4 = set(tv3);
      fusion.addOutput(tv4);
      tv2->setMemoryType(MemoryType::Tensor);
      tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
      tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

      for (auto tv : {tv1, tv2, tv3, tv4}) {
        tv->axis(0)->parallelize(ParallelType::TIDx);
      }

      for (auto tv : {tv2, tv1}) {
        tv->split(1, st_vec);
      }
      tv2->axis(-1)->parallelize(ParallelType::Vectorize);
      for (auto tv : {tv3, tv4}) {
        tv->split(1, ld_vec);
      }
      tv3->axis(-1)->parallelize(ParallelType::Vectorize);

      tv2->setAllocationDomain(tv2->getLoopDomain(), true);
      tv2->setTMemDimSepPos(1);

      inlineMost();

      if constexpr (verbose) {
        fusion.printKernel();
      }

      KernelExecutor ke;

      // Check the allocation size of tv2. When the load and store vectorization
      // factors are the same, the inlining position is one larger than the
      // case when they are different.
      const int64_t expected_ncols =
          st_vec == ld_vec ? std::max<int64_t>(st_vec, 32) : 256;
      checkAllocationSize(ke, expected_ncols);

      ke.compile(&fusion);

      at::Tensor t0 = at::rand({128, 256}, at::kCUDA);
      auto out = ke.run({t0});
      EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));

      // Check that vectorized PTX instructions are used
      GpuLower gpulw(&fusion);
      auto kernel_str = codegen::generateCudaKernel(gpulw.run());
      std::stringstream expect_st, expect_ld;
      expect_st << "tcgen05.st.sync.aligned.32x32b.x" << st_vec << ".b32";
      expect_ld << "tcgen05.ld.sync.aligned.32x32b.x" << ld_vec << ".b32";
      EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_st.str()));
      EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_ld.str()));
    }
  }
} /*
```

Note that the minimum unit of TMem load/store is 4 bytes, and the vectorization
factor used for TMem load/store must makes sure that the total size of the vector
is a multiple of 4 bytes. For example, char2, half1 are invalid vectorization
factors, but char4, half2 are valid:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, VectorizeMultipleOf4Bytes) {
  auto run = [](DataType dtype, int64_t vec) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    auto tv0 = makeContigConcreteTensor({128, 256}, dtype);
    fusion.addInput(tv0);
    auto tv1 = set(tv0);
    auto tv2 = set(tv1);
    auto tv3 = set(tv2);
    auto tv4 = set(tv3);
    fusion.addOutput(tv4);
    tv2->setMemoryType(MemoryType::Tensor);
    tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
    tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

    for (auto tv : {tv1, tv2, tv3, tv4}) {
      tv->axis(0)->parallelize(ParallelType::TIDx);
    }

    if (vec != 1) {
      for (auto tv : {tv1, tv2, tv3, tv4}) {
        tv->split(1, vec);
      }
      tv2->axis(-1)->parallelize(ParallelType::Vectorize);
      tv3->axis(-1)->parallelize(ParallelType::Vectorize);
    }

    tv2->setAllocationDomain(tv2->getLoopDomain(), true);
    tv2->setTMemDimSepPos(1);

    inlineMost();

    if constexpr (verbose) {
      fusion.printKernel();
    }

    KernelExecutor ke;

    ke.compile(&fusion);

    at::TensorOptions options = at::TensorOptions()
                                    .dtype(data_type_to_aten(dtype))
                                    .device(at::kCUDA, 0);
    at::Tensor t0 = dtype == DataType::Char
        ? at::randint(-128, 128, {128, 256}, options)
        : at::rand({128, 256}, options);
    auto out = ke.run({t0});
    EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));
  };

  EXPECT_THAT(
      [&]() { run(DataType::Char, 2); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Vectorize size is not a multiple of 4 bytes")));
  EXPECT_THAT(
      [&]() { run(DataType::Half, 1); },
      ::testing::ThrowsMessage<nvfuser::nvfError>(::testing::HasSubstr(
          "Vectorize size is not a multiple of 4 bytes")));
  run(DataType::Char, 4);
  run(DataType::Half, 2);
} /*
```

When using tensor memory for non-matmul purposes, the allocation domain of the
tensor memory tensor is usually dictated by the global scheduling of the
problem, and the vectorization factor used for TMem load/store are the product
of the unroll factor and the vectorization factor of the global memory load/store.
The following example demonstrates a performant copy kernel
gmem -> register -> tmem -> register -> gmem with vectorization 4 and unroll
factor 2:<!-- */ //-->\
```cpp
TEST_F(TMemTutorialR, PerformantVectorizedCopy) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeContigTensor(1);
  fusion.addInput(tv0);
  auto tv1 = set(tv0);
  auto tv2 = set(tv1);
  auto tv3 = set(tv2);
  auto tv4 = set(tv3);
  fusion.addOutput(tv4);
  tv2->setMemoryType(MemoryType::Tensor);
  tv2->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::StTMem);
  tv3->definition()->as<LoadStoreOp>()->setOpType(LoadStoreOpType::LdTMem);

  tv4->split(0, 4);
  tv4->axis(1)->parallelize(ParallelType::Vectorize);
  tv4->split(0, 128);
  tv4->axis(1)->parallelize(ParallelType::TIDx);
  tv4->split(0, 2);
  tv4->axis(1)->parallelize(ParallelType::TIDy);
  tv4->split(0, 2);
  tv4->axis(1)->parallelize(ParallelType::Serial);
  tv4->axis(0)->parallelize(ParallelType::BIDx);

  TransformPropagatorWithCheck propagator(tv4);
  MaxLogicalDomainInfoSpanningTree(tv4).traverse(&propagator);
  scheduler_utils::parallelizeAllLike(
      tv4, {ParallelType::TIDx, ParallelType::TIDy, ParallelType::BIDx});
  tv1->axis(-1)->parallelize(ParallelType::Vectorize);

  // [BIDx, Serial, TIDy, TIDx, Vec'] ->
  //   [BIDx, TIDx, |, TIDy, Vec{Serial * Vec'}]
  // Where the Vec' above are the vectorization dims of gmem access, not
  // the vectorization of the tmem access, and Vec is the the vectorization
  // of tmem access.
  for (auto tv : {tv2, tv3}) {
    tv->reorder({{1, 3}, {3, 1}});
    tv->merge(-2);
    tv->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  tv2->setAllocationDomain(tv2->getLoopDomain(), true);
  tv2->setTMemDimSepPos(2);

  inlineMost();

  if constexpr (verbose) {
    fusion.printKernel();
  }

  KernelExecutor ke;

  // Check that tv2 is allocated 32 columns. We actually only need
  // TIDy{2} * Serial{2} * Vec'{4} = 16 columns, but 32 is the minimum
  // unit of allocation.
  checkAllocationSize(ke, 32);

  ke.compile(&fusion);

  at::Tensor t0 = at::rand({256 * 1024 * 1024}, at::kCUDA);
  auto out = ke.run({t0});
  EXPECT_TRUE(at::equal(out[0].as<at::Tensor>(), t0));

  // Check that vectorized PTX instructions are used
  GpuLower gpulw(&fusion);
  auto kernel_str = codegen::generateCudaKernel(gpulw.run());
  std::stringstream expect_st, expect_ld;
  expect_st << "tcgen05.st.sync.aligned.32x32b.x8.b32";
  expect_ld << "tcgen05.ld.sync.aligned.32x32b.x8.b32";
  EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_st.str()));
  EXPECT_THAT(kernel_str, ::testing::HasSubstr(expect_ld.str()));
} /*
```

<!--*/
} // namespace nvfuser
// \-->
