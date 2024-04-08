<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# TMA Modeling In Depth

This document means to provide a deeper insight into how we model TMA, primarily at IterDomain level.
This document is not an introduction to our TMA support, which can be found [here](../dev/tma.md).
This document does not discuss topics like hardware, performance, computer architecture, etc.,
instead, we focus on mathematical correctness of our modeling here.
This document assumes familarity with the [introduction doc](../dev/tma.md) and ["Divisibility of Split"](../reading/divisibility-of-split.md).

We will focus on tiled TMA here, im2col TMA is not supported yet.

## What is TMA?

TMA is a hardware feature that allows the GPU to load a tile from a tensor of up to 5D.
The tile can be dense, or strided, as shown in Figure 1 below (the same figure as in the [introduction doc](../dev/tma.md)):

![Figure 1: TMA dense and strided tile](../dev/tma/dense-and-strided-tile.svg)

Conceptually, we can consider TMA as a function:
> $\mathrm{tma}(\vec{x}, sa; ga, \vec{gs}, \vec{bs}, \vec{gr}, \vec{er}, op)$.

In the parameter list of the above function signature,
we intentionally used `;` instead of `,` to separate $sa$ with $ga$.
we call everything before `;` "inputs", and everything after `;` except $op$ "parameters".

The meanings of inputs, parameters, and $op$ are:

- $op$ defines the direction of transfer, it can be either "load" (global -> shared) or "store" (shared -> global).
- $\vec{x}$ is the N-dimensional coordinate of the starting of the box in tensor.
  In the example in Figure 1, it is $(9, 1)$.
- $sa$ stands for "Shared memory base Address".
  In the example in Figure 1, it is the address of the purple item on the top left cornor of the box in shared memory.
- $ga$ stands for "Global memory base Address".
  In the example in Figure 1, it is the address of the blue item on the top left cornor of the tensor in global memory.
- $\vec{gs}$ stands for "Global Size". It is a vector of the same dimensionality as $\vec{x}$.
  In the example in Figure 1, it is $(12, 16)$.
- $\vec{bs}$ stands for "Box Size". It is a vector of the same dimensionality as $\vec{x}$.
  In the example in Figure 1, it is $(6, 4)$.
- $\vec{gr}$ stands for "Global stRide". It is a vector of the same dimensionality as $\vec{x}$.
  In the example in Figure 1, it is $(1, 14)$.
- $\vec{er}$ stands for "Element stRide". It is a vector of the same dimensionality as $\vec{x}$.
  In the example in Figure 1, it is $(1, 1)$ for the left diagram, and $(1, 3)$ for the right diagram.

We separate inputs, parameters, and $op$ because in the implementation,
inputs are provided as operands for the PTX instruction,
parameters are encoded inside the TensorMap descriptor,
and $op$ defines which PTX instruction to use.
When looking at the kernel level, only $\vec{x}$ and $sa$ can change;
parameters and op are predefined constants.

The thing that this $tma$ function does is demonstrated in the CodeBlock 1 below
(here, assuming the TMA is 2D, we can easily generalize to other dimensionalities):

```python
def tma(op, x, sa, *, ga, gs, bs, gr, er):
    ts = [ceildiv(bs[0], er[0]), ceildiv(bs[1], er[1])] # tile size
    for i0 in range(ts[0]):
        for i1 in range(ts[1]):
            smem_idx = i1 * ts[0] + i0
            gmem_idx0 = x[0] + i0 * er[0]
            gmem_idx1 = x[1] + i1 * er[1]
            gmem_idx = gmem_idx0 * gr[0] + gmem_idx1 * gr[1]
            if op == "load":
                if gmem_idx0 < gs[0] and gmem_idx1 < gs[1]:
                    sa[smem_idx] = ga[gmem_idx]
                else:
                    sa[smem_idx] = 0
            else:
                if gmem_idx0 < gs[0] and gmem_idx1 < gs[1]:
                    ga[gmem_idx] = sa[smem_idx]
```

## Predication and correctness

As we can see from CodeBlock 1, TMA has builtin predicates checking that the indices of all partitioned IterDomains are in bound.
That is, TMA will never do out-of-boundary access on global memory even if the indices of
some IterDomains may be out of boundary.
Therefore, we have:

**Theorem 1:** TMA provides weak correctness.

A common use case for TMA is to load data for tensor core,
which requires zero filling on out-of-boundary items.
So it is very important to know when TMA load provides strong correctness.
Strong correctness requires that, when an IterDomain expression create holes (indivisible split, resize),
the holes are filled with a certain value.

As we see in ["Divisibility of Split"](../reading/divisibility-of-split.md),
when we indivisibly split an IterDomain, we will need to predicate the IterDomain being split.
Also observe that when there are indivisible boxing splits,
TMA's builtin predicates are exactly the predicates needed for these indivisible boxing splits.
We therefore have:

**Theorem 2:** A schedule of TMA load provides strong correctness if in the consumer,
the boxing splits are the only IterDomain expressions between the allocation domain and the TMA domain that can create holes,
and the desired filling value is 0.

The condition in Theorem 2 is too strong and does not include some useful strong correctness cases.
Discovering other strong correctness cases needs the following definition:

**Definition 1 (TMA-protected IterDomain):** An IterDomain is TMA-protected if and only if it satisfies one of the following condition:

1. It is a partitioned IterDomain.
2. It is the outer output of a split of a TMA-protected IterDomain.
3. It is the output of a merge whose outer input is a TMA-protected IterDomain.
4. It is the output of a resize whose input is a TMA-protected IterDomain and `right_expand >= 0`.
5. It is the `X` output of a swizzle whose `X` input a TMA-protected IterDomain.

TMA-protected IterDomain has the following very important property:

**Theorem 4:** "TMA's builtin predicates are satisfied" implies "the indices of all TMA-protected IterDomains are in boundary".

**Proof:** This is a natural conclusion of Theorem 1-4 in ["Divisibility of Split"](../reading/divisibility-of-split.md). $\square$

With the above theorem, we can easily see that:

**Theorem 4:** A schedule of TMA load provides strong correctness if in the consumer,
the inputs of all hole-creating IterDomain expressions between the allocation domain and the TMA domain are TMA-protected,
and the desired filling value is 0.
