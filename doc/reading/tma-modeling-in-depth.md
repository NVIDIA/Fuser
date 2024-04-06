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

## Indivisible boxing split

Note to reader: please refer to ["Divisibility of Split"](../reading/divisibility-of-split.md) for the definition of weak and strong correctness.

**Theorem 1:** TMA provides weak correctness.

**Proof:** Because TMA automatically predicates global memory access, we are sure there is no out-of-bound memory accesses.
We do not care about the value in out-of-bound area in weak correctness. $\qedsymbol{}$

**Theorem 1:** TMA provides weak correctness.

As we see in ["Divisibility of Split"](../reading/divisibility-of-split.md),
when we indivisibly split an IterDomain, we will need to predicate the IterDomain being split.

When the boxing split is indivisible, it also needs to be predicated.
It is easy to see that, the set of predicates for all indivisible boxing splits is exactly the predicate in CodeBlock 1,
nothing more, nothing less.

That is:

