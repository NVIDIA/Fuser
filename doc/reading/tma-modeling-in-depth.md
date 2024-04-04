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

We will focus on tiled TMA here, im2col TMA is not supported yet.

## What is TMA?

TMA load and store are methods:

- $\mathrm{tmaload}(\vec{x}, sa; ga, \vec{gs}, \vec{bs}, \vec{gr}, \vec{er})$.
- $\mathrm{tmastore}(\vec{x}, sa; ga, \vec{gs}, \vec{bs}, \vec{gr}, \vec{er})$.

The inputs and parameters are:

- $\vec{x}$ is the N-dimensional coordinate of the starting of the box. It is a vector of up to 5D.
- $sa$ stands for "Shared memory base Address".
- $ga$ stands for "Global memory base Address".
- $\vec{gs}$ stands for "Global Size". It is a vector of the same dimensionality as $\vec{x}$.
- $\vec{bs}$ stands for "Box Size". It is a vector of the same dimensionality as $\vec{x}$.
- $\vec{gr}$ stands for "Global stRide". It is a vector of the same dimensionality as $\vec{x}$.
- $\vec{er}$ stands for "Element stRide". It is a vector of the same dimensionality as $\vec{x}$.

Note that we used `;` instead of `,` to separate $sa$ with $ga$.
This is because everything before `;` is provided as an operand for the PTX instruction,
and everything after that is encoded inside the TensorMap descriptor.
As a result, when looking at the kernel level, only $\vec{x}$ and $sa$ can change;
other parameters are predefined constants.
We call $\vec{x}$ and $sa$ "inputs", and others "parameters".

$tmaload$ and $tmastore$ does the following (assuming the TMA is 2D here, can easily generalize to other dimensionalities):

```python
def tma(op, x, sa, *, ga, gs, bs, gr, er):
    # tile size
    ts = [ceildiv(bs[0], er[0]), ceildiv(bs[1], er[1])]
    for i0 in ts[0]:
        for i1 in ts[1]:
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