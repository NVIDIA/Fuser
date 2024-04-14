<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# TMA Support in NVFuser

## Introduction

TMA is a hardware feature that allows the GPU to load a tile from a tensor of up to 5D.
The tile can be dense, or strided, as shown in Figure 1 below:

![Figure 1: TMA dense and strided tile](tma/dense-and-strided-tile.svg)

When using phrases like "tile size", it is important to be clear what we are referring to.
Here we use two separate words *box* and *tile* to refer to different things.

**Definition 1**: In an N-dimensional tensor, a *box* of size `(s1, s2, ..., sN)` at
`(x1, x2, ..., xN)` refers to all the `s1*s2*...*sN` items at `(x1, x2, ..., xN)`,
`(x1, x2, ..., xN + 1)`, ..., `(x1, x2, ..., xN + sN - 1)`, ..., `(x1 + s1 - 1, x2 + s2 - 1, ..., xN + sN - 1)`.

**Definition 2**: In an N-dimensional tensor, a *tile* of size `(s1, s2, ..., sN)` and stride
`(r1, r2, ..., rN)` at `(x1, x2, ..., xN)` refers to all the `s1*s2*...*sN` items at
`(x1, x2, ..., xN)`, `(x1, x2, ..., xN + rN)`, ..., `(x1, x2, ..., xN + rN * (sN - 1))`,
..., `(x1 + r1 * (s1 - 1), x2 + r2 * (s2 - 1), ..., xN + rN * (sN - 1))`.

In Figure 1, we have box size `(6, 4)` for both diagram.
For the diagram on the left, we have tile size `(6, 4)` and stride `(1, 1)`.
For the diagram on the right, we have tile size `(6, 2)`, and stride `(1, 3)`.

## Schedule

In order to use TMA, we need to tell the hardware what is the dimensionality of our tensor.
Most naively, we can make this dimensionality the size of the allocation domain.
This naive mental model does provide us an easy way to reason about and start using TMA,
but unfortunately it is not flexible enough and is not consistent with how we think about scheduling.
For example, if we want to schedule a fusion containing only pure pointwise (i.e. no broadcasting) operations and all input and output tensors are contiguous,
regardless of the actual dimensionality of the input and output tensors, in our mental model, we always
consider this problem as a 1D problem by viewing all tensors as flattened 1D tensor.
For this case, ideally, we should be using 1D TMA, instead of using the actual dimensionality of the tensor.
That is, the dimensionality of TMA is not necessarily the same as the dimensionality of the tensor.

In order to support the flexibility of using a dimensionality of TMA different from the dimensionality of the tensor,
we design the scheduling of TMA as a multiple-step process:

### Step 1: create TMA domain

When a user is ready to schedule the consumer of the TMA expression,
the user should already have an idea of how the problem should be viewed.
For example, if the user is scheduling a fusion with only pure pointwise ops,
the user would want to view the problem as a 1D problem.
If the user is scheduling a transpose, then the user might want to view the problem as 2D.
If the user is scheduling a matmul, then the user might want to view the problem as 3D.
From this view of the problem, the user should have an idea about what are the dimensionalities of tensors.

For example, if the user wants to schedule a matmul `(M1, M2, K1, K2, K3) x (K1, K2, K3, N) -> (M1, M2, N)`,
then in the mind of the user, this matmul will be a 3D problem `(M1*M2, K1*K2*K3, N)`.
In this mental model, the input and output tensors are all 2D:
`(M1*M2, K1*K2*K3)`, `(K1*K2*K3, N)`, and `(M1*M2, N)`.

The first step of scheduling TMA is to schedule the consumer of the TMA expression the way matching the mental model of the problem.
The result domain of this step is called the *TMA domain*.

The TMA domain for the above matmul example is shown in the Figure 2 below:

![Figure 2: The TMA domain of the matmul example](tma/matmul-tma-domain.svg)

Please note that the TMA domain is not a member of a TensorDomain like the root/rFactor/allocation/leaf domains.
Instead, it is a virtual domain that only exists in the user's mind.

Also note that the IterDomain expressions between the global tensor's allocation domain and the TMA domain must be a view,
for example, we can not merge discontiguous IterDomains ([why?](../reading/divisibility-of-split.md#merging-discontiguous-iterdomains)), and we can not have indivisible splits either.

### Step 2: Define box

After having scheduled a TMA domain, the next step is to define box.
There are two ways of defining box: partitioning and compositing.

#### Define box by partitioning

Defining box by partitioning is as simple as: select an IterDomain in the TMA domain, then
inner split that IterDomain by the box size of that dimension.

We call this split expression a "*boxing split*", the input of this split a "*partitioned IterDomain*",
the inner output of this split a "*box IterDomain*", and the outer output of this split a "*coordinate IterDomain*".

For the case of Figure 1, if both box dimensions are defined by partitioning,
the schedule should look like the Figure 3 below:

![Figure 3: Boxing by partitioning](tma/box-by-partitioning.svg)

Please note that, although in the above example, the split is divisible, this does not have to be the case in general.

#### Define box by compositing

TODO: this documentation is under construction
