<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Divisibility of Split

## Introduction

In nvFuser, `Split` is an IterDomain expression that partitions the original IterDomain into nested sub-IterDomains,
where the outer IterDomain iterates over the quotient and the inner IterDomain iterates over the remainder of the original extent divided by the split factor.

For example, suppose that I have an IterDomain whose extent is `6`.
It is helpful to think of this IterDomain as the following loop:

```python
for i in range(6):
    print(i)
```

If I do a `split(2)` on this IterDomain, I will get two IterDomains whose extents are `3` and `2`.
It is helpful to think of these two IterDomains as two nested loops:

```python
for i0 in range(3):
    for i1 in range(2):
        print(i0 * 2 + i1)
```

If the split is divisible, everything is simple and elegant like above.
However, when splits are indivisible, things start to get complicated.
For example, let's still consider the IterDomain with extent `6` in the example.
But this time, we do a `split(4)` instead of `split(2)`.
With `split(4)`, we will get two IterDomains whose extents are `2` (`ceilDiv(6, 4)`) and `4`.
These two extents can be think of as two nested loops:

```python
for i0 in range(2):
    for i1 in range(4):
        print(i0 * 4 + i1)
```

But wait, is this correct? No, this is not.
Because now we are printing `0 1 2 3 4 5 6 7` instead of `0 1 2 3 4 5`.
The correct code should be:

```python
for i0 in range(2):
    for i1 in range(4):
        i = i0 * 4 + i1
        if i < 6:
            print(i)
```

That is, whenever we do an indivisible split on an IterDomain, we will index out-of-bound for that IterDomain.
Therefore, we must generate a predicate for the IterDomain being split.

Personally, I feel it helpful to consider indivisible split as resize + divisible split.
For example, I can consider `Split(I{6}, 4)` as `DivisibleSplit(Resize(I{6}, 8), 4)`.
This way, `DivisibleSplit` just converts one loop into two loops without hurting the correctness,
and `Resize(I{size1}, size2)` converts a

```python
for i in range(size1):
    print(i)
```

into

```python
for i in range(size2):
    if i < size1:
        print(i)
```

Indivisible split also impact the allocation size.

For example, 

If I have a tensor `T0[I1, I2]`, are the following two schedules equivalent?

- **Schedule 1:** `[I1, I2] -- split -> [I1, I2/4, 4] -- merge -> [I1*(I2/4), 4]`.
- **Schedule 2:** `[I1, I2] -- merge -> [I1*I2] -- split -> [(I1*I2)/4, 4]`.

Where the divisions above are all ceildiv.

If the split is indivisible, the answer is no.
We can see this from a simple example where `I1` has extent `2`, and `I2` has extent `5`.

For schedule 1, after schedule, the extents of the leaf domain will be `[2*2, 4]`.
So for this schedule, we will be iterating the tensor as:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[0, 5], T[0, 6] , T[0, 7]
T[1, 0], T[1, 1], T[1, 2] , T[1, 3]
T[1, 4], T[1, 5], T[1, 6] , T[1, 7]
```

For schedule 2, after schedule, the extents of the leaf domain will be `[3, 4]`.
So for this schedule, we will be iterating the tensor as:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[1, 0], T[1, 1] , T[1, 2]
T[1, 3], T[1, 4], T[2, 0] , T[2, 1]
```

They are clearly not equivalent.

What if the split is divisible? They are equivalent!

TODO: explain why?

## Implications

### Merging discontiguous IterDomains

- Q1: Can I merge two discontiguous IterDomains to create a larger IterDomain, and split out a vectorization IterDomain from this larger IterDomain?
- Q2: In TMA scheduling, can I create the TMA domain by merging two discontiguous IterDomains, and then split out a box?

The answer is: yes if and only if the split size divide the extent of the inner IterDomain of the merge.
If we merge discontiguous IterDomains then do a split that does not divide the inner extent,
we will end up iterating the tensor like schedule 2.
From the above listing, we see that after `T[0, 4]`, we should be accessing `T[1, 0]`,
which is not contiguous to `T[0, 4]` in memory.
However, vectorization and TMA can only access memory contiguously.

TODO: what if divisible?
