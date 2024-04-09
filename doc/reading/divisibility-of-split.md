<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Divisibility of Split

> [!NOTE]
> We use $\div$ for true division, and $/$ for integer division. For example, $5\div 2 = 2.5$, $5/2=2$.

## Introduction

In nvFuser, `Split` is an IterDomain expression that partitions the original IterDomain into nested sub-IterDomains,
where the outer IterDomain iterates over the quotient and the inner IterDomain iterates over the remainder of the original extent divided by the split factor.

For example, suppose that I have an IterDomain `I0` whose extent is `6`.
It is helpful to think of this IterDomain as the following loop:

```python
for i in range(6):
    print(i)
```

If I do a `Split(I0, 2)` on this IterDomain, I will get two IterDomains whose extents are `3` and `2`.
It is helpful to think of these two IterDomains as two nested loops:

```python
for i0 in range(3):
    for i1 in range(2):
        print(i0 * 2 + i1)
```

If the split is divisible, everything is simple and elegant like above.
However, when splits are indivisible, things start to get complicated.
For example, let's still consider the `I0` as an example.
But this time, we do a `Split(I0, 4)` instead of `Split(I0, 2)`.
With `Split(I0, 4)`, we will get two IterDomains whose extents are `2` (`ceilDiv(6, 4)`) and `4`.
These two extents can be think of as two nested loops:

```python
for i0 in range(2):
    for i1 in range(4):
        print(i0 * 4 + i1)
```

But wait, is this correct? No, this is not.
Because now we are printing `0 1 2 3 4 5 6 7` instead of `0 1 2 3 4 5`.

That is, whenever we do an indivisible split on an IterDomain, we effectively changed its range as well.
We call this added extra range "holes".
In the above example, the indivisible split creates two holes in `I0`.

To maintain program semantics, we must add predicates checking whether we are at a valid item or at a hole:

```python
for i0 in range(2):
    for i1 in range(4):
        i = i0 * 4 + i1
        if i < 6:
            print(i)
```

Personally, I sometimes feel it helpful to consider indivisible split as resize + divisible split.
For example, I can consider `Split(I{6}, 4)` as `DivisibleSplit(Resize(I{6}, 0, 2), 4)`.
This way, hole creation and loop transformations are separated.
`DivisibleSplit` just converts one loop into two loops without while maintaining correctness.
`Resize(I{size1}, 0, right_expand)` create holes and introduce predicates:

```python
# Before resize
for i in range(size1):
    print(i)

# After resize
for i in range(size1 + right_expand):
    if i < size1:
        print(i)
```

## Predication

As we have seen above, indivisible splits create holes in the iteration,
therefore predicates must be introduced.
Let's consider the following example in Figure 1:

![Figure 1: Three Indivisible Splits](divisibility-of-split/three-indivisible-splits.svg)

In this example, there are three indivisible splits.
The indices of the inputs of all these splits, `i0`, `i1`, and `i2`, could potentially run out of boundary.

In order to maintain program semantics, the safest predication strategy is to predicate on all these three IterDomains:

```python
for i3 in range(2):
    for i4 in range(2):
        for i5 in range(2):
            for i6 in range(4):
                i1 = i3 * 2 + i4
                i2 = i5 * 4 + i6
                i0 = (i3 * 2 + i4) * 6 + (i5 * 4 + i6)
                if i1 < 3 and i2 < 6 and i0 < 15:
                    print(i0)
```

and this indeed works correctly.

Can this be simplified? What if we just predicate `i0` and nothing else?
We will get `0 1 2 3 4 5 6 7 6 7 8 9 10 11 12 13 12 13 14`.
We do print the correct set of values, but we are printing some values multiple times.
If all we care is to print the correct set of values, and we don't mind whether there are duplicates, this can be one strategy.
But this is clearly not equivalent to the program prior to transformation.

From the above example, we can see that overflow of an intermediate IterDomain does not necessarily results in an overflow of their ancestors.
To maintain program semantics, the predicate we use must be logically equivalent to predicating all holes.

The following theorems are useful tools to find mathematically simple predicates that are logically equivalent to predicating all holes:

**Theorem 1** Suppose that there is a split `I1, I2 = Split(I0, N)`.
Then "the index of `I0` is in bound" implies "the index of `I1` is in bound".

<details>

**<summary>Proof:</summary>**

Suppose the index of `I1` is $i_1$, the index of `I2` is $i_2$, the extent of `I0` is $S$.
The index of `I0` is then $i_0 = i_1 \times N + i_2$.
"the index of `I0` is in bound" means $i_0 < S$.
Because $i_2 \ge 0$,
$$i_0 < S \implies i_1 \times N < S \implies i_1 < S \div N \implies i_1 < \mathrm{ceilDiv}(S, N)$$
$\square$

</details>

The above theorem tells us that, if we are already predicating `I0`, then there is no need to additionally predicate `I1`.
We can derive similar theorems for other IterDomain expressions:

**Theorem 2** Suppose that there is a merge `I2 = Merge(I0, I1)`.
Then "the index of `I0` is in bound" is equivalent to "the index of `I2` is in bound".

<details>

**<summary>Proof:</summary>**

Suppose the index of `I2` is $i_2$, the extent of `I1` is $N$.
Then the index of `I0` is $i_0 = i_2 / N$.
Suppose that the extents of `I0` and `I2` are $P$ and $Q$, then $Q = N \times P$.
"the index of `I0` is in bound" means $i_0 < P$, which is:
$$i_2 / N < P$$
According to "Rule 1" in `[Simplification of boolean predicates]` in `csrc/expr_simplifier.h`,
(TODO: move this theorem to a md file)
$$i_2 / N < P \Leftrightarrow i_2 < Q$$
$\square$

</details>

**Theorem 3** Suppose that there is a resize `I1 = Resize(I0, L, R)`.
Then "the index of `I0` is in bound" implies "the index of `I1` is in bound" if $R >= 0$.

<details>

**<summary>Proof:</summary>**

Suppose the index of `I1` is $i_1$, the extent of `I0` is $N$.
The index of `I0` is then $i_0 = i_1 - L$.
The extent of `I1` is `N + L + R`
"the index of `I0` is in bound" means $i_0 < N$.
Because $R \ge 0$,
$$i_0 < N \Leftrightarrow i_1 < N + L \implies i_1 < N + L + R$$
$\square$

</details>

**Theorem 4** Suppose that there is a swizzle `I2, I3 = Swizzle(I0, I1)`.
Then "the index of `I0` is in bound" is equivalent to "the index of `I2` is in bound".

<details>

**<summary>Proof:</summary>**

Suppose the index of `I2` is $i_2$.
The index of `I0` is the same as `I2`: $i_0 = i_2$.
`I0` and `I2` also have the same extent.
$\square$

</details>

## Allocation and correctness model

Indivisible split also impact the allocation size.

For example, if I have a tensor `T0[I0{6}]`, when allocating this tensor,
I will allocate 6 items for this tensor.
If I do an indivisible split on `I0{6}` by `4` to get `I1{2}` and `I2{4}`,
and set the allocation domain of this tensor as `[I1, I2]`,
then I will need to allocate `2*4 = 8` items.
With the mental model of considering indivisible split as resize + divisible split,
we can consider the allocation of this example as Figure 2:

![Figure 2](divisibility-of-split/allocate-6-as-2,4.svg)

We call the above situation *over-allocated*, that is,
we are not only allocating data, but also allocating holes.

Because there are holes in the allocation due to indivisible split,
a natural question to ask is: When we write to an over-allocated buffer,
what value should we fill these holes as?
Possible answers are:

1. As long as there is no "Illegal Memory Access" killing my kernel,
   I don't care about what values these holes have.
2. The holes should be filled with 0.
3. The holes should be filled with x (something different from zero).

Which answer is correct? There is no certain answer.
It totally depends on how these values are read:
If the out-of-bound items are never read, then 1, 2, or 3 makes no difference.
If the out-of-bound items are actually read,
then it needs to be filled with some neutral value that effectively leads to a no-op.
For example, if we are doing an unpredicated reduction on an over-allocated buffer,
such as when using tensor core, we must fill the out-of-bound items correctly.
If the reduction is sum, then the out-of-bound items must be filled with 0.
If the reduction is product, then the out-of-bound items must be filled with 1.
This inspires the concept of weak and strong correctness:

**Definition 1:** A schedule/lowering strategy is weakly correct if all the in-bound items in the consumer's allocation domain are filled with the correct value,
and there is no error raise in the kernel.

**Definition 2:** A schedule/lowering strategy is strongly correct if all the in-bound items in the consumer's allocation domain are filled with the correct value,
and all the out-of-bound values are filled with a desired filling value.

## Case studies

### Merge-then-split vs split-then-merge

#### Question

If I have a tensor `T0[I1, I2]`, are the following two schedules equivalent?

- **Schedule 1:** `[I1, I2] -- split -> [I1, ceilDiv(I2,N), N] -- merge -> [I1*ceilDiv(I2,N), N]`.
- **Schedule 2:** `[I1, I2] -- merge -> [I1*I2] -- split -> [ceilDiv(I1*I2,N), N]`.

#### Answer

*If the split is indivisible, the answer is no.*

We can see this from a simple example where `I1` has extent `2`, and `I2` has extent `5`, and `N` is `4`.

For schedule 1, after schedule, the extents of the leaf domain will be `[2*2, 4]`.
So for this schedule, we will be iterating the tensor as the following Listing 1:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[0, 5], T[0, 6] , T[0, 7]
T[1, 0], T[1, 1], T[1, 2] , T[1, 3]
T[1, 4], T[1, 5], T[1, 6] , T[1, 7]
```

For schedule 2, after schedule, the extents of the leaf domain will be `[3, 4]`.
So for this schedule, we will be iterating the tensor as as the following Listing 2:

```python
T[0, 0], T[0, 1], T[0, 2] , T[0, 3]
T[0, 4], T[1, 0], T[1, 1] , T[1, 2]
T[1, 3], T[1, 4], T[2, 0] , T[2, 1]
```

They are clearly not equivalent.

*If the split is divisible then the answer is yes.*

Let's say the extents of `I1` and `I2` are $N_1$ and $N_2$.
Being divisible means $N$ divide $N_2$.

The extent of the first dimension of schedule 1 is then $N_1 \times (N_2 \div N)$,
which is the same as the extent of the first dimension in schedule 2 $(N_1 \times N_2) \div N$.

Assume the leaf indices are $i$ and $j$.

In schedule 1, the indices of `I1` and `I2` are
$i/(N_2 / N)$ and $i \mathbin{\\%} (N_2 / N) \times N + j$.
According to Theorem 2.10 in [Integer Division](../math/integer-division.md),
$$i/(N_2 / N) = (i \times N) / N_2$$
According to Theorem 2.13 in [Integer Division](../math/integer-division.md),
$$i \mathbin{\\%} (N_2 / N) \times N + j = (i \times N) \mathbin{\\%} N_2  + j$$

In schedule 2, the indices of `I1` and `I2` are
$(i \times N + j) / N_2$ and $(i \times N + j) \mathbin{\\%} N_2$.
According to the theorem in `expr_simplifier.cpp` before `distributeGcdRemainderDivMod` (TODO: move this to md file),
$$(i \times N + j) / N_2 = (i \times N) / N_2$$
$$(i \times N + j) \mathbin{\\%} N_2 = (i \times N) \mathbin{\\%} N_2  + j$$

We can see that `I1` and `I2` has both the same extent and the same indices.
They are therefore equivalent.

### Merging discontiguous IterDomains

#### Question

- Q1: Can I merge two discontiguous IterDomains to create a larger IterDomain, and split out a vectorization IterDomain from this larger IterDomain?
- Q2: In TMA scheduling, can I create the TMA domain by merging two discontiguous IterDomains, and then split out a box?

#### Answer

The answer is yes if and only if the split size divide the extent of the inner IterDomain of the merge.

If we merge discontiguous IterDomains then do a split that does not divide the inner extent,
we will end up iterating the tensor like schedule 2 in the [above example](#merge-then-split-vs-split-then-merge).
From the Listing 2 there, we see that after `T[0, 4]`, we should be accessing `T[1, 0]`,
which is not contiguous to `T[0, 4]` in memory.
However, vectorization and TMA can only access memory contiguously.

If divisible, then the merge-then-split is equivalent to a split-then-merge, which is totally valid.
