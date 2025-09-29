<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# The Mathematical Theory of IterDomain

> [!NOTE]
> We use $\div$ for true division, and $/$ for Euclidean division, and $\lceil/\rceil$ for ceil division. For example, $5\div 2 = 2.5$, $5/2=2$, $5\lceil/\rceil 2 = 3$.

## 1. IterDomain Transformations

**Definition 1.1 (IterDomain Transformation)**:
An *IterDomain transformation* of rank $(m, n)$ is a pair of two mappings $\langle\mathbb{Z}^{m}\to\mathbb{Z}^{n}, \mathbb{Z}^{m}\to\mathbb{Z}^{n}\rangle$,
called extent mapping and index mapping.
We use notation $t\langle E, I\rangle$ to denote an IterDomain transformation whose name is $t$, extent mapping is $E$ and index mapping is $I$.

**Definition 1.1.1 (Inner Split)**:
Given $d\in\mathbb{Z}^{+}$, an *inner split* with factor $d$ is an IterDomain transformation of rank $(1, 2)$,
where the extent map is $i \to (i \lceil/\rceil d, d)$,
and the index map is $i \to (i/d, i \mathbin{\\%} d)$.
We use notation $\mathrm{InnerSplit}(d)$ to denote inner split.

> [!CAUTION]
> TODO: outer split, merge, resize, swizzle, reorder

**Definition 1.2 (Equivalence of IterDomain Transformation)**:
Two IterDomain transformations of rank $(m, n)$ are equivalent if both the extent mapping and index mapping are equivalent.

We use the notation $t_1 = t_2$ to denote their equivalence.

**Theorem 1.1:**
For arbitrary three IterDomain transformations $t_1$, $t_2$ and $t_3$ of rank $(m, n)$, we have:

- Reflexivity: $t_1 = t_1$
- Symmetry: if $t_1 = t_2$, then $t_2 = t_1$.
- Transitivity: if $t_1 = t_2$ and $t_2 = t_3$, then $t_1 = t_3$.

That is, the equivalence of IterDomain transformations is an equivalence relation mathematically.

This theorem can be easily proved by applying the reflexivity, symmetry, and transitivity to both the extent mapping and index mapping.

**Definition 1.3: (Embedding of IterDomain Transformation)**:
Let $t_1\langle E, I\rangle$ be an IterDomain transformation of rank $(m, n)$, and $l, r \in \mathbb{N}$,
an IterDomain transformation $t_2$ of rank $(l + m + r, l + n + r)$ is an embedding of $t_1$ on dimensions $l$ to $l + m - 1$
if the extent mapping of $t_2$ is $(x_0, \ldots, x_{l + m + r - 1}) \to \left(x_0, \ldots, x_{l - 1}, E(x_l, \ldots, x_{l + m - 1}), x_{l + m}, \ldots, x_{l + m + r - 1}\right)$,
and the index mapping of $t_2$ is $(x_0, \ldots, x_{l + m + r - 1}) \to \left(x_0, \ldots, x_{l - 1}, I(x_l, \ldots, x_{l + m - 1}), x_{l + m}, \ldots, x_{l + m + r - 1}\right)$.

We use the notation $t_1[l, \ldots, l + m - 1]$ to the embedding of $t_1$ on dimensions $l$ to $l + m - 1$.

**Definition 1.4: (Composition of IterDomain Transformations)**:
Let $t_1\langle E_1, I_1\rangle$ be an IterDomain transformation of rank $(m, n)$,
and $t_2\langle E_2, I_2\rangle$ be an IterDomain transformation of rank $(n, l)$,
The composition of $t_1$ with $t_2$, denoted as $t_2 \circ t_1$, is the IterDomain transformation $\langle E_2 \circ E_1, I_2 \circ I_1\rangle$. The rank of $t_2 \circ t_1$ is $(m, l)$.

## 2. Properties of IterDomain Transformations

**Theorem 2.1 (Equivalence of Split-Split)**: Let $m, n \in \mathbb{Z}$, we have:
$$\mathrm{InnerSplit}(m)[0] \circ \mathrm{InnerSplit}(n) = \mathrm{InnerSplit}(n)[1] \circ \mathrm{InnerSplit}(m\cdot n)$$

Visually, we have:

![Equivalence of Split-Split](./iterdomain/split-split.svg)

<details>

**<summary>Proof:</summary>**

**The extent mapping:**

The extent mapping of $\mathrm{InnerSplit}(m)[0] \circ \mathrm{InnerSplit}(n)$ is $i \to (i \lceil/\rceil n \lceil/\rceil m, m, n)$.

The extent mapping of $\mathrm{InnerSplit}(n)[1] \circ \mathrm{InnerSplit}(m\cdot n)$ is $i \to (i \lceil/\rceil (m\cdot n), m, n)$.

According to Theorem 5.11 in [Integer Division](../math/integer-division.md): $i \lceil/\rceil n \lceil/\rceil m = i \lceil/\rceil (m\cdot n)$.

**The index mapping:**

The index mapping of $\mathrm{InnerSplit}(m)[0] \circ \mathrm{InnerSplit}(n)$ is $i \to (i / n / m, i / n \mathbin{\\%} m, i \mathbin{\\%} n)$.

The index mapping of $\mathrm{InnerSplit}(n)[1] \circ \mathrm{InnerSplit}(m\cdot n)$ is $i \to (i / (m\cdot n), i \mathbin{\\%} (m\cdot n) / n, i \mathbin{\\%} (m\cdot n) \mathbin{\\%} n)$.

According to Theorem 2.11 in [Integer Division](../math/integer-division.md): $i / n / m = i / (m\cdot n)$.

According to Theorem 2.12 in [Integer Division](../math/integer-division.md):

$$i \mathbin{\\%} (m\cdot n) = i \mathbin{\\%} n + ((i / n) \mathbin{\\%} m) \times n$$

According to Theorem 2.15.1 in [Integer Division](../math/integer-division.md):
$$i \mathbin{\\%} (m\cdot n) / n = (i / n) \mathbin{\\%} m$$

According to Theorem 2.7.1 in [Integer Division](../math/integer-division.md):
$$i \mathbin{\\%} (m\cdot n) \mathbin{\\%} n = i \mathbin{\\%} n$$

$\square$

</details>
