<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Mathematical Logic

**Theorem 1:** If $p \Rightarrow (q \Leftrightarrow r)$, then $p \land q \Leftrightarrow p \land r$.

<details>

**<summary>Proof:</summary>**

It is obvious that $p \land q \Rightarrow p$.
Also, $p \land q$ implies $q \land (q \leftrightarrow r)$, which further implies $r$.
Therefore, $p \land q \Rightarrow p \land r$.
The other direction can be proved similarily.
$\square$

</details>

Theorem 1 tells us that, when simplifying boolean predicates like
$p_1 \land p_2 \land \cdots \land p_n$,
we can simplifying each term separately, and when simplifying one term, we can assume other terms are true.

For example, if I have a predicate $i \ge 0 \land i < 6 \land i \mathbin{\\%} 6 < 3$,
then I can simplify $i \mathbin{\\%} 6 < 3$ assuming $i \ge 0 \land i < 6$.
Due to the assumption, $i \mathbin{\\%} 6 \equiv i$,
so $i \mathbin{\\%} 6 < 3$ is logically equivalent to $i < 3$.
Therefore, the overall predicate can be simplified as $i \ge 0 \land i < 6 \land i < 3$.
Applying the same rule, we can simplify $i < 6$ assuming $i < 3$, which leads to $i < 6$ being simplified as $\mathrm{true}$.
So the the overall predicate becomes $i \ge 0 \land \mathrm{true} \land i < 3$,
which can be further simplified as $i \ge 0 \land i < 3$.
