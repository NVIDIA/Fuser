<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Mathematical Logic

**Theorem 1:** If $p \rightarrow r$, then $p \land r \leftrightarrow p$.

<details>

**<summary>Proof:</summary>**

$p \land r \leftarrow p$: By [Absorption](https://en.wikipedia.org/wiki/Absorption_(logic))

$p \land r \rightarrow p$: By [Conjunction elimination](https://en.wikipedia.org/wiki/Conjunction_elimination)

$\square$

</details>

Theorem 1 can be used to simplify boolean predicates.
For example, if I have a predicate $i < 5 \land i < 6$,
let $p = i < 5$ and $r = i < 6$, then from Theorem 1,
we know that the original predicate is equivalent to $i < 5$.
So we have simplified the predicate without changing when this predicate is true and when it is false.
