<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Monotonic Function

Reference: [Wikipedia](https://en.wikipedia.org/wiki/Monotonic_function)

**Definition 1:** A function is called strictly increasing if for all $x < y$,
  we have $f(x) < f(y)$.

**Definition 2:** A function is called weakly increasing if for all $x \le y$,
  we have $f(x) \le f(y)$.

**Theorem 1:** if $f$ is strictly increasing, then

1. $x < y \Leftrightarrow f(x) < f(y)$
2. $x \le y \Leftrightarrow f(x) \le f(y)$

<details>

**<summary>Proof:</summary>**

$x < y \implies f(x) < f(y)$ by definition.

$x \le y \implies f(x) \le f(y)$:
$x \le y$ means either $x < y$ or $x = y$. For the first case, $f(x) < f(y)$.
For the second case, $f(x) = f(y)$, in combination, $f(x) \le f(y)$.

$f(x) < f(y) \implies x < y$:
Assume $x < y$ was not true, then $y \le x$, then $f(y) \le f(x)$,
which conflict with $f(x) < f(y)$.

$f(x) \le f(y) \implies x \le y$:
Assume $x \le y$ was not true, then $y < x$, then $f(y) < f(x)$, which conflict with $f(x) \le f(y)$.
$\square$

</details>

**Theorem 2:** if $f$ is weakly increasing, then
1. $x \le y \implies f(x) \le f(y)$
2. $x < y \implies f(x) \le f(y)$
3. $f(x) \le f(y)$ does not provide much information, especially, it is not guaranteed that $x \le y$
4. $f(x) < f(y) \implies x < y$

<details>

**<summary>Proof:</summary>**

$x \le y \implies f(x) \le f(y)$ by definition.

$x < y \implies f(x) \le f(y)$ because $x < y \implies x \le y$.

Consider $f(x) = 1$ (constant function), then $f$ is weakly increasing.
For this function, $f(x) \le f(y)$ is trivially true, and $x$ and $y$ can be arbitrary number.

$f(x) < f(y) \implies x < y$:
Assume that $x < y$ was not true, that is, $y \le x$, then $f(y) \le f(x)$,
which conflict with $f(x) < f(y)$
$\square$

</details>
