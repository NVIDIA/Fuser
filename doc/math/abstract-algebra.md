<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# A Brief Overview of Abstract Algebra

The real numbers together with binary operators $+$ $-$ $\times$ $/$ form a [field](https://en.wikipedia.org/wiki/Field_(mathematics)),
which is defined by its basic properties:

1. **Associativity:** $a + (b + c) = (a + b) + c$, and $a \times (b \times c) = (a \times b) \times c$.
2. **Commutativity:** $a + b = b + a$, and $a \times b = b \times a$.
3. **Distributivity:** $a \times (b + c) = (a \times b) + (a \times c)$.
4. **Identity:** there exist two *different* elements $0$ and $1$ such that $a + 0 = a$ and $a \times 1 = a$.
5. **Additive inverses:** for every $a$, there exists an element, denoted $-a$, such that $a + (-a) = 0$.
6. **Defition of subtraction:** $a - b \coloneqq a + (-b)$
7. **Multiplicative inverses:** for every $a \neq 0$, there exists an element, denoted by $1/a$, such that $a \times (1/a) = 1$.
8. **Defition of division:** $a / b \coloneqq a \times (1/b)$

Thanks to the field properties, div of real number has the following properties:

9. **Associativity:**
   - $a \times (b/c) = (a \times b)/c$
   - $a/(b \times c) = (a/b)/c$
   - $a/(b/c) = (a/b) \times c = (a \times c)/b $
10. **Right Distributivity:** $(a+b)/c = a/c + b/c$

For the set of integers $\mathbb{Z}$, (7) is no longer true, therefore,
the division can not be defined as (8), and as a result, neither (9) nor (10) is true anymore.
So, the mathematical structure of integer arithmetics is not as strong as a field.
The integer arithmetics form an [Euclidean domain](https://en.wikipedia.org/wiki/Euclidean_domain)

The relationship between Euclidean domains and other algebraic structures is as follows
$$\text{rings} > \text{commutative rings} > \text{[...]} > \text{Euclidean domains} > \text{fields}$$
where $A > B$ means $A$ is more general than $B$, that is, every $B$ is an $A$,
that is, $B$ has more structure/stronger properties than $A$

where "commutative rings" is almost a field except that:
- in (4), $0$ and $1$ are not required to be different
- in (7) and (8) are not required to be true

the [...] above can be further expanded as:
$$> \text{integral domains} > \text{GCD domains} > \text{unique factorization domains} >$$

The expanded [...] and Euclidean domains are made by gradually adding basic
properties of integers to the recipe. These properties include:
- **integral domains:**
  - $0 \neq 1$
  - if $a \neq 0$ and $b \neq 0$, then $ab \neq 0$.
  - **cancellation property:** if $a \neq 0$, then $ab=ac$ implies $b=c$.
- **GCD domains:** the existence of greatest common divisor
- **unique factorization domains:** every non-zero element can be uniquely written as a product of prime elements (for example, $12 = 2 \times 2 \times 3$)
- **Euclidean domains:** [the fundamental division-with-remainder property](integer-division.md)
