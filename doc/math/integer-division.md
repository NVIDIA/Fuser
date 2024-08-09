<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Integer Division

> [!NOTE]
> We use $\div$ for true division, and $/$ for integer division (which may refer to different things in different sections). For example, $5\div 2 = 2.5$, $5/2=2$.

We learnt arithmetic from as early as elementary school,
and have been used to simplify expressions using rules like $a(b+c) = ab+ac$,
but extra care is needed when dealing with integer divisions
(such as Euclidean division as defined in number theory, truncation division as used by C++, and floor division used by Python).
Because unlike real numbers, integer division can be anti-intuitive:
- $(a + b) / c \neq a/c + b/c$, for example: $(1 + 1) / 2 \neq 1/2 + 1/2$
- $(a / b) \times c \neq a / (b / c)$, for example: $(64 / 256) \times 4 \neq 64 / (256 / 4)$
- $(a / b) \times c \neq a \times (c / b)$, for example: $(1 / 4) \times 256 \neq 1 \times (256 / 4)$


## 1. A Review of Number theory

The definition of div and mod comes from the *Division Theorem*
(aka. the *Fundamental Division-With-Remainder Property*, aka. *Euclid's Division Lemma*)
described as below:

**Theorem 1.1 (Division Theorem):** For any integers $a$ and $b$ ($b \neq 0$), there
exist unique integers $q$ and $r$ such that:
1. $0 \leq r < |b|$
2. $a = bq + r$

if we define $a/b \coloneqq q$, and $a \mathbin{\\%} b \coloneqq r$, then we can write (2) as:
$$a = (a/b)b + a\mathbin{\\%}b$$
I will call the above equation "the fundamental division-with-remainder equation" in later contexts.
We call this definition of division Euclidean division.
Note that this is not how we define integer division in C++,
as explained [later](#3-implementations-of-div-and-mod).

A very important concept in number theory is "congruence", as described below:

**Definition 1.1 (Congruent):** For $c \neq 0$, if $(a-b)\div c$ is an integer, then we say $a$ is
congruent to $b$ modulo $c$, written as $a = b \pmod c$.

Note that the $\pmod c$ in the statement $a = b \pmod c$ should be understood
as a qualifier for the statement $a = b$ that modifies the meaning of the
original statement [P] to "[P] under the modular arithmetic". It should NOT
be understood as an operator operating on $b$, i.e. $a = (b \pmod c)$.

**Theorem 1.2:** The congruence is an equivalence relationship, that is:
- $a = a \pmod c$
- if $a = b \pmod c$, then $b = a \pmod c$
- if $a = b \pmod c$ and $b = d \pmod c$, then $a = d \pmod c$

**Theorem 1.3:** The congruence also has the following properties:
- $a = b \pmod c$ iff $a = b \pmod {(-c)}$
- if $a = a' \pmod c$, $b = b' \pmod c$, then
  - $a + b = a' + b' \pmod c$
  - $ab = a'b' \pmod c$
- if $bd = bd' \pmod c$ and $\mathrm{gcd}(b, c) = 1$, then $d = d' \pmod c$

## 2. More Theorems Of Euclidean Division

In this section, I will prove a few more theorems that I didn't find in textbooks but still feel useful for us.
The $/$ and $\mathbin{\\%}$ are defined under Euclidean division, not under truncation division as in C++.
The properties of $/$ and $\mathbin{\\%}$ under truncation division will be revisited [later](#4-properties-of-truncation-division).

**Theorem 2.1:** Euclidean division is NOT associative:
1) $a \times (b/c) \neq (a \times b)/c$
2) $a/(b \times c) \neq (a/b)/c$
3) $a/(b/c) \neq (a/b) \times c \neq (a \times c)/b$

<details>

**<summary>Proof:</summary>**

- $a \times (b/c) \neq (a \times b)/c$ because of the counter example $2 \times (3/2) \neq (2 \times 3)/2$.
- $a/(b \times c) \neq (a/b)/c$ because of the counter example $4/((-1) \times 5) \neq (4/(-1))/5$
  (note that $4/(-5) = 0$, $(-4)/5 = -1$).
- $a/(b/c) \neq (a/b) \times c \neq (a \times c)/b$ because of the counter example that
  $5/(3/2)$, $(5/3) \times 2$, and $(5 \times 2)/3$ are three different numbers: $5$, $2$, and $3$.

$\square$

</details>

**Theorem 2.2:** Euclidean division is NOT right distributive

<details>

**<summary>Proof:</summary>**

Counter example: $(1+1)/2 \neq 1/2 + 1/2$ $\square$

</details>

**Theorem 2.3:** $a\mathbin{\\%} b = a'\mathbin{\\%} b$ is equivalent to $a = a' \pmod b$

<details>

**<summary>Proof:</summary>**

*Direction ==>:*
$a\mathbin{\\%}b = a'\mathbin{\\%}b$ is equivalent to $a-(a/b)b = a'-(a'/b)b$
which is equivalent to $(a-a')\div b = (a/b-a'/b) = \text{integer}$
So $a\mathbin{\\%}b = a'\mathbin{\\%}b$ ==> $a = a' \pmod b$.

*Direction <==:*
if $a = a' \pmod b$, then $a = a' + kb$.
According to the Euclid's division lemma, $a' = q'b + r'$,
then $a = (q'+k)b + r'$, where $q'+k$ and $r'$.
It is easy to verify that, if we define $q = q'+k$ and $r = r'$,
then $a = qb + r$ also satisfies the condition in Euclid's division lemma.
Thanks to the uniqueness of $q$ and $r$, we have $a\mathbin{\\%}b = r = r' = a'\mathbin{\\%}b$.
$\square$

</details>

Thanks to the great property of Theorem 2.3 some theorems below can be easily proved by converting $\mathbin{\\%}$ into congruence.
But unfortunately, as we can see [later](#3-implementations-of-div-and-mod), for the truncation division in C++,
the beautiful Theorem 2.3 does not hold, so many theorems in this section needs to be modified if we are considering division in C++.

**Theorem 2.4:** $a = a \mathbin{\\%} b \pmod b$

<details>

**<summary>Proof:</summary>**

According to Euclid's division lemma, $(a - a \mathbin{\\%} b) \div b = q$ is integer.
$\square$

</details>

**Theorem 2.5:** if $0 \le r < |a|$, then $r \mathbin{\\%} a = r$, $r / a = 0$.

<details>

**<summary>Proof:</summary>**

This can be proved directly from Euclid's division lemma.
$\square$

</details>

**Theorem 2.6:** $a/(-b) = -a/b$, $a\mathbin{\\%}(-b) = a\mathbin{\\%}b$

<details>

**<summary>Proof:</summary>**

$a = bq+r$ is equivalent to $a = (-b)(-q) + r$
Due to the uniqueness of $q$ and $r$, we get our conclusion.
$\square$

</details>

**Theorem 2.7:** $(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c$

<details>

**<summary>Proof:</summary>**

According to Theorem 2.3, this is just to prove
$$a + b = a \mathbin{\\%} c + b \mathbin{\\%} c \pmod c$$
Because of Theorem 2.4, we have
$$a = a \mathbin{\\%} c \pmod c$$
$$b = b \mathbin{\\%} c \pmod c$$
applying Theorem 1.3, we get what we need.
$\square$

</details>

**Theorem 2.7.1:** If $a \mathbin{\\%} c = 0$, we have $(a + b) \mathbin{\\%} c = b \mathbin{\\%} c$

<details>

**<summary>Proof:</summary>**

$$(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c = b \mathbin{\\%} c \mathbin{\\%} c = b \mathbin{\\%} c$$
$\square$

</details>

**Theorem 2.7.2:** Let $g = gcd(a, c)$. If $0 \le b < |g|$, we have $(a + b) \mathbin{\\%} c = a \mathbin{\\%} c + b$.

<details>

**<summary>Proof:</summary>**

Because $0 \le b < |g|$, and $|g| \le |c|$, we know $0 \le b < |c|$.
According to Theorem 2.5, we have $$b \mathbin{\\%} c = b$$
According to Theorem 2.13, we have $$a \mathbin{\\%} c = ((a/|g|) \mathbin{\\%} (c/|g|)) \times |g|$$
So
$$a \mathbin{\\%} c + b \mathbin{\\%} c = ((a/|g|) \mathbin{\\%} (c/|g|)) \times |g| + b$$
Because $$0 \le (a/|g|) \mathbin{\\%} (c/|g|) < |c/g|$$
for integers, we have
$$0 \le (a/|g|) \mathbin{\\%} (c/|g|) \le |c/g| - 1$$
So
$$0 \le a \mathbin{\\%} c \le (|c/g| - 1) \times |g|$$
Therefore, we have
$$0 \le a \mathbin{\\%} c + b \mathbin{\\%} c < (|c/g| - 1) \times |g| + |g|$$
That is: $$0 \le a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$$

Therefore by Theorem 2.7:
$$(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c = a \mathbin{\\%} c + b \mathbin{\\%} c = a \mathbin{\\%} c + b$$
$\square$

</details>

**Theorem 2.8:** $(a \times b) \mathbin{\\%} c = (a \mathbin{\\%} c \times b \mathbin{\\%} c) \mathbin{\\%} c$

<details>

**<summary>Proof:</summary>**

Similar to above
$\square$

</details>

**Theorem 2.9:** If $a$ is a multiple of $b$, then $a \mathbin{\\%} b = 0$.

<details>

**<summary>Proof:</summary>**

This can be proved directly from Euclid's division lemma.
$\square$

</details>

**Theorem 2.10:** If $b$ is a multiple of $c$, then we have: $a\times (b/c) = (a\times b)/c$.

<details>

**<summary>Proof:</summary>**

If $b$ is a multiple of $c$, then:
$$(a \times b)\mathbin{\\%}c = a \times (b\mathbin{\\%}c) = 0$$

From the fundamental division-with-remainder equation, we know that:
$$b = (b/c) \times c + b\mathbin{\\%}c \text{ ... (eq 1)}$$
$$(a \times b) = ((a \times b)/c) \times c + (a \times b)\mathbin{\\%}c \text{ ... (eq 2)}$$

multiply $a$ to both side of (eq 1), we get:
$$(a \times b) = a \times (b/c) \times c + a \times (b\mathbin{\\%}c)$$

subtract (eq 2) by the above equation, we have:
$$0 = [(a \times b)/c - a \times (b/c)] \times c + [(a \times b)\mathbin{\\%}c - a \times (b\mathbin{\\%}c)]$$

The second term in the above equation is $0$, so we have
$$0 = [(a \times b)/c - a \times (b/c)]  \times  c$$

Because $c \neq 0$, we have $(a \times b)/c = a \times (b/c)$.
$\square$

</details>

**Theorem 2.11:** If $b > 0$, then $a/(b \times c) = (a/b)/c$.

<details>

**<summary>Proof:</summary>**

from the fundamental division-with-remainder equation, we have:
$$a = (a/b)b + a\mathbin{\\%}b = (((a/b)/c)c + (a/b)\mathbin{\\%}c)b + a\mathbin{\\%}b
  = ((a/b)/c) \times bc + (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) \text{ ... (eq 1)}$$
where
$$0 \le a\mathbin{\\%}b < b$$
$$0 \le (a/b)\mathbin{\\%}c \times b \le (|c| - 1) \times b$$

as a result, we have:
$$0 \le (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) < |bc|$$

from the fundamental division-with-remainder equation, we can uniquely decompose $a$ as
$$a = (a/(bc)) \times (bc) + a\mathbin{\\%}(bc) \text{ ... (eq 2)}$$

since $a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$ is in the correct range of $a\mathbin{\\%}(bc)$
and due to the uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have:
$$a/(bc) = (a/b)/c$$
$$a\mathbin{\\%}(bc) = a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$$
$\square$

</details>

**Theorem 2.12:** If $b > 0$, then $a \mathbin{\\%} (b \times c) = a \mathbin{\\%} b + ((a / b) \mathbin{\\%} c) \times b$.

<details>

**<summary>Proof:</summary>**

Already proved in the proof of Theorem 2.11
$\square$

</details>

**Theorem 2.13:** If $d > 0$ and $d$ divides $a$ and $b$, then $a \mathbin{\\%} b = ((a / d) \mathbin{\\%} (b / d)) \times d$.

<details>

**<summary>Proof:</summary>**

From the fundamental division-with-remainder equation, we have
$$b = d \times (b / d)$$
then
$$a \mathbin{\\%} b = a \mathbin{\\%} (d \times (b / d))$$
From Theorem 2.12, we have:
$$a \mathbin{\\%} b = a \mathbin{\\%} (d \times (b / d)) = a \mathbin{\\%} d + ((a / d) \mathbin{\\%} (b / d)) \times d$$
according to Theorem 2.9:
$$a \mathbin{\\%} d = 0$$
so we have:
$$a \mathbin{\\%} b = ((a / d) \mathbin{\\%} (b / d)) \times d$$
$\square$

</details>

**Theorem 2.14:** If $b$ is a multiple of $c$ and $c > 0$, then $a/(b/c) = (a \times c)/b$.

<details>

**<summary>Proof:</summary>**

If $b$ is a multiple of $c$, then we have:
$$b \mathbin{\\%} c = 0$$
Also, according to Theorem 2.13:
$$(ac) \mathbin{\\%} b = (a \mathbin{\\%} (b / c)) \times c$$
From the fundamental division-with-remainder equation, we have:
$$ac = ((ac)/b)b + (ac)\mathbin{\\%}b \text{ ... (eq 1)}$$
$$a = (a/(b/c))(b/c) + a\mathbin{\\%}(b/c) \text{ ... (eq 2)}$$
Multiply $c$ to both side of (eq 2), and note that $b$ is a multiple of $c$:
$$ac = (a/(b/c))b + (a\mathbin{\\%}(b/c)) \times c$$
subtract the above equation with (eq 1), we get
$$0 = ( a/(b/c) - (ac)/b ) \times b + ( (a\mathbin{\\%}(b/c)) \times c - (ac)\mathbin{\\%}b)$$
The second term in the above equation is $0$, so we have
$$0 = (a/(b/c) - (ac)/b) \times b$$
because $b \neq 0$, we get $a/(b/c) = (ac)/b$.
$\square$

</details>

**Theorem 2.15:** If $a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$, then $(a+b)/c = a/c + b/c$.

<details>

**<summary>Proof:</summary>**

From Theorem 2.7 and Theorem 2.5, we have
$$(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c = a \mathbin{\\%} c + b \mathbin{\\%} c$$
From the fundamental division-with-remainder equation, we have:
$$(a + b) = ((a + b) / c) * c + (a + b) \mathbin{\\%} c$$
$$a = (a / c) * c + a \mathbin{\\%} c$$
$$b = (b / c) * c + b \mathbin{\\%} c$$
Adding the last two equations and subtract with the first equation:
$$0 = ((a/c + b/c) - (a+b)/c) * c + (a \mathbin{\\%} c + b \mathbin{\\%} c - (a + b) \mathbin{\\%} c)$$
The second term in the above equation is $0$, so we have
$$0 = ((a/c + b/c) - (a+b)/c) * c$$
Because $c \neq 0$, we have $(a/c + b/c) = (a+b)/c$.
$\square$

</details>

**Theorem 2.15.1:** If $a \mathbin{\\%} c = 0$, we have $(a+b)/c = a/c + b/c$.

<details>

**<summary>Proof:</summary>**

If $a \mathbin{\\%} c = 0$, we have $$a \mathbin{\\%} c + b \mathbin{\\%} c = b \mathbin{\\%} c < |c|$$
From Theorem 2.15, we get the conclusion.
$\square$

</details>

**Theorem 2.15.2:** Let $g = gcd(a, c)$. If $0 \le b < |g|$, we have $(a + b) / c = a/c$.

<details>

**<summary>Proof:</summary>**

Similar to the proof of 2.7.2, we have
$$0 \le b < |c|$$
$$0 \le a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$$
So we have
$$(a + b) / c = a/c + b/c = a/c$$
$\square$

</details>

**Theorem 2.16:** If $d > 0$, we have $i / d < D \Leftrightarrow i < D \times d$.

<details>

**<summary>Proof:</summary>**

$i / d < D \implies i < D \times d$:

Consider the function $f(x) = x / d$, it is weakly increasing.
Also note that $D = (D \times d) / d$.
According to Theorem 2 (4) in [Monotonic Function](monotonic-function.md),
we have $f(i) < f(D \times d) \implies i < D \times d$.

$i < D \times d \implies i / d < D$:

According to the fundamental division-with-remainder equation,
$i < D \times d$ can be written as $i / d \times d + i \mathbin{\\%} d < D \times d$,
where $i \mathbin{\\%} d \ge 0$.
So $$i / d \times d \le i / d \times d + i \mathbin{\\%} d < D \times d$$
Consider the function $g(x) = x \times d$, which is strictly increasing.
According to Theorem 1 (1) in [Monotonic Function](monotonic-function.md),
$g(i / d) < g(D)$ implies $i/d < D$.
$\square$

</details>

## 3. Implementations of Div and Mod

Unfortunately, modern hardwares and programming languages does not implement div and mod consistent with Euclid's division lemma,
although these implementations can be converted with Euclid's division easily.
The implementations of div and mod depends on programming languages.
The comparison of these implementations and their properties are discussed in the following paper:

> Boute, Raymond T. "The Euclidean definition of the functions div and mod." ACM Transactions on Programming Languages and Systems (TOPLAS) 14.2 (1992): 127-144.

I will summarize some useful points from the above paper here, and add my own comments:

For $a \ge 0$ and $b > 0$, all implementation of $a/b$ and $a\mathbin{\\%}b$ are the same
and consistent with the Euclid's division.
So no brainer in this region.

Except for a few Languages (ISO Standard Pascal, Algol, Ada) all implementations are consistent with the fundamental division-with-remainder equation,
although the range and sign of $r$ can be different and the value of $q$ can be different by $1$.
Implementations not satisfying the fundamental division-with-remainder equation is considered wrong because it has no mathematical properties.
For all implementations, $|a \mathbin{\\%} b| < |b|$.

Common implementations are:

- **truncation division (round to zero):**
  - $a/b \coloneqq \mathrm{trunc}(a \div b)$
  - $a\mathbin{\\%}b$ defined by the fundamental division-with-remainder equation

- **floor division:**
  - $a/b \coloneqq \mathrm{floor}(a \div b)$
  - $a\mathbin{\\%}b$ defined by the fundamental division-with-remainder equation

For C89, the result of negative division is not specified. C99 and C++ uses truncation division.
Python and PyTorch uses floor division.
We will only be interested in truncation division in nvFuser because we use C++.

The properties of truncation division are:
1. Good: $(-a)/b = -(a/b) = a/(-b)$
2. Good: $(-a)\mathbin{\\%}b = -(a\mathbin{\\%}b) = a\mathbin{\\%}(-b)$
3) Bad: $a \mathbin{\\%} b = a' \mathbin{\\%} b$ is not equivalent to $a = a' \pmod b$

For all types of division (Euclidean/truncation/floor) $f(x) = x / d$,
$f$ is weakly increasing if $d > 0$, and weakly decreasing if $d < 0$.

Besides truncation and floor division, ceil division is also commonly used.
Although there is very little programming language implementing its division operator `/` as ceil division,
programs commonly implement ceil division as its utility.
The definition of ceil division is as follow:

- **ceil division:**
  - $a/b \coloneqq \mathrm{ceil}(a \div b)$
  - $a\mathbin{\\%}b$ defined by the fundamental division-with-remainder equation

## 4. Properties of Truncation Division

In this section, I will study truncation division and its properties.
I will first redefine truncation division using the same language as in Euclid's division lemma,
which will be convenient for proving theorems.
I will then prove that this new definition is equivalent to the definition of truncation division as described in [the previous section](#3-implementations-of-div-and-mod).
Then I will study the theorems in [section 2](#2-more-theorems-of-euclidean-division) to find out which is true and which needs change.
All $/$ and $\mathbin{\\%}$ in this section are using truncation division.

Note that the truncation division is the definition of div and mod in C and C++, as C99 standard says:
> When integers are divided, the result of the $/$ operator is the algebraic quotient with any fractional part discarded.
If the quotient $a/b$ is representable, the expression $(a/b)*b + a\mathbin{\\%}b$ shall equal $a$.

**Definition 4.0:** For any integers $a$ and $b$ ($b \neq 0$), there exist unique integers $q$ and $r$ such that
1. if $a \ge 0$, $0 \le r < |b|$; if $a < 0$, $-|b| < r \le 0$.
2. $a = bq + r$
We can then define $a/b \coloneqq q$, $a\mathbin{\\%}b \coloneqq r$.

**Theorem 4.0:** Definition 4.0 is equivalent to the definition of truncation division in
[the previous section](#3-implementations-of-div-and-mod).

<details>

**<summary>Proof:</summary>**

$\mathrm{trunc}(a\div b)$ is to remove the non-integer portion of $a \div b$.
The trunction function can be implemented by shifting $a$ towards zero by $r$,
which obtains $a - r$, to the nearest multiple of $b$.
We are shifting towards zero if and only if $r$ and $a$ have the same sign.
Also, $|r|$ must be smaller than $|b|$ otherwise it won't be the nearest multiple of $b$, and this shift is unique.
We therefore proved (1) and (2) above and the uniqueness.
$\square$

</details>

Now let's review [the theorems of Euclidean division](#2-more-theorems-of-euclidean-division) to see if they are still valid, and if not,
how they should be modified when translating to the language of truncation division.
Theorems will be numbered consistently, that is, theorem 4.x is the modified version of theorem 2.x.

**Theorem 4.1:** Associativity of truncation division:
1. $a \times (b/c) \neq (a \times b)/c$
2. $a/(b \times c) = (a/b)/c$
3. $a/(b/c) \neq (a/b) \times c \neq (a \times c)/b$

Note that for (2), it is now a " $=$ " instead of a " $\neq$ ".

<details>

**<summary>Proof:</summary>**

For (1) and (3), the same counter example as in Theorem 2.1 applies.

For (2), from Definition 4.0, we have
$$a = (a/b)b + a\mathbin{\\%}b = (((a/b)/c)c + (a/b)\mathbin{\\%}c)b + a\mathbin{\\%}b = ((a/b)/c) \times bc + (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) \text{ ... (eq 1)}$$

If $a \ge 0$, then
$$0 \le a\mathbin{\\%}b < |b|$$
$$0 \le ((a/b)\mathbin{\\%}c) \times b = ((a/|b|)\mathbin{\\%}c) \times |b| \le (|c| - 1)|b|$$
as a result, we have
$$0 \le (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) < |bc|$$

From Definition 4.0, we can uniquely decompose $a$ as
$$a = (a/(bc)) \times (bc) + a\mathbin{\\%}(bc) \text{ ... (eq 2)}$$
since $a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$ is in the correct range of $a\mathbin{\\%}(bc)$,
and due to the uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
$$a/(bc) = (a/b)/c$$
$$a\mathbin{\\%}(bc) = a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$$

If $a < 0$, then
$$-|b| < a\mathbin{\\%}b \le 0$$
$$-(|c| - 1)|b| \le ((a/b)\mathbin{\\%}c) \times b = ((a/|b|)\mathbin{\\%}c) \times |b| \le 0$$
as a result, we have
$$-|bc| \le (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) \le 0$$

From Definition 4.0, we can uniquely decompose $a$ as
$$a = (a/(bc)) \times (bc) + a\mathbin{\\%}(bc) ... (eq 3)$$
since $a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$ is in the correct range of $a\mathbin{\\%}(bc)$,
and due to the uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
$$a/(bc) = (a/b)/c$$
$$a\mathbin{\\%}(bc) = a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$$
$\square$

</details>

**Theorem 4.2:** Truncation division is NOT right distributive

<details>

**<summary>Proof:</summary>**

The same counter example as in Theorem 2.2 applies.
$\square$

</details>

For truncation division, Theorem 2.3 no longer holds, because $-3 = 2 \pmod 5$, however,
$-3 \mathbin{\\%} 5 = -3$, but $2 \mathbin{\\%} 5 = 2$.

**Theorem 4.3:**
1. $a\mathbin{\\%}b = a'\mathbin{\\%}b = 0$ is equivalent to $a = a' = 0 \pmod b$.
2. $a\mathbin{\\%}b = a'\mathbin{\\%}b \neq 0$ is equivalent to $a = a' \neq 0 \pmod b$ and $\mathrm{sign}(a)=\mathrm{sign}(a')$.
3. $a\mathbin{\\%}b = a'\mathbin{\\%}b + |b|$ is equivalent to $a = a' \neq 0 \pmod b$ and $a>0$ and $a'<0$.

<details>

**<summary>Proof:</summary>**

For (1):
$a\mathbin{\\%}b = a'\mathbin{\\%}b = 0$ is equivalent to $a=bq$ and $a'=bq'$, which is equivalent to
$a = a' = 0 \pmod b$

For (2) Direction ==>:

$a\mathbin{\\%}b = a'\mathbin{\\%}b$ is equivalent to $a-(a/b)b = a'-(a'/b)b$
which is equivalent to $(a-a')\div b = (a/b-a'/b) = \text{integer}$.
So $a\mathbin{\\%}b = a'\mathbin{\\%}b$ ==> $a = a' \pmod b$.
Also, from (1), we know that $a \neq 0 \pmod b$ and $a' \neq 0 \pmod b$.

From Definition 4.0, we know that since $a\mathbin{\\%}b$ is not $0$,
there is no overlap on the range of $a\mathbin{\\%}b$ for positive $a$ and negative $a$.
So the sign of $a$ and $a'$ must match, otherwise it is impossible to have $a\mathbin{\\%}b = a'\mathbin{\\%}b$.

Direction <==:

if $a = a' \pmod b$, then $a = a' + kb$.
According to Definition 0, $a' = q'b + r'$, then $a = (q'+k)b + r'$.
Because $\mathrm{sign}(a)=\mathrm{sign}(a')$, if $r'$ is in the correct range for $a'$,
then it will also be in the correct range for $a$.
Due to the uniqueness, we have $a\mathbin{\\%}b = r' = a'\mathbin{\\%}b$.

For (3) Direction ==>:

If $a\mathbin{\\%}b = a'\mathbin{\\%}b + |b|$, then $0 < a\mathbin{\\%}b < |b|$ and $|b| < a'\mathbin{\\%}b < 0$,
that is, $a>0$ and $a'<0$.
Also, we have $a-(a/b)b = a'-(a'/b)b + |b|$, that is,
$$(a-a')\div b = (a/b-a'/b+sign(b)) = \text{integer}$$
so $a = a' \pmod b$.

Also, from (1), we know that $a \neq 0 \pmod b$ and $a' \neq 0 \pmod b$.

Direction <==:

If $a = a' \neq 0 \pmod b$, then $a = a' + kb$.
According to Definition 0, $a' = q'b + r'$, where $-|b| < r' \le 0$.
from (1), we know that $r' \neq 0$, so $-|b| < r' < 0$.
So
$$a = (q'+k)b + r' = (q+k-sign(b))b + r' + |b|$$
Let $q = q' + k$, $r = r' + |b|$,
it is easy to verify that $0 < r < |b|$.
Due to the uniqueness:
$a\mathbin{\\%}b = r' + |b|$
$a/b = q+k-sign(b)$
$\square$

</details>

**Theorem 4.4:** $a = a \mathbin{\\%} b \pmod b$

<details>

**<summary>Proof:</summary>**

According to Definition 4.0, $(a - a \mathbin{\\%} b) \div b = q$ is integer.
$\square$

</details>

**Theorem 4.5:** If $-|a| < r < |a|$, then $r \mathbin{\\%} a = r$, $r / a = 0$.

<details>

**<summary>Proof:</summary>**

This can be proved directly from Definition 4.0.
$\square$

</details>

**Theorem 4.6:** $a/(-b) = -a/b$, $a\mathbin{\\%}(-b) = -a\mathbin{\\%}b$

<details>

**<summary>Proof:</summary>**

See [the previous section](#3-implementations-of-div-and-mod), this is written in the paper.
$\square$

</details>

**Theorem 4.7:** If $\mathrm{compatible\\_sign}(a, b)$, then $(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c$,
where $\mathrm{compatible\\_sign}(a, b)$ is defined as $ab \ge 0$.

<details>

**<summary>Proof:</summary>**

According to Theorem 4.3, this is just to prove
$$a + b = a \mathbin{\\%} c + b \mathbin{\\%} c \pmod c$$
Because of Theorem 4.4, we have
$$a = a \mathbin{\\%} c \pmod c$$
$$b = b \mathbin{\\%} c \pmod c$$
applying Theorem 1.3, we get what we want.
$\square$

</details>

**Theorem 4.7.1:** If $\mathrm{compatible\\_sign}(a, b)$ and $a \mathbin{\\%} c = 0$, we have $(a + b) \mathbin{\\%} c = b \mathbin{\\%} c$,
where $\mathrm{compatible\\_sign}(a, b)$ is defined as $ab \ge 0$.

<details>

**<summary>Proof:</summary>**

Similar to the proof of 2.7.1
$\square$

</details>

**Theorem 4.7.2:** Let $g = gcd(a, c)$. If $\mathrm{compatible\\_sign}(a, b)$ and $-|g| < b < |g|$, we have $(a + b) \mathbin{\\%} c = a \mathbin{\\%} c + b$.

<details>

**<summary>Proof:</summary>**

Similar to the proof of 2.7.2. We have
$$-|c| < b < |c|$$
and
$$-|c/g| + 1 \le (a/|g|) \mathbin{\\%} (c/|g|) \le |c/g| - 1$$
then
$$-|c| < a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$$

Therefore by Theorem 4.7:
$$(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c = a \mathbin{\\%} c + b \mathbin{\\%} c = a \mathbin{\\%} c + b$$
$\square$

</details>

**Theorem 4.8:** If $\mathrm{compatible\\_sign}(a, b)$, then $(a \times b) \mathbin{\\%} c = (a \mathbin{\\%} c \times b \mathbin{\\%} c) \mathbin{\\%} c$,
where $\mathrm{compatible\\_sign}(a, b)$ is defined as $ab \ge 0$.

<details>

**<summary>Proof:</summary>**

Similar to above.
$\square$

</details>

**Theorem 4.9:** If $a$ is a multiple of $b$, then $a \mathbin{\\%} b = 0$.

<details>

**<summary>Proof:</summary>**

This can be proved directly from Definition 4.0.
$\square$

</details>

**Theorem 4.10:** If $b$ is a multiple of $c$, then we have: $a \times (b/c) = (a \times b)/c$.

<details>

**<summary>Proof:</summary>**

Same proof as 2.10.
$\square$

</details>

**Theorem 4.11:** $a/(b \times c) = (a/b)/c$

<details>

**<summary>Proof:</summary>**

This is part of Theorem 4.1.
$\square$

</details>

**Theorem 4.12:** $a \mathbin{\\%} (b \times c) = a \mathbin{\\%} b + ((a / b) \mathbin{\\%} c)  \times b$

<details>

**<summary>Proof:</summary>**

Already proved in the proof of Theorem 4.1.
$\square$

</details>

**Theorem 4.13:** If $d$ divides $a$ and $b$, then $a \mathbin{\\%} b = ((a / d) \mathbin{\\%} (b / d)) \times d$.

<details>

**<summary>Proof:</summary>**

Same proof as 2.13.
$\square$

</details>

**Theorem 4.14:** If $b$ is a multiple of $c$, then $a/(b/c) = (a \times c)/b$.

<details>

**<summary>Proof:</summary>**

Proof: Same proof as 2.14.
$\square$

</details>

**Theorem 4.15:** If $\mathrm{compatible\\_sign}(a, b)$ and $-|c| < a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$,
then $(a+b)/c = a/c + b/c$. ,
The $\mathrm{compatible\\_sign}(a, b)$ is defined as $ab \ge 0$.

<details>

**<summary>Proof:</summary>**

From Theorem 4.7 and Theorem 4.5
$$(a + b) \mathbin{\\%} c = (a \mathbin{\\%} c + b \mathbin{\\%} c) \mathbin{\\%} c = a \mathbin{\\%} c + b \mathbin{\\%} c$$
The rest of the proof is the same as 2.15.
$\square$

</details>

**Theorem 4.15.1:** If $\mathrm{compatible\\_sign}(a, b)$ and $a \mathbin{\\%} c = 0$,
then $(a+b)/c = a/c + b/c$. ,
The $\mathrm{compatible\\_sign}(a, b)$ is defined as $ab \ge 0$.

<details>

**<summary>Proof:</summary>**

If $a \mathbin{\\%} c = 0$, we have $$-|c| < a \mathbin{\\%} c + b \mathbin{\\%} c = b \mathbin{\\%} c < |c|$$
From Theorem 4.15, we get the conclusion.
$\square$

</details>

**Theorem 4.15.2:** Let $g = gcd(a, c)$. If $\mathrm{compatible\\_sign}(a, b)$ and $-|g| < b < |g|$, we have $(a + b) / c = a / c$.

<details>

**<summary>Proof:</summary>**

Similar to the proof of 2.7.2, we have
$$-|c| < b < |c|$$
and
$$-|c| < a \mathbin{\\%} c + b \mathbin{\\%} c < |c|$$

Therefore:
$$(a + b) / c = a/c + b/c = a / c$$
$\square$

</details>

**Theorem 4.16:** If $i \ge 0$ and $d > 0$, we have $i / d < D \Leftrightarrow i < D \times d$.

<details>

**<summary>Proof:</summary>**

Similar to Theorem 2.16, except that we need both $i \ge 0$ and $d > 0$ to make $i \mathbin{\\%} d \ge 0$.
$\square$

</details>

## 5. Properties of Ceil Division

In this section, I will study ceil division and its properties.
I will first redefine ceil division using the same language as in Euclid's division lemma,
which will be convenient for proving theorems.
I will then prove that this new definition is equivalent to the definition of ceil division as described in [section 3](#3-implementations-of-div-and-mod).
Then I will study the theorems in [section 2](#2-more-theorems-of-euclidean-division) to find out which is true and which needs change.
All $/$ and $\mathbin{\\%}$ in this section are using ceil division.

**Definition 5.0:** For any integers $a$ and $b$ ($b \neq 0$), there exist unique integers $q$ and $r$ such that
1. if $b > 0$, $-b < r \le 0$, otherwise $0 \le r < |b|$.
2. $a = bq + r$
We can then define $a/b \coloneqq q$, $a\mathbin{\\%}b \coloneqq r$.

**Theorem 5.0:** Definition 5.0 is equivalent to the definition of ceil division in
[section 3](#3-implementations-of-div-and-mod).

<details>

**<summary>Proof:</summary>**

According to Theorem 1.1, there exist unique integers $q'$ and $r'$ such that:
1. $0 \leq r' < |b|$
2. $a = b \cdot q' + r'$

We can define $r$ as:

- $r'$ if $r' = 0$ or $b < 0$
- $r' - b$ otherwise

It is easy to verify that condition 1 is satisfied.

Also, we can define $q$ as $q'$ if $r' = 0$ or $b < 0$, otherwise define $q$ as $q' + 1$.
It is easy to verify that $a = b \cdot q + r$.

To verify that $q = \mathrm{ceil}(a \div b)$, we observe that $q = (a - r) \div b = a \div b - r \div b$.
If $r = 0$, $q = a \div b = \mathrm{ceil}(a \div b)$.
If $r \neq 0$, we always have $0 < - r \div b < 1$,
which is to "add a positive portition to $a \div b$ to make it an integer",
which matches with the definition of $\mathrm{ceil}$.

$\square$

</details>

**Theorem 5.11:** If $c > 0$, then $a/(b \times c) = (a/b)/c$.

<details>

**<summary>Proof:</summary>**

from the fundamental division-with-remainder equation, we have:
$$a = (a/b)b + a\mathbin{\\%}b = (((a/b)/c)c + (a/b)\mathbin{\\%}c)b + a\mathbin{\\%}b
  = ((a/b)/c) \times bc + (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) \text{ ... (eq 1)}$$

If $b > 0$, then $bc > 0$, we have:
$$-b < a\mathbin{\\%}b \le 0$$
$$-(c - 1) \times b \le (a/b)\mathbin{\\%}c \times b \le 0$$

as a result, we have:
$$- bc < (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) \le 0$$

If $b < 0$, then $bc < 0$, we have:
$$0 \le a\mathbin{\\%}b < -b$$
$$0 \le (a/b)\mathbin{\\%}c \times b \le -(c - 1) \times b$$

as a result, we have:
$$0 \le (a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b) < -bc$$

from the fundamental division-with-remainder equation, we can uniquely decompose $a$ as
$$a = (a/(bc)) \times (bc) + a\mathbin{\\%}(bc) \text{ ... (eq 2)}$$

since $a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$ is in the correct range of $a\mathbin{\\%}(bc)$
and due to the uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have:
$$a/(bc) = (a/b)/c$$
$$a\mathbin{\\%}(bc) = a\mathbin{\\%}b + ((a/b)\mathbin{\\%}c) \times b$$
$\square$

</details>

**Theorem 5.12:** If $c > 0$, then $a \mathbin{\\%} (b \times c) = a \mathbin{\\%} b + ((a / b) \mathbin{\\%} c) \times b$.

<details>

**<summary>Proof:</summary>**

Already proved in the proof of Theorem 5.11
$\square$

</details>
