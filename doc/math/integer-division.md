<!--
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
-->

# Integer Division

**Note on notation:**
- We use $\div$ for true division, and $/$ for integer division. For example, $5\div 2 = 2.5$, $5/2=2$.

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
$$a = (a/b)b + a\%b$$
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
- if $bd = bd' \pmod c$ and $gcd(b, c) = 1$, then $d = d' \pmod c$

## 2. More Theorems Of Euclidean Division

In this section, I will prove a few more theorems that I didn't find in textbooks but still feel useful for us.
The $/$ and $\%$ are defined under Euclidean division, not under truncation division as in C++.
The properties of $/$ and $\%$ under truncation division will be revisited [later](#4-properties-of-div-and-mod-under-trunc-div).

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

</details>

**Theorem 2.2:** Euclidean division is NOT right distributive

<details>

**<summary>Proof:</summary>**

Counter example: $(1+1)/2 \neq 1/2 + 1/2$

</details>

**Theorem 2.3:** $a\\% b = a'\% b$ is equivalent to $a = a' \pmod b$

<details>

**<summary>Proof:</summary>**

Direction ==>:
$a\%b = a'\%b$ is equivalent to $a-(a/b)b = a'-(a'/b)b$
which is equivalent to $(a-a')\div b = (a/b-a'/b) = \text{integer}$
So $a\%b = a'\%b$ ==> $a = a' \pmod b$
Direction <==:
if $a = a' \pmod b$, then $a = a' + kb$.
According to the Euclid's division lemma, $a' = q'b + r'$,
then $a = (q'+k)b + r'$, where $q'+k$ and $r'$.
It is easy to verify that, if we define $q = q'+k$ and $r = r'$,
then $a = qb + r$ also satisfies the condition in Euclid's division lemma.
Thanks to the uniqueness of $q$ and $r$, we have $a\%b = r = r' = a'\%b$

</details>

Thanks to the great property of Theorem 2.3 some theorems below can be easily proved by converting $\%$ into congruence.
But unfortunately, as we can see [later](#3-implementations-of-div-and-mod), for the truncation division in C++,
the beautiful Theorem 2.3 does not hold, so many theorems in this section needs to be modified if we are considering division in C++.

**Theorem 2.4:** $a = a \% b \pmod b$

<details>

**<summary>Proof:</summary>**

According to Euclid's division lemma, $(a - a \% b) \div b = q$ is integer

</details>

Theorem 4.5: if 0 <= r < |a|, then r % a = r, r / a = 0
Proof: This can be proved directly from Euclid's division lemma

Theorem 4.6: a/(-b) = -a/b, a%(-b) = a%b
Proof: a = bq+r is equivalent to a = (-b)(-q) + r
Due to the uniqueness of q and r, we get our conclusion

Theorem 4.7: (a + b) % c = (a % c + b % c) % c
Proof: According to Theorem 2.3, this is just to prove
a + b = a % c + b % c \pmod c
Because of Theorem 2.4, we have a = a % c \pmod c, b = b % c \pmod c
applying Theorem 3.3, we get what we need.

Theorem 4.8: (a * b) % c = (a % c * b % c) % c
Proof: Similar to above

Theorem 4.9: If a is a multiple of b, then a % b = 0
Proof: This can be proved directly from Euclid's division lemma

Theorem 2.10: If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
Proof: If b is a multiple of c, then (a*b)%c = a*(b%c) = 0
From the fundamental division-with-remainder equation, we know that:
b = (b/c)*c + b%c ... (eq 1)
(a*b) = ((a*b)/c)*c + (a*b)%c ... (eq 2)
multiply a to both side of (eq 1), we get:
(a*b) = a*(b/c)*c + a*(b%c)
subtract (eq 2) by the above equation, we have:
0 = [(a*b)/c - a*(b/c)]*c + [(a*b)%c - a*(b%c)]
The second term in the above equation is 0, so we have
0 = [(a*b)/c - a*(b/c)] * c
Because c \neq 0, we have (a*b)/c = a*(b/c)

Theorem 2.11: If b > 0, then a/(b*c) = (a/b)/c
Proof: from the fundamental division-with-remainder equation, we have
a = (a/b)b + a%b = (((a/b)/c)c + (a/b)%c)b + a%b
  = ((a/b)/c)*bc + (a%b + ((a/b)%c)*b) ... (eq 1)
where 0 <= a%b < b and 0 <= (a/b)%c*b <= (|c| - 1)*b,
as a result, we have 0 <= (a%b + ((a/b)%c)*b) < |bc|,
from the fundamental division-with-remainder equation, we can uniquely
decompose a as a = (a/(bc))*(bc) + a%(bc)  ... (eq 2)
since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
uniqueness of this decomposition, comparing (eq 1) and (eq 2) have
a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b

Theorem 2.12: If b > 0, then a % (b * c) = a % b + ((a / b) % c) * b
Proof: Already proved in the proof of Theorem 2.11

Theorem 2.13: If d > 0 and d divides a and b, then
a % b = ((a / d) % (b / d)) * d
Proof: From the fundamental division-with-remainder equation, we have
b = d * (b / d), then a % b = a % (d * (b / d))
From Theorem 2.12, we have:
a % b = a % (d * (b / d)) = a % d + ((a / d) % (b / d)) * d
according to Theorem 4.9, a % d = 0, so we have
a % b = ((a / d) % (b / d)) * d

Theorem 2.14: If b is a multiple of c and c > 0, then a/(b/c) = (a*c)/b
Proof: If b is a multiple of c, then we have b % c = 0
Also, according to Theorem 2.13, (ac) % b = (a % (b / c)) * c
From the fundamental division-with-remainder equation, we have:
ac = ((ac)/b)b + (ac)%b ... (eq 1)
a = (a/(b/c))(b/c) + a%(b/c) ... (eq 2)
Multiply c to both side of (eq 2), and note that b is a multiple of c:
ac = (a/(b/c))b + (a%(b/c)) * c
subtrace the above equation with (eq 1), we get
0 = [a/(b/c) - (ac)/b] * b + [(a%(b/c))*c - (ac)%b]
The second term in the above equation is 0, so we have
0 = [a/(b/c) - (ac)/b] * b
because b \neq 0, we get a/(b/c) = (ac)/b

Theorem 2.15: If a % c + b % c < |c|, then (a+b)/c = a/c + b/c
Proof: From Theorem 4.7 and Theorem 4.5, we have
(a + b) % c = (a % c + b % c) % c = a % c + b % c
From the fundamental division-with-remainder equation, we have:
(a + b) = ((a + b) / c) * c + (a + b) % c
a = (a / c) * c + a % c
b = (b / c) * c + b % c
Adding the last two equations and subtract with the first equation:
0 = [(a/c + b/c) - (a+b)/c] * c + [a % c + b % c - (a + b) % c]
The second term in the above equation is 0, so we have
0 = [(a/c + b/c) - (a+b)/c] * c
Because c \neq 0, we have (a/c + b/c) = (a+b)/c

## 3. Implementations of Div and Mod

Unfortunately, modern hardwares and programming languages does not implement
div and mod consistent with Euclid's division lemma, although these
implementations can be converted with Euclid's division easily. The
implementations of div and mod depends on programming languages. The
comparison of these implementations and their properties are discussed in the
following paper:

Boute, Raymond T. "The Euclidean definition of the functions div and mod." ACM Transactions on Programming Languages and Systems (TOPLAS) 14.2 (1992): 127-144.

I will summarize some useful points from the above paper here, and add my own
comments:

For a >= 0 and b > 0, all implementation of a/b and a%b are the same and
consistent with the Euclid's division. So no brainer in this region.

Except for a few Languages (ISO Standard Pascal, Algol, Ada) all
implementations are consistent with the fundamental division-with-remainder
equation, although the range and sign of r can be different and the value of
q can be different by 1. Implementations not satisfying the fundamental
division-with-remainder equation is considered wrong because it has no
mathematical properties. For all implementations, |a % b| < |b|.

Common implementations are:

trunc div (round to zero):
a/b \coloneqq trunc(a \ b)
a%b defined by the fundamental division-with-remainder equation

floor div:
a/b \coloneqq floor(a \ b)
a%b defined by the fundamental division-with-remainder equation

For C89, the result of negative div is not specified. C99 and C++ uses trunc
div. Python and PyTorch uses floor div. We will only be interested in trunc
div here because we use C++.

The properties of trunc div are:
1) Good: (-a)/b = -(a/b) = a/(-b)
2) Good: (-a)%b = -(a%b) = a%(-b)
3) Bad: a % b = a' % b is not equivalent to a = a' \pmod b)

## 4. Properties of Div and Mod Under Trunc Div

In this section, I will study trunc div and its properties. I will first
redefine trunc div using the same language as in Euclid's division lemma,
which will be convenient for proving theorems. I will then prove that this
new definition is equivalent to the definition of trunc div as described in
section "Implementations of Div and Mod". Then I will study the theorems in
section "Some More Theorems" to find out which is true and which needs
change. All / and % in this section are using trunc div.

Note that the trunc div is the definition of div and mod in C and C++, as C99
standard says: When integers are divided, the result of the / operator is the
algebraic quotient with any fractional part discarded. If the quotient a/b is
representable, the expression (a/b)*b + a%b shall equal a.

Definition 6.0: For any integers a and b (b \neq 0), there exist unique
integers q and r such that
1) if a >= 0, 0 <= r < |b|; if a < 0, -|b| < r <= 0.
2) a = bq + r
We can then define a/b \coloneqq q, a%b \coloneqq r

Theorem 6.0: Definition 6.0 is equivalent to the definition of trunc div in
section "Implementations of Div and Mod"
Proof: trunc(a\b) is to remove the non-integer portion of a\b. To shift a by
r, which obtains a - r, to the nearest multiple of b. To shift towards zero,
r and a should have the same sign. Also, |r| must be smaller than |b|
otherwise it won't be the nearest multiple of b, and this shift is unique.
We therefore proved 1) and 2) above and the uniqueness.

Now let's look at theorems in "Some More Theorems" to see how they should be
modified when translating to the language of trunc div. Theorems will be
numbered consistently, that is, theorem 6.x is the modified version of
theorem 4.x.

Theorem 6.1: Associativity of trunc div:
1) a*(b/c) \neq (a*b)/c
2) a/(b*c) = (a/b)/c
3) a/(b/c) \neq (a/b)*c \neq (a*c)/b
Note that for 2), it is now a "=" instead of a "\neq".
Proof: for 1) and 3), the same counter example as in Theorem 2.1 applies.
For 2), from Definition 6.0, we have
a = (a/b)b + a%b = (((a/b)/c)c + (a/b)%c)b + a%b
  = ((a/b)/c)*bc + (a%b + ((a/b)%c)*b)  ... (eq 1)
  = ((a/b)/c)*bc + (a%b + ((a/|b|)%c)*|b|)
if a >= 0, then 0 <= a%b < |b|, 0 <= (a/|b|)%c*|b| <= (|c| - 1)|b|,
as a result, we have 0 <= (a%b + ((a/|b|)%c)*|b|) < |bc|,
from Definition 6.0, we can uniquely decompose a as
a = (a/(bc))*(bc) + a%(bc) ... (eq 2)
since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b
if a < 0, then -|b| < a%b <= 0, -(|c| - 1)|b| <= ((a/|b|)%c)*|b| <= 0,
as a result, we have -|bc| <= (a%b + ((a/|b|)%c)*|b|) <= 0,
from Definition 6.0, we can uniquely decompose a as
a = (a/(bc))*(bc) + a%(bc) ... (eq 3)
since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b

Theorem 6.2: Integer div is NOT right distributive
Proof: the same counter example as in Theorem 2.2 applies.

For trunc div, Theorem 2.3 no longer holds, because -3 = 2 \pmod 5), however,
-3 % 5 = -3, but 2 % 5 = 2.

Theorem 6.3:
1) a%b = a'%b = 0 is equivalent to a = a' = 0 \pmod b)
2) a%b = a'%b \neq 0 is equivalent to a = a' \neq 0 \pmod b) and sign(a)=sign(a')
3) a%b = a'%b + |b| is equivalent to a = a' \neq 0 \pmod b) and a>0 and a'<0
Proof: For 1):
a%b = a'%b = 0 is equivalent to a=bq and a'=bq', which is equivalent to
a = a' = 0 \pmod b)
For 2) Direction ==>:
a%b = a'%b is equivalent to a-(a/b)b = a'-(a'/b)b
which is equivalent to (a-a')\b = (a/b-a'/b) = integer.
So a%b = a'%b ==> a = a' \pmod b),
also, from 1), we know that a \neq 0 \pmod b) and a' \neq 0 \pmod b)
From Definition 6.0, we know that since a%b is not 0,there is no overlap on
the range of a%b for positive a and negative a. So the sign of a and a' must
match, otherwise it is impossible to have a%b = a'%b.
Direction <==:
if a = a' \pmod b), then a = a' + kb.
According to Definition 0, a' = q'b + r', then a = (q'+k)b + r'.
Because sign(a)=sign(a'), if r' is in the correct range for a', then it will
also be in the correct range for a. Due to the uniqueness, a%b = r' = a'%b
For 3) Direction ==>:
If a%b = a'%b + |b|, then 0 < a%b < |b| and |b| < a'%b < 0,
that is, a>0 and a'<0.
Also, we have a-(a/b)b = a'-(a'/b)b + |b|, that is,
(a-a')\b = (a/b-a'/b+sign(b)) = integer, so a = a' \pmod b)
also, from 1), we know that a \neq 0 \pmod b) and a' \neq 0 \pmod b)
Direction <==:
if a = a' \neq 0 \pmod b), then a = a' + kb.
According to Definition 0, a' = q'b + r', where -|b| < r' <= 0.
from 1), we know that r' \neq 0, so -|b| < r' < 0.
So a = (q'+k)b + r' = (q+k-sign(b))b + r' + |b|.
Let q = q' + k, r = r' + |b|
it is easy to verify that 0 < r < |b|
Due to the uniqueness, a%b = r' + |b|, a/b = q+k-sign(b)

Theorem 6.4: a = a % b \pmod b)
Proof: According to Definition 0, (a - a % b) \ b = q is integer

Theorem 6.5: If -|a| < r < |a|, then r % a = r, r / a = 0
Proof: This can be proved directly from Definition 0

Theorem 6.6: a/(-b) = -a/b, a%(-b) = -a%b
Proof: See "Implementations of Div and Mod", this is a written in the paper

Theorem 6.7: If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
where compatible_sign(a, b) is defined as ab>=0
Proof: According to Theorem 6.3, this is just to prove
a + b = a % c + b % c \pmod c
Because of Theorem 6.4, we have a = a % c \pmod c, b = b % c \pmod c,
applying Theorem 3.3, we get what we want

Theorem 6.8: If compatible_sign(a, b), then (a * b) % c = (a % c * b % c) % c
where compatible_sign(a, b) is defined as ab>=0
Proof: Similar to above

Theorem 6.9: If a is a multiple of b, then a % b = 0
Proof:  This can be proved directly from Euclid's division lemma

Theorem 6.10: If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
Proof: Same proof as 2.10

Theorem 6.11: a/(b*c) = (a/b)/c
Proof: This is part of Theorem 6.1

Theorem 6.12: a % (b * c) = a % b + ((a / b) % c) * b
Proof: Already proved in the proof of Theorem 6.1

Theorem 6.13: If d divides a and b, then a % b = ((a / d) % (b / d)) * d
Proof: Same proof as 2.13

Theorem 6.14: If b is a multiple of c, then a/(b/c) = (a*c)/b
Proof: Same proof as 2.14

Theorem 6.15: If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
(a+b)/c = a/c + b/c
Proof: From Theorem 6.7 and Theorem 6.5
(a + b) % c = (a % c + b % c) % c = a % c + b % c
The rest of the proof is the same as 2.15
