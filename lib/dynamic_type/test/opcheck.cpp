// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "dynamic_type/type_traits.h"

#include <memory>

using namespace dynamic_type;

// Test opcheck

struct OpcheckSomeType {};
struct OpcheckSomeType2 {};

// Unary operators
static_assert(+opcheck<int>);
static_assert(!(+opcheck<OpcheckSomeType>));

static_assert(-opcheck<int>);
static_assert(!(-opcheck<OpcheckSomeType>));

static_assert(~opcheck<int>);
static_assert(!(~opcheck<OpcheckSomeType>));

static_assert(!opcheck<int>);
static_assert(!(!opcheck<OpcheckSomeType>));

static_assert(++opcheck<int&>);
static_assert(!(++opcheck<OpcheckSomeType&>));

static_assert(--opcheck<int&>);
static_assert(!(--opcheck<OpcheckSomeType&>));

static_assert(opcheck<int&> ++);
static_assert(!(opcheck<OpcheckSomeType&> ++));

static_assert(opcheck<int&> --);
static_assert(!(opcheck<OpcheckSomeType&> --));

// Comma
static_assert((opcheck<OpcheckSomeType>, opcheck<OpcheckSomeType>));
static_assert(
    std::is_same_v<decltype((opcheck<OpcheckSomeType>, opcheck<OpcheckSomeType>)), bool>);
// TODO: how to test negative case for comma operator? I can not think of any
// case where comma operator is not valid.

// Binary operators

static_assert(opcheck<int> + opcheck<float>);
static_assert(!(opcheck<int> + opcheck<OpcheckSomeType>));

static_assert(opcheck<int> - opcheck<float>);
static_assert(!(opcheck<int> - opcheck<OpcheckSomeType>));

static_assert(opcheck<int> * opcheck<float>);
static_assert(!(opcheck<int> * opcheck<OpcheckSomeType>));

static_assert(opcheck<int> / opcheck<float>);
static_assert(!(opcheck<int> / opcheck<OpcheckSomeType>));

static_assert(opcheck<int> % opcheck<int>);
static_assert(!(opcheck<int> % opcheck<OpcheckSomeType>));

static_assert(opcheck<int> & opcheck<int>);
static_assert(!(opcheck<int> & opcheck<OpcheckSomeType>));

static_assert(opcheck<int> | opcheck<int>);
static_assert(!(opcheck<int> | opcheck<OpcheckSomeType>));

static_assert(opcheck<int> ^ opcheck<int>);
static_assert(!(opcheck<int> ^ opcheck<OpcheckSomeType>));

static_assert(opcheck<int> && opcheck<int>);
static_assert(!(opcheck<int> && opcheck<OpcheckSomeType>));

static_assert(opcheck<int> || opcheck<int>);
static_assert(!(opcheck<int> || opcheck<OpcheckSomeType>));

static_assert(opcheck<int> << opcheck<int>);
static_assert(!(opcheck<int> << opcheck<OpcheckSomeType>));

static_assert(opcheck<int> >> opcheck<int>);
static_assert(!(opcheck<int> >> opcheck<OpcheckSomeType>));

static_assert(opcheck<int> == opcheck<float>);
static_assert(!(opcheck<int> == opcheck<OpcheckSomeType>));

static_assert(opcheck<int> != opcheck<float>);
static_assert(!(opcheck<int> != opcheck<OpcheckSomeType>));

static_assert(opcheck<int> < opcheck<float>);
static_assert(!(opcheck<int> < opcheck<OpcheckSomeType>));

static_assert(opcheck<int> > opcheck<float>);
static_assert(!(opcheck<int> > opcheck<OpcheckSomeType>));

static_assert(opcheck<int> <= opcheck<float>);
static_assert(!(opcheck<int> <= opcheck<OpcheckSomeType>));

static_assert(opcheck<int> >= opcheck<float>);
static_assert(!(opcheck<int> >= opcheck<OpcheckSomeType>));

#if !defined(__clang__)
// clang is incorrectly assuming that T& = must also return a T&, which is
// wrong. I never see such a restriction in the standard. In opcheck, all
// operators are by design intentionally returning a bool, regardless of the
// original semantics of that operator.

// Assignment operators
static_assert(opcheck<int&> = opcheck<int>);
static_assert(!(opcheck<int&> = opcheck<OpcheckSomeType>));

static_assert(opcheck<float&> += opcheck<int>);
static_assert(!(opcheck<int&> += opcheck<OpcheckSomeType>));

static_assert(opcheck<float&> -= opcheck<int>);
static_assert(!(opcheck<int&> -= opcheck<OpcheckSomeType>));

static_assert(opcheck<float&> *= opcheck<int>);
static_assert(!(opcheck<int&> *= opcheck<OpcheckSomeType>));

static_assert(opcheck<float&> /= opcheck<int>);
static_assert(!(opcheck<int&> /= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> %= opcheck<int>);
static_assert(!(opcheck<int&> %= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> &= opcheck<int>);
static_assert(!(opcheck<int&> &= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> |= opcheck<int>);
static_assert(!(opcheck<int&> |= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> ^= opcheck<int>);
static_assert(!(opcheck<int&> ^= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> <<= opcheck<int>);
static_assert(!(opcheck<int&> <<= opcheck<OpcheckSomeType>));

static_assert(opcheck<int&> >>= opcheck<int>);
static_assert(!(opcheck<int&> >>= opcheck<OpcheckSomeType>));
#endif

// Function call
int foo(int);
static_assert(opcheck<decltype(foo)>(opcheck<int>));
static_assert(!(opcheck<decltype(foo)>()));
static_assert(!(opcheck<decltype(foo)>(opcheck<OpcheckSomeType>)));
static_assert(!(opcheck<OpcheckSomeType>(opcheck<int>)));
int bar();
static_assert(opcheck<decltype(bar)>());
static_assert(!(opcheck<decltype(bar)>(opcheck<int>)));
static_assert(!(opcheck<decltype(bar)>(opcheck<OpcheckSomeType>)));
static_assert(!(opcheck<OpcheckSomeType>(opcheck<int>)));

// Array index
static_assert(opcheck<int[3]>[opcheck<int>]);
static_assert(!(opcheck<int[3]>[opcheck<OpcheckSomeType>]));
static_assert(!(opcheck<OpcheckSomeType>[opcheck<int>]));

// Arrow operator
static_assert(opcheck<std::unique_ptr<int>>->value());
static_assert(!opcheck<int>->value());

// Arrow star operator
struct OverloadArrowStar {
  auto operator->*(int OverloadArrowStar::* memberPtr) const -> int* {
    return nullptr;
  }
};

static_assert(!(opcheck<OverloadArrowStar>->*opcheck<int>));
static_assert(opcheck<OverloadArrowStar>->*opcheck<int OverloadArrowStar::*>);
static_assert(!(opcheck<int>->*opcheck<OverloadArrowStar>));

// Casting operators
static_assert(opcheck<float>.canCastTo(opcheck<float>));
static_assert(opcheck<float>.canCastTo(opcheck<int>));
static_assert(opcheck<int>.canCastTo(opcheck<float>));
static_assert(!opcheck<OpcheckSomeType>.canCastTo(opcheck<float>));
static_assert(!opcheck<float>.canCastTo(opcheck<OpcheckSomeType>));
static_assert(opcheck<OpcheckSomeType>.canCastTo(opcheck<OpcheckSomeType>));

static_assert(!opcheck<float>.hasExplicitCastTo(opcheck<double>));
struct OpcheckA {
  operator int() const {
    return 0;
  }
};
struct OpcheckB {
  OpcheckB(const OpcheckA&) {}
};
static_assert(opcheck<OpcheckA>.hasExplicitCastTo(opcheck<int>));
static_assert(!opcheck<OpcheckA>.hasExplicitCastTo(opcheck<OpcheckB>));
