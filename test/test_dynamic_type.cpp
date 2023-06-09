// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <type_traits.h>

#include <iostream>
#include <memory>

namespace nvfuser {

namespace opcheck_tests {

struct HasOperatorTestType {};

// Unary operators
static_assert(+opcheck<int>);
static_assert(!(+opcheck<HasOperatorTestType>));

static_assert(-opcheck<int>);
static_assert(!(-opcheck<HasOperatorTestType>));

static_assert(~opcheck<int>);
static_assert(!(~opcheck<HasOperatorTestType>));

static_assert(!opcheck<int>);
static_assert(!(!opcheck<HasOperatorTestType>));

static_assert(++opcheck<int&>);
static_assert(!(++opcheck<HasOperatorTestType&>));

static_assert(--opcheck<int&>);
static_assert(!(--opcheck<HasOperatorTestType&>));

static_assert(opcheck<int&> ++);
static_assert(!(opcheck<HasOperatorTestType&> ++));

static_assert(opcheck<int&> --);
static_assert(!(opcheck<HasOperatorTestType&> --));

// Comma
static_assert((opcheck<HasOperatorTestType>, opcheck<HasOperatorTestType>));
// TODO: how to test negative case for comma operator? I can not think of any
// case where comma operator is not valid.

// Binary operators

static_assert(opcheck<int> + opcheck<float>);
static_assert(!(opcheck<int> + opcheck<HasOperatorTestType>));

static_assert(opcheck<int> - opcheck<float>);
static_assert(!(opcheck<int> - opcheck<HasOperatorTestType>));

static_assert(opcheck<int> * opcheck<float>);
static_assert(!(opcheck<int> * opcheck<HasOperatorTestType>));

static_assert(opcheck<int> / opcheck<float>);
static_assert(!(opcheck<int> / opcheck<HasOperatorTestType>));

static_assert(opcheck<int> % opcheck<int>);
static_assert(!(opcheck<int> % opcheck<HasOperatorTestType>));

static_assert(opcheck<int> & opcheck<int>);
static_assert(!(opcheck<int> & opcheck<HasOperatorTestType>));

static_assert(opcheck<int> | opcheck<int>);
static_assert(!(opcheck<int> | opcheck<HasOperatorTestType>));

static_assert(opcheck<int> ^ opcheck<int>);
static_assert(!(opcheck<int> ^ opcheck<HasOperatorTestType>));

static_assert(opcheck<int> && opcheck<int>);
static_assert(!(opcheck<int> && opcheck<HasOperatorTestType>));

static_assert(opcheck<int> || opcheck<int>);
static_assert(!(opcheck<int> || opcheck<HasOperatorTestType>));

static_assert(opcheck<int> << opcheck<int>);
static_assert(!(opcheck<int> << opcheck<HasOperatorTestType>));

static_assert(opcheck<int> >> opcheck<int>);
static_assert(!(opcheck<int> >> opcheck<HasOperatorTestType>));

static_assert(opcheck<int> == opcheck<float>);
static_assert(!(opcheck<int> == opcheck<HasOperatorTestType>));

static_assert(opcheck<int> != opcheck<float>);
static_assert(!(opcheck<int> != opcheck<HasOperatorTestType>));

static_assert(opcheck<int> < opcheck<float>);
static_assert(!(opcheck<int> < opcheck<HasOperatorTestType>));

static_assert(opcheck<int> > opcheck<float>);
static_assert(!(opcheck<int> > opcheck<HasOperatorTestType>));

static_assert(opcheck<int> <= opcheck<float>);
static_assert(!(opcheck<int> <= opcheck<HasOperatorTestType>));

static_assert(opcheck<int> >= opcheck<float>);
static_assert(!(opcheck<int> >= opcheck<HasOperatorTestType>));

// Assignment operators
static_assert(opcheck<int&> = opcheck<int>);
static_assert(!(opcheck<int&> = opcheck<HasOperatorTestType>));

static_assert(opcheck<float&> += opcheck<int>);
static_assert(!(opcheck<int&> += opcheck<HasOperatorTestType>));

static_assert(opcheck<float&> -= opcheck<int>);
static_assert(!(opcheck<int&> -= opcheck<HasOperatorTestType>));

static_assert(opcheck<float&> *= opcheck<int>);
static_assert(!(opcheck<int&> *= opcheck<HasOperatorTestType>));

static_assert(opcheck<float&> /= opcheck<int>);
static_assert(!(opcheck<int&> /= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> %= opcheck<int>);
static_assert(!(opcheck<int&> %= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> &= opcheck<int>);
static_assert(!(opcheck<int&> &= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> |= opcheck<int>);
static_assert(!(opcheck<int&> |= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> ^= opcheck<int>);
static_assert(!(opcheck<int&> ^= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> <<= opcheck<int>);
static_assert(!(opcheck<int&> <<= opcheck<HasOperatorTestType>));

static_assert(opcheck<int&> >>= opcheck<int>);
static_assert(!(opcheck<int&> >>= opcheck<HasOperatorTestType>));

// Function call
int foo(int);
static_assert(opcheck<decltype(foo)>(opcheck<int>));
static_assert(!(opcheck<decltype(foo)>()));
static_assert(!(opcheck<decltype(foo)>(opcheck<HasOperatorTestType>)));
static_assert(!(opcheck<HasOperatorTestType>(opcheck<int>)));
int bar();
static_assert(opcheck<decltype(bar)>());
static_assert(!(opcheck<decltype(bar)>(opcheck<int>)));
static_assert(!(opcheck<decltype(bar)>(opcheck<HasOperatorTestType>)));
static_assert(!(opcheck<HasOperatorTestType>(opcheck<int>)));

// Array index
static_assert(opcheck<int[3]>[opcheck<int>]);
static_assert(!(opcheck<int[3]>[opcheck<HasOperatorTestType>]));
static_assert(!(opcheck<HasOperatorTestType>[opcheck<int>]));

// Arrow operator
static_assert(opcheck<std::unique_ptr<int>>->value());
static_assert(!opcheck<int>->value());

// Arrow star operator
struct OverloadArrowStar {
  auto operator->*(int OverloadArrowStar::*memberPtr) const -> int* {
    return nullptr;
  }
};

static_assert(opcheck<OverloadArrowStar>->*2);
static_assert(opcheck<OverloadArrowStar>->*true);
static_assert(opcheck<OverloadArrowStar>->*opcheck<int>);
static_assert(!(opcheck<int>->*2));
static_assert(!(opcheck<int>->*true));
static_assert(!(opcheck<int>->*opcheck<OverloadArrowStar>));

} // namespace opcheck_tests

} // namespace nvfuser