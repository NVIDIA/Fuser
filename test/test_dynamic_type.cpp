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

namespace has_operator_tests {

struct HasOperatorTestType {};

// Unary operators
static_assert(+has_operator<int>);
static_assert(!(+has_operator<HasOperatorTestType>));

static_assert(-has_operator<int>);
static_assert(!(-has_operator<HasOperatorTestType>));

static_assert(~has_operator<int>);
static_assert(!(~has_operator<HasOperatorTestType>));

static_assert(!has_operator<int>);
static_assert(!(!has_operator<HasOperatorTestType>));

static_assert(++has_operator<int&>);
static_assert(!(++has_operator<HasOperatorTestType&>));

static_assert(--has_operator<int&>);
static_assert(!(--has_operator<HasOperatorTestType&>));

static_assert(has_operator<int&> ++);
static_assert(!(has_operator<HasOperatorTestType&> ++));

static_assert(has_operator<int&> --);
static_assert(!(has_operator<HasOperatorTestType&> --));

// Comma
static_assert(
    (has_operator<HasOperatorTestType>, has_operator<HasOperatorTestType>));
// TODO: how to test negative case for comma operator? I can not think of any
// case where comma operator is not valid.

// Binary operators

static_assert(has_operator<int> + has_operator<float>);
static_assert(!(has_operator<int> + has_operator<HasOperatorTestType>));

static_assert(has_operator<int> - has_operator<float>);
static_assert(!(has_operator<int> - has_operator<HasOperatorTestType>));

static_assert(has_operator<int> * has_operator<float>);
static_assert(!(has_operator<int> * has_operator<HasOperatorTestType>));

static_assert(has_operator<int> / has_operator<float>);
static_assert(!(has_operator<int> / has_operator<HasOperatorTestType>));

static_assert(has_operator<int> % has_operator<int>);
static_assert(!(has_operator<int> % has_operator<HasOperatorTestType>));

static_assert(has_operator<int> & has_operator<int>);
static_assert(!(has_operator<int> & has_operator<HasOperatorTestType>));

static_assert(has_operator<int> | has_operator<int>);
static_assert(!(has_operator<int> | has_operator<HasOperatorTestType>));

static_assert(has_operator<int> ^ has_operator<int>);
static_assert(!(has_operator<int> ^ has_operator<HasOperatorTestType>));

static_assert(has_operator<int> && has_operator<int>);
static_assert(!(has_operator<int> && has_operator<HasOperatorTestType>));

static_assert(has_operator<int> || has_operator<int>);
static_assert(!(has_operator<int> || has_operator<HasOperatorTestType>));

static_assert(has_operator<int> << has_operator<int>);
static_assert(!(has_operator<int> << has_operator<HasOperatorTestType>));

static_assert(has_operator<int> >> has_operator<int>);
static_assert(!(has_operator<int> >> has_operator<HasOperatorTestType>));

static_assert(has_operator<int> == has_operator<float>);
static_assert(!(has_operator<int> == has_operator<HasOperatorTestType>));

static_assert(has_operator<int> != has_operator<float>);
static_assert(!(has_operator<int> != has_operator<HasOperatorTestType>));

static_assert(has_operator<int> < has_operator<float>);
static_assert(!(has_operator<int> < has_operator<HasOperatorTestType>));

static_assert(has_operator<int> > has_operator<float>);
static_assert(!(has_operator<int> > has_operator<HasOperatorTestType>));

static_assert(has_operator<int> <= has_operator<float>);
static_assert(!(has_operator<int> <= has_operator<HasOperatorTestType>));

static_assert(has_operator<int> >= has_operator<float>);
static_assert(!(has_operator<int> >= has_operator<HasOperatorTestType>));

// Assignment operators
static_assert(has_operator<int&> = has_operator<int>);
static_assert(!(has_operator<int&> = has_operator<HasOperatorTestType>));

static_assert(has_operator<float&> += has_operator<int>);
static_assert(!(has_operator<int&> += has_operator<HasOperatorTestType>));

static_assert(has_operator<float&> -= has_operator<int>);
static_assert(!(has_operator<int&> -= has_operator<HasOperatorTestType>));

static_assert(has_operator<float&> *= has_operator<int>);
static_assert(!(has_operator<int&> *= has_operator<HasOperatorTestType>));

static_assert(has_operator<float&> /= has_operator<int>);
static_assert(!(has_operator<int&> /= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> %= has_operator<int>);
static_assert(!(has_operator<int&> %= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> &= has_operator<int>);
static_assert(!(has_operator<int&> &= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> |= has_operator<int>);
static_assert(!(has_operator<int&> |= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> ^= has_operator<int>);
static_assert(!(has_operator<int&> ^= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> <<= has_operator<int>);
static_assert(!(has_operator<int&> <<= has_operator<HasOperatorTestType>));

static_assert(has_operator<int&> >>= has_operator<int>);
static_assert(!(has_operator<int&> >>= has_operator<HasOperatorTestType>));

// Function call
int foo(int);
static_assert(has_operator<decltype(foo)>(has_operator<int>));
static_assert(!(has_operator<decltype(foo)>()));
static_assert(
    !(has_operator<decltype(foo)>(has_operator<HasOperatorTestType>)));
static_assert(!(has_operator<HasOperatorTestType>(has_operator<int>)));
int bar();
static_assert(has_operator<decltype(bar)>());
static_assert(!(has_operator<decltype(bar)>(has_operator<int>)));
static_assert(
    !(has_operator<decltype(bar)>(has_operator<HasOperatorTestType>)));
static_assert(!(has_operator<HasOperatorTestType>(has_operator<int>)));

// Array index
static_assert(has_operator<int[3]>[has_operator<int>]);
static_assert(!(has_operator<int[3]>[has_operator<HasOperatorTestType>]));
static_assert(!(has_operator<HasOperatorTestType>[has_operator<int>]));

// Arrow operator
static_assert(has_operator<std::unique_ptr<int>>->value());
static_assert(!has_operator<int>->value());

// Arrow star operator
struct OverloadArrowStar
{
    auto operator->*(int OverloadArrowStar::*memberPtr) const -> int *
    {
        return nullptr;
    }
};

static_assert(has_operator<OverloadArrowStar>->*2);
static_assert(has_operator<OverloadArrowStar>->*true);
static_assert(has_operator<OverloadArrowStar>->*has_operator<int>);
static_assert(!(has_operator<int>->*2));
static_assert(!(has_operator<int>->*true));
static_assert(!(has_operator<int>->*has_operator<OverloadArrowStar>));


} // namespace has_operator_tests

} // namespace nvfuser