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

struct OperatorCheckerTestType {};

// Unary operators
static_assert(+opcheck<int>);
static_assert(!(+opcheck<OperatorCheckerTestType>));

static_assert(-opcheck<int>);
static_assert(!(-opcheck<OperatorCheckerTestType>));

static_assert(~opcheck<int>);
static_assert(!(~opcheck<OperatorCheckerTestType>));

static_assert(!opcheck<int>);
static_assert(!(!opcheck<OperatorCheckerTestType>));

static_assert(++opcheck<int&>);
static_assert(!(++opcheck<OperatorCheckerTestType&>));

static_assert(--opcheck<int&>);
static_assert(!(--opcheck<OperatorCheckerTestType&>));

static_assert(opcheck<int&> ++);
static_assert(!(opcheck<OperatorCheckerTestType&> ++));

static_assert(opcheck<int&> --);
static_assert(!(opcheck<OperatorCheckerTestType&> --));

// Comma
static_assert(
    (opcheck<OperatorCheckerTestType>, opcheck<OperatorCheckerTestType>));
// TODO: how to test negative case for comma operator? I can not think of any
// case where comma operator is not valid.

// Binary operators

static_assert(opcheck<int> + opcheck<float>);
static_assert(!(opcheck<int> + opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> - opcheck<float>);
static_assert(!(opcheck<int> - opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> * opcheck<float>);
static_assert(!(opcheck<int> * opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> / opcheck<float>);
static_assert(!(opcheck<int> / opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> % opcheck<int>);
static_assert(!(opcheck<int> % opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> & opcheck<int>);
static_assert(!(opcheck<int> & opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> | opcheck<int>);
static_assert(!(opcheck<int> | opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> ^ opcheck<int>);
static_assert(!(opcheck<int> ^ opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> && opcheck<int>);
static_assert(!(opcheck<int> && opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> || opcheck<int>);
static_assert(!(opcheck<int> || opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> << opcheck<int>);
static_assert(!(opcheck<int> << opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> >> opcheck<int>);
static_assert(!(opcheck<int> >> opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> == opcheck<float>);
static_assert(!(opcheck<int> == opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> != opcheck<float>);
static_assert(!(opcheck<int> != opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> < opcheck<float>);
static_assert(!(opcheck<int> < opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> > opcheck<float>);
static_assert(!(opcheck<int> > opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> <= opcheck<float>);
static_assert(!(opcheck<int> <= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int> >= opcheck<float>);
static_assert(!(opcheck<int> >= opcheck<OperatorCheckerTestType>));

// Assignment operators
static_assert(opcheck<int&> = opcheck<int>);
static_assert(!(opcheck<int&> = opcheck<OperatorCheckerTestType>));

static_assert(opcheck<float&> += opcheck<int>);
static_assert(!(opcheck<int&> += opcheck<OperatorCheckerTestType>));

static_assert(opcheck<float&> -= opcheck<int>);
static_assert(!(opcheck<int&> -= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<float&> *= opcheck<int>);
static_assert(!(opcheck<int&> *= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<float&> /= opcheck<int>);
static_assert(!(opcheck<int&> /= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> %= opcheck<int>);
static_assert(!(opcheck<int&> %= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> &= opcheck<int>);
static_assert(!(opcheck<int&> &= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> |= opcheck<int>);
static_assert(!(opcheck<int&> |= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> ^= opcheck<int>);
static_assert(!(opcheck<int&> ^= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> <<= opcheck<int>);
static_assert(!(opcheck<int&> <<= opcheck<OperatorCheckerTestType>));

static_assert(opcheck<int&> >>= opcheck<int>);
static_assert(!(opcheck<int&> >>= opcheck<OperatorCheckerTestType>));

// Function call
int foo(int);
static_assert(opcheck<decltype(foo)>(opcheck<int>));
static_assert(!(opcheck<decltype(foo)>()));
static_assert(!(opcheck<decltype(foo)>(opcheck<OperatorCheckerTestType>)));
static_assert(!(opcheck<OperatorCheckerTestType>(opcheck<int>)));
int bar();
static_assert(opcheck<decltype(bar)>());
static_assert(!(opcheck<decltype(bar)>(opcheck<int>)));
static_assert(!(opcheck<decltype(bar)>(opcheck<OperatorCheckerTestType>)));
static_assert(!(opcheck<OperatorCheckerTestType>(opcheck<int>)));

// Array index
static_assert(opcheck<int[3]>[opcheck<int>]);
static_assert(!(opcheck<int[3]>[opcheck<OperatorCheckerTestType>]));
static_assert(!(opcheck<OperatorCheckerTestType>[opcheck<int>]));

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

// Casting operators
static_assert(opcheck<float>.canCastTo(opcheck<float>));
static_assert(opcheck<float>.canCastTo(opcheck<int>));
static_assert(opcheck<int>.canCastTo(opcheck<float>));
static_assert(!opcheck<OperatorCheckerTestType>.canCastTo(opcheck<float>));
static_assert(!opcheck<float>.canCastTo(opcheck<OperatorCheckerTestType>));
static_assert(opcheck<OperatorCheckerTestType>.canCastTo(
    opcheck<OperatorCheckerTestType>));

} // namespace opcheck_tests

namespace ForAllTypes_tests {

// Find all primes < 10 using template deduction

using From2To10 = ForAllTypes<
    std::integral_constant<int, 2>,
    std::integral_constant<int, 3>,
    std::integral_constant<int, 4>,
    std::integral_constant<int, 5>,
    std::integral_constant<int, 6>,
    std::integral_constant<int, 7>,
    std::integral_constant<int, 8>,
    std::integral_constant<int, 9>,
    std::integral_constant<int, 10>>;

constexpr bool is_prime(int n) {
  return all(From2To10{}([n](auto* _) {
    auto divisor = std::remove_pointer_t<decltype(_)>::value;
    return n % divisor != 0 || n == divisor;
  }));
}

auto void_or_prime = [](auto* _) constexpr {
  constexpr auto value = std::remove_pointer_t<decltype(_)>::value;
  if constexpr (is_prime(value)) {
    return std::integral_constant<int, value>{};
  } else {
    return;
  }
};

// (2, 3, Void, 5, Void, 7, Void, Void, Void)
using result_with_void = decltype(From2To10{}(void_or_prime));

using result = decltype(remove_void_from_tuple(result_with_void{}));

static_assert(std::is_same_v<
              result,
              std::tuple<
                  std::integral_constant<int, 2>,
                  std::integral_constant<int, 3>,
                  std::integral_constant<int, 5>,
                  std::integral_constant<int, 7>>>);

} // namespace ForAllTypes_tests

namespace util_tests {

static_assert(all(true, true, true));
static_assert(all(std::make_tuple(true, true, true)));
static_assert(!all(true, false, true));
static_assert(!all(std::make_tuple(true, false, true)));

static_assert(
    remove_void_from_tuple(
        std::make_tuple(Void{}, 1, Void{}, 2, Void{}, 3, Void{})) ==
    std::make_tuple(1, 2, 3));

} // namespace util_tests

} // namespace nvfuser