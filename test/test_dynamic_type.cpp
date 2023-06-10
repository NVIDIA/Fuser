// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>
#include <type_traits.h>

#include <dynamic_type.h>

#include <iostream>
#include <memory>

namespace nvfuser {

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"

struct SomeType {};
struct SomeType2 {};

namespace OpCheckTests {

// Unary operators
static_assert(+opcheck<int>);
static_assert(!(+opcheck<SomeType>));

static_assert(-opcheck<int>);
static_assert(!(-opcheck<SomeType>));

static_assert(~opcheck<int>);
static_assert(!(~opcheck<SomeType>));

static_assert(!opcheck<int>);
static_assert(!(!opcheck<SomeType>));

static_assert(++opcheck<int&>);
static_assert(!(++opcheck<SomeType&>));

static_assert(--opcheck<int&>);
static_assert(!(--opcheck<SomeType&>));

static_assert(opcheck<int&> ++);
static_assert(!(opcheck<SomeType&> ++));

static_assert(opcheck<int&> --);
static_assert(!(opcheck<SomeType&> --));

// Comma
static_assert((opcheck<SomeType>, opcheck<SomeType>));
// TODO: how to test negative case for comma operator? I can not think of any
// case where comma operator is not valid.

// Binary operators

static_assert(opcheck<int> + opcheck<float>);
static_assert(!(opcheck<int> + opcheck<SomeType>));

static_assert(opcheck<int> - opcheck<float>);
static_assert(!(opcheck<int> - opcheck<SomeType>));

static_assert(opcheck<int> * opcheck<float>);
static_assert(!(opcheck<int> * opcheck<SomeType>));

static_assert(opcheck<int> / opcheck<float>);
static_assert(!(opcheck<int> / opcheck<SomeType>));

static_assert(opcheck<int> % opcheck<int>);
static_assert(!(opcheck<int> % opcheck<SomeType>));

static_assert(opcheck<int> & opcheck<int>);
static_assert(!(opcheck<int> & opcheck<SomeType>));

static_assert(opcheck<int> | opcheck<int>);
static_assert(!(opcheck<int> | opcheck<SomeType>));

static_assert(opcheck<int> ^ opcheck<int>);
static_assert(!(opcheck<int> ^ opcheck<SomeType>));

static_assert(opcheck<int> && opcheck<int>);
static_assert(!(opcheck<int> && opcheck<SomeType>));

static_assert(opcheck<int> || opcheck<int>);
static_assert(!(opcheck<int> || opcheck<SomeType>));

static_assert(opcheck<int> << opcheck<int>);
static_assert(!(opcheck<int> << opcheck<SomeType>));

static_assert(opcheck<int> >> opcheck<int>);
static_assert(!(opcheck<int> >> opcheck<SomeType>));

static_assert(opcheck<int> == opcheck<float>);
static_assert(!(opcheck<int> == opcheck<SomeType>));

static_assert(opcheck<int> != opcheck<float>);
static_assert(!(opcheck<int> != opcheck<SomeType>));

static_assert(opcheck<int> < opcheck<float>);
static_assert(!(opcheck<int> < opcheck<SomeType>));

static_assert(opcheck<int> > opcheck<float>);
static_assert(!(opcheck<int> > opcheck<SomeType>));

static_assert(opcheck<int> <= opcheck<float>);
static_assert(!(opcheck<int> <= opcheck<SomeType>));

static_assert(opcheck<int> >= opcheck<float>);
static_assert(!(opcheck<int> >= opcheck<SomeType>));

// Assignment operators
static_assert(opcheck<int&> = opcheck<int>);
#if 0
static_assert(!(opcheck<int&> = opcheck<SomeType>));

static_assert(opcheck<float&> += opcheck<int>);
static_assert(!(opcheck<int&> += opcheck<SomeType>));

static_assert(opcheck<float&> -= opcheck<int>);
static_assert(!(opcheck<int&> -= opcheck<SomeType>));

static_assert(opcheck<float&> *= opcheck<int>);
static_assert(!(opcheck<int&> *= opcheck<SomeType>));

static_assert(opcheck<float&> /= opcheck<int>);
static_assert(!(opcheck<int&> /= opcheck<SomeType>));

static_assert(opcheck<int&> %= opcheck<int>);
static_assert(!(opcheck<int&> %= opcheck<SomeType>));

static_assert(opcheck<int&> &= opcheck<int>);
static_assert(!(opcheck<int&> &= opcheck<SomeType>));

static_assert(opcheck<int&> |= opcheck<int>);
static_assert(!(opcheck<int&> |= opcheck<SomeType>));

static_assert(opcheck<int&> ^= opcheck<int>);
static_assert(!(opcheck<int&> ^= opcheck<SomeType>));

static_assert(opcheck<int&> <<= opcheck<int>);
static_assert(!(opcheck<int&> <<= opcheck<SomeType>));

static_assert(opcheck<int&> >>= opcheck<int>);
static_assert(!(opcheck<int&> >>= opcheck<SomeType>));

// Function call
int foo(int);
static_assert(opcheck<decltype(foo)>(opcheck<int>));
static_assert(!(opcheck<decltype(foo)>()));
static_assert(!(opcheck<decltype(foo)>(opcheck<SomeType>)));
static_assert(!(opcheck<SomeType>(opcheck<int>)));
int bar();
static_assert(opcheck<decltype(bar)>());
static_assert(!(opcheck<decltype(bar)>(opcheck<int>)));
static_assert(!(opcheck<decltype(bar)>(opcheck<SomeType>)));
static_assert(!(opcheck<SomeType>(opcheck<int>)));

// Array index
static_assert(opcheck<int[3]>[opcheck<int>]);
static_assert(!(opcheck<int[3]>[opcheck<SomeType>]));
static_assert(!(opcheck<SomeType>[opcheck<int>]));

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
static_assert(!opcheck<SomeType>.canCastTo(opcheck<float>));
static_assert(!opcheck<float>.canCastTo(opcheck<SomeType>));
static_assert(opcheck<SomeType>.canCastTo(opcheck<SomeType>));
#endif
} // namespace OpCheckTests

namespace ForAllTypesTests {

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

} // namespace ForAllTypesTests

class DynamicTypeTest : public NVFuserTest {};

using DoubleInt64Bool = DynamicType<double, int64_t, bool>;
using BoolSomeType = DynamicType<bool, SomeType>;
using IntSomeType = DynamicType<int, SomeType>;
using SomeTypes = DynamicType<SomeType, SomeType>;

TEST_F(DynamicTypeTest, Casting) {
  static_assert(IntSomeType(2).cast<double>() == 2.0);
  EXPECT_THAT(
      [&]() { IntSomeType(2).cast<SomeType>(); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Cannot cast to ")));
}

#define TEST_BINARY_OP_ALLTYPE(name, op)                                 \
  TEST_F(DynamicTypeTest, name) {                                        \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>); \
    static_assert(                                                       \
        (DoubleInt64Bool(2) op DoubleInt64Bool(2.5))                     \
            .as<decltype(2 op 2.5)>() == (2 op 2.5));                    \
    EXPECT_THAT(                                                         \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2); },              \
        ::testing::ThrowsMessage<c10::Error>(                            \
            ::testing::HasSubstr("Can not compute ")));                  \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);          \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));           \
    EXPECT_THAT(                                                         \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },    \
        ::testing::ThrowsMessage<c10::Error>(                            \
            ::testing::HasSubstr("Can not compute ")));                  \
  }

TEST_BINARY_OP_ALLTYPE(Add, +);
TEST_BINARY_OP_ALLTYPE(Minus, -);
TEST_BINARY_OP_ALLTYPE(Mul, *);
TEST_BINARY_OP_ALLTYPE(Div, /);
TEST_BINARY_OP_ALLTYPE(LogicalAnd, &&);
TEST_BINARY_OP_ALLTYPE(LogicalOr, ||);
TEST_BINARY_OP_ALLTYPE(Eq, ==);
TEST_BINARY_OP_ALLTYPE(Ne, !=);
TEST_BINARY_OP_ALLTYPE(Lt, <);
TEST_BINARY_OP_ALLTYPE(Gt, >);
TEST_BINARY_OP_ALLTYPE(Le, <=);
TEST_BINARY_OP_ALLTYPE(Ge, >=);

#define TEST_BINARY_OP_INT_ONLY(name, op)                                      \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(                                                             \
        (DoubleInt64Bool(3) op DoubleInt64Bool(2)).as<int64_t>() == (3 op 2)); \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2); },                    \
        ::testing::ThrowsMessage<c10::Error>(                                  \
            ::testing::HasSubstr("Can not compute ")));                        \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);                \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));                 \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },          \
        ::testing::ThrowsMessage<c10::Error>(                                  \
            ::testing::HasSubstr("Can not compute ")));                        \
  }

TEST_BINARY_OP_INT_ONLY(Mod, %);
TEST_BINARY_OP_INT_ONLY(BinaryAnd, &);
TEST_BINARY_OP_INT_ONLY(BinaryOr, |);
TEST_BINARY_OP_INT_ONLY(Xor, ^);
TEST_BINARY_OP_INT_ONLY(LShift, <<);
TEST_BINARY_OP_INT_ONLY(RShift, >>);

/*TODO: we should inline the definition of opname##_helper into enable_if,*/ /*but I can only do this in C++20 */

// template <typename DT>
// constexpr bool support_opname =
//     any_check([](auto x) { return +opcheck<decltype(x)>; }, typename DT::TypesAsTuple{}, typename DT::TypesAsTuple{});
// template <typename DT, typename = std::enable_if_t<support_opname<DT>>>
// inline constexpr DT operator+(DT x) {
//   DT ret(std::monostate{});
//   DT::for_all_types([&ret, x](auto* _) {
//     using Type = std::remove_pointer_t<decltype(_)>;
//     if constexpr (+opcheck<Type>) {
//       if (x.template is<Type>()) {
//         ret = DT(+x.template as<Type>());
//       }
//     }
//   });
//   TORCH_CHECK(
//       !ret.template is<std::monostate>(),
//       "Can not compute ",
//       "+",
//       " : incompatible type");
//   return ret;
// }

#define TEST_UNARY_OP(name, op)                                            \
  TEST_F(DynamicTypeTest, name) {                                          \
    static_assert(op opcheck<DoubleInt64Bool>);                            \
    static_assert((op DoubleInt64Bool(2)).as<decltype(op 2)>() == (op 2)); \
    EXPECT_THAT(                                                           \
        [&]() { op DoubleInt64Bool(); },                                   \
        ::testing::ThrowsMessage<c10::Error>(                              \
            ::testing::HasSubstr("Can not compute ")));                    \
    static_assert(op opcheck<IntSomeType>);                                \
    static_assert(!(op opcheck<SomeTypes>));                               \
    EXPECT_THAT(                                                           \
        [&]() { op IntSomeType(SomeType{}); },                             \
        ::testing::ThrowsMessage<c10::Error>(                              \
            ::testing::HasSubstr("Can not compute ")));                    \
  }

// TEST_UNARY_OP(Positive, +);
// TEST_UNARY_OP(Negative, -);
// TEST_UNARY_OP(BinaryNot, ~);
// TEST_UNARY_OP(LogicalNot, !);

#undef TEST_UNARY_OP

#pragma clang diagnostic pop

} // namespace nvfuser
