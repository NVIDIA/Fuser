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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif

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

#if !defined(__clang__)
// clang is incorrectly assuming that T& = must also return a T&, which is
// wrong. I never see such a restriction in the standard. In opcheck, all
// operators are by design intentionally returning a bool, regardless of the
// original semantics of that operator.

// Assignment operators
static_assert(opcheck<int&> = opcheck<int>);
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
#endif

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
using IntSomeType = DynamicType<int, SomeType>;
using BoolSomeType = DynamicType<bool, SomeType>;
using SomeTypes = DynamicType<SomeType, SomeType>;

TEST_F(DynamicTypeTest, Casting) {
  static_assert(IntSomeType(2).cast<double>() == 2.0);
  EXPECT_THAT(
      [&]() { IntSomeType(2).cast<SomeType>(); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Cannot cast to ")));
}

#define TEST_BINARY_OP_ALLTYPE(name, op)                                       \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int>);                   \
    static_assert(opcheck<int> op opcheck<DoubleInt64Bool>);                   \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.5))                          \
            .as<decltype(2L op 2.5)>() == (2L op 2.5));                        \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op 2L).as<decltype((3L op 2L))>() == (3L op 2L)); \
    static_assert(                                                             \
        (3L op DoubleInt64Bool(2L)).as<decltype((3L op 2L))>() == (3L op 2L)); \
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

#define TEST_BINARY_OP_INT_ONLY(name, op)                                   \
  TEST_F(DynamicTypeTest, name) {                                           \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);    \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int64_t>);            \
    static_assert(opcheck<int64_t> op opcheck<DoubleInt64Bool>);            \
    static_assert(                                                          \
        (DoubleInt64Bool(3L) op DoubleInt64Bool(2L)).as<int64_t>() ==       \
        (3L op 2L));                                                        \
    static_assert((DoubleInt64Bool(3L) op 2L).as<int64_t>() == (3L op 2L)); \
    static_assert((3L op DoubleInt64Bool(2L)).as<int64_t>() == (3L op 2L)); \
    EXPECT_THAT(                                                            \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2); },                 \
        ::testing::ThrowsMessage<c10::Error>(                               \
            ::testing::HasSubstr("Can not compute ")));                     \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);             \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));              \
    EXPECT_THAT(                                                            \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },       \
        ::testing::ThrowsMessage<c10::Error>(                               \
            ::testing::HasSubstr("Can not compute ")));                     \
  }

TEST_BINARY_OP_INT_ONLY(Mod, %);
TEST_BINARY_OP_INT_ONLY(BinaryAnd, &);
TEST_BINARY_OP_INT_ONLY(BinaryOr, |);
TEST_BINARY_OP_INT_ONLY(Xor, ^);
TEST_BINARY_OP_INT_ONLY(LShift, <<);
TEST_BINARY_OP_INT_ONLY(RShift, >>);

#define TEST_UNARY_OP(name, op, int_or_bool)                                  \
  TEST_F(DynamicTypeTest, name) {                                             \
    static_assert(op opcheck<DoubleInt64Bool>);                               \
    static_assert((op DoubleInt64Bool(2L)).as<decltype(op 2L)>() == (op 2L)); \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64Bool(); },                                      \
        ::testing::ThrowsMessage<c10::Error>(                                 \
            ::testing::HasSubstr("Can not compute ")));                       \
    static_assert(op opcheck<int_or_bool##SomeType>);                         \
    static_assert(!(op opcheck<SomeTypes>));                                  \
    EXPECT_THAT(                                                              \
        [&]() { op int_or_bool##SomeType(SomeType{}); },                      \
        ::testing::ThrowsMessage<c10::Error>(                                 \
            ::testing::HasSubstr("Can not compute ")));                       \
  }

TEST_UNARY_OP(Positive, +, Int);
TEST_UNARY_OP(Negative, -, Int);
TEST_UNARY_OP(BinaryNot, ~, Int);
TEST_UNARY_OP(LogicalNot, !, Bool);

#undef TEST_UNARY_OP

// This is the test for the examples in the note [Design of DynamicType], if you
// updated that note, please update this test as well. On the other hand, if you
// have to do something that breaks this test, please update the note as well.

struct bfloat16_zero {};
struct half_zero {};
float operator+(bfloat16_zero, half_zero) {
  return 0.0f;
}

TEST_F(DynamicTypeTest, ExamplesInNote) {
  // example 1
  using IntOrFloat = DynamicType<int, float>;
  {
    constexpr IntOrFloat x = 1;
    constexpr IntOrFloat y = 2.5f;
    constexpr IntOrFloat z = x + y;
    static_assert(z.as<float>() == 3.5f);
  }
  // example 2
  struct CustomType {};
  {
    using IntOrFloatOrCustom = DynamicType<int, float, CustomType>;
    constexpr IntOrFloatOrCustom i = 1;
    constexpr IntOrFloatOrCustom f = 2.5f;
    constexpr IntOrFloatOrCustom c = CustomType{};
    constexpr IntOrFloatOrCustom null;
    static_assert((i + i).as<int>() == 2);
    static_assert((i + f).as<float>() == 3.5f);
    static_assert((f + i).as<float>() == 3.5f);
    static_assert((f + f).as<float>() == 5.0f);
    EXPECT_THAT(
        [&]() { i + null; },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    EXPECT_THAT(
        [&]() { i + c; },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
  }
  // example 3
  {
    struct CustomType2 {};
    using Custom12 = DynamicType<CustomType, CustomType2>;
    static_assert(!(opcheck<Custom12> + opcheck<Custom12>));
  }
  // example 4
  {
    using BFloatOrHalfZero = DynamicType<bfloat16_zero, half_zero>;
    static_assert(!(opcheck<BFloatOrHalfZero> + opcheck<BFloatOrHalfZero>));
    using BFloatOrHalfZeroOrInt = DynamicType<bfloat16_zero, half_zero, int>;
    static_assert(
        opcheck<BFloatOrHalfZeroOrInt> + opcheck<BFloatOrHalfZeroOrInt>);
    EXPECT_THAT(
        [&]() {
          BFloatOrHalfZeroOrInt(half_zero{}) +
              BFloatOrHalfZeroOrInt(bfloat16_zero{});
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
  }
  // example 5
  {
    constexpr IntOrFloat x = 1;
    constexpr float y = 2.5f;
    static_assert(std::is_same_v<decltype(x + y), IntOrFloat>);
    static_assert((x + y).as<float>() == 3.5f);
    static_assert(std::is_same_v<decltype(y + x), IntOrFloat>);
    static_assert((y + x).as<float>() == 3.5f);
    static_assert(!(opcheck<IntOrFloat> + opcheck<double>));
    static_assert(!(opcheck<double> + opcheck<IntOrFloat>));
  }
}

TEST_F(DynamicTypeTest, UnaryOpAdvancedTyping) {
  struct Type1 {};
  struct Type2 {
    Type1 operator+() const {
      return Type1{};
    }
  };
  // not defined compile time because +Type2 is not in type list
  static_assert(!(+opcheck<DynamicType<Type2, SomeType>>));
  // defined compile time because +int is in type list
  static_assert(+opcheck<DynamicType<Type2, int>>);
  // runtime error because +Type2 is not in type list
  auto bad = [&]() { +DynamicType<Type2, int>(Type2{}); };
  EXPECT_THAT(
      bad,
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Can not compute ")));
}

TEST_F(DynamicTypeTest, BinaryOpAdvancedTyping) {
  struct Type1 {};
  struct Type2 {
    Type1 operator+(Type2) const {
      return Type1{};
    }
  };
  // not defined compile time because Type2+Type2 is not in type list
  static_assert(
      !(opcheck<DynamicType<Type2, SomeType>> +
        opcheck<DynamicType<Type2, SomeType>>));
  static_assert(!(opcheck<DynamicType<Type2, SomeType>> + opcheck<Type2>));
  static_assert(!(opcheck<Type2> + opcheck<DynamicType<Type2, SomeType>>));
  // defined compile time because int+int is in type list
  static_assert(
      opcheck<DynamicType<Type2, int>> + opcheck<DynamicType<Type2, int>>);
  // runtime error because Type2+Type2 is not in type list
  auto bad = [&]() {
    DynamicType<Type2, int> x(Type2{});
    x + x;
  };
  EXPECT_THAT(
      bad,
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Can not compute ")));
}

TEST_F(DynamicTypeTest, Printing) {
  std::stringstream ss;
  ss << DoubleInt64Bool(299792458L) << ", " << DoubleInt64Bool(3.14159) << ", "
     << DoubleInt64Bool(true);
  EXPECT_EQ(ss.str(), "299792458, 3.14159, 1");

  std::stringstream ss2;
  static_assert(opcheck<std::stringstream&> << opcheck<IntSomeType>);
  ss2 << IntSomeType(299792458);
  EXPECT_EQ(ss2.str(), "299792458");

  EXPECT_THAT(
      [&]() { ss << IntSomeType(); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Can not print")));
  EXPECT_THAT(
      [&]() { ss << IntSomeType(SomeType{}); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Can not print")));
  static_assert(!(opcheck<std::stringstream&> << opcheck<SomeTypes>));
}

TEST_F(DynamicTypeTest, PlusPlusMinusMinus) {
  // ++x
  {
    IntSomeType x(1);
    auto& y = ++x;
    EXPECT_EQ(x.as<int>(), 2);
    EXPECT_EQ(y.as<int>(), 2);
    EXPECT_EQ(&x, &y);
    EXPECT_THAT(
        []() {
          IntSomeType x;
          ++x;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          ++x;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    static_assert(!(++opcheck<SomeTypes&>));
  }
  // --x
  {
    IntSomeType x(1);
    auto& y = --x;
    EXPECT_EQ(x.as<int>(), 0);
    EXPECT_EQ(y.as<int>(), 0);
    EXPECT_EQ(&x, &y);
    EXPECT_THAT(
        []() {
          IntSomeType x;
          --x;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          --x;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    static_assert(!(--opcheck<SomeTypes&>));
  }
  // x++
  {
    IntSomeType x(1);
    auto y = x++;
    EXPECT_EQ(x.as<int>(), 2);
    EXPECT_EQ(y.as<int>(), 1);
    EXPECT_THAT(
        []() {
          IntSomeType x;
          x++;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          x++;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    static_assert(!(opcheck<SomeTypes&> ++));
  }
  // x--
  {
    IntSomeType x(1);
    auto y = x--;
    EXPECT_EQ(x.as<int>(), 0);
    EXPECT_EQ(y.as<int>(), 1);
    EXPECT_THAT(
        []() {
          IntSomeType x;
          x--;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          x--;
        },
        ::testing::ThrowsMessage<c10::Error>(
            ::testing::HasSubstr("Can not compute ")));
    static_assert(!(opcheck<SomeTypes&> --));
  }
}

#define TEST_ASSIGN_OP(op, assign_op, name)                \
  TEST_F(DynamicTypeTest, name) {                          \
    IntSomeType x(299792458);                              \
    auto& y = (x += 2);                                    \
    EXPECT_EQ(x.as<int>(), 299792458 + 2);                 \
    EXPECT_EQ(y.as<int>(), 299792458 + 2);                 \
    EXPECT_EQ(&x, &y);                                     \
    EXPECT_THAT(                                           \
        []() {                                             \
          IntSomeType x;                                   \
          x += 1;                                          \
        },                                                 \
        ::testing::ThrowsMessage<c10::Error>(              \
            ::testing::HasSubstr("Can not compute ")));    \
    EXPECT_THAT(                                           \
        []() {                                             \
          IntSomeType x(SomeType{});                       \
          x += 1;                                          \
        },                                                 \
        ::testing::ThrowsMessage<c10::Error>(              \
            ::testing::HasSubstr("Can not compute ")));    \
    static_assert(!(opcheck<SomeTypes&> += opcheck<int>)); \
  }

TEST_ASSIGN_OP(+, +=, AddAssign)
TEST_ASSIGN_OP(-, -=, MinusAssign);
TEST_ASSIGN_OP(*, *=, MulAssign);
TEST_ASSIGN_OP(/, /=, DivAssign);
TEST_ASSIGN_OP(%, %=, ModAssign);
TEST_ASSIGN_OP(&, &=, AndAssign);
TEST_ASSIGN_OP(|, |=, OrAssign);
TEST_ASSIGN_OP(^, ^=, XorAssign);
TEST_ASSIGN_OP(<<, <<=, LShiftAssign);
TEST_ASSIGN_OP(>>, >>=, RShiftAssign);

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace nvfuser
