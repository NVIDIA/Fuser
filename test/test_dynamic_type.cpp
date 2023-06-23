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
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

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
  return all(From2To10{}([n](auto _) {
    auto divisor = decltype(_)::type::value;
    return n % divisor != 0 || n == divisor;
  }));
}

auto void_or_prime = [](auto _) constexpr {
  constexpr auto value = decltype(_)::type::value;
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

struct NonInstantiable {
  NonInstantiable() = delete;
};

// Adding NonInstantiable as a member type to test that we never instantiate any
// member types when not necessary.
using DoubleInt64Bool =
    DynamicType<NoContainers, double, int64_t, bool, NonInstantiable>;
// Note: because std::vector does not has trivial destructor, we can not
// static_assert to test the following class:
using DoubleInt64BoolVec = DynamicType<
    Containers<std::vector>,
    double,
    int64_t,
    bool,
    NonInstantiable>;
using IntSomeType = DynamicType<NoContainers, int, SomeType, NonInstantiable>;
using BoolSomeType = DynamicType<NoContainers, bool, SomeType, NonInstantiable>;
using SomeTypes =
    DynamicType<NoContainers, SomeType, SomeType, NonInstantiable>;

// Utilities for testing if we have T->as<U> defined
template <typename T, typename U>
static auto hasAsHelper(int)
    -> decltype(std::declval<T>().template as<U>(), std::true_type{});

template <typename, typename>
static auto hasAsHelper(long) -> std::false_type;

template <typename T, typename U>
struct hasAs : decltype(hasAsHelper<T, U>(int{})) {};

TEST_F(DynamicTypeTest, Typing) {
  static_assert(DoubleInt64Bool().isNull());
  static_assert(!DoubleInt64Bool(1.0).isNull());
  static_assert(!DoubleInt64Bool().hasValue());
  static_assert(DoubleInt64Bool(1.0).hasValue());
  EXPECT_TRUE(DoubleInt64BoolVec().isNull());
  EXPECT_FALSE(DoubleInt64BoolVec(1.0).isNull());
  EXPECT_FALSE(DoubleInt64BoolVec().hasValue());
  EXPECT_TRUE(DoubleInt64BoolVec(1.0).hasValue());

  static_assert(hasAs<DoubleInt64BoolVec, double>::value);
  static_assert(hasAs<DoubleInt64BoolVec, int64_t>::value);
  static_assert(hasAs<DoubleInt64BoolVec, bool>::value);
  static_assert(
      hasAs<DoubleInt64BoolVec, std::vector<DoubleInt64BoolVec>>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, SomeType>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, int>::value);

  static_assert((int)DoubleInt64Bool(true) == 1);
  EXPECT_EQ((int)DoubleInt64BoolVec(true), 1);

  EXPECT_ANY_THROW(DoubleInt64Bool(1.0).as<bool>());
  EXPECT_ANY_THROW(
      DoubleInt64BoolVec(1.0).as<std::vector<DoubleInt64BoolVec>>());

  struct CustomType {};
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<double>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<int64_t>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<bool>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<int>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<float>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<SomeType>));
  static_assert(!opcheck<IntSomeType>.canCastTo(opcheck<CustomType>));
  static_assert((int64_t)IntSomeType(1) == 1);
  EXPECT_THAT(
      // suppress unused value warning
      []() { (void)(SomeType)IntSomeType(1); },
      ::testing::ThrowsMessage<c10::Error>(
          ::testing::HasSubstr("Cannot cast to ")));
}

#define TEST_BINARY_OP_ALLTYPE(name, op)                                       \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVec>); \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int>);                   \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<int>);                \
    static_assert(opcheck<int> op opcheck<DoubleInt64Bool>);                   \
    static_assert(opcheck<int> op opcheck<DoubleInt64BoolVec>);                \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.5))                          \
            .as<decltype(2L op 2.5)>() == (2L op 2.5));                        \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVec(2.5))                    \
            .as<decltype(2L op 2.5)>(),                                        \
        (2L op 2.5));                                                          \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op 2L).as<decltype((3L op 2L))>() == (3L op 2L)); \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(3L) op 2L).as<decltype((3L op 2L))>(),             \
        (3L op 2L));                                                           \
    static_assert(                                                             \
        (3L op DoubleInt64Bool(2L)).as<decltype((3L op 2L))>() == (3L op 2L)); \
    EXPECT_EQ(                                                                 \
        (3L op DoubleInt64BoolVec(2L)).as<decltype((3L op 2L))>(),             \
        (3L op 2L));                                                           \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2); },                    \
        ::testing::ThrowsMessage<c10::Error>(                                  \
            ::testing::HasSubstr("Can not compute ")));                        \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{})                \
              op DoubleInt64BoolVec(2);                                        \
        },                                                                     \
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

#define TEST_BINARY_OP_INT_ONLY(name, op)                                      \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVec>); \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int64_t>);               \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<int64_t>);            \
    static_assert(opcheck<int64_t> op opcheck<DoubleInt64Bool>);               \
    static_assert(opcheck<int64_t> op opcheck<DoubleInt64BoolVec>);            \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op DoubleInt64Bool(2L)).as<int64_t>() ==          \
        (3L op 2L));                                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(3L) op DoubleInt64BoolVec(2L)).as<int64_t>(),      \
        (3L op 2L));                                                           \
    static_assert((DoubleInt64Bool(3L) op 2L).as<int64_t>() == (3L op 2L));    \
    EXPECT_EQ((DoubleInt64BoolVec(3L) op 2L).as<int64_t>(), (3L op 2L));       \
    static_assert((3L op DoubleInt64Bool(2L)).as<int64_t>() == (3L op 2L));    \
    EXPECT_EQ((3L op DoubleInt64BoolVec(2L)).as<int64_t>(), (3L op 2L));       \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2); },                    \
        ::testing::ThrowsMessage<c10::Error>(                                  \
            ::testing::HasSubstr("Can not compute ")));                        \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{})                \
              op DoubleInt64BoolVec(2);                                        \
        },                                                                     \
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

#define TEST_UNARY_OP(name, op, int_or_bool)                                  \
  TEST_F(DynamicTypeTest, name) {                                             \
    static_assert(op opcheck<DoubleInt64Bool>);                               \
    static_assert(op opcheck<DoubleInt64BoolVec>);                            \
    static_assert((op DoubleInt64Bool(2L)).as<decltype(op 2L)>() == (op 2L)); \
    EXPECT_EQ((op DoubleInt64BoolVec(2L)).as<decltype(op 2L)>(), (op 2L));    \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64Bool(); },                                      \
        ::testing::ThrowsMessage<c10::Error>(                                 \
            ::testing::HasSubstr("Can not compute ")));                       \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}); },  \
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
  using IntOrFloat = DynamicType<NoContainers, int, float>;
  {
    constexpr IntOrFloat x = 1;
    constexpr IntOrFloat y = 2.5f;
    constexpr IntOrFloat z = x + y;
    static_assert(z.as<float>() == 3.5f);
  }
  // example 2
  struct CustomType {};
  {
    using IntOrFloatOrCustom =
        DynamicType<NoContainers, int, float, CustomType>;
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
    using Custom12 = DynamicType<NoContainers, CustomType, CustomType2>;
    static_assert(!(opcheck<Custom12> + opcheck<Custom12>));
  }
  // example 4
  {
    using BFloatOrHalfZero =
        DynamicType<NoContainers, bfloat16_zero, half_zero>;
    static_assert(!(opcheck<BFloatOrHalfZero> + opcheck<BFloatOrHalfZero>));
    using BFloatOrHalfZeroOrInt =
        DynamicType<NoContainers, bfloat16_zero, half_zero, int>;
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
  // example 6
  {
    using IntFloatVecList =
        DynamicType<Containers<std::vector, std::list>, int, float>;
    IntFloatVecList x = std::vector<IntFloatVecList>{1, 2.0f};
    IntFloatVecList y = std::list<IntFloatVecList>{3, x};
    EXPECT_TRUE(y.is<std::list<IntFloatVecList>>());
    EXPECT_EQ(y.as<std::list<IntFloatVecList>>().size(), 2);
    EXPECT_EQ(y.as<std::list<IntFloatVecList>>().front().as<int>(), 3);
    EXPECT_TRUE(y.as<std::list<IntFloatVecList>>()
                    .back()
                    .is<std::vector<IntFloatVecList>>());
    EXPECT_EQ(
        y.as<std::list<IntFloatVecList>>()
            .back()
            .as<std::vector<IntFloatVecList>>()
            .size(),
        2);
    EXPECT_EQ(
        y.as<std::list<IntFloatVecList>>()
            .back()
            .as<std::vector<IntFloatVecList>>()
            .front()
            .as<int>(),
        1);
    EXPECT_EQ(
        y.as<std::list<IntFloatVecList>>()
            .back()
            .as<std::vector<IntFloatVecList>>()
            .back()
            .as<float>(),
        2.0f);
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
  static_assert(!(+opcheck<DynamicType<NoContainers, Type2, SomeType>>));
  // defined compile time because +int is in type list
  static_assert(+opcheck<DynamicType<NoContainers, Type2, int>>);
  // runtime error because +Type2 is not in type list
  auto bad = [&]() { +DynamicType<NoContainers, Type2, int>(Type2{}); };
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
      !(opcheck<DynamicType<NoContainers, Type2, SomeType>> +
        opcheck<DynamicType<NoContainers, Type2, SomeType>>));
  static_assert(
      !(opcheck<DynamicType<NoContainers, Type2, SomeType>> + opcheck<Type2>));
  static_assert(
      !(opcheck<Type2> + opcheck<DynamicType<NoContainers, Type2, SomeType>>));
  // defined compile time because int+int is in type list
  static_assert(
      opcheck<DynamicType<NoContainers, Type2, int>> +
      opcheck<DynamicType<NoContainers, Type2, int>>);
  // runtime error because Type2+Type2 is not in type list
  auto bad = [&]() {
    DynamicType<NoContainers, Type2, int> x(Type2{});
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

namespace container_test {
// Testing containers support by implementing the set-theoretic definition of
// natural numbers:
// https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers

// TODO: unordered set is a better fit for this case, but it does not work with
// some old compilers (for example the old gcc on our CI). This is a workaround
#if !defined(__GNUC__) || __GNUC__ >= 12

struct StupidHash {
  template <typename T>
  size_t operator()(const T&) const {
    // This hash always collides, but who cares?
    return 0;
  }
};

template <typename T>
using UnorderedSetWithStupidHash = std::unordered_set<T, StupidHash>;

#else

template <typename T>
using UnorderedSetWithStupidHash = std::vector<T>;
#define insert push_back

#endif

using NaturalNumber = DynamicType<Containers<UnorderedSetWithStupidHash>>;

using Set = UnorderedSetWithStupidHash<NaturalNumber>;

#if 1
// DynamicType doesn't support operator== for containers, so we need to define
// it ourselves.
// TODO: add support for containers operator overloading in DynamicType, and
// remove this definition.

// operator== has to be in the top namespace for clang, otherwise clang will
// have trouble compiling it. operator== has to be in the container_test
// namespace for gcc, otherwise gcc will not compile.
#if defined(__clang__)
} // namespace container_test
#endif
bool operator==(
    const container_test::NaturalNumber& lhs,
    const container_test::NaturalNumber& rhs) {
  return lhs.as<container_test::Set>() == rhs.as<container_test::Set>();
}
#if defined(__clang__)
namespace container_test {
#endif
#endif

TEST_F(DynamicTypeTest, SetTheoreticNaturalNumbers) {
  auto next = [](const NaturalNumber& n) {
    // recursively define natural number n + 1 as n U {n}
    Set set = n.as<Set>();
    set.insert(n);
    return NaturalNumber(set);
  };

  NaturalNumber zero = Set{};
  NaturalNumber one = next(zero);
  NaturalNumber two = next(one);
  NaturalNumber three = next(two);
  NaturalNumber four = next(three);
  NaturalNumber five = next(four);
  NaturalNumber six = next(five);
  NaturalNumber seven = next(six);
  NaturalNumber eight = next(seven);
  NaturalNumber nine = next(eight);
  NaturalNumber ten = next(nine);

  EXPECT_EQ(zero.as<Set>().size(), 0);
  EXPECT_EQ(one.as<Set>().size(), 1);
  EXPECT_EQ(two.as<Set>().size(), 2);
  EXPECT_EQ(three.as<Set>().size(), 3);
  EXPECT_EQ(four.as<Set>().size(), 4);
  EXPECT_EQ(five.as<Set>().size(), 5);
  EXPECT_EQ(six.as<Set>().size(), 6);
  EXPECT_EQ(seven.as<Set>().size(), 7);
  EXPECT_EQ(eight.as<Set>().size(), 8);
  EXPECT_EQ(nine.as<Set>().size(), 9);
  EXPECT_EQ(ten.as<Set>().size(), 10);

  EXPECT_EQ(zero, NaturalNumber(Set{}));
  EXPECT_EQ(one, NaturalNumber(Set{zero}));
  EXPECT_EQ(two, NaturalNumber(Set{zero, one}));
  EXPECT_EQ(three, NaturalNumber(Set{zero, one, two}));
  EXPECT_EQ(four, NaturalNumber(Set{zero, one, two, three}));
  EXPECT_EQ(five, NaturalNumber(Set{zero, one, two, three, four}));
  EXPECT_EQ(six, NaturalNumber(Set{zero, one, two, three, four, five}));
  EXPECT_EQ(seven, NaturalNumber(Set{zero, one, two, three, four, five, six}));
  EXPECT_EQ(
      eight, NaturalNumber(Set{zero, one, two, three, four, five, six, seven}));
  EXPECT_EQ(
      nine,
      NaturalNumber(Set{zero, one, two, three, four, five, six, seven, eight}));
  EXPECT_EQ(
      ten,
      NaturalNumber(
          Set{zero, one, two, three, four, five, six, seven, eight, nine}));
}

#undef insert

} // namespace container_test

} // namespace nvfuser
