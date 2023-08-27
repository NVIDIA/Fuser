// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "dynamic_type.h"

#if defined(__GLIBCXX__) && __GLIBCXX__ >= 20230714
#define STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE 1
#endif

namespace dynamic_type {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif

struct SomeType {};
struct SomeType2 {};

class DynamicTypeTest : public ::testing::Test {};

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

// Utilities for testing if we have T->as<Template> defined
template <typename T, template <typename...> typename Template>
static auto hasAsTemplateHelper(int)
    -> decltype(std::declval<T>().template as<Template>(), std::true_type{});

template <typename, template <typename...> typename>
static auto hasAsTemplateHelper(long) -> std::false_type;

template <typename T, template <typename...> typename Template>
struct hasAsTemplate : decltype(hasAsTemplateHelper<T, Template>(int{})) {};

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
  static_assert(hasAsTemplate<DoubleInt64BoolVec, std::vector>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, SomeType>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, int>::value);
  static_assert(!hasAsTemplate<DoubleInt64BoolVec, std::list>::value);

  static_assert((int)DoubleInt64Bool(true) == 1);
  EXPECT_EQ((int)DoubleInt64BoolVec(true), 1);

  EXPECT_ANY_THROW(DoubleInt64Bool(1.0).as<bool>());
  EXPECT_ANY_THROW(DoubleInt64BoolVec(1.0).as<std::vector>());

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
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot cast from ")));
}

TEST_F(DynamicTypeTest, MoveCtor) {
  struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(const NonCopyable&) = delete;
    NonCopyable& operator=(NonCopyable&&) = default;
  };
  using NonCopyableType = DynamicType<NoContainers, NonCopyable>;
  static_assert(std::is_move_constructible_v<NonCopyableType>);
  static_assert(std::is_move_assignable_v<NonCopyableType>);
  static_assert(std::is_nothrow_move_constructible_v<NonCopyableType>);
  static_assert(std::is_nothrow_move_assignable_v<NonCopyableType>);
  NonCopyable a;
  // This should not compile:
  // NonCopyableType bb(a);
  NonCopyableType b(std::move(a));
  // Suppress unused var warning
  (void)b;
}

namespace null_tests {

constexpr DoubleInt64Bool a, b;
static_assert(a.isNull());
static_assert(!a.hasValue());
static_assert(b.isNull());
static_assert(!b.hasValue());
static_assert(a == b);
static_assert(b == a);
static_assert(!(a != b));
static_assert(!(b != a));
static_assert(!(a < b));
static_assert(!(b < a));
static_assert(!(a > b));
static_assert(!(b > a));
static_assert(a <= b);
static_assert(b <= a);
static_assert(a >= b);
static_assert(b >= a);
static_assert(a == std::monostate{});
static_assert(std::monostate{} == a);
static_assert(!(a != std::monostate{}));
static_assert(!(std::monostate{} != a));
static_assert(!(a < std::monostate{}));
static_assert(!(std::monostate{} < a));
static_assert(!(a > std::monostate{}));
static_assert(!(std::monostate{} > a));
static_assert(a <= std::monostate{});
static_assert(std::monostate{} <= a);
static_assert(a >= std::monostate{});
static_assert(std::monostate{} >= a);

} // namespace null_tests

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
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2L); },                   \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{})                \
              op DoubleInt64BoolVec(2L);                                       \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);                \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));                 \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },          \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
  }

TEST_BINARY_OP_ALLTYPE(Add, +);
TEST_BINARY_OP_ALLTYPE(Minus, -);
TEST_BINARY_OP_ALLTYPE(Mul, *);
TEST_BINARY_OP_ALLTYPE(Div, /);
TEST_BINARY_OP_ALLTYPE(LogicalAnd, &&);
TEST_BINARY_OP_ALLTYPE(LogicalOr, ||);

#define TEST_COMPARE_OP(name, op)                                              \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVec>); \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int>);                   \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<int>);                \
    static_assert(opcheck<int> op opcheck<DoubleInt64Bool>);                   \
    static_assert(opcheck<int> op opcheck<DoubleInt64BoolVec>);                \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.0)) == (2L op 2.0));         \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVec(2.0)), (2L op 2.0));     \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(std::vector<int64_t>{2})                           \
             op DoubleInt64BoolVec(std::vector<double>{2.0})),                 \
        (2L op 2.0));                                                          \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.5)) == (2L op 2.5));         \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVec(2.5)), (2L op 2.5));     \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(std::vector<int64_t>{2L})                          \
             op DoubleInt64BoolVec(std::vector<double>{2.5})),                 \
        (2L op 2.5));                                                          \
    static_assert((DoubleInt64Bool(3L) op 2L) == (3L op 2L));                  \
    EXPECT_EQ((DoubleInt64BoolVec(3L) op 2L), (3L op 2L));                     \
    static_assert((3L op DoubleInt64Bool(2L)) == (3L op 2L));                  \
    EXPECT_EQ((3L op DoubleInt64BoolVec(2L)), (3L op 2L));                     \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2L); },                   \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{})                \
              op DoubleInt64BoolVec(2L);                                       \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);                \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));                 \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },          \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
  }

TEST_COMPARE_OP(Eq, ==);
TEST_COMPARE_OP(Ne, !=);
TEST_COMPARE_OP(Lt, <);
TEST_COMPARE_OP(Gt, >);
TEST_COMPARE_OP(Le, <=);
TEST_COMPARE_OP(Ge, >=);

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
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(2L); },                   \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{})                \
              op DoubleInt64BoolVec(2L);                                       \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
    static_assert(opcheck<IntSomeType> + opcheck<IntSomeType>);                \
    static_assert(!(opcheck<SomeTypes> + opcheck<SomeTypes>));                 \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },          \
        ::testing::ThrowsMessage<std::runtime_error>(                          \
            ::testing::HasSubstr("Cannot compute ")));                         \
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
        ::testing::ThrowsMessage<std::runtime_error>(                         \
            ::testing::HasSubstr("Cannot compute ")));                        \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}); },  \
        ::testing::ThrowsMessage<std::runtime_error>(                         \
            ::testing::HasSubstr("Cannot compute ")));                        \
    static_assert(op opcheck<int_or_bool##SomeType>);                         \
    static_assert(!(op opcheck<SomeTypes>));                                  \
    EXPECT_THAT(                                                              \
        [&]() { op int_or_bool##SomeType(SomeType{}); },                      \
        ::testing::ThrowsMessage<std::runtime_error>(                         \
            ::testing::HasSubstr("Cannot compute ")));                        \
  }

TEST_UNARY_OP(Positive, +, Int);
TEST_UNARY_OP(Negative, -, Int);
TEST_UNARY_OP(BinaryNot, ~, Int);
TEST_UNARY_OP(LogicalNot, !, Bool);

#undef TEST_UNARY_OP

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
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot compute ")));
}

TEST_F(DynamicTypeTest, BinaryOpAdvancedTyping) {
  struct Type1 {};
  struct Type2 {
    constexpr Type1 operator+(Type2) const {
      return Type1{};
    }
    constexpr Type1 operator+() const {
      return Type1{};
    }
  };
  struct Type3 {
    constexpr Type3() = default;
    constexpr Type3(Type1) {}
    constexpr bool operator==(Type3) const {
      return true;
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
  // defined compile time because Type2+Type2 and +Type2 is constructible to
  // Type3
  using Type2Type3 = DynamicType<NoContainers, Type2, Type3>;
  static_assert(opcheck<Type2Type3> + opcheck<Type2Type3>);
  static_assert(Type2Type3(Type2{}) + Type2Type3(Type2{}) == Type3{});
  static_assert(opcheck<Type2Type3> + opcheck<Type2>);
  static_assert(Type2Type3(Type2{}) + Type2{} == Type3{});
  static_assert(opcheck<Type2> + opcheck<Type2Type3>);
  static_assert(Type2{} + Type2Type3(Type2{}) == Type3{});
  static_assert(+opcheck<Type2Type3>);
  static_assert(+Type2Type3(Type2{}) == Type3{});
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
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot compute ")));
  // test bool to int conversion
  using Int = DynamicType<NoContainers, int>;
  static_assert((Int(2) && Int(0)) == 0);
  static_assert((Int(2) && Int(3)) == 1);
}

TEST_F(DynamicTypeTest, CastToDynamicType) {
  using IntOrFloat = DynamicType<NoContainers, int, float>;
  struct A {
    constexpr operator IntOrFloat() const {
      return 1;
    }
  };
  static_assert((IntOrFloat)A{} == 1);
  IntOrFloat x = A{};
  EXPECT_EQ(x, 1);
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
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Can not print")));
  EXPECT_THAT(
      [&]() { ss << IntSomeType(SomeType{}); },
      ::testing::ThrowsMessage<std::runtime_error>(
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
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          ++x;
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
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
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          --x;
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
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
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          x++;
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
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
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
    EXPECT_THAT(
        []() {
          IntSomeType x(SomeType{});
          x--;
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot compute ")));
    static_assert(!(opcheck<SomeTypes&> --));
  }
}

TEST_F(DynamicTypeTest, Star) {
  using IntOrPtr = DynamicType<Containers<std::shared_ptr>, int>;
  static_assert(*opcheck<IntOrPtr>);
  static_assert(!(*opcheck<DoubleInt64Bool>));
  IntOrPtr x = 299792458;
  IntOrPtr y = std::make_shared<IntOrPtr>(x);
  EXPECT_EQ(*y, 299792458);
  (*y)--;
  EXPECT_EQ(*y, 299792457);
  EXPECT_THAT(
      [&]() { *x; },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot dereference ")));
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
        ::testing::ThrowsMessage<std::runtime_error>(      \
            ::testing::HasSubstr("Cannot compute ")));     \
    EXPECT_THAT(                                           \
        []() {                                             \
          IntSomeType x(SomeType{});                       \
          x += 1;                                          \
        },                                                 \
        ::testing::ThrowsMessage<std::runtime_error>(      \
            ::testing::HasSubstr("Cannot compute ")));     \
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
// some old compilers (for example the old gcc on our CI). This is a workaround.
// See [Incomplete type support in STL] for more details.
#if defined(STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE)

struct StupidHash {
  template <typename T>
  size_t operator()(const T&) const {
    // This hash always collides, but who cares?
    return 0;
  }
};

template <typename T>
using Set = std::unordered_set<T, StupidHash>;

#else

template <typename T>
using Set = std::vector<T>;
#define insert push_back

#endif

using NaturalNumber = DynamicType<Containers<Set>>;

TEST_F(DynamicTypeTest, SetTheoreticNaturalNumbers) {
  auto next = [](const NaturalNumber& n) {
    // recursively define natural number n + 1 as n U {n}
    auto set = n.as<Set>();
    set.insert(n);
    return NaturalNumber(set);
  };

  NaturalNumber zero = Set<NaturalNumber>{};
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

  EXPECT_TRUE(zero.is<Set>());
  EXPECT_TRUE(one.is<Set>());
  EXPECT_TRUE(two.is<Set>());
  EXPECT_TRUE(three.is<Set>());
  EXPECT_TRUE(four.is<Set>());
  EXPECT_TRUE(five.is<Set>());
  EXPECT_TRUE(six.is<Set>());
  EXPECT_TRUE(seven.is<Set>());
  EXPECT_TRUE(eight.is<Set>());
  EXPECT_TRUE(nine.is<Set>());
  EXPECT_TRUE(ten.is<Set>());

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

  EXPECT_EQ(zero, NaturalNumber(Set<NaturalNumber>{}));
  EXPECT_EQ(one, NaturalNumber(Set<NaturalNumber>{zero}));
  EXPECT_EQ(two, NaturalNumber(Set<NaturalNumber>{zero, one}));
  EXPECT_EQ(three, NaturalNumber(Set<NaturalNumber>{zero, one, two}));
  EXPECT_EQ(four, NaturalNumber(Set<NaturalNumber>{zero, one, two, three}));
  EXPECT_EQ(
      five, NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four}));
  EXPECT_EQ(
      six,
      NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four, five}));
  EXPECT_EQ(
      seven,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six}));
  EXPECT_EQ(
      eight,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six, seven}));
  EXPECT_EQ(
      nine,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight}));
  EXPECT_EQ(
      ten,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight, nine}));
}

#undef insert

TEST_F(DynamicTypeTest, FromContainerToContainer) {
  using IntOrVec = DynamicType<Containers<std::vector>, int>;
  using Vec = DynamicType<Containers<std::vector>>;

  static_assert(std::is_constructible_v<IntOrVec, std::vector<int>>);
  static_assert(
      std::is_constructible_v<IntOrVec, std::vector<std::vector<int>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<int>>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<std::vector<int>>>>>);

  static_assert(opcheck<IntOrVec>.canCastTo(opcheck<std::vector<double>>));
  static_assert(
      opcheck<IntOrVec>.canCastTo(opcheck<std::vector<std::vector<double>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<double>>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<std::vector<double>>>>>));

  static_assert(opcheck<IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<Vec>]);
  static_assert(opcheck<IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<const Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<const Vec>]);

  IntOrVec zero = 0;
  IntOrVec one = 1;
  IntOrVec two = 2;

  std::vector<std::vector<int>> vvi{{1, 2, 3}, {4, 5, 6}};
  IntOrVec x = vvi;
  EXPECT_EQ(x[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[0][0], 1);
  EXPECT_EQ(x[0][1], 2);
  EXPECT_EQ(x[0][2], 3);
  EXPECT_EQ(x[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[1][0], 4);
  EXPECT_EQ(x[1][1], 5);
  EXPECT_EQ(x[1][2], 6);

  EXPECT_EQ(x[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[zero][zero], 1);
  EXPECT_EQ(x[zero][one], 2);
  EXPECT_EQ(x[zero][two], 3);
  EXPECT_EQ(x[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[one][zero], 4);
  EXPECT_EQ(x[one][one], 5);
  EXPECT_EQ(x[one][two], 6);

  const IntOrVec xx = vvi;
  EXPECT_EQ(xx[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[0][0], 1);
  EXPECT_EQ(xx[0][1], 2);
  EXPECT_EQ(xx[0][2], 3);
  EXPECT_EQ(xx[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[1][0], 4);
  EXPECT_EQ(xx[1][1], 5);
  EXPECT_EQ(xx[1][2], 6);

  EXPECT_EQ(xx[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[zero][zero], 1);
  EXPECT_EQ(xx[zero][one], 2);
  EXPECT_EQ(xx[zero][two], 3);
  EXPECT_EQ(xx[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[one][zero], 4);
  EXPECT_EQ(xx[one][one], 5);
  EXPECT_EQ(xx[one][two], 6);

  std::vector<std::vector<double>> vvd{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ((std::vector<std::vector<double>>)x, vvd);
  EXPECT_EQ((std::vector<double>)x[0], vvd[0]);
  EXPECT_EQ((std::vector<double>)x[1], vvd[1]);
}

} // namespace container_test

TEST_F(DynamicTypeTest, Hash) {
  static_assert(has_cross_type_equality<DoubleInt64Bool>);
  using IntOrStr = DynamicType<NoContainers, int64_t, std::string>;
  static_assert(!has_cross_type_equality<IntOrStr>);
  std::unordered_map<IntOrStr, double> m;
  m[IntOrStr(0L)] = 0;
  m[IntOrStr(299792458L)] = 299792458;
  m[IntOrStr("speed of light")] = 299792458;
  m[IntOrStr("pi")] = 3.14159;
  EXPECT_EQ(m.at(IntOrStr(0L)), 0);
  EXPECT_EQ(m.at(IntOrStr(299792458L)), 299792458);
  EXPECT_EQ(m.at(IntOrStr("speed of light")), 299792458);
  EXPECT_EQ(m.at(IntOrStr("pi")), 3.14159);
}

} // namespace dynamic_type

template <>
struct std::hash<dynamic_type::DoubleInt64Bool> {
  size_t operator()(const dynamic_type::DoubleInt64Bool& x) const {
    return 0;
  }
};

namespace dynamic_type {

TEST_F(DynamicTypeTest, Hash2) {
  std::unordered_map<DoubleInt64Bool, double> m;
  m[DoubleInt64Bool(false)] = 0;
  m[DoubleInt64Bool(299792458L)] = 299792458;
  m[DoubleInt64Bool(3.14159)] = 3.14159;
  EXPECT_EQ(m.at(DoubleInt64Bool(false)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(0L)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(0.0)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(299792458L)), 299792458);
  EXPECT_EQ(m.at(DoubleInt64Bool(299792458.0)), 299792458);
  EXPECT_EQ(m.at(DoubleInt64Bool(3.14159)), 3.14159);
}

namespace member_pointer_test {
struct A {
  int x;
  int y;
};
struct B {
  int x;
  int y;
};
struct C {
  int x;
  int y;
};
struct D {
  int x;
  int y;
};
struct E {
  int x;
  int y;
};

struct CD {
  std::variant<C, D> v;

  constexpr const int& operator->*(int C::*member) const {
    return std::get<C>(v).*member;
  }

  constexpr const int& operator->*(int D::*member) const {
    return std::get<D>(v).*member;
  }

  constexpr int& operator->*(int C::*member) {
    return std::get<C>(v).*member;
  }

  constexpr int& operator->*(int D::*member) {
    return std::get<D>(v).*member;
  }
};

TEST_F(DynamicTypeTest, MemberPointer) {
  using ABCD = DynamicType<NoContainers, A, B, CD>;
  constexpr ABCD a = A{1, 2};
  static_assert(a->*&A::x == 1);
  static_assert(a->*&A::y == 2);
  constexpr ABCD b = B{3, 4};
  static_assert(b->*&B::x == 3);
  static_assert(b->*&B::y == 4);
  constexpr ABCD c = CD{C{5, 6}};
#if __cplusplus >= 202002L
  static_assert(c->*&C::x == 5);
  static_assert(c->*&C::y == 6);
#else
  EXPECT_EQ(c->*&C::x, 5);
  EXPECT_EQ(c->*&C::y, 6);
#endif
  constexpr ABCD d = CD{D{7, 8}};
#if __cplusplus >= 202002L
  static_assert(d->*&D::x == 7);
  static_assert(d->*&D::y == 8);
#else
  EXPECT_EQ(d->*&D::x, 7);
  EXPECT_EQ(d->*&D::y, 8);
#endif
  static_assert(opcheck<ABCD>->*opcheck<int A::*>);
  static_assert(opcheck<ABCD>->*opcheck<int B::*>);
  static_assert(opcheck<ABCD>->*opcheck<int C::*>);
  static_assert(opcheck<ABCD>->*opcheck<int D::*>);
  static_assert(!(opcheck<ABCD>->*opcheck<int E::*>));

  ABCD aa = a;
  EXPECT_EQ(aa->*&A::x, 1);
  EXPECT_EQ(aa->*&A::y, 2);
  aa->*& A::x = 299792458;
  aa->*& A::y = 314159;
  EXPECT_EQ(aa->*&A::x, 299792458);
  EXPECT_EQ(aa->*&A::y, 314159);

  ABCD cc = c;
  EXPECT_EQ(cc->*&C::x, 5);
  EXPECT_EQ(cc->*&C::y, 6);
  cc->*& C::x = 299792458;
  cc->*& C::y = 314159;
  EXPECT_EQ(cc->*&C::x, 299792458);
  EXPECT_EQ(cc->*&C::y, 314159);
}

struct F {
  int x;
  int y;
  constexpr const int& operator->*(std::string_view member) const {
    if (member == "x") {
      return x;
    } else if (member == "y") {
      return y;
    } else {
      throw std::runtime_error("invalid member");
    }
  }
  constexpr int& operator->*(std::string_view member) {
    if (member == "x") {
      return x;
    } else if (member == "y") {
      return y;
    } else {
      throw std::runtime_error("invalid member");
    }
  }
};

struct G : public F {};

TEST_F(DynamicTypeTest, NonMemberPointerArrowStarRef) {
  using EFG = DynamicType<NoContainers, E, F, G>;

  constexpr EFG f = F{1, 2};
#if __cplusplus >= 202002L
  static_assert(f->*"x" == 1);
  static_assert(f->*"y" == 2);
#else
  EXPECT_EQ(f->*"x", 1);
  EXPECT_EQ(f->*"y", 2);
#endif

  constexpr EFG g = G{3, 4};
#if __cplusplus >= 202002L
  static_assert(g->*"x" == 3);
  static_assert(g->*"y" == 4);
#else
  EXPECT_EQ(g->*"x", 3);
  EXPECT_EQ(g->*"y", 4);
#endif

  static_assert(opcheck<EFG>->*opcheck<std::string_view>);
  static_assert(!(opcheck<EFG>->*opcheck<int>));

  EFG ff = f;
  EXPECT_EQ(ff->*"x", 1);
  EXPECT_EQ(ff->*"y", 2);
  ff->*"x" = 299792458;
  ff->*"y" = 314159;
  EXPECT_EQ(ff->*"x", 299792458);
  EXPECT_EQ(ff->*"y", 314159);
}

class ConstAccessor {
  std::function<int()> getter_;

 public:
  ConstAccessor(std::function<int()> getter) : getter_(getter) {}

  operator int() const {
    return getter_();
  }
};

class Accessor {
  std::function<int()> getter_;
  std::function<void(int)> setter_;

 public:
  Accessor(std::function<int()> getter, std::function<void(int)> setter)
      : getter_(getter), setter_(setter) {}

  const Accessor& operator=(int value) const {
    setter_(value);
    return *this;
  }
  operator int() const {
    return getter_();
  }
};

struct H {
  int x;
  int y;
  ConstAccessor operator->*(std::string_view member) const {
    if (member == "x") {
      return ConstAccessor{[this]() { return x; }};
    } else if (member == "y") {
      return ConstAccessor{[this]() { return y; }};
    } else {
      throw std::runtime_error("invalid member");
    }
  }
  Accessor operator->*(std::string_view member) {
    if (member == "x") {
      return Accessor{[this]() { return x; }, [this](int value) { x = value; }};
    } else if (member == "y") {
      return Accessor{[this]() { return y; }, [this](int value) { y = value; }};
    } else {
      throw std::runtime_error("invalid member");
    }
  }
};

struct I : public H {};

TEST_F(DynamicTypeTest, NonMemberPointerArrowStaAccessor) {
  using EHI = DynamicType<NoContainers, E, H, I>;

  EHI h = H{1, 2};
  EXPECT_EQ(h->*"x", 1);
  EXPECT_EQ(h->*"y", 2);

  EHI i = I{3, 4};
  EXPECT_EQ(i->*"x", 3);
  EXPECT_EQ(i->*"y", 4);

  static_assert(opcheck<EHI>->*opcheck<std::string_view>);
  static_assert(!(opcheck<EHI>->*opcheck<int>));

  EHI hh = h;
  EXPECT_EQ(hh->*"x", 1);
  EXPECT_EQ(hh->*"y", 2);
  hh->*"x" = 299792458;
  hh->*"y" = 314159;
  EXPECT_EQ(hh->*"x", 299792458);
  EXPECT_EQ(hh->*"y", 314159);
}

TEST_F(DynamicTypeTest, MemberFunctions) {
  struct J {
    constexpr std::string_view no_qualifiers() {
      return "no qualifiers";
    }

    constexpr std::string_view const_qualifiers() const {
      return "const qualifiers";
    }

    constexpr std::string_view volatile_qualifiers() volatile {
      return "volatile qualifiers";
    }

    constexpr std::string_view const_volatile_qualifiers() const volatile {
      return "const volatile qualifiers";
    }

    constexpr std::string_view lvalue_ref_qualifiers() & {
      return "lvalue ref qualifiers";
    }

    constexpr std::string_view const_lvalue_ref_qualifiers() const& {
      return "const lvalue ref qualifiers";
    }

    constexpr std::string_view volatile_lvalue_ref_qualifiers() volatile& {
      return "volatile lvalue ref qualifiers";
    }

    constexpr std::string_view noexcept_qualifiers() noexcept {
      return "noexcept qualifiers";
    }

    constexpr std::string_view noexcept_false_qualifiers() noexcept(false) {
      return "noexcept(false) qualifiers";
    }

    constexpr std::string_view noexcept_true_qualifiers() noexcept(true) {
      return "noexcept(true) qualifiers";
    }

    constexpr int two_arguments(int a, int b) const {
      return a + b;
    }

    constexpr int three_arguments(int a, int b, int c) const {
      return a + b + c;
    }
  };

  using EJ = DynamicType<NoContainers, E, J>;
  constexpr EJ j = J{};
  static_assert((j->*&J::const_qualifiers)() == "const qualifiers");
  static_assert(
      (j->*&J::const_volatile_qualifiers)() == "const volatile qualifiers");
  static_assert(
      (j->*&J::const_lvalue_ref_qualifiers)() == "const lvalue ref qualifiers");
  static_assert((j->*&J::two_arguments)(10, 2) == 12);
  static_assert((j->*&J::three_arguments)(10, 2, 300) == 312);

  // Not using static_assert below because we can not call functions without
  // const qualifier in the constant evaluation context
  EJ jj = j;
  EXPECT_EQ((jj->*&J::no_qualifiers)(), "no qualifiers");
  EXPECT_EQ((jj->*&J::volatile_qualifiers)(), "volatile qualifiers");
  EXPECT_EQ((jj->*&J::lvalue_ref_qualifiers)(), "lvalue ref qualifiers");
  EXPECT_EQ(
      (jj->*&J::volatile_lvalue_ref_qualifiers)(),
      "volatile lvalue ref qualifiers");
  EXPECT_EQ((jj->*&J::noexcept_qualifiers)(), "noexcept qualifiers");
  EXPECT_EQ(
      (jj->*&J::noexcept_false_qualifiers)(), "noexcept(false) qualifiers");
  EXPECT_EQ((jj->*&J::noexcept_true_qualifiers)(), "noexcept(true) qualifiers");
}

} // namespace member_pointer_test

} // namespace dynamic_type
