// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "dynamic_type/dynamic_type.h"

#include "utils.h"

// Helper to check if unary operator is available (SFINAE-friendly)
template <typename T, typename Op>
struct has_unary_op : std::false_type {};

template <typename T>
struct has_unary_op<T, std::integral_constant<int, '+'>> : std::bool_constant<requires { +std::declval<T>(); }> {};

template <typename T>
struct has_unary_op<T, std::integral_constant<int, '-'>> : std::bool_constant<requires { -std::declval<T>(); }> {};

template <typename T>
struct has_unary_op<T, std::integral_constant<int, '~'>> : std::bool_constant<requires { ~std::declval<T>(); }> {};

template <typename T>
struct has_unary_op<T, std::integral_constant<int, '!'>> : std::bool_constant<requires { !std::declval<T>(); }> {};

#define TEST_UNARY_OP(name, op, int_or_bool, opchar)                          \
  TEST_F(DynamicTypeTest, name) {                                             \
    static_assert(has_unary_op<DoubleInt64Bool, std::integral_constant<int, opchar>>::value); \
    static_assert(has_unary_op<DoubleInt64BoolVec, std::integral_constant<int, opchar>>::value); \
    static_assert((op DoubleInt64Bool(2L)).as<decltype(op 2L)>() == (op 2L)); \
    EXPECT_EQ((op DoubleInt64BoolVec(2L)).as<decltype(op 2L)>(), (op 2L));    \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64Bool(); },                                      \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(    \
            "Result is dynamic but not convertible to result type")));        \
    EXPECT_THAT(                                                              \
        [&]() { op DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}); },  \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(    \
            "Result is dynamic but not convertible to result type")));        \
    static_assert(has_unary_op<int_or_bool##SomeType, std::integral_constant<int, opchar>>::value); \
    EXPECT_THAT(                                                              \
        [&]() { op int_or_bool##SomeType(SomeType{}); },                      \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(    \
            "Result is dynamic but not convertible to result type")));        \
  }

TEST_UNARY_OP(Positive, +, Int, '+');
TEST_UNARY_OP(Negative, -, Int, '-');
TEST_UNARY_OP(BinaryNot, ~, Int, '~');
#undef TEST_UNARY_OP

TEST_F(DynamicTypeTest, LogicalNot) {
  static_assert(has_unary_op<DoubleInt64Bool, std::integral_constant<int, '!'>>::value);
  static_assert(has_unary_op<DoubleInt64BoolVec, std::integral_constant<int, '!'>>::value);
  static_assert(std::is_same_v<decltype(!DoubleInt64Bool(2L)), bool>);
  static_assert((!DoubleInt64Bool(2L)) == (!2L));
  static_assert(std::is_same_v<decltype(!DoubleInt64BoolVec(2L)), bool>);
  EXPECT_EQ(!DoubleInt64BoolVec(2L), (!2L));
  EXPECT_THAT(
      [&]() { !DoubleInt64Bool(); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
  EXPECT_THAT(
      [&]() { !DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
  static_assert(has_unary_op<BoolSomeType, std::integral_constant<int, '!'>>::value);
  EXPECT_THAT(
      [&]() { !BoolSomeType(SomeType{}); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
}

TEST_F(DynamicTypeTest, UnaryOpAdvancedTyping) {
  struct Type1 {};
  struct Type2 {
    Type1 operator+() const {
      return Type1{};
    }
  };
  // defined compile time because +Type2 is defined
  static_assert(has_unary_op<DynamicType<NoContainers, Type2, SomeType>, std::integral_constant<int, '+'>>::value);
  static_assert(
      std::is_same_v<
          decltype(+std::declval<DynamicType<NoContainers, Type2, SomeType>>()),
          Type1>);
  // defined compile time because +int is in type list
  static_assert(has_unary_op<DynamicType<NoContainers, Type2, int>, std::integral_constant<int, '+'>>::value);
  // runtime error because +Type2 is not in type list
  auto bad = [&]() { +DynamicType<NoContainers, Type2, int>(Type2{}); };
  EXPECT_THAT(
      bad,
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
}

// Helper for dereference operator
template <typename T, typename = void>
struct has_dereference : std::false_type {};

template <typename T>
struct has_dereference<T, std::void_t<decltype(*std::declval<T>())>> : std::true_type {};

TEST_F(DynamicTypeTest, Star) {
  using IntOrPtr = DynamicType<Containers<std::shared_ptr>, int>;
  static_assert(has_dereference<IntOrPtr>::value);
  static_assert(!has_dereference<DoubleInt64Bool>::value);
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
  }
}
