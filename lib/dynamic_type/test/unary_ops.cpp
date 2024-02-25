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
