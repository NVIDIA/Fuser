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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

#define TEST_BINARY_OP_ALLTYPE(name, op)                                       \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64Bool>);       \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVec>); \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<int>);                   \
    static_assert(opcheck<DoubleInt64BoolVec> op opcheck<int>);                \
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64BoolTwo>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVecTwo>);        \
    static_assert(opcheck<int> op opcheck<DoubleInt64Bool>);                   \
    static_assert(opcheck<int> op opcheck<DoubleInt64BoolVec>);                \
    static_assert(opcheck<DoubleInt64BoolTwo> op opcheck<DoubleInt64Bool>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVecTwo> op opcheck<DoubleInt64BoolVec>);        \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.5))                          \
            .as<decltype(2L op 2.5)>() == (2L op 2.5));                        \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64BoolTwo{})                          \
            .as<decltype(2L op 2L)>() == (2L op 2L));                          \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(2L))                         \
            .as<decltype(2L op 2L)>() == (2L op 2L));                          \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVec(2.5))                    \
            .as<decltype(2L op 2.5)>(),                                        \
        (2L op 2.5));                                                          \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVecTwo{})                    \
            .as<decltype(2L op 2L)>(),                                         \
        (2L op 2L));                                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(2L))                   \
            .as<decltype(2L op 2L)>(),                                         \
        (2L op 2L));                                                           \
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
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64BoolTwo>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVecTwo>);        \
    static_assert(opcheck<int> op opcheck<DoubleInt64Bool>);                   \
    static_assert(opcheck<int> op opcheck<DoubleInt64BoolVec>);                \
    static_assert(opcheck<DoubleInt64BoolTwo> op opcheck<DoubleInt64Bool>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVecTwo> op opcheck<DoubleInt64BoolVec>);        \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64Bool(2.0)) == (2L op 2.0));         \
    static_assert(                                                             \
        (DoubleInt64Bool(2L) op DoubleInt64BoolTwo{}) == (2L op 2L));          \
    static_assert(                                                             \
        (DoubleInt64Bool(1L) op DoubleInt64BoolTwo{}) == (1L op 2L));          \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op DoubleInt64BoolTwo{}) == (3L op 2L));          \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(2L)) == (2L op 2L));         \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(1L)) == (2L op 1L));         \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(3L)) == (2L op 3L));         \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVec(2.0)), (2L op 2.0));     \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(2L) op DoubleInt64BoolVecTwo{}), (2L op 2L));      \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(1L) op DoubleInt64BoolVecTwo{}), (1L op 2L));      \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(3L) op DoubleInt64BoolVecTwo{}), (3L op 2L));      \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(2L)), (2L op 2L));     \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(1L)), (1L op 2L));     \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(3L)), (3L op 2L));     \
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
    static_assert(opcheck<DoubleInt64Bool> op opcheck<DoubleInt64BoolTwo>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVec> op opcheck<DoubleInt64BoolVecTwo>);        \
    static_assert(opcheck<int64_t> op opcheck<DoubleInt64Bool>);               \
    static_assert(opcheck<int64_t> op opcheck<DoubleInt64BoolVec>);            \
    static_assert(opcheck<DoubleInt64BoolTwo> op opcheck<DoubleInt64Bool>);    \
    static_assert(                                                             \
        opcheck<DoubleInt64BoolVecTwo> op opcheck<DoubleInt64BoolVec>);        \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op DoubleInt64Bool(2L)).as<int64_t>() ==          \
        (3L op 2L));                                                           \
    static_assert(                                                             \
        (DoubleInt64Bool(3L) op DoubleInt64BoolTwo{})                          \
            .as<decltype(3L op 2L)>() == (3L op 2L));                          \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(3L))                         \
            .as<decltype(2L op 3L)>() == (2L op 3L));                          \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(3L) op DoubleInt64BoolVec(2L)).as<int64_t>(),      \
        (3L op 2L));                                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(3L) op DoubleInt64BoolVecTwo{})                    \
            .as<decltype(3L op 2L)>(),                                         \
        (3L op 2L));                                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(3L))                   \
            .as<decltype(2L op 3L)>(),                                         \
        (2L op 3L));                                                           \
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

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
