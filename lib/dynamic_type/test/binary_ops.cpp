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

#include <cstdint>

#include "utils.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#endif

namespace {

template <typename T, typename U>
concept HasPlus = requires(T t, U u) {
  t + u;
};

struct AdvancedType1 {};
struct AdvancedType2 {
  constexpr AdvancedType1 operator+(AdvancedType2) const {
    return AdvancedType1{};
  }
  constexpr AdvancedType1 operator+() const {
    return AdvancedType1{};
  }
};
struct AdvancedType3 {
  constexpr AdvancedType3() = default;
  constexpr AdvancedType3(AdvancedType1) {}
  constexpr bool operator==(AdvancedType3) const {
    return true;
  }
};

using AdvancedType2SomeType =
    DynamicType<NoContainers, AdvancedType2, SomeType>;
using AdvancedType2Type3 =
    DynamicType<NoContainers, AdvancedType2, AdvancedType3>;
using AdvancedType2Int = DynamicType<NoContainers, AdvancedType2, int>;

} // namespace

#define TEST_BINARY_OP_ALLTYPE(name, op)                                       \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(requires(DoubleInt64Bool a, DoubleInt64Bool b) { a op b; }); \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVec b) { a op b; });     \
    static_assert(requires(DoubleInt64Bool a, int b) { a op b; });             \
    static_assert(requires(DoubleInt64BoolVec a, int b) { a op b; });          \
    static_assert(                                                             \
        requires(DoubleInt64Bool a, DoubleInt64BoolTwo b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVecTwo b) { a op b; });  \
    static_assert(requires(int a, DoubleInt64Bool b) { a op b; });             \
    static_assert(requires(int a, DoubleInt64BoolVec b) { a op b; });          \
    static_assert(                                                             \
        requires(DoubleInt64BoolTwo a, DoubleInt64Bool b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVecTwo a, DoubleInt64BoolVec b) { a op b; });  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{2}) op DoubleInt64Bool(2.5))                  \
            .as<decltype(int64_t{2} op 2.5)>() == (int64_t{2} op 2.5));        \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{2}) op DoubleInt64BoolTwo{})                  \
            .as<decltype(int64_t{2} op int64_t{2})>() ==                       \
        (int64_t{2} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(int64_t{2}))                 \
            .as<decltype(int64_t{2} op int64_t{2})>() ==                       \
        (int64_t{2} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{2}) op DoubleInt64BoolVec(2.5))            \
            .as<decltype(int64_t{2} op 2.5)>(),                                \
        (int64_t{2} op 2.5));                                                  \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{2}) op DoubleInt64BoolVecTwo{})            \
            .as<decltype(int64_t{2} op int64_t{2})>(),                         \
        (int64_t{2} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(int64_t{2}))           \
            .as<decltype(int64_t{2} op int64_t{2})>(),                         \
        (int64_t{2} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op int64_t{2})                            \
            .as<decltype((int64_t{3} op int64_t{2}))>() ==                     \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op int64_t{2})                         \
            .as<decltype((int64_t{3} op int64_t{2}))>(),                       \
        (int64_t{3} op int64_t{2}));                                           \
    static_assert(                                                             \
        (int64_t{3} op DoubleInt64Bool(int64_t{2}))                            \
            .as<decltype((int64_t{3} op int64_t{2}))>() ==                     \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (int64_t{3} op DoubleInt64BoolVec(int64_t{2}))                         \
            .as<decltype((int64_t{3} op int64_t{2}))>(),                       \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(int64_t{2}); },           \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}) op             \
          DoubleInt64BoolVec(int64_t{2});                                      \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    static_assert(requires(IntSomeType a, IntSomeType b) { a op b; });         \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) op IntSomeType(SomeType{}); },         \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
  }

TEST_BINARY_OP_ALLTYPE(Add, +);
TEST_BINARY_OP_ALLTYPE(Minus, -);
TEST_BINARY_OP_ALLTYPE(Mul, *);
TEST_BINARY_OP_ALLTYPE(Div, /);
TEST_BINARY_OP_ALLTYPE(LogicalAnd, &&);
TEST_BINARY_OP_ALLTYPE(LogicalOr, ||);

#define TEST_COMPARE_OP(name, op)                                              \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(requires(DoubleInt64Bool a, DoubleInt64Bool b) { a op b; }); \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVec b) { a op b; });     \
    static_assert(requires(DoubleInt64Bool a, int b) { a op b; });             \
    static_assert(requires(DoubleInt64BoolVec a, int b) { a op b; });          \
    static_assert(                                                             \
        requires(DoubleInt64Bool a, DoubleInt64BoolTwo b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVecTwo b) { a op b; });  \
    static_assert(requires(int a, DoubleInt64Bool b) { a op b; });             \
    static_assert(requires(int a, DoubleInt64BoolVec b) { a op b; });          \
    static_assert(                                                             \
        requires(DoubleInt64BoolTwo a, DoubleInt64Bool b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVecTwo a, DoubleInt64BoolVec b) { a op b; });  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{2}) op DoubleInt64Bool(2.0)) ==               \
        (int64_t{2} op 2.0));                                                  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{2}) op DoubleInt64BoolTwo{}) ==               \
        (int64_t{2} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{1}) op DoubleInt64BoolTwo{}) ==               \
        (int64_t{1} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op DoubleInt64BoolTwo{}) ==               \
        (int64_t{3} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(int64_t{2})) ==              \
        (int64_t{2} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(int64_t{1})) ==              \
        (int64_t{2} op int64_t{1}));                                           \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(int64_t{3})) ==              \
        (int64_t{2} op int64_t{3}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{2}) op DoubleInt64BoolVec(2.0)),           \
        (int64_t{2} op 2.0));                                                  \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{2}) op DoubleInt64BoolVecTwo{}),           \
        (int64_t{2} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{1}) op DoubleInt64BoolVecTwo{}),           \
        (int64_t{1} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op DoubleInt64BoolVecTwo{}),           \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(int64_t{2})),          \
        (int64_t{2} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(int64_t{1})),          \
        (int64_t{2} op int64_t{1}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(int64_t{3})),          \
        (int64_t{2} op int64_t{3}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(std::vector<int64_t>{2})                           \
             op DoubleInt64BoolVec(std::vector<double>{2.0})),                 \
        (int64_t{2} op 2.0));                                                  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{2}) op DoubleInt64Bool(2.5)) ==               \
        (int64_t{2} op 2.5));                                                  \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{2}) op DoubleInt64BoolVec(2.5)),           \
        (int64_t{2} op 2.5));                                                  \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(std::vector<int64_t>{int64_t{2}})                  \
             op DoubleInt64BoolVec(std::vector<double>{2.5})),                 \
        (int64_t{2} op 2.5));                                                  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op int64_t{2}) ==                         \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op int64_t{2}),                        \
        (int64_t{3} op int64_t{2}));                                           \
    static_assert(                                                             \
        (int64_t{3} op DoubleInt64Bool(int64_t{2})) ==                         \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (int64_t{3} op DoubleInt64BoolVec(int64_t{2})),                        \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(int64_t{2}); },           \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}) op             \
          DoubleInt64BoolVec(int64_t{2});                                      \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    static_assert(requires(IntSomeType a, IntSomeType b) { a op b; });         \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) op IntSomeType(SomeType{}); },         \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
  }

TEST_COMPARE_OP(Eq, ==);
TEST_COMPARE_OP(Ne, !=);
TEST_COMPARE_OP(Lt, <);
TEST_COMPARE_OP(Gt, >);
TEST_COMPARE_OP(Le, <=);
TEST_COMPARE_OP(Ge, >=);

#define TEST_NAMED_COMPARE_OP(name, op, func)                              \
  TEST_F(DynamicTypeTest, name) {                                          \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{2}), DoubleInt64Bool(2.0)) ==         \
        (int64_t{2} op 2.0));                                              \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{2}), DoubleInt64BoolTwo{}) ==         \
        (int64_t{2} op int64_t{2}));                                       \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{1}), DoubleInt64BoolTwo{}) ==         \
        (int64_t{1} op int64_t{2}));                                       \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{3}), DoubleInt64BoolTwo{}) ==         \
        (int64_t{3} op int64_t{2}));                                       \
    static_assert(                                                         \
        func(DoubleInt64BoolTwo{}, DoubleInt64Bool(int64_t{2})) ==         \
        (int64_t{2} op int64_t{2}));                                       \
    static_assert(                                                         \
        func(DoubleInt64BoolTwo{}, DoubleInt64Bool(int64_t{1})) ==         \
        (int64_t{2} op int64_t{1}));                                       \
    static_assert(                                                         \
        func(DoubleInt64BoolTwo{}, DoubleInt64Bool(int64_t{3})) ==         \
        (int64_t{2} op int64_t{3}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{2}), DoubleInt64BoolVec(2.0)),     \
        (int64_t{2} op 2.0));                                              \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{2}), DoubleInt64BoolVecTwo{}),     \
        (int64_t{2} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{1}), DoubleInt64BoolVecTwo{}),     \
        (int64_t{1} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{3}), DoubleInt64BoolVecTwo{}),     \
        (int64_t{3} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVecTwo{}, DoubleInt64BoolVec(int64_t{2})),     \
        (int64_t{2} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVecTwo{}, DoubleInt64BoolVec(int64_t{1})),     \
        (int64_t{2} op int64_t{1}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVecTwo{}, DoubleInt64BoolVec(int64_t{3})),     \
        (int64_t{2} op int64_t{3}));                                       \
    EXPECT_EQ(                                                             \
        func(                                                              \
            DoubleInt64BoolVec(std::vector<int64_t>{2}),                   \
            DoubleInt64BoolVec(std::vector<double>{2.0})),                 \
        (int64_t{2} op 2.0));                                              \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{2}), DoubleInt64Bool(2.5)) ==         \
        (int64_t{2} op 2.5));                                              \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{2}), DoubleInt64BoolVec(2.5)),     \
        (int64_t{2} op 2.5));                                              \
    EXPECT_EQ(                                                             \
        func(                                                              \
            DoubleInt64BoolVec(std::vector<int64_t>{int64_t{2}}),          \
            DoubleInt64BoolVec(std::vector<double>{2.5})),                 \
        (int64_t{2} op 2.5));                                              \
    static_assert(                                                         \
        func(DoubleInt64Bool(int64_t{3}), int64_t{2}) ==                   \
        (int64_t{3} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(DoubleInt64BoolVec(int64_t{3}), int64_t{2}),                  \
        (int64_t{3} op int64_t{2}));                                       \
    static_assert(                                                         \
        func(int64_t{3}, DoubleInt64Bool(int64_t{2})) ==                   \
        (int64_t{3} op int64_t{2}));                                       \
    EXPECT_EQ(                                                             \
        func(int64_t{3}, DoubleInt64BoolVec(int64_t{2})),                  \
        (int64_t{3} op int64_t{2}));                                       \
    EXPECT_THAT(                                                           \
        [&]() { func(DoubleInt64Bool(), DoubleInt64Bool(int64_t{2})); },   \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr( \
            "Result is dynamic but not convertible to result type")));     \
    EXPECT_THAT(                                                           \
        [&]() {                                                            \
          func(                                                            \
              DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}),       \
              DoubleInt64BoolVec(int64_t{2}));                             \
        },                                                                 \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr( \
            "Result is dynamic but not convertible to result type")));     \
    EXPECT_THAT(                                                           \
        [&]() { func(IntSomeType(SomeType{}), IntSomeType(SomeType{})); }, \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr( \
            "Result is dynamic but not convertible to result type")));     \
  }

TEST_NAMED_COMPARE_OP(NamedEq, ==, eq);
TEST_NAMED_COMPARE_OP(NamedNe, !=, ne);
TEST_NAMED_COMPARE_OP(NamedLt, <, lt);
TEST_NAMED_COMPARE_OP(NamedGt, >, gt);
TEST_NAMED_COMPARE_OP(NamedLe, <=, le);
TEST_NAMED_COMPARE_OP(NamedGe, >=, ge);

#define TEST_BINARY_OP_INT_ONLY(name, op)                                      \
  TEST_F(DynamicTypeTest, name) {                                              \
    static_assert(requires(DoubleInt64Bool a, DoubleInt64Bool b) { a op b; }); \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVec b) { a op b; });     \
    static_assert(requires(DoubleInt64Bool a, int64_t b) { a op b; });         \
    static_assert(requires(DoubleInt64BoolVec a, int64_t b) { a op b; });      \
    static_assert(                                                             \
        requires(DoubleInt64Bool a, DoubleInt64BoolTwo b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVec a, DoubleInt64BoolVecTwo b) { a op b; });  \
    static_assert(requires(int64_t a, DoubleInt64Bool b) { a op b; });         \
    static_assert(requires(int64_t a, DoubleInt64BoolVec b) { a op b; });      \
    static_assert(                                                             \
        requires(DoubleInt64BoolTwo a, DoubleInt64Bool b) { a op b; });        \
    static_assert(                                                             \
        requires(DoubleInt64BoolVecTwo a, DoubleInt64BoolVec b) { a op b; });  \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op DoubleInt64Bool(int64_t{2}))           \
            .as<int64_t>() == (int64_t{3} op int64_t{2}));                     \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op DoubleInt64BoolTwo{})                  \
            .as<decltype(int64_t{3} op int64_t{2})>() ==                       \
        (int64_t{3} op int64_t{2}));                                           \
    static_assert(                                                             \
        (DoubleInt64BoolTwo {} op DoubleInt64Bool(int64_t{3}))                 \
            .as<decltype(int64_t{2} op int64_t{3})>() ==                       \
        (int64_t{2} op int64_t{3}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op DoubleInt64BoolVec(int64_t{2}))     \
            .as<int64_t>(),                                                    \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op DoubleInt64BoolVecTwo{})            \
            .as<decltype(int64_t{3} op int64_t{2})>(),                         \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVecTwo {} op DoubleInt64BoolVec(int64_t{3}))           \
            .as<decltype(int64_t{2} op int64_t{3})>(),                         \
        (int64_t{2} op int64_t{3}));                                           \
    static_assert(                                                             \
        (DoubleInt64Bool(int64_t{3}) op int64_t{2}).as<int64_t>() ==           \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (DoubleInt64BoolVec(int64_t{3}) op int64_t{2}).as<int64_t>(),          \
        (int64_t{3} op int64_t{2}));                                           \
    static_assert(                                                             \
        (int64_t{3} op DoubleInt64Bool(int64_t{2})).as<int64_t>() ==           \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_EQ(                                                                 \
        (int64_t{3} op DoubleInt64BoolVec(int64_t{2})).as<int64_t>(),          \
        (int64_t{3} op int64_t{2}));                                           \
    EXPECT_THAT(                                                               \
        [&]() { DoubleInt64Bool() op DoubleInt64Bool(int64_t{2}); },           \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    EXPECT_THAT(                                                               \
        [&]() {                                                                \
          DoubleInt64BoolVec(std::vector<DoubleInt64BoolVec>{}) op             \
          DoubleInt64BoolVec(int64_t{2});                                      \
        },                                                                     \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
    static_assert(requires(IntSomeType a, IntSomeType b) { a + b; });          \
    EXPECT_THAT(                                                               \
        [&]() { IntSomeType(SomeType{}) + IntSomeType(SomeType{}); },          \
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(     \
            "Result is dynamic but not convertible to result type")));         \
  }

TEST_BINARY_OP_INT_ONLY(Mod, %);
TEST_BINARY_OP_INT_ONLY(BinaryAnd, &);
TEST_BINARY_OP_INT_ONLY(BinaryOr, |);
TEST_BINARY_OP_INT_ONLY(Xor, ^);
TEST_BINARY_OP_INT_ONLY(LShift, <<);
TEST_BINARY_OP_INT_ONLY(RShift, >>);

TEST_F(DynamicTypeTest, BinaryOpAdvancedTyping) {
  static_assert(!HasPlus<AdvancedType2SomeType, AdvancedType2SomeType>);
  static_assert(!HasPlus<AdvancedType2SomeType, AdvancedType2>);
  static_assert(!HasPlus<AdvancedType2, AdvancedType2SomeType>);

  // defined compile time because Type2+Type2 is constructible to Type3
  static_assert(
      requires(AdvancedType2Type3 a, AdvancedType2Type3 b) { a + b; });
  static_assert(
      AdvancedType2Type3(AdvancedType2{}) +
          AdvancedType2Type3(AdvancedType2{}) ==
      AdvancedType3{});
  static_assert(requires(AdvancedType2Type3 a, AdvancedType2 b) { a + b; });
  static_assert(
      AdvancedType2Type3(AdvancedType2{}) + AdvancedType2{} == AdvancedType3{});
  static_assert(requires(AdvancedType2 a, AdvancedType2Type3 b) { a + b; });
  static_assert(
      AdvancedType2{} + AdvancedType2Type3(AdvancedType2{}) == AdvancedType3{});
  // defined compile time because int+int is in type list
  static_assert(requires(AdvancedType2Int a, AdvancedType2Int b) { a + b; });
  // runtime error because Type2+Type2 is not in type list
  auto bad = [&]() {
    AdvancedType2Int x(AdvancedType2{});
    x + x;
  };
  EXPECT_THAT(
      bad,
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
  // test bool to int conversion
  using Int = DynamicType<NoContainers, int>;
  static_assert((Int(2) && Int(0)) == 0);
  static_assert((Int(2) && Int(3)) == 1);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
