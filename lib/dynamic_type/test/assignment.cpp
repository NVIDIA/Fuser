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
