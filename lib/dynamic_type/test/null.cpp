// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gtest/gtest.h>

#include "dynamic_type/dynamic_type.h"

#include "utils.h"

using namespace dynamic_type;

// Using local definition since utils.h's DoubleInt64Bool has NonInstantiable
using LocalDoubleInt64Bool = DynamicType<NoContainers, double, int64_t, bool>;

TEST_F(DynamicTypeTest, NullComparisons) {
  // Use local type to avoid dependency on NonInstantiable from utils.h
  using DoubleInt64Bool = LocalDoubleInt64Bool;
  DoubleInt64Bool a, b;
  EXPECT_TRUE(a.isNull());
  EXPECT_FALSE(a.hasValue());
  EXPECT_TRUE(b.isNull());
  EXPECT_FALSE(b.hasValue());
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);
  EXPECT_FALSE(a != b);
  EXPECT_FALSE(b != a);
  EXPECT_FALSE(a < b);
  EXPECT_FALSE(b < a);
  EXPECT_FALSE(a > b);
  EXPECT_FALSE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b <= a);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a == std::monostate{});
  EXPECT_TRUE(std::monostate{} == a);
  EXPECT_FALSE(a != std::monostate{});
  EXPECT_FALSE(std::monostate{} != a);
  EXPECT_FALSE(a < std::monostate{});
  EXPECT_FALSE(std::monostate{} < a);
  EXPECT_FALSE(a > std::monostate{});
  EXPECT_FALSE(std::monostate{} > a);
  EXPECT_TRUE(a <= std::monostate{});
  EXPECT_TRUE(std::monostate{} <= a);
  EXPECT_TRUE(a >= std::monostate{});
  EXPECT_TRUE(std::monostate{} >= a);
}
