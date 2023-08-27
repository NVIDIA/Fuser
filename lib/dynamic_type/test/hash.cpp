// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "dynamic_type.h"

using namespace dynamic_type;

using DoubleInt64Bool = DynamicType<NoContainers, double, int64_t, bool>;

class DynamicTypeTest : public ::testing::Test {};

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

template <>
struct std::hash<DoubleInt64Bool> {
  size_t operator()(const DoubleInt64Bool& x) const {
    return 0;
  }
};

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
