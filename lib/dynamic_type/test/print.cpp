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

#include <sstream>
#include <string>

#include "utils.h"

using namespace dynamic_type;

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
