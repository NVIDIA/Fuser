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

#include <string>

using namespace dynamic_type;

class DynamicTypeTest : public ::testing::Test {};

TEST_F(DynamicTypeTest, DispatchBasic) {
  using IntOrStr = DynamicType<NoContainers, int64_t, std::string>;
  auto concat = [](std::string a, auto x, auto y, std::string b) {
    std::string result = a;
    if constexpr (std::is_same_v<decltype(x), int64_t>) {
      result += std::to_string(x);
    } else if constexpr (std::is_same_v<decltype(x), std::string>) {
      result += x;
    }
    if constexpr (std::is_same_v<decltype(y), int64_t>) {
      result += std::to_string(y);
    } else if constexpr (std::is_same_v<decltype(y), std::string>) {
      result += y;
    }
    return result + b;
  };

  IntOrStr one(std::string("1"));
  IntOrStr two(2);
  EXPECT_TRUE(one.is<std::string>());
  EXPECT_TRUE(two.is<int64_t>());

  auto concated_result11 = IntOrStr::dispatch(concat, "0", one, one, "3");
  static_assert(std::is_same_v<decltype(concated_result11), std::string>);
  EXPECT_EQ(concated_result11, "0113");

  auto concated_result12 = IntOrStr::dispatch(concat, "0", one, two, "3");
  static_assert(std::is_same_v<decltype(concated_result12), std::string>);
  EXPECT_EQ(concated_result12, "0123");

  auto concated_result21 = IntOrStr::dispatch(concat, "0", two, one, "3");
  static_assert(std::is_same_v<decltype(concated_result21), std::string>);
  EXPECT_EQ(concated_result21, "0213");

  auto concated_result22 = IntOrStr::dispatch(concat, "0", two, two, "3");
  static_assert(std::is_same_v<decltype(concated_result22), std::string>);
  EXPECT_EQ(concated_result22, "0223");
}

TEST_F(DynamicTypeTest, DispatchArgumentPerfectForwarding) {
  using IntOrStr = DynamicType<NoContainers, int64_t, std::string>;
  struct NonCopyable {
    NonCopyable() = default;
    NonCopyable(const NonCopyable&) = delete;
  };
  auto concat =
      [](std::string& result, NonCopyable&& nc, auto x, auto y, std::string b) {
        if constexpr (std::is_same_v<decltype(x), int64_t>) {
          result += std::to_string(x);
        } else if constexpr (std::is_same_v<decltype(x), std::string>) {
          result += x;
        }
        if constexpr (std::is_same_v<decltype(y), int64_t>) {
          result += std::to_string(y);
        } else if constexpr (std::is_same_v<decltype(y), std::string>) {
          result += y;
        }
        result += b;
      };

  IntOrStr one(std::string("1"));
  IntOrStr two(2);
  EXPECT_TRUE(one.is<std::string>());
  EXPECT_TRUE(two.is<int64_t>());

  std::string concated_result11;
  IntOrStr::dispatch(concat, concated_result11, NonCopyable{}, one, one, "3");
  static_assert(std::is_same_v<decltype(concated_result11), std::string>);
  EXPECT_EQ(concated_result11, "113");

  std::string concated_result12;
  IntOrStr::dispatch(concat, concated_result12, NonCopyable{}, one, two, "3");
  static_assert(std::is_same_v<decltype(concated_result12), std::string>);
  EXPECT_EQ(concated_result12, "123");

  std::string concated_result21;
  IntOrStr::dispatch(concat, concated_result21, NonCopyable{}, two, one, "3");
  static_assert(std::is_same_v<decltype(concated_result21), std::string>);
  EXPECT_EQ(concated_result21, "213");

  std::string concated_result22;
  IntOrStr::dispatch(concat, concated_result22, NonCopyable{}, two, two, "3");
  static_assert(std::is_same_v<decltype(concated_result22), std::string>);
  EXPECT_EQ(concated_result22, "223");
}

TEST_F(DynamicTypeTest, DispatchReturnsDynamicType) {
  using IntOrFloat = DynamicType<NoContainers, int64_t, float>;
  auto add = [](auto x, auto y) {
    if constexpr (opcheck<decltype(x)> + opcheck<decltype(y)>) {
      return x + y;
    } else {
      return;
    }
  };
  constexpr IntOrFloat one(1);
  constexpr IntOrFloat two(2.0f);
  static_assert(one.is<int64_t>());
  static_assert(two.is<float>());

  auto r11 = IntOrFloat::dispatch(add, one, one);
  static_assert(std::is_same_v<decltype(r11), IntOrFloat>);
  EXPECT_TRUE(r11.is<int64_t>());
  EXPECT_EQ(r11, 2);

  constexpr auto ce_r11 = IntOrFloat::dispatch(add, one, one);
  static_assert(std::is_same_v<decltype(ce_r11), const IntOrFloat>);
  static_assert(ce_r11.is<int64_t>());
  static_assert(ce_r11 == 2);

  auto r12 = IntOrFloat::dispatch(add, one, two);
  static_assert(std::is_same_v<decltype(r12), IntOrFloat>);
  EXPECT_TRUE(r12.is<float>());
  EXPECT_EQ(r12, 3.0f);

  constexpr auto ce_r12 = IntOrFloat::dispatch(add, one, two);
  static_assert(std::is_same_v<decltype(ce_r12), const IntOrFloat>);
  static_assert(ce_r12.is<float>());
  static_assert(ce_r12 == 3.0f);
}

TEST_F(DynamicTypeTest, DispatchReturnsReference) {
  using IntOrFloat = DynamicType<NoContainers, int64_t, float>;
  auto add = [](std::vector<int>& a, auto x, int64_t y) -> decltype(auto) {
    if constexpr (std::is_integral_v<decltype(x)>) {
      return a[x + y];
    } else {
      return;
    }
  };
  constexpr IntOrFloat one(1);
  constexpr IntOrFloat two(2.0f);
  static_assert(one.is<int64_t>());
  static_assert(two.is<float>());

  std::vector<int> a = {0, 1, 2, 3};

  auto& r1 = IntOrFloat::dispatch(add, a, one, 1);
  static_assert(std::is_same_v<decltype(r1), int&>);
  EXPECT_EQ(r1, 2);
  EXPECT_EQ(&r1, &a[2]);

  EXPECT_THAT(
      [&]() { IntOrFloat::dispatch(add, a, two, 1); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Result is dynamic but not convertible to result type")));
}
