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

#include <list>
#include <vector>

using namespace dynamic_type;

// This is the test for the examples in the README.md, if you updated that note,
// please update this test as well. On the other hand, if you have to do
// something that breaks this test, please update the note as well.

class Examples : public ::testing::Test {};

namespace example_1 {

using IntOrFloat = DynamicType<NoContainers, int, float>;
constexpr IntOrFloat x = 1;
constexpr IntOrFloat y = 2.5f;
constexpr IntOrFloat z = x + y;
static_assert(z.as<float>() == 3.5f);

} // namespace example_1

TEST_F(Examples, Example2) {
  struct CustomType {};
  using IntOrFloatOrCustom = DynamicType<NoContainers, int, float, CustomType>;
  constexpr IntOrFloatOrCustom i = 1;
  constexpr IntOrFloatOrCustom f = 2.5f;
  constexpr IntOrFloatOrCustom c = CustomType{};
  constexpr IntOrFloatOrCustom null;
  static_assert(i + i == 2);
  static_assert(i + f == 3.5f);
  static_assert(f + i == 3.5f);
  static_assert(f + f == 5.0f);
  EXPECT_THAT(
      [&]() { i + null; },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot compute ")));
  EXPECT_THAT(
      [&]() { i + c; },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot compute ")));
}

namespace example_3 {

struct CustomType {};
struct CustomType2 {};
using Custom12 = DynamicType<NoContainers, CustomType, CustomType2>;
static_assert(!(opcheck<Custom12> + opcheck<Custom12>));

} // namespace example_3

struct bfloat16_zero {};
struct half_zero {};
float operator+(bfloat16_zero, half_zero) {
  return 0.0f;
}

TEST_F(Examples, Example4) {
  using BFloatOrHalfZero = DynamicType<NoContainers, bfloat16_zero, half_zero>;
  static_assert(!(opcheck<BFloatOrHalfZero> + opcheck<BFloatOrHalfZero>));
  using BFloatOrHalfZeroOrInt =
      DynamicType<NoContainers, bfloat16_zero, half_zero, int>;
  static_assert(
      opcheck<BFloatOrHalfZeroOrInt> + opcheck<BFloatOrHalfZeroOrInt>);
  EXPECT_THAT(
      [&]() {
        BFloatOrHalfZeroOrInt(half_zero{}) +
            BFloatOrHalfZeroOrInt(bfloat16_zero{});
      },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot compute ")));
}

namespace example_5 {

using IntOrFloat = DynamicType<NoContainers, int, float>;
constexpr IntOrFloat x = 1;
constexpr float y = 2.5f;
static_assert(std::is_same_v<decltype(x + y), IntOrFloat>);
static_assert((x + y).as<float>() == 3.5f);
static_assert(std::is_same_v<decltype(y + x), IntOrFloat>);
static_assert((y + x).as<float>() == 3.5f);
static_assert(!(opcheck<IntOrFloat> + opcheck<double>));
static_assert(!(opcheck<double> + opcheck<IntOrFloat>));

} // namespace example_5

TEST_F(Examples, Example6) {
  using IntFloatVecList =
      DynamicType<Containers<std::vector, std::list>, int, float>;
  IntFloatVecList x = std::vector<IntFloatVecList>{1, 2.0f};
  IntFloatVecList y = std::list<IntFloatVecList>{3, x};
  EXPECT_TRUE(y.is<std::list>());
  EXPECT_EQ(y.as<std::list>().size(), 2);
  EXPECT_EQ(y.as<std::list>().front().as<int>(), 3);
  EXPECT_TRUE(y.as<std::list>().back().is<std::vector>());
  EXPECT_EQ(y.as<std::list>().back().as<std::vector>().size(), 2);
  EXPECT_EQ(y.as<std::list>().back()[0], 1);
  EXPECT_EQ(y.as<std::list>().back()[1], 2.0f);
  EXPECT_THAT(
      // std::list can not be indexed
      [&]() { y[0]; },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot index ")));
}
