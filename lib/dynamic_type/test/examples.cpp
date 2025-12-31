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

#include <list>
#include <vector>

using namespace dynamic_type;

// This is the test for the examples in the README.md, if you updated that note,
// please update this test as well. On the other hand, if you have to do
// something that breaks this test, please update the note as well.

class Examples : public ::testing::Test {};

TEST_F(Examples, Example1) {
  using IntOrFloat = DynamicType<NoContainers, int, float>;
  IntOrFloat x = 1;
  IntOrFloat y = 2.5f;
  IntOrFloat z = x + y;
  EXPECT_EQ(z.as<float>(), 3.5f);
}

TEST_F(Examples, Example2) {
  struct CustomType {};
  using IntOrFloatOrCustom = DynamicType<NoContainers, int, float, CustomType>;
  IntOrFloatOrCustom i = 1;
  IntOrFloatOrCustom f = 2.5f;
  IntOrFloatOrCustom c = CustomType{};
  IntOrFloatOrCustom null;
  EXPECT_EQ(i + i, 2);
  EXPECT_EQ(i + f, 3.5f);
  EXPECT_EQ(f + i, 3.5f);
  EXPECT_EQ(f + f, 5.0f);
  EXPECT_THAT(
      [&]() { i + null; },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Cannot compute")));
  EXPECT_THAT(
      [&]() { i + c; },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Cannot compute")));
}

TEST_F(Examples, Example3) {
  // Test that operations on unsupported types fail at runtime
  struct CustomType {};
  struct CustomType2 {};
  using Custom12 = DynamicType<NoContainers, CustomType, CustomType2>;
  // With friend operators, the operator is always defined at compile time
  // but will fail at runtime for unsupported type combinations
  EXPECT_THAT(
      []() { Custom12(CustomType{}) + Custom12(CustomType2{}); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Cannot compute")));
}

struct bfloat16_zero {};
struct half_zero {};
float operator+(bfloat16_zero, half_zero) {
  return 0.0f;
}

TEST_F(Examples, Example4) {
  using BFloatOrHalfZero = DynamicType<NoContainers, bfloat16_zero, half_zero>;
  // With friend operators, the operator is always defined at compile time
  // Runtime error for unsupported type combinations
  EXPECT_THAT(
      []() { BFloatOrHalfZero(bfloat16_zero{}) + BFloatOrHalfZero(half_zero{}); },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Cannot compute")));
  using BFloatOrHalfZeroOrInt =
      DynamicType<NoContainers, bfloat16_zero, half_zero, int>;
  static_assert(
      opcheck<BFloatOrHalfZeroOrInt> + opcheck<BFloatOrHalfZeroOrInt>);
  EXPECT_THAT(
      [&]() {
        BFloatOrHalfZeroOrInt(half_zero{}) +
            BFloatOrHalfZeroOrInt(bfloat16_zero{});
      },
      ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
          "Cannot compute")));
}

TEST_F(Examples, Example5) {
  using IntOrFloat = DynamicType<NoContainers, int, float>;
  IntOrFloat x = 1;
  float y = 2.5f;
  static_assert(std::is_same_v<decltype(x + y), IntOrFloat>);
  EXPECT_EQ((x + y).as<float>(), 3.5f);
  static_assert(std::is_same_v<decltype(y + x), IntOrFloat>);
  EXPECT_EQ((y + x).as<float>(), 3.5f);
  static_assert(!(opcheck<IntOrFloat> + opcheck<double>));
  static_assert(!(opcheck<double> + opcheck<IntOrFloat>));
}

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

TEST_F(Examples, Example7) {
  using IntFloatVec = DynamicType<Containers<std::vector>, int, float>;
  IntFloatVec x = std::vector<int>{1, 2};
  IntFloatVec y = std::vector<std::vector<int>>{{1, 2}, {3, 4}};
  EXPECT_TRUE(x.is<std::vector>());
  EXPECT_EQ(x.as<std::vector>().size(), 2);
  EXPECT_EQ(x.as<std::vector>()[0], 1);
  EXPECT_EQ(x.as<std::vector>()[1], 2);
  EXPECT_TRUE(y.is<std::vector>());
  EXPECT_EQ(y.as<std::vector>().size(), 2);
  EXPECT_TRUE(y.as<std::vector>()[0].is<std::vector>());
  EXPECT_EQ(y.as<std::vector>()[0].as<std::vector>().size(), 2);
  EXPECT_EQ(y.as<std::vector>()[0].as<std::vector>()[0], 1);
  EXPECT_EQ(y.as<std::vector>()[0].as<std::vector>()[1], 2);
  EXPECT_TRUE(y.as<std::vector>()[1].is<std::vector>());
  EXPECT_EQ(y.as<std::vector>()[1].as<std::vector>().size(), 2);
  EXPECT_EQ(y.as<std::vector>()[1].as<std::vector>()[0], 3);
  EXPECT_EQ(y.as<std::vector>()[1].as<std::vector>()[1], 4);
}

TEST_F(Examples, Example8) {
  using IntFloatVec = DynamicType<Containers<std::vector>, int, float>;

  IntFloatVec x = std::vector<IntFloatVec>{1, 2.3f};
  auto y = (std::vector<int>)x;
  EXPECT_EQ(y.size(), 2);
  EXPECT_EQ(y[0], 1);
  EXPECT_EQ(y[1], 2);

  IntFloatVec z = std::vector<std::vector<IntFloatVec>>{{1, 2.3f}, {3.4f, 5}};
  auto q = (std::vector<std::vector<int>>)z;
  EXPECT_EQ(q.size(), 2);
  EXPECT_EQ(q[0].size(), 2);
  EXPECT_EQ(q[0][0], 1);
  EXPECT_EQ(q[0][1], 2);
  EXPECT_EQ(q[1].size(), 2);
  EXPECT_EQ(q[1][0], 3);
  EXPECT_EQ(q[1][1], 5);
}

TEST_F(Examples, Example9) {
  using IntFloatVec = DynamicType<Containers<std::vector>, int, float>;
  IntFloatVec x = std::vector<IntFloatVec>{1, 2.3f};
  EXPECT_EQ(x[0], 1);
  EXPECT_EQ(x[1], 2.3f);
}

TEST_F(Examples, Example10) {
  struct A {
    int x;
    std::string name() const {
      return "A";
    }
  };

  struct B {
    double y;
    std::string name() const {
      return "B";
    }
  };

  using AB = DynamicType<NoContainers, A, B>;
  AB a = A{1};
  EXPECT_EQ(a->*&A::x, 1);
  EXPECT_EQ((a->*&A::name)(), "A");
  AB b = B{2.5};
  EXPECT_EQ(b->*&B::y, 2.5);
  EXPECT_EQ((b->*&B::name)(), "B");
}

TEST_F(Examples, Example11) {
  using IntDoubleVec = DynamicType<Containers<std::vector>, int, double>;
  auto get_size = [](auto x) { return sizeof(x); };
  IntDoubleVec mydata1 = 3.0;
  EXPECT_EQ(IntDoubleVec::dispatch(get_size, mydata1), 8);
  IntDoubleVec mydata2 = 2;
  EXPECT_EQ(IntDoubleVec::dispatch(get_size, mydata2), 4);

  auto get_total_size = [](auto x, size_t num_x, auto y, size_t num_y) {
    return sizeof(x) * num_x + sizeof(y) * num_y;
  };
  EXPECT_EQ(IntDoubleVec::dispatch(get_total_size, mydata1, 3, mydata2, 5), 44);

  auto my_pow = [](auto x, auto exp) {
    if constexpr (
        std::is_arithmetic_v<decltype(x)> &&
        std::is_arithmetic_v<decltype(exp)>) {
      if constexpr (std::is_integral_v<decltype(exp)>) {
        decltype(x) result = 1;
        while (exp-- > 0) {
          result *= x;
        }
        return result;
      } else {
        return std::pow(x, exp);
      }
    } else {
      throw std::runtime_error("Unsupported type");
      return;
    }
  };
  auto r11 = IntDoubleVec::dispatch(my_pow, mydata1, mydata1);
  static_assert(std::is_same_v<decltype(r11), IntDoubleVec>);
  EXPECT_TRUE(r11.is<double>());
  EXPECT_EQ(r11, 27.0);
  auto r12 = IntDoubleVec::dispatch(my_pow, mydata1, mydata2);
  static_assert(std::is_same_v<decltype(r12), IntDoubleVec>);
  EXPECT_TRUE(r12.is<double>());
  EXPECT_EQ(r12, 9.0);
  auto r21 = IntDoubleVec::dispatch(my_pow, mydata2, mydata1);
  static_assert(std::is_same_v<decltype(r21), IntDoubleVec>);
  EXPECT_TRUE(r21.is<double>());
  EXPECT_EQ(r21, 8.0);
  auto r22 = IntDoubleVec::dispatch(my_pow, mydata2, mydata2);
  static_assert(std::is_same_v<decltype(r22), IntDoubleVec>);
  EXPECT_TRUE(r22.is<int>());
  EXPECT_EQ(r22, 4);

  std::vector<float> vec = {0.0, 1.0, 2.0, 3.0};
  auto get_item = [](auto& v, auto index) -> decltype(auto) {
    if constexpr (std::is_integral_v<decltype(index)>) {
      return v[index];
    } else {
      throw std::runtime_error("Illegal index type");
      return;
    }
  };
  IntDoubleVec::dispatch(get_item, vec, mydata2) = 100.0;
  std::vector<float> expect{0.0, 1.0, 100.0, 3.0};
  EXPECT_EQ(vec, expect);
}
