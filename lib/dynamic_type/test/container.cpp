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

#include <unordered_set>
#include <vector>

#include "utils.h"

TEST_F(DynamicTypeTest, FromContainerToContainer) {
  using IntOrVec = DynamicType<Containers<std::vector>, int>;
  using Vec = DynamicType<Containers<std::vector>>;

  static_assert(std::is_constructible_v<IntOrVec, std::vector<int>>);
  static_assert(
      std::is_constructible_v<IntOrVec, std::vector<std::vector<int>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<int>>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<std::vector<int>>>>>);

  static_assert(opcheck<IntOrVec>.canCastTo(opcheck<std::vector<double>>));
  static_assert(
      opcheck<IntOrVec>.canCastTo(opcheck<std::vector<std::vector<double>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<double>>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<std::vector<double>>>>>));

  static_assert(opcheck<IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<Vec>]);
  static_assert(opcheck<IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<const Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<const Vec>]);

  IntOrVec zero = 0;
  IntOrVec one = 1;
  IntOrVec two = 2;

  std::vector<std::vector<int>> vvi{{1, 2, 3}, {4, 5, 6}};
  IntOrVec x = vvi;
  EXPECT_EQ(x[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[0][0], 1);
  EXPECT_EQ(x[0][1], 2);
  EXPECT_EQ(x[0][2], 3);
  EXPECT_EQ(x[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[1][0], 4);
  EXPECT_EQ(x[1][1], 5);
  EXPECT_EQ(x[1][2], 6);

  EXPECT_EQ(x[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[zero][zero], 1);
  EXPECT_EQ(x[zero][one], 2);
  EXPECT_EQ(x[zero][two], 3);
  EXPECT_EQ(x[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[one][zero], 4);
  EXPECT_EQ(x[one][one], 5);
  EXPECT_EQ(x[one][two], 6);

  const IntOrVec xx = vvi;
  EXPECT_EQ(xx[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[0][0], 1);
  EXPECT_EQ(xx[0][1], 2);
  EXPECT_EQ(xx[0][2], 3);
  EXPECT_EQ(xx[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[1][0], 4);
  EXPECT_EQ(xx[1][1], 5);
  EXPECT_EQ(xx[1][2], 6);

  EXPECT_EQ(xx[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[zero][zero], 1);
  EXPECT_EQ(xx[zero][one], 2);
  EXPECT_EQ(xx[zero][two], 3);
  EXPECT_EQ(xx[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[one][zero], 4);
  EXPECT_EQ(xx[one][one], 5);
  EXPECT_EQ(xx[one][two], 6);

  std::vector<std::vector<double>> vvd{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ((std::vector<std::vector<double>>)x, vvd);
  EXPECT_EQ((std::vector<double>)x[0], vvd[0]);
  EXPECT_EQ((std::vector<double>)x[1], vvd[1]);
}

#if defined(__GLIBCXX__) && __GLIBCXX__ >= 20230714
#define STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE 1
#endif

// Testing containers support by implementing the set-theoretic definition of
// natural numbers:
// https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers

// TODO: unordered set is a better fit for this case, but it does not work with
// some old compilers (for example the old gcc on our CI). This is a workaround.
// See [Incomplete type support in STL] for more details.
#if defined(STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE)

struct StupidHash {
  template <typename T>
  size_t operator()(const T&) const {
    // This hash always collides, but who cares?
    return 0;
  }
};

template <typename T>
using Set = std::unordered_set<T, StupidHash>;

#else

template <typename T>
using Set = std::vector<T>;
#define insert push_back

#endif

using NaturalNumber = DynamicType<Containers<Set>>;

TEST_F(DynamicTypeTest, SetTheoreticNaturalNumbers) {
  auto next = [](const NaturalNumber& n) {
    // recursively define natural number n + 1 as n U {n}
    auto set = n.as<Set>();
    set.insert(n);
    return NaturalNumber(set);
  };

  NaturalNumber zero = Set<NaturalNumber>{};
  NaturalNumber one = next(zero);
  NaturalNumber two = next(one);
  NaturalNumber three = next(two);
  NaturalNumber four = next(three);
  NaturalNumber five = next(four);
  NaturalNumber six = next(five);
  NaturalNumber seven = next(six);
  NaturalNumber eight = next(seven);
  NaturalNumber nine = next(eight);
  NaturalNumber ten = next(nine);

  EXPECT_TRUE(zero.is<Set>());
  EXPECT_TRUE(one.is<Set>());
  EXPECT_TRUE(two.is<Set>());
  EXPECT_TRUE(three.is<Set>());
  EXPECT_TRUE(four.is<Set>());
  EXPECT_TRUE(five.is<Set>());
  EXPECT_TRUE(six.is<Set>());
  EXPECT_TRUE(seven.is<Set>());
  EXPECT_TRUE(eight.is<Set>());
  EXPECT_TRUE(nine.is<Set>());
  EXPECT_TRUE(ten.is<Set>());

  EXPECT_EQ(zero.as<Set>().size(), 0);
  EXPECT_EQ(one.as<Set>().size(), 1);
  EXPECT_EQ(two.as<Set>().size(), 2);
  EXPECT_EQ(three.as<Set>().size(), 3);
  EXPECT_EQ(four.as<Set>().size(), 4);
  EXPECT_EQ(five.as<Set>().size(), 5);
  EXPECT_EQ(six.as<Set>().size(), 6);
  EXPECT_EQ(seven.as<Set>().size(), 7);
  EXPECT_EQ(eight.as<Set>().size(), 8);
  EXPECT_EQ(nine.as<Set>().size(), 9);
  EXPECT_EQ(ten.as<Set>().size(), 10);

  EXPECT_EQ(zero, NaturalNumber(Set<NaturalNumber>{}));
  EXPECT_EQ(one, NaturalNumber(Set<NaturalNumber>{zero}));
  EXPECT_EQ(two, NaturalNumber(Set<NaturalNumber>{zero, one}));
  EXPECT_EQ(three, NaturalNumber(Set<NaturalNumber>{zero, one, two}));
  EXPECT_EQ(four, NaturalNumber(Set<NaturalNumber>{zero, one, two, three}));
  EXPECT_EQ(
      five, NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four}));
  EXPECT_EQ(
      six,
      NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four, five}));
  EXPECT_EQ(
      seven,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six}));
  EXPECT_EQ(
      eight,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six, seven}));
  EXPECT_EQ(
      nine,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight}));
  EXPECT_EQ(
      ten,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight, nine}));
}

#undef insert
