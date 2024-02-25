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

#include <list>
#include <vector>

// Utilities for testing if we have T->as<U> defined
template <typename T, typename U>
static auto hasAsHelper(int)
    -> decltype(std::declval<T>().template as<U>(), std::true_type{});

template <typename, typename>
static auto hasAsHelper(long) -> std::false_type;

template <typename T, typename U>
struct hasAs : decltype(hasAsHelper<T, U>(int{})) {};

// Utilities for testing if we have T->as<Template> defined
template <typename T, template <typename...> typename Template>
static auto hasAsTemplateHelper(int)
    -> decltype(std::declval<T>().template as<Template>(), std::true_type{});

template <typename, template <typename...> typename>
static auto hasAsTemplateHelper(long) -> std::false_type;

template <typename T, template <typename...> typename Template>
struct hasAsTemplate : decltype(hasAsTemplateHelper<T, Template>(int{})) {};

TEST_F(DynamicTypeTest, Typing) {
  static_assert(DoubleInt64Bool().isNull());
  static_assert(!DoubleInt64Bool(1.0).isNull());
  static_assert(!DoubleInt64Bool().hasValue());
  static_assert(DoubleInt64Bool(1.0).hasValue());
  EXPECT_TRUE(DoubleInt64BoolVec().isNull());
  EXPECT_FALSE(DoubleInt64BoolVec(1.0).isNull());
  EXPECT_FALSE(DoubleInt64BoolVec().hasValue());
  EXPECT_TRUE(DoubleInt64BoolVec(1.0).hasValue());

  static_assert(hasAs<DoubleInt64BoolVec, double>::value);
  static_assert(hasAs<DoubleInt64BoolVec, int64_t>::value);
  static_assert(hasAs<DoubleInt64BoolVec, bool>::value);
  static_assert(
      hasAs<DoubleInt64BoolVec, std::vector<DoubleInt64BoolVec>>::value);
  static_assert(hasAsTemplate<DoubleInt64BoolVec, std::vector>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, SomeType>::value);
  static_assert(!hasAs<DoubleInt64BoolVec, int>::value);
  static_assert(!hasAsTemplate<DoubleInt64BoolVec, std::list>::value);

  static_assert((int)DoubleInt64Bool(true) == 1);
  EXPECT_EQ((int)DoubleInt64BoolVec(true), 1);

  EXPECT_ANY_THROW(DoubleInt64Bool(1.0).as<bool>());
  EXPECT_ANY_THROW(DoubleInt64BoolVec(1.0).as<std::vector>());

  struct CustomType {};
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<double>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<int64_t>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<bool>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<int>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<float>));
  static_assert(opcheck<IntSomeType>.canCastTo(opcheck<SomeType>));
  static_assert(!opcheck<IntSomeType>.canCastTo(opcheck<CustomType>));
  static_assert((int64_t)IntSomeType(1) == 1);
  EXPECT_THAT(
      // suppress unused value warning
      []() { (void)(SomeType)IntSomeType(1); },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Cannot cast from ")));
}

TEST_F(DynamicTypeTest, CastToDynamicType) {
  using IntOrFloat = DynamicType<NoContainers, int, float>;
  struct A {
    constexpr operator IntOrFloat() const {
      return 1;
    }
  };
  static_assert((IntOrFloat)A{} == 1);
  IntOrFloat x = A{};
  EXPECT_EQ(x, 1);
}
