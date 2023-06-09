// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <type_traits>
#include <utility>

namespace nvfuser {

namespace has_operator_impl {

struct HasOperatorHelper {};

struct IsNullaryFunc {
  template <typename T>
  static constexpr auto check(int) -> decltype((std::declval<T>()()), true) {
    return true;
  }

  template <typename T>
  static constexpr bool check(long) {
    return false;
  }
};

struct HasArrowOperator {
  template <typename T>
  static constexpr auto check(int)
      -> decltype((std::declval<decltype(&T::operator->)>()), true) {
    return true;
  }

  template <typename T>
  static constexpr bool check(long) {
    return false;
  }
};

struct HasArrowStarOperator {
  template <typename T>
  static constexpr auto check(int)
      -> decltype((std::declval<decltype(&T::operator->*)>()), true) {
    return true;
  }

  template <typename T>
  static constexpr bool check(long) {
    return false;
  }
};

struct TrueType {
  static constexpr bool value() {
    return true;
  }
};

struct FalseType {
  static constexpr bool value() {
    return false;
  }
};

template <typename T>
struct HasOperator {
  constexpr operator HasOperatorHelper() const {
    return {};
  }

  template <typename T1>
  constexpr auto operator=(HasOperator<T1>) const
      -> decltype((std::declval<T>() = std::declval<T1>()), true) {
    return true;
  }

  constexpr bool operator=(HasOperatorHelper) const {
    return false;
  }

  template <typename... Ts>
  constexpr auto operator()(HasOperator<Ts>... args) const
      -> decltype((std::declval<T>()(std::declval<Ts>()...)), true) {
    return true;
  }

  constexpr bool operator()(HasOperatorHelper) const {
    return false;
  }

  template <typename... Ts>
  constexpr bool operator()(HasOperatorHelper, Ts... args) const {
    return false && operator()(args...);
  }

  template <
      typename T1 = int,
      std::enable_if_t<!IsNullaryFunc::check<T>(int{}), T1> = 0>
  constexpr bool operator()() const {
    return false;
  }

  template <typename T1>
  constexpr auto operator[](HasOperator<T1>) const
      -> decltype((std::declval<T>()[std::declval<T1>()]), true) {
    return true;
  }

  constexpr bool operator[](HasOperatorHelper) const {
    return false;
  }

  template <
      typename T1 = int,
      std::enable_if_t<HasArrowOperator::check<T>(int{}), T1> = 0>
  constexpr auto operator->() const -> TrueType* {
    return nullptr;
  }

  template <
      typename T1 = int,
      std::enable_if_t<!HasArrowOperator::check<T>(int{}), T1> = 0>
  constexpr auto operator->() const -> FalseType* {
    return nullptr;
  }

  template <
      typename T1,
      typename T2 = int,
      std::enable_if_t<HasArrowStarOperator::check<T>(int{}), T2> = 0>
  constexpr bool operator->*(T1) const {
    return true;
  }

  template <
      typename T1,
      typename T2 = int,
      std::enable_if_t<!HasArrowStarOperator::check<T>(int{}), T2> = 0>
  constexpr bool operator->*(T1) const {
    return false;
  }
};

#define DEFINE_UNARY_OP(op)                       \
  template <typename T1>                          \
  constexpr auto operator op(HasOperator<T1>)     \
      ->decltype(op std::declval<T1>(), true) {   \
    return true;                                  \
  }                                               \
                                                  \
  constexpr bool operator op(HasOperatorHelper) { \
    return false;                                 \
  }

#define DEFINE_UNARY_SUFFIX_OP(op)                     \
  template <typename T1>                               \
  constexpr auto operator op(HasOperator<T1>, int)     \
      ->decltype(std::declval<T1>() op, true) {        \
    return true;                                       \
  }                                                    \
                                                       \
  constexpr bool operator op(HasOperatorHelper, int) { \
    return false;                                      \
  }

#define DEFINE_BINARY_OP(op)                                         \
  template <typename T1, typename T2>                                \
  constexpr auto operator op(HasOperator<T1>, HasOperator<T2>)       \
      ->decltype((std::declval<T1>() op std::declval<T2>()), true) { \
    return true;                                                     \
  }                                                                  \
                                                                     \
  constexpr bool operator op(HasOperatorHelper, HasOperatorHelper) { \
    return false;                                                    \
  }

// Unary operators
DEFINE_UNARY_OP(+);
DEFINE_UNARY_OP(-);
DEFINE_UNARY_OP(~);
DEFINE_UNARY_OP(!);
DEFINE_UNARY_OP(++);
DEFINE_UNARY_OP(--);
DEFINE_UNARY_SUFFIX_OP(++);
DEFINE_UNARY_SUFFIX_OP(--);
DEFINE_UNARY_OP(*);
DEFINE_UNARY_OP(&);

// Binary operators
DEFINE_BINARY_OP(+);
DEFINE_BINARY_OP(-);
DEFINE_BINARY_OP(*);
DEFINE_BINARY_OP(/);
DEFINE_BINARY_OP(%);
DEFINE_BINARY_OP(&);
DEFINE_BINARY_OP(|);
DEFINE_BINARY_OP(^);
DEFINE_BINARY_OP(&&);
DEFINE_BINARY_OP(||);
DEFINE_BINARY_OP(<<);
DEFINE_BINARY_OP(>>);
DEFINE_BINARY_OP(==);
DEFINE_BINARY_OP(!=);
DEFINE_BINARY_OP(<);
DEFINE_BINARY_OP(>);
DEFINE_BINARY_OP(<=);
DEFINE_BINARY_OP(>=);

// Assignment operators
DEFINE_BINARY_OP(+=);
DEFINE_BINARY_OP(-=);
DEFINE_BINARY_OP(*=);
DEFINE_BINARY_OP(/=);
DEFINE_BINARY_OP(%=);
DEFINE_BINARY_OP(&=);
DEFINE_BINARY_OP(|=);
DEFINE_BINARY_OP(^=);
DEFINE_BINARY_OP(<<=);
DEFINE_BINARY_OP(>>=);

#undef DEFINE_UNARY_OP
#undef DEFINE_UNARY_SUFFIX_OP
#undef DEFINE_BINARY_OP

// comma operator
// This is essentailly just DEFINE_BINARY_OP(,); But because of the C++
// preprocessor, comma is treated as a separator for arguments, so we need to do
// it manually.
template <typename T1, typename T2>
constexpr auto operator,(HasOperator<T1>, HasOperator<T2>)
    -> decltype((std::declval<T1>(), std::declval<T2>()), true) {
  return true;
}

constexpr bool operator,(HasOperatorHelper, HasOperatorHelper) {
  return false;
}

// TODO: overload the following operators:
// <=> (requires C++20)

} // namespace has_operator_impl

// reference: https://en.cppreference.com/w/cpp/language/operators
template <typename T>
constexpr has_operator_impl::HasOperator<T> has_operator;

} // namespace nvfuser