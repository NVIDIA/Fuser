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

// See note [Operator checker] below

namespace opcheck_impl {

struct OperatorCheckerHelper {};

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
struct OperatorChecker {
  constexpr operator OperatorCheckerHelper() const {
    return {};
  }

  template <typename T1>
  constexpr auto operator=(OperatorChecker<T1>) const
      -> decltype((std::declval<T>() = std::declval<T1>()), true) {
    return true;
  }

  constexpr bool operator=(OperatorCheckerHelper) const {
    return false;
  }

  template <typename... Ts>
  constexpr auto operator()(OperatorChecker<Ts>... args) const
      -> decltype((std::declval<T>()(std::declval<Ts>()...)), true) {
    return true;
  }

  constexpr bool operator()(OperatorCheckerHelper) const {
    return false;
  }

  template <typename... Ts>
  constexpr bool operator()(OperatorCheckerHelper, Ts... args) const {
    return false && operator()(args...);
  }

  template <
      typename T1 = int,
      std::enable_if_t<!IsNullaryFunc::check<T>(int{}), T1> = 0>
  constexpr bool operator()() const {
    return false;
  }

  template <typename T1>
  constexpr auto operator[](OperatorChecker<T1>) const
      -> decltype((std::declval<T>()[std::declval<T1>()]), true) {
    return true;
  }

  constexpr bool operator[](OperatorCheckerHelper) const {
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

#define DEFINE_UNARY_OP(op)                           \
  template <typename T1>                              \
  constexpr auto operator op(OperatorChecker<T1>)     \
      ->decltype(op std::declval<T1>(), true) {       \
    return true;                                      \
  }                                                   \
                                                      \
  constexpr bool operator op(OperatorCheckerHelper) { \
    return false;                                     \
  }

#define DEFINE_UNARY_SUFFIX_OP(op)                         \
  template <typename T1>                                   \
  constexpr auto operator op(OperatorChecker<T1>, int)     \
      ->decltype(std::declval<T1>() op, true) {            \
    return true;                                           \
  }                                                        \
                                                           \
  constexpr bool operator op(OperatorCheckerHelper, int) { \
    return false;                                          \
  }

#define DEFINE_BINARY_OP(op)                                                 \
  template <typename T1, typename T2>                                        \
  constexpr auto operator op(OperatorChecker<T1>, OperatorChecker<T2>)       \
      ->decltype((std::declval<T1>() op std::declval<T2>()), true) {         \
    return true;                                                             \
  }                                                                          \
                                                                             \
  constexpr bool operator op(OperatorCheckerHelper, OperatorCheckerHelper) { \
    return false;                                                            \
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
constexpr auto operator,(OperatorChecker<T1>, OperatorChecker<T2>)
    -> decltype((std::declval<T1>(), std::declval<T2>()), true) {
  return true;
}

constexpr bool operator,(OperatorCheckerHelper, OperatorCheckerHelper) {
  return false;
}

// TODO: overload the following operators:
// <=> (requires C++20)

} // namespace opcheck_impl

template <typename T>
constexpr opcheck_impl::OperatorChecker<T> opcheck;

// Note [Operator checker]
//
// "opcheck" is a utility to check if an operator for certain type is defined.
// For example, if you want to check if int > float is defined, you can do:
constexpr bool int_gt_float_is_defined = (opcheck<int> > opcheck<float>);
static_assert(int_gt_float_is_defined);
// This will return true because int > float is defined. However, if you do
constexpr bool int_gt_pair_is_defined =
    (opcheck<int> > opcheck<std::pair<int, int>>);
static_assert(!int_gt_pair_is_defined);
// This will return false because int > pair is not defined.
//
// This utility works for all overloadable operators in C++. Just use these ops
// on opcheck and you will know if it is defined for the underlying type.
//
// Due to the limitiation of C++'s operator overloading, some operators'
// interface might not be as clean as others. For example, the arrow operator ->
// is a special one. If you want to check if int has ->, you need to do:
constexpr bool int_has_arrow = (opcheck<int>->value());
static_assert(!int_has_arrow);
//
// For more examples, see test_dynamic_type.cpp namespace opcheck_tests
//
// reference: https://en.cppreference.com/w/cpp/language/operators

} // namespace nvfuser
