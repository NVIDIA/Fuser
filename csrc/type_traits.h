// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace nvfuser {

// See note [Operator checker] below

namespace opcheck_impl {

struct OperatorCheckerHelper {};

// Implementation detail: The pattern like IsNullaryFunc repeats multiple times
// and the idea behind it is the key to the implementation of operator checker.
// The idea is to use SFINAE. I will add detailed comment to IsNullaryFunc, and
// other classes should have the same principle.
struct IsNullaryFunc {
  // In this struct, we define two check functions, one takes an int, and one
  // takes a long. To use this struct, we should call
  // IsNullaryFunc::check<T>(int{}). Be sure that the argument has type int
  // instead of long. Because the argument is int, the compiler will try to do
  // pattern matching on the check that takes an int argument first, because
  // this variant does not require any automatic conversion for argument. If the
  // pattern matching succeeds, then this variant will be chosen. Otherwise, the
  // compiler will try the other variant, which takes a long, and this variant
  // is designed that pattern matching always succeeds. So if the compiler
  // decide to convert int{} to long, this variant will be generated to be
  // chosen.

  template <typename T>
  static constexpr auto check(int) -> decltype((std::declval<T>()()), true) {
    // When trying to match this variant, the compiler will try to evaluate the
    // expression inside decltype. If the expression is valid, then the pattern
    // matching succeeds, and this variant will be chosen. std::declval<T>() is
    // a value of type T, and if T is a nullary function type, then
    // std::declval<T>()() is well-formed, and the expression is valid.
    // Otherwise the expression is invalid, and the pattern matching fails. The
    // comma ensures that the result of decltype is always bool.
    return true;
  }

  template <typename T>
  static constexpr bool check(long) {
    // The compiler will only consider this variant if the pattern matching on
    // the previous variant fails.
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

  template <typename T1>
  constexpr auto canCastTo(OperatorChecker<T1>) const
      -> decltype(((T1)(std::declval<T>())), true) {
    return true;
  }

  constexpr bool canCastTo(OperatorCheckerHelper) const {
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

// Run the given function on each type in the variadic template list.
// The function should take a single argument of type T*. Note that the argument
// of the function should only be used for type deduction, and its value should
// not be used.
//
// The returned value of the function is a tuple of the return values of the
// function calls, with void replaced by Void.
//
// For example, if you want to print 0.2 as bool, int, and float, you can do the
// following:
//   auto f = [](auto* x) {
//     using T = std::remove_pointer_t<decltype(x)>;
//     std::cout << T(0.2) << std::endl;
//   };
//   ForAllTypes<bool, int, float>{}(f);

struct Void {};

template <typename T, typename... Ts>
struct ForAllTypes {
  template <typename Fun>
  constexpr auto operator()(Fun f) {
    using RetT = decltype(f((T*)nullptr));
    if constexpr (std::is_void_v<RetT>) {
      f((T*)nullptr);
      return std::tuple_cat(std::tuple<Void>{}, ForAllTypes<Ts...>{}(f));
    } else {
      return std::tuple_cat(
          std::make_tuple(f((T*)nullptr)), ForAllTypes<Ts...>{}(f));
    }
  }
};

template <typename T>
struct ForAllTypes<T> {
  template <typename Fun>
  constexpr auto operator()(Fun f) {
    using RetT = decltype(f((T*)nullptr));
    if constexpr (std::is_void_v<RetT>) {
      f((T*)nullptr);
      return std::tuple<Void>{};
    } else {
      return std::make_tuple(f((T*)nullptr));
    }
  }
};

// Check if all the given booleans are true.

template <typename... Ts>
constexpr bool all(Ts... bs) {
  return (bs && ...);
}

template <typename... Ts>
constexpr bool all(std::tuple<Ts...> bs) {
  return std::apply([](auto... bs) { return all(bs...); }, bs);
}

// (Void, T1, Void, T2, Void, T3, ...) -> (T1, T2, T3, ...)

void remove_void_from_tuple(std::tuple<>) {}

template <typename T, typename... Ts>
auto remove_void_from_tuple(std::tuple<T, Ts...> t) {
  std::tuple<Ts...> others = std::apply(
      [](auto head, auto... tail) { return std::make_tuple(tail...); }, t);
  if constexpr (std::is_same_v<T, Void>) {
    return remove_void_from_tuple();
  } else {
    using first = std::tuple<T>{std::get<0>(t)};
    using others_t = decltype(remove_void_from_tuple(others));
    if constexpr (std::is_void_v<others_t>) {
      return first;
    } else {
      return std::tuple_cat(first, others);
    }
  }
}

} // namespace nvfuser
