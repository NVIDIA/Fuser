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

// Implementation detail: The pattern like IsFunc repeats multiple times and the
// idea behind it is the key to the implementation of operator checker. The idea
// is to use SFINAE. I will add detailed comment to IsFunc, and other classes
// should have the same principle.
struct IsFunc {
  // In this struct, we define two check functions, one takes an int, and one
  // takes a long. To use this struct, we should call
  // IsFunc::check<Fun, Arg1, Arg2>(int{})
  // Be sure that the argument has type int instead of long. Because the
  // argument is int, the compiler will try to do pattern matching on the check
  // that takes an int argument first, because this variant does not require any
  // automatic conversion for argument. If the pattern matching succeeds, then
  // this variant will be chosen. Otherwise, the compiler will try the other
  // variant, which takes a long, and this variant is designed that pattern
  // matching always succeeds. So if the compiler decide to convert int{} to
  // long, this variant will be generated to be chosen.

  template <typename Fun, typename... Ts>
  static constexpr auto check(int)
      -> decltype((std::declval<Fun>()(std::declval<Ts>()...)), true) {
    // When trying to match this variant, the compiler will try to evaluate the
    // expression inside decltype. If the expression is valid, then the pattern
    // matching succeeds, and this variant will be chosen. std::declval<Fun>()
    // is a value of type Fun, and if Fun is a desired function type, then
    // std::declval<Fun>()(std::declval<Ts>()...) is well-formed, and the
    // expression is valid. Otherwise the expression is invalid, and the pattern
    // matching fails. The comma ensures that the result of decltype is always
    // bool.
    return true;
  }

  template <typename Fun, typename... Ts>
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

  template <
      typename T1 = int,
      typename... Ts,
      std::enable_if_t<IsFunc::check<T, Ts...>(int{}), T1> = 0>
  constexpr bool operator()(OperatorChecker<Ts>... args) const {
    return true;
  }

  template <
      typename T1 = int,
      typename... Ts,
      std::enable_if_t<!IsFunc::check<T, Ts...>(int{}), T1> = 0>
  constexpr bool operator()(OperatorChecker<Ts>... args) const {
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

namespace opcheck_note {

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

} // namespace opcheck_note

// Basically just "void". We need this because if we have something like
// std::tuple<void, int> we will be unable to create an instance of it.
// So we have to use something like std::tuple<Void, int> instead.
struct Void {};

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
// And the output will be:
//  1
//  0
//  0.2

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

// Check if all the booleans in the arguments are true. There are two versions:
// one for variadic arguments, and one for std::tuple.

template <typename... Ts>
constexpr bool all(Ts... bs) {
  return (bs && ...);
}

template <typename... Ts>
constexpr bool all(std::tuple<Ts...> bs) {
  return std::apply([](auto... bs) { return all(bs...); }, bs);
}

// Check if all the booleans in the arguments are true. There are two versions:
// one for variadic arguments, and one for std::tuple.

template <typename... Ts>
constexpr bool any(Ts... bs) {
  return (bs || ...);
}

template <typename... Ts>
constexpr bool any(std::tuple<Ts...> bs) {
  return std::apply([](auto... bs) { return any(bs...); }, bs);
}

// Remove all the voids from a tuple. For example:
// (Void, T1, Void, T2, Void, T3, ...) -> (T1, T2, T3, ...)

template <typename... Ts>
constexpr auto remove_void_from_tuple(std::tuple<Ts...> t) {
  if constexpr (sizeof...(Ts) == 0) {
    return std::tuple<>{};
  } else {
    auto [head, others] = std::apply(
        [](auto head, auto... tail) {
          return std::make_tuple(
              std::make_tuple(head), std::make_tuple(tail...));
        },
        t);
    auto proccessed_others = remove_void_from_tuple(others);
    if constexpr (std::is_same_v<
                      std::tuple_element_t<0, decltype(head)>,
                      Void>) {
      return proccessed_others;
    } else {
      return std::tuple_cat(head, proccessed_others);
    }
  }
}

// Implementation helper for belongs_to. See below for the actual definition of
// belongs_to.
namespace belongs_to_impl {

// Given a tuple of Ts, return a tuple with the same size as Ts. The tuple
// contains either true or void. (true if T is the same as the corresponding
// type in Ts, void otherwise). For example, if T = int, Ts is (int, float,
// bool), then the return type is (true, void, void).
template <typename T, typename... Ts>
auto get_match_tuple() {
  auto true_or_void = [](auto* x) {
    using U = std::remove_pointer_t<decltype(x)>;
    if constexpr (std::is_same_v<T, U>) {
      return true;
    } else {
      return;
    }
  };
  return ForAllTypes<Ts...>{}(true_or_void);
}

} // namespace belongs_to_impl

// Check if T belongs to the given type list Ts. For example
// belongs_to<int, int, float, bool> is true, but
// belongs_to<int, float, bool> is false.
template <typename T, typename... Ts>
constexpr bool belongs_to =
    (std::tuple_size_v<decltype(remove_void_from_tuple(
         belongs_to_impl::get_match_tuple<T, Ts...>()))> > 0);

// Take the cartesion product of two tuples.
// For example:
// cartesian_product((1, 2), (3, 4)) = ((1, 3), (1, 4), (2, 3), (2, 4))
template <typename... Ts, typename... Us>
constexpr auto cartesian_product(std::tuple<Ts...> t, std::tuple<Us...> u) {
  return std::apply(
      [u](auto... ts) constexpr {
        return std::tuple_cat(std::apply(
            [ts](auto... us) constexpr {
              return std::make_tuple(std::make_tuple(ts, us)...);
            },
            u)...);
      },
      t);
}

template <typename... Tuples, typename Fun>
constexpr bool any_defined(Fun f, Tuples... tuples) {
  constexpr auto c = decltype(cartesian_product(tuples...)){};
  return std::apply(
      [f](auto... candidates) constexpr {
        return any(std::apply(
            [&](auto... args) constexpr {
              return f(opcheck<decltype(args)>...);
            },
            candidates)...);
      },
      c);
}

} // namespace nvfuser
