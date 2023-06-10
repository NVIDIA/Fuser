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

// Note on the coding style of this file:
// - I use `namespace nvfuser` and `} // namespace nvfuser` a lot to separate
//   different parts of the code, so that we can easily fold and unfold them in
//   editors that support folding.
// - Many tests are done with static_assert, so I just put it here, instead of
//   writing a separate test file. Because I think these tests serves as a good
//   documentation of the usage.

namespace nvfuser {

// Implementation detail
namespace can_use_args_impl {

// For how to use std::void_t for SFINAE, see
// https://en.cppreference.com/w/cpp/types/void_t

template <typename, typename Fun, typename... Ts>
struct CanUseArg : std::false_type {};

template <typename Fun, typename... Ts>
struct CanUseArg<
    std::void_t<decltype(std::declval<Fun>()(std::declval<Ts>()...))>,
    Fun,
    Ts...> : std::true_type {};

} // namespace can_use_args_impl

// Check if a function Fun can be called with arguments Ts...

template <typename Fun, typename... Ts>
constexpr bool can_use_args =
    can_use_args_impl::CanUseArg<void, Fun, Ts...>::value;

// For example, `float sin(float)` can be called with int, but not with float*
// so:
static_assert(can_use_args<float (*)(float), int>);
static_assert(!can_use_args<float (*)(float), float*>);

} // namespace nvfuser

namespace nvfuser {

// Implementation detail for opcheck. This implementation is very long, I
// recommend read the usage doc below first before reading this implementation.
namespace opcheck_impl {

// A type that is purposely made implicitly convertible from OperatorChecker
struct CastableFromOperatorChecker {};

template <typename T, typename = void>
struct HasArrowOperator : std::false_type {};

template <typename T>
struct HasArrowOperator<
    T,
    std::void_t<decltype(std::declval<decltype(&T::operator->)>())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasArrowStarOperator : std::false_type {};

template <typename T>
struct HasArrowStarOperator<
    T,
    std::void_t<decltype(std::declval<decltype(&T::operator->*)>())>>
    : std::true_type {};

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
  constexpr operator CastableFromOperatorChecker() const {
    return {};
  }

  // The trick here is, when the compiler sees `operator=`, It will first try
  // with the function signature that does not require implicit conversion, that
  // is, the first candidate. And only if the first candidate fails to match, it
  // will try the function signatures that requires implicit conversion from
  // OperatorChecker to CastableFromOperatorChecker. In the code below, if the
  // expression (std::declval<T>() = std::declval<T1>()) is well-formed, then
  // template deduction for the first candidate will succeed, so it will be
  // chosen. If not, then the compiler will find that the second candidate is
  // also a match by casting OperatorChecker to CastableFromOperatorChecker.
  // So the second candidate will be chosen.
  template <typename T1>
  constexpr auto operator=(OperatorChecker<T1>) const
      -> decltype((std::declval<T>() = std::declval<T1>()), true) {
    return true;
  }
  constexpr bool operator=(CastableFromOperatorChecker) const {
    return false;
  }

  template <
      typename T1 = int,
      typename... Ts,
      std::enable_if_t<can_use_args<T, Ts...>, T1> = 0>
  constexpr bool operator()(OperatorChecker<Ts>... args) const {
    return true;
  }
  template <
      typename T1 = int,
      typename... Ts,
      std::enable_if_t<!can_use_args<T, Ts...>, T1> = 0>
  constexpr bool operator()(OperatorChecker<Ts>... args) const {
    return false;
  }

  template <typename T1>
  constexpr auto operator[](OperatorChecker<T1>) const
      -> decltype((std::declval<T>()[std::declval<T1>()]), true) {
    return true;
  }
  constexpr bool operator[](CastableFromOperatorChecker) const {
    return false;
  }

  template <
      typename T1 = int,
      std::enable_if_t<HasArrowOperator<T>::value, T1> = 0>
  constexpr auto operator->() const -> TrueType* {
    return nullptr;
  }
  template <
      typename T1 = int,
      std::enable_if_t<!HasArrowOperator<T>::value, T1> = 0>
  constexpr auto operator->() const -> FalseType* {
    return nullptr;
  }

  template <
      typename T1,
      typename T2 = int,
      std::enable_if_t<HasArrowStarOperator<T>::value, T2> = 0>
  constexpr bool operator->*(T1) const {
    return true;
  }
  template <
      typename T1,
      typename T2 = int,
      std::enable_if_t<!HasArrowStarOperator<T>::value, T2> = 0>
  constexpr bool operator->*(T1) const {
    return false;
  }

  template <typename T1>
  constexpr auto canCastTo(OperatorChecker<T1>) const
      -> decltype(((T1)(std::declval<T>())), true) {
    return true;
  }
  constexpr bool canCastTo(CastableFromOperatorChecker) const {
    return false;
  }
};

#define DEFINE_UNARY_OP(op)                                 \
  template <typename T1>                                    \
  constexpr auto operator op(OperatorChecker<T1>)           \
      ->decltype(op std::declval<T1>(), true) {             \
    return true;                                            \
  }                                                         \
                                                            \
  constexpr bool operator op(CastableFromOperatorChecker) { \
    return false;                                           \
  }

#define DEFINE_UNARY_SUFFIX_OP(op)                               \
  template <typename T1>                                         \
  constexpr auto operator op(OperatorChecker<T1>, int)           \
      ->decltype(std::declval<T1>() op, true) {                  \
    return true;                                                 \
  }                                                              \
                                                                 \
  constexpr bool operator op(CastableFromOperatorChecker, int) { \
    return false;                                                \
  }

#define DEFINE_BINARY_OP(op)                                           \
  template <typename T1, typename T2>                                  \
  constexpr auto operator op(OperatorChecker<T1>, OperatorChecker<T2>) \
      ->decltype((std::declval<T1>() op std::declval<T2>()), true) {   \
    return true;                                                       \
  }                                                                    \
                                                                       \
  constexpr bool operator op(                                          \
      CastableFromOperatorChecker, CastableFromOperatorChecker) {      \
    return false;                                                      \
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

constexpr bool operator,(
    CastableFromOperatorChecker,
    CastableFromOperatorChecker) {
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
static_assert(opcheck<int> > opcheck<float>);
// This will be true because int > float is defined. However, if you do
static_assert(!(opcheck<int> > opcheck<std::pair<int, int>>));
// This will be false because int > pair is not defined.
//
// This utility works for all overloadable operators in C++. Just use these ops
// on opcheck and you will know if it is defined for the underlying type.
//
// Due to the limitiation of C++'s operator overloading, some operators'
// interface might not be as clean as others. For example, the arrow operator ->
// is a special one. If you want to check if int has ->, you need to do:
static_assert(!(opcheck<int>->value()));
//
// For more examples, see test_dynamic_type.cpp namespace opcheck_tests
//
// Reference about operator overloading:
// https://en.cppreference.com/w/cpp/language/operators

} // namespace nvfuser

namespace nvfuser {

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

template <typename... Ts>
struct ForAllTypes;

template <typename T, typename... Ts>
struct ForAllTypes<T, Ts...> {
  template <typename Fun>
  constexpr auto operator()(Fun f) const {
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

template <>
struct ForAllTypes<> {
  template <typename Fun>
  constexpr auto operator()(Fun f) const {
    return std::tuple<>{};
  }
};

} // namespace nvfuser

namespace nvfuser {

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

// For example:
static_assert(all(true, true, true));
static_assert(all(std::make_tuple(true, true, true)));
static_assert(!all(true, false, true));
static_assert(!all(std::make_tuple(true, false, true)));

} // namespace nvfuser

namespace nvfuser {

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

// For example:
static_assert(any(true, true, true));
static_assert(any(std::make_tuple(true, true, true)));
static_assert(any(true, false, true));
static_assert(any(std::make_tuple(true, false, true)));
static_assert(!any(false, false, false));
static_assert(!any(std::make_tuple(false, false, false)));

} // namespace nvfuser

namespace nvfuser {

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

// For example:
static_assert(
    remove_void_from_tuple(
        std::make_tuple(Void{}, 1, Void{}, true, Void{}, 3.5, Void{})) ==
    std::make_tuple(1, true, 3.5));

} // namespace nvfuser

namespace nvfuser {

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

// For example:

static_assert(belongs_to<int, float, double, int>);
static_assert(!belongs_to<int, float, double, long>);

} // namespace nvfuser

namespace nvfuser {

// Take the cartesion product of two tuples.
// For example:
// cartesian_product((1, 2), (3, 4)) = ((1, 3), (1, 4), (2, 3), (2, 4))
template <typename Tuple>
constexpr auto cartesian_product(Tuple t) {
  return std::apply(
      [](auto... ts) constexpr {
        return std::make_tuple(std::make_tuple(ts)...);
      },
      t);
}

template <typename Tuple1, typename... OtherTuples>
constexpr auto cartesian_product(Tuple1 first, OtherTuples... others) {
  auto c_first = cartesian_product(first);
  auto c_others = cartesian_product(others...);
  return std::apply(
      [c_others](auto... ts) constexpr {
        // cat one item in c_first with all the items in c_others
        auto cat_one_first_all_others = [c_others](auto first_item) {
          return std::apply(
              [first_item](auto... other_item) constexpr {
                return std::make_tuple(
                    std::tuple_cat(first_item, other_item)...);
              },
              c_others);
        };
        return std::tuple_cat(cat_one_first_all_others(ts)...);
      },
      c_first);
}

// For example:

static_assert(
    cartesian_product(std::make_tuple(1.0, true)) ==
    std::make_tuple(std::make_tuple(1.0), std::make_tuple(true)));

static_assert(
    cartesian_product(std::make_tuple(1.0, true), std::make_tuple(2.0f, 4)) ==
    std::make_tuple(
        std::make_tuple(1.0, 2.0f),
        std::make_tuple(1.0, 4),
        std::make_tuple(true, 2.0f),
        std::make_tuple(true, 4)));

static_assert(
    cartesian_product(
        std::make_tuple(1.0, true),
        std::make_tuple(2.0f, 4),
        std::make_tuple(std::size_t(0), nullptr)) ==
    std::make_tuple(
        std::make_tuple(1.0, 2.0f, std::size_t(0)),
        std::make_tuple(1.0, 2.0f, nullptr),
        std::make_tuple(1.0, 4, std::size_t(0)),
        std::make_tuple(1.0, 4, nullptr),
        std::make_tuple(true, 2.0f, std::size_t(0)),
        std::make_tuple(true, 2.0f, nullptr),
        std::make_tuple(true, 4, std::size_t(0)),
        std::make_tuple(true, 4, nullptr)));

} // namespace nvfuser

namespace nvfuser {

// Can I find a T1 from Tuple1 and a T2 from Tuple2 such that f(T1, T2) is
// defined? For example, if Tuple1 = (int, float), Tuple2 = (int, bool), and
// f(T1, T2) = T1 + T2, then the answer is yes, because I can find T1 = int and
// T2 = int such that f(T1, T2) is defined.
template <typename... Tuples, typename Fun>
constexpr bool any_defined(Fun f, Tuples... tuples) {
  auto c = cartesian_product(tuples...);
  return std::apply(
      [f](auto... candidates) constexpr {
        return any(std::apply(
            [&](auto... args) constexpr {
              // TODO: I should actually use
              //   return opcheck<Fun>(opcheck<decltype(args)>...);
              // or
              //   return can_use_args<Fun, decltype(args)...>;
              // because the current implementation only support f being a
              // single op, but unfortunately this won't compile. I have a
              // feeling that this is a bug of SFINAE in the compiler, but I'm
              // not sure.
              return f(opcheck<decltype(args)>...);
            },
            candidates)...);
      },
      c);
}

// For example:

// Can I find a T from {int, std::pair<int, int>} such that -T is defined?
static_assert(any_defined(
    [](auto x) constexpr { return -x; },
    std::tuple<int, std::pair<int, int>>{}));

// Can I find a T from {std::pair<int, int>, std::pair<int, float>} such that -T
// is defined?
static_assert(!any_defined(
    [](auto x) constexpr { return -x; },
    std::tuple<std::pair<int, int>, std::pair<int, float>>{}));

// Can I find a T from {int, std::pair<int, int>} and a U from
// {std::pair<int, int>, float} such that T + U is defined?
static_assert(any_defined(
    [](auto x, auto y) constexpr { return x + y; },
    std::tuple<int, std::pair<int, int>>{},
    std::tuple<std::pair<int, int>, float>{}));

// Can I find a T from {int, std::pair<int, int>} and a U from
// {std::pair<int, int>, std::pair<int, float>} such that T + U is defined?
static_assert(!any_defined(
    [](auto x, auto y) constexpr { return x + y; },
    std::tuple<int, std::pair<int, int>>{},
    std::tuple<std::pair<int, int>, std::pair<int, float>>{}));

} // namespace nvfuser
