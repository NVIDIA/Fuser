// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

// Heavy operator implementations for DynamicType.
// This file will contain operator implementations that can be
// instantiated once via explicit template instantiation.

#include "decl.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbool-operation"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbool-operation"
#endif

namespace dynamic_type {

// Stream output operator implementation
template <typename DT, typename>
std::ostream& operator<<(std::ostream& os, const DT& dt) {
  bool printed = false;
  DT::for_all_types([&printed, &os, &dt](auto _) {
    using T = typename decltype(_)::type;
    if constexpr (opcheck<std::ostream&> << opcheck<T>) {
      if constexpr (std::is_same_v<
                        decltype(os << std::declval<T>()),
                        std::ostream&>) {
        if (dt.template is<T>()) {
          os << dt.template as<T>();
          printed = true;
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(
      printed, "Can not print ", dt.type().name(), " : incompatible type");
  return os;
}

// NOTE: Unary operators (+, -, ~, !) are now friend functions inside DynamicType class.

// Dereference operator implementation
template <typename DT, typename>
DT& operator*(const DT& x) {
  std::optional<std::reference_wrapper<DT>> ret = std::nullopt;
  DT::for_all_types([&ret, &x](auto t) {
    using T = typename decltype(t)::type;
    if constexpr (*opcheck<T>) {
      if constexpr (std::is_same_v<decltype(*std::declval<T>()), DT&>) {
        if (x.template is<T>()) {
          ret = std::ref(*(x.template as<T>()));
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(ret.has_value(), "Cannot dereference ", x.type().name());
  return ret.value();
}

// NOTE: Prefix/postfix ++/-- are now friend functions inside DynamicType class.

// NOTE: Compound assignment operators are now friend functions inside DynamicType class.

// Binary operator implementations
#define DEFINE_BINARY_OP_IMPL(opname, op, func_name, return_type, check_existence) \
  template <typename LHS, typename RHS, typename DT, typename>                 \
  inline constexpr return_type func_name(LHS&& x, RHS&& y) {                   \
    constexpr bool lhs_is_dt = is_dynamic_type_v<std::decay_t<LHS>>;           \
    constexpr bool rhs_is_dt = is_dynamic_type_v<std::decay_t<RHS>>;           \
    if constexpr (                                                             \
        lhs_is_dt && !rhs_is_dt &&                                             \
        opcheck<std::decay_t<RHS>>.hasExplicitCastTo(                          \
            opcheck<std::decay_t<LHS>>)) {                                     \
      return x op(DT) y;                                                       \
    } else if constexpr (                                                      \
        !lhs_is_dt && rhs_is_dt &&                                             \
        opcheck<std::decay_t<LHS>>.hasExplicitCastTo(                          \
            opcheck<std::decay_t<RHS>>)) {                                     \
      return (DT)x op y;                                                       \
    } else {                                                                   \
      return DT::dispatch(                                                     \
          [](auto&& x, auto&& y) -> decltype(auto) {                           \
            using X = decltype(x);                                             \
            using Y = decltype(y);                                             \
            if constexpr (false) {                                             \
              /* TODO: This doesn't work on gcc 11.4 with C++20, temporarily   \
               * disabled and use the more verbose implementation below. We    \
               * should reenable this when we upgrade our compilers. */        \
              if constexpr (opname##_type_compatible<X, Y, return_type>()) {   \
                return std::forward<X>(x) op std::forward<Y>(y);               \
              }                                                                \
            } else {                                                           \
              if constexpr (opcheck<X> op opcheck<Y>) {                        \
                if constexpr (std::is_convertible_v<                           \
                                  decltype(std::declval<X>()                   \
                                               op std::declval<Y>()),          \
                                  return_type>) {                              \
                  return std::forward<X>(x) op std::forward<Y>(y);             \
                }                                                              \
              }                                                                \
            }                                                                  \
          },                                                                   \
          std::forward<LHS>(x),                                                \
          std::forward<RHS>(y));                                               \
    }                                                                          \
  }

// NOTE: Most binary operators are now friend functions inside DynamicType class.
// Only operator&& and operator|| remain as templates to avoid ambiguity with
// built-in bool && bool when one operand is bool.
DEFINE_BINARY_OP_IMPL(land, &&, operator&&, DT, true);
DEFINE_BINARY_OP_IMPL(lor, ||, operator||, DT, true);

#undef DEFINE_BINARY_OP_IMPL

// ============================================================================
// Static member implementations for friend operators
// These are covered by extern template, so only instantiated once
// ============================================================================

#define DEFINE_BINARY_OP_FRIEND_IMPL(opname, op, return_type)                  \
  template <typename Containers, typename... Ts>                               \
  auto DynamicType<Containers, Ts...>::opname##_impl(                          \
      const DynamicType& a, const DynamicType& b) -> return_type {             \
    std::optional<return_type> result;                                         \
    dispatch(                                                                  \
        [&result](auto&& x, auto&& y) {                                        \
          using X = std::decay_t<decltype(x)>;                                 \
          using Y = std::decay_t<decltype(y)>;                                 \
          if constexpr (opcheck<X> op opcheck<Y>) {                            \
            if constexpr (std::is_convertible_v<decltype(x op y), return_type>) { \
              result = static_cast<return_type>(x op y);                       \
            }                                                                  \
          }                                                                    \
        },                                                                     \
        a, b);                                                                 \
    DYNAMIC_TYPE_CHECK(                                                        \
        result.has_value(),                                                    \
        "Cannot compute ", a.type().name(), " " #op " ", b.type().name());     \
    return *result;                                                            \
  }

// Comparison operators (return bool)
DEFINE_BINARY_OP_FRIEND_IMPL(eq, ==, bool)
DEFINE_BINARY_OP_FRIEND_IMPL(neq, !=, bool)
DEFINE_BINARY_OP_FRIEND_IMPL(lt, <, bool)
DEFINE_BINARY_OP_FRIEND_IMPL(gt, >, bool)
DEFINE_BINARY_OP_FRIEND_IMPL(le, <=, bool)
DEFINE_BINARY_OP_FRIEND_IMPL(ge, >=, bool)

// Arithmetic operators (return DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(add, +, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(sub, -, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(mul, *, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(div, /, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(mod, %, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(band, &, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(bor, |, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(bxor, ^, DynamicType)
// NOTE: land and lor kept as template functions (see DEFINE_BINARY_OP_IMPL above)
DEFINE_BINARY_OP_FRIEND_IMPL(lshift, <<, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(rshift, >>, DynamicType)

// Named comparison functions (return DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_eq, ==, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_neq, !=, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_lt, <, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_gt, >, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_le, <=, DynamicType)
DEFINE_BINARY_OP_FRIEND_IMPL(named_ge, >=, DynamicType)

#undef DEFINE_BINARY_OP_FRIEND_IMPL

// ============================================================================
// Unary operator static member implementations
// ============================================================================

#define DEFINE_UNARY_OP_FRIEND_IMPL(opname, op)                                \
  template <typename Containers, typename... Ts>                               \
  auto DynamicType<Containers, Ts...>::opname##_impl(                          \
      const DynamicType& x) -> DynamicType {                                   \
    std::optional<DynamicType> result;                                         \
    for_all_types([&result, &x](auto t) {                                      \
      using Type = typename decltype(t)::type;                                 \
      if constexpr (op opcheck<Type>) {                                        \
        if constexpr (std::is_constructible_v<VariantType, decltype(op std::declval<Type>())>) { \
          if (x.template is<Type>()) {                                         \
            result = DynamicType(op x.template as<Type>());                    \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        result.has_value(),                                                    \
        "Cannot compute " #op, x.type().name(), " : incompatible type");       \
    return *result;                                                            \
  }

DEFINE_UNARY_OP_FRIEND_IMPL(pos, +)
DEFINE_UNARY_OP_FRIEND_IMPL(neg, -)
DEFINE_UNARY_OP_FRIEND_IMPL(bnot, ~)
DEFINE_UNARY_OP_FRIEND_IMPL(lnot, !)

#undef DEFINE_UNARY_OP_FRIEND_IMPL

// ============================================================================
// Prefix increment/decrement static member implementations (++x, --x)
// ============================================================================

template <typename Containers, typename... Ts>
auto DynamicType<Containers, Ts...>::lpp_impl(DynamicType& x) -> DynamicType& {
  bool computed = false;
  for_all_types([&computed, &x](auto t) {
    using Type = typename decltype(t)::type;
    if constexpr (++opcheck<Type&>) {
      if constexpr (std::is_same_v<decltype(++std::declval<Type&>()), Type&>) {
        if (x.template is<Type>()) {
          ++x.template as<Type>();
          computed = true;
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(computed, "Cannot compute ++", x.type().name());
  return x;
}

template <typename Containers, typename... Ts>
auto DynamicType<Containers, Ts...>::lmm_impl(DynamicType& x) -> DynamicType& {
  bool computed = false;
  for_all_types([&computed, &x](auto t) {
    using Type = typename decltype(t)::type;
    if constexpr (--opcheck<Type&>) {
      if constexpr (std::is_same_v<decltype(--std::declval<Type&>()), Type&>) {
        if (x.template is<Type>()) {
          --x.template as<Type>();
          computed = true;
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(computed, "Cannot compute --", x.type().name());
  return x;
}

// ============================================================================
// Postfix increment/decrement static member implementations (x++, x--)
// ============================================================================

template <typename Containers, typename... Ts>
auto DynamicType<Containers, Ts...>::rpp_impl(DynamicType& x) -> DynamicType {
  std::optional<DynamicType> result;
  for_all_types([&result, &x](auto t) {
    using Type = typename decltype(t)::type;
    if constexpr (opcheck<Type&>++) {
      if constexpr (std::is_constructible_v<VariantType, decltype(std::declval<Type&>()++)>) {
        if (x.template is<Type>()) {
          result = DynamicType(x.template as<Type>()++);
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(result.has_value(), "Cannot compute ", x.type().name(), "++");
  return *result;
}

template <typename Containers, typename... Ts>
auto DynamicType<Containers, Ts...>::rmm_impl(DynamicType& x) -> DynamicType {
  std::optional<DynamicType> result;
  for_all_types([&result, &x](auto t) {
    using Type = typename decltype(t)::type;
    if constexpr (opcheck<Type&>--) {
      if constexpr (std::is_constructible_v<VariantType, decltype(std::declval<Type&>()--)>) {
        if (x.template is<Type>()) {
          result = DynamicType(x.template as<Type>()--);
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(result.has_value(), "Cannot compute ", x.type().name(), "--");
  return *result;
}

} // namespace dynamic_type

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

