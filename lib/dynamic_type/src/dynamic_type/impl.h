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

// Unary operator implementations
#define DEFINE_UNARY_OP_IMPL(opname, op)                                       \
  template <typename DT, typename>                                             \
  inline constexpr decltype(auto) operator op(DT&& x) {                        \
    return std::decay_t<DT>::dispatch(                                         \
        [](auto&& x) -> decltype(auto) {                                       \
          if constexpr (op opcheck<std::decay_t<decltype(x)>>) {               \
            return op std::forward<decltype(x)>(x);                            \
          }                                                                    \
        },                                                                     \
        std::forward<DT>(x));                                                  \
  }

DEFINE_UNARY_OP_IMPL(pos, +);
DEFINE_UNARY_OP_IMPL(neg, -);
DEFINE_UNARY_OP_IMPL(bnot, ~);
DEFINE_UNARY_OP_IMPL(lnot, !);
#undef DEFINE_UNARY_OP_IMPL

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

// Prefix ++/-- operator implementations
#define DEFINE_LEFT_PPMM_IMPL(opname, op)                                      \
  template <typename DT, typename>                                             \
  inline constexpr DT& operator op(DT & x) {                                   \
    bool computed = false;                                                     \
    DT::for_all_types([&computed, &x](auto _) {                                \
      using Type = typename decltype(_)::type;                                 \
      if constexpr (op opcheck<Type&>) {                                       \
        if constexpr (std::is_same_v<                                          \
                          decltype(op std::declval<Type&>()),                  \
                          Type&>) {                                            \
          if (x.template is<Type>()) {                                         \
            op x.template as<Type>();                                          \
            computed = true;                                                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        computed,                                                              \
        "Cannot compute ",                                                     \
        #op,                                                                   \
        x.type().name(),                                                       \
        " : incompatible type");                                               \
    return x;                                                                  \
  }

DEFINE_LEFT_PPMM_IMPL(lpp, ++);
DEFINE_LEFT_PPMM_IMPL(lmm, --);
#undef DEFINE_LEFT_PPMM_IMPL

// Postfix ++/-- operator implementations
#define DEFINE_RIGHT_PPMM_IMPL(opname, op)                                     \
  template <typename DT>                                                       \
  inline constexpr std::enable_if_t<                                           \
      is_dynamic_type_v<DT> &&                                                 \
          any_check(                                                           \
              opname##_helper<typename DT::VariantType>,                       \
              DT::type_identities_as_tuple),                                   \
      DT> operator op(DT & x, int) {                                           \
    DT ret;                                                                    \
    DT::for_all_types([&ret, &x](auto _) {                                     \
      using Type = typename decltype(_)::type;                                 \
      if constexpr (opcheck<Type&> op) {                                       \
        if constexpr (std::is_constructible_v<                                 \
                          typename DT::VariantType,                            \
                          decltype(std::declval<Type&>() op)>) {               \
          if (x.template is<Type>()) {                                         \
            ret = DT(x.template as<Type>() op);                                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        !ret.template is<std::monostate>(),                                    \
        "Cannot compute ",                                                     \
        x.type().name(),                                                       \
        #op,                                                                   \
        " : incompatible type");                                               \
    return ret;                                                                \
  }

DEFINE_RIGHT_PPMM_IMPL(rpp, ++);
DEFINE_RIGHT_PPMM_IMPL(rmm, --);
#undef DEFINE_RIGHT_PPMM_IMPL

// Compound assignment operator implementations
#define DEFINE_ASSIGNMENT_OP_IMPL(op, assign_op)                 \
  template <typename DT, typename T, typename>                   \
  inline constexpr DT& operator assign_op(DT & x, const T & y) { \
    return x = x op y;                                           \
  }

DEFINE_ASSIGNMENT_OP_IMPL(+, +=);
DEFINE_ASSIGNMENT_OP_IMPL(-, -=);
DEFINE_ASSIGNMENT_OP_IMPL(*, *=);
DEFINE_ASSIGNMENT_OP_IMPL(/, /=);
DEFINE_ASSIGNMENT_OP_IMPL(%, %=);
DEFINE_ASSIGNMENT_OP_IMPL(&, &=);
DEFINE_ASSIGNMENT_OP_IMPL(|, |=);
DEFINE_ASSIGNMENT_OP_IMPL(^, ^=);
DEFINE_ASSIGNMENT_OP_IMPL(<<, <<=);
DEFINE_ASSIGNMENT_OP_IMPL(>>, >>=);
#undef DEFINE_ASSIGNMENT_OP_IMPL

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

// Binary operator implementations - all 24 operators
DEFINE_BINARY_OP_IMPL(add, +, operator+, DT, true);
DEFINE_BINARY_OP_IMPL(minus, -, operator-, DT, true);
DEFINE_BINARY_OP_IMPL(mul, *, operator*, DT, true);
DEFINE_BINARY_OP_IMPL(div, /, operator/, DT, true);
DEFINE_BINARY_OP_IMPL(mod, %, operator%, DT, true);
DEFINE_BINARY_OP_IMPL(band, &, operator&, DT, true);
DEFINE_BINARY_OP_IMPL(bor, |, operator|, DT, true);
DEFINE_BINARY_OP_IMPL(xor, ^, operator^, DT, true);
DEFINE_BINARY_OP_IMPL(land, &&, operator&&, DT, true);
DEFINE_BINARY_OP_IMPL(lor, ||, operator||, DT, true);
DEFINE_BINARY_OP_IMPL(lshift, <<, operator<<, DT, true);
DEFINE_BINARY_OP_IMPL(rshift, >>, operator>>, DT, true);

// Named comparison (return DT)
DEFINE_BINARY_OP_IMPL(named_eq, ==, eq, DT, true);
DEFINE_BINARY_OP_IMPL(named_neq, !=, ne, DT, true);
DEFINE_BINARY_OP_IMPL(named_lt, <, lt, DT, true);
DEFINE_BINARY_OP_IMPL(named_gt, >, gt, DT, true);
DEFINE_BINARY_OP_IMPL(named_le, <=, le, DT, true);
DEFINE_BINARY_OP_IMPL(named_ge, >=, ge, DT, true);

// Comparison (return bool)
DEFINE_BINARY_OP_IMPL(eq, ==, operator==, bool, false);
DEFINE_BINARY_OP_IMPL(neq, !=, operator!=, bool, false);
DEFINE_BINARY_OP_IMPL(lt, <, operator<, bool, false);
DEFINE_BINARY_OP_IMPL(gt, >, operator>, bool, false);
DEFINE_BINARY_OP_IMPL(le, <=, operator<=, bool, false);
DEFINE_BINARY_OP_IMPL(ge, >=, operator>=, bool, false);

#undef DEFINE_BINARY_OP_IMPL

} // namespace dynamic_type

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

