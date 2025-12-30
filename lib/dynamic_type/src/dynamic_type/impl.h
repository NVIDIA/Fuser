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

} // namespace dynamic_type

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

