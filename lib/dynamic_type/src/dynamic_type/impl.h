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

} // namespace dynamic_type

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

