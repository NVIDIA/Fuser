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

// ============================================================================
// Index-based switch dispatch macro for binary operators
// This eliminates ForAllTypes/Void/tuple overhead by using direct switch
// dispatch based on variant index.
//
// Parameters:
//   opname      - Function name prefix (eq, neq, lt, add, etc.)
//   op          - Operator symbol (==, !=, <, +, etc.)
//   return_type - Return type (bool for comparison, DynamicType for arithmetic)
// ============================================================================

#define DEFINE_BINARY_OP_INDEX_DISPATCH(opname, op, return_type)               \
  template <typename Containers, typename... Ts>                               \
  return_type DynamicType<Containers, Ts...>::opname##_impl(                   \
      const DynamicType& a, const DynamicType& b) {             \
    /* Compute result for specific type indices I, J */                        \
    auto compute_at_indices = [&]<std::size_t I, std::size_t J>()              \
        -> std::pair<bool, std::optional<return_type>> {                       \
      using X = std::variant_alternative_t<I, VariantType>;                    \
      using Y = std::variant_alternative_t<J, VariantType>;                    \
      if constexpr ((opcheck<X> op opcheck<Y>)) {                              \
        using ResultT = decltype((std::declval<X>() op std::declval<Y>()));      \
        /* Skip if result type is DynamicType - indicates recursive call       \
           through operators returning DynamicType */                          \
        constexpr bool result_is_dt =                                          \
            std::is_same_v<std::decay_t<ResultT>, DynamicType>;                \
        /* Skip if X and Y are DIFFERENT types and one/both are                \
           constructible to DynamicType but aren't base types.                 \
           This indicates opcheck success is via implicit conversion           \
           to DynamicType, which would cause infinite recursion.               \
           If X == Y (same type), the native operator is used - safe. */       \
        constexpr bool x_is_base = (std::is_same_v<X, Ts> || ...);             \
        constexpr bool y_is_base = (std::is_same_v<Y, Ts> || ...);             \
        constexpr bool mixed_with_container =                                  \
            !std::is_same_v<X, Y> && /* different types */                     \
            ((std::is_constructible_v<DynamicType, X> && !x_is_base) ||        \
             (std::is_constructible_v<DynamicType, Y> && !y_is_base));         \
        if constexpr (!result_is_dt && !mixed_with_container &&                \
                      std::is_convertible_v<ResultT, return_type>) {           \
          return {true, static_cast<return_type>(                              \
              (std::get<I>(a.value) op std::get<J>(b.value)))};                 \
        }                                                                      \
      }                                                                        \
      return {false, std::nullopt};                                            \
    };                                                                         \
                                                                               \
    /* Inner dispatch on b's index */                                          \
    auto dispatch_b = [&]<std::size_t I, std::size_t... Js>(                   \
        std::index_sequence<Js...>)                                            \
        -> std::pair<bool, std::optional<return_type>> {                       \
      std::pair<bool, std::optional<return_type>> result{false, std::nullopt}; \
      const std::size_t b_idx = b.value.index();                               \
      ((b_idx == Js                                                            \
            ? (result = compute_at_indices.template operator()<I, Js>(), true) \
            : false) ||                                                        \
       ...);                                                                   \
      return result;                                                           \
    };                                                                         \
                                                                               \
    /* Outer dispatch on a's index */                                          \
    auto dispatch_a = [&]<std::size_t... Is>(std::index_sequence<Is...> seq)   \
        -> std::pair<bool, std::optional<return_type>> {                       \
      std::pair<bool, std::optional<return_type>> result{false, std::nullopt}; \
      const std::size_t a_idx = a.value.index();                               \
      ((a_idx == Is                                                            \
            ? (result = dispatch_b.template operator()<Is>(seq), true)         \
            : false) ||                                                        \
       ...);                                                                   \
      return result;                                                           \
    };                                                                         \
                                                                               \
    auto [found, result] = dispatch_a(std::make_index_sequence<num_types>{});  \
                                                                               \
    DYNAMIC_TYPE_CHECK(                                                        \
        found && result.has_value(),                                           \
        "Cannot compute ",                                                     \
        a.type().name(),                                                       \
        " " #op " ",                                                           \
        b.type().name());                                                      \
    return *result;                                                            \
  }

// operator== also uses template-based dispatch (converted to switch for consistency)
// Note: eq is kept using template dispatch as it was originally working.
// If linking issues occur, convert to DEFINE_BINARY_OP_SWITCH_BOOL(eq, ==) below.
DEFINE_BINARY_OP_INDEX_DISPATCH(eq, ==, bool)

#undef DEFINE_BINARY_OP_INDEX_DISPATCH

// ============================================================================
// Macro-based switch dispatch for binary operators
// Uses explicit switch statements instead of template fold expressions to
// avoid deep template nesting that crashes Clang 18.1.8.
// Supports up to 16 variant alternatives (increase cases if more needed).
// ============================================================================

// Helper: try binary op at indices I, J - works for any return type
#define SWITCH_DISPATCH_TRY(OP, I, J, RET_TYPE, result_var, found_var)         \
  do {                                                                         \
    if constexpr ((I) < num_types && (J) < num_types) {                        \
      using X = std::variant_alternative_t<(I), VariantType>;                  \
      using Y = std::variant_alternative_t<(J), VariantType>;                  \
      if constexpr (opcheck<X> OP opcheck<Y>) {                                \
        using ResultT = decltype(std::declval<X>() OP std::declval<Y>());      \
        constexpr bool result_is_dt =                                          \
            std::is_same_v<std::decay_t<ResultT>, DynamicType>;                \
        constexpr bool x_is_base = (std::is_same_v<X, Ts> || ...);             \
        constexpr bool y_is_base = (std::is_same_v<Y, Ts> || ...);             \
        constexpr bool mixed_with_container =                                  \
            !std::is_same_v<X, Y> &&                                           \
            ((std::is_constructible_v<DynamicType, X> && !x_is_base) ||        \
             (std::is_constructible_v<DynamicType, Y> && !y_is_base));         \
        if constexpr (!result_is_dt && !mixed_with_container &&                \
                      std::is_convertible_v<ResultT, RET_TYPE>) {              \
          result_var = static_cast<RET_TYPE>(                                  \
              std::get<(I)>(a.value) OP std::get<(J)>(b.value));               \
          found_var = true;                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

// Inner switch on b's index (supports up to 16 types)
#define SWITCH_DISPATCH_B(OP, I, RET_TYPE, result_var, found_var)              \
  switch (b.value.index()) {                                                   \
    case 0: SWITCH_DISPATCH_TRY(OP, I, 0, RET_TYPE, result_var, found_var); break; \
    case 1: SWITCH_DISPATCH_TRY(OP, I, 1, RET_TYPE, result_var, found_var); break; \
    case 2: SWITCH_DISPATCH_TRY(OP, I, 2, RET_TYPE, result_var, found_var); break; \
    case 3: SWITCH_DISPATCH_TRY(OP, I, 3, RET_TYPE, result_var, found_var); break; \
    case 4: SWITCH_DISPATCH_TRY(OP, I, 4, RET_TYPE, result_var, found_var); break; \
    case 5: SWITCH_DISPATCH_TRY(OP, I, 5, RET_TYPE, result_var, found_var); break; \
    case 6: SWITCH_DISPATCH_TRY(OP, I, 6, RET_TYPE, result_var, found_var); break; \
    case 7: SWITCH_DISPATCH_TRY(OP, I, 7, RET_TYPE, result_var, found_var); break; \
    case 8: SWITCH_DISPATCH_TRY(OP, I, 8, RET_TYPE, result_var, found_var); break; \
    case 9: SWITCH_DISPATCH_TRY(OP, I, 9, RET_TYPE, result_var, found_var); break; \
    case 10: SWITCH_DISPATCH_TRY(OP, I, 10, RET_TYPE, result_var, found_var); break; \
    case 11: SWITCH_DISPATCH_TRY(OP, I, 11, RET_TYPE, result_var, found_var); break; \
    case 12: SWITCH_DISPATCH_TRY(OP, I, 12, RET_TYPE, result_var, found_var); break; \
    case 13: SWITCH_DISPATCH_TRY(OP, I, 13, RET_TYPE, result_var, found_var); break; \
    case 14: SWITCH_DISPATCH_TRY(OP, I, 14, RET_TYPE, result_var, found_var); break; \
    case 15: SWITCH_DISPATCH_TRY(OP, I, 15, RET_TYPE, result_var, found_var); break; \
    default: break;                                                            \
  }

// Outer switch on a's index (supports up to 16 types)
#define SWITCH_DISPATCH_A(OP, RET_TYPE, result_var, found_var)                 \
  switch (a.value.index()) {                                                   \
    case 0: SWITCH_DISPATCH_B(OP, 0, RET_TYPE, result_var, found_var); break;  \
    case 1: SWITCH_DISPATCH_B(OP, 1, RET_TYPE, result_var, found_var); break;  \
    case 2: SWITCH_DISPATCH_B(OP, 2, RET_TYPE, result_var, found_var); break;  \
    case 3: SWITCH_DISPATCH_B(OP, 3, RET_TYPE, result_var, found_var); break;  \
    case 4: SWITCH_DISPATCH_B(OP, 4, RET_TYPE, result_var, found_var); break;  \
    case 5: SWITCH_DISPATCH_B(OP, 5, RET_TYPE, result_var, found_var); break;  \
    case 6: SWITCH_DISPATCH_B(OP, 6, RET_TYPE, result_var, found_var); break;  \
    case 7: SWITCH_DISPATCH_B(OP, 7, RET_TYPE, result_var, found_var); break;  \
    case 8: SWITCH_DISPATCH_B(OP, 8, RET_TYPE, result_var, found_var); break;  \
    case 9: SWITCH_DISPATCH_B(OP, 9, RET_TYPE, result_var, found_var); break;  \
    case 10: SWITCH_DISPATCH_B(OP, 10, RET_TYPE, result_var, found_var); break; \
    case 11: SWITCH_DISPATCH_B(OP, 11, RET_TYPE, result_var, found_var); break; \
    case 12: SWITCH_DISPATCH_B(OP, 12, RET_TYPE, result_var, found_var); break; \
    case 13: SWITCH_DISPATCH_B(OP, 13, RET_TYPE, result_var, found_var); break; \
    case 14: SWITCH_DISPATCH_B(OP, 14, RET_TYPE, result_var, found_var); break; \
    case 15: SWITCH_DISPATCH_B(OP, 15, RET_TYPE, result_var, found_var); break; \
    default: break;                                                            \
  }

// Generate a binary operator returning bool (comparison operators)
#define DEFINE_BINARY_OP_SWITCH_BOOL(opname, op)                               \
  template <typename Containers, typename... Ts>                               \
  bool DynamicType<Containers, Ts...>::opname##_impl(                          \
      const DynamicType& a, const DynamicType& b) {                            \
    static_assert(num_types <= 16,                                             \
        "Switch dispatch supports max 16 types. Increase cases in impl.h.");   \
    bool result = false;                                                       \
    bool found = false;                                                        \
    SWITCH_DISPATCH_A(op, bool, result, found);                                \
    DYNAMIC_TYPE_CHECK(found, "Cannot compute ",                               \
        a.type().name(), " " #op " ", b.type().name());                        \
    return result;                                                             \
  }

// Generate a binary operator returning DynamicType (arithmetic/bitwise)
#define DEFINE_BINARY_OP_SWITCH_DT(opname, op)                                 \
  template <typename Containers, typename... Ts>                               \
  auto DynamicType<Containers, Ts...>::opname##_impl(                          \
      const DynamicType& a, const DynamicType& b) -> DynamicType {             \
    static_assert(num_types <= 16,                                             \
        "Switch dispatch supports max 16 types. Increase cases in impl.h.");   \
    std::optional<DynamicType> result;                                         \
    bool found = false;                                                        \
    SWITCH_DISPATCH_A(op, DynamicType, result, found);                         \
    DYNAMIC_TYPE_CHECK(found && result.has_value(), "Cannot compute ",         \
        a.type().name(), " " #op " ", b.type().name());                        \
    return *result;                                                            \
  }

// Comparison operators (return bool)
DEFINE_BINARY_OP_SWITCH_BOOL(neq, !=)
DEFINE_BINARY_OP_SWITCH_BOOL(lt, <)
DEFINE_BINARY_OP_SWITCH_BOOL(gt, >)
DEFINE_BINARY_OP_SWITCH_BOOL(le, <=)
DEFINE_BINARY_OP_SWITCH_BOOL(ge, >=)

// Arithmetic operators (return DynamicType)
DEFINE_BINARY_OP_SWITCH_DT(add, +)
DEFINE_BINARY_OP_SWITCH_DT(sub, -)
DEFINE_BINARY_OP_SWITCH_DT(mul, *)
DEFINE_BINARY_OP_SWITCH_DT(div, /)
DEFINE_BINARY_OP_SWITCH_DT(mod, %)

// Bitwise operators (return DynamicType)
DEFINE_BINARY_OP_SWITCH_DT(band, &)
DEFINE_BINARY_OP_SWITCH_DT(bor, |)
DEFINE_BINARY_OP_SWITCH_DT(bxor, ^)
DEFINE_BINARY_OP_SWITCH_DT(lshift, <<)
DEFINE_BINARY_OP_SWITCH_DT(rshift, >>)

// Named comparison functions (return DynamicType)
DEFINE_BINARY_OP_SWITCH_DT(named_eq, ==)
DEFINE_BINARY_OP_SWITCH_DT(named_neq, !=)
DEFINE_BINARY_OP_SWITCH_DT(named_lt, <)
DEFINE_BINARY_OP_SWITCH_DT(named_gt, >)
DEFINE_BINARY_OP_SWITCH_DT(named_le, <=)
DEFINE_BINARY_OP_SWITCH_DT(named_ge, >=)

#undef SWITCH_DISPATCH_TRY
#undef SWITCH_DISPATCH_B
#undef SWITCH_DISPATCH_A
#undef DEFINE_BINARY_OP_SWITCH_BOOL
#undef DEFINE_BINARY_OP_SWITCH_DT

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
        using ResultT = decltype(op std::declval<Type>());                     \
        /* Skip if result type is DynamicType - indicates recursion */         \
        constexpr bool result_is_dt =                                          \
            std::is_same_v<std::decay_t<ResultT>, DynamicType>;                \
        /* Skip if Type is a container (not a base type) that converts to      \
           DynamicType - this would cause infinite recursion */                \
        constexpr bool is_base = (std::is_same_v<Type, Ts> || ...);            \
        constexpr bool converts_to_dt =                                        \
            std::is_constructible_v<DynamicType, Type>;                        \
        constexpr bool is_container_type = converts_to_dt && !is_base;         \
        if constexpr (!result_is_dt && !is_container_type &&                   \
                      std::is_constructible_v<VariantType, ResultT>) {         \
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

#undef DEFINE_UNARY_OP_FRIEND_IMPL

// Logical not - returns bool
template <typename Containers, typename... Ts>
bool DynamicType<Containers, Ts...>::lnot_impl(const DynamicType& x) {
  std::optional<bool> result;
  for_all_types([&result, &x](auto t) {
    using Type = typename decltype(t)::type;
    if constexpr (!opcheck<Type>) {
      using ResultT = decltype(!std::declval<Type>());
      // Skip if Type is a container (not a base type) that would convert
      // to DynamicType - this would cause infinite recursion
      constexpr bool is_base = (std::is_same_v<Type, Ts> || ...);
      constexpr bool converts_to_dt = std::is_constructible_v<DynamicType, Type>;
      constexpr bool is_container_type = converts_to_dt && !is_base;
      if constexpr (!is_container_type &&
                    std::is_convertible_v<ResultT, bool>) {
        if (x.template is<Type>()) {
          result = static_cast<bool>(!x.template as<Type>());
        }
      }
    }
  });
  DYNAMIC_TYPE_CHECK(
      result.has_value(),
      "Cannot compute !", x.type().name(), " : incompatible type");
  return *result;
}

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

