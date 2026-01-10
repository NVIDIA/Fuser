// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <variant>

#include "error.h"
#include "type_traits.h"

// Visibility attribute for exported symbols.
// Static member functions need default visibility to be exported from
// shared libraries built with -fvisibility=hidden.
#if defined _WIN32 || defined __CYGWIN__
#define DT_API __declspec(dllexport)
#else
#define DT_API __attribute__((visibility("default")))
#endif

namespace dynamic_type {

// We must disable a lot of compiler warnings to make this work. The reason for
// the need to disable these warnings is not because the code quality in this
// file is bad, but because these apparently "bad" practices are necessary. For
// example, if you have a dynamic type that can be either a bool or a class
// SomeType{}, then we should support the ~ operator on it, because in the C++
// standard bool supports it. Usually, when people write code like ~bool, they
// are making a mistake, and the compiler will want you to use !bool instead.
// However, in our case here we will allow everything that the C++ standard
// allows. The compiler should yell at the user who uses DynamicType with ~
// but not at us for implementing it.

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wbool-operation"
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbool-operation"
// gcc, even the latest version (13.1.1), is complaining about the following
// code:
//   std::optional<bool> ret = std::nullopt;
//   ...
//   DYNAMIC_TYPE_CHECK(ret.has_value(), ...);
//   return ret.value();
// saying that ret.value() is used uninitialized. This complaint is totoally
// nonsense.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

template <template <typename...> typename... Templates>
// Note: `Templates` is a list of templates, not a list of types.
// Just like std::vector is a template, std::vector<int> is a type.
struct Containers {
  template <typename DynamicType, typename... MemberTypes>
  using VariantType =
      std::variant<std::monostate, MemberTypes..., Templates<DynamicType>...>;

  template <typename DynamicType, typename... MemberTypes>
  using TypeIdentitiesAsTuple = std::tuple<
      std::type_identity<std::monostate>,
      std::type_identity<MemberTypes>...,
      std::type_identity<Templates<DynamicType>>...>;

  template <typename DynamicType, typename... MemberTypes>
  using ForAllTypes = dynamic_type::
      ForAllTypes<std::monostate, MemberTypes..., Templates<DynamicType>...>;

  // Check if T is one of the types in the type list MemberTypes..., or a
  // container
  template <typename T, typename DynamicType, typename... MemberTypes>
  static constexpr auto is_candidate_type = dynamic_type::
      belongs_to<T, std::monostate, MemberTypes..., Templates<DynamicType>...>;

  template <typename ItemT>
  using ForAllContainerTypes = dynamic_type::ForAllTypes<Templates<ItemT>...>;

  template <typename ItemT>
  static constexpr auto
  all_container_type_identities_constructible_from_initializer_list() {
    return dynamic_type::remove_void_from_tuple(ForAllContainerTypes<
                                                ItemT>{}([](auto t) {
      using T = typename decltype(t)::type;
      if constexpr (std::is_constructible_v<T, std::initializer_list<ItemT>>) {
        return std::type_identity<T>{};
      } else {
        return;
      }
    }));
  }

  template <typename ItemT>
  using AllContainerTypeIdentitiesConstructibleFromInitializerList =
      decltype(all_container_type_identities_constructible_from_initializer_list<
               ItemT>());
};

using NoContainers = Containers<>;

template <typename Containers, typename... Ts>
struct DynamicType {
  using VariantType =
      typename Containers::template VariantType<DynamicType, Ts...>;
  VariantType value;

  using TypeIdentitiesAsTuple =
      typename Containers::template TypeIdentitiesAsTuple<DynamicType, Ts...>;
  static constexpr TypeIdentitiesAsTuple type_identities_as_tuple{};

  static constexpr std::size_t num_types =
      std::tuple_size_v<TypeIdentitiesAsTuple>;

  using ForAllTypes =
      typename Containers::template ForAllTypes<DynamicType, Ts...>;
  static constexpr ForAllTypes for_all_types{};

  // Check if T is one of the types in the type list Ts... or a container
  template <typename T>
  static constexpr auto is_candidate_type =
      Containers::template is_candidate_type<T, DynamicType, Ts...>;

  template <typename T>
  static constexpr bool can_cast_to = any_check(
      [](auto t) {
        return opcheck<typename decltype(t)::type>.canCastTo(opcheck<T>);
      },
      type_identities_as_tuple);

  template <typename ItemT>
  using AllContainerTypeIdentitiesConstructibleFromInitializerList =
      typename Containers::
          template AllContainerTypeIdentitiesConstructibleFromInitializerList<
              ItemT>;

  template <typename ItemT>
  static constexpr auto
      num_container_types_constructible_from_initializer_list =
          std::tuple_size_v<
              AllContainerTypeIdentitiesConstructibleFromInitializerList<
                  ItemT>>;

  // ============================================================================
  // Switch-based dispatch helpers for dispatch() method.
  // Uses explicit return type template parameter - no inference overhead.
  // ============================================================================

  // Helper macro: try to execute f0 at index I for void return
#define DISPATCH_EXEC_VOID(I, f0_ref, arg_ref)                                 \
  case I: {                                                                    \
    if constexpr ((I) < num_types) {                                           \
      using T = std::variant_alternative_t<(I), VariantType>;                  \
      f0_ref(arg_ref.template as<T>());                                        \
    }                                                                          \
    break;                                                                     \
  }

  // Helper macro: try to execute f0 at index I for value/reference return
#define DISPATCH_EXEC_VALUE(I, f0_ref, arg_ref, ret_ref, result_type)          \
  case I: {                                                                    \
    if constexpr ((I) < num_types) {                                           \
      using T = std::variant_alternative_t<(I), VariantType>;                  \
      const T& a0 = arg_ref.template as<T>();                                  \
      if constexpr (std::is_convertible_v<decltype(f0_ref(a0)), result_type>) {\
        ret_ref = f0_ref(a0);                                                  \
      } else {                                                                 \
        DYNAMIC_TYPE_CHECK(                                                    \
            false, "Result is dynamic but not convertible to result type");    \
      }                                                                        \
    }                                                                          \
    break;                                                                     \
  }

  // Helper macro: try to execute f0 at index I for DynamicType return
#define DISPATCH_EXEC_DYNAMIC(I, f0_ref, arg_ref, ret_ref)                     \
  case I: {                                                                    \
    if constexpr ((I) < num_types) {                                           \
      using T = std::variant_alternative_t<(I), VariantType>;                  \
      const T& a0 = arg_ref.template as<T>();                                  \
      using CallRetT = decltype(f0_ref(a0));                                   \
      if constexpr (!std::is_void_v<CallRetT>) {                               \
        ret_ref = DynamicType(f0_ref(a0));                                     \
      } else {                                                                 \
        DYNAMIC_TYPE_CHECK(                                                    \
            false, "Result is dynamic but not convertible to result type");    \
      }                                                                        \
    }                                                                          \
    break;                                                                     \
  }

  // Switch for void return (up to 16 types)
#define DISPATCH_SWITCH_VOID(f0_ref, arg_ref)                                  \
  switch (arg_ref.value.index()) {                                             \
    DISPATCH_EXEC_VOID(0, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(1, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(2, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(3, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(4, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(5, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(6, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(7, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(8, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(9, f0_ref, arg_ref)                                     \
    DISPATCH_EXEC_VOID(10, f0_ref, arg_ref)                                    \
    DISPATCH_EXEC_VOID(11, f0_ref, arg_ref)                                    \
    DISPATCH_EXEC_VOID(12, f0_ref, arg_ref)                                    \
    DISPATCH_EXEC_VOID(13, f0_ref, arg_ref)                                    \
    DISPATCH_EXEC_VOID(14, f0_ref, arg_ref)                                    \
    DISPATCH_EXEC_VOID(15, f0_ref, arg_ref)                                    \
    default: break;                                                            \
  }

  // Switch for value/reference return (up to 16 types)
#define DISPATCH_SWITCH_VALUE(f0_ref, arg_ref, ret_ref, result_type)           \
  switch (arg_ref.value.index()) {                                             \
    DISPATCH_EXEC_VALUE(0, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(1, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(2, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(3, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(4, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(5, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(6, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(7, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(8, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(9, f0_ref, arg_ref, ret_ref, result_type)              \
    DISPATCH_EXEC_VALUE(10, f0_ref, arg_ref, ret_ref, result_type)             \
    DISPATCH_EXEC_VALUE(11, f0_ref, arg_ref, ret_ref, result_type)             \
    DISPATCH_EXEC_VALUE(12, f0_ref, arg_ref, ret_ref, result_type)             \
    DISPATCH_EXEC_VALUE(13, f0_ref, arg_ref, ret_ref, result_type)             \
    DISPATCH_EXEC_VALUE(14, f0_ref, arg_ref, ret_ref, result_type)             \
    DISPATCH_EXEC_VALUE(15, f0_ref, arg_ref, ret_ref, result_type)             \
    default: break;                                                            \
  }

  // Switch for DynamicType return (up to 16 types)
#define DISPATCH_SWITCH_DYNAMIC(f0_ref, arg_ref, ret_ref)                      \
  switch (arg_ref.value.index()) {                                             \
    DISPATCH_EXEC_DYNAMIC(0, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(1, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(2, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(3, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(4, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(5, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(6, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(7, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(8, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(9, f0_ref, arg_ref, ret_ref)                         \
    DISPATCH_EXEC_DYNAMIC(10, f0_ref, arg_ref, ret_ref)                        \
    DISPATCH_EXEC_DYNAMIC(11, f0_ref, arg_ref, ret_ref)                        \
    DISPATCH_EXEC_DYNAMIC(12, f0_ref, arg_ref, ret_ref)                        \
    DISPATCH_EXEC_DYNAMIC(13, f0_ref, arg_ref, ret_ref)                        \
    DISPATCH_EXEC_DYNAMIC(14, f0_ref, arg_ref, ret_ref)                        \
    DISPATCH_EXEC_DYNAMIC(15, f0_ref, arg_ref, ret_ref)                        \
    default: break;                                                            \
  }

  // ============================================================================
  // dispatch() - Requires explicit return type template parameter.
  //
  // Usage:
  //   dispatch<ReturnT>(lambda, dynamicArg1, dynamicArg2, ...)
  //
  // ReturnT can be:
  //   - void: No return value
  //   - DynamicType: Wrap result in DynamicType
  //   - Any other type: Direct return (int64_t, bool, std::string, etc.)
  // ============================================================================
  template <typename ReturnT, typename FuncT, typename FirstArg, typename... OtherArgs>
  static inline constexpr ReturnT dispatch(
      FuncT&& f,
      FirstArg&& arg0,
      OtherArgs&&... args) {
    static_assert(
        num_types <= 16,
        "dispatch() supports max 16 types. Increase switch cases in decl.h.");

    // Recursively dispatch on `args`, only leaving arg0 as undispatched
    // argument
    auto f0 = [&](auto&& a0) -> decltype(auto) {
      if constexpr (sizeof...(OtherArgs) == 0) {
        return std::forward<FuncT>(f)(std::forward<decltype(a0)>(a0));
      } else {
        auto f_others = [&](auto&&... others) -> decltype(auto) {
          return std::forward<FuncT>(f)(
              std::forward<decltype(a0)>(a0),
              std::forward<decltype(others)>(others)...);
        };
        return dispatch<ReturnT>(f_others, std::forward<OtherArgs>(args)...);
      }
    };

    // Does arg0 need dispatch?
    if constexpr (std::is_same_v<std::decay_t<FirstArg>, DynamicType>) {
      // Dispatch based on explicit return type
      if constexpr (std::is_void_v<ReturnT>) {
        // Void return path
        DISPATCH_SWITCH_VOID(f0, arg0)
        return;
      } else if constexpr (std::is_same_v<std::decay_t<ReturnT>, DynamicType>) {
        // DynamicType return path - wrap results
        DynamicType ret{};
        DISPATCH_SWITCH_DYNAMIC(f0, arg0, ret)
        return ret;
      } else {
        // Direct type return path (int64_t, bool, std::string, references, etc.)
        using result_type = ReturnT;
        // Needs to wrap reference as optional<reference_wrapper<T>> because
        // C++ does not allow rebinding a reference.
        constexpr bool is_reference = std::is_reference_v<result_type>;
        using ret_storage_t = std::conditional_t<
            is_reference,
            std::optional<
                std::reference_wrapper<std::remove_reference_t<result_type>>>,
            result_type>;
        ret_storage_t ret{};
        DISPATCH_SWITCH_VALUE(f0, arg0, ret, result_type)
        if constexpr (is_reference) {
          return ret->get();
        } else {
          return ret;
        }
      }
    } else {
      // No need to dispatch arg0, just perfectly forwarding it.
      if constexpr (std::is_void_v<ReturnT>) {
        f0(std::forward<FirstArg>(arg0));
        return;
      } else {
        using f0_return_t = decltype(f0(std::forward<FirstArg>(arg0)));
        if constexpr (std::is_void_v<f0_return_t>) {
          // Lambda returns void but we need a return value
          f0(std::forward<FirstArg>(arg0));
          if constexpr (std::is_reference_v<ReturnT>) {
            // Can't default-construct a reference; this is a type mismatch error
            DYNAMIC_TYPE_CHECK(
                false, "Lambda returned void but reference return type expected");
            // Unreachable, but needed for compilation
            return *static_cast<std::remove_reference_t<ReturnT>*>(nullptr);
          } else {
            return ReturnT{};
          }
        } else {
          return f0(std::forward<FirstArg>(arg0));
        }
      }
    }
  }

  // NOTE: dispatch_deduce() has been removed for compile-time performance.
  // It was previously used for operator->() and SFINAE checks but:
  // - operator->() was removed (use .as<T>() instead)
  // - SFINAE checks now use pure constexpr any_check()

#undef DISPATCH_EXEC_VOID
#undef DISPATCH_EXEC_VALUE
#undef DISPATCH_EXEC_DYNAMIC
#undef DISPATCH_SWITCH_VOID
#undef DISPATCH_SWITCH_VALUE
#undef DISPATCH_SWITCH_DYNAMIC

  constexpr DynamicType() = default;

  template <typename T, typename = decltype(VariantType(std::declval<T>()))>
  constexpr DynamicType(T&& value) : value(std::forward<T>(value)) {}

  template <
      template <typename...> typename Template,
      typename ItemT,
      typename = std::enable_if_t<
          is_candidate_type<Template<DynamicType>> &&
          !std::is_same_v<ItemT, DynamicType>>>
  constexpr DynamicType(Template<ItemT> value)
      : value([](auto input) {
          Template<DynamicType> result;
          std::transform(
              input.begin(),
              input.end(),
              std::back_inserter(result),
              [](auto& item) { return DynamicType(std::move(item)); });
          return result;
        }(std::move(value))) {}

  template <
      typename ItemT = DynamicType,
      typename = std::enable_if_t<
          // enable this ctor only when there is only one container supporting
          // initializer_list, otherwise it is ambiguous to tell which container
          // to use.
          num_container_types_constructible_from_initializer_list<ItemT> == 1>>
  constexpr DynamicType(std::initializer_list<DynamicType> list)
      : DynamicType(typename std::tuple_element_t<
                    0,
                    AllContainerTypeIdentitiesConstructibleFromInitializerList<
                        DynamicType>>::type(list)) {}

  // Returns the type_info of the actual type of the variant value. For
  // example, if value holds an int, then this will return typeid(int).
  const std::type_info& type() const {
    return std::visit(
        [](auto value) -> const std::type_info& { return typeid(value); },
        value);
  }

  template <typename T>
  constexpr bool is() const {
    return std::holds_alternative<T>(value);
  }

  template <template <typename...> typename Template>
  constexpr bool is() const {
    return is<Template<DynamicType>>();
  }

  constexpr bool isNull() const {
    return std::holds_alternative<std::monostate>(value);
  }

  constexpr bool hasValue() const {
    return !isNull();
  }

  template <typename T, typename = std::enable_if_t<is_candidate_type<T>>>
  constexpr const T& as() const {
    return std::get<T>(value);
  }

  template <typename T, typename = std::enable_if_t<is_candidate_type<T>>>
  constexpr T& as() {
    return std::get<T>(value);
  }

  template <
      template <typename...> typename Template,
      typename = std::enable_if_t<is_candidate_type<Template<DynamicType>>>>
  constexpr const Template<DynamicType>& as() const {
    return as<Template<DynamicType>>();
  }

  template <
      template <typename...> typename Template,
      typename = std::enable_if_t<is_candidate_type<Template<DynamicType>>>>
  constexpr Template<DynamicType>& as() {
    return as<Template<DynamicType>>();
  }

  template <typename T, typename = std::enable_if_t<can_cast_to<T>>>
  explicit constexpr operator T() const {
    return dispatch<T>(
        [](auto x) -> decltype(auto) {
          using X = decltype(x);
          if constexpr (opcheck<X>.canCastTo(opcheck<T>)) {
            return (T)x;
          }
        },
        *this);
  }

  template <
      template <typename...> typename Template,
      typename ItemT,
      typename = std::enable_if_t<
          is_candidate_type<Template<DynamicType>> && can_cast_to<ItemT>>>
  explicit constexpr operator Template<ItemT>() const {
    DYNAMIC_TYPE_CHECK(
        is<Template<DynamicType>>(),
        "Cannot cast from ",
        type().name(),
        " to ",
        typeid(Template<ItemT>).name(),
        " : incompatible type");
    Template<ItemT> result;
    std::transform(
        as<Template<DynamicType>>().begin(),
        as<Template<DynamicType>>().end(),
        std::back_inserter(result),
        [](const auto& item) { return (ItemT)item; });
    return result;
  }

  // Intentionally not overloading operator=, because the compiler generated
  // default behavior usually makes more sense than the overloaded one. For
  // example, if we have
  //   struct SomeType {};
  //   using IntOrCustom = DynamicType<int, SomeType>;
  //   IntOrCustom x(1);
  //   IntOrCustom y(SomeType{});
  //   x = y;
  // Then the compiler generated behavior will get us SomeType{} for x, but if
  // we overload based on the underlying type, we will get a runtime error,
  // because it is not possible to assign SomeType{} to an int.

  // NOTE: operator->() was removed because:
  // 1. It required dispatch_deduce() which is expensive at compile time
  // 2. nvFuser production code doesn't use it (uses .as<T>() instead)
  // 3. For pointer access, use: dt.as<T*>()->member
  // 4. For smart pointers, use: dt.as<std::shared_ptr<T>>()->member

  template <typename IndexT>
  static constexpr bool has_square_bracket = any_check(
      [](auto t) {
        using T = typename decltype(t)::type;
        if constexpr (opcheck<T>[opcheck<IndexT>]) {
          return std::is_same_v<
              decltype(std::declval<T>()[std::declval<IndexT>()]),
              DynamicType&>;
        }
        return false;
      },
      type_identities_as_tuple);

#define DEFINE_SQUARE_BRACKET_OPERATOR(__const)                                \
  template <typename IndexT>                                                   \
  std::enable_if_t<                                                            \
      !std::is_same_v<IndexT, DynamicType> && has_square_bracket<IndexT>,      \
      __const DynamicType&>                                                    \
  operator[](const IndexT& i) __const {                                        \
    std::optional<std::reference_wrapper<__const DynamicType>> ret =           \
        std::nullopt;                                                          \
    for_all_types([this, &ret, &i](auto t) {                                   \
      using T = typename decltype(t)::type;                                    \
      if constexpr (opcheck<T>[opcheck<IndexT>]) {                             \
        if constexpr (std::is_same_v<                                          \
                          decltype(std::declval<T>()[std::declval<IndexT>()]), \
                          DynamicType&>) {                                     \
          if (is<T>()) {                                                       \
            ret = std::ref(as<T>()[i]);                                        \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        ret.has_value(),                                                       \
        "Cannot index ",                                                       \
        type().name(),                                                         \
        " with ",                                                              \
        typeid(IndexT).name(),                                                 \
        " : incompatible type");                                               \
    return ret.value();                                                        \
  }

  DEFINE_SQUARE_BRACKET_OPERATOR()
  DEFINE_SQUARE_BRACKET_OPERATOR(const)
#undef DEFINE_SQUARE_BRACKET_OPERATOR

  static constexpr bool has_any_square_bracket = any_check(
      [](auto t) {
        using IndexT = typename decltype(t)::type;
        return has_square_bracket<IndexT>;
      },
      type_identities_as_tuple);

#define DEFINE_SQUARE_BRACKET_OPERATOR(__const)                      \
  template <typename DT>                                             \
  std::enable_if_t<                                                  \
      std::is_same_v<DT, DynamicType> && has_any_square_bracket,     \
      __const DynamicType&>                                          \
  operator[](const DT& i) __const {                                  \
    std::optional<std::reference_wrapper<__const DynamicType>> ret = \
        std::nullopt;                                                \
    for_all_types([this, &ret, &i](auto t) {                         \
      using IndexT = typename decltype(t)::type;                     \
      if constexpr (has_square_bracket<IndexT>) {                    \
        if (i.template is<IndexT>()) {                               \
          ret = std::ref((*this)[i.template as<IndexT>()]);          \
        }                                                            \
      }                                                              \
    });                                                              \
    DYNAMIC_TYPE_CHECK(                                              \
        ret.has_value(),                                             \
        "Cannot index ",                                             \
        type().name(),                                               \
        " with ",                                                    \
        i.type().name(),                                             \
        " : incompatible type");                                     \
    return ret.value();                                              \
  }

  DEFINE_SQUARE_BRACKET_OPERATOR()
  DEFINE_SQUARE_BRACKET_OPERATOR(const)
#undef DEFINE_SQUARE_BRACKET_OPERATOR

  // ->* over for accessing candidate members. This will be converted as a .*
  // with a candidate type. For example, if you have:
  // DynamicType<NoContainers, A, B, C> abc;
  // then you can use abc->*A::x to access the member x of A. Member access also
  // support functions, just make sure that you get the correct precedence. For
  // example: use (abc->*A::print)() instead of abc->*A::print().

#define DEFINE_ARROW_STAR_OPERATOR(__const)                                    \
  template <                                                                   \
      typename Ret,                                                            \
      typename Class,                                                          \
      typename = std::enable_if_t<is_candidate_type<Class>>>                   \
  constexpr decltype(auto) operator->*(Ret Class::* member) __const {          \
    /* Use decltype(auto) instead of auto as return type so that references */ \
    /* and qualifiers are preserved*/                                          \
    if constexpr (std::is_function_v<Ret>) {                                   \
      return [this, member](auto&&... args) {                                  \
        return (as<Class>().*member)(std::forward<decltype(args)>(args)...);   \
      };                                                                       \
    } else {                                                                   \
      return as<Class>().*member;                                              \
    }                                                                          \
  }

  DEFINE_ARROW_STAR_OPERATOR()
  DEFINE_ARROW_STAR_OPERATOR(const)
#undef DEFINE_ARROW_STAR_OPERATOR

  // ->* operator for non-candidate access. This will just forward the argument
  // to the overloaded ->* of candidates. Due to limitations of C++'s type
  // system, we can only enable this when all the types in the type list that
  // support this operator have the same return type.

#define DEFINE_ARROW_STAR_OPERATOR(__const)                                     \
  template <typename MemberT>                                                   \
  static constexpr auto all_arrow_star_ret_types##__const =                     \
      remove_void_from_tuple(for_all_types([](auto t) {                         \
        using T = typename decltype(t)::type;                                   \
        if constexpr (opcheck<T>->*opcheck<MemberT>) {                          \
          return std::type_identity<                                            \
              decltype(std::declval<__const T>()->*std::declval<MemberT>())>{}; \
        }                                                                       \
      }));                                                                      \
                                                                                \
  template <typename MemberT>                                                   \
  using AllArrowStarRetTypes##__const =                                         \
      decltype(all_arrow_star_ret_types##__const<MemberT>);                     \
                                                                                \
  template <typename MemberT>                                                   \
  static constexpr bool all_arrow_star_ret_types_are_same##__const =            \
      all_same_type(all_arrow_star_ret_types##__const<MemberT>);                \
                                                                                \
  template <typename MemberT>                                                   \
  using ArrowStarRetType##__const =                                             \
      typename first_or_void<AllArrowStarRetTypes##__const<MemberT>>::type;     \
                                                                                \
  template <typename MemberT>                                                   \
  constexpr std::enable_if_t<                                                   \
      all_arrow_star_ret_types_are_same##__const<MemberT>,                      \
      typename ArrowStarRetType##__const<MemberT>::type>                        \
  operator->*(const MemberT& member) __const {                                  \
    using RetT = typename ArrowStarRetType##__const<MemberT>::type;             \
    std::optional<wrap_reference_t<RetT>> ret = std::nullopt;                   \
    for_all_types([this, &member, &ret](auto t) {                               \
      using T = typename decltype(t)::type;                                     \
      if constexpr (opcheck<T>->*opcheck<MemberT>) {                            \
        if (is<T>()) {                                                          \
          ret = as<T>()->*member;                                               \
        }                                                                       \
      }                                                                         \
    });                                                                         \
    DYNAMIC_TYPE_CHECK(                                                         \
        ret.has_value(),                                                        \
        "Cannot access member with type ",                                      \
        typeid(RetT).name(),                                                    \
        " : incompatible type");                                                \
    return ret.value();                                                         \
  }

  DEFINE_ARROW_STAR_OPERATOR()
  DEFINE_ARROW_STAR_OPERATOR(const)
#undef DEFINE_ARROW_STAR_OPERATOR

  // TODO: support operator(). This is not supported yet because it is the most
  // difficulty one to implement because it can has arbitrary number of
  // arguments. I believe it is doable, but I decide to leave it for future.

  // ========================================================================
  // Friend operators with static member implementations
  // - Static _impl functions: declarations here, definitions in impl.h
  // - Friend operators: trivial forwarding, fine to inline everywhere
  // - Mixed types work via implicit constructor
  // - Static members are covered by extern template
  // ========================================================================

  // General macro for binary operators (works for both operators and named functions)
  // return_type: bool for comparison operators, DynamicType for arithmetic
  // func_name: operator+ or named function like eq, add, etc.
  // DT_API ensures the static member is exported from shared libraries.
#define DEFINE_BINARY_OP_FRIEND(opname, op, func_name, return_type)            \
  DT_API static return_type opname##_impl(const DynamicType& a, const DynamicType& b);\
  friend return_type func_name(const DynamicType& a, const DynamicType& b) {   \
    return opname##_impl(a, b);                                                \
  }

  // Comparison operators (return bool)
  DEFINE_BINARY_OP_FRIEND(eq, ==, operator==, bool)
  DEFINE_BINARY_OP_FRIEND(neq, !=, operator!=, bool)
  DEFINE_BINARY_OP_FRIEND(lt, <, operator<, bool)
  DEFINE_BINARY_OP_FRIEND(gt, >, operator>, bool)
  DEFINE_BINARY_OP_FRIEND(le, <=, operator<=, bool)
  DEFINE_BINARY_OP_FRIEND(ge, >=, operator>=, bool)

  // Arithmetic operators (return DynamicType)
  DEFINE_BINARY_OP_FRIEND(add, +, operator+, DynamicType)
  DEFINE_BINARY_OP_FRIEND(sub, -, operator-, DynamicType)
  DEFINE_BINARY_OP_FRIEND(mul, *, operator*, DynamicType)
  DEFINE_BINARY_OP_FRIEND(div, /, operator/, DynamicType)
  DEFINE_BINARY_OP_FRIEND(mod, %, operator%, DynamicType)
  DEFINE_BINARY_OP_FRIEND(band, &, operator&, DynamicType)
  DEFINE_BINARY_OP_FRIEND(bor, |, operator|, DynamicType)
  DEFINE_BINARY_OP_FRIEND(bxor, ^, operator^, DynamicType)
  // NOTE: operator&& and operator|| are kept as template functions (below)
  // to avoid ambiguity with built-in bool && bool when one operand is bool
  DEFINE_BINARY_OP_FRIEND(lshift, <<, operator<<, DynamicType)
  DEFINE_BINARY_OP_FRIEND(rshift, >>, operator>>, DynamicType)

  // Named comparison functions (return DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_eq, ==, eq, DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_neq, !=, ne, DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_lt, <, lt, DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_gt, >, gt, DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_le, <=, le, DynamicType)
  DEFINE_BINARY_OP_FRIEND(named_ge, >=, ge, DynamicType)

#undef DEFINE_BINARY_OP_FRIEND

  // Unary operators (return DynamicType)
#define DEFINE_UNARY_OP_FRIEND(opname, op)                                     \
  DT_API static DynamicType opname##_impl(const DynamicType& x);               \
  friend DynamicType operator op(const DynamicType& x) {                       \
    return opname##_impl(x);                                                   \
  }

  DEFINE_UNARY_OP_FRIEND(pos, +)
  DEFINE_UNARY_OP_FRIEND(neg, -)
  DEFINE_UNARY_OP_FRIEND(bnot, ~)

#undef DEFINE_UNARY_OP_FRIEND

  // Logical not - returns bool
  DT_API static bool lnot_impl(const DynamicType& x);
  friend bool operator!(const DynamicType& x) { return lnot_impl(x); }

  // Prefix increment/decrement (++x, --x) - return reference
  DT_API static DynamicType& lpp_impl(DynamicType& x);  // ++x
  DT_API static DynamicType& lmm_impl(DynamicType& x);  // --x
  friend DynamicType& operator++(DynamicType& x) { return lpp_impl(x); }
  friend DynamicType& operator--(DynamicType& x) { return lmm_impl(x); }

  // Postfix increment/decrement (x++, x--) - return copy of original
  DT_API static DynamicType rpp_impl(DynamicType& x);  // x++
  DT_API static DynamicType rmm_impl(DynamicType& x);  // x--
  friend DynamicType operator++(DynamicType& x, int) { return rpp_impl(x); }
  friend DynamicType operator--(DynamicType& x, int) { return rmm_impl(x); }

  // Compound assignment operators - use the binary operators
  friend DynamicType& operator+=(DynamicType& x, const DynamicType& y) { return x = x + y; }
  friend DynamicType& operator-=(DynamicType& x, const DynamicType& y) { return x = x - y; }
  friend DynamicType& operator*=(DynamicType& x, const DynamicType& y) { return x = x * y; }
  friend DynamicType& operator/=(DynamicType& x, const DynamicType& y) { return x = x / y; }
  friend DynamicType& operator%=(DynamicType& x, const DynamicType& y) { return x = x % y; }
  friend DynamicType& operator&=(DynamicType& x, const DynamicType& y) { return x = x & y; }
  friend DynamicType& operator|=(DynamicType& x, const DynamicType& y) { return x = x | y; }
  friend DynamicType& operator^=(DynamicType& x, const DynamicType& y) { return x = x ^ y; }
  friend DynamicType& operator<<=(DynamicType& x, const DynamicType& y) { return x = x << y; }
  friend DynamicType& operator>>=(DynamicType& x, const DynamicType& y) { return x = x >> y; }
};

template <typename T>
struct is_dynamic_type : std::false_type {};

template <typename... Ts>
struct is_dynamic_type<DynamicType<Ts...>> : std::true_type {};

template <typename T>
constexpr bool is_dynamic_type_v = is_dynamic_type<T>::value;

// Helper to get type identities tuple - uses if constexpr for lazy evaluation
template <typename T, bool is_dt>
struct get_type_identities {
  // Non-DynamicType case: wrap single type in tuple
  using type = std::tuple<std::type_identity<T>>;
};

template <typename T>
struct get_type_identities<T, true> {
  // DynamicType case: use its TypeIdentitiesAsTuple
  using type = typename T::TypeIdentitiesAsTuple;
};

template <typename T>
using get_type_identities_t =
    typename get_type_identities<std::decay_t<T>, is_dynamic_type_v<std::decay_t<T>>>::type;

// Declaration macro for binary operators - implementation in impl.h
#define DEFINE_BINARY_OP_DECL(opname, op, func_name, return_type, check_existence) \
  template <typename X, typename Y, typename RetT>                             \
  constexpr bool opname##_type_compatible() {                                  \
    if constexpr (opcheck<X> op opcheck<Y>) {                                  \
      if constexpr (std::is_convertible_v<                                     \
                        decltype(std::declval<X>() op std::declval<Y>()),      \
                        RetT>) {                                               \
        return true;                                                           \
      }                                                                        \
    }                                                                          \
    return false;                                                              \
  }                                                                            \
  template <typename LHS, typename RHS>                                        \
  constexpr bool opname##_defined() {                                          \
    constexpr bool lhs_is_dt = is_dynamic_type_v<std::decay_t<LHS>>;           \
    constexpr bool rhs_is_dt = is_dynamic_type_v<std::decay_t<RHS>>;           \
    using DT =                                                                 \
        std::conditional_t<lhs_is_dt, std::decay_t<LHS>, std::decay_t<RHS>>;   \
    if constexpr (!lhs_is_dt && !rhs_is_dt) {                                  \
      return false;                                                            \
    } else if constexpr (                                                      \
        (lhs_is_dt && !rhs_is_dt &&                                            \
         opcheck<std::decay_t<RHS>>.hasExplicitCastTo(                         \
             opcheck<std::decay_t<LHS>>)) ||                                   \
        (!lhs_is_dt && rhs_is_dt &&                                            \
         opcheck<std::decay_t<LHS>>.hasExplicitCastTo(                         \
             opcheck<std::decay_t<RHS>>))) {                                   \
      return opname##_defined<DT, DT>();                                       \
    } else {                                                                   \
      if constexpr (check_existence) {                                         \
        /* Pure constexpr check using any_check - no dispatch_deduce needed */ \
        /* Uses get_type_identities_t for lazy evaluation to avoid           */ \
        /* instantiating TypeIdentitiesAsTuple on non-DynamicType types.     */ \
        using lhs_types = get_type_identities_t<LHS>;                          \
        using rhs_types = get_type_identities_t<RHS>;                          \
        return any_check(                                                      \
            [](auto lhs_t, auto rhs_t) constexpr {                             \
              using L = typename decltype(lhs_t)::type;                        \
              using R = typename decltype(rhs_t)::type;                        \
              return opname##_type_compatible<L, R, DT>();                     \
            },                                                                 \
            lhs_types{}, rhs_types{});                                         \
      } else {                                                                 \
        return true;                                                           \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  template <                                                                   \
      typename LHS,                                                            \
      typename RHS,                                                            \
      typename DT = std::conditional_t<                                        \
          is_dynamic_type_v<std::decay_t<LHS>>,                                \
          std::decay_t<LHS>,                                                   \
          std::decay_t<RHS>>,                                                  \
      typename = std::enable_if_t<opname##_defined<LHS, RHS>()>>               \
  inline constexpr return_type func_name(LHS&& x, RHS&& y);

// NOTE: Most binary operators are now friend functions inside DynamicType class.
// Only operator&& and operator|| remain as templates to avoid ambiguity with
// built-in bool && bool when one operand is bool.
DEFINE_BINARY_OP_DECL(land, &&, operator&&, DT, true);
DEFINE_BINARY_OP_DECL(lor, ||, operator||, DT, true);

#undef DEFINE_BINARY_OP_DECL

// NOTE: Unary operators (+, -, ~, !) are now friend functions inside DynamicType class.

// Intentionally not supporting the following unary ops:
// DEFINE_UNARY_OP(addr, &);
// Because it only makes sense if and only if both T& and T* are included in
// the type list, however, std::variant does not allow reference type to be
// an alternative. Also, if we overloaded the operator&, how can we get the
// address of the dynamic type itself?

template <typename DT>
auto star_defined_checker = [](auto t) {
  using T = typename decltype(t)::type;
  if constexpr (*opcheck<T>) {
    return std::is_same_v<decltype(*std::declval<T>()), DT&>;
  }
  return false;
};

// Declaration only - implementation in impl.h
template <
    typename DT,
    typename = std::enable_if_t<
        is_dynamic_type_v<DT> &&
        any_check(star_defined_checker<DT>, DT::type_identities_as_tuple)>>
DT& operator*(const DT& x);

// Printing
// TODO: we should inline the definition of can_print into enable_if, but I can
// only do this in C++20
constexpr auto can_print = [](auto x) constexpr {
  using T = typename decltype(x)::type;
  if constexpr (opcheck<std::ostream&> << opcheck<T>) {
    return std::is_same_v<
        decltype(std::declval<std::ostream&>() << std::declval<T>()),
        std::ostream&>;
  }
  return false;
};

// Declaration only - implementation in impl.h
template <
    typename DT,
    typename = std::enable_if_t<
        is_dynamic_type_v<DT> &&
        any_check(can_print, DT::type_identities_as_tuple)>>
std::ostream& operator<<(std::ostream& os, const DT& dt);

// NOTE: Prefix/postfix ++/-- and compound assignment operators are now
// friend functions inside DynamicType class.

// Intentionally not overloading operator comma",". This operator is rarely
// overloaded, and the automatically defined version by the compiler usually
// does what we want.

// Check that, whether there exist two different types T and U, where both T and
// U are contained in the type list of dynamic type DT, and T == U is defined.
template <typename DT>
constexpr bool has_cross_type_equality =
    any(remove_void_from_tuple(DT::for_all_types([](auto t) {
      using T = typename decltype(t)::type;
      return any(remove_void_from_tuple(DT::for_all_types([](auto u) {
        using U = typename decltype(u)::type;
        if constexpr (std::is_same_v<T, U>) {
          return;
        } else {
          return opcheck<T> == opcheck<U>;
        }
      })));
    })));

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#undef DT_API

} // namespace dynamic_type

// Hashing:

template <typename Containers, typename... Ts>
struct std::hash<dynamic_type::DynamicType<Containers, Ts...>> {
  // The hashing should be consistent with the equality operator. That is, if
  // a == b, then a and b should always has the same hash. However, because we
  // are using the hashing function for std::variant as our hasing function,
  // there is no way for us to guarantee this if there are cross-type
  // equality. For example, 0 == 0.0, but they don't have the same hash value.
  // So the hashing function for DynamicType<NoContainers, int, double> as
  // defined here is illegal.
  static_assert(
      !dynamic_type::has_cross_type_equality<
          dynamic_type::DynamicType<Containers, Ts...>>,
      "Hash function of DynamicType can not be automatically defined while "
      "there are cross-type equality.");
  using DT = dynamic_type::DynamicType<Containers, Ts...>;
  std::size_t operator()(DT const& dt) const noexcept {
    return std::hash<typename DT::VariantType>{}(dt.value);
  }
};
