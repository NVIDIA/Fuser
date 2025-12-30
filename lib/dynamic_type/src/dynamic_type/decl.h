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

  template <typename FuncT, typename FirstArg, typename... OtherArgs>
  static inline constexpr decltype(auto) dispatch(
      FuncT&& f,
      FirstArg&& arg0,
      OtherArgs&&... args) {
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
        return dispatch(f_others, std::forward<OtherArgs>(args)...);
      }
    };
    // Does arg0 need dispatch?
    if constexpr (std::is_same_v<std::decay_t<FirstArg>, DynamicType>) {
      // Infer return result: if f always returns the same type, then we return
      // the same type as well. Otherwise, we return DynamicType assuming that
      // DynamicType is the common holder of these types. Void is treated
      // specially here: if for some case the function returns some type, and
      // for other cases the function returns void, then we ignore void and use
      // the cases with return value for inference. We decide to do this because
      // non-void return values can be ignored, but void returning can never
      // pass any information. There is no single best inference strategy that
      // fits all cases, ignoring void seems to be good tradeoff.
      auto get_single_result_type = [](auto t) {
        using T = typename decltype(t)::type;
        using RetT = decltype(f0(std::declval<T>()));
        if constexpr (!std::is_void_v<RetT>) {
          return std::type_identity<RetT>{};
        } else {
          // return void instead of std::type_identity<void> so that we can use
          // remove_void_from_tuple to remove it.
          return;
        }
      };
      using result_types = decltype(remove_void_from_tuple(
          DynamicType::for_all_types(get_single_result_type)));
      constexpr bool returns_void = (std::tuple_size_v<result_types> == 0);
      if constexpr (returns_void) {
        DynamicType::for_all_types([&](auto t) -> decltype(auto) {
          using T = typename decltype(t)::type;
          if (arg0.template is<T>()) {
            f0(arg0.template as<T>());
          }
        });
        return;
      } else {
        constexpr bool has_single_return_type =
            are_all_same<result_types>::value;
        using result_type = std::conditional_t<
            has_single_return_type,
            typename std::tuple_element_t<0, result_types>::type,
            DynamicType>;
        // Needs to wrap reference as optional<reference_wrapper<T>> because
        // C++ does not allow rebinding a reference.
        constexpr bool is_reference = std::is_reference_v<result_type>;
        using ret_storage_t = std::conditional_t<
            is_reference,
            std::optional<
                std::reference_wrapper<std::remove_reference_t<result_type>>>,
            result_type>;
        ret_storage_t ret{};
        DynamicType::for_all_types([&](auto t) -> decltype(auto) {
          using T = typename decltype(t)::type;
          if (arg0.template is<T>()) {
            const T& a0 = arg0.template as<T>();
            if constexpr (std::
                              is_convertible_v<decltype(f0(a0)), result_type>) {
              ret = f0(a0);
            } else {
              DYNAMIC_TYPE_CHECK(
                  false,
                  "Result is dynamic but not convertible to result type");
            }
          }
        });
        if constexpr (is_reference) {
          return ret->get();
        } else {
          return ret;
        }
      }
    } else {
      // No need to dispatch arg0, just perfectly forwarding it.
      return f0(std::forward<FirstArg>(arg0));
    }
  }

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
    return dispatch(
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

  constexpr decltype(auto) operator->() {
    return dispatch(
        [](auto&& x) -> decltype(auto) {
          using X = decltype(x);
          using XD = std::decay_t<X>;
          if constexpr (std::is_pointer_v<XD>) {
            return (std::decay_t<X>)(x);
          } else if constexpr (opcheck<XD>->value()) {
            return std::forward<X>(x).operator->();
          }
        },
        *this);
  }

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
#define DEFINE_BINARY_OP_FRIEND(opname, op, func_name, return_type)            \
  static return_type opname##_impl(const DynamicType& a, const DynamicType& b);\
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
  static DynamicType opname##_impl(const DynamicType& x);                      \
  friend DynamicType operator op(const DynamicType& x) {                       \
    return opname##_impl(x);                                                   \
  }

  DEFINE_UNARY_OP_FRIEND(pos, +)
  DEFINE_UNARY_OP_FRIEND(neg, -)
  DEFINE_UNARY_OP_FRIEND(bnot, ~)
  DEFINE_UNARY_OP_FRIEND(lnot, !)

#undef DEFINE_UNARY_OP_FRIEND

  // Prefix increment/decrement (++x, --x) - return reference
  static DynamicType& lpp_impl(DynamicType& x);  // ++x
  static DynamicType& lmm_impl(DynamicType& x);  // --x
  friend DynamicType& operator++(DynamicType& x) { return lpp_impl(x); }
  friend DynamicType& operator--(DynamicType& x) { return lmm_impl(x); }

  // Postfix increment/decrement (x++, x--) - return copy of original
  static DynamicType rpp_impl(DynamicType& x);  // x++
  static DynamicType rmm_impl(DynamicType& x);  // x--
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
  template <typename RetT>                                                     \
  constexpr auto opname##_is_valid = [](auto&& x, auto&& y) {                  \
    using X = decltype(x);                                                     \
    using Y = decltype(y);                                                     \
    if constexpr (opname##_type_compatible<X, Y, RetT>()) {                    \
      return std::true_type{};                                                 \
    } else {                                                                   \
      return;                                                                  \
    }                                                                          \
  };                                                                           \
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
        using should_define_t = decltype(DT::dispatch(                         \
            opname##_is_valid<DT>, std::declval<LHS>(), std::declval<RHS>())); \
        return std::is_same_v<should_define_t, std::true_type>;                \
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
