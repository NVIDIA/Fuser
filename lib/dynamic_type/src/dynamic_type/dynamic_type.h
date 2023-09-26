// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <algorithm>
#include <optional>
#include <ostream>
#include <type_traits>
#include <typeinfo>
#include <variant>

#include "C++20/type_traits"
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

  constexpr DynamicType() = default;

  template <typename T, typename = decltype(VariantType(std::declval<T>()))>
  constexpr DynamicType(T value) : value(std::move(value)) {}

  template <
      template <typename...>
      typename Template,
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
      template <typename...>
      typename Template,
      typename = std::enable_if_t<is_candidate_type<Template<DynamicType>>>>
  constexpr const Template<DynamicType>& as() const {
    return as<Template<DynamicType>>();
  }

  template <
      template <typename...>
      typename Template,
      typename = std::enable_if_t<is_candidate_type<Template<DynamicType>>>>
  constexpr Template<DynamicType>& as() {
    return as<Template<DynamicType>>();
  }

  template <typename T, typename = std::enable_if_t<can_cast_to<T>>>
  explicit constexpr operator T() const {
    std::optional<T> ret = std::nullopt;
    for_all_types([this, &ret](auto from) {
      using From = typename decltype(from)::type;
      if constexpr (opcheck<From>.canCastTo(opcheck<T>)) {
        if (is<From>()) {
          ret = (T)as<From>();
        }
      }
    });
    DYNAMIC_TYPE_CHECK(
        ret.has_value(),
        "Cannot cast from ",
        type().name(),
        " to ",
        typeid(T).name(),
        " : incompatible type");
    return ret.value();
  }

  template <
      template <typename...>
      typename Template,
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

  // Intentionally not overloading operator-> because it only makes sense when
  // returning pointers, however, if we have a DynamicType that can be either a
  // Type1 or Type2, then it is ambiguous to return a pointer to Type1 vs Type2

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
  constexpr decltype(auto) operator->*(Ret Class::*member) __const {           \
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
};

template <typename T>
struct is_dynamic_type : std::false_type {};

template <typename... Ts>
struct is_dynamic_type<DynamicType<Ts...>> : std::true_type {};

template <typename T>
constexpr bool is_dynamic_type_v = is_dynamic_type<T>::value;

#define DEFINE_BINARY_OP(opname, op)                                       \
  /*TODO: we should inline the definition of lambdas into enable_if,*/     \
  /*but I can only do this in C++20 */                                     \
  template <typename DTVariantType>                                        \
  constexpr auto opname##_defined_checker = [](auto x, auto y) constexpr { \
    using X = typename decltype(x)::type;                                  \
    using Y = typename decltype(y)::type;                                  \
    if constexpr (opcheck<X> op opcheck<Y>) {                              \
      return std::is_constructible_v<                                      \
          DTVariantType,                                                   \
          decltype(std::declval<X>() op std::declval<Y>())>;               \
    }                                                                      \
    return false;                                                          \
  };                                                                       \
  template <typename DT>                                                   \
  inline constexpr std::enable_if_t<                                       \
      is_dynamic_type_v<DT> &&                                             \
          any_check(                                                       \
              opname##_defined_checker<typename DT::VariantType>,          \
              DT::type_identities_as_tuple,                                \
              DT::type_identities_as_tuple),                               \
      DT>                                                                  \
  operator op(const DT& x, const std::type_identity_t<DT>& y) {            \
    DT ret(std::monostate{});                                              \
    DT::for_all_types([&ret, &x, &y](auto lhs) {                           \
      using LHS = typename decltype(lhs)::type;                            \
      DT::for_all_types([&ret, &x, &y](auto rhs) {                         \
        using RHS = typename decltype(rhs)::type;                          \
        if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                    \
          if constexpr (std::is_constructible_v<                           \
                            typename DT::VariantType,                      \
                            decltype(std::declval<LHS>()                   \
                                         op std::declval<RHS>())>) {       \
            if (x.template is<LHS>() && y.template is<RHS>()) {            \
              ret = DT(x.template as<LHS>() op y.template as<RHS>());      \
            }                                                              \
          }                                                                \
        }                                                                  \
      });                                                                  \
    });                                                                    \
    DYNAMIC_TYPE_CHECK(                                                    \
        !ret.template is<std::monostate>(),                                \
        "Cannot compute ",                                                 \
        x.type().name(),                                                   \
        " ",                                                               \
        #op,                                                               \
        " ",                                                               \
        y.type().name(),                                                   \
        " : incompatible type");                                           \
    return ret;                                                            \
  }                                                                        \
  /*TODO: we should inline the definition of lambdas into enable_if,*/     \
  /*but I can only do this in C++20 */                                     \
  template <typename RHS, typename DTVariantType>                          \
  constexpr auto opname##_rdefined_checker = [](auto x) constexpr {        \
    using X = typename decltype(x)::type;                                  \
    if constexpr (opcheck<X> op opcheck<RHS>) {                            \
      return std::is_constructible_v<                                      \
          DTVariantType,                                                   \
          decltype(std::declval<X>() op std::declval<RHS>())>;             \
    }                                                                      \
    return false;                                                          \
  };                                                                       \
  template <typename DT, typename RHS>                                     \
  inline constexpr std::enable_if_t<                                       \
      is_dynamic_type_v<DT> && !is_dynamic_type_v<RHS> &&                  \
          any_check(                                                       \
              opname##_rdefined_checker<RHS, typename DT::VariantType>,    \
              DT::type_identities_as_tuple),                               \
      DT>                                                                  \
  operator op(const DT& x, const RHS& y) {                                 \
    DT ret(std::monostate{});                                              \
    DT::for_all_types([&ret, &x, &y](auto lhs) {                           \
      using LHS = typename decltype(lhs)::type;                            \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                      \
        if constexpr (std::is_constructible_v<                             \
                          typename DT::VariantType,                        \
                          decltype(std::declval<LHS>()                     \
                                       op std::declval<RHS>())>) {         \
          if (x.template is<LHS>()) {                                      \
            ret = DT(x.template as<LHS>() op y);                           \
          }                                                                \
        }                                                                  \
      }                                                                    \
    });                                                                    \
    DYNAMIC_TYPE_CHECK(                                                    \
        !ret.template is<std::monostate>(),                                \
        "Cannot compute ",                                                 \
        x.type().name(),                                                   \
        " ",                                                               \
        #op,                                                               \
        " ",                                                               \
        typeid(RHS).name(),                                                \
        " : incompatible type");                                           \
    return ret;                                                            \
  }                                                                        \
  /*TODO: we should inline the definition of lambdas into enable_if,*/     \
  /*but I can only do this in C++20 */                                     \
  template <typename LHS, typename DTVariantType>                          \
  constexpr auto opname##_ldefined_checker = [](auto y) constexpr {        \
    using Y = typename decltype(y)::type;                                  \
    if constexpr (opcheck<LHS> op opcheck<Y>) {                            \
      return std::is_constructible_v<                                      \
          DTVariantType,                                                   \
          decltype(std::declval<LHS>() op std::declval<Y>())>;             \
    }                                                                      \
    return false;                                                          \
  };                                                                       \
  template <typename LHS, typename DT>                                     \
  inline constexpr std::enable_if_t<                                       \
      is_dynamic_type_v<DT> && !is_dynamic_type_v<LHS> &&                  \
          (opcheck<LHS>.hasExplicitCastTo(opcheck<DT>) ||                  \
           any_check(                                                      \
               opname##_ldefined_checker<LHS, typename DT::VariantType>,   \
               DT::type_identities_as_tuple)),                             \
      DT>                                                                  \
  operator op(const LHS& x, const DT& y) {                                 \
    DT ret(std::monostate{});                                              \
    DT::for_all_types([&ret, &x, &y](auto rhs) {                           \
      using RHS = typename decltype(rhs)::type;                            \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                      \
        if constexpr (std::is_constructible_v<                             \
                          typename DT::VariantType,                        \
                          decltype(std::declval<LHS>()                     \
                                       op std::declval<RHS>())>) {         \
          if (y.template is<RHS>()) {                                      \
            ret = DT(x op y.template as<RHS>());                           \
          }                                                                \
        }                                                                  \
      }                                                                    \
    });                                                                    \
    if (ret.hasValue()) {                                                  \
      return ret;                                                          \
    }                                                                      \
    if constexpr (opcheck<LHS>.hasExplicitCastTo(opcheck<DT>)) {           \
      return (DT)x op y;                                                   \
    }                                                                      \
    DYNAMIC_TYPE_CHECK(                                                    \
        false,                                                             \
        "Cannot compute ",                                                 \
        typeid(LHS).name(),                                                \
        " ",                                                               \
        #op,                                                               \
        " ",                                                               \
        y.type().name(),                                                   \
        " : incompatible type");                                           \
    return ret;                                                            \
  }

DEFINE_BINARY_OP(add, +);
DEFINE_BINARY_OP(minus, -);
DEFINE_BINARY_OP(mul, *);
DEFINE_BINARY_OP(div, /);
DEFINE_BINARY_OP(mod, %);
DEFINE_BINARY_OP(band, &);
DEFINE_BINARY_OP(bor, |);
DEFINE_BINARY_OP(xor, ^);
DEFINE_BINARY_OP(land, &&);
DEFINE_BINARY_OP(lor, ||);
DEFINE_BINARY_OP(lshift, <<);
DEFINE_BINARY_OP(rshift, >>);

#undef DEFINE_BINARY_OP

#define DEFINE_COMPARE_OP(opname, op)                                          \
  /*TODO: we should inline the definition of lambdas into enable_if,*/         \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_defined_checker = [](auto x, auto y) constexpr {     \
    using X = typename decltype(x)::type;                                      \
    using Y = typename decltype(y)::type;                                      \
    if constexpr (opcheck<X> op opcheck<Y>) {                                  \
      return std::is_convertible_v<                                            \
          decltype(std::declval<X>() op std::declval<Y>()),                    \
          bool>;                                                               \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename = std::enable_if_t<                                             \
          is_dynamic_type_v<DT> &&                                             \
          any_check(                                                           \
              opname##_defined_checker,                                        \
              DT::type_identities_as_tuple,                                    \
              DT::type_identities_as_tuple)>>                                  \
  inline constexpr bool operator op(                                           \
      const DT& x, const std::type_identity_t<DT>& y) {                        \
    std::optional<bool> ret = std::nullopt;                                    \
    DT::for_all_types([&ret, &x, &y](auto lhs) {                               \
      using LHS = typename decltype(lhs)::type;                                \
      DT::for_all_types([&ret, &x, &y](auto rhs) {                             \
        using RHS = typename decltype(rhs)::type;                              \
        if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                        \
          if constexpr (std::is_convertible_v<                                 \
                            decltype(std::declval<LHS>()                       \
                                         op std::declval<RHS>()),              \
                            bool>) {                                           \
            if (x.template is<LHS>() && y.template is<RHS>()) {                \
              ret = x.template as<LHS>() op y.template as<RHS>();              \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        ret.has_value(),                                                       \
        "Cannot compute ",                                                     \
        x.type().name(),                                                       \
        " ",                                                                   \
        #op,                                                                   \
        " ",                                                                   \
        y.type().name(),                                                       \
        " : incompatible type");                                               \
    return ret.value();                                                        \
  }                                                                            \
  /*TODO: we should inline the definition of lambdas into enable_if,*/         \
  /*but I can only do this in C++20 */                                         \
  template <typename T>                                                        \
  constexpr auto opname##_rdefined_checker = [](auto x) constexpr {            \
    using X = typename decltype(x)::type;                                      \
    if constexpr (opcheck<X> op opcheck<T>) {                                  \
      return std::is_convertible_v<                                            \
          decltype(std::declval<X>() op std::declval<T>()),                    \
          bool>;                                                               \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename RHS,                                                            \
      typename = std::enable_if_t<                                             \
          is_dynamic_type_v<DT> && !is_dynamic_type_v<RHS> &&                  \
          any_check(                                                           \
              opname##_rdefined_checker<RHS>, DT::type_identities_as_tuple)>>  \
  inline constexpr bool operator op(const DT& x, const RHS& y) {               \
    std::optional<bool> ret = std::nullopt;                                    \
    DT::for_all_types([&ret, &x, &y](auto lhs) {                               \
      using LHS = typename decltype(lhs)::type;                                \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                          \
        if constexpr (std::is_convertible_v<                                   \
                          decltype(std::declval<LHS>()                         \
                                       op std::declval<RHS>()),                \
                          bool>) {                                             \
          if (x.template is<LHS>()) {                                          \
            ret = x.template as<LHS>() op y;                                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        ret.has_value(),                                                       \
        "Cannot compute ",                                                     \
        x.type().name(),                                                       \
        " ",                                                                   \
        #op,                                                                   \
        " ",                                                                   \
        typeid(RHS).name(),                                                    \
        " : incompatible type");                                               \
    return ret.value();                                                        \
  }                                                                            \
  /*TODO: we should inline the definition of lambdas into enable_if,*/         \
  /*but I can only do this in C++20 */                                         \
  template <typename T>                                                        \
  constexpr auto opname##_ldefined_checker = [](auto y) constexpr {            \
    using Y = typename decltype(y)::type;                                      \
    if constexpr (opcheck<T> op opcheck<Y>) {                                  \
      return std::is_convertible_v<                                            \
          decltype(std::declval<T>() op std::declval<Y>()),                    \
          bool>;                                                               \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <typename LHS, typename DT>                                         \
  inline constexpr std::enable_if_t<                                           \
      is_dynamic_type_v<DT> && !is_dynamic_type_v<LHS> &&                      \
          (opcheck<LHS>.hasExplicitCastTo(opcheck<DT>) ||                      \
           any_check(                                                          \
               opname##_ldefined_checker<LHS>, DT::type_identities_as_tuple)), \
      bool>                                                                    \
  operator op(const LHS& x, const DT& y) {                                     \
    std::optional<bool> ret = std::nullopt;                                    \
    DT::for_all_types([&ret, &x, &y](auto rhs) {                               \
      using RHS = typename decltype(rhs)::type;                                \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                          \
        if constexpr (std::is_convertible_v<                                   \
                          decltype(std::declval<LHS>()                         \
                                       op std::declval<RHS>()),                \
                          bool>) {                                             \
          if (y.template is<RHS>()) {                                          \
            ret = x op y.template as<RHS>();                                   \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    if (ret.has_value()) {                                                     \
      return ret.value();                                                      \
    }                                                                          \
    if constexpr (opcheck<LHS>.hasExplicitCastTo(opcheck<DT>)) {               \
      return (DT)x op y;                                                       \
    }                                                                          \
    DYNAMIC_TYPE_CHECK(                                                        \
        false,                                                                 \
        "Cannot compute ",                                                     \
        typeid(LHS).name(),                                                    \
        " ",                                                                   \
        #op,                                                                   \
        " ",                                                                   \
        y.type().name(),                                                       \
        " : incompatible type");                                               \
  }

DEFINE_COMPARE_OP(eq, ==);
DEFINE_COMPARE_OP(neq, !=);
DEFINE_COMPARE_OP(lt, <);
DEFINE_COMPARE_OP(gt, >);
DEFINE_COMPARE_OP(le, <=);
DEFINE_COMPARE_OP(ge, >=);

#undef DEFINE_COMPARE_OP

#define DEFINE_UNARY_OP(opname, op)                                            \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  template <typename DTVariantType>                                            \
  constexpr auto opname##_helper = [](auto x) constexpr {                      \
    using X = typename decltype(x)::type;                                      \
    if constexpr (op opcheck<X>) {                                             \
      return std::                                                             \
          is_constructible_v<DTVariantType, decltype(op std::declval<X>())>;   \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <typename DT>                                                       \
  inline constexpr std::enable_if_t<                                           \
      is_dynamic_type_v<DT> &&                                                 \
          any_check(                                                           \
              opname##_helper<typename DT::VariantType>,                       \
              DT::type_identities_as_tuple),                                   \
      DT>                                                                      \
  operator op(const DT& x) {                                                   \
    DT ret(std::monostate{});                                                  \
    DT::for_all_types([&ret, &x](auto _) {                                     \
      using Type = typename decltype(_)::type;                                 \
      if constexpr (op opcheck<Type>) {                                        \
        if constexpr (std::is_constructible_v<                                 \
                          typename DT::VariantType,                            \
                          decltype(op std::declval<Type>())>) {                \
          if (x.template is<Type>()) {                                         \
            ret = DT(op x.template as<Type>());                                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    DYNAMIC_TYPE_CHECK(                                                        \
        !ret.template is<std::monostate>(),                                    \
        "Cannot compute ",                                                     \
        #op,                                                                   \
        x.type().name(),                                                       \
        " : incompatible type");                                               \
    return ret;                                                                \
  }

DEFINE_UNARY_OP(pos, +);
DEFINE_UNARY_OP(neg, -);
DEFINE_UNARY_OP(bnot, ~);
DEFINE_UNARY_OP(lnot, !);

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

template <
    typename DT,
    typename = std::enable_if_t<
        is_dynamic_type_v<DT> &&
        any_check(star_defined_checker<DT>, DT::type_identities_as_tuple)>>
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

#undef DEFINE_UNARY_OP

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
template <
    typename DT,
    typename = std::enable_if_t<
        is_dynamic_type_v<DT> &&
        any_check(can_print, DT::type_identities_as_tuple)>>
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

#define DEFINE_LEFT_PPMM(opname, op)                                           \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_helper = [](auto x) constexpr {                      \
    using X = typename decltype(x)::type;                                      \
    if constexpr (op opcheck<X&>) {                                            \
      return std::is_same_v<decltype(op std::declval<X&>()), X&>;              \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename = std::enable_if_t<                                             \
          is_dynamic_type_v<DT> &&                                             \
          any_check(opname##_helper, DT::type_identities_as_tuple)>>           \
  inline constexpr DT& operator op(DT& x) {                                    \
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

DEFINE_LEFT_PPMM(lpp, ++);
DEFINE_LEFT_PPMM(lmm, --);

#undef DEFINE_LEFT_PPMM

#define DEFINE_RIGHT_PPMM(opname, op)                                          \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  template <typename DTVariantType>                                            \
  constexpr auto opname##_helper = [](auto x) constexpr {                      \
    using X = typename decltype(x)::type;                                      \
    if constexpr (opcheck<X&> op) {                                            \
      return std::                                                             \
          is_constructible_v<DTVariantType, decltype(std::declval<X&>() op)>;  \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <typename DT>                                                       \
  inline constexpr std::enable_if_t<                                           \
      is_dynamic_type_v<DT> &&                                                 \
          any_check(                                                           \
              opname##_helper<typename DT::VariantType>,                       \
              DT::type_identities_as_tuple),                                   \
      DT>                                                                      \
  operator op(DT& x, int) {                                                    \
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

DEFINE_RIGHT_PPMM(rpp, ++);
DEFINE_RIGHT_PPMM(rmm, --);

#undef DEFINE_RIGHT_PPMM

#define DEFINE_ASSIGNMENT_OP(op, assign_op)                      \
  template <                                                     \
      typename DT,                                               \
      typename T,                                                \
      typename = std::enable_if_t<                               \
          is_dynamic_type_v<DT> && (opcheck<DT> op opcheck<T>)>> \
  inline constexpr DT& operator assign_op(DT& x, const T& y) {   \
    return x = x op y;                                           \
  }

DEFINE_ASSIGNMENT_OP(+, +=);
DEFINE_ASSIGNMENT_OP(-, -=);
DEFINE_ASSIGNMENT_OP(*, *=);
DEFINE_ASSIGNMENT_OP(/, /=);
DEFINE_ASSIGNMENT_OP(%, %=);
DEFINE_ASSIGNMENT_OP(&, &=);
DEFINE_ASSIGNMENT_OP(|, |=);
DEFINE_ASSIGNMENT_OP(^, ^=);
DEFINE_ASSIGNMENT_OP(<<, <<=);
DEFINE_ASSIGNMENT_OP(>>, >>=);

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
      "Hash function of DynamicType can not be automatically defined while there are cross-type equality.");
  using DT = dynamic_type::DynamicType<Containers, Ts...>;
  std::size_t operator()(DT const& dt) const noexcept {
    return std::hash<typename DT::VariantType>{}(dt.value);
  }
};
