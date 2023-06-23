// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <type_traits.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <optional>
#include <type_traits>
#include <variant>

// Note [Design of DynamicType]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// DynamicType is a type that can be one of a set of types. It is similar to
// std::variant, but it is designed to be used in a way that is more similar to
// how dynamic types are used in Python. For example, in Python, you can do
// something like this:
//   x = 1; y = 2.5; z = x + y
// and z will be a dynamic float. However in C++, you will not be able to do:
//   using IntOrFloat = std::variant<int, float>;
//   IntOrFloat x = 1; IntOrFloat y = 2.5f; IntOrFloat z = x + y;
// because the operator+ on std::variant is not defined. The goal of DynamicType
// is to fill this gap. So you can do:
//   using IntOrFloat = DynamicType<int, float>;
//   IntOrFloat x = 1; IntOrFloat y = 2.5f; IntOrFloat z = x + y;
//
// The design purpose of DynamicType is to allow the user to forget about the
// actual type as much as possible, and use operators seamlessly just like if
// they are using Python. DynamicType should support arbitrary types, including
// user-defined types, pointers, but excluding references, due to the limitation
// of the C++ standard. The definition of operators on DynamicType should be
// automatic. For example, if you have:
//   struct CustomType {};
//   using IntOrFloatOrCustom = DynamicType<int, float, CustomType>;
// The the operator+ on IntOrFloatOrCustom should be defined, and it should be
// equivalent to one of the following:
//  - operator+(int, int)
//  - operator+(float, float)
//  - operator+(int, float)
//  - operator+(float, int)
// depending on the actual type of the DynamicType. If the actual type is
// CustomType which does not have operator+, or if the value is null, then this
// is a runtime error.
// However, if have:
//   struct CustomType2 {};
//   using Custom12 = DynamicType<CustomType, CustomType2>;
// Then the operator+ on Custom12 should not be defined at compile time, and
// doing Custom12{} + Custom12{} results in a compilation error. It is a
// compilation error because we know at compile time that none of them are
// defined:
//  - operator+(CustomType, CustomType)
//  - operator+(CustomType, CustomType2)
//  - operator+(CustomType2, CustomType)
//  - operator+(CustomType2, CustomType2)
// So we decide to not create the operator+ for Custom12.
//
// Also, besides requiring operator+(T1, T2) to be defined for some T1 and T2 in
// the type list, it is also required that the result type of operator+(T1, T2)
// is also in the type list. For example, if you have:
//   struct bfloat16_zero {}; struct half_zero {};
//   float operator+(bfloat16_zero, half_zero) { return 0.0f; }
//   using BFloatOrHalfZero = DynamicType<bfloat16_zero, half_zero>;
// Then the operator+ on BFloatOrHalf should not be defined, because the result
// type is not in the type list. However, if you have:
//   using BFloatOrHalfZeroOrInt = DynamicType<bfloat16_zero, half_zero, int>;
// Then the operator+ on BFloatOrHalfZeroOrInt should be defined at compile time
// because int+int is defined, but
// BFloatOrHalfZeroOrInt(half_zero{}) + BFloatOrHalfZeroOrInt(bfloat16_zero{})
// should be a runtime error, because the the result of half_zero+bfloat16_zero,
// i.e. float, is not in the type list.
//
// Besides the operators within DynamicType, such as DynamicType + DynamicType,
// DynamicType also support operators with static type. For example, if you have
//   IntOrFloat x = 1; float y = 2.5f;
// then x + y or y + x should be an IntOrFloat with value 3.5f. However, if you
// have
//   IntOrFloat x = 1; double y = 2.5;
// then you will get a compilation error for doing x + y or y + x, because int +
// double and double + int are double, which is not in the list of types of
// IntOrFloat.
//
// All the above behaviors are handled by template meta-programming, so they are
// automatic. Adding a new type to the list of types does not introduce any
// extra work. All the behaviors mentioned in this note is tested in
// DynamicTypeTest.ExamplesInNote, so if you want to change anything in this
// doc, please make sure to update the test as well.
//
// Also, operations on DynamicType should be as constexpr as possible. So most
// tests in DynamicTypeTest are static_assert tests.

namespace nvfuser {

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
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbool-operation"
#endif

template <typename... Ts>
struct DynamicType;

template <typename... Ts>
// not using template <typename... Ts> to make sure there is at least one type
struct DynamicType {
  std::variant<std::monostate, Ts...> value_;

  using TypesAsTuple = std::tuple<Ts...>;
  static constexpr TypesAsTuple types_as_tuple{};
  using ForAllTypes = nvfuser::ForAllTypes<Ts...>;
  static constexpr ForAllTypes for_all_types{};

  // Check if T is one of the types in the type list Ts...
  template <typename T>
  static constexpr auto is_candidate_type = nvfuser::belongs_to<T, Ts...>;

  template <typename T>
  static constexpr bool can_cast_to = any_check(
      [](auto t) { return opcheck<decltype(t)>.canCastTo(opcheck<T>); },
      types_as_tuple);

  constexpr DynamicType() = default;

  template <typename T>
  constexpr DynamicType(T value) : value_(value) {}

  template <typename T>
  constexpr bool is() const {
    return std::holds_alternative<T>(value_);
  }

  constexpr bool isNull() const {
    return std::holds_alternative<std::monostate>(value_);
  }

  constexpr bool hasValue() const {
    return !isNull();
  }

  template <typename T, typename = std::enable_if_t<is_candidate_type<T>>>
  constexpr T as() const {
    return std::get<T>(value_);
  }

  template <typename T, typename = std::enable_if_t<is_candidate_type<T>>>
  constexpr T& as() {
    return std::get<T>(value_);
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
    TORCH_CHECK(ret.has_value(), "Cannot cast to ", typeid(T).name());
    return ret.value();
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

  // Intentionally not supporting operator[], because this has similar issue as
  // operator* for requiring reference type to be in the type list, which is not
  // allowed by the C++ standard. On the other hand, instead of supporting
  // things like std::vector<int> being in the type list, we are more interested
  // in having recursive dynamic types, i.e. std::vector<DynamicType> being in
  // the type list. In this case, operator[] will behave differently from the
  // case of std::vector<int>. We need to think more about the interface before
  // implementing it.

  // TODO: support ->* operator. This operator is rarely used, so we don't
  // implement it yet. But if in the future, it turns to be useful, we should
  // implement it.

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

#define DEFINE_BINARY_OP(opname, op)                                        \
  /*TODO: we should inline the definition of lambdas into enable_if,*/      \
  /*but I can only do this in C++20 */                                      \
  constexpr auto opname##_defined_checker =                                 \
      [](auto x, auto y, auto z) constexpr {                                \
        if constexpr (opcheck<decltype(x)> op opcheck<decltype(y)>) {       \
          return std::is_same_v<decltype(x op y), decltype(z)>;             \
        }                                                                   \
        return false;                                                       \
      };                                                                    \
  template <                                                                \
      typename DT,                                                          \
      typename = std::enable_if_t<any_check(                                \
          opname##_defined_checker,                                         \
          DT::types_as_tuple,                                               \
          DT::types_as_tuple,                                               \
          DT::types_as_tuple)>>                                             \
  inline constexpr DT operator op(DT x, DT y) {                             \
    DT ret(std::monostate{});                                               \
    DT::for_all_types([&ret, x, y](auto lhs) {                              \
      using LHS = typename decltype(lhs)::type;                             \
      DT::for_all_types([&ret, x, y](auto rhs) {                            \
        using RHS = typename decltype(rhs)::type;                           \
        if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                     \
          if constexpr (DT::template is_candidate_type<                     \
                            decltype(std::declval<LHS>()                    \
                                         op std::declval<RHS>())>) {        \
            if (x.template is<LHS>() && y.template is<RHS>()) {             \
              ret = DT(x.template as<LHS>() op y.template as<RHS>());       \
            }                                                               \
          }                                                                 \
        }                                                                   \
      });                                                                   \
    });                                                                     \
    TORCH_CHECK(                                                            \
        !ret.template is<std::monostate>(),                                 \
        "Can not compute ",                                                 \
        #op,                                                                \
        " : incompatible type");                                            \
    return ret;                                                             \
  }                                                                         \
  /*TODO: we should inline the definition of lambdas into enable_if,*/      \
  /*but I can only do this in C++20 */                                      \
  template <typename T>                                                     \
  constexpr auto opname##_rdefined_checker = [](auto x, auto z) constexpr { \
    if constexpr (opcheck<decltype(x)> op opcheck<T>) {                     \
      return std::is_same_v<decltype(x op std::declval<T>()), decltype(z)>; \
    }                                                                       \
    return false;                                                           \
  };                                                                        \
  template <                                                                \
      typename DT,                                                          \
      typename RHS,                                                         \
      typename = std::enable_if_t<any_check(                                \
          opname##_rdefined_checker<RHS>,                                   \
          DT::types_as_tuple,                                               \
          DT::types_as_tuple)>>                                             \
  inline constexpr DT operator op(DT x, RHS y) {                            \
    DT ret(std::monostate{});                                               \
    DT::for_all_types([&ret, x, y](auto lhs) {                              \
      using LHS = typename decltype(lhs)::type;                             \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                       \
        if constexpr (DT::template is_candidate_type<                       \
                          decltype(std::declval<LHS>()                      \
                                       op std::declval<RHS>())>) {          \
          if (x.template is<LHS>()) {                                       \
            ret = DT(x.template as<LHS>() op y);                            \
          }                                                                 \
        }                                                                   \
      }                                                                     \
    });                                                                     \
    TORCH_CHECK(                                                            \
        !ret.template is<std::monostate>(),                                 \
        "Can not compute ",                                                 \
        #op,                                                                \
        " : incompatible type");                                            \
    return ret;                                                             \
  }                                                                         \
  /*TODO: we should inline the definition of lambdas into enable_if,*/      \
  /*but I can only do this in C++20 */                                      \
  template <typename T>                                                     \
  constexpr auto opname##_ldefined_checker = [](auto y, auto z) constexpr { \
    if constexpr (opcheck<T> op opcheck<decltype(y)>) {                     \
      return std::is_same_v<decltype(std::declval<T>() op y), decltype(z)>; \
    }                                                                       \
    return false;                                                           \
  };                                                                        \
  template <                                                                \
      typename LHS,                                                         \
      typename DT,                                                          \
      typename = std::enable_if_t<any_check(                                \
          opname##_ldefined_checker<LHS>,                                   \
          DT::types_as_tuple,                                               \
          DT::types_as_tuple)>>                                             \
  inline constexpr DT operator op(LHS x, DT y) {                            \
    DT ret(std::monostate{});                                               \
    DT::for_all_types([&ret, x, y](auto rhs) {                              \
      using RHS = typename decltype(rhs)::type;                             \
      if constexpr ((opcheck<LHS> op opcheck<RHS>)) {                       \
        if constexpr (DT::template is_candidate_type<                       \
                          decltype(std::declval<LHS>()                      \
                                       op std::declval<RHS>())>) {          \
          if (y.template is<RHS>()) {                                       \
            ret = DT(x op y.template as<RHS>());                            \
          }                                                                 \
        }                                                                   \
      }                                                                     \
    });                                                                     \
    TORCH_CHECK(                                                            \
        !ret.template is<std::monostate>(),                                 \
        "Can not compute ",                                                 \
        #op,                                                                \
        " : incompatible type");                                            \
    return ret;                                                             \
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
DEFINE_BINARY_OP(eq, ==);
DEFINE_BINARY_OP(neq, !=);
DEFINE_BINARY_OP(lt, <);
DEFINE_BINARY_OP(gt, >);
DEFINE_BINARY_OP(le, <=);
DEFINE_BINARY_OP(ge, >=);

#undef DEFINE_BINARY_OP

#define DEFINE_UNARY_OP(opname, op)                                            \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_helper = [](auto x, auto y) constexpr {              \
    if constexpr (op opcheck<decltype(x)>) {                                   \
      return std::is_same_v<decltype(op x), decltype(y)>;                      \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename = std::enable_if_t<any_check(                                   \
          opname##_helper, DT::types_as_tuple, DT::types_as_tuple)>>           \
  inline constexpr DT operator op(DT x) {                                      \
    DT ret(std::monostate{});                                                  \
    DT::for_all_types([&ret, x](auto _) {                                      \
      using Type = typename decltype(_)::type;                                 \
      if constexpr (op opcheck<Type>) {                                        \
        if constexpr (DT::template is_candidate_type<                          \
                          decltype(op std::declval<Type>())>) {                \
          if (x.template is<Type>()) {                                         \
            ret = DT(op x.template as<Type>());                                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    TORCH_CHECK(                                                               \
        !ret.template is<std::monostate>(),                                    \
        "Can not compute ",                                                    \
        #op,                                                                   \
        " : incompatible type");                                               \
    return ret;                                                                \
  }

DEFINE_UNARY_OP(pos, +);
DEFINE_UNARY_OP(neg, -);
DEFINE_UNARY_OP(bnot, ~);
DEFINE_UNARY_OP(lnot, !);

// Intentionally not supporting the following unary ops:
// DEFINE_UNARY_OP(addr, &);
// DEFINE_UNARY_OP(deref, *);
// Because it only makes sense if and only if both T& and T* are included in
// the type list, however, std::variant does not allow reference type to be
// an alternative. Also, if we overloaded the operator&, how can we get the
// address of the dynamic type itself?
// TODO: even if we can not have T& in the type list, should we just let * to
// return T instead of T&?

#undef DEFINE_UNARY_OP

// Printing
// TODO: we should inline the definition of can_print into enable_if, but I can
// only do this in C++20
constexpr auto can_print = [](auto x) constexpr {
  using T = decltype(x);
  if constexpr (opcheck<std::ostream&> << opcheck<T>) {
    return std::
        is_same_v<decltype(std::declval<std::ostream&>() << x), std::ostream&>;
  }
  return false;
};
template <
    typename DT,
    typename = std::enable_if_t<any_check(can_print, DT::types_as_tuple)>>
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
  TORCH_CHECK(printed, "Can not print: incompatible type");
  return os;
}

#define DEFINE_LEFT_PPMM(opname, op)                                           \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_helper = [](auto x) constexpr {                      \
    if constexpr (op opcheck<decltype(x)&>) {                                  \
      return std::is_same_v<decltype(op x), decltype(x)&>;                     \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename =                                                               \
          std::enable_if_t<any_check(opname##_helper, DT::types_as_tuple)>>    \
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
    TORCH_CHECK(computed, "Can not compute ", #op, " : incompatible type");    \
    return x;                                                                  \
  }

DEFINE_LEFT_PPMM(lpp, ++);
DEFINE_LEFT_PPMM(lmm, --);

#undef DEFINE_LEFT_PPMM

#define DEFINE_RIGHT_PPMM(opname, op)                                          \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_helper = [](auto x, auto y) constexpr {              \
    if constexpr (opcheck<decltype(x)&> op) {                                  \
      return std::is_same_v<decltype(x op), decltype(y)>;                      \
    }                                                                          \
    return false;                                                              \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename = std::enable_if_t<any_check(                                   \
          opname##_helper, DT::types_as_tuple, DT::types_as_tuple)>>           \
  inline constexpr DT operator op(DT& x, int) {                                \
    DT ret;                                                                    \
    DT::for_all_types([&ret, &x](auto _) {                                     \
      using Type = typename decltype(_)::type;                                 \
      if constexpr (opcheck<Type&> op) {                                       \
        if constexpr (DT::template is_candidate_type<                          \
                          decltype(std::declval<Type&>() op)>) {               \
          if (x.template is<Type>()) {                                         \
            ret = DT(x.template as<Type>() op);                                \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    });                                                                        \
    TORCH_CHECK(                                                               \
        !ret.template is<std::monostate>(),                                    \
        "Can not compute ",                                                    \
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

using EvaluatorValue = DynamicType<double, int64_t, bool>;

namespace EvaluatorValue_functions {

inline EvaluatorValue ceildiv(
    const EvaluatorValue& a,
    const EvaluatorValue& b) {
  if (a.is<int64_t>() && b.is<int64_t>()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return EvaluatorValue((aa + bb - 1) / bb);
    } else {
      return EvaluatorValue((aa + bb + 1) / bb);
    }
  }
  return EvaluatorValue(std::ceil((a / b).as<double>()));
}

inline EvaluatorValue max(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a > b).as<bool>() ? a : b);
}

inline EvaluatorValue min(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue((a < b).as<bool>() ? a : b);
}

inline EvaluatorValue gcd(const EvaluatorValue& a, const EvaluatorValue& b) {
  return EvaluatorValue(std::gcd(a.as<int64_t>(), b.as<int64_t>()));
}

inline EvaluatorValue notExpr(const EvaluatorValue& a) {
  if (a.is<int64_t>()) {
    return EvaluatorValue(~a.as<int64_t>());
  }
  if (a.is<bool>()) {
    return EvaluatorValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(false);
}

inline EvaluatorValue abs(const EvaluatorValue& a) {
  if (a.is<int64_t>()) {
    return EvaluatorValue(std::abs(a.as<int64_t>()));
  }
  if (a.is<double>()) {
    return EvaluatorValue(std::abs(a.as<double>()));
  }
  if (a.is<bool>()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace EvaluatorValue_functions

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

} // namespace nvfuser
