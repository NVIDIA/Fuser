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
// So we decide decide to not create the operator+ for Custom12.
//
// All the above behaviors are handled by template meta-programming, so they are
// automatic. Adding a new type to the list of types does not introduce any
// extra work. All the behaviors mentioned in this note is tested in
// DynamicTypeTest.ExamplesInNote, so if you want to change anything in this
// doc, please make sure to update the test as well.

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

// template <typename... Ts>
// std::ostream& operator<<(std::ostream& os, const DynamicType<Ts...>& dt);

template <typename... Ts>
// not using template <typename... Ts> to make sure there is at least one type
struct DynamicType {
  std::variant<std::monostate, Ts...> value_;

  using TypesAsTuple = std::tuple<Ts...>;
  static constexpr TypesAsTuple types_as_tuple{};
  using ForAllTypes = nvfuser::ForAllTypes<Ts...>;
  static constexpr ForAllTypes for_all_types{};

  constexpr DynamicType() = default;

  template <typename T>
  constexpr DynamicType(T value) : value_(value) {}

  template <typename T>
  constexpr bool is() const {
    return std::holds_alternative<T>(value_);
  }

  template <typename T>
  constexpr T as() const {
    return std::get<T>(value_);
  }

  template <typename T>
  constexpr T cast() const {
    std::optional<T> ret = std::nullopt;
    for_all_types([this, &ret](auto* from) {
      using From = std::remove_pointer_t<decltype(from)>;
      if constexpr (opcheck<From>.canCastTo(opcheck<T>)) {
        if (is<From>()) {
          ret = (T)as<From>();
        }
      }
    });
    TORCH_CHECK(ret.has_value(), "Cannot cast to ", typeid(T).name());
    return ret.value();
  }
};

#define DEFINE_BINARY_OP(opname, op)                                           \
  /*TODO: we should inline the definition of opname##_helper into enable_if,*/ \
  /*but I can only do this in C++20 */                                         \
  constexpr auto opname##_helper = [](auto x, auto y) constexpr {              \
    return opcheck<decltype(x)> op opcheck<decltype(y)>;                       \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename = std::enable_if_t<any_check(                                   \
          opname##_helper, DT::types_as_tuple, DT::types_as_tuple)>>           \
  inline constexpr DT operator op(DT x, DT y) {                                \
    DT ret(std::monostate{});                                                  \
    DT::for_all_types([&ret, x, y](auto* lhs) {                                \
      using LHS = std::remove_pointer_t<decltype(lhs)>;                        \
      DT::for_all_types([&ret, x, y](auto* rhs) {                              \
        using RTS = std::remove_pointer_t<decltype(rhs)>;                      \
        if constexpr (opcheck<LHS> op opcheck<RTS>) {                          \
          if (x.template is<LHS>() && y.template is<RTS>()) {                  \
            ret = DT(x.template as<LHS>() op y.template as<RTS>());            \
          }                                                                    \
        }                                                                      \
      });                                                                      \
    });                                                                        \
    TORCH_CHECK(                                                               \
        !ret.template is<std::monostate>(),                                    \
        "Can not compute ",                                                    \
        #op,                                                                   \
        " : incompatible type");                                               \
    return ret;                                                                \
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
  constexpr auto opname##_helper = [](auto x) constexpr {                      \
    return op opcheck<decltype(x)>;                                            \
  };                                                                           \
  template <                                                                   \
      typename DT,                                                             \
      typename =                                                               \
          std::enable_if_t<any_check(opname##_helper, DT::types_as_tuple)>>    \
  inline constexpr DT operator op(DT x) {                                      \
    DT ret(std::monostate{});                                                  \
    DT::for_all_types([&ret, x](auto* _) {                                     \
      using Type = std::remove_pointer_t<decltype(_)>;                         \
      if constexpr (op opcheck<Type>) {                                        \
        if (x.template is<Type>()) {                                           \
          ret = DT(op x.template as<Type>());                                  \
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

#undef DEFINE_UNARY_OP

// DEFINE_UNARY_OP(pp, ++);
// DEFINE_UNARY_OP(mm, --);
// DEFINE_UNARY_SUFFIX_OP(spp, ++);
// DEFINE_UNARY_SUFFIX_OP(smm, --);

class TORCH_CUDA_CU_API EvaluatorValue {
  std::variant<double, int64_t, bool> value_;

 public:
  explicit EvaluatorValue(int64_t i) : value_(i) {}
  explicit EvaluatorValue(double d) : value_(d) {}
  explicit EvaluatorValue(bool b) : value_(b) {}
  explicit EvaluatorValue(int i) : value_((int64_t)i) {}
  explicit EvaluatorValue(size_t i) : value_((int64_t)i) {}
  EvaluatorValue() : EvaluatorValue(0) {}

  bool isInt() const {
    return std::holds_alternative<int64_t>(value_);
  }

  bool isDouble() const {
    return std::holds_alternative<double>(value_);
  }

  bool isBool() const {
    return std::holds_alternative<bool>(value_);
  }

  template <typename T>
  T as() const {
    TORCH_CHECK(
        std::holds_alternative<T>(value_),
        "The expected dtype and the actual dtype does not match in EvaluatorValue");
    return std::get<T>(value_);
  }

  template <typename T>
  T cast() const {
    if (isInt()) {
      return (T)as<int64_t>();
    }
    if (isBool()) {
      return (T)as<bool>();
    }
    if (isDouble()) {
      return (T)as<double>();
    }
    TORCH_INTERNAL_ASSERT(false);
  }

#define DEFINE_ARITHMETIC_OP(op)                                  \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    if (isInt()) {                                                \
      return EvaluatorValue(as<int64_t>() op other);              \
    }                                                             \
    if (isDouble()) {                                             \
      return EvaluatorValue(as<double>() op other);               \
    }                                                             \
    if (isBool()) {                                               \
      return EvaluatorValue(as<bool>() op other);                 \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    if (other.isInt()) {                                          \
      return operator op(other.as<int64_t>());                    \
    }                                                             \
    if (other.isDouble()) {                                       \
      return operator op(other.as<double>());                     \
    }                                                             \
    if (other.isBool()) {                                         \
      return operator op(other.as<bool>());                       \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }

  DEFINE_ARITHMETIC_OP(+)
  DEFINE_ARITHMETIC_OP(-)
  DEFINE_ARITHMETIC_OP(*)
  DEFINE_ARITHMETIC_OP(/)
  DEFINE_ARITHMETIC_OP(>)
  DEFINE_ARITHMETIC_OP(>=)
  DEFINE_ARITHMETIC_OP(<)
  DEFINE_ARITHMETIC_OP(<=)
  DEFINE_ARITHMETIC_OP(==)
  DEFINE_ARITHMETIC_OP(!=)

#undef DEFINE_ARITHMETIC_OP

#define DEFINE_BITWISE_OP(op)                                     \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    if (isInt()) {                                                \
      return EvaluatorValue(as<int64_t>() op other);              \
    }                                                             \
    if (isBool()) {                                               \
      return EvaluatorValue(as<bool>() op other);                 \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    if (other.isInt()) {                                          \
      return operator op(other.as<int64_t>());                    \
    }                                                             \
    if (other.isBool()) {                                         \
      return operator op(other.as<bool>());                       \
    }                                                             \
    TORCH_INTERNAL_ASSERT(false);                                 \
  }

  DEFINE_BITWISE_OP(|)
  DEFINE_BITWISE_OP(^)
  DEFINE_BITWISE_OP(&)

#undef DEFINE_BITWISE_OP

#define DEFINE_LOGICAL_OP(op)                                     \
  template <typename T>                                           \
  EvaluatorValue operator op(T other) const {                     \
    return EvaluatorValue(cast<bool>() op other);                 \
  }                                                               \
  EvaluatorValue operator op(const EvaluatorValue& other) const { \
    return operator op(other.cast<bool>());                       \
  }

  DEFINE_LOGICAL_OP(||)
  DEFINE_LOGICAL_OP(&&)

#undef DEFINE_LOGICAL_OP

#define DEFINE_ASSIGN_OP(assign, op)                             \
  EvaluatorValue& operator assign(const EvaluatorValue& other) { \
    *this = *this op other;                                      \
    return *this;                                                \
  }                                                              \
  template <typename T>                                          \
  EvaluatorValue& operator assign(T other) {                     \
    *this = *this op other;                                      \
    return *this;                                                \
  }

  DEFINE_ASSIGN_OP(+=, +)
  DEFINE_ASSIGN_OP(-=, -)
  DEFINE_ASSIGN_OP(*=, *)
  DEFINE_ASSIGN_OP(/=, /)
  DEFINE_ASSIGN_OP(&=, &)
  DEFINE_ASSIGN_OP(|=, |)
  DEFINE_ASSIGN_OP(^=, ^)

#undef DEFINE_ASSIGN_OP

  EvaluatorValue operator%(const EvaluatorValue& other) const {
    if (isInt() && other.isInt()) {
      return EvaluatorValue(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue operator%(int64_t other) const {
    if (isInt()) {
      return EvaluatorValue(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue& operator%=(const EvaluatorValue& other) {
    if (isInt() && other.isInt()) {
      return *this = EvaluatorValue(as<int64_t>() % other.as<int64_t>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }
  EvaluatorValue& operator%=(int64_t other) {
    if (isInt()) {
      return *this = EvaluatorValue(as<int64_t>() % other);
    }
    TORCH_INTERNAL_ASSERT(false);
  }

  EvaluatorValue operator-() const {
    if (isInt()) {
      return EvaluatorValue(-as<int64_t>());
    }
    if (isDouble()) {
      return EvaluatorValue(-as<double>());
    }
    if (isBool()) {
      return EvaluatorValue(-as<bool>());
    }
    TORCH_INTERNAL_ASSERT(false);
  }

  explicit operator double() const;
  explicit operator int64_t() const;
  explicit operator size_t() const;
  explicit operator int() const;
  explicit operator bool() const;
}; // namespace cuda

#define DEFINE_ARITHMETIC_OP(op)                                 \
  template <typename T>                                          \
  inline EvaluatorValue operator op(T lhs, EvaluatorValue rhs) { \
    return EvaluatorValue(lhs) op rhs;                           \
  }

DEFINE_ARITHMETIC_OP(+)
DEFINE_ARITHMETIC_OP(-)
DEFINE_ARITHMETIC_OP(*)
DEFINE_ARITHMETIC_OP(/)
DEFINE_ARITHMETIC_OP(&&)
DEFINE_ARITHMETIC_OP(&)
DEFINE_ARITHMETIC_OP(||)
DEFINE_ARITHMETIC_OP(|)
DEFINE_ARITHMETIC_OP(^)
DEFINE_ARITHMETIC_OP(>)
DEFINE_ARITHMETIC_OP(>=)
DEFINE_ARITHMETIC_OP(<)
DEFINE_ARITHMETIC_OP(<=)
DEFINE_ARITHMETIC_OP(==)
DEFINE_ARITHMETIC_OP(!=)

#undef DEFINE_ARITHMETIC_OP

inline EvaluatorValue::operator double() const {
  return as<double>();
}

inline EvaluatorValue::operator int64_t() const {
  return as<int64_t>();
}

inline EvaluatorValue::operator size_t() const {
  return as<int64_t>();
}

inline EvaluatorValue::operator int() const {
  return (int)as<int64_t>();
}

inline EvaluatorValue::operator bool() const {
  return as<bool>();
}

#undef DEFINE_EQ_OP

inline std::ostream& operator<<(std::ostream& os, const EvaluatorValue& val) {
  if (val.isInt()) {
    return os << val.as<int64_t>();
  }
  if (val.isBool()) {
    return os << val.as<bool>();
  }
  if (val.isDouble()) {
    return os << val.as<double>();
  }
  TORCH_INTERNAL_ASSERT(false);
}

namespace EvaluatorValue_functions {

inline EvaluatorValue ceildiv(
    const EvaluatorValue& a,
    const EvaluatorValue& b) {
  if (a.isInt() && b.isInt()) {
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
  if (a.isInt()) {
    return EvaluatorValue(~a.as<int64_t>());
  }
  if (a.isBool()) {
    return EvaluatorValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(false);
}

inline EvaluatorValue abs(const EvaluatorValue& a) {
  if (a.isInt()) {
    return EvaluatorValue(std::abs(a.as<int64_t>()));
  }
  if (a.isDouble()) {
    return EvaluatorValue(std::abs(a.as<double>()));
  }
  if (a.isBool()) {
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
