
#ifdef __NVCC__
#include <complex>
#endif // __NVCC__
namespace {

using int8_t = signed char;
using uint8_t = unsigned char;
using int16_t = short int;
using uint16_t = unsigned short int;
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long int;
using uint64_t = unsigned long long int;

// Modified from cuda.h
struct TensorMap {
  alignas(64)
  uint64_t opaque[16];
};
typedef int nvfuser_index_t;

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef __NVCC__
#include <type_traits>
#else
// The following namespace std is modified from LLVM, see the following
// copyright information
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// copy-pasted from some llvm files:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/type_traits
// -
// https://github.com/llvm/llvm-project/blob/main/clang/test/Headers/Inputs/include/type_traits
namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp, _Tp __v>
struct integral_constant {
  static const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

// is_same, functional
template <class _Tp, class _Up>
struct is_same : public false_type {};
template <class _Tp>
struct is_same<_Tp, _Tp> : public true_type {};

// is_integral, for some types.
template <class _Tp>
struct is_integral : public integral_constant<bool, false> {};
template <>
struct is_integral<bool> : public integral_constant<bool, true> {};
template <>
struct is_integral<char> : public integral_constant<bool, true> {};
template <>
struct is_integral<short> : public integral_constant<bool, true> {};
template <>
struct is_integral<int> : public integral_constant<bool, true> {};
template <>
struct is_integral<long> : public integral_constant<bool, true> {};
template <>
struct is_integral<long long> : public integral_constant<bool, true> {};

// enable_if, functional
template <bool _C, typename _Tp>
struct enable_if {};
template <typename _Tp>
struct enable_if<true, _Tp> {
  using type = _Tp;
};
template <bool b, class T = void>
using enable_if_t = typename enable_if<b, T>::type;

template <class _Tp>
struct remove_const {
  typedef _Tp type;
};
template <class _Tp>
struct remove_const<const _Tp> {
  typedef _Tp type;
};
template <class _Tp>
using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp>
struct remove_volatile {
  typedef _Tp type;
};
template <class _Tp>
struct remove_volatile<volatile _Tp> {
  typedef _Tp type;
};
template <class _Tp>
using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp>
struct remove_cv {
  typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;
};
template <class _Tp>
using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp>
struct __libcpp_is_floating_point : public false_type {};
template <>
struct __libcpp_is_floating_point<float> : public true_type {};
template <>
struct __libcpp_is_floating_point<double> : public true_type {};
template <>
struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp>
struct is_floating_point
    : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};

template <class _Tp>
struct is_arithmetic
    : public integral_constant<
          bool,
          is_integral<_Tp>::value || is_floating_point<_Tp>::value> {};
template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type {
  static void __test(...);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);
  static long double __test(long double);

  typedef decltype(__test(declval<_Tp>())) type;
  static const bool value = !is_same<type, void>::value;
};

template <>
struct __numeric_type<void> {
  static const bool value = true;
};

// __promote

template <
    class _A1,
    class _A2 = void,
    class _A3 = void,
    bool = __numeric_type<_A1>::value&& __numeric_type<_A2>::value&&
        __numeric_type<_A3>::value>
class __promote_imp {
 public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
 private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

 public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true> {
 private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

 public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true> {
 public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std
#endif

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef __NVCC__
#include <bit>
#else

namespace std {

template <class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From), To> bit_cast(
    const From& src) noexcept {
  return *reinterpret_cast<const To*>(&src);
}

} // namespace std

#endif

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifndef __NVCC__
#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#define NAN __int_as_float(0x7fffffff)
//===----------------------------------------------------------------------===//
// The following namespace std is modified from LLVM, see the following
// copyright information
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// copy-pasted from the following llvm file:
// https://github.com/llvm/llvm-project/blob/main/libcxx/include/complex
namespace std {

template <class _Tp>
class complex;

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
template <class _Tp>
complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template <class _Tp>
class complex {
 public:
  typedef _Tp value_type;

 private:
  value_type __re_;
  value_type __im_;

 public:
  constexpr complex(
      const value_type& __re = value_type(),
      const value_type& __im = value_type())
      : __re_(__re), __im_(__im) {}
  template <class _Xp>
  constexpr complex(const complex<_Xp>& __c)
      : __re_(__c.real()), __im_(__c.imag()) {}

  constexpr value_type real() const {
    return __re_;
  }
  constexpr value_type imag() const {
    return __im_;
  }

  void real(value_type __re) {
    __re_ = __re;
  }
  void imag(value_type __im) {
    __im_ = __im;
  }

  constexpr operator bool() const {
    return real() || imag();
  }

  complex& operator=(const value_type& __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  complex& operator+=(const value_type& __re) {
    __re_ += __re;
    return *this;
  }
  complex& operator-=(const value_type& __re) {
    __re_ -= __re;
    return *this;
  }
  complex& operator*=(const value_type& __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  complex& operator/=(const value_type& __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  complex& operator=(const complex<_Xp>& __c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator+=(const complex<_Xp>& __c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator-=(const complex<_Xp>& __c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator*=(const complex<_Xp>& __c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  complex& operator/=(const complex<_Xp>& __c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <>
class complex<double>;

template <>
class complex<float> {
  float __re_;
  float __im_;

 public:
  typedef float value_type;

  constexpr complex(float __re = 0.0f, float __im = 0.0f)
      : __re_(__re), __im_(__im) {}

  explicit constexpr complex(const complex<double>& __c);

  // copy volatile to non-volatile
  constexpr complex(const volatile complex<float>& other)
      : __re_(other.__re_), __im_(other.__im_) {}

  constexpr complex(const complex<float>& other)
      : __re_(other.__re_), __im_(other.__im_) {}

  constexpr float real() const {
    return __re_;
  }
  constexpr float imag() const {
    return __im_;
  }

  void real(value_type __re) {
    __re_ = __re;
  }
  void imag(value_type __im) {
    __im_ = __im;
  }

  constexpr operator bool() const {
    return real() || imag();
  }

  complex& operator=(float __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  complex& operator+=(float __re) {
    __re_ += __re;
    return *this;
  }
  complex& operator-=(float __re) {
    __re_ -= __re;
    return *this;
  }
  complex& operator*=(float __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  complex& operator/=(float __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  complex& operator=(const complex<_Xp>& __c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }

  // non-volatile to volatile
  template <class _Xp>
  volatile complex& operator=(const complex<_Xp>& __c) volatile {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  // volatile to non-volatile
  template <class _Xp>
  complex& operator=(const volatile complex<_Xp>& __c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  // volatile to volatile
  template <class _Xp>
  volatile complex& operator=(const volatile complex<_Xp>& __c) volatile {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }

  template <class _Xp>
  complex& operator+=(const complex<_Xp>& __c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator-=(const complex<_Xp>& __c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator*=(const complex<_Xp>& __c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  complex& operator/=(const complex<_Xp>& __c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <>
class complex<double> {
  double __re_;
  double __im_;

 public:
  typedef double value_type;

  constexpr complex(double __re = 0.0, double __im = 0.0)
      : __re_(__re), __im_(__im) {}

  constexpr complex(const complex<float>& __c);

  // copy volatile to non-volatile
  constexpr complex(const volatile complex<double>& other)
      : __re_(other.__re_), __im_(other.__im_) {}

  constexpr complex(const complex<double>& other)
      : __re_(other.__re_), __im_(other.__im_) {}

  constexpr double real() const {
    return __re_;
  }
  constexpr double imag() const {
    return __im_;
  }

  void real(value_type __re) {
    __re_ = __re;
  }
  void imag(value_type __im) {
    __im_ = __im;
  }

  constexpr operator bool() const {
    return real() || imag();
  }

  complex& operator=(double __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  complex& operator+=(double __re) {
    __re_ += __re;
    return *this;
  }
  complex& operator-=(double __re) {
    __re_ -= __re;
    return *this;
  }
  complex& operator*=(double __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  complex& operator/=(double __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  complex& operator=(const complex<_Xp>& __c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }

  // non-volatile to volatile
  template <class _Xp>
  volatile complex& operator=(const complex<_Xp>& __c) volatile {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  // volatile to non-volatile
  template <class _Xp>
  complex& operator=(const volatile complex<_Xp>& __c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  // volatile to volatile
  template <class _Xp>
  volatile complex& operator=(const volatile complex<_Xp>& __c) volatile {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }

  template <class _Xp>
  complex& operator+=(const complex<_Xp>& __c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator-=(const complex<_Xp>& __c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  complex& operator*=(const complex<_Xp>& __c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  complex& operator/=(const complex<_Xp>& __c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

inline constexpr complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

// 26.3.6 operators:

template <class _Tp>
inline complex<_Tp> operator+(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator+(const complex<_Tp>& __x, const _Tp& __y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator+(const _Tp& __x, const complex<_Tp>& __y) {
  complex<_Tp> __t(__y);
  __t += __x;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator-(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator-(const complex<_Tp>& __x, const _Tp& __y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator-(const _Tp& __x, const complex<_Tp>& __y) {
  complex<_Tp> __t(-__y);
  __t += __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w) {
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __ac = __a * __c;
  _Tp __bd = __b * __d;
  _Tp __ad = __a * __d;
  _Tp __bc = __b * __c;
  _Tp __x = __ac - __bd;
  _Tp __y = __ad + __bc;
  if (isnan(__x) && isnan(__y)) {
    bool __recalc = false;
    if (isinf(__a) || isinf(__b)) {
      __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
      if (isnan(__c))
        __c = copysign(_Tp(0), __c);
      if (isnan(__d))
        __d = copysign(_Tp(0), __d);
      __recalc = true;
    }
    if (isinf(__c) || isinf(__d)) {
      __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
      if (isnan(__a))
        __a = copysign(_Tp(0), __a);
      if (isnan(__b))
        __b = copysign(_Tp(0), __b);
      __recalc = true;
    }
    if (!__recalc &&
        (isinf(__ac) || isinf(__bd) || isinf(__ad) || isinf(__bc))) {
      if (isnan(__a))
        __a = copysign(_Tp(0), __a);
      if (isnan(__b))
        __b = copysign(_Tp(0), __b);
      if (isnan(__c))
        __c = copysign(_Tp(0), __c);
      if (isnan(__d))
        __d = copysign(_Tp(0), __d);
      __recalc = true;
    }
    if (__recalc) {
      __x = _Tp(INFINITY) * (__a * __c - __b * __d);
      __y = _Tp(INFINITY) * (__a * __d + __b * __c);
    }
  }
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
inline complex<_Tp> operator*(const complex<_Tp>& __x, const _Tp& __y) {
  complex<_Tp> __t(__x);
  __t *= __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator*(const _Tp& __x, const complex<_Tp>& __y) {
  complex<_Tp> __t(__y);
  __t *= __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator/(const complex<_Tp>& __z, const complex<_Tp>& __w) {
  int __ilogbw = 0;
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
  if (isfinite(__logbw)) {
    __ilogbw = static_cast<int>(__logbw);
    __c = scalbn(__c, -__ilogbw);
    __d = scalbn(__d, -__ilogbw);
  }
  _Tp __denom = __c * __c + __d * __d;
  _Tp __x = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
  _Tp __y = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (isnan(__x) && isnan(__y)) {
    if ((__denom == _Tp(0)) && (!isnan(__a) || !isnan(__b))) {
      __x = copysign(_Tp(INFINITY), __c) * __a;
      __y = copysign(_Tp(INFINITY), __c) * __b;
    } else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d)) {
      __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
      __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
      __x = _Tp(INFINITY) * (__a * __c + __b * __d);
      __y = _Tp(INFINITY) * (__b * __c - __a * __d);
    } else if (
        isinf(__logbw) && __logbw > _Tp(0) && isfinite(__a) && isfinite(__b)) {
      __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
      __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
      __x = _Tp(0) * (__a * __c + __b * __d);
      __y = _Tp(0) * (__b * __c - __a * __d);
    }
  }
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
inline complex<_Tp> operator/(const complex<_Tp>& __x, const _Tp& __y) {
  return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template <class _Tp>
inline complex<_Tp> operator/(const _Tp& __x, const complex<_Tp>& __y) {
  complex<_Tp> __t(__x);
  __t /= __y;
  return __t;
}

template <class _Tp>
inline complex<_Tp> operator+(const complex<_Tp>& __x) {
  return __x;
}

template <class _Tp>
inline complex<_Tp> operator-(const complex<_Tp>& __x) {
  return complex<_Tp>(-__x.real(), -__x.imag());
}

template <class _Tp>
inline constexpr bool operator==(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template <class _Tp>
inline constexpr bool operator==(const complex<_Tp>& __x, const _Tp& __y) {
  return __x.real() == __y && __x.imag() == 0;
}

template <class _Tp>
inline constexpr bool operator==(const _Tp& __x, const complex<_Tp>& __y) {
  return __x == __y.real() && 0 == __y.imag();
}

template <class _Tp>
inline constexpr bool operator!=(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  return !(__x == __y);
}

template <class _Tp>
inline constexpr bool operator!=(const complex<_Tp>& __x, const _Tp& __y) {
  return !(__x == __y);
}

template <class _Tp>
inline constexpr bool operator!=(const _Tp& __x, const complex<_Tp>& __y) {
  return !(__x == __y);
}

template <class _Tp>
inline constexpr bool operator&&(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  return bool(__x) && bool(__y);
}

template <class _Tp>
inline constexpr bool isnan(const complex<_Tp>& __x) {
  return isnan(__x.real()) || isnan(__x.imag());
}

template <class _Tp>
inline constexpr bool operator||(
    const complex<_Tp>& __x,
    const complex<_Tp>& __y) {
  return bool(__x) || bool(__y);
}

// 26.3.7 values:

template <
    class _Tp,
    bool = is_integral<_Tp>::value,
    bool = is_floating_point<_Tp>::value>
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, true, false> {
  typedef double _ValueType;
  typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, false, true> {
  typedef _Tp _ValueType;
  typedef complex<_Tp> _ComplexType;
};

// real

template <class _Tp>
inline constexpr _Tp real(const complex<_Tp>& __c) {
  return __c.real();
}

template <class _Tp>
inline constexpr typename __libcpp_complex_overload_traits<_Tp>::_ValueType real(
    _Tp __re) {
  return __re;
}

// imag


template <class _Tp>
inline constexpr _Tp imag(const complex<_Tp>& __c) {
  return __c.imag();
}

template <class _Tp>
inline constexpr typename __libcpp_complex_overload_traits<_Tp>::_ValueType imag(
    _Tp) {
  return 0;
}

// abs

template <class _Tp>
inline _Tp abs(const complex<_Tp>& __c) {
  return hypot(__c.real(), __c.imag());
}

// arg

template <class _Tp>
inline _Tp arg(const complex<_Tp>& __c) {
  return atan2(__c.imag(), __c.real());
}

template <class _Tp>
inline typename enable_if<
    is_integral<_Tp>::value || is_same<_Tp, double>::value,
    double>::type
arg(_Tp __re) {
  return atan2(0., __re);
}

template <class _Tp>
inline typename enable_if<is_same<_Tp, float>::value, float>::type arg(
    _Tp __re) {
  return atan2f(0.F, __re);
}

} // namespace std

namespace std {

using ::isfinite;
using ::isinf;
using ::isnan;
using ::signbit;

using ::abs;

using ::acos;
using ::acosf;
using ::asin;
using ::asinf;
using ::atan;
using ::atan2;
using ::atan2f;
using ::atanf;
using ::ceil;
using ::ceilf;
using ::cos;
using ::cosf;
using ::cosh;
using ::coshf;

using ::exp;
using ::expf;

using ::fabs;
using ::fabsf;
using ::floor;
using ::floorf;

using ::fmod;
using ::fmodf;

using ::frexp;
using ::frexpf;
using ::ldexp;
using ::ldexpf;

using ::log;
using ::logf;

using ::log10;
using ::log10f;
using ::modf;
using ::modff;

using ::pow;
using ::powf;

using ::sin;
using ::sinf;
using ::sinh;
using ::sinhf;

using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;

using ::tanh;
using ::tanhf;

using ::acosh;
using ::acoshf;
using ::asinh;
using ::asinhf;
using ::atanh;
using ::atanhf;
using ::cbrt;
using ::cbrtf;

using ::copysign;
using ::copysignf;

using ::erf;
using ::erfc;
using ::erfcf;
using ::erff;
using ::exp2;
using ::exp2f;
using ::expm1;
using ::expm1f;
using ::fdim;
using ::fdimf;
using ::fma;
using ::fmaf;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::hypot;
using ::hypotf;
using ::ilogb;
using ::ilogbf;
using ::lgamma;
using ::lgammaf;
using ::llrint;
using ::llrintf;
using ::llround;
using ::llroundf;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logb;
using ::logbf;
using ::lrint;
using ::lrintf;
using ::lround;
using ::lroundf;

using ::nan;
using ::nanf;

using ::nearbyint;
using ::nearbyintf;
using ::nextafter;
using ::nextafterf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
using ::rint;
using ::rintf;
using ::round;
using ::roundf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
using ::tgamma;
using ::tgammaf;
using ::trunc;
using ::truncf;

} // namespace std

namespace std {

// norm

template <class _Tp>
inline _Tp norm(const complex<_Tp>& __c) {
  if (isinf(__c.real()))
    return abs(__c.real());
  if (isinf(__c.imag()))
    return abs(__c.imag());
  return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
inline typename __libcpp_complex_overload_traits<_Tp>::_ValueType norm(
    _Tp __re) {
  typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
  return static_cast<_ValueType>(__re) * __re;
}

// conj

template <class _Tp>
inline complex<_Tp> conj(const complex<_Tp>& __c) {
  return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
inline typename __libcpp_complex_overload_traits<_Tp>::_ComplexType conj(
    _Tp __re) {
  typedef
      typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// proj

template <class _Tp>
inline complex<_Tp> proj(const complex<_Tp>& __c) {
  complex<_Tp> __r = __c;
  if (isinf(__c.real()) || isinf(__c.imag()))
    __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
  return __r;
}

template <class _Tp>
inline typename enable_if<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType>::type
proj(_Tp __re) {
  if (isinf(__re))
    __re = abs(__re);
  return complex<_Tp>(__re);
}

template <class _Tp>
inline typename enable_if<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType>::type
proj(_Tp __re) {
  typedef
      typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
  return _ComplexType(__re);
}

// polar

template <class _Tp>
complex<_Tp> polar(const _Tp& __rho, const _Tp& __theta = _Tp()) {
  if (isnan(__rho) || signbit(__rho))
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  if (isnan(__theta)) {
    if (isinf(__rho))
      return complex<_Tp>(__rho, __theta);
    return complex<_Tp>(__theta, __theta);
  }
  if (isinf(__theta)) {
    if (isinf(__rho))
      return complex<_Tp>(__rho, _Tp(NAN));
    return complex<_Tp>(_Tp(NAN), _Tp(NAN));
  }
  _Tp __x = __rho * cos(__theta);
  if (isnan(__x))
    __x = 0;
  _Tp __y = __rho * sin(__theta);
  if (isnan(__y))
    __y = 0;
  return complex<_Tp>(__x, __y);
}

// log

template <class _Tp>
inline complex<_Tp> log(const complex<_Tp>& __x) {
  return complex<_Tp>(log(abs(__x)), arg(__x));
}

// log10

template <class _Tp>
inline complex<_Tp> log10(const complex<_Tp>& __x) {
  return log(__x) / log(_Tp(10));
}

// log2

template <class _Tp>
inline complex<_Tp> log2(const complex<_Tp>& __x) {
  return log(__x) / log(_Tp(2));
}

// sqrt

template <class _Tp>
complex<_Tp> sqrt(const complex<_Tp>& __x) {
  if (isinf(__x.imag()))
    return complex<_Tp>(_Tp(INFINITY), __x.imag());
  if (isinf(__x.real())) {
    if (__x.real() > _Tp(0))
      return complex<_Tp>(
          __x.real(),
          isnan(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
    return complex<_Tp>(
        isnan(__x.imag()) ? __x.imag() : _Tp(0),
        copysign(__x.real(), __x.imag()));
  }
  return polar(sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template <class _Tp>
complex<_Tp> exp(const complex<_Tp>& __x) {
  _Tp __i = __x.imag();
  if (__i == 0) {
    return complex<_Tp>(exp(__x.real()), copysign(_Tp(0), __x.imag()));
  }
  if (isinf(__x.real())) {
    if (__x.real() < _Tp(0)) {
      if (!isfinite(__i))
        __i = _Tp(1);
    } else if (__i == 0 || !isfinite(__i)) {
      if (isinf(__i))
        __i = _Tp(NAN);
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = exp(__x.real());
  return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

template <class _Tp>
inline complex<_Tp> pow(const complex<_Tp>& __x, const complex<_Tp>& __y) {
  return exp(__y * log(__x));
}

template <class _Tp, class _Up>
inline complex<typename __promote<_Tp, _Up>::type> pow(
    const complex<_Tp>& __x,
    const complex<_Up>& __y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return std::pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up>
inline typename enable_if<
    is_arithmetic<_Up>::value,
    complex<typename __promote<_Tp, _Up>::type>>::type
pow(const complex<_Tp>& __x, const _Up& __y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return std::pow(result_type(__x), result_type(__y));
}

template <class _Tp, class _Up>
inline typename enable_if<
    is_arithmetic<_Tp>::value,
    complex<typename __promote<_Tp, _Up>::type>>::type
pow(const _Tp& __x, const complex<_Up>& __y) {
  typedef complex<typename __promote<_Tp, _Up>::type> result_type;
  return std::pow(result_type(__x), result_type(__y));
}

// __sqr, computes pow(x, 2)

template <class _Tp>
inline complex<_Tp> __sqr(const complex<_Tp>& __x) {
  return complex<_Tp>(
      (__x.real() - __x.imag()) * (__x.real() + __x.imag()),
      _Tp(2) * __x.real() * __x.imag());
}

// asinh

template <class _Tp>
complex<_Tp> asinh(const complex<_Tp>& __x) {
  const _Tp __pi(atan2(+0., -0.));
  if (isinf(__x.real())) {
    if (isnan(__x.imag()))
      return __x;
    if (isinf(__x.imag()))
      return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
    return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
  }
  if (isnan(__x.real())) {
    if (isinf(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (__x.imag() == 0)
      return __x;
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (isinf(__x.imag()))
    return complex<_Tp>(
        copysign(__x.imag(), __x.real()), copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
  return complex<_Tp>(
      copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

template <class _Tp>
complex<_Tp> acosh(const complex<_Tp>& __x) {
  const _Tp __pi(atan2(+0., -0.));
  if (isinf(__x.real())) {
    if (isnan(__x.imag()))
      return complex<_Tp>(abs(__x.real()), __x.imag());
    if (isinf(__x.imag())) {
      if (__x.real() > 0)
        return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
      else
        return complex<_Tp>(
            -__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
    }
    if (__x.real() < 0)
      return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
    return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
  }
  if (isnan(__x.real())) {
    if (isinf(__x.imag()))
      return complex<_Tp>(abs(__x.imag()), __x.real());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (isinf(__x.imag()))
    return complex<_Tp>(abs(__x.imag()), copysign(__pi / _Tp(2), __x.imag()));
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
  return complex<_Tp>(
      copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
}

// atanh

template <class _Tp>
complex<_Tp> atanh(const complex<_Tp>& __x) {
  const _Tp __pi(atan2(+0., -0.));
  if (isinf(__x.imag())) {
    return complex<_Tp>(
        copysign(_Tp(0), __x.real()), copysign(__pi / _Tp(2), __x.imag()));
  }
  if (isnan(__x.imag())) {
    if (isinf(__x.real()) || __x.real() == 0)
      return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
    return complex<_Tp>(__x.imag(), __x.imag());
  }
  if (isnan(__x.real())) {
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (isinf(__x.real())) {
    return complex<_Tp>(
        copysign(_Tp(0), __x.real()), copysign(__pi / _Tp(2), __x.imag()));
  }
  if (abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0)) {
    return complex<_Tp>(
        copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
  }
  complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
  return complex<_Tp>(
      copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh

template <class _Tp>
complex<_Tp> sinh(const complex<_Tp>& __x) {
  if (isinf(__x.real()) && !isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.real() == 0 && !isfinite(__x.imag()))
    return complex<_Tp>(__x.real(), _Tp(NAN));
  if (__x.imag() == 0 && !isfinite(__x.real()))
    return __x;
  return complex<_Tp>(
      sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh

template <class _Tp>
complex<_Tp> cosh(const complex<_Tp>& __x) {
  if (isinf(__x.real()) && !isfinite(__x.imag()))
    return complex<_Tp>(abs(__x.real()), _Tp(NAN));
  if (__x.real() == 0 && !isfinite(__x.imag()))
    return complex<_Tp>(_Tp(NAN), __x.real());
  if (__x.real() == 0 && __x.imag() == 0)
    return complex<_Tp>(_Tp(1), __x.imag());
  if (__x.imag() == 0 && !isfinite(__x.real()))
    return complex<_Tp>(abs(__x.real()), __x.imag());
  return complex<_Tp>(
      cosh(__x.real()) * cos(__x.imag()), sinh(__x.real()) * sin(__x.imag()));
}

// tanh

template <class _Tp>
complex<_Tp> tanh(const complex<_Tp>& __x) {
  if (isinf(__x.real())) {
    if (!isfinite(__x.imag()))
      return complex<_Tp>(copysign(_Tp(1), __x.real()), _Tp(0));
    return complex<_Tp>(
        copysign(_Tp(1), __x.real()),
        copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
  }
  if (isnan(__x.real()) && __x.imag() == 0)
    return __x;
  _Tp __2r(_Tp(2) * __x.real());
  _Tp __2i(_Tp(2) * __x.imag());
  _Tp __d(cosh(__2r) + cos(__2i));
  _Tp __2rsh(sinh(__2r));
  if (isinf(__2rsh) && isinf(__d))
    return complex<_Tp>(
        __2rsh > _Tp(0) ? _Tp(1) : _Tp(-1), __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
  return complex<_Tp>(__2rsh / __d, sin(__2i) / __d);
}

// asin

template <class _Tp>
complex<_Tp> asin(const complex<_Tp>& __x) {
  complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template <class _Tp>
complex<_Tp> acos(const complex<_Tp>& __x) {
  const _Tp __pi(atan2(+0., -0.));
  if (isinf(__x.real())) {
    if (isnan(__x.imag()))
      return complex<_Tp>(__x.imag(), __x.real());
    if (isinf(__x.imag())) {
      if (__x.real() < _Tp(0))
        return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
      return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
    }
    if (__x.real() < _Tp(0))
      return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
    return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
  }
  if (isnan(__x.real())) {
    if (isinf(__x.imag()))
      return complex<_Tp>(__x.real(), -__x.imag());
    return complex<_Tp>(__x.real(), __x.real());
  }
  if (isinf(__x.imag()))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
    return complex<_Tp>(__pi / _Tp(2), -__x.imag());
  complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
  if (signbit(__x.imag()))
    return complex<_Tp>(abs(__z.imag()), abs(__z.real()));
  return complex<_Tp>(abs(__z.imag()), -abs(__z.real()));
}

// atan

template <class _Tp>
complex<_Tp> atan(const complex<_Tp>& __x) {
  complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template <class _Tp>
complex<_Tp> sin(const complex<_Tp>& __x) {
  complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template <class _Tp>
inline complex<_Tp> cos(const complex<_Tp>& __x) {
  return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template <class _Tp>
complex<_Tp> tan(const complex<_Tp>& __x) {
  complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
  return complex<_Tp>(__z.imag(), -__z.real());
}

// Literal suffix for complex number literals [complex.literals]
inline namespace literals {
inline namespace complex_literals {
constexpr complex<double> operator""i(long double __im) {
  return {0.0, static_cast<double>(__im)};
}

constexpr complex<double> operator""i(unsigned long long __im) {
  return {0.0, static_cast<double>(__im)};
}

constexpr complex<float> operator""if(long double __im) {
  return {0.0f, static_cast<float>(__im)};
}

constexpr complex<float> operator""if(unsigned long long __im) {
  return {0.0f, static_cast<float>(__im)};
}
} // namespace complex_literals
} // namespace literals

} // namespace std

__device__ std::complex<double> lerp(
    std::complex<double> start,
    std::complex<double> end,
    std::complex<double> weight) {
  if (abs(weight) < 0.5) {
    return start + weight * (end - start);
  } else {
    return end - (end - start) * (1.0 - weight);
  }
}

__device__ std::complex<float> lerp(
    std::complex<float> start,
    std::complex<float> end,
    std::complex<float> weight) {
  if (abs(weight) < 0.5f) {
    return start + weight * (end - start);
  } else {

    return end - (end - start) * (1.0f - weight);
  }
}

__device__ std::complex<double> reciprocal(std::complex<double> x) {
  return 1.0 / x;
}

__device__ std::complex<float> reciprocal(std::complex<float> x) {
  return 1.0f / x;
}

__device__ std::complex<double> sigmoid(std::complex<double> x) {
  return 1.0 / (1.0 + exp(-x));
}

__device__ std::complex<float> sigmoid(std::complex<float> x) {
  return 1.0f / (1.0f + exp(-x));
}

// The reciprocal of a complex number z is
//    1/z = conj(z)/|z|^2.
// The principal square root of a complex number z can be obtained by [1]
//    sqrt(z) = sqrt(|z|) (z + |z|) / |z + |z||.
// Combining these formulas we have
//    1/sqrt(z) = (conj(z) + |z|) / (sqrt(|z|) |z + |z||).
// [1] https://math.stackexchange.com/a/44500
__device__ std::complex<float> rsqrt(std::complex<float> z) {
  auto a = std::real(z);
  auto b = std::imag(z);
  auto absa = ::fabsf(a);
  auto absb = ::fabsf(b);
  // scale to avoid precision loss due to underflow/overflow
  auto scale = fmax(absa, absb);
  a /= scale;
  b /= scale;
  auto a_sq = a * a;
  auto b_sq = b * b;
  auto modz_sq = a_sq + b_sq;
  auto modz = ::sqrtf(modz_sq);
  auto a_plus_modz = a + modz;
  auto mod_zplusmodz_sq = a_plus_modz * a_plus_modz + b_sq;
  auto fac = ::rsqrtf(scale * modz * mod_zplusmodz_sq);
  return std::complex<float>(a_plus_modz * fac, -b * fac);
}

__device__ std::complex<double> rsqrt(std::complex<double> z) {
  auto a = std::real(z);
  auto b = std::imag(z);
  auto absa = ::abs(a);
  auto absb = ::abs(b);
  // scale to avoid precision loss due to underflow/overflow
  auto scale = fmax(absa, absb);
  a /= scale;
  b /= scale;
  auto a_sq = a * a;
  auto b_sq = b * b;
  auto modz_sq = a_sq + b_sq;
  auto modz = ::sqrt(modz_sq);
  auto a_plus_modz = a + modz;
  auto mod_zplusmodz_sq = a_plus_modz * a_plus_modz + b_sq;
  auto fac = ::rsqrt(scale * modz * mod_zplusmodz_sq);
  return std::complex<double>(a_plus_modz * fac, -b * fac);
}

template <typename T>
bool isfinite(std::complex<T> x) {
  return ::isfinite(std::real(x)) && ::isfinite(std::imag(x));
}

template <typename T>
bool isinf(std::complex<T> x) {
  return ::isinf(std::real(x)) || ::isinf(std::imag(x));
}

template <typename T>
bool isreal(std::complex<T> x) {
  return std::imag(x) == 0;
}
#endif // __NVCC__

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#define __NVFUSER_HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __NVFUSER_HALF_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

struct __half;
__device__ __inline__ __half __float2half(const float);

struct __align__(2) __half {
  __half() = default;

  __device__ __half(const float f) {
    __x = __float2half(f).__x;
  }

  __device__ uint16_t raw() const {
    return __x;
  }

 protected:
  unsigned short __x;
};

__device__ __inline__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "f"(f));
  return val;
}

__device__ __inline__ __half __double2half(const double d) {
  __half val;
  asm("{  cvt.rn.f16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "d"(d));
  return val;
}

__device__ __inline__ __half __int2half(const int i) {
  __half val;
  asm("{  cvt.rn.f16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __half __int2half(const int64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __half __int2half(const uint32_t i) {
  __half val;
  asm("{  cvt.rn.f16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "r"(i));
  return val;
}

__device__ __inline__ __half __int2half(const uint64_t i64) {
  __half val;
  asm("{  cvt.rn.f16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "l"(i64));
  return val;
}

__device__ __inline__ __half __bool2half(const bool b) {
  return __int2half((int)b);
}

__device__ __inline__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ double __half2double(const __half h) {
  double val;
  asm("{  cvt.f64.f16 %0, %1;}\n" : "=d"(val) : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __half2int32(const __half h) {
  int val;
  asm("{  cvt.rzi.s32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __half2int(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.s64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ int __half2uint32(const __half h) {
  int val;
  asm("{  cvt.rzi.u32.f16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ int64_t __half2uint(const __half h) {
  int64_t val;
  asm("{  cvt.rzi.u64.f16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
}

__device__ __inline__ void __half2int(const __half h, int& output) {
  output = __half2int32(h);
}

__device__ __inline__ void __half2int(const __half h, int64_t& output) {
  output = __half2int(h);
}

__device__ __inline__ void __half2int(const __half h, uint32_t& output) {
  output = __half2uint32(h);
}

__device__ __inline__ void __half2int(const __half h, uint64_t& output) {
  output = __half2uint(h);
}

__device__ __inline__ nvfuser_index_t __half2index(const __half h) {
  nvfuser_index_t result;
  __half2int(h, result);
  return result;
}

__device__ __inline__ bool __half2bool(const __half h) {
  return (bool)__half2float(h) != 0;
}

__device__ __inline__ __half __real_then_2half(const std::complex<float> c) {
  return __float2half(std::real(c));
}

__device__ __inline__ __half __real_then_2half(const std::complex<double> c) {
  return __double2half(std::real(c));
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#define __NVFUSER_BFLOAT_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __NVFUSER_BFLOAT_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

struct __bfloat;
__device__ __inline__ __bfloat __float2bfloat(const float);

struct __align__(2) __bfloat {
  __bfloat() = default;

  __device__ __bfloat(const float f) {
    __x = __float2bfloat(f).__x;
  }

  __device__ uint16_t raw() const {
    return __x;
  }

 protected:
  unsigned short __x;
};

__device__ __inline__ __bfloat __float2bfloat(const float f) {
  __bfloat val;
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "f"(f));
  return val;
}

__device__ __inline__ __bfloat __double2bfloat(const double d) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.f64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "d"(d));
  return val;
#else
  return __float2bfloat(static_cast<float>(d));
#endif
}

__device__ __inline__ __bfloat __int2bfloat(const int i) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.s32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "r"(i));
  return val;
#else
  return __float2bfloat(static_cast<float>(i));
#endif
}

__device__ __inline__ __bfloat __int2bfloat(const int64_t i64) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.s64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "l"(i64));
  return val;
#else
  return __float2bfloat(static_cast<float>(i64));
#endif
}

__device__ __inline__ __bfloat __int2bfloat(const uint32_t i) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.u32 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "r"(i));
  return val;
#else
  return __float2bfloat(static_cast<float>(i));
#endif
}

__device__ __inline__ __bfloat __int2bfloat(const uint64_t i64) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.u64 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "l"(i64));
  return val;
#else
  return __float2bfloat(static_cast<float>(i64));
#endif
}

__device__ __inline__ __bfloat __bool2bfloat(const bool b) {
  return __int2bfloat((int)b);
}

__device__ __inline__ float __bfloat2float(const __bfloat h) {
  float val;
  asm("{  mov.b32 %0, {0,%1};}\n"
      : "=f"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
}

__device__ __inline__ double __bfloat2double(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  double val;
  asm("{  cvt.f64.bf16 %0, %1;}\n"
      : "=d"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<double>(__bfloat2float(h));
#endif
}

__device__ int __bfloat2int32(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int val;
  asm("{  cvt.rzi.s32.bf16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int>(__bfloat2float(h));
#endif
}

__device__ __inline__ int64_t __bfloat2int(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int64_t val;
  asm("{  cvt.rzi.s64.bf16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int64_t>(__bfloat2float(h));
#endif
}

__device__ int __bfloat2uint32(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int val;
  asm("{  cvt.rzi.u32.bf16 %0, %1;}\n"
      : "=r"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int>(__bfloat2float(h));
#endif
}

__device__ __inline__ int64_t __bfloat2uint(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  int64_t val;
  asm("{  cvt.rzi.u64.bf16 %0, %1;}\n"
      : "=l"(val)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return static_cast<int64_t>(__bfloat2float(h));
#endif
}

__device__ __inline__ void __bfloat2int(const __bfloat h, int& output) {
  output = __bfloat2int32(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, int64_t& output) {
  output = __bfloat2int(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, uint32_t& output) {
  output = __bfloat2uint32(h);
}

__device__ __inline__ void __bfloat2int(const __bfloat h, uint64_t& output) {
  output = __bfloat2uint(h);
}

__device__ __inline__ nvfuser_index_t __bfloat2index(
    const __bfloat h,
    bool& output) {
  nvfuser_index_t result;
  __bfloat2int(h, result);
  return result;
}

__device__ __inline__ bool __bfloat2bool(const __bfloat h) {
  return (bool)__bfloat2float(h) != 0;
}

__device__ __inline__ __bfloat __half2bfloat(const __half h) {
#if __CUDA_ARCH__ >= 900
  __bfloat val;
  asm("{  cvt.rn.bf16.f16 %0, %1;}\n"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(__NVFUSER_HALF_TO_CUS(h)));
  return val;
#else
  return __float2bfloat(__half2float(h));
#endif
}

__device__ __inline__ __half __bfloat2half(const __bfloat h) {
#if __CUDA_ARCH__ >= 900
  __half val;
  asm("{  cvt.rn.f16.bf16 %0, %1;}\n"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  return val;
#else
  return __float2half(__bfloat2float(h));
#endif
}

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<float> c) {
  return __float2bfloat(std::real(c));
}

__device__ __inline__ __bfloat __real_then_2bfloat(
    const std::complex<double> c) {
  return __double2bfloat(std::real(c));
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

struct __e4m3;
__device__ __inline__ __e4m3 __float2e4m3(const float);

struct __align__(1) __e4m3 {
  __e4m3() = default;

  __device__ __e4m3(const float f) {
    __x = __float2e4m3(f).__x;
  }

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

__device__ __inline__ __e4m3 __double2e4m3(const double f) {
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{\n\t"
      ".reg .b16 buf0;\n\t"
      ".reg .b32 buf1;\n\t"
      "cvt.rn.f16.f64 buf0, %1;\n\t"
      "cvt.u32.u16 buf1, buf0;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, buf1;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "d"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ double __e4m32double(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  double val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f64.f16 %0, %1;"
      "}"
      : "=d"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __float2e4m3(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e4m32float(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  float val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f32.f16 %0, %1;\n\t"
      "}"
      : "=f"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __half2e4m3(const __half h) {
  uint32_t buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "r"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e4m32half(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __half val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "}"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e4m3 __bfloat2e4m3(const __bfloat h) {
  unsigned short _tmp_buffer;
  __e4m3 val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16.bf16 %1, %1;\n\t"
      "cvt.u32.u16 buf0, %1;\n\t"
      "cvt.rn.satfinite.e4m3x2.f16x2 %0, buf0;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ __bfloat __e4m32bfloat(const __e4m3 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __bfloat val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e4m3x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "cvt.bf16.f16 %0, %0;\n\t"
      "}"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

struct __e5m2;
__device__ __inline__ __e5m2 __float2e5m2(const float);

struct __align__(1) __e5m2 {
  __e5m2() = default;

  __device__ __e5m2(const float f) {
    __x = __float2e5m2(f).__x;
  }

  __device__ uint8_t raw() const {
    return __x;
  }

 protected:
  uint8_t __x;
};

__device__ __inline__ __e5m2 __double2e5m2(const double f) {
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{\n\t"
      ".reg .b16 buf0;\n\t"
      ".reg .b32 buf1;\n\t"
      "cvt.rn.f16.f64 buf0, %1;\n\t"
      "cvt.u32.u16 buf1, buf0;\n\t"
      "cvt.rn.satfinite.e5m2x2.f16x2 %0, buf1;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "d"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ double __e5m22double(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  double val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f64.f16 %0, %1;"
      "}"
      : "=d"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __float2e5m2(const float f) {
  constexpr float f_const_zero = 0.f;
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;}"
      : "=h"(_tmp_buffer)
      : "f"(f_const_zero), "f"(f));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ float __e5m22float(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  float val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %1, buf0;\n\t"
      "cvt.f32.f16 %0, %1;\n\t"
      "}"
      : "=f"(val)
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __half2e5m2(const __half h) {
  uint32_t buffer;
  memcpy(&buffer, &h, sizeof(__half));
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}\n\t"
      : "=h"(_tmp_buffer)
      : "r"(buffer));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));

  return val;
}

__device__ __inline__ __half __e5m22half(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __half val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "}"
      : "=h"(__NVFUSER_HALF_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

__device__ __inline__ __e5m2 __bfloat2e5m2(const __bfloat h) {
  unsigned short _tmp_buffer;
  __e5m2 val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16.bf16 %1, %1;\n\t"
      "cvt.u32.u16 buf0, %1;\n\t"
      "cvt.rn.satfinite.e5m2x2.f16x2 %0, buf0;\n\t"
      "}"
      : "=h"(_tmp_buffer)
      : "h"(__NVFUSER_BFLOAT_TO_CUS(h)));
  memcpy(&val, &_tmp_buffer, sizeof(uint8_t));
  return val;
}

__device__ __inline__ __bfloat __e5m22bfloat(const __e5m2 h) {
  unsigned short _tmp_buffer;
  memcpy(&_tmp_buffer, &h, sizeof(uint8_t));
  __bfloat val;
  asm("{\n\t"
      ".reg .b32 buf0;\n\t"
      "cvt.rn.f16x2.e5m2x2 buf0, %1;\n\t"
      "cvt.u16.u32 %0, buf0;\n\t"
      "cvt.bf16.f16 %0, %0;\n\t"
      "}"
      : "=h"(__NVFUSER_BFLOAT_TO_US(val))
      : "h"(_tmp_buffer));

  return val;
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Type trait utils
template <typename Type, bool is_volatile>
struct MaybeVolatile;

template <typename Type>
struct MaybeVolatile<Type, true> {
  using type = volatile Type;
};

template <typename Type>
struct MaybeVolatile<Type, false> {
  using type = Type;
};

template <typename... Types>
struct TypeList {};

template <int idx, typename T, typename... Types>
struct TypeSelector {
  using type = typename TypeSelector<idx - 1, Types...>::type;
};

template <typename T, typename... Types>
struct TypeSelector<0, T, Types...> {
  using type = T;
};

template <typename T0, typename T1>
struct IsSameType {
  static constexpr bool value = false;
};

template <typename T0>
struct IsSameType<T0, T0> {
  static constexpr bool value = true;
};

template <typename T>
struct IsPointerType {
  static constexpr bool value = false;
};

template <typename T>
struct IsPointerType<T*> {
  static constexpr bool value = true;
};

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size = 1>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }

  __device__ const scalar_t& operator[](const unsigned int i) const {
    return array[i];
  }

  Array& operator=(const Array& a) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = a[i];
    }
    return *this;
  }
};

// Used for vectorized allocations that are not in registers
template <typename scalar_t, int vec_size>
__device__ void arraySet(scalar_t* buff, scalar_t val) {
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    buff[i] = val;
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadGeneric(scalar_t* to, scalar_t* from) {
  // It would be really nice to use memcpy here, but one example was failing
  // with:
  //
  //  memcpy(to, from, vec_size * sizeof(scalar_t));
  //
  // Yet passing with:
  //
  // for(int i = 0; i < vec_size; i++){
  //   to[i] = from[i];
  // }

  switch (sizeof(scalar_t) * vec_size) {
    case 1:
      *reinterpret_cast<uchar1*>(to) = *reinterpret_cast<uchar1*>(from);
      break;
    case 2:
      *reinterpret_cast<uchar2*>(to) = *reinterpret_cast<uchar2*>(from);
      break;
    case 4:
      *reinterpret_cast<uint1*>(to) = *reinterpret_cast<uint1*>(from);
      break;
    case 8:
      *reinterpret_cast<uint2*>(to) = *reinterpret_cast<uint2*>(from);
      break;
    case 12:
      *reinterpret_cast<uint3*>(to) = *reinterpret_cast<uint3*>(from);
      break;
    case 16:
      *reinterpret_cast<uint4*>(to) = *reinterpret_cast<uint4*>(from);
      break;
  }
}

// Volatile version only works with c++ fundamnetal types
template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGenericVolatile(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 2:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 4:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 8:
      *reinterpret_cast<typename MaybeVolatile<double, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<double, is_volatile_from>::type*>(from);
      break;
  }
}

template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadLocalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, is_volatile, false>(to, from);
      break;
    case 8: {
      uint2 const& data = *reinterpret_cast<uint2*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      } else {
        asm volatile(
            "st.global.cs.v2.s32 [%0], {%1,%2};" ::"l"(
                (typename MaybeVolatile<uint2, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y));
      }
      break;
    }
    case 16: {
      uint4 const& data = *reinterpret_cast<uint4*>(from);
      if (is_volatile) {
        asm volatile(
            "st.volatile.global.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      } else {
        asm volatile(
            "st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"(
                (typename MaybeVolatile<uint4, is_volatile>::type*)to),
            "r"(data.x),
            "r"(data.y),
            "r"(data.z),
            "r"(data.w));
      }
      break;
    }
  }
}

// This is copied from csrc/type.h and should be kept consistent.
enum class CacheOp {
  AllLevels,
  Streaming,
  Global,
};

template <typename T, CacheOp cache_op>
__device__ void loadGlobalToLocalCached(void* to, void* from) {
  T* typed_to = reinterpret_cast<T*>(to);
  T* typed_from = reinterpret_cast<T*>(from);
  switch (cache_op) {
    case CacheOp::AllLevels:
      *typed_to = __ldca(typed_from);
      break;
    case CacheOp::Streaming:
      *typed_to = __ldcs(typed_from);
      break;
    case CacheOp::Global:
      *typed_to = __ldcg(typed_from);
      break;
  }
}

// For simplicity, cache_op is only used for non-volatile loads written in
// inline assembly. Other loads are done with the default cache operator --
// cache all levels. ld.volatile doesn't accept cache operator anyway.
template <typename scalar_t, int vec_size, bool is_volatile, CacheOp cache_op>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, is_volatile>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2& data = *reinterpret_cast<uint2*>(to);
        asm volatile("ld.volatile.global.v2.s32 {%0,%1}, [%2];"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"((uint2*)from));
      } else {
        loadGlobalToLocalCached<uint2, cache_op>(
            to, const_cast<scalar_t*>(from));
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4& data = *reinterpret_cast<uint4*>(to);
        asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"((uint4*)from));
      } else {
        loadGlobalToLocalCached<uint4, cache_op>(
            to, const_cast<scalar_t*>(from));
      }
      break;
    }
  }
}

template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGlobalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
    case 2:
    case 4:
    case 8:
      loadGenericVolatile<scalar_t, vec_size, is_volatile_to, is_volatile_from>(
          to, from);
      break;
    case 12: {
      uint3 local_intermediate;
      loadGlobalToLocal<
          scalar_t,
          vec_size,
          is_volatile_from,
          CacheOp::Streaming>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
    case 16: {
      uint4 local_intermediate;
      loadGlobalToLocal<
          scalar_t,
          vec_size,
          is_volatile_from,
          CacheOp::Streaming>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
  }
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
template <typename T, int Dims, int AllocDims = Dims>
struct Tensor {
  __device__ T& operator[](nvfuser_index_t ind) {
    return data[ind];
  };

  T* data;
  Array<nvfuser_index_t, Dims, 1> logical_size;
  Array<nvfuser_index_t, AllocDims, 1> alloc_stride;
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](nvfuser_index_t i) {
    return *data;
  };

  T* data;
};

// Specialization for 0-dim case that's easy to pass in a CPU based tensor.
template <typename T>
struct CpuScalarTensor {
  __device__ T& operator[](int i) {
    return data;
  };

  T data;
};

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
__device__ unsigned int mulhilo32(
    unsigned int a,
    unsigned int b,
    unsigned int* result_high) {
  *result_high = __umulhi(a, b);
  return a * b;
}

__device__ uint4 single_round(uint4 ctr, uint2 key) {
  constexpr unsigned long kPhiloxSA = 0xD2511F53;
  constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
  unsigned int hi0;
  unsigned int hi1;
  unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
  unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
  uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
  return ret;
}

__device__ uint4 philox(
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset) {
  constexpr unsigned long kPhilox10A = 0x9E3779B9;
  constexpr unsigned long kPhilox10B = 0xBB67AE85;
  uint2 key = {};
  key.x = (unsigned int)seed;
  key.y = (unsigned int)(seed >> 32);
  uint4 counter = make_uint4(0, 0, 0, 0);
  counter.x = (unsigned int)(offset);
  counter.y = (unsigned int)(offset >> 32);
  counter.z = (unsigned int)(subsequence);
  counter.w = (unsigned int)(subsequence >> 32);

  uint4 output = {};
  uint2 key_ = key;
  uint4 counter_ = counter;
  for (int i = 0; i < 9; i++) {
    counter_ = single_round(counter_, key_);
    key_.x += (kPhilox10A);
    key_.y += (kPhilox10B);
  }
  output = single_round(counter_, key_);
  return output;
}

__device__ float uniformf(unsigned int x) {
  constexpr float kRanInvM32 = 2.3283064e-10f; // Inverse of 2^32.
  float result = x * kRanInvM32 + kRanInvM32 / 2.0f;
  return result == 1 ? 0.0f : result;
}

__device__ double uniform(unsigned int x, unsigned int y) {
  constexpr double kRan2Pow53Inv = 1.1102230246251565e-16;
  const unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  double result = z * kRan2Pow53Inv + (kRan2Pow53Inv / 2.0);
  return result == 1 ? 0.0 : result;
}

__device__ double rng_uniform(const uint4& rng_result, int rng_component) {
  return uniform(
      (&rng_result.x)[rng_component * 2],
      (&rng_result.x)[rng_component * 2 + 1]);
}

__device__ float rng_uniformf(const uint4& rng_result, int rng_component) {
  return uniformf((&rng_result.x)[rng_component]);
}

__device__ double rng_uniform_range(
    const uint4& rng_result,
    int rng_component,
    double from,
    double to) {
  auto range = to - from;
  auto uniform01 = rng_uniform(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ float rng_uniform_rangef(
    const uint4& rng_result,
    int rng_component,
    float from,
    float to) {
  auto range = to - from;
  auto uniform01 = rng_uniformf(rng_result, rng_component);
  return from + range * uniform01;
}

__device__ float normalf(unsigned int x, unsigned int y, int rng_component) {
  float u = uniformf(x);
  float v = uniformf(y) * 6.2831855f;

  if (rng_component % 2 == 0) {
    return sqrtf(-2.0f * logf(u)) * sinf(v);
  } else {
    return sqrtf(-2.0f * logf(u)) * cosf(v);
  }
}

__device__ double normal(
    unsigned int x0,
    unsigned int x1,
    unsigned int y0,
    unsigned int y1,
    int rng_component) {
  double u = uniform(x0, x1);
  double v = uniform(y0, y1) * 6.2831853071795860;

  if (rng_component % 2 == 0) {
    return sqrt(-2.0 * log(u)) * sin(v);
  } else {
    return sqrt(-2.0 * log(u)) * cos(v);
  }
}

__device__ double rng_normal_standard(
    const uint4& rng_result,
    int rng_component) {
  return normal(
      rng_result.x, rng_result.y, rng_result.z, rng_result.w, rng_component);
}

__device__ float rng_normal_standardf(
    const uint4& rng_result,
    int rng_component) {
  return normalf(
      (&rng_result.x)[rng_component / 2 * 2],
      (&rng_result.y)[rng_component / 2 * 2],
      rng_component);
}

__device__ double rng_normal_general(
    const uint4& rng_result,
    int rng_component,
    double mean,
    double std) {
  auto normal01 = rng_normal_standard(rng_result, rng_component);
  return normal01 * std + mean;
}

__device__ float rng_normal_generalf(
    const uint4& rng_result,
    int rng_component,
    float mean,
    float std) {
  auto normal01 = rng_normal_standardf(rng_result, rng_component);
  return normal01 * std + mean;
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#define NVFUSER_DEFINE_MAGIC_ZERO          \
  __shared__ int nvfuser_zero_s;           \
  if (threadIdx.x == 0)                    \
    nvfuser_zero_s = 0;                    \
  __syncthreads();                         \
  atomicMin(&nvfuser_zero_s, threadIdx.x); \
  int nvfuser_zero = nvfuser_zero_s;

#define NVFUSER_UPDATE_MAGIC_ZERO \
  do {                            \
    nvfuser_zero <<= 1;           \
  } while (0);

#ifdef __NVCC__
#include <assert.h>
#endif // __NVCC__

__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ constexpr int64_t ceilDiv(int64_t a, int b) {
  return ceilDiv(a, (int64_t)b);
}

__device__ constexpr int64_t ceilDiv(int a, int64_t b) {
  return ceilDiv((int64_t)a, b);
}

__device__ constexpr double ceilDiv(double a, double b) {
  return std::ceil(a / b);
}

__device__ constexpr double ceilDiv(double a, int64_t b) {
  return std::ceil(a / b);
}

__device__ constexpr double ceilDiv(int64_t a, double b) {
  return std::ceil(a / b);
}

// Monotonic and precise lerp is described here:
// https://math.stackexchange.com/a/1798323
__device__ double lerp(double start, double end, double weight) {
  if (weight < 0.5) {
    return start + weight * (end - start);
  } else {
    return end - (end - start) * (1.0 - weight);
  }
}

__device__ float lerp(float start, float end, float weight) {
  if (weight < 0.5f) {
    return start + weight * (end - start);
  } else {
    return end - (end - start) * (1.0f - weight);
  }
}

__device__ float lerp(float start, float end, double weight) {
  return lerp(start, end, static_cast<float>(weight));
}

__device__ constexpr int max(int a, int b) {
  return a > b ? a : b;
}

__device__ constexpr int64_t max(int64_t a, int b) {
  return a > (int64_t)b ? a : (int64_t)b;
}

__device__ constexpr int64_t max(int a, int64_t b) {
  return (int64_t)a > b ? (int64_t)a : b;
}

__device__ constexpr int64_t max(int64_t a, int64_t b) {
  return a > b ? a : b;
}

__device__ double fmax(double a, double b) {
  // check and propagate NaN
  if (a != a) {
    return a;
  } else { // If b is nan, it will be returned in the next line
    return a > b ? a : b;
  }
}

__device__ float fmax(float a, float b) {
  // check and propagate NaN
  if (a != a) {
    return a;
  } else { // If b is nan, it will be returned in the next line
    return a > b ? a : b;
  }
}

__device__ constexpr int min(int a, int b) {
  return a > b ? b : a;
}

__device__ constexpr int64_t min(int64_t a, int b) {
  return (int64_t)a > b ? b : (int64_t)a;
}

__device__ constexpr int64_t min(int a, int64_t b) {
  return a > (int64_t)b ? (int64_t)b : a;
}

__device__ constexpr int64_t min(int64_t a, int64_t b) {
  return a > b ? b : a;
}

__device__ double fmin(double a, double b) {
  // check and propagate NaN
  if (b != b) {
    return b;
  } else { // If a is nan, it will be returned in the next line
    return a > b ? b : a;
  }
}

__device__ float fmin(float a, float b) {
  // check and propagate NaN
  if (b != b) {
    return b;
  } else { // If a is nan, it will be returned in the next line
    return a > b ? b : a;
  }
}

__device__ constexpr int alignBufferSize(int buffer, int size) {
  return (buffer + (size - 1)) & ~(size - 1);
}

__device__ double clamp(double x, double minv, double maxv) {
  return fmin(fmax(x, minv), maxv);
}

__device__ float clamp(float x, double minv, double maxv) {
  return fmin(fmax((double)x, minv), maxv);
}

__device__ int clamp(int x, int64_t minv, int64_t maxv) {
  return min(max((int64_t)x, minv), maxv);
}

__device__ int64_t clamp(int64_t x, int64_t minv, int64_t maxv) {
  return min(max(x, minv), maxv);
}

__device__ double frac(double x) {
  return x - trunc(x);
}

__device__ float frac(float x) {
  return x - trunc(x);
}

__device__ double reciprocal(double x) {
  return 1 / x;
}

__device__ float reciprocal(float x) {
  return 1 / x;
}

__device__ double relu(double x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(float x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int64_t x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(int x) {
  return x <= 0 ? 0 : x;
}

__device__ double remainder(double a, double b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ float remainder(float a, float b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

__device__ double silu(double x) {
  return x * sigmoid(x);
}

__device__ float silu(float x) {
  return x * sigmoid(x);
}

__device__ double threshold(double x, double t, double v) {
  return x <= t ? v : x;
}

__device__ float threshold(float x, double t, double v) {
  return x <= t ? v : x;
}

__device__ int threshold(int x, int64_t t, int64_t v) {
  return x <= t ? v : x;
}

__device__ int64_t threshold(int64_t x, int64_t t, int64_t v) {
  return x <= t ? v : x;
}

__device__ constexpr int64_t remainder(int64_t a, int64_t b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ constexpr int remainder(int a, int b) {
  auto mod = a % b;
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ constexpr int64_t fmod(int64_t a, int64_t b) {
  return a % b;
}

__device__ constexpr int fmod(int a, int b) {
  return a % b;
}

__device__ constexpr double fmod(double a, double b) {
  return ::fmod(a, b);
}

__device__ constexpr float fmod(float a, float b) {
  return ::fmod(a, b);
}

__device__ constexpr double nextafter(double a, double b) {
  return ::nextafter(a, b);
}

__device__ constexpr float nextafter(float a, float b) {
  return ::nextafterf(a, b);
}

template <typename T>
__device__ T pow(T a, T b) {
  if (b < 0) {
    if (a == 1) {
      return 1;
    } else if (a == -1) {
      auto negative = (-b) % static_cast<T>(2);
      return negative ? -1 : 1;
    } else {
      return 0;
    }
  } else {
    T result = 1;
    while (b) {
      if (b & 1) {
        result *= a;
      }
      b /= 2;
      a *= a;
    }
    return result;
  }
}

template __device__ int pow<int>(int a, int b);
template __device__ int64_t pow<int64_t>(int64_t a, int64_t b);

template <>
__device__ float pow<float>(float a, float b) {
  return ::pow(a, b);
}

template <>
__device__ double pow<double>(double a, double b) {
  return ::pow(a, b);
}

__device__ float pow(float a, int b) {
  return pow(a, (float)b);
}

__device__ double pow(double a, int b) {
  return pow(a, (double)b);
}

__device__ float pow(float a, int64_t b) {
  return pow(a, (float)b);
}

__device__ double pow(double a, int64_t b) {
  return pow(a, (double)b);
}

__device__ int64_t pow(int64_t a, int b) {
  return pow(a, (int64_t)b);
}

__device__ int64_t pow(int a, int64_t b) {
  return pow((int64_t)a, b);
}

__device__ double rsqrt(double z) {
  return ::rsqrt(z);
}

__device__ float rsqrt(float z) {
  return ::rsqrtf(z);
}

__device__ int rsqrt(int z) {
  return ::rsqrtf((float)z);
}

__device__ int64_t rsqrt(int64_t z) {
  return ::rsqrt((double)z);
}

__device__ double signbit(double a) {
  return ::signbit(a);
}

__device__ float signbit(float a) {
  return ::signbit(a);
}

__device__ int signbit(int a) {
  return a < 0;
}

__device__ int64_t signbit(int64_t a) {
  return a < 0;
}

// Reference:
// https://en.wikipedia.org/wiki/Euclidean_algorithm#Implementations
// https://github.com/pytorch/pytorch/blob/c9f4f01981fd73fcc7c27676cc50230cd1b5bc22/aten/src/ATen/native/Math.h#L1232
template <typename T>
__device__ T gcd(T a, T b) {
  a = abs(a);
  b = abs(b);
  while (b != 0) {
    auto t = b;
    b = a % b;
    a = t;
  }
  return a;
}

template <typename T>
bool isfinite(T x) {
  return ::isfinite(x);
}

// ref:
// https://github.com/NVIDIA/cutlass/blob/6fbc0d33800008d3180d3fefed4e1a653e5f72a0/include/cutlass/bfloat16.h#L213
template <>
bool isfinite<__bfloat>(__bfloat x) {
  const auto exponent_biased = int((x.raw() >> 7) & 0x0ff);
  return exponent_biased != 0x0ff;
}

// ref:
// https://github.com/NVIDIA/cutlass/blob/6fbc0d33800008d3180d3fefed4e1a653e5f72a0/include/cutlass/half.h#L511
template <>
bool isfinite<__half>(__half x) {
  const auto exponent_biased = int((x.raw() >> 10) & 0x1f);
  return exponent_biased != 0x1f;
}

template <typename T>
bool isinf(T x) {
  return ::isinf(x);
}

////////////////////////////////////////////////////////////
// TODO: the following overloads are only needed for CUDA //
// 10.2 Please remove when CUDA 10.2 support is dropped   //
////////////////////////////////////////////////////////////

bool isinf(int64_t x) {
  return false;
}

bool isinf(int x) {
  return false;
}

bool isinf(short x) {
  return false;
}

bool isinf(char x) {
  return false;
}

bool isinf(unsigned char x) {
  return false;
}

bool isinf(bool x) {
  return false;
}

bool isfinite(int64_t x) {
  return true;
}

bool isfinite(int x) {
  return true;
}

bool isfinite(short x) {
  return true;
}

bool isfinite(char x) {
  return true;
}

bool isfinite(unsigned char x) {
  return true;
}

bool isfinite(bool x) {
  return true;
}

////////////////////////////////////////////////////////////
//                        End TODO                        //
////////////////////////////////////////////////////////////

template <typename T>
bool isnan(T x) {
  return x != x;
}

template <typename T>
bool isneginf(T x) {
  return x < 0 && isinf(x);
}

template <typename T>
bool isposinf(T x) {
  return x > 0 && isinf(x);
}

template <typename T>
bool isreal(T x) {
  return true;
}

// Return the current value of the cycle counter
__device__ inline int64_t readCycleCounter() {
  // Ensures preceding memory operations are completed. Doing this
  // would make sense for measuring elapsed times enclosed with this
  // function.
  __threadfence();
  return clock64();
}

__device__ float print_impl(const char* name, float value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ double print_impl(const char* name, double value) {
  printf(
      "%s = %lf @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ int print_impl(const char* name, int value) {
  printf(
      "%s = %d @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ int64_t print_impl(const char* name, int64_t value) {
  printf(
      "%s = %ld @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value,
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ bool print_impl(const char* name, bool value) {
  printf(
      "%s = %s @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      value ? "true" : "false",
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

__device__ __half print_impl(const char* name, __half value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      __half2float(value),
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}

#if __CUDACC_VER_MAJOR__ >= 11
__device__ __bfloat print_impl(const char* name, __bfloat value) {
  printf(
      "%s = %f @ threadIdx=(%d,%d,%d), blockIdx=(%d,%d,%d)\n",
      name,
      __bfloat2float(value),
      (int)threadIdx.x,
      (int)threadIdx.y,
      (int)threadIdx.z,
      (int)blockIdx.x,
      (int)blockIdx.y,
      (int)blockIdx.z);
  return value;
}
#endif

#define print(...) print_impl(#__VA_ARGS__, (__VA_ARGS__))

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace index_utils {

// Utility functions

// Total size of provided dimension
template <typename _dim3>
__device__ __forceinline__ nvfuser_index_t size(const _dim3& d) {
  return (nvfuser_index_t)d.x * (nvfuser_index_t)d.y * (nvfuser_index_t)d.z;
}

// Linearized indexing of idx based on dim, if bool==false that dimension does
// not participate
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ nvfuser_index_t maskedOffset(const _dim3& idx, const _dim3_2& dim) {
  nvfuser_index_t offset = 0;
  if (Z)
    offset += idx.z;
  if (Y)
    offset = offset * dim.y + idx.y;
  if (X)
    offset = offset * dim.x + idx.x;
  return offset;
}

// Linearized indexing of idx based on dim. All dimensions participate.
template <typename _dim3, typename _dim3_2>
__device__ nvfuser_index_t offset(const _dim3& idx, const _dim3_2& dim) {
  nvfuser_index_t offset = idx.z;
  offset = offset * dim.y + idx.y;
  offset = offset * dim.x + idx.x;
  return offset;
}

// Masks the provided dim3, those == false get truncated to 1
template <bool X, bool Y, bool Z, typename _dim3>
__device__ dim3 maskedDims(const _dim3& dim) {
  return dim3{
      X ? (unsigned)dim.x : 1U,
      Y ? (unsigned)dim.y : 1U,
      Z ? (unsigned)dim.z : 1U};
}

// Provides total size of dim with masking, those dims == false do not
// participate in the size calculation
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ nvfuser_index_t maskedSize(const _dim3& dim) {
  return size(maskedDims<X_BLOCK, Y_BLOCK, Z_BLOCK>(dim));
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3>
__device__ bool maskedIsZero(const _dim3& idx) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == 0;
  if (Y)
    isZero = isZero && idx.y == 0;
  if (Z)
    isZero = isZero && idx.z == 0;
  return isZero;
}

// Checks if provided idx is zero on those dims == true
template <bool X, bool Y, bool Z, typename _dim3, typename _dim3_2>
__device__ bool maskedIsLast(const _dim3& idx, const _dim3_2& dim) {
  bool isZero = true;
  if (X)
    isZero = isZero && idx.x == dim.x - 1;
  if (Y)
    isZero = isZero && idx.y == dim.y - 1;
  if (Z)
    isZero = isZero && idx.z == dim.z - 1;
  return isZero;
}

} // namespace index_utils

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// std::tuple-like type
template <typename... Types>
struct Tuple;

#define TUPLE_INCREMENT_PTR(idx)                                        \
  do {                                                                  \
    static_assert(                                                      \
        IsPointerType<T##idx>::value, "Invalid for non-pointer types"); \
    val##idx += offset;                                                 \
  } while (0)

template <typename T0>
struct Tuple<T0> {
  T0 val0;

  Tuple() = default;

  __device__ Tuple(T0 _val0) : val0(_val0) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
  }
};

template <typename T0, typename T1>
struct Tuple<T0, T1> {
  T0 val0;
  T1 val1;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1) : val0(_val0), val1(_val1) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
  }
};

template <typename T0, typename T1, typename T2>
struct Tuple<T0, T1, T2> {
  T0 val0;
  T1 val1;
  T2 val2;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2)
      : val0(_val0), val1(_val1), val2(_val2) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
  }
};

template <typename T0, typename T1, typename T2, typename T3>
struct Tuple<T0, T1, T2, T3> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3)
      : val0(_val0), val1(_val1), val2(_val2), val3(_val3) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
  }
};

template <typename T0, typename T1, typename T2, typename T3, typename T4>
struct Tuple<T0, T1, T2, T3, T4> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3, T4 _val4)
      : val0(_val0), val1(_val1), val2(_val2), val3(_val3), val4(_val4) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5>
struct Tuple<T0, T1, T2, T3, T4, T5> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;

  Tuple() = default;

  __device__ Tuple(T0 _val0, T1 _val1, T2 _val2, T3 _val3, T4 _val4, T5 _val5)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6>
struct Tuple<T0, T1, T2, T3, T4, T5, T6> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;
  T6 val6;

  Tuple() = default;

  __device__ Tuple(
      T0 _val0,
      T1 _val1,
      T2 _val2,
      T3 _val3,
      T4 _val4,
      T5 _val5,
      T6 _val6)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5),
        val6(_val6) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
    TUPLE_INCREMENT_PTR(6);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7>
struct Tuple<T0, T1, T2, T3, T4, T5, T6, T7> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;
  T6 val6;
  T7 val7;

  Tuple() = default;

  __device__ Tuple(
      T0 _val0,
      T1 _val1,
      T2 _val2,
      T3 _val3,
      T4 _val4,
      T5 _val5,
      T6 _val6,
      T7 _val7)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5),
        val6(_val6),
        val7(_val7) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
    TUPLE_INCREMENT_PTR(6);
    TUPLE_INCREMENT_PTR(7);
  }
};

template <
    typename T0,
    typename T1,
    typename T2,
    typename T3,
    typename T4,
    typename T5,
    typename T6,
    typename T7,
    typename T8,
    typename T9,
    typename T10,
    typename T11,
    typename T12,
    typename T13,
    typename T14,
    typename T15>
struct Tuple<
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    T8,
    T9,
    T10,
    T11,
    T12,
    T13,
    T14,
    T15> {
  T0 val0;
  T1 val1;
  T2 val2;
  T3 val3;
  T4 val4;
  T5 val5;
  T6 val6;
  T7 val7;
  T8 val8;
  T9 val9;
  T10 val10;
  T11 val11;
  T12 val12;
  T13 val13;
  T14 val14;
  T15 val15;

  Tuple() = default;

  __device__ Tuple(
      T0 _val0,
      T1 _val1,
      T2 _val2,
      T3 _val3,
      T4 _val4,
      T5 _val5,
      T6 _val6,
      T7 _val7,
      T8 _val8,
      T9 _val9,
      T10 _val10,
      T11 _val11,
      T12 _val12,
      T13 _val13,
      T14 _val14,
      T15 _val15)
      : val0(_val0),
        val1(_val1),
        val2(_val2),
        val3(_val3),
        val4(_val4),
        val5(_val5),
        val6(_val6),
        val7(_val7),
        val8(_val8),
        val9(_val9),
        val10(_val10),
        val11(_val11),
        val12(_val12),
        val13(_val13),
        val14(_val14),
        val15(_val15) {}

  // Only valid when instantiated for pointer types
  __device__ void operator+=(nvfuser_index_t offset) {
    TUPLE_INCREMENT_PTR(0);
    TUPLE_INCREMENT_PTR(1);
    TUPLE_INCREMENT_PTR(2);
    TUPLE_INCREMENT_PTR(3);
    TUPLE_INCREMENT_PTR(4);
    TUPLE_INCREMENT_PTR(5);
    TUPLE_INCREMENT_PTR(6);
    TUPLE_INCREMENT_PTR(7);
    TUPLE_INCREMENT_PTR(8);
    TUPLE_INCREMENT_PTR(9);
    TUPLE_INCREMENT_PTR(10);
    TUPLE_INCREMENT_PTR(11);
    TUPLE_INCREMENT_PTR(12);
    TUPLE_INCREMENT_PTR(13);
    TUPLE_INCREMENT_PTR(14);
    TUPLE_INCREMENT_PTR(15);
  }
};

#undef TUPLE_INCREMENT_PTR

// Accessor for Tuple
template <int idx>
struct get;

#define DEFINE_TUPLE_GET(idx)                              \
  template <>                                              \
  struct get<idx> {                                        \
    template <typename Tuple>                              \
    __device__ auto& operator()(Tuple& vals) {             \
      return vals.val##idx;                                \
    }                                                      \
    template <typename Tuple>                              \
    __device__ const auto& operator()(const Tuple& vals) { \
      return vals.val##idx;                                \
    }                                                      \
  };

DEFINE_TUPLE_GET(0);
DEFINE_TUPLE_GET(1);
DEFINE_TUPLE_GET(2);
DEFINE_TUPLE_GET(3);
DEFINE_TUPLE_GET(4);
DEFINE_TUPLE_GET(5);
DEFINE_TUPLE_GET(6);
DEFINE_TUPLE_GET(7);
DEFINE_TUPLE_GET(8);
DEFINE_TUPLE_GET(9);
DEFINE_TUPLE_GET(10);
DEFINE_TUPLE_GET(11);
DEFINE_TUPLE_GET(12);
DEFINE_TUPLE_GET(13);
DEFINE_TUPLE_GET(14);
DEFINE_TUPLE_GET(15);
#undef DEFINE_TUPLE_GET

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset = 0);

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset = 0);

template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    typename DstType::template ValType<0> src);

template <typename... Types>
class LocalTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;

  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;

  LocalTuple() = default;

  __device__ explicit LocalTuple(Types... args) : vals_(args...) {}

  __device__ LocalTuple(const LocalTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ LocalTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  __device__ LocalTuple& operator=(const LocalTuple<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ LocalTuple& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ auto& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<Types...> vals_;
};

template <bool is_volatile, typename... Types>
class PtrTupleBase {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;
  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;
  template <int val_idx>
  using TypeIMaybeVolatile = typename MaybeVolatile<
      typename TypeSelector<val_idx, Types...>::type,
      is_volatile>::type;

  __device__ PtrTupleBase(Types*... args) : vals_(args...) {}

  __device__ PtrTupleBase(const PtrTupleBase& other) : vals_(other.vals_) {}

  // Note: this is a deep copy
  __device__ PtrTupleBase& operator=(
      const PtrTupleBase<is_volatile, Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ PtrTupleBase& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ TypeIMaybeVolatile<val_idx>& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return ((TypeIMaybeVolatile<val_idx>*)get<val_idx>()(vals_))[ptr_offset];
  }

  template <int val_idx>
  __device__ const TypeIMaybeVolatile<val_idx>& val(
      nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return ((TypeIMaybeVolatile<val_idx>*)get<val_idx>()(vals_))[ptr_offset];
  }

  __device__ void operator+=(nvfuser_index_t ptr_offset) {
    vals_ += ptr_offset;
  }

 private:
  Tuple<Types*...> vals_;
};

template <typename... Types>
class RefTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;
  template <int idx>
  using ValType = typename TypeSelector<idx, Types...>::type;

  __device__ RefTuple(Types&... args) : vals_(args...) {}

  __device__ RefTuple(const RefTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ RefTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  __device__ RefTuple& operator=(const RefTuple<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <template <typename...> typename TupleType>
  __device__ RefTuple& operator=(const TupleType<Types...>& other) {
    copyTuple(*this, other);
    return *this;
  }

  template <int val_idx>
  __device__ auto& val(nvfuser_index_t ptr_offset = 0) {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<Types&...> vals_;
};

template <typename DstType, typename SrcType, int num_vals>
struct TupleCopy {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset) {
    static_assert(
        IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::
            value,
        "Invalid value types");
    TupleCopy<DstType, SrcType, num_vals - 1>::copy(
        dst, dst_offset, src, src_offset);
    dst.val<num_vals - 1>(dst_offset) = src.val<num_vals - 1>(src_offset);
  }
};

template <typename DstType, typename SrcType>
struct TupleCopy<DstType, SrcType, 0> {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset) {}
};

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset) {
  static_assert(
      IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::value,
      "Invalid value types");
  TupleCopy<DstType, SrcType, DstType::num_vals>::copy(
      dst, dst_offset, src, src_offset);
};

template <typename DstType, typename SrcType>
__inline__ __device__ static void copyTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset) {
  copyTuple<DstType, SrcType>(dst, 0, src, src_offset);
};

template <typename DstType, int num_vals>
struct TupleSet {
  __inline__ __device__ static void set(
      DstType& dst,
      nvfuser_index_t dst_offset,
      typename DstType::template ValType<0> src) {
    static_assert(
        IsSameType<
            typename DstType::template ValType<num_vals - 1>,
            typename DstType::template ValType<0>>::value,
        "Invalid value types");
    TupleSet<DstType, num_vals - 1>::set(dst, dst_offset, src);
    dst.val<num_vals - 1>(dst_offset) = src;
  }
};

template <typename DstType>
struct TupleSet<DstType, 0> {
  __inline__ __device__ static void set(
      DstType& dst,
      nvfuser_index_t dst_offset,
      typename DstType::template ValType<0> src) {}
};

template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    typename DstType::template ValType<0> src) {
  TupleSet<DstType, DstType::num_vals>::set(dst, dst_offset, src);
};


template <typename DstType>
__inline__ __device__ static void setTuple(
    DstType& dst,
    typename DstType::template ValType<0> src) {
  setTuple(dst, 0, src);
};

template <typename DstType, typename SrcType, typename PredType, int num_vals>
struct PredicatedTupleCopy {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset,
      const PredType& pred) {
    static_assert(
        IsSameType<typename PredType::template ValType<num_vals - 1>, bool>::
            value,
        "Invalid predicate type");
    PredicatedTupleCopy<DstType, SrcType, PredType, num_vals - 1>::copy(
        dst, dst_offset, src, src_offset, pred);
    if (pred.val<num_vals - 1>(0)) {
      dst.val<num_vals - 1>(dst_offset) = src.val<num_vals - 1>(src_offset);
    }
  }
};

template <typename DstType, typename SrcType, typename PredType>
struct PredicatedTupleCopy<DstType, SrcType, PredType, 0> {
  __inline__ __device__ static void copy(
      DstType& dst,
      nvfuser_index_t dst_offset,
      const SrcType& src,
      nvfuser_index_t src_offset,
      const PredType& pred) {}
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset,
    const PredType& pred) {
  static_assert(
      IsSameType<typename DstType::ValTypes, typename SrcType::ValTypes>::value,
      "Invalid value types");
  static_assert(
      PredType::num_vals == DstType::num_vals, "Invalid predicate type");
  PredicatedTupleCopy<DstType, SrcType, PredType, DstType::num_vals>::copy(
      dst, dst_offset, src, src_offset, pred);
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset,
    const PredType& pred) {
  copyTupleIf(dst, 0, src, src_offset, pred);
};

template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyTupleIf(
    DstType& dst,
    const SrcType& src,
    const PredType& pred) {
  copyTupleIf(dst, 0, src, 0, pred);
};

// Can a generic const and non-const RefTupe be defined?
template <typename... Types>
class ConstRefTuple {
 public:
  static constexpr int num_vals = sizeof...(Types);
  using ValTypes = TypeList<Types...>;

  __device__ ConstRefTuple(const Types&... args) : vals_(args...) {}

  __device__ ConstRefTuple(const ConstRefTuple& other) : vals_(other.vals_) {}

  template <template <typename...> typename TupleType>
  __device__ ConstRefTuple(const TupleType<Types...>& other) {
    copyTuple(*this, other);
  }

  template <int val_idx>
  __device__ const auto& val(nvfuser_index_t ptr_offset = 0) const {
    static_assert(val_idx < num_vals, "Out-of-range value index");
    return get<val_idx>()(vals_);
  }

 private:
  Tuple<const Types&...> vals_;
};

template <typename... Types>
using PtrTuple = PtrTupleBase<false, Types...>;

template <typename... Types>
using VolatilePtrTuple = PtrTupleBase<true, Types...>;

// Define a LocalTuple of NumVals values of type Type
template <int NumVals, typename Type>
struct MakeLocalTuple;

template <typename Type>
struct MakeLocalTuple<1, Type> {
  using type = LocalTuple<Type>;
};

template <typename Type>
struct MakeLocalTuple<2, Type> {
  using type = LocalTuple<Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<3, Type> {
  using type = LocalTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<4, Type> {
  using type = LocalTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<5, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<6, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<7, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<8, Type> {
  using type = LocalTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeLocalTuple<16, Type> {
  using type = LocalTuple<
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type>;
};

template <int NumVals, typename Type>
struct MakeRefTuple;

template <typename Type>
struct MakeRefTuple<1, Type> {
  using type = RefTuple<Type>;
};

template <typename Type>
struct MakeRefTuple<2, Type> {
  using type = RefTuple<Type, Type>;
};

template <typename Type>
struct MakeRefTuple<3, Type> {
  using type = RefTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<4, Type> {
  using type = RefTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<5, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<6, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<7, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<8, Type> {
  using type = RefTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeRefTuple<16, Type> {
  using type = RefTuple<
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type>;
};

template <int NumVals, typename Type>
struct MakeConstRefTuple;

template <typename Type>
struct MakeConstRefTuple<1, Type> {
  using type = ConstRefTuple<Type>;
};

template <typename Type>
struct MakeConstRefTuple<2, Type> {
  using type = ConstRefTuple<Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<3, Type> {
  using type = ConstRefTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<4, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<5, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<6, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<7, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<8, Type> {
  using type = ConstRefTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeConstRefTuple<16, Type> {
  using type = ConstRefTuple<
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type>;
};

template <int NumVals, typename Type>
struct MakeVolatilePtrTuple;

template <typename Type>
struct MakeVolatilePtrTuple<1, Type> {
  using type = VolatilePtrTuple<Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<2, Type> {
  using type = VolatilePtrTuple<Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<3, Type> {
  using type = VolatilePtrTuple<Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<4, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<5, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<6, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<7, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<8, Type> {
  using type = VolatilePtrTuple<Type, Type, Type, Type, Type, Type, Type, Type>;
};

template <typename Type>
struct MakeVolatilePtrTuple<16, Type> {
  using type = VolatilePtrTuple<
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type,
      Type>;
};

// Utility definitions. Currently only used with LocalTuple

template <int idx, typename BinaryFunc, typename... DataTypes>
struct TupleBinaryOp {
  static __inline__ __device__ void apply(
      BinaryFunc func,
      const LocalTuple<DataTypes...>& lhs,
      const LocalTuple<DataTypes...>& rhs,
      LocalTuple<DataTypes...>& result) {
    TupleBinaryOp<idx - 1, BinaryFunc, DataTypes...>::apply(
        func, lhs, rhs, result);
    result.val<idx - 1>(0) = func(lhs.val<idx - 1>(0), rhs.val<idx - 1>(0));
  }
};

template <typename BinaryFunc, typename... DataTypes>
struct TupleBinaryOp<0, BinaryFunc, DataTypes...> {
  static __inline__ __device__ void apply(
      BinaryFunc func,
      const LocalTuple<DataTypes...>& lhs,
      const LocalTuple<DataTypes...>& rhs,
      LocalTuple<DataTypes...>& result) {}
};

template <typename BinaryFunc, typename... DataTypes>
__inline__ __device__ LocalTuple<DataTypes...> apply(
    BinaryFunc func,
    const LocalTuple<DataTypes...>& lhs,
    const LocalTuple<DataTypes...>& rhs) {
  LocalTuple<DataTypes...> result = lhs;
  TupleBinaryOp<sizeof...(DataTypes), BinaryFunc, DataTypes...>::apply(
      func, result, rhs, result);
  return result;
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    const LocalTuple<BoolTypes...>& lhs,
    const LocalTuple<BoolTypes...>& rhs) {
  return apply([](bool x, bool y) { return x && y; }, lhs, rhs);
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    bool lhs,
    const LocalTuple<BoolTypes...>& rhs) {
  LocalTuple<BoolTypes...> lhs_tuple;
  setTuple(lhs_tuple, lhs);
  return lhs_tuple && rhs;
}

template <typename... BoolTypes>
__inline__ __device__ LocalTuple<BoolTypes...> operator&&(
    const LocalTuple<BoolTypes...>& lhs,
    bool rhs) {
  LocalTuple<BoolTypes...> rhs_tuple;
  setTuple(rhs_tuple, rhs);
  return lhs && rhs_tuple;
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Default block synchronization. Just use __barrier_sync
namespace block_sync {

__forceinline__ __device__ void init() {}

// Thread-block synchronization
template <bool aligned>
__forceinline__ __device__ void sync() {
  if constexpr (aligned) {
    __syncthreads();
  } else {
    __barrier_sync(0);
  }
}

} // namespace block_sync

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace grid_sync {

// Get the first bit in a 64 bit integer
#define FIRST_UINT64_BIT ((uint64_t)1 << (sizeof(uint64_t) * 8 - 1))

template <typename T>
__device__ T globalAsVolatile(volatile T& global_val) {
  return global_val;
}

// A grid synchronization that can be called multiple times in a kernel assuming
// all the blocks fit on device at once. The semaphore is an integer semaphore
// assumed to be initialized to 0 before launching the kernel. The persistent
// option should be envoked if this sync will be called multiple times in one
// kernel (i.e. having a grid reduce within a loop). Having multiple grid syncs
// called once in the same kernel does not require persistent mode. Segment size
// is the number of blocks participating in the sync in the dimensions marked by
// [X,Y,Z]_BLOCK. The granularity of this sync are those dimensions. I.E.
// Marking X and Y but not Z means there should be Z semaphores of size X*Y.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool PERSISTENT,
    bool Aligned>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const bool last_block) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync<Aligned>();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Get increment value, only want a single block to have the large
    // increment, doesn't really matter which one, the goal is to flip/flop the
    // first bit of a uint64_t value, since our semaphores are actualy int64_t
    // we will just reinterpret_cast it to act as a uint64_t
    uint64_t semaphore_increment = 1;

    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    if (last_block) {
      semaphore_increment = FIRST_UINT64_BIT - (segment_size - 1);
    }

    uint64_t oldArrive =
        atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), semaphore_increment);

    // If for persistent kernels, lock all blocks until the semaphore has been
    // reached. Make sure we access semaphore as a volatile address so we get
    // the global memory updates.
    unsigned int ns = 8;
    while ((PERSISTENT || last_block) &&
           ((oldArrive ^ globalAsVolatile(semaphore)) & FIRST_UINT64_BIT) ==
               0) {
      // Put a sleep here so we have some breaks in probing the global
      // semaphore, giving a better chance for other warps/blocks to catch up.
#if __CUDA_ARCH__ >= 700
      // __nanosleep only available on compute capability 7.0 or higher
      __nanosleep(ns); // avoids busy waiting
      if (ns < 256) {
        ns *= 2;
      }
#endif
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync<Aligned>();
}

template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool PERSISTENT,
    bool Aligned>
__device__ void sync(int64_t& semaphore, const uint64_t& segment_size) {
  sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT, Aligned>(
      semaphore,
      segment_size,
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim));
}

// Grid sync that can be called multiple times in the same kernel without all
// blocks being resident on device. This allows grid sync to be called multiple
// times as long as it's not broadcasted on the parallel axis it was reduced on.
//
// n_entrances is how many times every block is expected to enter into this
// function. All blocks must enter n_entrances times. The last block is only
// allowed to proceed once all other blocks have entered n_entrance
// times.
//
// Note that this is not currently used by grid and welford reduction
// as they use a separate sync flag for each each grid sync call.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, bool Aligned>
__device__ void sync(
    int64_t& semaphore,
    const uint64_t& segment_size,
    const nvfuser_index_t n_entrances) {
  // Finish all global memory transactions before synchronizing
  __threadfence();

  // Synchronize all threads in a block before synchronizing blocks
  block_sync::sync<Aligned>();

  // Only allow linear_tid == 0 to participate in the synchronization
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // Makes the assumption that blocks are in increasing order, this is not
    // guaranteed by CUDA but this is the current behavior, and unlikely to
    // change.
    bool last_block =
        index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    if (last_block) {
      int64_t finished_val =
          ((int64_t)(index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(
                         gridDim) -
                     1)) *
          ((int64_t)n_entrances);

      unsigned int ns = 8;
      // Last block needs to wait for all other blocks to finish
      while (globalAsVolatile(semaphore) < finished_val) {
#if __CUDA_ARCH__ >= 700
        // __nanosleep only available on compute capability 7.0 or higher
        __nanosleep(ns); // avoids busy waiting
        if (ns < 256) {
          ns *= 2;
        }
#endif
      }
    } else {
      auto old = atomicAdd(reinterpret_cast<uint64_t*>(&semaphore), 1);
    }
  }

  // Sync block to make sure all other threads are waiting on the sync
  block_sync::sync<Aligned>();
}

// Non-blocking function to read the semaphore value in each calling thread
__device__ int64_t semaphoreFetch(int64_t* semaphore) {
  int64_t state;
  // NOTE: acquire/release operations require sm_70 or higher
  // https://docs.nvidia.com/cuda/archive/12.3.0/parallel-thread-execution/index.html#scopes-and-applicability
  asm volatile("ld.global.acquire.gpu.b64 %0, [%1];\n"
               : "=l"(state)
               : "l"(semaphore));
  return state;
}

// Non-blocking function to set semaphore to new_value
__device__ void semaphoreRelease(int64_t* semaphore, int64_t new_value) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    // NOTE: acquire/release operations require sm_70 or higher
    // https://docs.nvidia.com/cuda/archive/12.3.0/parallel-thread-execution/index.html#scopes-and-applicability
    asm volatile("st.global.release.gpu.b64 [%0], %1;\n"
                 :
                 : "l"(semaphore), "l"(new_value));
  }
}

// First thread waits until fetched semaphore value matches trigger
__device__ void semaphoreWait(int64_t* semaphore, int64_t trigger_value) {
  int64_t status = -1;
  // Cutlass uses a loop like this, and has a facility where any thread can
  // fetch the semaphore value ahead of waiting. This could reduce the wait
  // time potentially but requires placement of the early fetch.
  // https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/semaphore.h
  // while (__syncthreads_and(status != trigger_value)) {
  // As soon as any thread in the block observes the trigger then it is
  // safe to proceed
  // Instead, we simply use the first thread in the block to do busy waiting.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    while (status != trigger_value) {
      status = semaphoreFetch(semaphore);
    }
  }
}

// Serialize blocks in segments indicated by the [XYZ]_BLOCK template arguments.
// This should be called at the beginning of the section to be serialized.
// Assumes semaphore is initialized to zero. This function always synchronizes
// the thread block.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__device__ void blockSerializeWait(int64_t* semaphore) {
  int segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);
  int block_idx_in_segment =
      index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (block_idx_in_segment > 0) {
    semaphoreWait(semaphore, block_idx_in_segment);
  }
  __syncthreads();
}

// Serialize blocks in segments indicated by the [XYZ]_BLOCK template arguments.
// This should be called at the end of the section to be serialized.
// This function always cleans up the semaphore; i.e. the last block writes the
// value 0 to the semaphore when complete. This function always synchronizes
// the thread block.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK>
__device__ void blockSerializeRelease(int64_t* semaphore) {
  int segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);
  int block_idx_in_segment =
      index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
  bool last_block = block_idx_in_segment == segment_size - 1;

  __syncthreads();
  semaphoreRelease(semaphore, last_block ? 0 : block_idx_in_segment + 1);
}

} // namespace grid_sync

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Reference:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-barrier
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

namespace mbarrier {

__device__ inline void init(
    uint32_t smem_barrier_ptr,
    uint32_t thread_count = 1) {
  asm volatile(
      "mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_barrier_ptr),
      "r"(thread_count));
}

__device__ inline void inval(uint32_t smem_barrier_ptr) {
  asm volatile("mbarrier.inval.shared.b64 [%0];\n" ::"r"(smem_barrier_ptr));
}

__device__ inline uint64_t arrive(uint32_t smem_barrier_ptr) {
  volatile uint64_t state;
  asm volatile("mbarrier.arrive.shared.b64 %0, [%1];\n"
               : "=l"(state)
               : "r"(smem_barrier_ptr));
  return state;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__device__ inline uint64_t arriveExpectTX(
    uint32_t smem_barrier_ptr,
    uint32_t tx_count) {
  volatile uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;\n"
               : "=l"(state)
               : "r"(smem_barrier_ptr), "r"(tx_count));
  return state;
}
#endif

__device__ inline void wait(uint32_t smem_barrier_ptr, uint64_t state) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile(
      "{\n"
      ".reg .pred                complete;\n"
      "waitLoop:\n"
      "mbarrier.try_wait.shared.b64 complete, [%0], %1;\n"
      "@!complete bra waitLoop;\n"
      "}\n" ::"r"(smem_barrier_ptr),
      "l"(state));
#else
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.test_wait.shared.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "nanosleep.u32 20;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_barrier_ptr),
      "l"(state));
#endif
}

} // namespace mbarrier

#endif // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to false the dimension doesn't
// participate in the reduction. We could start with warp reductions, then
// reduce the warps, this could save some shared memory, but could be slower in
// some instances.
//
//  EXAMPLE USAGE:
//  blockReduceSum<X_THREADS, Y_THREADS, Z_THREADS>
//    (output[output_index], inputs[input_index],
//      [] __device__ (T& a, const T b) { a += b; });
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // If this thread will output a final result
  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  // Size of the reduction segments
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  // Index into the reduction segment
  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  // Index of the reduction segment
  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  // Initialize shared memory
  if (read_pred) {
    shared_mem[smem_offset] = inp_val;
  } else {
    shared_mem[smem_offset] = init_val;
  }

  block_sync::sync<Aligned>();
  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + np2]);
  }
  block_sync::sync<Aligned>();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(shared_mem[smem_offset], shared_mem[smem_offset + factor]);
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
    T result = out;
    reduction_op(result, shared_mem[smem_offset]);
    if (reduction_size > 1) {
      reduction_op(result, shared_mem[smem_offset + 1]);
    }
    out = result;
  }
  block_sync::sync<Aligned>();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename Func>
__device__ void blockReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, T, Func>(
      out,
      inp_val,
      reduction_op,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}

// Each thread in the iteration dimension processes N elements
// Typical usage is in outer reduction where the iteration dimension
// is parallelized by vectorized loads, bidmx. The reduction dimension
// is parallelized by bdimy. This function works as follows:
// (1) Each thread vectorized loads N elements from input register array to
// smem. (2) do N * bdimx parallel reductions in smem.

// TODO: merge `blockIterGroupedReduce` with `blockReduce`
// (1) for-loops are fully unrolled should not cause overhead for `blockReduce`
// (2) used in gridReduce, needs to change correspodning gridReduce function
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    int N, // Number of elements per input array
    typename T,
    typename Func>
__device__ void blockIterGroupedReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // N should be a valid vectorization factor
  static_assert(
      N == 2 || N == 4 || N == 8 || N == 16,
      "N should be a valid vectorization factor, one of (2, 4, 8, 16)!");

  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Adjust shared memory offset for array processing
  unsigned int smem_offset =
      (reduction_idx * reduction_size + reduction_tid) * N;
  if (read_pred) {
    // This section calculates the number of vectorized load operations required
    // to fetch all elements of an array into shared memory, assuming each load
    // can transfer up to 16 bytes. For example, with fusion input vectorized by
    // 8 (N = 8) and computations in fp32 (sizeof(T) = 4 bytes), the total data
    // size is 4 * 8 = 32 bytes, necessitating 32 / 16 = 2 load transactions.
    // Each transaction loads 16 / 4 (bytes per element) = 4 elements.
    if constexpr (sizeof(T) * N <= 16) {
      loadGeneric<T, N>(shared_mem + smem_offset, const_cast<T*>(inp_val));
    } else {
      constexpr unsigned int total_loads = sizeof(T) * N / 16;
      constexpr unsigned int elements_per_load = 16 / sizeof(T);
      static_assert(
          sizeof(T) * N == 16 * total_loads,
          "This combination of vectorization factor and data type is not supported!");
      static_assert(
          sizeof(T) * elements_per_load == 16,
          "This data type is not supported!");
#pragma unroll
      for (unsigned int i = 0; i < total_loads; ++i) {
        loadGeneric<T, elements_per_load>(
            shared_mem + smem_offset + i * elements_per_load,
            const_cast<T*>(inp_val) + i * elements_per_load);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      shared_mem[smem_offset + i] = init_val;
    }
  }

  block_sync::sync<Aligned>();

  // Reduce down to nearest power of 2 for the tree reduction:
  int np2 = 1 << (31 - __clz(reduction_size));

  // Perform parallel reduction for each element in the array
  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      reduction_op(
          shared_mem[smem_offset + i], shared_mem[smem_offset + np2 * N + i]);
    }
  }

  block_sync::sync<Aligned>();

  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        reduction_op(
            shared_mem[smem_offset + i],
            shared_mem[smem_offset + factor * N + i]);
      }
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      T result = out[i];
      reduction_op(result, shared_mem[smem_offset + i]);
      if (reduction_size > 1) {
        reduction_op(
            result,
            shared_mem[smem_offset + N + i]); // Handle the last element if
                                              // reduction size is odd
      }
      out[i] = result;
    }
  }
  block_sync::sync<Aligned>();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    int N, // Number of elements per input array
    typename T,
    typename Func>
__device__ void blockIterGroupedReduce(
    T out[N],
    const T inp_val[N],
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  blockIterGroupedReduce<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, N, T, Func>(
      out,
      inp_val,
      reduction_op,
      shared_mem,
      read_write_pred,
      read_write_pred,
      init_val);
}

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
// Inter-block reduction.
//
// The gridReduce function performs point-wise reductions of scalars across
// thread blocks. Thread blocks are disjointly partitioned into groups,
// "reduction segments", that are collectively defined by boolean template
// parameters, X_BLOCK, Y_BLOCK and Z_BLOCK. Each of X/Y/Z_BLOCK determines
// whether thread blocks along the dimension should be grouped into the same
// reduction segment. Cross-block reducitons are independently done within each
// segment and generates distinctive results per segment. For instance, if all
// of X/Y/Z_BLOCK are true, reductions will be done across all thread blocks
// since there will be just a single segment consisting of all thread blocks. If
// none of them are true, each thread block will become a segment by itself, so
// no reduction will be performed.
//
// The input scalars to reduce within each segment are a certain subset of
// thread-private scalars provided as part of the gridReduce function
// parameters. Boolean template parameters, X_THREAD, Y_THREAD and Z_THREAD,
// determine which subset of the scalars should be used for inter-block
// reductions. Specifically, all the input scalars of threads along each
// dimension will be used when X/Y/Z_THREAD are true. Otherwise, only the value
// held at offset 0 of each dimension will be used. Thus, for example, if all of
// X/Y/Z_THREAD are true, the scalars of all threads in each block will
// participate in inter-block reductions. If all of them are false, only one
// scalar of the thread at threadIdx.x == threadIdx.y == threadIdx.z == 0 will
// be used. In the code below, we call the subset of threads a "reduction
// block". "Participating" thread dimensions here are similar to the
// "non-participating" block dimensions. They come from a block dimension that
// has not been reduced before hitting this grid reduction.
//
// Inter-block reductions perform point-wise reductions of scalars of reduction
// blocks within each reduction segment. More specifically, let rb be a
// reduction block and rs be a reduction segment. Let IN(thread_idx, block_idx)
// denote the input scalar of thread at thread_idx and block_idx. The result of
// each reduction segment, OUT(thread_idx, block_idx_out), is defined only for
// each thread_idx in thread block block_idx_out in the segment as follows:
//
//   OUT(thread_idx, block_idx_out) =
//     Reduction of IN(thread_idx, block_idx) for
//       all block_idx in a reduction segment
//
// OUT is not given for all threads that are not in block_idx_out and the
// reduction block.
//
// See also the function comment of gridReduce.

namespace reduction {

// Reduces all the reduction blocks in each reduction segment. This is the
// "cleanup" stage of a grid reduction.
//
// This is only called by one thread block per reduction segment. The input
// reduction blocks of the segment are stored in an intermediate buffer pointed
// by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
// block is formed.
//
// The size of a reduction block is by definition smaller or equal to the size
// of a thread block. We use the remaining threads to parallelize reductions
// across reduction blocks. For example, when X/Y/Z_THREAD = {true, false,
// false}, we use blockDim.y*blockDim.z threads for each output value. This is
// done first by loading the input values in parallel and then by reducing
// across threads of dimensions whose XYZ_THREAD are false.
//
// Note that what is done here after the loading from global memory is similar
// to what the existing blockReduce function does.
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename Func>
__device__ void gridReduceLastBlock(
    T& out,
    const volatile T* in,
    const nvfuser_index_t
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const nvfuser_index_t
        block_reduction_segment_size, // Number of reductions across the block
    Func reduction_op,
    T* shared_buf,
    bool write_pred,
    T init_val) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  T inp = init_val;

  // Block stride across the reduction until we only have one value per thread
  for (nvfuser_index_t reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    auto work_buf_offset = reduction_i * block_reduction_segment_size +
        block_reduction_segment_idx;
    reduction_op(inp, in[work_buf_offset]);
  }

  // Block reduce the per thread values into per "participating" thread values
  T inp_tmp = init_val;
  blockReduce<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
      inp_tmp, inp, reduction_op, shared_buf, true, init_val);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
    reduction_op(out, inp_tmp);
  }
}

// Reduces per-thread values across threads and thread blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - reduction_op: Scalar reduction function
// - work_buf: Temporary buffer for cross-block reductions
// - sync_flags: A vector of integers for synchronizations
// - shared_buf: Shared memory buffer for intra-block reduction
//
// Thread has valid results based on if it's the last block in the grid
// reduction dimension
//
// Template parameters:
// - X/Y/Z_BLOCK/THREAD: When true, reduces across thread blocks along the X/Y/Z
//   dimensions
// - PERSISTENT_REDUCTION: Indicates grid reduction will be called in a loop, or
//   the result of the grid reduction will be broadcasted and used across the
//   grid. These requires cross grid communication and the grid synchronizations
//   here to actually synchronize across the entire grid. When false the grid is
//   not synchronized, the last block just waits for everyone else to finish and
//   the other blocks can exit early.
// - T: Scalar data type of input/output data
// - Func: Type of scalara reduction function
//
// Template parameters X/Y/Z_BLOCK define a group of thread blocks that are
// reduced together. We call it a reduction segment. Some examples are:
//
// Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which
// includes all thread blocks. It is effecively the same as the grid.
//
// Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an
// individual segment by itself.
//
// Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread
// blocks that have the same blockDim.x. There will be blockDim.y*blockDim.z
// such segments.
//
// X/Y/Z_THREAD also works similarly as X/Y/Z_BLOCK and defines a
// group of threads that are reduced togather.
//
// After the function completes, only one thread block per reduction segment
// gets valid reduction results. There is no guarantee which particular block
// gets the final results.
//
// entrance_ind and n_entrances are allowed when PERSISTENT_REDUCTION = false.
// If a grid reduction call is only called once per thread, entrance_ind == 0
// and n_entrances == 1. However, grid reduction can be called in a loop in a
// thread, in that case entrance_ind is the count of times the function has been
// called, and n_entrances is the total number of times it will be called.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T,
    typename Func>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD, Aligned>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);

  } else {
    // Use a different sync flag for each call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out,
        (T*)work_buf,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op,
        shared_buf,
        write_pred,
        init_val);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

// This is just a wrapper of the above grid reduction routine to
// measure the elapsed cycles. The measurement must be done just by
// one thread, and in this case it should be done by one of the
// threads in the last thread block.
#ifdef NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T,
    typename Func>
__device__ void gridReduce(
    T& out,
    const T& inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduce<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      Aligned,
      T,
      Func>(
      out,
      inp_val,
      reduction_op,
      work_buf,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      init_val,
      entrance_ind,
      n_entrances);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // NVFUSER_PROFILE_KERNEL

template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename Func>
__device__ void gridReduce2PartialReduction(
    const T& inp_val,
    T init_val,
    Func reduction_op,
    volatile T* work_buf,
    T* shared_buf,
    bool read_pred,
    nvfuser_index_t grid_reduction_segment_size,
    nvfuser_index_t idx_in_grid_segment,
    nvfuser_index_t block_reduction_segment_size) {
  T block_reduction_val = init_val;

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
    blockReduce<X_THREAD, Y_THREAD, Z_THREAD, Aligned>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val);
  } else if (read_pred) {
    block_reduction_val = inp_val;
  }

  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    work_buf[work_buf_offset] = block_reduction_val;
  }
}

// 2-way horizontally fused grid reduction
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,

    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,
    bool write_pred,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances) {
  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf1 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  work_buf2 += (entrance_ind * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      Aligned>(
      inp_val1,
      init_val1,
      reduction_op1,
      work_buf1,
      (T1*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  gridReduce2PartialReduction<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      Aligned>(
      inp_val2,
      init_val2,
      reduction_op2,
      work_buf2,
      (T2*)shared_buf,
      read_pred,
      grid_reduction_segment_size,
      idx_in_grid_segment,
      block_reduction_segment_size);

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  } else {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[entrance_ind * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out1,
        work_buf1,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op1,
        (T1*)shared_buf,
        write_pred,
        init_val1);
    gridReduceLastBlock<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
        out2,
        work_buf2,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op2,
        (T2*)shared_buf,
        write_pred,
        init_val2);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

#ifdef NVFUSER_PROFILE_KERNEL
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T1,
    typename Func1,
    typename T2,
    typename Func2>
__device__ void gridReduceGroup(
    T1& out1,
    const T1& inp_val1,
    T1 init_val1,
    Func1 reduction_op1,
    volatile T1* work_buf1,
    T2& out2,
    const T2& inp_val2,
    T2 init_val2,
    Func2 reduction_op2,
    volatile T2* work_buf2,
    int64_t* sync_flags,
    void* shared_buf,
    bool read_pred,
    bool write_pred,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances,
    int64_t& cycles,
    int64_t& count) {
  int64_t start_counter = 0;

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  gridReduceGroup<
      X_BLOCK,
      Y_BLOCK,
      Z_BLOCK,
      X_THREAD,
      Y_THREAD,
      Z_THREAD,
      PERSISTENT_REDUCTION,
      Aligned,
      T1,
      Func1,
      T2,
      Func2>(
      out1,
      inp_val1,
      init_val1,
      reduction_op1,
      work_buf1,
      out2,
      inp_val2,
      init_val2,
      reduction_op2,
      work_buf2,
      sync_flags,
      shared_buf,
      read_pred,
      write_pred,
      entrance_ind,
      n_entrances);

  if (index_utils::maskedIsLast<true, true, true>(blockIdx, gridDim) &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}
#endif // NVFUSER_PROFILE_KERNEL

// This performs a single reduction step, combining a single element "in" with
// a previous value "work". For a serial grid reduction, "work" resides in
// global memory, while "in" and "out" are in registers.
//
// If the write predicate is false, this function returns early (noop). If the
// read predicate is false, "init" is used in place of "in".
//
// If first_step is false, "work" will be read and reduction_op will be called.
// The result will be written back to "work" unless last_step is true.
template <int64_t vec_size, typename T, typename Func>
__device__ void serialReductionStep(
    T* out,
    T* in,
    T init,
    volatile T* work,
    Func reduction_op,
    bool first_step,
    bool last_step,
    bool read_pred,
    bool write_pred) {
  if (!write_pred) {
    return;
  }
  if (read_pred) {
    loadGeneric<T, vec_size>(out, in);
  } else {
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      out[i] = init;
    }
  }
  if (!first_step) {
    T work_reg[vec_size];
    loadGlobalToLocal<T, vec_size, true, CacheOp::Global>(work_reg, work);
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      reduction_op(out[i], work_reg[i]);
    }
  }
  if (!last_step) {
    loadLocalToGlobal<T, vec_size, true>(work, out);
  }
}

// vectorized reduction
template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    int vec_size,
    typename T,
    typename Func>
__device__ void iterGroupedGridReduceLastBlock(
    T* out,
    const volatile T* in,
    const nvfuser_index_t
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const nvfuser_index_t
        block_reduction_segment_size, // Number of reductions across the block
    Func reduction_op,
    T* shared_buf,
    bool write_pred,
    T init_val,
    const nvfuser_index_t grid_segment_size,
    const nvfuser_index_t idx_in_grid_segment) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  // index into iteration dim.
  // Its calculation is same to that in [iterGroupedGridReduce]. Becuase when
  // [iterGroupedGridReduceLastBlock] is called from [iterGroupedGridReduce],
  // X_THREAD, Y_THREAD, Z_THREAD are flipped.
  const auto thread_offset =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // T inp = init_val;
  T inp[vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    inp[i] = init_val;
  }
  // Block stride across the reduction until we only have one value per thread
  for (nvfuser_index_t reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    // index to global memory
    auto work_buf_offset =
        reduction_i * (block_reduction_segment_size * grid_segment_size) +
        block_reduction_segment_size * idx_in_grid_segment + thread_offset;
    work_buf_offset *= vec_size;

    // vectorized load from global memory
    constexpr int total_bytes = vec_size * sizeof(T);
    constexpr int n_loads = total_bytes <= 16 ? 1 : total_bytes / 16;
    constexpr int n_elements = total_bytes <= 16 ? vec_size : 16 / sizeof(T);
#pragma unroll
    for (int i = 0; i < n_loads; i++) {
      int i_offset = i * n_elements;
      T in_reg[n_elements];
      loadGlobalToLocal<T, n_elements, true, CacheOp::Global>(
          &in_reg[0], const_cast<T*>(in + work_buf_offset + i_offset));
#pragma unroll
      for (int j = 0; j < n_elements; j++) {
        reduction_op(inp[i_offset + j], in_reg[j]);
      }
    }
  }

  // Block reduce the per thread values into per "participating" thread values
  // T inp_tmp = init_val;
  T inp_tmp[vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    inp_tmp[i] = init_val;
  }
  blockIterGroupedReduce<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned, vec_size>(
      inp_tmp, inp, reduction_op, shared_buf, true, init_val);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      reduction_op(out[i], inp_tmp[i]);
    }
  }
}

// Main algorithm is same to gridReduce: start with block reduce then write
// results to gmem, the last block load from gmem and finalize with a block
// reduction. Main differences:
// (1) each thread in the iter dim does [vec_size] reductions instead of 1.
// (2) using [blockIterGroupedReduce] instead of [blockReduce].
// (3) ensures vectorized load/store to gmem.
// Specifically, the new para [vec_size] is the vecotrization factor in the
// iteration dimension. It is used in outer reduction to reduce calling this
// grid reduction from [vec_size] times to only 1 time. Its value is limited
// to 1, 2, 4, 8, 16 based on the hardware support and input data type.
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    int vec_size,
    typename T,
    typename Func>
__device__ void iterGroupedGridReduce(
    T* out,
    const T* inp_val,
    Func reduction_op,
    volatile T* work_buf,
    int64_t* sync_flags,
    T* shared_buf,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // inp or block reduction results
  T block_reduction_val[vec_size];

  // Do block reduction when required
  if (X_THREAD || Y_THREAD || Z_THREAD) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      block_reduction_val[i] = init_val;
    }
    blockIterGroupedReduce<X_THREAD, Y_THREAD, Z_THREAD, Aligned, vec_size>(
        block_reduction_val,
        inp_val,
        reduction_op,
        shared_buf,
        read_pred,
        true,
        init_val);
  } else if (read_pred) {
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      block_reduction_val[i] = inp_val[i];
    }
  }

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  if ((!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
            threadIdx, blockDim);

    // index to the right position in [work_buf] to store block reduction
    // results. Consider a typical outer reduction case where iteration dim is
    // TIDx and BIDx and reduction dim is TIDy and BIDy. block_offset = BIDy
    // block_reduction_segment_size = blockDim.x
    // grid_segment_size = gridDim.x
    // idx_in_grid_segment = BIDx
    // thread_offset = TIDx
    auto work_buf_offset =
        block_offset * (block_reduction_segment_size * grid_segment_size) +
        block_reduction_segment_size * idx_in_grid_segment + thread_offset;
    work_buf_offset *= vec_size;

    constexpr int total_bytes = vec_size * sizeof(T);
    constexpr int n_loads = total_bytes <= 16 ? 1 : total_bytes / 16;
    constexpr int n_elements = total_bytes <= 16 ? vec_size : 16 / sizeof(T);
#pragma unroll
    for (int i = 0; i < n_loads; i++) {
      loadLocalToGlobal<T, n_elements, true>(
          &work_buf[work_buf_offset + i * n_elements],
          &block_reduction_val[i * n_elements]);
    }
  }

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);

  } else {
    // there is only one vectorized call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // Cleanup with block reduction
    iterGroupedGridReduceLastBlock<
        !X_THREAD,
        !Y_THREAD,
        !Z_THREAD,
        Aligned,
        vec_size>(
        out,
        (T*)work_buf,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        reduction_op,
        shared_buf,
        write_pred,
        init_val,
        grid_segment_size,
        idx_in_grid_segment);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(

        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

} // namespace reduction

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace grid_broadcast {

// Broadcasts per-thread values across threads and blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - work_buf: Temporary buffer for communication across threads/blocks
// - sync_flags: A vector of integers for synchronizations
//
// Template parameters:
// - X/Y/Z_BLOCK: When true, broadcasts across thread blocks along the X/Y/Z
//   dimensions
// - X/Y/Z_THREAD: When true, broadcasts across threads along the X/Y/Z
//   dimensions
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T>
__device__ void broadcast(
    T& out,
    const T& inp_val,
    volatile T* work_buf,
    Tensor<int64_t, 1> sync_flags,
    bool read_write_pred) {
  // Number of values broadcasted in the grid dimensions
  const auto grid_seg_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the broadcast we're performing out of the grid_seg_size
  const auto grid_seg_idx =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads not participating in a broadcast dimension, this is the
  // number of thread entries to expect in the work buffer, therefore a striding
  const auto block_stride =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  // Which broadcast in the block this is to line up the entry with the work
  // buffer
  const auto thread_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  const bool has_valid_data = (!X_BLOCK || blockIdx.x == gridDim.x - 1) &&
      (!Y_BLOCK || blockIdx.y == gridDim.y - 1) &&
      (!Z_BLOCK || blockIdx.z == gridDim.z - 1) &&
      (!X_THREAD || threadIdx.x == 0) && (!Y_THREAD || threadIdx.y == 0) &&
      (!Z_THREAD || threadIdx.z == 0);

  if (has_valid_data && read_write_pred) {
    work_buf[grid_seg_idx * block_stride + thread_offset] = inp_val;
    __threadfence();
  }

  grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, true, Aligned>(
      sync_flags[grid_seg_idx], grid_seg_size);

  if (read_write_pred) {
    out = work_buf[grid_seg_idx * block_stride + thread_offset];
  }

  // Make sure everyone has read from the buffer before continuing the kernel
  // and potentially overwriting
  grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, true, Aligned>(
      sync_flags[grid_seg_idx], grid_seg_size);
}
} // namespace grid_broadcast

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

namespace broadcast {
// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// Aligned: Called from aligned threads if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, bool Aligned, typename T>
__device__ void blockBroadcast(
    T& out,
    const T& inp_val,
    T* shared_mem,
    bool read_write_pred) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  if (has_valid_data && read_write_pred) {
    shared_mem[shared_offset] = inp_val;
  }

  block_sync::sync<Aligned>();

  if (read_write_pred) {
    out = shared_mem[shared_offset];
  }

  block_sync::sync<Aligned>();
}

} // namespace broadcast

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
template <typename DataType>
struct WelfordTriplet {
  DataType avg;
  DataType var;
  nvfuser_index_t N;
};

template <typename DataType>
__inline__ __device__ void copyTriplet(
    DataType* dst_avg,
    DataType* dst_var,
    nvfuser_index_t* dst_N,
    const WelfordTriplet<DataType>& src) {
  *dst_avg = src.avg;
  *dst_var = src.var;
  *dst_N = src.N;
}

template <typename DataType>
__inline__ __device__ void copyTriplet(
    WelfordTriplet<DataType>& dst,
    const DataType* src_avg,
    const DataType* src_var,
    const nvfuser_index_t* src_N) {
  dst.avg = *src_avg;
  dst.var = *src_var;
  dst.N = *src_N;
}

template <typename DataType>
__inline__ __device__ void copyTriplet(
    WelfordTriplet<DataType>& dst,
    const WelfordTriplet<DataType>& src) {
  dst.avg = src.avg;
  dst.var = src.var;
  dst.N = src.N;
}

// -----------------------------------------------------------------------------------------------
//  Block Welford Primitives
// -----------------------------------------------------------------------------------------------
// Basic utility for welford update. Can be used to scan one value, or two merge
// two welford results
template <typename T, typename TN>
__inline__ __device__ void welfordCombine(
    T& a_avg,
    T& a_M2,
    TN& a_N,
    const T b_avg,
    const T b_M2,
    TN b_N) {
  if (b_N == 0) {
    return;
  }
  TN ab_N = a_N + b_N;
  T b_N_div_ab_N = ((T)(nvfuser_index_t)(b_N)) / ((T)(nvfuser_index_t)(ab_N));
  T delta = b_avg - a_avg;
  a_avg += delta * b_N_div_ab_N;
  a_M2 += b_M2 + delta * delta * ((T)(nvfuser_index_t)(a_N)) * b_N_div_ab_N;
  a_N = ab_N;
}

template <typename T, bool OutputGmem>
__inline__ __device__ void welfordVectorized(
    T& a_avg,
    T& a_M2,
    nvfuser_index_t& a_N,
    const T b_avg,
    const T b_N_div_ab_N,
    const nvfuser_index_t ab_N,
    const bool pred) {
  // Want only predicated statements and don't want to have
  // "if", but for gmem output writes can be illegal, so needs to
  // bail out here.
  if (OutputGmem && !pred) {
    return;
  }
  T predicated_b_avg = pred ? b_avg : a_avg;
  T delta0 = predicated_b_avg - a_avg;
  a_avg += delta0 * b_N_div_ab_N;
  T delta1 = predicated_b_avg - a_avg;
  a_M2 += delta0 * delta1;
  a_N = ab_N;
}

// Non predicated version
template <typename T>
__inline__ __device__ void welfordVectorized(
    T& a_avg,
    T& a_M2,
    nvfuser_index_t& a_N,
    const T b_avg,
    const T b_N_div_ab_N,
    const nvfuser_index_t ab_N) {
  T delta0 = b_avg - a_avg;
  a_avg += delta0 * b_N_div_ab_N;
  T delta1 = b_avg - a_avg;
  a_M2 += delta0 * delta1;
  a_N = ab_N;
}

// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block.
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename TN>
__inline__ __device__ void blockWelford(
    T& out_avg,
    T& out_M2,
    TN& out_N,
    const T& in_avg,
    const T& in_M2,
    const TN& in_N,
    T* shared_mem_avg,
    T* shared_mem_M2,
    TN* shared_mem_N,
    bool read_pred,
    bool write_pred,
    T init_val) {
  // If this thread will output a final result
  bool should_write =
      index_utils::maskedIsZero<X_REDUCE, Y_REDUCE, Z_REDUCE>(threadIdx);

  // Size of the reduction segments
  unsigned int reduction_size =
      index_utils::maskedSize<X_REDUCE, Y_REDUCE, Z_REDUCE>(blockDim);

  // Index into the reduction segment
  unsigned int reduction_tid =
      index_utils::maskedOffset<X_REDUCE, Y_REDUCE, Z_REDUCE>(
          threadIdx, blockDim);

  // Index of the reduction segment
  unsigned int reduction_idx =
      index_utils::maskedOffset<!X_REDUCE, !Y_REDUCE, !Z_REDUCE>(
          threadIdx, blockDim);

  // Offset into smem for the current thread
  unsigned int smem_offset = reduction_idx * reduction_size + reduction_tid;

  if (read_pred) {
    shared_mem_avg[smem_offset] = in_avg;
    shared_mem_M2[smem_offset] = in_M2;
    shared_mem_N[smem_offset] = in_N;
  } else {
    shared_mem_avg[smem_offset] = init_val;
    shared_mem_M2[smem_offset] = init_val;
    shared_mem_N[smem_offset] = 0;
  }

  block_sync::sync<Aligned>();
  // Reduce down to nearest power of 2:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2 && reduction_tid + np2 < reduction_size) {
    welfordCombine(
        shared_mem_avg[smem_offset],
        shared_mem_M2[smem_offset],
        shared_mem_N[smem_offset],
        shared_mem_avg[smem_offset + np2],
        shared_mem_M2[smem_offset + np2],
        shared_mem_N[smem_offset + np2]);
  }
  block_sync::sync<Aligned>();

  // loop peel the final iteration to save one syncthread for the end
  for (int factor = np2 / 2; factor > 1; factor >>= 1) {
    if (reduction_tid < factor) {
      welfordCombine(
          shared_mem_avg[smem_offset],
          shared_mem_M2[smem_offset],
          shared_mem_N[smem_offset],
          shared_mem_avg[smem_offset + factor],
          shared_mem_M2[smem_offset + factor],
          shared_mem_N[smem_offset + factor]);
    }
    block_sync::sync<Aligned>();
  }

  if (should_write && write_pred) {
    T res_avg = out_avg;
    T res_M2 = out_M2;
    TN res_N = out_N;
    welfordCombine(
        res_avg,
        res_M2,
        res_N,
        shared_mem_avg[smem_offset],
        shared_mem_M2[smem_offset],
        shared_mem_N[smem_offset]);
    if (reduction_size > 1) {
      welfordCombine(
          res_avg,
          res_M2,
          res_N,
          shared_mem_avg[smem_offset + 1],
          shared_mem_M2[smem_offset + 1],
          shared_mem_N[smem_offset + 1]);
    }
    out_avg = res_avg;
    out_M2 = res_M2;
    out_N = res_N;
  }
  block_sync::sync<Aligned>();
}

// Use the same pred for both reads and writes
template <
    bool X_REDUCE,
    bool Y_REDUCE,
    bool Z_REDUCE,
    bool Aligned,
    typename T,
    typename TN>
__inline__ __device__ void blockWelford(
    T& out_avg,
    T& out_M2,
    TN& out_N,
    const T& in_avg,
    const T& in_M2,
    const TN& in_N,
    T* shared_mem_avg,
    T* shared_mem_M2,
    TN* shared_mem_N,
    bool read_write_pred,
    T init_val) {
  blockWelford<X_REDUCE, Y_REDUCE, Z_REDUCE, Aligned, T, TN>(
      out_avg,
      out_M2,
      out_N,
      in_avg,
      in_M2,
      in_N,
      shared_mem_avg,
      shared_mem_M2,
      shared_mem_N,
      read_write_pred,
      read_write_pred,
      init_val);
}
// -----------------------------------------------------------------------------------------------
//  Grid Welford Prototype
// -----------------------------------------------------------------------------------------------
namespace welford {

template <
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool Aligned,
    typename T,
    typename TN>
__device__ void gridWelfordLastBlock(
    T& out_avg,
    T& out_M2,
    TN& out_N,
    const volatile T* in_avg,
    const volatile T* in_M2,
    const volatile TN* in_N,
    const nvfuser_index_t
        grid_reduction_segment_size, // Number of reductions across
                                     // grid reduce dimensions
    const nvfuser_index_t
        block_reduction_segment_size, // Number of reductions across the block
    T* shared_buf_avg,
    T* shared_buf_M2,
    TN* shared_buf_N,
    bool write_pred,
    T init_val) {
  // We have to do num_reductions across reduction_size. The reductions are
  // contiguous, but offset by reduction_size. There is an entry in "in" for
  // every block, and every thread marked as true. Threads in dimensions marked
  // as false can be used to parallelize the reduction.

  // Find the reduction id of the participating threads
  const auto block_reduction_segment_idx =
      index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim);

  // Find an id associated within a reduction segment for all
  // "non-participating" threads, which will parallelize the reductions for the
  // "participating" threads
  const auto id_in_block_segment =
      index_utils::maskedOffset<!X_THREAD, !Y_THREAD, !Z_THREAD>(
          threadIdx, blockDim);

  // Stride by the "non-participating" threads
  const auto input_stride_for_thread_in_segment =
      index_utils::maskedSize<!X_THREAD, !Y_THREAD, !Z_THREAD>(blockDim);

  T inp_avg = init_val;
  T inp_M2 = init_val;
  TN inp_N = 0;

  // Block stride across the reduction until we only have one value per thread
  for (nvfuser_index_t reduction_i = id_in_block_segment;
       reduction_i < grid_reduction_segment_size;
       reduction_i += input_stride_for_thread_in_segment) {
    auto work_buf_offset = reduction_i * block_reduction_segment_size +
        block_reduction_segment_idx;
    welfordCombine(
        inp_avg,
        inp_M2,
        inp_N,
        in_avg[work_buf_offset],
        in_M2[work_buf_offset],
        in_N[work_buf_offset]);
  }

  // Block reduce the per thread values into per "participating" thread values
  T inp_avg_tmp = init_val;
  T inp_M2_tmp = init_val;
  TN inp_N_tmp = 0;
  blockWelford<!X_THREAD, !Y_THREAD, !Z_THREAD, Aligned>(
      inp_avg_tmp,
      inp_M2_tmp,
      inp_N_tmp,
      inp_avg,
      inp_M2,
      inp_N,
      shared_buf_avg,
      shared_buf_M2,
      shared_buf_N,
      true,
      init_val);
  const bool should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);
  if (should_write && write_pred) {
    welfordCombine(out_avg, out_M2, out_N, inp_avg_tmp, inp_M2_tmp, inp_N_tmp);
  }
}

// Grid welford combine. See GridReduction for more information
template <
    bool X_BLOCK,
    bool Y_BLOCK,
    bool Z_BLOCK,
    bool X_THREAD,
    bool Y_THREAD,
    bool Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool Aligned,
    typename T,
    typename TN>
__device__ void gridWelford(
    T& out_avg,
    T& out_M2,
    TN& out_N,
    const T& inp_avg,
    const T& inp_M2,
    const TN& inp_N,
    volatile T* work_buf_avg,
    volatile T* work_buf_M2,
    volatile TN* work_buf_N,
    Tensor<int64_t, 1> sync_flags,
    T* shared_buf_avg,
    T* shared_buf_M2,
    TN* shared_buf_N,
    bool read_pred,
    bool write_pred,
    T init_val,
    const nvfuser_index_t entrance_ind,
    const nvfuser_index_t n_entrances) {
  // entrance index only matters for non-persistent re-entrant grid reductions.
  const nvfuser_index_t entrance_ind_ = PERSISTENT_REDUCTION ? 0 : entrance_ind;
  const nvfuser_index_t n_entrances_ = PERSISTENT_REDUCTION ? 1 : n_entrances;

  // Number of values to reduce in the reduction segment
  const auto grid_reduction_segment_size =
      index_utils::maskedSize<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the
  // grid_reduction_segment_size
  const auto idx_in_grid_segment =
      index_utils::maskedOffset<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(
          blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto block_reduction_segment_size =
      index_utils::maskedSize<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  // Number of reductions in the grid
  const nvfuser_index_t grid_segment_size = PERSISTENT_REDUCTION
      ? 1
      : index_utils::maskedSize<!X_BLOCK, !Y_BLOCK, !Z_BLOCK>(gridDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf_avg += (entrance_ind_ * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;
  work_buf_M2 += (entrance_ind_ * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;
  work_buf_N += (entrance_ind_ * grid_segment_size + idx_in_grid_segment) *
      grid_reduction_segment_size * block_reduction_segment_size;

  if ((X_THREAD || threadIdx.x == 0) && (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto block_offset =
        index_utils::maskedOffset<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);
    auto thread_offset =
        index_utils::maskedOffset<X_THREAD, Y_THREAD, Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset =
        block_offset * block_reduction_segment_size + thread_offset;
    if (read_pred) {
      work_buf_avg[work_buf_offset] = inp_avg;
      work_buf_M2[work_buf_offset] = inp_M2;
      work_buf_N[work_buf_offset] = inp_N;
    } else {
      work_buf_avg[work_buf_offset] = init_val;
      work_buf_M2[work_buf_offset] = init_val;
      work_buf_N[work_buf_offset] = 0;
    }
  }

  if (PERSISTENT_REDUCTION) {
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  } else {
    // Use a different sync flag for each call
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[entrance_ind_ * grid_segment_size + idx_in_grid_segment],
        grid_reduction_segment_size);
  }

  bool last_block =
      index_utils::maskedIsLast<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  if (last_block) {
    // final reduction
    gridWelfordLastBlock<X_THREAD, Y_THREAD, Z_THREAD, Aligned>(
        out_avg,
        out_M2,
        out_N,
        work_buf_avg,
        work_buf_M2,
        work_buf_N,
        grid_reduction_segment_size,
        block_reduction_segment_size,
        shared_buf_avg,
        shared_buf_M2,
        shared_buf_N,
        write_pred,
        init_val);
  }

  if (PERSISTENT_REDUCTION) {
    // Make sure we're done with global memory before we allow the kernel to
    // continue
    grid_sync::sync<X_BLOCK, Y_BLOCK, Z_BLOCK, PERSISTENT_REDUCTION, Aligned>(
        sync_flags[idx_in_grid_segment], grid_reduction_segment_size);
  }
}

} // namespace welford

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace warp {

template <typename T>
__device__ __forceinline__ T shfl_xor(T var, int laneMask, int width = 32) {
  return __shfl_xor_sync(0xffffffff, var, laneMask, width);
}
template <typename T>
__device__ __forceinline__ std::complex<T> shfl_xor(
    std::complex<T> var,
    int laneMask,
    int width = 32) {
  T real = __shfl_xor_sync(0xffffffff, var.real(), laneMask, width);
  T imag = __shfl_xor_sync(0xffffffff, var.imag(), laneMask, width);
  return std::complex<T>(real, imag);
}

template <bool SINGLE_WARP, bool Aligned, typename T, typename Func>
__device__ void warpReduceTIDX(
    T& out,
    const T& inp_val,
    Func reduction_op,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  constexpr int WARP_SIZE = 32;

  // Assume input padded to multiples of a warp
  T reduce_val = init_val;

  // Do warp reduction
  if (read_write_pred) {
    reduce_val = inp_val;
  }

  // Reduce within each warp
  for (int i = 16; i >= 1; i /= 2) {
    reduction_op(reduce_val, shfl_xor(reduce_val, i, WARP_SIZE));
  }

  // Reduce across warp if needed
  // Load value to shared mem
  if (!SINGLE_WARP) {
    unsigned int warp_idx = threadIdx.x / WARP_SIZE;
    unsigned int lane_idx = threadIdx.x % WARP_SIZE;
    unsigned int reduce_group_id = threadIdx.z * blockDim.y + threadIdx.y;
    bool is_warp_head = lane_idx == 0;
    unsigned int reduction_size = blockDim.x;
    unsigned int num_of_warps = reduction_size / WARP_SIZE;
    unsigned int smem_offset = reduce_group_id * num_of_warps;

    block_sync::sync<Aligned>();

    if (is_warp_head) {
      shared_mem[smem_offset + warp_idx] = reduce_val;
    }

    block_sync::sync<Aligned>();

    if (warp_idx == 0) {
      // This assumes num_of_warps will be < 32, meaning < 1024 threads.
      //  Should be true for long enough.
      assert(num_of_warps <= 32);

      reduce_val = lane_idx < num_of_warps ? shared_mem[smem_offset + lane_idx]
                                           : init_val;

      // Reduce within warp 0
      for (int i = 16; i >= 1; i /= 2) {
        reduction_op(reduce_val, shfl_xor(reduce_val, i, 32));
      }
    }

    if (is_warp_head) {
      reduction_op(out, reduce_val);
    }
    // needs sync, otherwise other warps may access shared memory before this
    // reduction is done.
    block_sync::sync<Aligned>();
  } else {
    reduction_op(out, reduce_val);
  }
}

} // namespace warp

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

// Utility for converting generic pointer to SMEM pointer in PTX.
//  We should review vectorized load/stores with shared memory.
//  SMEM memory movement PTX is only Global -> SMEM, SMEM -> Local, Local ->
//  SMEM, and this is needed for these PTX instructions to provide the SMEM
//  pointer.
__device__ inline unsigned toSmem(const void* raw_ptr) {
  unsigned smem_ptr_uint;
  asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
      : "=r"(smem_ptr_uint)
      : "l"(raw_ptr));

  return smem_ptr_uint;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))

namespace Turing {

// LdMatrix has .x1, .x2 and .x4 options, currently we actively use .x2 and
//  .x4. In .x2 option. the the address register of upper half warp (lane 16-31)
//  are un-used but on Turing [sm75,sm80) architecture these un-used addresses
//  need to be valid, in the sense that:
//     1. The data it points to has to be within allocated shared mem buffer.
//     2. The address needs to be aligned to 16 byte.
//  See also:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
//  This function addresses 2. above by masking out the sub-16B component
//    of the address in upper warp and 1. is guaranteed by ldmatrix swizzle
//    util.
//  This will **not** affect any functionality. This is just modification
//    of unused pointers to satisfy the alignment requirement on Turing
//    hardware.
//  The alignment requirement is lifted on sm80+,
//    so this function is a no-op on Ampere or above.
template <unsigned num_valid_addresses>
__device__ inline unsigned adjustPartialLdMatrixAddrInTuring(
    unsigned addr_in_byte) {
  const unsigned lane = threadIdx.x % 32;
  if (lane >= num_valid_addresses) {
    return 0;
  }
  return addr_in_byte;
}

} // namespace Turing

#endif // Arch 75

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

namespace Hopper {

// References:
//
// TMA:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp
//
// Tensor map:
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

// TMA Loads:

template <int dim>
struct CpAsyncBulkTensorTileG2SIndex {
  const TensorMap* descriptor;
  Array<int32_t, dim> crds;
  uint32_t mbarrier;
};

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<1>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3}], [%2];"
      :
      : "r"(smem_addr), "l"(gmem_int_desc), "r"(src.mbarrier), "r"(src.crds[0])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<2>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<3>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<4>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileG2S(
    const CpAsyncBulkTensorTileG2SIndex<5>& src,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(src.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(smem_addr),
        "l"(gmem_int_desc),
        "r"(src.mbarrier),
        "r"(src.crds[0]),
        "r"(src.crds[1]),
        "r"(src.crds[2]),
        "r"(src.crds[3]),
        "r"(src.crds[4])
      : "memory");
}

// TMA Stores:

template <int dim>
struct CpAsyncBulkTensorTileS2GIndex {
  const TensorMap* descriptor;
  Array<int32_t, dim> crds;
};

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<1>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%2}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_addr), "r"(dest.crds[0])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<2>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(gmem_int_desc), "r"(smem_addr), "r"(dest.crds[0]), "r"(dest.crds[1])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<3>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%2, %3, %4}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<4>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2]),
        "r"(dest.crds[3])
      : "memory");
}

__device__ inline void cpAsyncBulkTensorTileS2G(
    const CpAsyncBulkTensorTileS2GIndex<5>& dest,
    uint32_t smem_addr) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(dest.descriptor);
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];"
      :
      : "l"(gmem_int_desc),
        "r"(smem_addr),
        "r"(dest.crds[0]),
        "r"(dest.crds[1]),
        "r"(dest.crds[2]),
        "r"(dest.crds[3]),
        "r"(dest.crds[4])
      : "memory");
}

} // namespace Hopper

#endif // Arch 90

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace fused_reduction {

// Tuple of Welford avg, var and N parameters.
//
// Template parameters:
// - DataTypeT: Type of avg and var
// - IndexTypeT: Type of N
// - MakeTuple: Template template parameter to define Tuple types
// (e.g., MakeLocalTuple>
template <
    int NumVals,
    typename DataTypeT,
    typename IndexTypeT,
    template <int, typename>
    typename MakeTuple>
struct WelfordTripletTuple {
  static constexpr int num_vals = NumVals;
  using DataType = DataTypeT;
  using IndexType = IndexTypeT;
  using DataTuple = typename MakeTuple<NumVals, DataType>::type;
  using IndexTuple = typename MakeTuple<NumVals, IndexType>::type;

  DataTuple avg;
  DataTuple var;
  IndexTuple N;

  WelfordTripletTuple(
      const DataTuple& avg,
      const DataTuple& var,
      const IndexTuple& N)
      : avg(avg), var(var), N(N) {}
};

template <int NumVals, typename DataType, typename IndexType>
using LocalWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeLocalTuple>;

template <int NumVals, typename DataType, typename IndexType>
using RefWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeRefTuple>;

template <int NumVals, typename DataType, typename IndexType>
using ConstRefWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataType, IndexType, MakeConstRefTuple>;

template <int NumVals, typename DataTypeT, typename IndexTypeT>
using VolatilePtrWelfordTripletTuple =
    WelfordTripletTuple<NumVals, DataTypeT, IndexTypeT, MakeVolatilePtrTuple>;

// Advance pointer offsets of WelfordTripleTuple. Only valid when the
// values are pointer values.
template <typename WelfordTripletTupleType>
__inline__ __device__ static void operator+=(
    WelfordTripletTupleType& triplet,
    nvfuser_index_t offset) {
  triplet.avg += offset;
  triplet.var += offset;
  triplet.N += offset;
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType>
__inline__ __device__ static void copyWelfordTripletTuple(
    DstType& dst,
    nvfuser_index_t dst_offset,
    const SrcType& src,
    nvfuser_index_t src_offset = 0) {
  copyTuple(dst.avg, dst_offset, src.avg, src_offset);
  copyTuple(dst.var, dst_offset, src.var, src_offset);
  copyTuple(dst.N, dst_offset, src.N, src_offset);
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType>
__inline__ __device__ static void copyWelfordTripletTuple(
    DstType& dst,
    const SrcType& src,
    nvfuser_index_t src_offset = 0) {
  copyWelfordTripletTuple(dst, 0, src, src_offset);
}

// Copy each of the triplet tuples
template <typename DstType, typename SrcType, typename PredType>
__inline__ __device__ static void copyWelfordTripletTupleIf(
    DstType& dst,
    const SrcType& src,
    const PredType& pred) {
  copyTupleIf(dst.avg, src.avg, pred);
  copyTupleIf(dst.var, src.var, pred);
  copyTupleIf(dst.N, src.N, pred);
}

} // namespace fused_reduction

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace fused_reduction {

namespace impl {

//! Suppose f_i be the i-th function of the binary function
//! parameters. Call the function as: f_i(x, y)
template <int i, typename DataType, typename Func, typename... Funcs>
struct FuncSelector {
  static __device__ void call(
      DataType& x,
      const DataType y,
      Func f,
      Funcs... funcs) {
    // Here, i is guaranteed to be larger than 0 as there's a
    // specialization for i == 0 below. Recursively call FuncSelector
    // by dropping f and decrementing i.
    FuncSelector<i - 1, DataType, Funcs...>::call(x, y, funcs...);
  }
};

//! Specialization of FuncSelector when i == 0, so f_i is f.
template <typename DataType, typename Func, typename... Funcs>
struct FuncSelector<0, DataType, Func, Funcs...> {
  static __device__ void call(
      DataType& x,
      const DataType y,
      Func f,
      Funcs... funcs) {
    f(x, y);
  }
};

//! Call each of the first i+1 functions with the first i+1 values of
//! tuples. Here, i is guaranteed to be larger than -1 as there's a
//! specialization for i == -1.
template <int i, typename TupleType0, typename TupleType1, typename... Funcs>
struct FuncForEach {
  static __device__ void call(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Funcs... funcs) {
    static_assert(
        IsSameType<
            typename TupleType0::template ValType<i>,
            typename TupleType1::template ValType<i>>::value,
        "Invalid tuple types");
    // Process the first i functions first.
    FuncForEach<i - 1, TupleType0, TupleType1, Funcs...>::call(
        val0, offset0, val1, offset1, funcs...);
    // Call the i+1-th function
    FuncSelector<i, typename TupleType0::template ValType<i>, Funcs...>::call(
        val0.val<i>(offset0), val1.val<i>(offset1), funcs...);
  }
};

//! Specialization of FuncForEach when i == -1, which means no
//! function to call. Just for stopping the recursive pattern here.
template <typename TupleType0, typename TupleType1, typename... Funcs>
struct FuncForEach<-1, TupleType0, TupleType1, Funcs...> {
  static __device__ void call(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Funcs... funcs) {}
};

//! Reduce one value of a tuple using one of the reduction ops. The
//! value at val_idx is reduced by the function at func_idx.
template <
    int func_idx,
    int val_idx,
    typename TupleType0,
    typename TupleType1,
    typename... Funcs>
__inline__ __device__ static void reduceVal(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Funcs... reduction_ops) {
  static_assert(
      IsSameType<
          typename TupleType0::template ValType<val_idx>,
          typename TupleType1::template ValType<val_idx>>::value,
      "Invalid tuple types");
  FuncSelector<
      func_idx,
      typename TupleType0::template ValType<val_idx>,
      Funcs...>::
      call(
          val0.val<val_idx>(offset0),
          val1.val<val_idx>(offset1),
          reduction_ops...);
}

//! Accumulate each value of a given pair of tuples using its corresponding
//! function. Suppose f_i be the i-th reduciton function. Call f_i as:
//! f_i(val0.val<i>(offset0), val1.val<i>(offset1)).
template <typename TupleType0, typename TupleType1, typename... Funcs>
__inline__ __device__ static void reduceEach(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Funcs... reduction_ops) {
  constexpr int num_funcs = sizeof...(reduction_ops);
  FuncForEach<num_funcs - 1, TupleType0, TupleType1, Funcs...>::call(
      val0, offset0, val1, offset1, reduction_ops...);
}

template <typename TupleType0, typename TupleType1, typename Func, int num_vals>
struct TupleReduce {};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 1> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(val0.val<0>(offset0), val1.val<0>(offset1));
  }
};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 2> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(
        val0.val<0>(offset0),
        val0.val<1>(offset0),
        val1.val<0>(offset1),
        val1.val<1>(offset1));
  }
};

template <typename TupleType0, typename TupleType1, typename Func>
struct TupleReduce<TupleType0, TupleType1, Func, 3> {
  __inline__ __device__ static void reduce(
      TupleType0& val0,
      nvfuser_index_t offset0,
      const TupleType1& val1,
      nvfuser_index_t offset1,
      Func reduction_op) {
    static_assert(
        IsSameType<
            typename TupleType0::ValTypes,
            typename TupleType1::ValTypes>::value,
        "Invalid value types");
    reduction_op(
        val0.val<0>(offset0),
        val0.val<1>(offset0),
        val0.val<2>(offset0),
        val1.val<0>(offset1),
        val1.val<1>(offset1),
        val1.val<2>(offset1));
  }
};

//! Reduce all values of a tuple together. The reduction function must
//! have the same number of inputs as the number of values of each tuple.
template <typename TupleType0, typename TupleType1, typename Func>
__inline__ __device__ void reduceTuple(
    TupleType0& val0,
    nvfuser_index_t offset0,
    const TupleType1& val1,
    nvfuser_index_t offset1,
    Func reduction_op) {
  static_assert(
      TupleType0::num_vals == TupleType1::num_vals, "Invalid number of values");
  TupleReduce<TupleType0, TupleType1, Func, TupleType0::num_vals>::reduce(
      val0, offset0, val1, offset1, reduction_op);
}

// Reduces all of the first (idx+1) values by a thread block
template <
    int idx,
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    typename LocalTupleT,
    typename... Funcs>
struct BlockReduceEach {
  __inline__ __device__ static void reduce(
      LocalTupleT& block_result,
      const LocalTupleT& partial_result,
      void* shared_mem,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx,
      Funcs... funcs) {
    // Finish the reduction of each tuple value with a smaller offset
    BlockReduceEach<idx - 1, BROADCAST, true, Aligned, LocalTupleT, Funcs...>::
        reduce(
            block_result,
            partial_result,
            shared_mem,
            has_block_result,
            tid_in_reduction,
            num_threads_per_reduction,
            num_elements_per_reduction,
            reduction_idx,
            funcs...);

    if (num_elements_per_reduction == 1) {
      if (has_block_result) {
        block_result.val<idx>(0) = partial_result.val<idx>(0);
      }
      return;
    }

    using DataType = typename LocalTupleT::template ValType<idx>;

    PtrTuple<DataType> shared_buf(static_cast<DataType*>(shared_mem));

    LocalTuple<DataType> block_result_i(partial_result.val<idx>(0));

    const auto smem_offset =
        reduction_idx * num_threads_per_reduction + tid_in_reduction;

    const int np2 = 1 << (31 - __clz(num_elements_per_reduction));

    // Threads values are initialized, so all can participate here
    if (tid_in_reduction >= np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    block_sync::sync<Aligned>();

    if (tid_in_reduction < np2 &&
        tid_in_reduction + np2 < num_elements_per_reduction) {
      impl::reduceVal<idx, 0>(
          block_result_i, 0, shared_buf, smem_offset + np2, funcs...);
    }

    if (tid_in_reduction < np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    // Always sync when communicating across smem
    block_sync::sync<Aligned>();

    // Reduce down to 2 values, last thread will do the final reduction and
    // can save a syncthreads this way
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_reduction < factor) {
        impl::reduceVal<idx, 0>(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            funcs...);
      }
      block_sync::sync<Aligned>();
    }

    copyTuple(block_result_i, shared_buf, smem_offset);

    // Do the last reduction
    if (has_block_result) {
      impl::reduceVal<idx, 0>(
          block_result_i, 0, shared_buf, smem_offset + 1, funcs...);
    }

    if (BROADCAST) {
      if (has_block_result) {
        // Put result back in shared memory, put in the first entry of the
        // reduction segment's buffer
        copyTuple(
            shared_buf,
            reduction_idx * num_threads_per_reduction,
            block_result_i);
      }

      // Sync threads to make sure result is in smem
      block_sync::sync<Aligned>();

      copyTuple(
          block_result_i,
          shared_buf,
          reduction_idx * num_threads_per_reduction);
    }

    block_result.val<idx>(0) = block_result_i.val<0>(0);

    if (FORWARD_PROTECT_SMEM) {
      block_sync::sync<Aligned>();
    }
  }
};

// Specialization for idx == -1, i.e., no value to reduce.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    typename LocalTupleT,
    typename... Funcs>
struct BlockReduceEach<
    -1,
    BROADCAST,
    FORWARD_PROTECT_SMEM,
    Aligned,
    LocalTupleT,
    Funcs...> {
  __inline__ __device__ static void reduce(
      LocalTupleT& block_result,
      const LocalTupleT& partial_result,
      void* shared_mem,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx,
      Funcs... funcs) {}
};

//! Reduce each value of a tuple by a thread block.
//!
//! The final result is broadcast when BROADCAST is true.
//!
//! \param block_result result of the block reduction
//! \param partial_result Per-thread input tuple
//! \param shared_mem
//! \param has_block_result
//! \param tid_in_reduction
//! \param num_threads_per_reduction
//! \param num_elements_per_reduction
//! \param reduction_idx
//! \param reduction_ops
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    typename LocalTupleT,
    typename... Funcs>
__inline__ __device__ void blockReduceEach(
    LocalTupleT& block_result,
    const LocalTupleT& partial_result,
    void* shared_mem,
    bool has_block_result,
    int tid_in_reduction,
    int num_threads_per_reduction,
    int num_elements_per_reduction,
    int reduction_idx,
    Funcs... reduction_ops) {
  BlockReduceEach<
      LocalTupleT::num_vals - 1,
      BROADCAST,
      FORWARD_PROTECT_SMEM,
      Aligned,
      LocalTupleT,
      Funcs...>::
      reduce(
          block_result,
          partial_result,
          shared_mem,
          has_block_result,
          tid_in_reduction,
          num_threads_per_reduction,
          num_elements_per_reduction,
          reduction_idx,
          reduction_ops...);
}

} // namespace impl

// We have 6 dimensions, 3 in the grid, 3 in the block
// They can be 1 of 3 states,
// Reduction Domain - TEMPLATE STATE 0
//   - Participating in the reduction, has values coming in, one value coming
//     out across the dimension
// Iteration Domain - TEMPLATE STATE 1
//   - Not participating in the reduction, has values across the dimension after
//     the reduction
// Collapsed Domain - TEMPLATE STATE 2
//   - Previously reduced, doesn't need to be reduced on that dimension, doesn't
//     have values across that dimension
constexpr __device__ bool isReduce(int STATE) {
  return STATE == 0;
}

constexpr __device__ bool isIter(int STATE) {
  return STATE == 1;
}

constexpr __device__ bool isPred(int STATE) {
  return STATE == 2;
}

constexpr __device__ bool inactive(int STATE) {
  return STATE == 3;
}

constexpr __device__ bool activeNotIter(int STATE) {
  return STATE != 3 && STATE != 1;
}

constexpr __device__ bool isReduceOrIter(int STATE) {
  return isReduce(STATE) || isIter(STATE);
}

// When generating an index into the reduction, we have to stride by iteration
// domains and reduction domains. Collapsed domains we can ignore, but we need
// to make sure they never read or write (need to be predicated to correct
// participation).

// All inclusive reduction with option to re-broadcast. This reduction class
// does not use predication of parallelization in the read or write predicates.
// Instead there are 3 states each dimension of parallelization can have,
// described above. Predication, indexing, and reduction will be done based on
// this information.
template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
class ParallelReduce {
  static_assert(
      !BROADCAST || PERSISTENT_REDUCTION,
      "Broadcast requires persistent reduction");

  static constexpr bool BLOCK_REDUCE =
      isReduce(X_THREAD) || isReduce(Y_THREAD) || isReduce(Z_THREAD);

  static constexpr bool GRID_REDUCE =
      isReduce(X_BLOCK) || isReduce(Y_BLOCK) || isReduce(Z_BLOCK);

  // ping-pong between global buffers to avoid a second sync
  bool flip = false;

 public:
  __device__ ParallelReduce() {}

  // reduceGroup does not support Welford-style reductions that reduce
  // all values of a tuple together, so this is the only entry point
  // for Welford for now.
  template <bool Aligned, typename Func, typename... Types>
  __device__ __inline__ void reduce(
      RefTuple<Types...> out,
      const ConstRefTuple<Types...>& inp,
      VolatilePtrTuple<Types...> global_work_buffer,
      int64_t* global_sync_buffer, // Allocated as product of all
                                   // non-participating Grid dimension
      PtrTuple<Types...> shared_buf,
      bool read_pred, // Prevent reading from out of bounds memory
      bool write_pred, // Prevent from writing out of bounds
      const LocalTuple<Types...>& init_val,
      Func reduction_op);

  //! Profiled version
  template <bool Aligned, typename Func, typename... Types>
  __device__ __inline__ void reduce(
      RefTuple<Types...> out,
      const ConstRefTuple<Types...>& inp,
      VolatilePtrTuple<Types...> global_work_buffer,
      int64_t* global_sync_buffer, // Allocated as product of all
                                   // non-participating Grid dimension
      PtrTuple<Types...> shared_buf,
      bool read_pred, // Prevent reading from out of bounds memory
      bool write_pred, // Prevent from writing out of bounds

      const LocalTuple<Types...>& init_val,
      Func reduction_op,
      int64_t& cycles,
      int64_t& count);

  //! Each value of a tuple is independently reduced by the
  //! corresponding reduction op. Thus, Welford-like reductions are
  //! not supported by this interface.
  //!
  //! Note that out is purely used as the output parameter, and its
  //! initial value is not used but just overwritten. Since grid
  //! reductions do not allow serial reduction IterDomains, there is
  //! no need to accumulate into the out parameter.
  template <
      bool Aligned,
      typename... DataTypes,
      typename... Funcs,
      typename... BoolTypes>
  __device__ __inline__ void reduceGroup(
      RefTuple<DataTypes...> out,
      const ConstRefTuple<DataTypes...>& inp,
      VolatilePtrTuple<DataTypes...> global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      int64_t* global_sync_buffer,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      const LocalTuple<BoolTypes...>& write_preds,
      Funcs... funcs);

  //! Profiled version
  template <
      bool Aligned,
      typename... DataTypes,
      typename... Funcs,
      typename... BoolTypes>
  __device__ __inline__ void reduceGroup(
      RefTuple<DataTypes...> out,
      const ConstRefTuple<DataTypes...>& inp,
      VolatilePtrTuple<DataTypes...> global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      int64_t* global_sync_buffer,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      const LocalTuple<BoolTypes...>& write_preds,
      int64_t& cycles,
      int64_t& count,
      Funcs... funcs);

  // User-visible entry point of grouped grid welford +
  // broadcast. Mostly the same as reduceGroup, and it would be
  // possible to combine this to reduceGroup, but it might make the
  // templated data structures even more complicated and difficult to
  // understand. For now, keep it as a separate function.
  //
  // Unlike reduceGroup, though, the data types of welford ops must be
  // the same. For example, reduceGroup can be used to reduce half and
  // float values by passing a tuple of, e.g., LocalTuple<half,
  // float>, but that's not supported here for implementation
  // simplicity. In practice, it should be really uncommon to group
  // welford ops with different data types, so this restriction
  // shouldn't be an issue.
  template <bool Aligned, int NumArgs, typename DataType, typename IndexType>
  __device__ __inline__ void welfordGroup(
      typename MakeRefTuple<NumArgs, DataType>::type out_avg,
      typename MakeRefTuple<NumArgs, DataType>::type out_var,
      typename MakeRefTuple<NumArgs, IndexType>::type out_N,
      const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_avg,
      const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_var,
      const typename MakeConstRefTuple<NumArgs, IndexType>::type& inp_N,
      const typename MakeLocalTuple<NumArgs, DataType>::type& init_avg,
      const typename MakeLocalTuple<NumArgs, DataType>::type& init_var,
      const typename MakeLocalTuple<NumArgs, IndexType>::type& init_N,
      typename MakeVolatilePtrTuple<NumArgs, DataType>::type
          global_work_buffer_avg,
      typename MakeVolatilePtrTuple<NumArgs, DataType>::type
          global_work_buffer_var,
      typename MakeVolatilePtrTuple<NumArgs, IndexType>::type
          global_work_buffer_N,
      int64_t* global_sync_buffer,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      const typename MakeLocalTuple<NumArgs, bool>::type& read_preds,
      const typename MakeLocalTuple<NumArgs, bool>::type& write_preds);

  //! Profiled version
  template <bool Aligned, int NumArgs, typename DataType, typename IndexType>
  __device__ __inline__ void welfordGroup(
      typename MakeRefTuple<NumArgs, DataType>::type out_avg,
      typename MakeRefTuple<NumArgs, DataType>::type out_var,
      typename MakeRefTuple<NumArgs, IndexType>::type out_N,
      const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_avg,
      const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_var,
      const typename MakeConstRefTuple<NumArgs, IndexType>::type& inp_N,
      const typename MakeLocalTuple<NumArgs, DataType>::type& init_avg,
      const typename MakeLocalTuple<NumArgs, DataType>::type& init_var,
      const typename MakeLocalTuple<NumArgs, IndexType>::type& init_N,
      typename MakeVolatilePtrTuple<NumArgs, DataType>::type
          global_work_buffer_avg,
      typename MakeVolatilePtrTuple<NumArgs, DataType>::type
          global_work_buffer_var,
      typename MakeVolatilePtrTuple<NumArgs, IndexType>::type
          global_work_buffer_N,
      int64_t* global_sync_buffer,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      const typename MakeLocalTuple<NumArgs, bool>::type& read_preds,
      const typename MakeLocalTuple<NumArgs, bool>::type& write_preds,
      int64_t& cycles,
      int64_t& count);

  // This is highly specific to the outer-reduction pattern. All the
  // assumptions should be asserted with static_assert at the begging of
  // the fuction.
  template <bool Aligned, int NumVals, typename DataType, int BDIMX, int BDIMY>
  __device__ __inline__ void welfordGroupOuter(
      DataType out_avg[NumVals],
      DataType out_var[NumVals],
      nvfuser_index_t out_N[NumVals],
      const DataType in_avg[NumVals],
      const DataType in_var[NumVals],
      nvfuser_index_t in_N,
      DataType* global_buf_avg,
      DataType* global_buf_var,
      nvfuser_index_t* global_buf_N,
      DataType* shared_buf,
      int64_t* global_sync_buffer);

  // Profiled version
  template <bool Aligned, int NumVals, typename DataType, int BDIMX, int BDIMY>
  __device__ __inline__ void welfordGroupOuter(
      DataType out_avg[NumVals],
      DataType out_var[NumVals],
      nvfuser_index_t out_N[NumVals],
      const DataType in_avg[NumVals],
      const DataType in_var[NumVals],
      nvfuser_index_t in_N,
      DataType* global_buf_avg,
      DataType* global_buf_var,
      nvfuser_index_t* global_buf_N,
      DataType* shared_buf,
      int64_t* global_sync_buffer,
      int64_t& cycles,
      int64_t& count);

 private:
  __device__ static bool isLastBlockInGrid() {
    return index_utils::maskedIsLast<
               isReduceOrIter(X_BLOCK),
               isReduceOrIter(Y_BLOCK),
               isReduceOrIter(Z_BLOCK)>(blockIdx, gridDim) &&
        index_utils::maskedIsZero<
               !isReduceOrIter(X_BLOCK),
               !isReduceOrIter(Y_BLOCK),
               !isReduceOrIter(Z_BLOCK)>(blockIdx);
  }

  //! Initial per-CTA reduction of each value of a tuple. Each value
  //! is reduced individually, so the shared memory buffer just needs
  //! to be large enough for each value. NOTE that the smem buffer is
  //! not forward protected.
  template <
      bool BLOCK_BROADCAST,
      bool Aligned,
      typename... DataTypes,
      typename... Funcs,
      typename... BoolTypes>
  __device__ __inline__ static LocalTuple<DataTypes...> reduceGroupBlock(
      const ConstRefTuple<DataTypes...>& inp,
      const LocalTuple<DataTypes...>& init_val,
      void* shared_mem,
      const LocalTuple<BoolTypes...>& read_preds,
      bool block_reduce_participate,
      Funcs... funcs);

  //! Final reduction of partial results. Done by all blocks
  //! redundantly when BROADCAST is true, or just one block otherwise.
  //! The smem buffer is assumed synchronized when it is passed in,
  //! but it isn't synchronized when returning from this function.
  template <
      bool Aligned,
      typename... DataTypes,
      typename... Funcs,
      typename... BoolTypes>
  __device__ __inline__ static void reduceGroupLastBlock(
      RefTuple<DataTypes...>& out,
      const VolatilePtrTuple<DataTypes...>& global_work_buffer,
      const LocalTuple<DataTypes...>& init_val,
      void* shared_mem,
      nvfuser_index_t block_red_idx_offset,
      nvfuser_index_t num_thread_iters,
      nvfuser_index_t num_block_iters,
      nvfuser_index_t thread_red_idx_offset,
      nvfuser_index_t grid_red_size,
      const LocalTuple<BoolTypes...>& write_preds,
      bool block_reduce_participate,
      bool grid_reduce_participate,
      Funcs... reduction_ops);

  //! Welford version of reduceGroupBlock
  template <
      bool BLOCK_BROADCAST,
      bool Aligned,
      int NumVals,
      typename DataType,
      typename IndexType>
  __device__ __inline__ static void welfordGroupBlock(
      LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
      const ConstRefWelfordTripletTuple<NumVals, DataType, IndexType>& inp,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      const typename MakeLocalTuple<NumVals, bool>::type& read_preds,
      bool block_reduce_participate);

  //! Welford version of reduceGrouplLastBlock
  template <bool Aligned, int NumVals, typename DataType, typename IndexType>
  __device__ __inline__ static void welfordGroupLastBlock(
      RefWelfordTripletTuple<NumVals, DataType, IndexType>& out,
      const VolatilePtrWelfordTripletTuple<NumVals, DataType, IndexType>&
          global_work_buffer,
      const LocalWelfordTripletTuple<NumVals, DataType, IndexType>& init_val,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      nvfuser_index_t block_red_idx_offset,
      nvfuser_index_t num_thread_iters,
      nvfuser_index_t num_block_iters,
      nvfuser_index_t thread_red_idx_offset,
      nvfuser_index_t grid_red_size,
      const typename MakeLocalTuple<NumVals, bool>::type& write_preds,
      bool block_reduce_participate,
      bool grid_reduce_participate);

  // End Parallel reduce class
};

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, typename Func, typename... Types>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduce(
        RefTuple<Types...> out,
        const ConstRefTuple<Types...>& inp,
        VolatilePtrTuple<Types...> global_work_buffer,
        int64_t* global_sync_buffer, // Allocated as product of all
        // non-participating Grid dimension
        PtrTuple<Types...> shared_buf,
        bool read_pred, // Prevent reading from out of bounds memory
        bool write_pred, // Prevent from writing out of bounds
        const LocalTuple<Types...>& init_val,
        Func reduction_op) {
  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    if (read_pred && write_pred) {
      out = inp;
    }
    return;
  }

  // Don't read/write in temporary buffers if in a predicated dimension
  bool block_reduce_participate = index_utils::
      maskedIsZero<isPred(X_THREAD), isPred(Y_THREAD), isPred(Z_THREAD)>(
          threadIdx);

  // Initialize block result
  LocalTuple<Types...> block_result = init_val;

  // Grab input data if participating in the reduction, set to block_result in
  // the case there is no block reduction
  if (block_reduce_participate && read_pred) {
    block_result = inp;
  }

  // Only threads that with id == 0 in the dimensions being reduced will
  // have a valid result
  bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  if (BLOCK_REDUCE) {
    // -- START BLOCK REDUCTION -- //

    // Size of the block reduction segment, can be an int since it's limited
    // to number of threads
    int block_reduction_size = index_utils::
        maskedSize<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
            blockDim);

    // Index in the reduction segment, can be an int since it's limited to
    // number of threads
    int tid_in_block_reduction = index_utils::maskedOffset<
        isReduce(X_THREAD),
        isReduce(Y_THREAD),
        isReduce(Z_THREAD)>(threadIdx, blockDim);

    // ID of the block reduction this thread is participating in
    //
    // If any of the parallel dimensions are predicated out, that means
    // they've already been reduced, so we only care about the first thread in
    // that dimension. Therefore don't expand the reduction_idx by that
    // dimension
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    // Shared memory buffer is 2D
    // [iter dimension, reduction dimension]

    // Offset into smem for the current thread
    int block_reduce_smem_offset =
        block_reduction_idx * block_reduction_size + tid_in_block_reduction;

    // Initialize shared memory
    if (block_reduce_participate) {
      copyTuple(shared_buf, block_reduce_smem_offset, block_result);
    }

    // Sync to make sure smem is completely initialized
    block_sync::sync<Aligned>();

    // Round reduction size down to nearest power of 2
    int np2 = 1 << (31 - __clz(block_reduction_size));

    // Perform an initial reduction leaving np2 elements
    if (block_reduce_participate && tid_in_block_reduction < np2 &&
        tid_in_block_reduction + np2 < block_reduction_size) {
      impl::reduceTuple(
          shared_buf,
          block_reduce_smem_offset,
          shared_buf,
          block_reduce_smem_offset + np2,
          reduction_op);
    }

    // Always need to sync while operating on shared memory
    block_sync::sync<Aligned>();

    // Reduce down until 2 values, leaving 2 values allows us to manually
    // perform the last reduction and avoid a syncthreads
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_block_reduction < factor && block_reduce_participate) {
        impl::reduceTuple(
            shared_buf,
            block_reduce_smem_offset,
            shared_buf,
            block_reduce_smem_offset + factor,
            reduction_op);
      }
      block_sync::sync<Aligned>();
    }

    // Accumulate that last valid result
    if (has_block_result) {
      copyTuple(block_result, shared_buf, block_reduce_smem_offset);
      if (block_reduction_size > 1) {
        impl::reduceTuple(
            block_result,
            0,
            shared_buf,
            block_reduce_smem_offset + 1,
            reduction_op);
      }
    }

    // ===== BLOCK REDUCTION CLEANUP =======
    if (!GRID_REDUCE) {
      // If no grid reduction, we don't have to continue. Either broadcast
      // back across the block or return the correct reduction
      if (has_block_result && write_pred) {
        impl::reduceTuple(block_result, 0, out, 0, reduction_op);
        out = block_result;
      }
      if (BROADCAST) {
        // No grid reduce, but need to broadcast, perform block broadcast
        if (has_block_result && write_pred) {
          // Put result back in shared memory, put in the first entry of the
          // reduction segment's buffer
          copyTuple(
              shared_buf,
              block_reduction_idx * block_reduction_size,
              block_result);
        }

        // Sync threads to make sure result is in smem
        block_sync::sync<Aligned>();
        // If the thread is participating, and is not attempting to write out
        // of bounds, return the broadcasted value.
        if (block_reduce_participate && write_pred) {
          copyTuple(
              out, shared_buf, block_reduction_idx * block_reduction_size);

        }
      }

      // Forward protect shared memory, don't want threads to continue to
      // another reduction/broadcast and pollute shared memory before the
      // reduction is completely finished.
      //
      // This could be avoided in some cases if we added thread syncs from
      // block reductions in the syncthread insertion pass.
      block_sync::sync<Aligned>();
      return;
    }
  }

  // -- START GRID REDUCTION -- //
  // Grid reductions are more challenging for two reasons, (1) the reduction
  // itself is 3D instead of 2D because we now have an iter domain space in
  // the grid dimension. (2) a tree reduction isn't performed, instead all
  // blocks will populate GMEM and one  block will finish the grid reduction.

  // What is the grid reduction size, block reduction already performed so
  // that doesn't have to be taken into consideration
  const auto grid_red_size = index_utils::
      maskedSize<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          gridDim);

  // Which ID in the reduction is this block. Threads can participate in
  // multiple grid reductions, but the block will have the same relative index
  // in those reductions
  const auto idx_in_grid_red = index_utils::
      maskedOffset<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if (PERSISTENT_REDUCTION && flip) {
    auto global_buffer_size =
        index_utils::
            maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
                gridDim) *
        index_utils::
            maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
                blockDim) *
        grid_red_size;
    global_work_buffer += global_buffer_size;
  }
  flip = !flip;

  // How many grid reductions have to be performed, in the grid dimension
  const auto num_block_iters = index_utils::
      maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(gridDim);

  // Which grid reduction does this block participate in, in the grid
  // dimension
  const auto block_red_idx_offset = index_utils::
      maskedOffset<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the block dimension
  const auto num_thread_iters = index_utils::
      maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          blockDim);

  // Which grid reduction does this thread participate in, in the block
  // dimension
  const auto thread_red_idx_offset = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // 3D buffer of reductions:
  //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
  // Offset into the work buffer
  const auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *
          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

  if (grid_reduce_participate && block_reduce_participate) {
    if (has_block_result) {
      copyTuple(global_work_buffer, work_buf_offset, block_result);
    }
  }

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if (grid_reduce_participate) {
    // Don't need to sync up blocks that are not participating in this
    // reduction
    grid_sync::sync<
        isReduce(X_BLOCK),
        isReduce(Y_BLOCK),
        isReduce(Z_BLOCK),
        PERSISTENT_REDUCTION,
        Aligned>(
        global_sync_buffer[block_red_idx_offset], grid_red_size, last_block);
  }

  // -- START BLOCK CLEANUP -- //
  // All blocks perform the last cleanup, so every block, and every thread
  // will have the final result

  // Initialize block result
  LocalTuple<Types...> last_block_result(init_val);

  if ((PERSISTENT_REDUCTION || last_block) && grid_reduce_participate) {
    // Can use the last block to reduce all the values the blocks filled in.
    // Can use any thread that has been predicated, or has been reduced to do
    // this reduction, cannot use any block that's associated with an
    // iteration domain

    // Start with non-block reduction

    // Index in the reduction segment
    int tid_in_block_reduction_2 = index_utils::maskedOffset<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx, blockDim);

    int block_reduction_size_2 = index_utils::maskedSize<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(blockDim);

    // 3D buffer of reductions:
    //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
    // Change the offset, we want to keep the last two dimensions, but the
    // first dimension is what we will reduce over
    const auto work_buf_offset_2 =
        block_red_idx_offset * num_thread_iters + thread_red_idx_offset;
    for (auto reduction_i = tid_in_block_reduction_2;
         reduction_i < grid_red_size;
         reduction_i += block_reduction_size_2) {
      impl::reduceTuple(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset_2 +
              reduction_i * num_block_iters *
                  num_thread_iters, // Iterating over the outer most
          // dimension, so need to stride by the
          // total number of grid reductions. Could
          // come back and change it so this is the
          // contiguous dimension
          reduction_op);
    }

    // -- START LAST BLOCK - BLOCK REDUCTION -- //

    // Reduced so we have one value per thread, we need to further reduce any
    // dimension that is not an iter dimension

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    // Offset in smem for this thread's result
    auto smem_offset =
        block_reduction_idx * block_reduction_size_2 + tid_in_block_reduction_2;

    // Similar as before, reduce down to nearest power of 2 so we can do a
    // tree reduction
    int np2 = 1 << (31 - __clz(min(block_reduction_size_2, grid_red_size)));

    // Threads values are initialized, so all can participate here
    if (tid_in_block_reduction_2 >= np2) {
      copyTuple(shared_buf, smem_offset, last_block_result);
    }

    block_sync::sync<Aligned>();

    if (tid_in_block_reduction_2 < np2 &&
        tid_in_block_reduction_2 + np2 <
            min(block_reduction_size_2, grid_red_size)) {
      impl::reduceTuple(
          last_block_result, 0, shared_buf, smem_offset + np2, reduction_op);
    }

    if (tid_in_block_reduction_2 < np2) {
      copyTuple(shared_buf, smem_offset, last_block_result);
    }

    // Always sync when communicating across smem
    block_sync::sync<Aligned>();

    // Reduce down to 2 values, last thread will do the final reduction and
    // can save a syncthreads this way
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_block_reduction_2 < factor) {
        impl::reduceTuple(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            reduction_op);
      }
      block_sync::sync<Aligned>();
    }

    // If this thread in each block has the final result before broadcasting
    // to all other threads in block
    bool has_block_result_2 = index_utils::maskedIsZero<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx);
    // Do the last reduction, protected by the write predicate
    copyTuple(last_block_result, shared_buf, smem_offset);
    if (has_block_result && grid_reduce_participate) {
      impl::reduceTuple(last_block_result, 0, out, 0, reduction_op);
      if (min(block_reduction_size_2, grid_red_size) > 1) {
        impl::reduceTuple(
            last_block_result, 0, shared_buf, smem_offset + 1, reduction_op);
      }
    }

    if (grid_reduce_participate && PERSISTENT_REDUCTION) {
      // If persistent reduction, always broadcast reduced values
      copyTuple(shared_buf, smem_offset, last_block_result);
      block_sync::sync<Aligned>();
      if (write_pred && block_reduce_participate) {
        copyTuple(
            out, shared_buf, block_reduction_idx * block_reduction_size_2);
      }
      // For persistent kernels we double the global buffer allocation so we
      // don't need to protect those buffers every iteration preventing the
      // need of an additional grid_sync. Since we flip back and forth between
      // sections of the buffer, the one grid sync protects the other part of
      // the buffer.
    } else {
      if (grid_reduce_participate) {
        if (last_block && has_block_result && block_reduce_participate &&
            write_pred) {
          copyTuple(
              out, shared_buf, block_reduction_idx * block_reduction_size_2);
        }
      }
    }
    // Forward protect the smem used in this reduction
    block_sync::sync<Aligned>();
  }
}

//! Profiled version
template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, typename Func, typename... Types>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduce(
        RefTuple<Types...> out,
        const ConstRefTuple<Types...>& inp,
        VolatilePtrTuple<Types...> global_work_buffer,
        int64_t* global_sync_buffer, // Allocated as product of all
        // non-participating Grid dimension
        PtrTuple<Types...> shared_buf,
        bool read_pred, // Prevent reading from out of bounds memory
        bool write_pred, // Prevent from writing out of bounds
        const LocalTuple<Types...>& init_val,
        Func reduction_op,
        int64_t& cycles,
        int64_t& count) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  reduce<Aligned>(
      out,
      inp,
      global_work_buffer,
      global_sync_buffer,
      shared_buf,
      read_pred,
      write_pred,
      init_val,
      reduction_op);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <
    bool Aligned,
    typename... DataTypes,
    typename... Funcs,
    typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroup(
        RefTuple<DataTypes...> out,
        const ConstRefTuple<DataTypes...>& inp,
        VolatilePtrTuple<DataTypes...> global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        int64_t* global_sync_buffer,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        const LocalTuple<BoolTypes...>& write_preds,
        Funcs... funcs) {
  static_assert(
      sizeof...(DataTypes) == sizeof...(Funcs),
      "Mismatched number of Tuple values and functions");
  static_assert(
      sizeof...(DataTypes) == sizeof...(BoolTypes),
      "Mismatched number of Tuple values and predicate values");

  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    copyTupleIf(out, inp, read_preds && write_preds);
    return;
  }

  // Don't read/write in temporary buffers if in a predicated dimension
  const bool block_reduce_participate = index_utils::
      maskedIsZero<isPred(X_THREAD), isPred(Y_THREAD), isPred(Z_THREAD)>(
          threadIdx);

  // Only threads that with id == 0 in the dimensions being reduced will
  // have a valid result
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  // Initial per-block reduction. Result is broadcast if specified
  // and this call is block reduction only.
  const auto block_result = reduceGroupBlock < !GRID_REDUCE && BROADCAST,
             Aligned > (inp,
                        init_val,
                        shared_mem,
                        read_preds,
                        block_reduce_participate,
                        funcs...);
  // If block reduction only, save to out and exit
  if (!GRID_REDUCE) {
    copyTupleIf(
        out,
        block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));

    // Need a block sync here as reduceGroupBlock does not
    // forward-protect the smem buffer. This block sync is not
    // necessary when a grid reduction follows since a block sync is
    // done just before the grid sync.
    block_sync::sync<Aligned>();
    return;
  }

  // -- START GRID REDUCTION -- //
  // Grid reductions are more challenging for two reasons, (1) the reduction
  // itself is 3D instead of 2D because we now have an iter domain space in
  // the grid dimension. (2) a tree reduction isn't performed, instead all
  // blocks will populate GMEM and one  block will finish the grid reduction.

  // What is the grid reduction size, block reduction already performed so
  // that doesn't have to be taken into consideration
  const auto grid_red_size = index_utils::
      maskedSize<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          gridDim);

  // Which ID in the reduction is this block. Threads can participate in
  // multiple grid reductions, but the block will have the same relative index
  // in those reductions
  const auto idx_in_grid_red = index_utils::
      maskedOffset<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the grid dimension
  const auto num_block_iters = index_utils::
      maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(gridDim);

  // Which grid reduction does this block participate in, in the grid
  // dimension
  const auto block_red_idx_offset = index_utils::
      maskedOffset<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the block dimension
  const auto num_thread_iters = index_utils::
      maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          blockDim);

  // Which grid reduction does this thread participate in, in the block
  // dimension
  const auto thread_red_idx_offset = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // 3D buffer of reductions:
  //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
  // Offset into the work buffer
  const auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *

          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

  if (PERSISTENT_REDUCTION && flip) {
    auto global_buffer_size =
        index_utils::
            maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
                gridDim) *
        index_utils::
            maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
                blockDim) *
        grid_red_size;
    global_work_buffer += global_buffer_size;
  }
  flip = !flip;

  // Per-block partial reduction to global work buffer
  if (grid_reduce_participate && block_reduce_participate && has_block_result) {
    copyTuple(global_work_buffer, work_buf_offset, block_result);
  }

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if (grid_reduce_participate) {
    // Don't need to sync up blocks that are not participating in this
    // reduction
    grid_sync::sync<
        isReduce(X_BLOCK),
        isReduce(Y_BLOCK),
        isReduce(Z_BLOCK),
        PERSISTENT_REDUCTION,
        Aligned>(
        global_sync_buffer[block_red_idx_offset], grid_red_size, last_block);
  }

  // -- START BLOCK CLEANUP -- //
  reduceGroupLastBlock<Aligned>(
      out,
      global_work_buffer,
      init_val,
      shared_mem,
      block_red_idx_offset,
      num_thread_iters,
      num_block_iters,
      thread_red_idx_offset,
      grid_red_size,
      write_preds,
      block_reduce_participate,
      grid_reduce_participate,
      funcs...);

  // Forward protect the smem buffer
  block_sync::sync<Aligned>();
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <
    bool Aligned,
    typename... DataTypes,
    typename... Funcs,
    typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroup(
        RefTuple<DataTypes...> out,
        const ConstRefTuple<DataTypes...>& inp,
        VolatilePtrTuple<DataTypes...> global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        int64_t* global_sync_buffer,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        const LocalTuple<BoolTypes...>& write_preds,
        int64_t& cycles,
        int64_t& count,
        Funcs... funcs) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  reduceGroup<Aligned>(
      out,
      inp,
      global_work_buffer,
      init_val,
      global_sync_buffer,
      shared_mem,
      read_preds,
      write_preds,
      funcs...);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <
    bool BLOCK_BROADCAST,
    bool Aligned,
    typename... DataTypes,
    typename... Funcs,
    typename... BoolTypes>
__device__ __inline__ LocalTuple<DataTypes...> ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroupBlock(
        const ConstRefTuple<DataTypes...>& inp,
        const LocalTuple<DataTypes...>& init_val,
        void* shared_mem,
        const LocalTuple<BoolTypes...>& read_preds,
        bool block_reduce_participate,
        Funcs... funcs) {
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  // Initialize block result
  LocalTuple<DataTypes...> block_result = init_val;

  copyTupleIf(block_result, inp, block_reduce_participate && read_preds);

  // Size of the block reduction segment, can be an int since it's limited
  // to number of threads
  const int block_reduction_size = index_utils::
      maskedSize<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          blockDim);

  // Index in the reduction segment, can be an int since it's limited to
  // number of threads
  const int tid_in_block_reduction = index_utils::
      maskedOffset<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx, blockDim);

  // ID of the block reduction this thread is participating in
  //
  // If any of the parallel dimensions are predicated out, that means
  // they've already been reduced, so we only care about the first thread in
  // that dimension. Therefore don't expand the reduction_idx by that
  // dimension
  const int block_reduction_idx = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // Do not protect the smem buffer as it's not always necessary.
  impl::blockReduceEach<
      BLOCK_BROADCAST,
      false,
      Aligned,
      LocalTuple<DataTypes...>,
      Funcs...>(
      block_result,
      block_result,
      shared_mem,
      has_block_result,
      tid_in_block_reduction,
      block_reduction_size,
      block_reduction_size,
      block_reduction_idx,
      funcs...);

  return block_result;
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <
    bool Aligned,
    typename... DataTypes,
    typename... Funcs,
    typename... BoolTypes>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    reduceGroupLastBlock(
        RefTuple<DataTypes...>& out,
        const VolatilePtrTuple<DataTypes...>& global_work_buffer,
        const LocalTuple<DataTypes...>& init_val,
        void* shared_mem,
        nvfuser_index_t block_red_idx_offset,
        nvfuser_index_t num_thread_iters,
        nvfuser_index_t num_block_iters,
        nvfuser_index_t thread_red_idx_offset,
        nvfuser_index_t grid_red_size,
        const LocalTuple<BoolTypes...>& write_preds,
        bool block_reduce_participate,
        bool grid_reduce_participate,
        Funcs... reduction_ops) {
  // Initialize block result
  LocalTuple<DataTypes...> last_block_result(init_val);

  const bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if ((PERSISTENT_REDUCTION || last_block) && grid_reduce_participate) {
    // Can use the last block to reduce all the values the blocks filled in.
    // Can use any thread that has been predicated, or has been reduced to do
    // this reduction, cannot use any block that's associated with an
    // iteration domain

    // Start with non-block reduction

    // Index in the reduction segment
    int tid_in_block_reduction = index_utils::maskedOffset<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx, blockDim);

    int block_reduction_size = index_utils::maskedSize<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(blockDim);

    bool has_block_result = index_utils::maskedIsZero<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx);

    // 3D buffer of reductions:
    //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
    // Change the offset, we want to keep the last two dimensions, but the
    // first dimension is what we will reduce over
    const auto work_buf_offset =
        block_red_idx_offset * num_thread_iters + thread_red_idx_offset;
    for (auto reduction_i = tid_in_block_reduction; reduction_i < grid_red_size;
         reduction_i += block_reduction_size) {
      impl::reduceEach(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset +
              reduction_i * num_block_iters *
                  num_thread_iters, // Iterating over the outer most
                                    // dimension, so need to stride by the
                                    // total number of grid reductions. Could
                                    // come back and change it so this is the
                                    // contiguous dimension
          reduction_ops...);
    }

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    impl::blockReduceEach<
        BROADCAST,
        false,
        Aligned,
        LocalTuple<DataTypes...>,
        Funcs...>(
        last_block_result,
        last_block_result,
        shared_mem,
        has_block_result,
        tid_in_block_reduction,
        block_reduction_size,
        min(grid_red_size, block_reduction_size),
        block_reduction_idx,
        reduction_ops...);

    copyTupleIf(
        out,
        last_block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));
  }
}

} // namespace fused_reduction

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace fused_reduction {

namespace impl {

//! Implementation helper for welfordEach.
template <int ValIdx, typename Triplet0, typename Triplet1>
struct WelfordForEach {
  static __inline__ __device__ void call(
      Triplet0& triplet0,
      nvfuser_index_t offset0,
      const Triplet1& triplet1,
      nvfuser_index_t offset1) {
    static_assert(
        Triplet0::num_vals == Triplet1::num_vals, "Invalid Triplet types");
    static_assert(
        IsSameType<typename Triplet0::DataType, typename Triplet1::DataType>::
            value,
        "Invalid Triplet types");
    static_assert(
        IsSameType<typename Triplet0::IndexType, typename Triplet1::IndexType>::
            value,
        "Invalid Triplet types");

    using DataType = typename Triplet0::DataType;
    using IndexType = typename Triplet0::IndexType;

    WelfordForEach<ValIdx - 1, Triplet0, Triplet1>::call(
        triplet0, offset0, triplet1, offset1);
    welfordCombine<DataType, IndexType>(
        triplet0.avg.val<ValIdx>(offset0),
        triplet0.var.val<ValIdx>(offset0),
        triplet0.N.val<ValIdx>(offset0),
        triplet1.avg.val<ValIdx>(offset1),
        triplet1.var.val<ValIdx>(offset1),
        triplet1.N.val<ValIdx>(offset1));
  }
};

template <typename Triplet0, typename Triplet1>
struct WelfordForEach<-1, Triplet0, Triplet1> {
  __inline__ __device__ static void call(
      Triplet0& triplet0,
      nvfuser_index_t offset0,
      const Triplet1& triplet1,
      nvfuser_index_t offset1) {}
};

//! Call welfordCombine with each of the triplet tuples. This is a
//! welford version of reduceEach.
template <typename Triplet0, typename Triplet1>
__inline__ __device__ static void welfordEach(
    Triplet0& triplet0,
    nvfuser_index_t offset0,
    const Triplet1& triplet1,
    nvfuser_index_t offset1) {
  WelfordForEach<Triplet0::num_vals - 1, Triplet0, Triplet1>::call(
      triplet0, offset0, triplet1, offset1);
}

// Welford version of BlockReduceEach
template <
    int idx,
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    int NumVals,
    typename DataType,
    typename IndexType>
struct BlockWelfordEach {
  __inline__ __device__ static void reduce(
      LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
      const LocalWelfordTripletTuple<NumVals, DataType, IndexType>&
          partial_result,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx) {
    // Finish the reduction of each tuple value with a smaller offset
    BlockWelfordEach<
        idx - 1,
        BROADCAST,
        true,
        Aligned,
        NumVals,
        DataType,
        IndexType>::
        reduce(
            block_result,
            partial_result,
            shared_buf,
            has_block_result,
            tid_in_reduction,
            num_threads_per_reduction,
            num_elements_per_reduction,
            reduction_idx);

    if (num_elements_per_reduction == 1) {
      if (has_block_result) {
        copyWelfordTripletTuple(block_result, partial_result);
      }
      return;
    }

    LocalTuple<DataType, DataType, IndexType> block_result_i(
        partial_result.avg.val<idx>(0),
        partial_result.var.val<idx>(0),
        partial_result.N.val<idx>(0));

    const auto smem_offset =
        reduction_idx * num_threads_per_reduction + tid_in_reduction;

    const int np2 = 1 << (31 - __clz(num_elements_per_reduction));

    // Threads values are initialized, so all can participate here
    if (tid_in_reduction >= np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    block_sync::sync<Aligned>();
    if (tid_in_reduction < np2 &&
        tid_in_reduction + np2 < num_elements_per_reduction) {
      impl::reduceTuple(
          block_result_i,
          0,
          shared_buf,
          smem_offset + np2,
          welfordCombine<DataType, IndexType>);
    }

    if (tid_in_reduction < np2) {
      copyTuple(shared_buf, smem_offset, block_result_i);
    }

    // Always sync when communicating across smem
    block_sync::sync<Aligned>();

    // Reduce down to 2 values, last thread will do the final reduction and
    // can save a syncthreads this way
    for (int factor = np2 / 2; factor > 1; factor >>= 1) {
      if (tid_in_reduction < factor) {
        impl::reduceTuple(
            shared_buf,
            smem_offset,
            shared_buf,
            smem_offset + factor,
            welfordCombine<DataType, IndexType>);
      }
      block_sync::sync<Aligned>();
    }

    copyTuple(block_result_i, shared_buf, smem_offset);

    // Do the last reduction
    if (has_block_result) {
      impl::reduceTuple(
          block_result_i,
          0,
          shared_buf,
          smem_offset + 1,
          welfordCombine<DataType, IndexType>);
    }

    if (BROADCAST) {
      if (has_block_result) {
        // Put result back in shared memory, put in the first entry of the
        // reduction segment's buffer
        copyTuple(
            shared_buf,
            reduction_idx * num_threads_per_reduction,
            block_result_i);
      }

      // Sync threads to make sure result is in smem
      block_sync::sync<Aligned>();

      copyTuple(
          block_result_i,
          shared_buf,
          reduction_idx * num_threads_per_reduction);
    }

    block_result.avg.val<idx>(0) = block_result_i.val<0>(0);
    block_result.var.val<idx>(0) = block_result_i.val<1>(0);
    block_result.N.val<idx>(0) = block_result_i.val<2>(0);

    if (FORWARD_PROTECT_SMEM) {
      block_sync::sync<Aligned>();
    }
  }
};

// Specialization for idx == -1, i.e., no value to reduce.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    int NumVals,
    typename DataType,
    typename IndexType>
struct BlockWelfordEach<
    -1,
    BROADCAST,
    FORWARD_PROTECT_SMEM,
    Aligned,
    NumVals,
    DataType,
    IndexType> {
  __inline__ __device__ static void reduce(
      LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
      const LocalWelfordTripletTuple<NumVals, DataType, IndexType>&
          partial_result,
      PtrTuple<DataType, DataType, IndexType> shared_buf,
      bool has_block_result,
      int tid_in_reduction,
      int num_threads_per_reduction,
      int num_elements_per_reduction,
      int reduction_idx) {}
};

//! Welford version of blockReduceEach. Perform block-parallel Welford
//! reduction of each Welford triplet.
template <
    bool BROADCAST,
    bool FORWARD_PROTECT_SMEM,
    bool Aligned,
    int NumVals,
    typename DataType,
    typename IndexType>
__inline__ __device__ void blockWelfordEach(
    LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
    const LocalWelfordTripletTuple<NumVals, DataType, IndexType>&
        partial_result,
    PtrTuple<DataType, DataType, IndexType> shared_buf,
    bool has_block_result,
    int tid_in_reduction,
    int num_threads_per_reduction,
    int num_elements_per_reduction,
    int reduction_idx) {
  BlockWelfordEach<
      NumVals - 1,
      BROADCAST,
      FORWARD_PROTECT_SMEM,
      Aligned,
      NumVals,
      DataType,
      IndexType>::
      reduce(
          block_result,
          partial_result,
          shared_buf,
          has_block_result,
          tid_in_reduction,
          num_threads_per_reduction,
          num_elements_per_reduction,
          reduction_idx);
}

} // namespace impl

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, int NumArgs, typename DataType, typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroup(
        typename MakeRefTuple<NumArgs, DataType>::type out_avg,
        typename MakeRefTuple<NumArgs, DataType>::type out_var,
        typename MakeRefTuple<NumArgs, IndexType>::type out_N,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_avg,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_var,
        const typename MakeConstRefTuple<NumArgs, IndexType>::type& inp_N,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_avg,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_var,
        const typename MakeLocalTuple<NumArgs, IndexType>::type& init_N,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_avg,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_var,
        typename MakeVolatilePtrTuple<NumArgs, IndexType>::type
            global_work_buffer_N,
        int64_t* global_sync_buffer,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        const typename MakeLocalTuple<NumArgs, bool>::type& read_preds,
        const typename MakeLocalTuple<NumArgs, bool>::type& write_preds) {
  const ConstRefWelfordTripletTuple<NumArgs, DataType, IndexType> inp(
      inp_avg, inp_var, inp_N);
  RefWelfordTripletTuple<NumArgs, DataType, IndexType> out(
      out_avg, out_var, out_N);

  // If no reduction needed, just return input
  if (!BLOCK_REDUCE && !GRID_REDUCE) {
    copyWelfordTripletTupleIf(out, inp, read_preds && write_preds);
    return;
  }

  // Don't read/write in temporary buffers if in a predicated dimension
  const bool block_reduce_participate = index_utils::
      maskedIsZero<isPred(X_THREAD), isPred(Y_THREAD), isPred(Z_THREAD)>(
          threadIdx);

  // Only threads that with id == 0 in the dimensions being reduced will
  // have a valid result
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  LocalWelfordTripletTuple<NumArgs, DataType, IndexType> block_result(
      init_avg, init_var, init_N);

  // Initial per-block reduction. Result is broadcast if specified
  // and this call is block reduction only.
  welfordGroupBlock<
      !GRID_REDUCE && BROADCAST,
      Aligned,
      NumArgs,
      DataType,
      IndexType>(
      block_result, inp, shared_buf, read_preds, block_reduce_participate);

  // If block reduction only, save to out and exit
  if (!GRID_REDUCE) {
    copyWelfordTripletTupleIf(
        out,
        block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));

    // Need a block sync here as reduceGroupBlock does not
    // forward-protect the smem buffer. This block sync is not
    // necessary when a grid reduction follows since a block sync is
    // done just before the grid sync.
    block_sync::sync<Aligned>();
    return;
  }

  // -- START GRID REDUCTION -- //
  // Grid reductions are more challenging for two reasons, (1) the reduction
  // itself is 3D instead of 2D because we now have an iter domain space in
  // the grid dimension. (2) a tree reduction isn't performed, instead all
  // blocks will populate GMEM and one  block will finish the grid reduction.

  // What is the grid reduction size, block reduction already performed so
  // that doesn't have to be taken into consideration
  const auto grid_red_size = index_utils::
      maskedSize<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          gridDim);

  // Which ID in the reduction is this block. Threads can participate in
  // multiple grid reductions, but the block will have the same relative index
  // in those reductions
  const auto idx_in_grid_red = index_utils::
      maskedOffset<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the grid dimension
  const auto num_block_iters = index_utils::
      maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(gridDim);

  // Which grid reduction does this block participate in, in the grid
  // dimension
  const auto block_red_idx_offset = index_utils::
      maskedOffset<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
          blockIdx, gridDim);

  // How many grid reductions have to be performed, in the block dimension
  const auto num_thread_iters = index_utils::
      maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          blockDim);

  // Which grid reduction does this thread participate in, in the block
  // dimension
  const auto thread_red_idx_offset = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // 3D buffer of reductions:
  //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
  // Offset into the work buffer
  auto work_buf_offset =
      (idx_in_grid_red * num_block_iters + block_red_idx_offset) *
          num_thread_iters +
      thread_red_idx_offset;

  // Don't read/write in temporary buffers if in a predicated dimension
  bool grid_reduce_participate = index_utils::
      maskedIsZero<isPred(X_BLOCK), isPred(Y_BLOCK), isPred(Z_BLOCK)>(blockIdx);

  VolatilePtrWelfordTripletTuple<NumArgs, DataType, IndexType>
      global_work_buffer(
          global_work_buffer_avg, global_work_buffer_var, global_work_buffer_N);

  if (PERSISTENT_REDUCTION && flip) {
    auto global_buffer_size =
        index_utils::
            maskedSize<isIter(X_BLOCK), isIter(Y_BLOCK), isIter(Z_BLOCK)>(
                gridDim) *
        index_utils::
            maskedSize<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
                blockDim) *
        grid_red_size;
    global_work_buffer += global_buffer_size;
  }
  flip = !flip;

  // Per-block partial reduction to global work buffer
  if (grid_reduce_participate && block_reduce_participate && has_block_result) {
    copyWelfordTripletTuple(global_work_buffer, work_buf_offset, block_result);
  }

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if (grid_reduce_participate) {
    // Don't need to sync up blocks that are not participating in this
    // reduction
    grid_sync::sync<
        isReduce(X_BLOCK),
        isReduce(Y_BLOCK),
        isReduce(Z_BLOCK),
        PERSISTENT_REDUCTION,
        Aligned>(
        global_sync_buffer[block_red_idx_offset], grid_red_size, last_block);
  }

  // -- START BLOCK CLEANUP -- //
  welfordGroupLastBlock<Aligned, NumArgs, DataType, IndexType>(
      out,
      global_work_buffer,
      LocalWelfordTripletTuple<NumArgs, DataType, IndexType>(
          init_avg, init_var, init_N),
      shared_buf,
      block_red_idx_offset,
      num_thread_iters,
      num_block_iters,
      thread_red_idx_offset,
      grid_red_size,
      write_preds,
      block_reduce_participate,
      grid_reduce_participate);

  // Forward protect the smem buffer
  block_sync::sync<Aligned>();
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,

    bool BROADCAST>
template <bool Aligned, int NumArgs, typename DataType, typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroup(
        typename MakeRefTuple<NumArgs, DataType>::type out_avg,
        typename MakeRefTuple<NumArgs, DataType>::type out_var,
        typename MakeRefTuple<NumArgs, IndexType>::type out_N,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_avg,
        const typename MakeConstRefTuple<NumArgs, DataType>::type& inp_var,
        const typename MakeConstRefTuple<NumArgs, IndexType>::type& inp_N,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_avg,
        const typename MakeLocalTuple<NumArgs, DataType>::type& init_var,
        const typename MakeLocalTuple<NumArgs, IndexType>::type& init_N,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_avg,
        typename MakeVolatilePtrTuple<NumArgs, DataType>::type
            global_work_buffer_var,
        typename MakeVolatilePtrTuple<NumArgs, IndexType>::type
            global_work_buffer_N,
        int64_t* global_sync_buffer,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        const typename MakeLocalTuple<NumArgs, bool>::type& read_preds,
        const typename MakeLocalTuple<NumArgs, bool>::type& write_preds,
        int64_t& cycles,
        int64_t& count) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  welfordGroup<Aligned, NumArgs, DataType, IndexType>(
      out_avg,
      out_var,
      out_N,
      inp_avg,
      inp_var,
      inp_N,
      init_avg,
      init_var,
      init_N,
      global_work_buffer_avg,
      global_work_buffer_var,
      global_work_buffer_N,
      global_sync_buffer,
      shared_buf,
      read_preds,
      write_preds);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <
    bool BLOCK_BROADCAST,
    bool Aligned,
    int NumVals,
    typename DataType,
    typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupBlock(
        LocalWelfordTripletTuple<NumVals, DataType, IndexType>& block_result,
        const ConstRefWelfordTripletTuple<NumVals, DataType, IndexType>& inp,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        const typename MakeLocalTuple<NumVals, bool>::type& read_preds,
        bool block_reduce_participate) {
  const bool has_block_result = index_utils::
      maskedIsZero<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx);

  copyWelfordTripletTupleIf(
      block_result, inp, block_reduce_participate && read_preds);

  // Size of the block reduction segment, can be an int since it's limited
  // to number of threads
  const int block_reduction_size = index_utils::
      maskedSize<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          blockDim);

  // Index in the reduction segment, can be an int since it's limited to
  // number of threads
  const int tid_in_block_reduction = index_utils::
      maskedOffset<isReduce(X_THREAD), isReduce(Y_THREAD), isReduce(Z_THREAD)>(
          threadIdx, blockDim);

  // ID of the block reduction this thread is participating in
  //
  // If any of the parallel dimensions are predicated out, that means
  // they've already been reduced, so we only care about the first thread in
  // that dimension. Therefore don't expand the reduction_idx by that
  // dimension
  const int block_reduction_idx = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  // Do not protect the smem buffer as it's not always necessary.
  impl::blockWelfordEach<
      BLOCK_BROADCAST,
      false,
      Aligned,
      NumVals,
      DataType,
      IndexType>(
      block_result,
      block_result,
      shared_buf,
      has_block_result,
      tid_in_block_reduction,
      block_reduction_size,
      block_reduction_size,
      block_reduction_idx);
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, int NumVals, typename DataType, typename IndexType>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupLastBlock(
        RefWelfordTripletTuple<NumVals, DataType, IndexType>& out,
        const VolatilePtrWelfordTripletTuple<NumVals, DataType, IndexType>&
            global_work_buffer,
        const LocalWelfordTripletTuple<NumVals, DataType, IndexType>& init_val,
        PtrTuple<DataType, DataType, IndexType> shared_buf,
        nvfuser_index_t block_red_idx_offset,
        nvfuser_index_t num_thread_iters,
        nvfuser_index_t num_block_iters,
        nvfuser_index_t thread_red_idx_offset,
        nvfuser_index_t grid_red_size,
        const typename MakeLocalTuple<NumVals, bool>::type& write_preds,
        bool block_reduce_participate,
        bool grid_reduce_participate) {
  // Initialize block result
  auto last_block_result = init_val;

  const bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  if ((PERSISTENT_REDUCTION || last_block) && grid_reduce_participate) {
    // Can use the last block to reduce all the values the blocks filled in.
    // Can use any thread that has been predicated, or has been reduced to do
    // this reduction, cannot use any block that's associated with an
    // iteration domain

    // Start with non-block reduction

    // Index in the reduction segment
    int tid_in_block_reduction = index_utils::maskedOffset<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx, blockDim);

    int block_reduction_size = index_utils::maskedSize<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(blockDim);

    bool has_block_result = index_utils::maskedIsZero<
        activeNotIter(X_THREAD),
        activeNotIter(Y_THREAD),
        activeNotIter(Z_THREAD)>(threadIdx);

    // 3D buffer of reductions:
    //    [reduction_offset(grid), iter_offset(grid), iter_offset(block)]
    // Change the offset, we want to keep the last two dimensions, but the
    // first dimension is what we will reduce over
    const auto work_buf_offset =
        block_red_idx_offset * num_thread_iters + thread_red_idx_offset;
    for (auto reduction_i = tid_in_block_reduction; reduction_i < grid_red_size;
         reduction_i += block_reduction_size) {
      impl::welfordEach(
          last_block_result,
          0,
          global_work_buffer,
          work_buf_offset + reduction_i * num_block_iters * num_thread_iters);
    }

    // Which block reduction this thread is participating in
    int block_reduction_idx = index_utils::
        maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
            threadIdx, blockDim);

    impl::blockWelfordEach<
        BROADCAST,
        false,
        Aligned,
        NumVals,
        DataType,
        IndexType>(
        last_block_result,
        last_block_result,
        shared_buf,
        has_block_result,
        tid_in_block_reduction,
        block_reduction_size,
        min(grid_red_size, block_reduction_size),
        block_reduction_idx);

    copyWelfordTripletTupleIf(
        out,
        last_block_result,
        write_preds &&
            (block_reduce_participate && (BROADCAST || has_block_result)));
  }
}

} // namespace fused_reduction

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace fused_reduction {
namespace impl {

// Grouped block welford optimized for outer reductions with
// TIDx and TIDy mapped to non-reduction and reduction domains,
// respectively with unused TIDz.
//
// The main motivation of this optimized version is the same as the
// grouped grid reduction, i.e, by doing multiple reductions together,
// it is possible to reduce the number of synchronizations. However,
// unlike the grouped grid reduction, the cost of grouping can be
// prohitively high, i.e., the size of the work buffer must be
// expanded by a factor of grouping. In the case of grid
// reductions, the buffer is on global memory, so the space requirement
// is not a concern, but that isn't the case with block reductions,
// since the buffer is on shared memory, which has a limited
// capacity.
//
// This implementation tries to benefit from aggregated block
// synchronizations while minimizing the cost of the expanded buffer
// size by first partially reducing the input within each warp. It
// would save the required buffer size by a factor of WARP_SIZE /
// blockDim.x as the reduction is done along threadIdx.y. So to be
// effective, blockDim.x needs to be smaller than WARP_SIZE, and in the
// case of grouped grid welford, it should be typically 8 or 16.
//
// The algorithm is an adaptation of scattered butterfly reduction,
// aka recursive halving, commonly used for implementing
// MPI_Reduce_scatter. For a visual illustration of the data
// organization, see, for example, page 22 of Solomonik,
// Design of Parallel and High-Performance Computing:
// Distributed-Memory Models and Algorithms, 2015
// (https://solomonik.cs.illinois.edu/talks/dphpc-dec-2015.pdf)
//
// Assumptions:
// - blockDim.x and blockDim.y are statically known values so that all
// loops can be completely unrolled
// - blockDim.x is smaller than WARP_SIZE
// - blockDim.x evenly divides WARP_SIZE
// - There are multiple warps per block
// - The gouping factor, NumVals, is at least as large as the warp
// dimY and is divisible by the warp dimY.
//
// This is meant to be used as part of the grouped grid welford
// reduction but should be usable as a standalone block welford routine as
// long as the above assumptions hold.
//
// Note: Having an output reference parameter resulted in using more
// registers than just returing the output. Results would vary
// depending on compiler versions, but it seems safer to return outputs
// as a new value.
template <bool Aligned, int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ WelfordTriplet<DataType> blockWelfordOuter(
    DataType* inp_avg,
    DataType* inp_var,
    nvfuser_index_t inp_N,
    DataType* smem) {
  constexpr int num_warps = BDIMX * BDIMY / 32;
  static_assert(num_warps >= 1, "There must be at least a single warp");
  static_assert(32 % BDIMX == 0, "blockDimx.x must be able to divide 32");

  const int tid = threadIdx.x + threadIdx.y * BDIMX;
  const int wid = tid / 32;

  // Dimension of the Y axis within each warp
  constexpr int wdimy = 32 / BDIMX;
  static_assert(NumVals >= wdimy, "NumVals must be >= 32 / blockDim.x");
  static_assert(
      NumVals % wdimy == 0, "NumVals must be divisible by 32 / blockDim.x");
  // There must be at least a single warp

  // Y index within each warp
  const int warp_tidy = threadIdx.y % wdimy;

  // Thread index in each warp
  const int lane_id = threadIdx.x + warp_tidy * BDIMX;

  constexpr int smem_var_offset = num_warps * BDIMX * NumVals;
  constexpr int smem_N_offset = num_warps * BDIMX * NumVals * 2;

  // We define a chunk as a value in a group and a chunk size as the
  // number of group values per thread. Initially, the chunk size is
  // NumVals. After the initial warp reduction, the chunk size is
  // reduced to NumVals/wdimy. For example, suppose NumVals=8,
  // blockDim.x=8, blockDim.y=32, then wdimy=4, so after the initial
  // warp reduction, the chunk size is 2. This is the number of
  // elements each thread stores to shared memory.

  int chunk_size = NumVals;

  // Butterfly reduction, a.k.a. recursive halving as each iteration
  // halves the number of values
#pragma unroll
  for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
    chunk_size /= 2;

    const auto peer_N = __shfl_xor_sync(0xffffffff, inp_N, lane_mask);
    const auto updated_N = inp_N + peer_N;
    const DataType b_N_div_ab_N =
        updated_N != 0 ? ((DataType)peer_N) / ((DataType)updated_N) : 0;

#pragma unroll
    for (int index_in_chunk = 0; index_in_chunk < chunk_size;
         ++index_in_chunk) {
      DataType pushed_avg = 0;
      DataType pushed_var = 0;
      DataType self_avg = 0;
      DataType self_var = 0;
      // Divergent branch. Not a big deal with independent scheduling?
      if (lane_id & lane_mask) {
        // Push first half
        auto push_offset = index_in_chunk;
        auto self_offset = index_in_chunk + chunk_size;
        pushed_avg = inp_avg[push_offset];
        pushed_var = inp_var[push_offset];
        self_avg = inp_avg[self_offset];
        self_var = inp_var[self_offset];
      } else {
        // Push second half
        auto push_offset = index_in_chunk + chunk_size;
        auto self_offset = index_in_chunk;
        pushed_avg = inp_avg[push_offset];
        pushed_var = inp_var[push_offset];
        self_avg = inp_avg[self_offset];
        self_var = inp_var[self_offset];
      }
      auto peer_avg = __shfl_xor_sync(0xffffffff, pushed_avg, lane_mask);
      auto peer_var = __shfl_xor_sync(0xffffffff, pushed_var, lane_mask);

      auto delta = peer_avg - self_avg;
      self_avg += delta * b_N_div_ab_N;
      self_var += peer_var + delta * delta * ((DataType)(inp_N)) * b_N_div_ab_N;

      inp_avg[index_in_chunk] = self_avg;
      inp_var[index_in_chunk] = self_var;
    }
    inp_N = updated_N;
  }

  // At this point, chunk_size is reduced to NumVals/wdimy as
  // mentioned above. Each thread has warp-reduced chunk_size values
  // in array inp. This chunk_size_post_reduction should be equal to
  // chunk_size at this point.
  constexpr int chunk_size_post_reduction = NumVals / wdimy;

  // More specifically, the warp_tidy of each thread defines
  // the chunk IDs held by the thread as follows:
  //
  // [warp_tidy * chunk_size_post_reduction, warp_tidy *
  // chunk_size_post_reduction + chunk_size_post_reduction]
  //
  // Each thread uploads the chunk_size_post_reduction values one by
  // one. Each chunk is spread by BDIMX * BDIMY values. The data
  // layout of the shared memory is:
  //
  // [chunk_size, wid, warp_tidy, TIDx]
  //
  // The remaining reduction is done on the WID
  // dimension. More specifically, we assign one warp per chunk (or
  // a value of the group). The wdimy threads of the same threadId.x
  // collectively reduce num_warps partial results, each of which is
  // stored with stride 32. This means that there will be wdimy-way
  // bank conflicts, so to avoid that, swizzling is also employed.
#pragma unroll
  for (int i = 0; i < chunk_size; ++i) {
    // Accumulating smem offset from the innermost dimension
    int smem_offset = 0;
    // TIDx
    smem_offset += threadIdx.x;
    // Warp_TIDy with swizzle
    smem_offset += ((warp_tidy + wid) % wdimy) * BDIMX;
    // WID
    smem_offset += wid * 32;
    // chunk_size
    smem_offset += i * BDIMX * BDIMY;
    smem[smem_offset] = inp_avg[i];
    smem[smem_var_offset + smem_offset] = inp_var[i];
    // Upload N only when threadIdx.x == 0 && chunk_index == 0
    if (threadIdx.x == 0 && i == 0 && warp_tidy == 0) {
      reinterpret_cast<nvfuser_index_t*>(smem + smem_N_offset)[wid] = inp_N;
    }
  }

  block_sync::sync<Aligned>();

  // The next step is to let each thread of a warp independently
  // accumulate the partial results on the shared memory
  // reduction. A single warp is used to accumulate of the partial
  // results for a single chunk, so warp wid takes care of the wid-th
  // chunk.
  //
  // The starting offset of partial results of a chunk is:
  //
  // (wid % chunk_size_post_reduction) * BDIMX * BDIMY + (wid /
  // chunk_size_post_reduction) * BDIMX
  //
  // Note that each thread had chunk_size_post_reduction contiguous
  // chunks, so when uploaded to shmem, they are strided by
  // BDIMX*BDIMY, hence (wid % chunk_size_post_reduction) * BDIMX *
  // BDIMY.

  // The vector width is likely at least 4, so at least 4 warps should
  // be used, which is
  // enough to occupy an SM. When NumVals=8, it might be more
  // efficient to use just 4 warps with each warp taking care of two
  // groups, but the difference would be pretty small.

  // Also, the number of warps should be at least 8 and can be 16
  // too. NumVals should be 8 at largest, so it's always num_warps >=
  // NumVals.

  DataType avg = 0;
  DataType var = 0;
  nvfuser_index_t N = 0;

  static_assert(
      num_warps >= NumVals,
      "Number of warps must be at least as large as NumVals");

  if (wid < NumVals) {
#pragma unroll
    for (int i = warp_tidy; i < num_warps; i += wdimy) {
      int offset = 0;
      offset += threadIdx.x;
      // Offset to the partial results of the i-th warp
      offset += i * 32;
      // Offset to the chunk for this warp. Swizzled to avoid bank
      // conflicts.
      offset += ((wid / chunk_size + i) % wdimy) * BDIMX;
      offset += (wid % chunk_size) * BDIMX * BDIMY;

      DataType avg_smem = smem[offset];
      DataType var_smem = smem[smem_var_offset + offset];
      nvfuser_index_t N_smem =
          reinterpret_cast<nvfuser_index_t*>(&smem[smem_N_offset])[i];

      welfordCombine(avg, var, N, avg_smem, var_smem, N_smem);
    }
  }

  block_sync::sync<Aligned>();

  // Nothing to do for warps whose wid is larger than NunVals
  if (wid >= NumVals) {
    WelfordTriplet<DataType> out = {0, 0, 0};
    return out;
  }

  // Standard binary-exchange reduction within wdimy intra-warp
  // threads.
#pragma unroll
  for (int lane_mask = 16; lane_mask >= BDIMX; lane_mask /= 2) {
    auto avg_peer = __shfl_xor_sync(0xffffffff, avg, lane_mask);
    auto var_peer = __shfl_xor_sync(0xffffffff, var, lane_mask);
    auto N_peer = __shfl_xor_sync(0xffffffff, N, lane_mask);

    welfordCombine(avg, var, N, avg_peer, var_peer, N_peer);
  }

  WelfordTriplet<DataType> out = {avg, var, N};
  return out;
}

} // namespace impl
} // namespace fused_reduction

// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
namespace fused_reduction {

namespace impl {

// Utility struct to hold multiple values for grouped Welford. The
// count is uniform, so there's only one N value.
template <int NumVals, typename DataType>
struct WelfordTripletVector {
  Array<DataType, NumVals, NumVals> avg_;
  Array<DataType, NumVals, NumVals> var_;
  nvfuser_index_t N_;

  WelfordTripletVector() = default;

  __device__ WelfordTripletVector(
      const DataType avg[NumVals],
      const DataType var[NumVals],
      const nvfuser_index_t N) {
    memcpy(avg_.array, avg, sizeof(DataType) * NumVals);
    memcpy(var_.array, var, sizeof(DataType) * NumVals);
    N_ = N;
  }

  __device__ WelfordTripletVector& operator=(
      const WelfordTripletVector<NumVals, DataType>& other) {
    avg_ = other.avg_;
    var_ = other.var_;
    N_ = other.N_;
    return *this;
  }

  __device__ void init() {
    avg_.set((DataType)0);
    var_.set((DataType)0);
    N_ = 0;
  }

  __device__ DataType& avg(int idx) {
    return avg_[idx];
  }

  __device__ DataType avg(int idx) const {
    return avg_.array[idx];
  }

  __device__ DataType& var(int idx) {
    return var_[idx];
  }

  __device__ DataType var(int idx) const {
    return var_.array[idx];
  }

  __device__ nvfuser_index_t& N() {
    return N_;
  }

  __device__ nvfuser_index_t N() const {
    return N_;
  }
};

// The offset in smem buffer to broadcast final results within a
// thread block
template <int BDIMX>
__inline__ __device__ int getSmemGroupOffset(int iter_idx, int group_idx) {
  return group_idx * BDIMX + iter_idx;
}

// Upload the final results to smem for intra-block broadcasting
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ void copyFromTripletToSmem(
    DataType* smem,
    int iter_idx,
    int group_idx,
    const WelfordTriplet<DataType>& local_triplet) {
  int offset = getSmemGroupOffset<BDIMX>(iter_idx, group_idx);
  smem[offset] = local_triplet.avg;
  int smem_stride = BDIMX * NumVals;
  smem[smem_stride + offset] = local_triplet.var;
  if (iter_idx == 0 && group_idx == 0) {
    reinterpret_cast<nvfuser_index_t*>(smem + smem_stride * 2)[0] =
        local_triplet.N;
  }
}

// Fetch the final results from smem for intra-block broadcasting
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__inline__ __device__ void copyFromSmemToTriplet(
    WelfordTriplet<DataType>& local_triplet,
    const DataType* smem,
    int iter_idx,
    int group_idx) {
  int offset = getSmemGroupOffset<BDIMX>(iter_idx, group_idx);
  local_triplet.avg = smem[offset];
  int smem_stride = BDIMX * NumVals;
  local_triplet.var = smem[smem_stride + offset];
  local_triplet.N =
      reinterpret_cast<const nvfuser_index_t*>(smem + smem_stride * 2)[0];
}

// Per-thread accumulation of the per-block partial results in global
// memory. There's gridDim.y partial results, which is accumulated in
// parallel by threadIdx.y. This should be followed by a block reduction.
template <int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ WelfordTripletVector<NumVals, DataType>
welfordGroupAccumulateGlobalBuffer(
    volatile DataType* global_buf_avg,
    volatile DataType* global_buf_var,
    volatile nvfuser_index_t* global_buf_N,
    bool flip) {
  const int grid_size = gridDim.x * gridDim.y;
  const int iter_idx = threadIdx.x;
  const int red_idx = threadIdx.y;
  const int num_threads_per_reduction = BDIMY;

  WelfordTripletVector<NumVals, DataType> results;

  results.init();

  // Reduction is done cooperatively with the thread blocks with the
  // same blockIdx.x. Thread blocks with the same blockIdx.x uses a
  // global buffer of size blockDim.x * gridDim.y for each value in a
  // group.

  // Advance the global buffer pointers to the location of the values
  // to accumulate for the first group value (i.e., gi == 0 in the
  // below NumVals loop)
  global_buf_avg += iter_idx + blockIdx.x * BDIMX * gridDim.y;
  global_buf_var += iter_idx + blockIdx.x * BDIMX * gridDim.y;
  global_buf_N += iter_idx + blockIdx.x * BDIMX * gridDim.y;

  if (flip) {
    global_buf_avg += BDIMX * grid_size * NumVals;
    global_buf_var += BDIMX * grid_size * NumVals;
    global_buf_N += BDIMX * grid_size * NumVals;
  }

  // Since there's gridDim.y elements to reduce using blockDim.y
  // threads, loop over gridDim.y with stride blockDim.y. First, just
  // grab the values in the global memory.

  if (red_idx < gridDim.y) {
    int work_buf_offset = red_idx * BDIMX;
    // N is constant across NumVals
    const auto g_N = global_buf_N[work_buf_offset];
    results.N() = g_N;

    // Just copy the first elements
#pragma unroll
    for (int gi = 0; gi < NumVals; ++gi) {
      auto& a_avg = results.avg(gi);
      auto& a_var = results.var(gi);
      auto b_avg = global_buf_avg[work_buf_offset];
      auto b_var = global_buf_var[work_buf_offset];
      work_buf_offset += grid_size * BDIMX;

      results.avg(gi) = b_avg;
      results.var(gi) = b_var;
    }
  }

  // Accumulate into results by looping over the remaining results in
  // the global buffer
  for (int ri = red_idx + num_threads_per_reduction; ri < gridDim.y;
       ri += num_threads_per_reduction) {
    int work_buf_offset = ri * BDIMX;
    // N is constant across NumVals
    const auto g_N = global_buf_N[work_buf_offset];
    nvfuser_index_t updated_N = results.N() + g_N;

    // Hoist the division by updated_N as it's invariant over the
    // NumVals loop
    DataType b_N_div_ab_N = updated_N != 0
        ? (((DataType)g_N) / ((DataType)updated_N))
        : (DataType)0;
    DataType a_N_b_N_div_ab_N = ((DataType)results.N()) * b_N_div_ab_N;

#pragma unroll
    for (int gi = 0; gi < NumVals; ++gi) {
      auto& a_avg = results.avg(gi);
      auto& a_var = results.var(gi);
      auto b_avg = global_buf_avg[work_buf_offset];
      auto b_var = global_buf_var[work_buf_offset];
      work_buf_offset += grid_size * BDIMX;

      auto delta = b_avg - a_avg;
      a_avg += delta * b_N_div_ab_N;
      a_var += b_var + delta * delta * a_N_b_N_div_ab_N;
    }
    results.N() = updated_N;
  }

  return results;
}

} // namespace impl

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupOuter(
        DataType out_avg[NumVals],
        DataType out_var[NumVals],
        nvfuser_index_t out_N[NumVals],
        const DataType in_avg[NumVals],
        const DataType in_var[NumVals],
        nvfuser_index_t in_N,
        DataType* global_buf_avg,
        DataType* global_buf_var,
        nvfuser_index_t* global_buf_N,
        DataType* shared_buf,
        int64_t* global_sync_buffer) {
  using namespace fused_reduction::impl;

  static_assert(
      isIter(X_BLOCK) && isReduce(Y_BLOCK) && inactive(Z_BLOCK) &&
          isIter(X_THREAD) && isReduce(Y_THREAD) && inactive(Z_THREAD),
      "Invalid parallelization for outer welford reduction");

  static_assert(
      BDIMY % NumVals == 0, "blockDim.y must be divisible by group count");
  static_assert(BDIMX <= 32, "blockDim.x must be up to 32.");
  static_assert(
      (BDIMX * BDIMY) % 32 == 0, "Number of threads must be a multiple of 32.");
  static_assert(32 % BDIMX == 0, "blockDim.x must be able to divide 32.");
  static_assert(
      NumVals >= (32 / BDIMX), "Group count must be >= 32 / blockDim.x");

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    out_avg[i] = in_avg[i];
    out_var[i] = in_var[i];
  }

  auto iter_tid = index_utils::
      maskedOffset<isIter(X_THREAD), isIter(Y_THREAD), isIter(Z_THREAD)>(
          threadIdx, blockDim);

  auto per_block_result =
      impl::blockWelfordOuter<Aligned, NumVals, DataType, BDIMX, BDIMY>(
          out_avg, out_var, in_N, shared_buf);

  // At this point, threads with tid_in_group == 0 has valid partial
  // results. Store them to global buffer.

  const int grid_size = gridDim.x * gridDim.y;
  const int iter_idx = threadIdx.x;

  // Stores the partial results into the global work buffer. Only
  // threads with tid_in_group have the valid partial results
  const int wid = (threadIdx.x + threadIdx.y * BDIMX) / 32;
  constexpr int wdimy = 32 / BDIMX;
  const int warp_tidy = threadIdx.y % wdimy;
  const bool has_valid_block_reduction_result = warp_tidy == 0 && wid < NumVals;
  // Each valid result is held by a warp
  const int valid_group_idx = wid;

  if (has_valid_block_reduction_result) {
    int work_buf_offset = iter_idx + blockIdx.y * BDIMX +
        blockIdx.x * BDIMX * gridDim.y + valid_group_idx * BDIMX * grid_size;
    if (PERSISTENT_REDUCTION && flip) {
      auto global_buffer_size = BDIMX * grid_size * NumVals;
      work_buf_offset += global_buffer_size;
    }

    global_buf_avg[work_buf_offset] = per_block_result.avg;
    global_buf_var[work_buf_offset] = per_block_result.var;

    // the count values should be the same across the group, so just
    // store once
    if (valid_group_idx == 0) {
      global_buf_N[work_buf_offset] = per_block_result.N;
    }
  }

  flip = !flip;

  // -- GLOBAL BUFFER FILLED -- //

  bool last_block = index_utils::
      maskedIsLast<isReduce(X_BLOCK), isReduce(Y_BLOCK), isReduce(Z_BLOCK)>(
          blockIdx, gridDim);

  grid_sync::sync<
      isReduce(X_BLOCK),
      isReduce(Y_BLOCK),
      isReduce(Z_BLOCK),
      PERSISTENT_REDUCTION,
      Aligned>(global_sync_buffer[blockIdx.x], gridDim.y, last_block);

  auto partial_results =
      welfordGroupAccumulateGlobalBuffer<NumVals, DataType, BDIMX, BDIMY>(
          global_buf_avg, global_buf_var, global_buf_N, !flip);

  auto per_block_final_result =
      impl::blockWelfordOuter<Aligned, NumVals, DataType, BDIMX, BDIMY>(
          partial_results.avg_.array,
          partial_results.var_.array,
          partial_results.N_,
          shared_buf);

  // At this point, each thread of the groups with tid_in_group=0
  // has the final reduction result. We need to upload them to
  // shmem for broadcasting.
  if (has_valid_block_reduction_result) {
    copyFromTripletToSmem<NumVals, DataType, BDIMX, BDIMY>(
        shared_buf, iter_idx, valid_group_idx, per_block_final_result);
  }

  __syncthreads();

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    WelfordTriplet<DataType> final_result;
    copyFromSmemToTriplet<NumVals, DataType, BDIMX, BDIMY>(
        final_result, shared_buf, iter_idx, i);
    out_avg[i] = final_result.avg;
    out_var[i] = final_result.var;
    in_N = final_result.N;
  }

#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    out_N[i] = in_N;
  }

  // Forward protect the smem buffer
  __syncthreads();
}

template <
    int X_BLOCK,
    int Y_BLOCK,
    int Z_BLOCK,
    int X_THREAD,
    int Y_THREAD,
    int Z_THREAD,
    bool PERSISTENT_REDUCTION,
    bool BROADCAST>
template <bool Aligned, int NumVals, typename DataType, int BDIMX, int BDIMY>
__device__ __inline__ void ParallelReduce<
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    X_THREAD,
    Y_THREAD,
    Z_THREAD,
    PERSISTENT_REDUCTION,
    BROADCAST>::
    welfordGroupOuter(
        DataType out_avg[NumVals],
        DataType out_var[NumVals],
        nvfuser_index_t out_N[NumVals],
        const DataType in_avg[NumVals],
        const DataType in_var[NumVals],
        nvfuser_index_t in_N,
        DataType* global_buf_avg,
        DataType* global_buf_var,
        nvfuser_index_t* global_buf_N,
        DataType* shared_buf,
        int64_t* global_sync_buffer,
        int64_t& cycles,
        int64_t& count) {
  int64_t start_counter = 0;

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    start_counter = readCycleCounter();
  }

  welfordGroupOuter<Aligned, NumVals, DataType, BDIMX, BDIMY>(
      out_avg,
      out_var,
      out_N,
      in_avg,
      in_var,
      in_N,
      global_buf_avg,
      global_buf_var,
      global_buf_N,
      shared_buf,
      global_sync_buffer);

  if (isLastBlockInGrid() &&
      index_utils::maskedIsZero<true, true, true>(threadIdx)) {
    cycles += readCycleCounter() - start_counter;
    ++count;
  }
}

} // namespace fused_reduction
__global__ void nvfuser_none_f0_c0_r0_g0(Tensor<__half, 2, 2> T0, Tensor<__half, 2, 2> T1, Tensor<__half, 2, 2> T5, Tensor<int64_t, 1, 1> T13, Tensor<float, 7, 7> T14) {
  alignas(16) extern __shared__ char array[];
  const unsigned smem_offset = 0;
  nvfuser_index_t i0;
  int splitk_factor = gridDim.z;
  i0 = ceilDiv((ceilDiv(T0.logical_size[1LL], 32)), splitk_factor);
  nvfuser_index_t i1;
  i1 = ((nvfuser_index_t)threadIdx.x) / 4;
  nvfuser_index_t i2;
  i2 = ((nvfuser_index_t)threadIdx.x) % 4;
  nvfuser_index_t i3;
  i3 = 8LL * i2;
  nvfuser_index_t i4;
  i4 = T0.logical_size[1LL] * i1;
  nvfuser_index_t i5;
  i5 = (8 * T0.logical_size[1LL]) * ((nvfuser_index_t)threadIdx.y);
  nvfuser_index_t i6;
  i6 = (16 * T0.logical_size[1LL]) * ((nvfuser_index_t)threadIdx.z);
  nvfuser_index_t i7;
  i7 = (32LL * i0) * ((nvfuser_index_t)blockIdx.z);
  nvfuser_index_t i8;
  i8 = 128LL * T0.logical_size[1LL];
  nvfuser_index_t i9;
  i9 = i8 * ((nvfuser_index_t)blockIdx.y);
  __half* ptr10;
  ptr10 = (((((T1.data + i3) + i4) + i5) + i6) + i7) + i9;
  nvfuser_index_t i11;
  i11 = 32 * T0.logical_size[1LL];
  nvfuser_index_t i12;
  i12 = 64 * i1;
  nvfuser_index_t i13;
  i13 = i2 / 4;
  nvfuser_index_t i14;
  i14 = 64 * i13;
  nvfuser_index_t i15;
  i15 = i2 % 4;
  nvfuser_index_t i16;
  i16 = 16 * (((i1 / 2) % 4) ^ i15);
  nvfuser_index_t i17;
  i17 = 512 * ((nvfuser_index_t)threadIdx.y);
  nvfuser_index_t i18;
  i18 = 1024 * ((nvfuser_index_t)threadIdx.z);
  __half* T7 = reinterpret_cast<__half*>(array + smem_offset + 40960LL);
  unsigned i19;
  i19 = toSmem(T7);
  unsigned i20;
  i20 = ((((i19 + i12) + i14) + i16) + i17) + i18;
  nvfuser_index_t i21;
  i21 = i8 * ((nvfuser_index_t)blockIdx.x);
  __half* ptr22;
  ptr22 = (((((T0.data + i3) + i4) + i5) + i6) + i7) + i21;
  __half* T6 = reinterpret_cast<__half*>(array + smem_offset + 0);
  unsigned i23;
  i23 = toSmem(T6);
  unsigned i24;
  i24 = ((((i23 + i12) + i14) + i16) + i17) + i18;
  nvfuser_index_t i25;
  i25 = ((nvfuser_index_t)threadIdx.x) / 8;
  nvfuser_index_t i26;
  i26 = ((nvfuser_index_t)threadIdx.x) % 8;
  nvfuser_index_t i27;
  i27 = i25 / 2;
  nvfuser_index_t i28;
  i28 = 64 * i26;
  nvfuser_index_t i29;
  i29 = i25 % 2;
  nvfuser_index_t i30;
  i30 = (i26 / 2) % 4;
  unsigned i31;
  i31 = (i23 + (512 * i29)) + i28;
  nvfuser_index_t i32;
  i32 = 4096 * ((nvfuser_index_t)threadIdx.z);
  unsigned i33;
  i33 = ((i31 + (64 * (i27 / 4))) + (16 * (i30 ^ (i27 % 4)))) + i32;
  unsigned i34;
  i34 = (i19 + (512 * i27)) + i28;
  nvfuser_index_t i35;
  i35 = 4096 * ((nvfuser_index_t)threadIdx.y);
  unsigned i36;
  i36 = ((i34 + (64 * (i29 / 4))) + (16 * (i30 ^ (i29 % 4)))) + i35;
  __half* ptr37;
  ptr37 = ((((((128 + T1.data) + i3) + i4) + i5) + i6) + i7) + i9;
  __half* ptr38;
  ptr38 = ((((((128 + T0.data) + i3) + i4) + i5) + i6) + i7) + i21;
  nvfuser_index_t i39;
  i39 = 2LL + i27;
  unsigned i40;
  i40 = ((i31 + (64 * (i39 / 4))) + (16 * (i30 ^ (i39 % 4)))) + i32;
  nvfuser_index_t i41;
  i41 = 2LL + i29;
  unsigned i42;
  i42 = ((i34 + (64 * (i41 / 4))) + (16 * (i30 ^ (i41 % 4)))) + i35;
  nvfuser_index_t i43;
  i43 = i1 % 8;
  nvfuser_index_t i44;
  i44 = ((((2048LL * (i1 / 8)) + (128LL * i43)) + (2 * i15)) + (64 * ((nvfuser_index_t)threadIdx.y))) + (8192LL * ((nvfuser_index_t)threadIdx.z));
  nvfuser_index_t i45;
  i45 = i43 % 4;
  nvfuser_index_t i46;
  i46 = ((nvfuser_index_t)threadIdx.x) / 2;
  nvfuser_index_t i47;
  i47 = ((((32LL * (i46 / 4)) + (4 * (((nvfuser_index_t)threadIdx.x) % 2))) + (128LL * ((nvfuser_index_t)threadIdx.y))) + (256LL * ((nvfuser_index_t)threadIdx.z))) + (8LL * ((i46 % 4) ^ (((nvfuser_index_t)threadIdx.y) + (2 * ((nvfuser_index_t)threadIdx.z)))));
  nvfuser_index_t i48;
  i48 = 128LL * ((nvfuser_index_t)blockIdx.y);
  nvfuser_index_t i49;
  i49 = 4 * ((nvfuser_index_t)threadIdx.x);
  nvfuser_index_t i50;
  i50 = (((i49 + (T1.logical_size[0LL] * ((nvfuser_index_t)threadIdx.y))) + ((2LL * T1.logical_size[0LL]) * ((nvfuser_index_t)threadIdx.z))) + i48) + ((128LL * T1.logical_size[0LL]) * ((nvfuser_index_t)blockIdx.x));
  nvfuser_index_t i51;
  i51 = 4LL * T1.logical_size[0LL];
  nvfuser_index_t i52;
  i52 = 8 * i2;
  nvfuser_index_t i53;
  i53 = ((7 - T0.logical_size[1LL]) + i52) + i7;
  nvfuser_index_t i54;
  i54 = 8 * ((nvfuser_index_t)threadIdx.y);
  nvfuser_index_t i55;
  i55 = 16 * ((nvfuser_index_t)threadIdx.z);
  nvfuser_index_t i56;
  i56 = ((((-T1.logical_size[0LL]) + i1) + i54) + i55) + i48;
  nvfuser_index_t i57;
  i57 = -T0.logical_size[0LL];
  nvfuser_index_t i58;
  i58 = 128LL * ((nvfuser_index_t)blockIdx.x);
  nvfuser_index_t i59;
  i59 = (((i57 + i1) + i54) + i55) + i58;
  nvfuser_index_t i60;
  i60 = (135 + i52) + i7;
  bool b61;
  b61 = ((3 + i49) + i48) < T1.logical_size[0LL];
  nvfuser_index_t i62;
  i62 = ((i57 + ((nvfuser_index_t)threadIdx.y)) + (2LL * ((nvfuser_index_t)threadIdx.z))) + i58;
  bool b63;
  b63 = (((nvfuser_index_t)blockIdx.z) == (((nvfuser_index_t)gridDim.z) + -1)) && b61;
  float* T12 = reinterpret_cast<float*>(array + smem_offset + 0);
  Array<float, 128, 2> T11;
  #pragma unroll
  for(nvfuser_index_t i64 = 0; i64 < 4; ++i64) {
    nvfuser_index_t i65;
    i65 = 32 * i64;
    #pragma unroll
    for(nvfuser_index_t i66 = 0; i66 < 4; ++i66) {
      ((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i65 + (8 * i66))]))).set(0);
    }
  }
  #pragma unroll
  for(nvfuser_index_t i67 = 0; i67 < 4; ++i67) {
    nvfuser_index_t i68;
    i68 = 32LL * i67;
    __half* ptr69;
    ptr69 = ptr10 + i68;
    nvfuser_index_t i70;
    i70 = 8192 * i67;
    unsigned i71;
    i71 = i20 + i70;
    __half* ptr72;
    ptr72 = ptr22 + i68;
    unsigned i73;
    i73 = i24 + i70;
    bool b74;
    b74 = i53 < (-i68);
    #pragma unroll
    for(nvfuser_index_t i75 = 0; i75 < 4; ++i75) {
      bool b76;
      b76 = b74 && (i56 < (-(32 * i75)));
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i71 + (2048 * i75)))),
         "l"((ptr69 + (i11 * i75))),
         "n"(16LL),
         "r"((uint32_t)((!b76)))
      );
    }
    #pragma unroll
    for(nvfuser_index_t i77 = 0; i77 < 4; ++i77) {
      bool b78;
      b78 = b74 && (i59 < (-(32 * i77)));
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i73 + (2048 * i77)))),
         "l"((ptr72 + (i11 * i77))),
         "n"(16LL),
         "r"((uint32_t)((!b78)))
      );
    }
    asm volatile("cp.async.commit_group;\n");
  }
  asm volatile("cp.async.wait_group %0;\n"::"n"(3LL));
  __syncthreads();
  Array<__half, 64, 8> T9;
  Array<__half, 64, 8> T10;
  #pragma unroll
  for(nvfuser_index_t i79 = 0; i79 < 4; ++i79) {
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[0]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[1]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[2]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[3])
      :"r"((uint32_t)((i33 + (1024 * i79))))
    );
  }
  #pragma unroll
  for(nvfuser_index_t i80 = 0; i80 < 4; ++i80) {
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[0]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[1]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[2]),
       "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[3])
      :"r"((uint32_t)((i36 + (1024 * i80))))
    );
  }
  #pragma unroll 1
  for(nvfuser_index_t i81 = 0; i81 < i0; ++i81) {
    nvfuser_index_t i82;
    i82 = 32LL * i81;
    __half* ptr83;
    ptr83 = ptr37 + i82;
    nvfuser_index_t i84;
    i84 = 8192 * ((4 + i81) % 5);
    unsigned i85;
    i85 = i20 + i84;
    __half* ptr86;
    ptr86 = ptr38 + i82;
    unsigned i87;
    i87 = i24 + i84;
    nvfuser_index_t i88;
    i88 = 8192 * (i81 % 5);
    unsigned i89;
    i89 = i40 + i88;
    unsigned i90;
    i90 = i42 + i88;
    nvfuser_index_t i91;
    i91 = 8192 * ((1 + i81) % 5);
    unsigned i92;
    i92 = i33 + i91;
    unsigned i93;
    i93 = i36 + i91;
    bool b94;
    b94 = (i60 + i82) < T0.logical_size[1LL];
    #pragma unroll
    for(nvfuser_index_t i75 = 0; i75 < 4; ++i75) {
      bool b95;
      b95 = b94 && (i56 < (-(32 * i75)));
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i85 + (2048 * i75)))),
         "l"((ptr83 + (i11 * i75))),
         "n"(16LL),
         "r"((uint32_t)((!b95)))
      );
    }
    #pragma unroll
    for(nvfuser_index_t i77 = 0; i77 < 4; ++i77) {
      bool b96;
      b96 = b94 && (i59 < (-(32 * i77)));
      asm volatile(
        "{\n"
        "  .reg .pred p0; \n"
        "  setp.ne.b32 p0, %3, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], %2, p0;\n"
        "}\n"
        :
        :"r"((uint32_t)((i87 + (2048 * i77)))),
         "l"((ptr86 + (i11 * i77))),
         "n"(16LL),
         "r"((uint32_t)((!b96)))
      );
    }
    asm volatile("cp.async.commit_group;\n");
    #pragma unroll
    for(nvfuser_index_t i79 = 0; i79 < 4; ++i79) {
      asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(32 + (8LL * i79))]))[0]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(32 + (8LL * i79))]))[1]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(32 + (8LL * i79))]))[2]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(32 + (8LL * i79))]))[3])
        :"r"((uint32_t)((i89 + (1024 * i79))))
      );
    }
    __half T2[32];
    #pragma unroll
    for(nvfuser_index_t i97 = 0; i97 < 4; ++i97) {
      nvfuser_index_t i98;
      i98 = 8 * i97;
      #pragma unroll
      for(nvfuser_index_t i99 = 0; i99 < 2; ++i99) {
        nvfuser_index_t i100;
        i100 = i98 + (4LL * i99);
        #pragma unroll
        for(nvfuser_index_t i101 = 0; i101 < 2; ++i101) {
          nvfuser_index_t i102;
          i102 = i100 + (2LL * i101);
          #pragma unroll
          for(nvfuser_index_t i103 = 0; i103 < 2; ++i103) {
            T2[(i102 + i103)] = 0.000000000e+00f;
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i97 = 0; i97 < 4; ++i97) {
      nvfuser_index_t i104;
      i104 = 8 * i97;
      #pragma unroll
      for(nvfuser_index_t i99 = 0; i99 < 2; ++i99) {
        nvfuser_index_t i105;
        i105 = i104 + (4LL * i99);
        #pragma unroll
        for(nvfuser_index_t i101 = 0; i101 < 2; ++i101) {
          nvfuser_index_t i106;
          i106 = i105 + (2LL * i101);
          #pragma unroll
          for(nvfuser_index_t i103 = 0; i103 < 2; ++i103) {
            nvfuser_index_t i107;
            i107 = i106 + i103;
            T2[i107]
               = T9[i107];
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i80 = 0; i80 < 4; ++i80) {
      asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(32 + (8LL * i80))]))[0]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(32 + (8LL * i80))]))[1]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(32 + (8LL * i80))]))[2]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(32 + (8LL * i80))]))[3])
        :"r"((uint32_t)((i90 + (1024 * i80))))
      );
    }
    __half T3[32];
    #pragma unroll
    for(nvfuser_index_t i108 = 0; i108 < 4; ++i108) {
      nvfuser_index_t i109;
      i109 = 8 * i108;
      #pragma unroll
      for(nvfuser_index_t i110 = 0; i110 < 2; ++i110) {
        nvfuser_index_t i111;
        i111 = i109 + (4 * i110);
        #pragma unroll
        for(nvfuser_index_t i112 = 0; i112 < 2; ++i112) {
          nvfuser_index_t i113;
          i113 = i111 + (2LL * i112);
          #pragma unroll
          for(nvfuser_index_t i114 = 0; i114 < 2; ++i114) {
            T3[(i113 + i114)] = 0.000000000e+00f;
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i108 = 0; i108 < 4; ++i108) {
      nvfuser_index_t i115;
      i115 = 8 * i108;
      #pragma unroll
      for(nvfuser_index_t i110 = 0; i110 < 2; ++i110) {
        nvfuser_index_t i116;
        i116 = i115 + (4 * i110);
        #pragma unroll
        for(nvfuser_index_t i112 = 0; i112 < 2; ++i112) {
          nvfuser_index_t i117;
          i117 = i116 + (2LL * i112);
          #pragma unroll
          for(nvfuser_index_t i114 = 0; i114 < 2; ++i114) {
            nvfuser_index_t i118;
            i118 = i117 + i114;
            T3[i118]
               = T10[i118];
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i64 = 0; i64 < 4; ++i64) {
      nvfuser_index_t i119;
      i119 = 8 * i64;
      nvfuser_index_t i120;
      i120 = 32 * i64;
      #pragma unroll
      for(nvfuser_index_t i66 = 0; i66 < 4; ++i66) {
        nvfuser_index_t i121;
        i121 = 8 * i66;
        asm(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          :"=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][0]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][1]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][2]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][3])
          :"r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[0]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[1]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[2]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[3]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i121])))))[0LL][0]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i121])))))[0LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][0]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][2]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[0LL][3])
        );
        asm(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          :"=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][0]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][1]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][2]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][3])
          :"r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[0]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[1]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[2]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i119]))[3]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i121])))))[1LL][0]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i121])))))[1LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][0]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][2]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i120 + i121)])))))[1LL][3])
        );
      }
    }
    #pragma unroll
    for(nvfuser_index_t i97 = 0; i97 < 4; ++i97) {
      nvfuser_index_t i122;
      i122 = 8 * i97;
      #pragma unroll
      for(nvfuser_index_t i99 = 0; i99 < 2; ++i99) {
        nvfuser_index_t i123;
        i123 = i122 + (4LL * i99);
        #pragma unroll
        for(nvfuser_index_t i101 = 0; i101 < 2; ++i101) {
          nvfuser_index_t i124;
          i124 = i123 + (2LL * i101);
          #pragma unroll
          for(nvfuser_index_t i103 = 0; i103 < 2; ++i103) {
            T2[(i124 + i103)] = 0.000000000e+00f;
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i97 = 0; i97 < 4; ++i97) {
      nvfuser_index_t i125;
      i125 = 8 * i97;
      nvfuser_index_t i126;
      i126 = 32 + i125;
      #pragma unroll
      for(nvfuser_index_t i99 = 0; i99 < 2; ++i99) {
        nvfuser_index_t i127;
        i127 = 4LL * i99;
        nvfuser_index_t i128;
        i128 = i125 + i127;
        nvfuser_index_t i129;
        i129 = i126 + i127;
        #pragma unroll
        for(nvfuser_index_t i101 = 0; i101 < 2; ++i101) {
          nvfuser_index_t i130;
          i130 = 2LL * i101;
          nvfuser_index_t i131;
          i131 = i128 + i130;
          nvfuser_index_t i132;
          i132 = i129 + i130;
          #pragma unroll
          for(nvfuser_index_t i103 = 0; i103 < 2; ++i103) {
            T2[(i131 + i103)]
               = T9[(i132 + i103)];
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i108 = 0; i108 < 4; ++i108) {
      nvfuser_index_t i133;
      i133 = 8 * i108;
      #pragma unroll
      for(nvfuser_index_t i110 = 0; i110 < 2; ++i110) {
        nvfuser_index_t i134;
        i134 = i133 + (4 * i110);
        #pragma unroll
        for(nvfuser_index_t i112 = 0; i112 < 2; ++i112) {
          nvfuser_index_t i135;
          i135 = i134 + (2LL * i112);
          #pragma unroll
          for(nvfuser_index_t i114 = 0; i114 < 2; ++i114) {
            T3[(i135 + i114)] = 0.000000000e+00f;
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i108 = 0; i108 < 4; ++i108) {
      nvfuser_index_t i136;
      i136 = 8 * i108;
      nvfuser_index_t i137;
      i137 = 32 + i136;
      #pragma unroll
      for(nvfuser_index_t i110 = 0; i110 < 2; ++i110) {
        nvfuser_index_t i138;
        i138 = 4 * i110;
        nvfuser_index_t i139;
        i139 = i136 + i138;
        nvfuser_index_t i140;
        i140 = i137 + i138;
        #pragma unroll
        for(nvfuser_index_t i112 = 0; i112 < 2; ++i112) {
          nvfuser_index_t i141;
          i141 = 2LL * i112;
          nvfuser_index_t i142;
          i142 = i139 + i141;
          nvfuser_index_t i143;
          i143 = i140 + i141;
          #pragma unroll
          for(nvfuser_index_t i114 = 0; i114 < 2; ++i114) {
            T3[(i142 + i114)]
               = T10[(i143 + i114)];
          }
        }
      }
    }
    #pragma unroll
    for(nvfuser_index_t i64 = 0; i64 < 4; ++i64) {
      nvfuser_index_t i144;
      i144 = 8 * i64;
      nvfuser_index_t i145;
      i145 = 32 * i64;
      #pragma unroll
      for(nvfuser_index_t i66 = 0; i66 < 4; ++i66) {
        nvfuser_index_t i146;
        i146 = 8 * i66;
        asm(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          :"=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][0]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][1]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][2]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][3])
          :"r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[0]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[1]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[2]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[3]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i146])))))[0LL][0]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i146])))))[0LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][0]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][2]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[0LL][3])
        );
        asm(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          :"=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][0]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][1]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][2]),
           "=f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][3])
          :"r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[0]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[1]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[2]),
           "r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T2[i144]))[3]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i146])))))[1LL][0]),
           "r"((*reinterpret_cast<Array<Array<uint32_t, 2, 1>, 2, 1>*>(&((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T3[i146])))))[1LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][0]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][1]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][2]),
           "f"((*reinterpret_cast<Array<Array<float, 4, 1>, 2, 1>*>(&((*reinterpret_cast<Array<float, 8, 1>*>(&T11[(i145 + i146)])))))[1LL][3])
        );
      }
    }
    asm volatile("cp.async.wait_group %0;\n"::"n"(3LL));
    __syncthreads();
    #pragma unroll
    for(nvfuser_index_t i79 = 0; i79 < 4; ++i79) {
      asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[0]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[1]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[2]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T9[(8LL * i79)]))[3])
        :"r"((uint32_t)((i92 + (1024 * i79))))
      );
    }
    #pragma unroll
    for(nvfuser_index_t i80 = 0; i80 < 4; ++i80) {
      asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        :"=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[0]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[1]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[2]),
         "=r"((*reinterpret_cast<Array<uint32_t, 4, 1>*>(&T10[(8LL * i80)]))[3])
        :"r"((uint32_t)((i93 + (1024 * i80))))
      );
    }
  }
  asm volatile("cp.async.wait_all;\n");
  __syncthreads();
  #pragma unroll
  for(nvfuser_index_t i147 = 0; i147 < 4; ++i147) {
    nvfuser_index_t i148;
    i148 = 32 * i147;
    nvfuser_index_t i149;
    i149 = i44 + (2048LL * i147);
    #pragma unroll
    for(nvfuser_index_t i150 = 0; i150 < 4; ++i150) {
      nvfuser_index_t i151;
      i151 = i148 + (8 * i150);
      nvfuser_index_t i152;
      i152 = i13 + (2 * i150);
      #pragma unroll
      for(nvfuser_index_t i153 = 0; i153 < 2; ++i153) {
        nvfuser_index_t i154;
        i154 = i151 + (4LL * i153);
        nvfuser_index_t i155;
        i155 = i152 + i153;
        nvfuser_index_t i156;
        i156 = (i149 + (32LL * (i155 / 4))) + (8LL * (i45 ^ (i155 % 4)));
        #pragma unroll
        for(nvfuser_index_t i157 = 0; i157 < 2; ++i157) {
          loadGeneric<float, 2>( &T12[(i156 + (1024LL * i157))],  &T11[(i154 + (2LL * i157))]);
        }
      }
    }
  }
  // Allocate global tensor T13
  int num_segments = index_utils::maskedSize<true, true, false>(gridDim);
  int segment_id = index_utils::maskedOffset<true, true, false>(blockIdx, gridDim);
  //int64_t* semaphore = &T13[index_utils::maskedOffset<true, true, false>(blockIdx, gridDim)];
  int64_t cta_rank = index_utils::maskedOffset<false, false, true>(blockIdx, gridDim);
  bool first_block = cta_rank == 0;
  bool last_block = cta_rank == index_utils::maskedSize<false, false, true>(gridDim) - 1;

  // [hierarchical shuffle-up reduction]
  // Here we implement the "shuffle-up" pattern for reduction to the last
  // thread in a warp. However in this case instead of reducing over a warp we
  // are reducing over blocks containing the split-K partitions.
  //
  //   For example, for S = 8 this looks like the following
  //
  //    Each CTA starts with its current sum value. In the first step CTAs 0-3
  //    write their sum value to work buffer.
  //    CTA:          0   1   2   3   4   5   6   7
  //                  |   |   |   |
  //                  v   v   v   v
  //    work buffer:  0   1   2   3
  //
  //    In the next step, the other half of the CTAs reads the values that were
  //    written in the first step. This repeats until completion.
  //    CTA:                          4   5   6   7
  //    work buffer (shift by 4):     0   1   2   3
  //    CTAs 4-7 read shifted work buffer and update sums
  //    CTA:                          4   5   6   7
  //    CTAs 4&5 write new sums       |   |
  //                                  v   v
  //    work buffer:                  0   1
  //   
  //    CTA:                                  6   7
  //    work buffer (shift by 2):             0   1
  //    CTAs 6&7 read shifted work buffer and update sums
  //    CTA:                                  6   7
  //    CTA 6 writes new sum                  |
  //                                          v
  //    work buffer:                          0
  //   
  //    CTA:                                      7
  //    work buffer (shift by 1):                 0
  //    CTA 7 reads shifted work buffer, updates to get total sum
  //
  // This algorithm uses shuffle-up instead of shuffle-down since we assume
  // that CTAs are launched in ascending order. Note that the first half of the
  // CTAs does not need to wait for any earlier results: those CTAs can write
  // their results immediately to the work buffer then exit.
  //
  // If the split-K factor is not a power of two, we must imagine there are
  // empty _earlier_ CTAs: e.g. for splitk_factor=5:
  //
  //    In the first write step, CTAs 0&1 write to positions 2&3
  //    CTA:         -3  -2  -1   0   1   2   3   4
  //                              |
  //                              v
  //    work buffer:  0   1   2   3
  //
  //    In the first read step, CTAs 1-3 we do not read positions 0-2. CTA 4
  //    reads position 3 since it was written to by CTA 0.
  //    CTA:                          1   2   3   4
  //    work buffer (shift by 4):     0   1   2   3
  //    CTA:                          1   2   3   4
  //    CTAs 1&2 write sums           |   |
  //                                  v   v
  //    work buffer:                  0   1
  //
  //    The algorithm proceeds as normal from here.
  //
  //    CTA:                                  3   4
  //    work buffer (shift by 2):             0   1
  //    CTAs 3&4 read shifted work buffer and update sums
  //    CTA:                                  3   4
  //    CTA 3 writes sum                      |
  //                                          v
  //    work buffer:                          0
  //
  //    CTA:                                      4
  //    work buffer (shift by 1):                 0
  //    CTA 4 reads shifted work buffer, updates to get total sum
  //
  // It is easy to see above that there are 7 stores and 7 loads in the S=8
  // case. For S=5 there are 4 stores and 4 loads. In general, there are always
  // S-1 stores and S-1 loads, and CTAs never load data written by a later CTA.
  //
  // Let S be the split-K factor and let D = ceil(log2(S)) be the depth of a
  // binary tree representation of the split-K partitions.
  // We use a work buffer whose size is 2**(D-1). For each output tile, we
  // use 2**(D-1) many counting semaphores: one for each work buffer
  // position. These can be reset to 0 by the CTA doing the last read from that
  // position.
  //
  // We call E=2**D-S the "offset" as it corresponds to the number of "negative
  // CTAs" that we need to ignore.
  //
  // [semaphore usage]
  // CTAs start "active" and become "inactive" after their writes if they are
  // in the bottom half of the "active" CTAs in the present stage. We hold one
  // semaphore for each work buffer position. When a semaphore is zero it means
  // the corresponding work buffer position is available for writing to. Once
  // that work buffer slot is filled with data, the _target CTA_'s rank is
  // written to the semaphore (note that rank 0 is never a target CTA). At each
  // stage, each active CTA watches its semaphore to see when its rank appears.
  // At that point, it reads the data and resets the semaphore to zero so that
  // the work buffer slot can be re-used.
  //
  // Note one exception. In the case that splitk_factor is not a factor of two,
  // there will be no CTA to write some entries in the stage 1. This would lead
  // to a deadlock if the above described method were used. To mitigate this
  // and to prevent reading uninitialized data, at the beginning of each stage
  // each CTA computes the maximum CTA rank in the partial sum corresponding to
  // its present work buffer location. If that rank is negative, then since
  // these are partial prefix sums it guarantees that no data has yet been
  // written to that work buffer location. In that case, there is no read done
  // and the sum is not updated.
  //
  // Notice that the semaphores are set to zero whenever the target CTA reads a
  // result, so the semaphores are reset to zero at the end of the algorithm.


  // Compute D=ceil(log2(S)) which is the position of the most significant high
  // bit of S-1 (starting with 1 for the position of the least significant bit)
  int D = (8 * sizeof(long long int)) - __clzll(splitk_factor - 1);
  int padded_size = 1 << D;
  int E = padded_size - splitk_factor;
  // final pointer looks like &T14[work_buffer_base + work_buffer_pos * work_buffer_size + work_buffer_step * loop_index]
  int work_buffer_size = T5.logical_size[0] * T5.logical_size[1];
  int work_buffer_pos = E + cta_rank;
  int work_buffer_base = i50;
  int work_buffer_step = i51;
  int64_t* semaphore = nullptr;

  // Initialize register buffer with this CTA's result
  __syncthreads();
  Array<float, 128, 2>& sum = T11;
  constexpr int vec_size = 4;
  #pragma unroll
  for(nvfuser_index_t i158 = 0; i158 < 32; ++i158) {
    const nvfuser_index_t off = 4 * i158;
    bool b160;
    b160 = i62 < (-off);
    // smem->reg load
    // NOTE: we could instead keep the running sum in smem
    if (b61 && b160) {
      loadGeneric<float, vec_size>(&T11[4 * i158], &T12[(i47 + (512LL * i158))]);
    }
  }
  float* work_buf =  nullptr;

  for (int stage = D - 1; stage >= 0; --stage) {
    int stage_shift = 1 << stage;

    // Only the top stage_shift * 2 ranks are active. Other ranks skip stage
    bool active = work_buffer_pos >= 0;

    // Whether to write sum to work buffer
    bool write_sum = active && (work_buffer_pos < stage_shift);

    // Whether to read work buffer and update sum
    // The sender used previous iteration's stage_shift
    bool read_sum = active && work_buffer_pos >= stage_shift && cta_rank >= stage_shift;

    // All active ranks update their semaphores. The first half writes the
    // receiver rank to which their data will be shuffled and the second half
    // writes 0 to reset the semaphore.
    int64_t new_semaphore_value = write_sum ? cta_rank + stage_shift : 0;

    // Each stage consists of:
    //  - write sum
    //  - write semaphore with receiver semaphore
    //  - shift
    //  - wait for semaphore to equal cta_rank
    //  - read and update sum
    if (write_sum) {
      semaphore = &T13[work_buffer_pos * num_segments + segment_id];
      // Wait for the semaphore to be reset before overwriting data
      grid_sync::semaphoreWait(semaphore, 0);
      __syncthreads();

      work_buf = &T14[work_buffer_base + work_buffer_pos * work_buffer_size];
      Array<float, 128, 2>& sum = T11;
      #pragma unroll
      for(nvfuser_index_t i158 = 0; i158 < 32; ++i158) {
        const nvfuser_index_t off = 4 * i158;
        bool b160;
        b160 = i62 < (-off);
        if (b61 && b160) {
          loadLocalToGlobal<float, vec_size, true>(work_buf + work_buffer_step * i158, &T11[4 * i158]);
        }
      }
      
      // Signal that data is ready to be read from receiver rank
      __syncthreads();
      grid_sync::semaphoreRelease(semaphore, cta_rank + stage_shift);
    }

    // Shift work buffer
    work_buffer_pos -= stage_shift;

    if (read_sum) {
      work_buf = &T14[work_buffer_base + work_buffer_pos * work_buffer_size];

      // wait for semaphore
      semaphore = &T13[work_buffer_pos * num_segments + segment_id];
      grid_sync::semaphoreWait(semaphore, cta_rank);
      __syncthreads();

      for(nvfuser_index_t i158 = 0; i158 < 32; ++i158) {
        const nvfuser_index_t off = 4 * i158;
        bool b160;
        b160 = i62 < (-off);
        if (b61 && b160) {
          Array<float, 4, 4> T8;
          loadGlobalToLocal<float, vec_size, true, CacheOp::Global>(&T8[0], work_buf + work_buffer_step * i158);
          nvfuser_index_t off = 4 * i158;
          for (nvfuser_index_t j = 0; j < 4; ++j) {
            T11[off + j] += T8[j];
          }
        }
      }

      // Reset the semaphore. This indicates that the read is complete and it is safe to overwrite this slot
      __syncthreads();
      grid_sync::semaphoreRelease(semaphore, 0);
    }
  }
  if (cta_rank == splitk_factor - 1) {
    // Write output
    #pragma unroll
    for(nvfuser_index_t i158 = 0; i158 < 32; ++i158) {
      nvfuser_index_t i159;
      i159 = i50 + (i51 * i158);
      const nvfuser_index_t off = 4 * i158;
      bool b160;
      b160 = i62 < (-off);
      Array<__half, 4, 4> T8;
      #pragma unroll
      for(nvfuser_index_t i162 = 0; i162 < 4; ++i162) {
        T8[i162]
           = __float2half(T11[off + i162]);
      }
      if ((b63 && b160)) {
        loadLocalToGlobal<__half, /*vec_size=*/4, /*is_volatile=*/false>( &T5[i159], &T8[0]);
      }
    }
  }

  // Original
  if (false) {
  grid_sync::blockSerializeWait<false, false, true>(semaphore);
  __syncthreads();
  #pragma unroll
  for(nvfuser_index_t i158 = 0; i158 < 32; ++i158) {
    nvfuser_index_t i159;
    i159 = i50 + (i51 * i158);
    bool b160;
    b160 = i62 < (-(4LL * i158));
    bool b161;
    b161 = b61 && b160;
    Array<float, 4, 4> T4;
    T4.set(float(0.000000000e+00f));
    // Allocate global tensor T14
    reduction::serialReductionStep</*vec_size=*/4>(
      &T4[0],
      &T12[(i47 + (512LL * i158))],
      0.000000000e+00f,
      &T14[i159],
      [](float &a, float b) { a = a + b; },
      first_block,
      last_block,
      b161,
      b161);
    Array<__half, 4, 4> T8;
    #pragma unroll
    for(nvfuser_index_t i162 = 0; i162 < 4; ++i162) {
      T8[i162]
         = __float2half(T4[i162]);
    }
    if ((b63 && b160)) {
      loadLocalToGlobal<__half, /*vec_size=*/4, /*is_volatile=*/false>( &T5[i159], &T8[0]);
    }
  }
  grid_sync::blockSerializeRelease<false, false, true>(semaphore);
  } // original
}
}

