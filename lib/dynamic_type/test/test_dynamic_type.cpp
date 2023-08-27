// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "dynamic_type.h"

#include "utils.h"

#if defined(__GLIBCXX__) && __GLIBCXX__ >= 20230714
#define STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE 1
#endif

namespace dynamic_type {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif

TEST_F(DynamicTypeTest, Printing) {
  std::stringstream ss;
  ss << DoubleInt64Bool(299792458L) << ", " << DoubleInt64Bool(3.14159) << ", "
     << DoubleInt64Bool(true);
  EXPECT_EQ(ss.str(), "299792458, 3.14159, 1");

  std::stringstream ss2;
  static_assert(opcheck<std::stringstream&> << opcheck<IntSomeType>);
  ss2 << IntSomeType(299792458);
  EXPECT_EQ(ss2.str(), "299792458");

  EXPECT_THAT(
      [&]() { ss << IntSomeType(); },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Can not print")));
  EXPECT_THAT(
      [&]() { ss << IntSomeType(SomeType{}); },
      ::testing::ThrowsMessage<std::runtime_error>(
          ::testing::HasSubstr("Can not print")));
  static_assert(!(opcheck<std::stringstream&> << opcheck<SomeTypes>));
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace container_test {
// Testing containers support by implementing the set-theoretic definition of
// natural numbers:
// https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers

// TODO: unordered set is a better fit for this case, but it does not work with
// some old compilers (for example the old gcc on our CI). This is a workaround.
// See [Incomplete type support in STL] for more details.
#if defined(STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE)

struct StupidHash {
  template <typename T>
  size_t operator()(const T&) const {
    // This hash always collides, but who cares?
    return 0;
  }
};

template <typename T>
using Set = std::unordered_set<T, StupidHash>;

#else

template <typename T>
using Set = std::vector<T>;
#define insert push_back

#endif

using NaturalNumber = DynamicType<Containers<Set>>;

TEST_F(DynamicTypeTest, SetTheoreticNaturalNumbers) {
  auto next = [](const NaturalNumber& n) {
    // recursively define natural number n + 1 as n U {n}
    auto set = n.as<Set>();
    set.insert(n);
    return NaturalNumber(set);
  };

  NaturalNumber zero = Set<NaturalNumber>{};
  NaturalNumber one = next(zero);
  NaturalNumber two = next(one);
  NaturalNumber three = next(two);
  NaturalNumber four = next(three);
  NaturalNumber five = next(four);
  NaturalNumber six = next(five);
  NaturalNumber seven = next(six);
  NaturalNumber eight = next(seven);
  NaturalNumber nine = next(eight);
  NaturalNumber ten = next(nine);

  EXPECT_TRUE(zero.is<Set>());
  EXPECT_TRUE(one.is<Set>());
  EXPECT_TRUE(two.is<Set>());
  EXPECT_TRUE(three.is<Set>());
  EXPECT_TRUE(four.is<Set>());
  EXPECT_TRUE(five.is<Set>());
  EXPECT_TRUE(six.is<Set>());
  EXPECT_TRUE(seven.is<Set>());
  EXPECT_TRUE(eight.is<Set>());
  EXPECT_TRUE(nine.is<Set>());
  EXPECT_TRUE(ten.is<Set>());

  EXPECT_EQ(zero.as<Set>().size(), 0);
  EXPECT_EQ(one.as<Set>().size(), 1);
  EXPECT_EQ(two.as<Set>().size(), 2);
  EXPECT_EQ(three.as<Set>().size(), 3);
  EXPECT_EQ(four.as<Set>().size(), 4);
  EXPECT_EQ(five.as<Set>().size(), 5);
  EXPECT_EQ(six.as<Set>().size(), 6);
  EXPECT_EQ(seven.as<Set>().size(), 7);
  EXPECT_EQ(eight.as<Set>().size(), 8);
  EXPECT_EQ(nine.as<Set>().size(), 9);
  EXPECT_EQ(ten.as<Set>().size(), 10);

  EXPECT_EQ(zero, NaturalNumber(Set<NaturalNumber>{}));
  EXPECT_EQ(one, NaturalNumber(Set<NaturalNumber>{zero}));
  EXPECT_EQ(two, NaturalNumber(Set<NaturalNumber>{zero, one}));
  EXPECT_EQ(three, NaturalNumber(Set<NaturalNumber>{zero, one, two}));
  EXPECT_EQ(four, NaturalNumber(Set<NaturalNumber>{zero, one, two, three}));
  EXPECT_EQ(
      five, NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four}));
  EXPECT_EQ(
      six,
      NaturalNumber(Set<NaturalNumber>{zero, one, two, three, four, five}));
  EXPECT_EQ(
      seven,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six}));
  EXPECT_EQ(
      eight,
      NaturalNumber(
          Set<NaturalNumber>{zero, one, two, three, four, five, six, seven}));
  EXPECT_EQ(
      nine,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight}));
  EXPECT_EQ(
      ten,
      NaturalNumber(Set<NaturalNumber>{
          zero, one, two, three, four, five, six, seven, eight, nine}));
}

#undef insert

TEST_F(DynamicTypeTest, FromContainerToContainer) {
  using IntOrVec = DynamicType<Containers<std::vector>, int>;
  using Vec = DynamicType<Containers<std::vector>>;

  static_assert(std::is_constructible_v<IntOrVec, std::vector<int>>);
  static_assert(
      std::is_constructible_v<IntOrVec, std::vector<std::vector<int>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<int>>>>);
  static_assert(std::is_constructible_v<
                IntOrVec,
                std::vector<std::vector<std::vector<std::vector<int>>>>>);

  static_assert(opcheck<IntOrVec>.canCastTo(opcheck<std::vector<double>>));
  static_assert(
      opcheck<IntOrVec>.canCastTo(opcheck<std::vector<std::vector<double>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<double>>>>));
  static_assert(opcheck<IntOrVec>.canCastTo(
      opcheck<std::vector<std::vector<std::vector<std::vector<double>>>>>));

  static_assert(opcheck<IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<Vec>]);
  static_assert(opcheck<IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<Vec>[opcheck<const Vec>]);
  static_assert(opcheck<const IntOrVec>[opcheck<const IntOrVec>]);
  static_assert(!opcheck<const Vec>[opcheck<const Vec>]);

  IntOrVec zero = 0;
  IntOrVec one = 1;
  IntOrVec two = 2;

  std::vector<std::vector<int>> vvi{{1, 2, 3}, {4, 5, 6}};
  IntOrVec x = vvi;
  EXPECT_EQ(x[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[0][0], 1);
  EXPECT_EQ(x[0][1], 2);
  EXPECT_EQ(x[0][2], 3);
  EXPECT_EQ(x[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[1][0], 4);
  EXPECT_EQ(x[1][1], 5);
  EXPECT_EQ(x[1][2], 6);

  EXPECT_EQ(x[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(x[zero][zero], 1);
  EXPECT_EQ(x[zero][one], 2);
  EXPECT_EQ(x[zero][two], 3);
  EXPECT_EQ(x[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(x[one][zero], 4);
  EXPECT_EQ(x[one][one], 5);
  EXPECT_EQ(x[one][two], 6);

  const IntOrVec xx = vvi;
  EXPECT_EQ(xx[0], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[0][0], 1);
  EXPECT_EQ(xx[0][1], 2);
  EXPECT_EQ(xx[0][2], 3);
  EXPECT_EQ(xx[1], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[1][0], 4);
  EXPECT_EQ(xx[1][1], 5);
  EXPECT_EQ(xx[1][2], 6);

  EXPECT_EQ(xx[zero], IntOrVec(std::vector<int>{1, 2, 3}));
  EXPECT_EQ(xx[zero][zero], 1);
  EXPECT_EQ(xx[zero][one], 2);
  EXPECT_EQ(xx[zero][two], 3);
  EXPECT_EQ(xx[one], IntOrVec(std::vector<int>{4, 5, 6}));
  EXPECT_EQ(xx[one][zero], 4);
  EXPECT_EQ(xx[one][one], 5);
  EXPECT_EQ(xx[one][two], 6);

  std::vector<std::vector<double>> vvd{{1, 2, 3}, {4, 5, 6}};
  EXPECT_EQ((std::vector<std::vector<double>>)x, vvd);
  EXPECT_EQ((std::vector<double>)x[0], vvd[0]);
  EXPECT_EQ((std::vector<double>)x[1], vvd[1]);
}

} // namespace container_test

TEST_F(DynamicTypeTest, Hash) {
  static_assert(has_cross_type_equality<DoubleInt64Bool>);
  using IntOrStr = DynamicType<NoContainers, int64_t, std::string>;
  static_assert(!has_cross_type_equality<IntOrStr>);
  std::unordered_map<IntOrStr, double> m;
  m[IntOrStr(0L)] = 0;
  m[IntOrStr(299792458L)] = 299792458;
  m[IntOrStr("speed of light")] = 299792458;
  m[IntOrStr("pi")] = 3.14159;
  EXPECT_EQ(m.at(IntOrStr(0L)), 0);
  EXPECT_EQ(m.at(IntOrStr(299792458L)), 299792458);
  EXPECT_EQ(m.at(IntOrStr("speed of light")), 299792458);
  EXPECT_EQ(m.at(IntOrStr("pi")), 3.14159);
}

} // namespace dynamic_type

template <>
struct std::hash<DoubleInt64Bool> {
  size_t operator()(const DoubleInt64Bool& x) const {
    return 0;
  }
};

namespace dynamic_type {

TEST_F(DynamicTypeTest, Hash2) {
  std::unordered_map<DoubleInt64Bool, double> m;
  m[DoubleInt64Bool(false)] = 0;
  m[DoubleInt64Bool(299792458L)] = 299792458;
  m[DoubleInt64Bool(3.14159)] = 3.14159;
  EXPECT_EQ(m.at(DoubleInt64Bool(false)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(0L)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(0.0)), 0);
  EXPECT_EQ(m.at(DoubleInt64Bool(299792458L)), 299792458);
  EXPECT_EQ(m.at(DoubleInt64Bool(299792458.0)), 299792458);
  EXPECT_EQ(m.at(DoubleInt64Bool(3.14159)), 3.14159);
}

namespace member_pointer_test {
struct A {
  int x;
  int y;
};
struct B {
  int x;
  int y;
};
struct C {
  int x;
  int y;
};
struct D {
  int x;
  int y;
};
struct E {
  int x;
  int y;
};

struct CD {
  std::variant<C, D> v;

  constexpr const int& operator->*(int C::*member) const {
    return std::get<C>(v).*member;
  }

  constexpr const int& operator->*(int D::*member) const {
    return std::get<D>(v).*member;
  }

  constexpr int& operator->*(int C::*member) {
    return std::get<C>(v).*member;
  }

  constexpr int& operator->*(int D::*member) {
    return std::get<D>(v).*member;
  }
};

TEST_F(DynamicTypeTest, MemberPointer) {
  using ABCD = DynamicType<NoContainers, A, B, CD>;
  constexpr ABCD a = A{1, 2};
  static_assert(a->*&A::x == 1);
  static_assert(a->*&A::y == 2);
  constexpr ABCD b = B{3, 4};
  static_assert(b->*&B::x == 3);
  static_assert(b->*&B::y == 4);
  constexpr ABCD c = CD{C{5, 6}};
#if __cplusplus >= 202002L
  static_assert(c->*&C::x == 5);
  static_assert(c->*&C::y == 6);
#else
  EXPECT_EQ(c->*&C::x, 5);
  EXPECT_EQ(c->*&C::y, 6);
#endif
  constexpr ABCD d = CD{D{7, 8}};
#if __cplusplus >= 202002L
  static_assert(d->*&D::x == 7);
  static_assert(d->*&D::y == 8);
#else
  EXPECT_EQ(d->*&D::x, 7);
  EXPECT_EQ(d->*&D::y, 8);
#endif
  static_assert(opcheck<ABCD>->*opcheck<int A::*>);
  static_assert(opcheck<ABCD>->*opcheck<int B::*>);
  static_assert(opcheck<ABCD>->*opcheck<int C::*>);
  static_assert(opcheck<ABCD>->*opcheck<int D::*>);
  static_assert(!(opcheck<ABCD>->*opcheck<int E::*>));

  ABCD aa = a;
  EXPECT_EQ(aa->*&A::x, 1);
  EXPECT_EQ(aa->*&A::y, 2);
  aa->*& A::x = 299792458;
  aa->*& A::y = 314159;
  EXPECT_EQ(aa->*&A::x, 299792458);
  EXPECT_EQ(aa->*&A::y, 314159);

  ABCD cc = c;
  EXPECT_EQ(cc->*&C::x, 5);
  EXPECT_EQ(cc->*&C::y, 6);
  cc->*& C::x = 299792458;
  cc->*& C::y = 314159;
  EXPECT_EQ(cc->*&C::x, 299792458);
  EXPECT_EQ(cc->*&C::y, 314159);
}

struct F {
  int x;
  int y;
  constexpr const int& operator->*(std::string_view member) const {
    if (member == "x") {
      return x;
    } else if (member == "y") {
      return y;
    } else {
      throw std::runtime_error("invalid member");
    }
  }
  constexpr int& operator->*(std::string_view member) {
    if (member == "x") {
      return x;
    } else if (member == "y") {
      return y;
    } else {
      throw std::runtime_error("invalid member");
    }
  }
};

struct G : public F {};

TEST_F(DynamicTypeTest, NonMemberPointerArrowStarRef) {
  using EFG = DynamicType<NoContainers, E, F, G>;

  constexpr EFG f = F{1, 2};
#if __cplusplus >= 202002L
  static_assert(f->*"x" == 1);
  static_assert(f->*"y" == 2);
#else
  EXPECT_EQ(f->*"x", 1);
  EXPECT_EQ(f->*"y", 2);
#endif

  constexpr EFG g = G{3, 4};
#if __cplusplus >= 202002L
  static_assert(g->*"x" == 3);
  static_assert(g->*"y" == 4);
#else
  EXPECT_EQ(g->*"x", 3);
  EXPECT_EQ(g->*"y", 4);
#endif

  static_assert(opcheck<EFG>->*opcheck<std::string_view>);
  static_assert(!(opcheck<EFG>->*opcheck<int>));

  EFG ff = f;
  EXPECT_EQ(ff->*"x", 1);
  EXPECT_EQ(ff->*"y", 2);
  ff->*"x" = 299792458;
  ff->*"y" = 314159;
  EXPECT_EQ(ff->*"x", 299792458);
  EXPECT_EQ(ff->*"y", 314159);
}

class ConstAccessor {
  std::function<int()> getter_;

 public:
  ConstAccessor(std::function<int()> getter) : getter_(getter) {}

  operator int() const {
    return getter_();
  }
};

class Accessor {
  std::function<int()> getter_;
  std::function<void(int)> setter_;

 public:
  Accessor(std::function<int()> getter, std::function<void(int)> setter)
      : getter_(getter), setter_(setter) {}

  const Accessor& operator=(int value) const {
    setter_(value);
    return *this;
  }
  operator int() const {
    return getter_();
  }
};

struct H {
  int x;
  int y;
  ConstAccessor operator->*(std::string_view member) const {
    if (member == "x") {
      return ConstAccessor{[this]() { return x; }};
    } else if (member == "y") {
      return ConstAccessor{[this]() { return y; }};
    } else {
      throw std::runtime_error("invalid member");
    }
  }
  Accessor operator->*(std::string_view member) {
    if (member == "x") {
      return Accessor{[this]() { return x; }, [this](int value) { x = value; }};
    } else if (member == "y") {
      return Accessor{[this]() { return y; }, [this](int value) { y = value; }};
    } else {
      throw std::runtime_error("invalid member");
    }
  }
};

struct I : public H {};

TEST_F(DynamicTypeTest, NonMemberPointerArrowStaAccessor) {
  using EHI = DynamicType<NoContainers, E, H, I>;

  EHI h = H{1, 2};
  EXPECT_EQ(h->*"x", 1);
  EXPECT_EQ(h->*"y", 2);

  EHI i = I{3, 4};
  EXPECT_EQ(i->*"x", 3);
  EXPECT_EQ(i->*"y", 4);

  static_assert(opcheck<EHI>->*opcheck<std::string_view>);
  static_assert(!(opcheck<EHI>->*opcheck<int>));

  EHI hh = h;
  EXPECT_EQ(hh->*"x", 1);
  EXPECT_EQ(hh->*"y", 2);
  hh->*"x" = 299792458;
  hh->*"y" = 314159;
  EXPECT_EQ(hh->*"x", 299792458);
  EXPECT_EQ(hh->*"y", 314159);
}

TEST_F(DynamicTypeTest, MemberFunctions) {
  struct J {
    constexpr std::string_view no_qualifiers() {
      return "no qualifiers";
    }

    constexpr std::string_view const_qualifiers() const {
      return "const qualifiers";
    }

    constexpr std::string_view volatile_qualifiers() volatile {
      return "volatile qualifiers";
    }

    constexpr std::string_view const_volatile_qualifiers() const volatile {
      return "const volatile qualifiers";
    }

    constexpr std::string_view lvalue_ref_qualifiers() & {
      return "lvalue ref qualifiers";
    }

    constexpr std::string_view const_lvalue_ref_qualifiers() const& {
      return "const lvalue ref qualifiers";
    }

    constexpr std::string_view volatile_lvalue_ref_qualifiers() volatile& {
      return "volatile lvalue ref qualifiers";
    }

    constexpr std::string_view noexcept_qualifiers() noexcept {
      return "noexcept qualifiers";
    }

    constexpr std::string_view noexcept_false_qualifiers() noexcept(false) {
      return "noexcept(false) qualifiers";
    }

    constexpr std::string_view noexcept_true_qualifiers() noexcept(true) {
      return "noexcept(true) qualifiers";
    }

    constexpr int two_arguments(int a, int b) const {
      return a + b;
    }

    constexpr int three_arguments(int a, int b, int c) const {
      return a + b + c;
    }
  };

  using EJ = DynamicType<NoContainers, E, J>;
  constexpr EJ j = J{};
  static_assert((j->*&J::const_qualifiers)() == "const qualifiers");
  static_assert(
      (j->*&J::const_volatile_qualifiers)() == "const volatile qualifiers");
  static_assert(
      (j->*&J::const_lvalue_ref_qualifiers)() == "const lvalue ref qualifiers");
  static_assert((j->*&J::two_arguments)(10, 2) == 12);
  static_assert((j->*&J::three_arguments)(10, 2, 300) == 312);

  // Not using static_assert below because we can not call functions without
  // const qualifier in the constant evaluation context
  EJ jj = j;
  EXPECT_EQ((jj->*&J::no_qualifiers)(), "no qualifiers");
  EXPECT_EQ((jj->*&J::volatile_qualifiers)(), "volatile qualifiers");
  EXPECT_EQ((jj->*&J::lvalue_ref_qualifiers)(), "lvalue ref qualifiers");
  EXPECT_EQ(
      (jj->*&J::volatile_lvalue_ref_qualifiers)(),
      "volatile lvalue ref qualifiers");
  EXPECT_EQ((jj->*&J::noexcept_qualifiers)(), "noexcept qualifiers");
  EXPECT_EQ(
      (jj->*&J::noexcept_false_qualifiers)(), "noexcept(false) qualifiers");
  EXPECT_EQ((jj->*&J::noexcept_true_qualifiers)(), "noexcept(true) qualifiers");
}

} // namespace member_pointer_test

} // namespace dynamic_type
