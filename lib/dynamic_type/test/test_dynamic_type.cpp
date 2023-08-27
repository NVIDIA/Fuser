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

namespace dynamic_type {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-comparison"
#pragma clang diagnostic ignored "-Wbitwise-instead-of-logical"
#pragma clang diagnostic ignored "-Wliteral-conversion"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

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
