// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <memory>

#include "dynamic_type/dynamic_type.h"

using namespace dynamic_type;

class MemberTest : public ::testing::Test {};

struct MemberA {
  int x;
  int y;
};
struct MemberB {
  int x;
  int y;
};
struct MemberC {
  int x;
  int y;
};
struct MemberD {
  int x;
  int y;
};
struct MemberE {
  int x;
  int y;
};

struct MemberCD {
  std::variant<MemberC, MemberD> v;

  constexpr const int& operator->*(int MemberC::* member) const {
    return std::get<MemberC>(v).*member;
  }

  constexpr const int& operator->*(int MemberD::* member) const {
    return std::get<MemberD>(v).*member;
  }

  constexpr int& operator->*(int MemberC::* member) {
    return std::get<MemberC>(v).*member;
  }

  constexpr int& operator->*(int MemberD::* member) {
    return std::get<MemberD>(v).*member;
  }
};

TEST_F(MemberTest, MemberPointer) {
  using ABCD = DynamicType<NoContainers, MemberA, MemberB, MemberCD>;
  constexpr ABCD a = MemberA{1, 2};
  static_assert(a->*&MemberA::x == 1);
  static_assert(a->*&MemberA::y == 2);
  constexpr ABCD b = MemberB{3, 4};
  static_assert(b->*&MemberB::x == 3);
  static_assert(b->*&MemberB::y == 4);
  constexpr ABCD c = MemberCD{MemberC{5, 6}};
#if __cplusplus >= 202002L
  static_assert(c->*&MemberC::x == 5);
  static_assert(c->*&MemberC::y == 6);
#else
  EXPECT_EQ(c->*&MemberC::x, 5);
  EXPECT_EQ(c->*&MemberC::y, 6);
#endif
  constexpr ABCD d = MemberCD{MemberD{7, 8}};
#if __cplusplus >= 202002L
  static_assert(d->*&MemberD::x == 7);
  static_assert(d->*&MemberD::y == 8);
#else
  EXPECT_EQ(d->*&MemberD::x, 7);
  EXPECT_EQ(d->*&MemberD::y, 8);
#endif
  static_assert(opcheck<ABCD>->*opcheck<int MemberA::*>);
  static_assert(opcheck<ABCD>->*opcheck<int MemberB::*>);
  static_assert(opcheck<ABCD>->*opcheck<int MemberC::*>);
  static_assert(opcheck<ABCD>->*opcheck<int MemberD::*>);
  static_assert(!(opcheck<ABCD>->*opcheck<int MemberE::*>));

  ABCD aa = a;
  EXPECT_EQ(aa->*&MemberA::x, 1);
  EXPECT_EQ(aa->*&MemberA::y, 2);
  aa->*& MemberA::x = 299792458;
  aa->*& MemberA::y = 314159;
  EXPECT_EQ(aa->*&MemberA::x, 299792458);
  EXPECT_EQ(aa->*&MemberA::y, 314159);

  ABCD cc = c;
  EXPECT_EQ(cc->*&MemberC::x, 5);
  EXPECT_EQ(cc->*&MemberC::y, 6);
  cc->*& MemberC::x = 299792458;
  cc->*& MemberC::y = 314159;
  EXPECT_EQ(cc->*&MemberC::x, 299792458);
  EXPECT_EQ(cc->*&MemberC::y, 314159);
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

TEST_F(MemberTest, NonMemberPointerArrowStarRef) {
  using EFG = DynamicType<NoContainers, MemberE, F, G>;

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

TEST_F(MemberTest, NonMemberPointerArrowStaAccessor) {
  using EHI = DynamicType<NoContainers, MemberE, H, I>;

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

TEST_F(MemberTest, MemberFunctions) {
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

  using EJ = DynamicType<NoContainers, MemberE, J>;
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

TEST_F(MemberTest, ArrowOp) {
  int num_dtor_calls = 0;
  struct S {
    int aaa;
    int& num_dtor_calls;
    S(int aaa, int& num_dtor_calls)
        : aaa(aaa), num_dtor_calls(num_dtor_calls) {}
    ~S() {
      num_dtor_calls++;
    };
  } s(12, num_dtor_calls);
  EXPECT_EQ(num_dtor_calls, 0);
  using IntSVec = DynamicType<Containers<std::vector>, int, S*>;
  IntSVec x(&s);
  EXPECT_EQ(x->aaa, 12);

  using Pointer =
      DynamicType<NoContainers, S*, std::shared_ptr<S>, std::unique_ptr<S>>;
  S s1(34, num_dtor_calls);
  auto s2 = std::make_shared<S>(56, num_dtor_calls);
  auto s3 = std::make_unique<S>(78, num_dtor_calls);
  Pointer ptr;
  ptr = &s1;
  EXPECT_EQ(ptr->aaa, 34);
  ptr = s2;
  EXPECT_EQ(ptr->aaa, 56);
  s2 = nullptr;
  EXPECT_EQ(num_dtor_calls, 0);
  ptr = std::move(s3);
  EXPECT_EQ(num_dtor_calls, 1);
  EXPECT_EQ(ptr->aaa, 78);
  EXPECT_EQ(num_dtor_calls, 1);
  ptr = {};
  EXPECT_EQ(num_dtor_calls, 2);
}
