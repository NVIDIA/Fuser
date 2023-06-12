// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <csrc/exceptions.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/utils.h>

#include <polymorphic_value.h>
#include <type.h>

namespace nvfuser {

using dynamic_type::opcheck;

class PolymorphicValueTest : public NVFuserTest {};

TEST_F(PolymorphicValueTest, OpaqueEquality) {
  Opaque a{DataType::Int}, b{DataType::Int};
  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, a);
}

TEST_F(PolymorphicValueTest, Struct) {
  struct A : public Struct {
    int64_t x;
    double y;

    StructType type() const override {
      std::vector<StructType::FieldInfo> fields(2);
      fields.at(0) = {"x", std::make_shared<DataType>(DataType::Int), true};
      fields.at(1) = {"y", std::make_shared<DataType>(DataType::Double), false};
      return StructType::make<A>(fields, "A");
    }

    std::function<PolymorphicValue()> getter(
        const std::string& key) const override {
      if (key == "x") {
        return [this]() { return PolymorphicValue(x); };
      } else if (key == "y") {
        return [this]() { return PolymorphicValue(y); };
      } else {
        NVF_ERROR(false, "Invalid key");
      }
    }

    std::function<void(const PolymorphicValue&)> setter(
        const std::string& key) override {
      if (key == "x") {
        return [this](const PolymorphicValue& value) { x = (int64_t)value; };
      } else if (key == "y") {
        return [this](const PolymorphicValue& value) { y = (double)value; };
      } else {
        NVF_ERROR(false, "Invalid key");
      }
    }
  };

  static_assert(opcheck<PolymorphicValue>->*opcheck<int64_t A::*>);
  static_assert(opcheck<PolymorphicValue>->*opcheck<double A::*>);

  // In a "static context", i.e. we know the C++ type of the struct, we can
  // use pointer-to-member syntax to access fields. This is the most efficient
  // way to access fields. Accessing fields in this way will give references to
  // the fields, so the types of fields are also static.
  PolymorphicValue a = std::static_pointer_cast<Struct>(std::make_shared<A>());
  static_assert(std::is_same_v<decltype(a->*&A::x), int64_t&>);
  static_assert(std::is_same_v<decltype(a->*&A::y), double&>);
  a->*& A::x = 299792458;
  a->*& A::y = 3.1415926;
  EXPECT_EQ(a->*&A::x, 299792458);
  EXPECT_EQ(a->*&A::y, 3.1415926);
  a->*& A::x = 2788;
  a->*& A::y = 2.71828;
  EXPECT_EQ(a->*&A::x, 2788);
  EXPECT_EQ(a->*&A::y, 2.71828);

  StructType type = (a->*&StructHandle::type)();
  EXPECT_EQ(type.name, "A");
  EXPECT_EQ(type.fields.size(), 2);
  EXPECT_EQ(type.fields.at(0).name, "x");
  EXPECT_EQ(*type.fields.at(0).type, DataType::Int);
  EXPECT_TRUE(type.fields.at(0).used_in_kernel);
  EXPECT_EQ(type.fields.at(1).name, "y");
  EXPECT_EQ(*type.fields.at(1).type, DataType::Double);
  EXPECT_FALSE(type.fields.at(1).used_in_kernel);

  {
    // intentionally create a new scope and define another struct with the same
    // name to make sure the previous struct is not accessible
    struct A {
      int64_t x;
      double y;
    };
    static_assert(!(opcheck<PolymorphicValue>->*opcheck<int64_t A::*>));
    static_assert(!(opcheck<PolymorphicValue>->*opcheck<double A::*>));

    // In a "dynamic context", i.e. we don't know the C++ type of the struct, we
    // can use string keys to access fields, and PolymorphicValue for values.
    // This is less efficient than the static context because type conversions
    // and key checking are required, but it is the only way to access fields if
    // we don't know the C++ type of the struct.
    PolymorphicValue b = type.create();
    b->*"x" = 2788;
    b->*"y" = 2.71828;
    EXPECT_EQ((PolymorphicValue)(b->*"x"), 2788);
    EXPECT_EQ((PolymorphicValue)(b->*"y"), 2.71828);
    b->*"x" = 299792458;
    b->*"y" = 3.1415926;
    EXPECT_EQ((PolymorphicValue)(b->*"x"), 299792458);
    EXPECT_EQ((PolymorphicValue)(b->*"y"), 3.1415926);

    EXPECT_EQ(type, (b->*&StructHandle::type)());
  }
}

} // namespace nvfuser
