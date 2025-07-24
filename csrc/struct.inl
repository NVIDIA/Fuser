// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <visibility.h>
#include <functional>
#include <memory>
#include <string>

namespace nvfuser {

// Note [Struct Support in PolymorphicValue]
//
// PolymorphicValue supports structs, which is just a list of named fields. The
// most straightforward way to support structs is to use a map from field name
// to value, something like:
//   template <typename T>
//   using Struct = std::unordered_map<std::string, T>;
//   using PolymorphicValue = DynamicType<Containers<Struct>, ...>;
// However, the performance of this approach is not ideal. So instead of making
// the struct support truly dynamic fields by using a map, we decide to make it
// semi-dynamic: each struct type in nvFuser must be backed by a real struct in
// C++, which mean, the fields have static storage types. But, on the other
// hand, struct fields can also be accessed dynamically, that is, you can get or
// set a struct field without knowing the actual C++ struct and the type of the
// field. Instead, by using solely the string name of the field, you shall be
// able to access fields as a PolymorphicValue. For example, if your struct is
// defined as:
//   struct A { int64_t x; double y; };
//   PolymorphicValue v = some struct of type A;
// Then you can access the fields statically like:
//   const int64_t& x = v->*&A::x;
//   v->*&A::x = 1;
// Static accesses should be very efficient, as fast as dynamic casts + pointer
// dereferences. However, if you don't have access to the definition of `A`, you
// can still access the fields dynamically:
//   PolymorphicValue x = v->*"x";
//   v->*"x" = 1;
// Dynamic accesses are slower than static accesses, because you need to do
// string comparisons to find the field, and do casts between the actual field
// type and PolymorphicValue. This can be slow especially when the struct has
// some fields of containers like std::vector<int64_t>, because you need to do
// the conversion between std::vector<PolymorphicValue> and std::vector<int64_t>
// every time you get or set a field.
//
// The implementation of this feature requires a few components working
// together:
// 1. StructType: a data type that describes the name and fields of a struct.
//    More importantly, it stores a function that can create an instance of a
//    struct without requiring the caller to know the actual struct type.
// 2. Struct: a base class for all structs, which provides the virtual interface
//    for accessing fields dynamically, as well as an interface for getting the
//    StructType of the struct.
// 3. StructHandle: a wrapper around Struct, which maintains the ownership of
//    struct objects and provides the overloaded ->* operator for accessing
//    fields statically and dynamically. StructHandle is a candidate type for
//    PolymorphicValue.
// 4. Accessor: a helper class returned by the dynamic ->* operator, which
//    provides the overloaded casting to PolymorphicValue and = operator for
//    getting and setting fields dynamically.
//
// With the above components, define a struct type that supports dynamic access
// to fields is basically subclassing Struct and implementing the virtual
// methods. Please check the test PolymorphicValueTest.Struct for an example.

struct Struct {
  virtual ~Struct() = default;

  virtual StructType type() const = 0;
  virtual std::function<PolymorphicValue()> getter(
      const std::string& key) const = 0;
  virtual std::function<void(const PolymorphicValue&)> setter(
      const std::string& key) = 0;
};

class Accessor {
  std::function<PolymorphicValue()> getter_;
  std::function<void(const PolymorphicValue&)> setter_;

 public:
  Accessor(
      std::function<PolymorphicValue()> getter,
      std::function<void(const PolymorphicValue&)> setter)
      : getter_(std::move(getter)), setter_(std::move(setter)) {}
  Accessor(const Accessor& value) = default;
  Accessor(Accessor&& value) = default;
  Accessor& operator=(const Accessor& value) = default;
  Accessor& operator=(Accessor&& value) = default;

  inline const Accessor& operator=(const PolymorphicValue& value) const {
    setter_(std::move(value));
    return *this;
  }

  inline operator PolymorphicValue() const {
    return getter_();
  }
};

inline Accessor StructHandle::operator->*(const std::string& key) const {
  return Accessor(struct_ptr_->getter(key), struct_ptr_->setter(key));
}

// If a struct type is only used in kernel and we will never create an instance
// on the host, we can just use this dummy struct as a placeholder for the
// convenience
struct NVF_API NotImplementedStruct : public Struct {
  StructType type() const override;

  std::function<PolymorphicValue()> getter(
      const std::string& key) const override {
    NVF_THROW("Not implemented");
  }

  std::function<void(const PolymorphicValue&)> setter(
      const std::string& key) override {
    NVF_THROW("Not implemented");
  }
};

} // namespace nvfuser
