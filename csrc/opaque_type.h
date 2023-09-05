// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <dynamic_type/type_traits.h>

#include <any>
#include <cstddef>
#include <cstring>
#include <functional>
#include <ostream>

namespace nvfuser {

class Opaque;

template <typename T>
struct OpaqueEquals {
  bool operator()(const Opaque& a, const Opaque& b) const;
};

template <typename T>
struct OpaqueToBytes {
  std::vector<std::byte> operator()(const Opaque& a) const;
};

class Opaque {
  std::any value_;
  std::function<bool(const Opaque&, const Opaque&)> equals_;
  std::function<std::vector<std::byte>(const Opaque&)> to_bytes_;
  size_t size_;

 public:
  template <typename T>
  explicit Opaque(T value)
      : value_(std::move(value)),
        equals_(OpaqueEquals<T>{}),
        to_bytes_(OpaqueToBytes<T>{}),
        size_(sizeof(T)) {}

  bool operator==(const Opaque& other) const {
    if (this == &other) {
      return true;
    }
    if (value_.type() != other.value_.type()) {
      // Note that because C++ is a statically typed language, there is no way
      // to completely accurately compare equality of opaque values. The
      // behavior here is just an approximation. For example 1 == 1.0 but
      // Opaque(1) != Opaque(1.0).
      return false;
    }
    return equals_(*this, other);
  }

  bool operator!=(const Opaque& other) const {
    return !(*this == other);
  }

  const std::any& any() const {
    return value_;
  }

  template <typename T>
  const T& as() const {
    return std::any_cast<const T&>(value_);
  }

  template <typename T>
  T& as() {
    return std::any_cast<T&>(value_);
  }

  std::vector<std::byte> bytes() const {
    return to_bytes_(*this);
  }

  size_t size() const {
    return size_;
  }
};

template <typename T>
bool OpaqueEquals<T>::operator()(const Opaque& a, const Opaque& b) const {
  if constexpr (dynamic_type::opcheck<T> == dynamic_type::opcheck<T>) {
    // If T == T exists, use it
    return a.as<T>() == b.as<T>();
  } else {
    // Otherwise, do bitwise compare. Note that bitwise comparison is not always
    // correct. So this is only an approximation. For example:
    //   struct A {
    //     int64_t x;
    //     std::vector<float> y;
    //   };
    //   Opaque(A{1, {2.0}}) != Opaque(A{1, {2.0}});
    // Another example:
    //   struct A {
    //     int32_t i;
    //     double d;
    //   };
    //   /*maybe:*/ Opaque(A{1, 2.0}) == Opaque(A{1, 2.0});
    //   /*maybe:*/ Opaque(A{1, 2.0}) != Opaque(A{1, 2.0});
    // Because the struct is not packed, usually C++ compiler will allocate A as
    // something like below:
    // [=== i (32bits) ===][=== empty (32bits) ===][====== d (64bits) ======]
    // The padding bits are not initialized and can be different between two
    // instances of A. So the comparison result is not even deterministic.
    // This path should only be used for packed POD structs. For other types,
    // the user should provide an overloaded operator==.
    return std::memcmp(&a.as<T>(), &b.as<T>(), sizeof(T)) == 0;
  }
}

template <typename T>
std::vector<std::byte> OpaqueToBytes<T>::operator()(const Opaque& a) const {
  return std::vector<std::byte>(
      (const std::byte*)&a.as<T>(), (const std::byte*)(&a.as<T>() + 1));
}

inline std::ostream& operator<<(std::ostream& os, const Opaque& opaque) {
  os << "Opaque<" << opaque.any().type().name() << ">";
  return os;
}

} // namespace nvfuser
