// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <any>
#include <functional>
#include <ostream>

namespace nvfuser {

class Opaque;

template <typename T>
struct OpaqueEquals {
  bool operator()(const Opaque& a, const Opaque& b) const;
};

class Opaque {
  std::any value_;
  std::function<bool(const Opaque&, const Opaque&)> equals_;

 public:
  template <typename T>
  explicit Opaque(T value)
      : value_(std::move(value)), equals_(OpaqueEquals<T>{}) {}

  bool operator==(const Opaque& other) const {
    if (this == &other) {
      return true;
    }
    if (value_.type() != other.value_.type()) {
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
};

template <typename T>
bool OpaqueEquals<T>::operator()(const Opaque& a, const Opaque& b) const {
  return a.as<T>() == b.as<T>();
}

inline std::ostream& operator<<(std::ostream& os, const Opaque& opaque) {
  os << "Opaque<" << opaque.any().type().name() << ">";
  return os;
}

} // namespace nvfuser
