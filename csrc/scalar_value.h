// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <macros.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <dynamic_type.h>
#include <complex>
#include <cstddef>
#include <unordered_map>

#include <type.h>

namespace nvfuser {

template <typename T>
struct Struct {
  // In theory, we should just use std::unordered_map<std::string, T>, but this
  // doesn't work on old gcc. See also SetTheoreticNaturalNumbers
#if defined(STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE)
  std::unordered_map<std::string, T> fields;
#define MAYBE_STAR
#else
  std::unordered_map<std::string, std::shared_ptr<T>> fields;
#define MAYBE_STAR *
#endif

  const T& operator[](const std::string& key) const {
    return MAYBE_STAR fields.at(key);
  }

  T& operator[](const std::string& key) {
#if defined(STD_UNORDERED_SET_SUPPORTS_INCOMPLETE_TYPE)
    return fields[key];
#else
    if (fields.find(key) == fields.end()) {
      fields[key] = std::make_shared<T>();
    }
    return *fields.at(key);
#endif
  }

#undef MAYBE_STAR
};

// Use a single pointer type to represent all pointers, otherwise we would need
// exponential compilation time for all pointer types in ScalarValue.
class Pointer {
  std::byte* ptr_;
  size_t size_;

 public:
  template <typename T>
  Pointer(T* ptr) : ptr_(reinterpret_cast<std::byte*>(ptr)), size_(sizeof(T)) {}

  Pointer(void* ptr, DataType dtype)
      : ptr_(reinterpret_cast<std::byte*>(ptr)), size_(dataTypeSize(dtype)) {}

  template <typename T>
  explicit operator T*() const {
    TORCH_INTERNAL_ASSERT(size_ == sizeof(T));
    return static_cast<T*>(ptr_);
  }

  Pointer& operator+=(int64_t offset) {
    ptr_ += offset * size_;
    return *this;
  }

  Pointer& operator-=(int64_t offset) {
    ptr_ -= offset * size_;
    return *this;
  }

  Pointer& operator++() {
    ptr_ += size_;
    return *this;
  }

  Pointer& operator--() {
    ptr_ -= size_;
    return *this;
  }

  Pointer operator++(int) {
    Pointer tmp = *this;
    ++*this;
    return tmp;
  }

  Pointer operator--(int) {
    Pointer tmp = *this;
    --*this;
    return tmp;
  }

  Pointer operator+(int64_t offset) const {
    Pointer tmp = *this;
    tmp += offset;
    return tmp;
  }

  Pointer operator-(int64_t offset) const {
    Pointer tmp = *this;
    tmp -= offset;
    return tmp;
  }

  int64_t operator-(const Pointer& other) const {
    TORCH_INTERNAL_ASSERT(size_ == other.size_);
    return (ptr_ - other.ptr_) / (int64_t)size_;
  }

  bool operator==(const Pointer& other) const {
    return ptr_ == other.ptr_;
  }

  bool operator==(std::nullptr_t) const {
    return ptr_ == nullptr;
  }

  bool operator!=(const Pointer& other) const {
    return ptr_ != other.ptr_;
  }

  bool operator!=(std::nullptr_t) const {
    return ptr_ != nullptr;
  }

  bool operator<(const Pointer& other) const {
    return ptr_ < other.ptr_;
  }

  bool operator>(const Pointer& other) const {
    return ptr_ > other.ptr_;
  }

  bool operator<=(const Pointer& other) const {
    return ptr_ <= other.ptr_;
  }

  bool operator>=(const Pointer& other) const {
    return ptr_ >= other.ptr_;
  }

  bool operator!() const {
    return !ptr_;
  }

  explicit operator bool() const {
    return ptr_;
  }
};

inline Pointer operator+(int64_t offset, const Pointer& ptr) {
  return ptr + offset;
}

using ScalarValue = DynamicType<
    Containers<std::vector, Struct>,
    std::complex<double>,
    double,
    int64_t,
    bool,
    Pointer>;

namespace ScalarValue_functions {

inline ScalarValue ceildiv(const ScalarValue& a, const ScalarValue& b) {
  if (a.is<int64_t>() && b.is<int64_t>()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return ScalarValue((aa + bb - 1) / bb);
    } else {
      return ScalarValue((aa + bb + 1) / bb);
    }
  }
  return ScalarValue(std::ceil((a / b).as<double>()));
}

inline ScalarValue max(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(a > b ? a : b);
}

inline ScalarValue min(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(a < b ? a : b);
}

inline ScalarValue gcd(const ScalarValue& a, const ScalarValue& b) {
  return ScalarValue(std::gcd(a.as<int64_t>(), b.as<int64_t>()));
}

inline ScalarValue notExpr(const ScalarValue& a) {
  if (a.is<int64_t>()) {
    return ScalarValue(~a.as<int64_t>());
  }
  if (a.is<bool>()) {
    return ScalarValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(false);
}

inline ScalarValue abs(const ScalarValue& a) {
  if (a.is<int64_t>()) {
    return ScalarValue(std::abs(a.as<int64_t>()));
  }
  if (a.is<double>()) {
    return ScalarValue(std::abs(a.as<double>()));
  }
  if (a.is<bool>()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(false);
}

} // namespace ScalarValue_functions

} // namespace nvfuser
