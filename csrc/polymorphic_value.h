// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <macros.h>

#include <dynamic_type.h>
#include <any>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ostream>
#include <unordered_map>

#include <ATen/ATen.h>

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

struct DataType;

// Use a single pointer type to represent all pointers, otherwise we would need
// exponential compilation time for all pointer types in PolymorphicValue.
class Pointer {
  std::byte* ptr_;
  int64_t size_;

 public:
  template <typename T>
  Pointer(T* ptr) : ptr_(reinterpret_cast<std::byte*>(ptr)), size_(sizeof(T)) {}

  inline Pointer(void* ptr, DataType dtype);

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

  explicit operator int64_t() const {
    return reinterpret_cast<int64_t>(ptr_);
  }

  explicit operator unsigned() const {
    return (unsigned)(int64_t)(*this);
  }
};

inline Pointer operator+(int64_t offset, const Pointer& ptr) {
  return ptr + offset;
}

struct Opaque {
  std::any value;

  // Because the type information is not available at compile time, we can't
  // accurately compare the values of two opaque values. So, by default,
  // equality check is done by pointer compare. However, we also support
  // manually specifying the equality comparator.
  std::function<bool(const Opaque&, const Opaque&)> equals =
      [](const Opaque& a, const Opaque& b) { return &a == &b; };

  bool operator==(const Opaque& other) const {
    if (this == &other) {
      return true;
    }
    if (value.type() != other.value.type()) {
      return false;
    }
    bool result1 = equals(*this, other);
    bool result2 = equals(other, *this);
    TORCH_INTERNAL_ASSERT(
        result1 == result2, "Opaque equality is not symmetric");
    return result1;
  }

  bool operator!=(const Opaque& other) const {
    return !(*this == other);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Opaque& opaque) {
  os << "Opaque<" << opaque.value.type().name() << ">";
  return os;
}

template <typename T>
struct OpaqueEquals {
  bool operator()(const Opaque& a, const Opaque& b) const {
    return std::any_cast<T>(a.value) == std::any_cast<T>(b.value);
  }
};

using PolymorphicValue = DynamicType<
    Containers<std::vector, Struct>,
    Pointer,
    Opaque,
    at::Tensor,
    std::complex<double>,
    double,
    int64_t,
    bool>;

namespace PolymorphicValue_functions {

inline PolymorphicValue ceildiv(
    const PolymorphicValue& a,
    const PolymorphicValue& b) {
  if (a.is<int64_t>() && b.is<int64_t>()) {
    auto aa = a.as<int64_t>();
    auto bb = b.as<int64_t>();
    if (bb > 0) {
      return PolymorphicValue((aa + bb - 1) / bb);
    } else {
      return PolymorphicValue((aa + bb + 1) / bb);
    }
  }
  return PolymorphicValue(std::ceil((a / b).as<double>()));
}

inline PolymorphicValue max(
    const PolymorphicValue& a,
    const PolymorphicValue& b) {
  return PolymorphicValue(a > b ? a : b);
}

inline PolymorphicValue min(
    const PolymorphicValue& a,
    const PolymorphicValue& b) {
  return PolymorphicValue(a < b ? a : b);
}

inline PolymorphicValue gcd(
    const PolymorphicValue& a,
    const PolymorphicValue& b) {
  return PolymorphicValue(std::gcd(a.as<int64_t>(), b.as<int64_t>()));
}

inline PolymorphicValue notExpr(const PolymorphicValue& a) {
  if (a.is<int64_t>()) {
    return PolymorphicValue(~a.as<int64_t>());
  }
  if (a.is<bool>()) {
    return PolymorphicValue(!a.as<bool>());
  }
  TORCH_INTERNAL_ASSERT(
      false, "PolymorphicValue notExpr not implemented for ", a.type().name());
}

inline PolymorphicValue abs(const PolymorphicValue& a) {
  if (a.is<int64_t>()) {
    return PolymorphicValue(std::abs(a.as<int64_t>()));
  }
  if (a.is<double>()) {
    return PolymorphicValue(std::abs(a.as<double>()));
  }
  if (a.is<bool>()) {
    return a;
  }
  TORCH_INTERNAL_ASSERT(
      false, "PolymorphicValue abs not implemented for ", a.type().name());
}

inline PolymorphicValue erf(const PolymorphicValue& a) {
  if (a.is<at::Tensor>()) {
    return PolymorphicValue(a.as<at::Tensor>().erf());
  }
  TORCH_INTERNAL_ASSERT(
      false, "PolymorphicValue erf not implemented for ", a.type().name());
}

// Convert scalars, vector of scalars, vector of vector of scalars, etc., into
// an at::Tensor
inline PolymorphicValue toTensor(const PolymorphicValue& x) {
  if (x.is<at::Tensor>()) {
    return x;
  }
  // TODO: allow specifying device
  auto options = at::TensorOptions().device(at::kCUDA, 0);
  if (x.is<int64_t>()) {
    return PolymorphicValue(
        at::tensor(x.as<int64_t>(), options.dtype(at::kLong)).squeeze());
  }
  if (x.is<double>()) {
    return PolymorphicValue(
        at::tensor(x.as<double>(), options.dtype(at::kDouble)).squeeze());
  }
  if (x.is<bool>()) {
    return PolymorphicValue(
        at::tensor(x.as<bool>(), options.dtype(at::kBool)).squeeze());
  }
  if (x.is<std::complex<double>>()) {
    return PolymorphicValue(
        at::tensor(
            (c10::complex<double>)x.as<std::complex<double>>(),
            options.dtype(at::kComplexDouble))
            .squeeze());
  }
  if (x.is<std::vector>()) {
    auto vec = x.as<std::vector>();
    std::vector<at::Tensor> tensors;
    tensors.reserve(vec.size());
    for (const auto& elem : vec) {
      tensors.push_back(toTensor(elem).as<at::Tensor>());
    }
    return PolymorphicValue(at::stack(tensors));
  }
  TORCH_INTERNAL_ASSERT(
      false, "PolymorphicValue toTensor not implemented for ", x.type().name());
}

} // namespace PolymorphicValue_functions

} // namespace nvfuser
