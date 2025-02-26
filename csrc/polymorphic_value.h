// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <python_frontend/distributed_tensor.h>
#include <any>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ostream>
#include <unordered_map>

#include <ATen/ATen.h>

#ifndef DYNAMIC_TYPE_CHECK
#define DYNAMIC_TYPE_CHECK NVF_ERROR
#endif

#include <dynamic_type/dynamic_type.h>
#include <macros.h>
#include <opaque_type.h>

namespace nvfuser {

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

  Pointer() : ptr_(nullptr), size_(-1) {}

  int64_t size() const {
    return size_;
  }

  template <typename T>
  explicit operator T*() const {
    return reinterpret_cast<T*>(ptr_);
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
    NVF_ERROR(size_ == other.size_);
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

  explicit operator size_t() const {
    return reinterpret_cast<size_t>(ptr_);
  }
};

inline Pointer operator+(int64_t offset, const Pointer& ptr) {
  return ptr + offset;
}

inline std::ostream& operator<<(std::ostream& os, const Pointer& ptr) {
  os << (void*)ptr;
  return os;
}

struct Struct;
class Accessor;
struct StructType;

// See Note [Struct Support in PolymorphicValue] for documentation.
class StructHandle {
  std::shared_ptr<Struct> struct_ptr_;

 public:
  StructHandle(std::shared_ptr<Struct> struct_ptr)
      : struct_ptr_(std::move(struct_ptr)) {}
  StructHandle& operator=(std::shared_ptr<Struct> struct_ptr) {
    struct_ptr_ = std::move(struct_ptr);
    return *this;
  }

  StructHandle(const StructHandle& other) = default;
  StructHandle(StructHandle&& other) = default;
  StructHandle& operator=(const StructHandle& other) = default;
  StructHandle& operator=(StructHandle&& other) = default;

  bool operator==(const StructHandle& other) const;

  template <typename T>
  bool is() const {
    return std::dynamic_pointer_cast<T>(struct_ptr_) != nullptr;
  }

  template <typename T>
  inline T& as() const {
    return *std::dynamic_pointer_cast<T>(struct_ptr_);
  }

  inline StructType type() const;

  template <typename Ret, typename Class>
  inline std::enable_if_t<std::is_base_of_v<Struct, Class>, Ret&> operator->*(
      Ret Class::* member) const {
    return as<Class>().*member;
  }

  inline Accessor operator->*(const std::string& key) const;
};

using PolymorphicValue = dynamic_type::DynamicType<
    dynamic_type::Containers<std::vector>,
    StructHandle,
    Pointer,
    Opaque,
    python_frontend::DistributedTensor,
    at::Tensor,
    std::complex<double>,
    double,
    int64_t,
    bool>;

namespace PolymorphicValue_functions {

NVF_API std::string toString(const PolymorphicValue& v);

template <typename T>
inline bool isNan(const T& a) {
  return std::isnan(a);
}

// For example, `nan+i` and `nan-i` are treated equal because both are NaNs.
// This is consistent with pytorch's implementation:
// https://github.com/pytorch/pytorch/blob/6d8e0c4b5a3be8201cab731dfd1e6513162cf25c/c10/util/complex_utils.h#L43.
template <typename T>
inline bool isNan(const std::complex<T>& a) {
  return std::isnan(a.real()) || std::isnan(a.imag());
}

// NaNs are treated equal.
template <typename T>
inline bool isSameNanSensitive(const T& a, const T& b) {
  if (isNan(a) && isNan(b)) {
    return true;
  }
  return a == b;
}

bool isSame(const PolymorphicValue& a, const PolymorphicValue& b);

// Convert scalars, vector of scalars, vector of vector of scalars, etc., into
// an at::Tensor. device argument allows for the creation of CPU Scalars.
PolymorphicValue toTensor(
    const PolymorphicValue& x,
    at::DeviceType device_type = at::kCUDA,
    int8_t device_index = 0);

constexpr bool isScalar(const PolymorphicValue& x) {
  return x.is<int64_t>() || x.is<double>() || x.is<bool>() ||
      x.is<std::complex<double>>();
}

// Convert PolymorphicValue to c10::Scalar.
inline c10::Scalar toScalar(const PolymorphicValue& x) {
  if (x.is<double>()) {
    return (c10::Scalar)x.as<double>();
  } else if (x.is<int64_t>()) {
    return (c10::Scalar)x.as<int64_t>();
  } else if (x.is<bool>()) {
    return (c10::Scalar)x.as<bool>();
  } else if (x.is<std::complex<double>>()) {
    return (c10::complex<double>)x.as<std::complex<double>>();
  }
  NVF_THROW("Cannot convert ", x, " to a scalar.");
}

PolymorphicValue IValueToPolymorphicValue(const c10::IValue& val);

c10::IValue toIValue(const PolymorphicValue& x);

PolymorphicValue castToDtype(PolymorphicValue value, const DataType& dtype);

} // namespace PolymorphicValue_functions

} // namespace nvfuser

#include <struct.inl>
