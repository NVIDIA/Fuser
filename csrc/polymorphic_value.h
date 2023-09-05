// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <exceptions.h>
#include <any>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ostream>
#include <unordered_map>

#include <ATen/ATen.h>

#define DYNAMIC_TYPE_CHECK NVF_ERROR

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
      Ret Class::*member) const {
    return as<Class>().*member;
  }

  inline Accessor operator->*(const std::string& key) const;
};

using PolymorphicValue = dynamic_type::DynamicType<
    dynamic_type::Containers<std::vector>,
    StructHandle,
    Pointer,
    Opaque,
    at::Tensor,
    std::complex<double>,
    double,
    int64_t,
    bool>;

namespace PolymorphicValue_functions {

inline std::string toString(const PolymorphicValue& v) {
  std::stringstream ss;
  if (v.is<at::Tensor>()) {
    const auto& t = v.as<at::Tensor>();
    ss << "Tensor(sizes=" << t.sizes() << ", "
       << "stride=" << t.strides() << ", " << t.dtype() << ", " << t.device()
       << ")";
  } else {
    ss << v;
  }
  return ss.str();
}

inline bool isSame(const PolymorphicValue& a, const PolymorphicValue& b) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.is<at::Tensor>() && b.is<at::Tensor>()) {
    return (a.as<at::Tensor>().is_same(b.as<at::Tensor>()));
  } else {
    return (a == b);
  }
}

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
  if (a.is<std::complex<double>>()) {
    return std::abs(a.as<std::complex<double>>());
  }
  NVF_ERROR(
      false, "PolymorphicValue abs not implemented for ", a.type().name());
}

inline PolymorphicValue erf(const PolymorphicValue& a) {
  if (a.is<at::Tensor>()) {
    return PolymorphicValue(a.as<at::Tensor>().erf());
  }
  NVF_ERROR(
      false, "PolymorphicValue erf not implemented for ", a.type().name());
}

// Convert scalars, vector of scalars, vector of vector of scalars, etc., into
// an at::Tensor. device argument allows for the creation of CPU Scalars.
inline PolymorphicValue toTensor(
    const PolymorphicValue& x,
    at::DeviceType device_type = at::kCUDA,
    int8_t device_index = 0) {
  if (x.is<at::Tensor>()) {
    return x;
  }
  auto options = at::TensorOptions().device(device_type, device_index);
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
  NVF_ERROR(
      false, "PolymorphicValue toTensor not implemented for ", x.type().name());
}

} // namespace PolymorphicValue_functions

} // namespace nvfuser

#include <struct.inl>
