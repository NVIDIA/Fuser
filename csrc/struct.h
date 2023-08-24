// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <functional>
#include <memory>
#include <string>

namespace nvfuser {

struct StructType;

template <typename PolymorphicValue>
class ConstAccessorTemplate {
  const std::function<PolymorphicValue()> getter_;

 public:
  ConstAccessorTemplate(std::function<PolymorphicValue()> getter)
      : getter_(std::move(getter)) {}
  operator PolymorphicValue() const {
    return getter_();
  }
};

template <typename PolymorphicValue>
class AccessorTemplate {
  const std::function<PolymorphicValue()> getter_;
  const std::function<void(PolymorphicValue)> setter_;

 public:
  AccessorTemplate(
      std::function<PolymorphicValue()> getter,
      std::function<void(PolymorphicValue)> setter)
      : getter_(std::move(getter)), setter_(std::move(setter)) {}
  const AccessorTemplate& operator=(PolymorphicValue value) const {
    setter_(std::move(value));
    return *this;
  }
  operator PolymorphicValue() const {
    return getter_();
  }
};

template <typename PolymorphicValue>
struct StructTemplate {
  using ConstAccessor = ConstAccessorTemplate<PolymorphicValue>;
  using Accessor = AccessorTemplate<PolymorphicValue>;
  virtual ~StructTemplate() = default;

  virtual StructType type() const = 0;
  virtual std::function<PolymorphicValue()> getter(std::string key) const = 0;
  virtual std::function<void(PolymorphicValue)> setter(std::string key) = 0;

  ConstAccessor operator[](std::string key) const {
    return ConstAccessor(getter(std::move(key)));
  }
  Accessor operator[](const std::string& key) {
    return Accessor(getter(key), setter(key));
  }
};

template <typename PolymorphicValue>
class StructHolder {
  using Accessor = AccessorTemplate<PolymorphicValue>;
  using Struct = StructTemplate<PolymorphicValue>;
  std::shared_ptr<Struct> struct_ptr_;

 public:
  StructHolder(std::shared_ptr<Struct> struct_ptr) : struct_ptr_(struct_ptr) {}
  StructHolder& operator=(std::shared_ptr<Struct> struct_ptr) {
    struct_ptr_ = std::move(struct_ptr);
    return *this;
  }

  StructHolder(const StructHolder& other) = default;
  StructHolder(StructHolder&& other) = default;
  StructHolder& operator=(const StructHolder& other) = default;
  StructHolder& operator=(StructHolder&& other) = default;

  template <typename T>
  bool is() const {
    return std::dynamic_pointer_cast<T>(struct_ptr_) != nullptr;
  }

  template <typename T>
  inline const T& as() const {
    return *std::dynamic_pointer_cast<T>(struct_ptr_);
  }
  template <typename T>
  inline T& as() {
    return *std::dynamic_pointer_cast<T>(struct_ptr_);
  }
  template <typename Ret, typename Class>
  inline const Ret& operator->*(Ret Class::*member) const {
    return as<Class>().*member;
  }
  template <typename Ret, typename Class>
  inline Ret& operator->*(Ret Class::*member) {
    return as<Class>().*member;
  }
  inline PolymorphicValue operator[](const std::string& key) const;
  inline StructType type() const;
};

} // namespace nvfuser
