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
struct StructTemplate {
  virtual ~StructTemplate() = default;

  virtual StructType type() const = 0;
  virtual std::function<PolymorphicValue()> getter(
      const std::string& key) const = 0;
  virtual std::function<void(const PolymorphicValue&)> setter(
      const std::string& key) = 0;
};

template <typename PolymorphicValue>
class ConstAccessorTemplate {
  const std::function<PolymorphicValue()> getter_;

 public:
  ConstAccessorTemplate() = default;
  ConstAccessorTemplate(std::function<PolymorphicValue()> getter)
      : getter_(std::move(getter)) {}
  ConstAccessorTemplate(const ConstAccessorTemplate& value) = default;
  ConstAccessorTemplate(ConstAccessorTemplate&& value) = default;
  ConstAccessorTemplate& operator=(const ConstAccessorTemplate& value) =
      default;
  ConstAccessorTemplate& operator=(ConstAccessorTemplate&& value) = default;

  operator PolymorphicValue() const {
    return getter_();
  }
};

template <typename PolymorphicValue>
class AccessorTemplate {
  const std::function<PolymorphicValue()> getter_;
  const std::function<void(const PolymorphicValue&)> setter_;

 public:
  AccessorTemplate() = default;
  AccessorTemplate(
      std::function<PolymorphicValue()> getter,
      std::function<void(const PolymorphicValue&)> setter)
      : getter_(std::move(getter)), setter_(std::move(setter)) {}
  AccessorTemplate(const AccessorTemplate& value) = default;
  AccessorTemplate(AccessorTemplate&& value) = default;
  AccessorTemplate& operator=(const AccessorTemplate& value) = default;
  AccessorTemplate& operator=(AccessorTemplate&& value) = default;

  const AccessorTemplate& operator=(const PolymorphicValue& value) const {
    setter_(std::move(value));
    return *this;
  }
  operator PolymorphicValue() const {
    return getter_();
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

  inline StructType type() const;

  template <typename Ret, typename Class>
  inline const Ret& operator->*(Ret Class::*member) const {
    return as<Class>().*member;
  }
  template <typename Ret, typename Class>
  inline Ret& operator->*(Ret Class::*member) {
    return as<Class>().*member;
  }
  Accessor operator->*(const std::string& key) const {
    return Accessor(struct_ptr_->getter(key), struct_ptr_->setter(key));
  }
};

} // namespace nvfuser
