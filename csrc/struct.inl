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

inline Accessor StructHolder::operator->*(const std::string& key) const {
  return Accessor(struct_ptr_->getter(key), struct_ptr_->setter(key));
}

// If a struct type is only used in kernel and we will never create an instance
// on the host, we can just use this dummy struct as a placeholder for the
// convenience
struct NotImplementedStruct : public Struct {
  StructType type() const override;

  std::function<PolymorphicValue()> getter(
      const std::string& key) const override {
    TORCH_INTERNAL_ASSERT(false, "Not implemented");
  }

  std::function<void(const PolymorphicValue&)> setter(
      const std::string& key) override {
    TORCH_INTERNAL_ASSERT(false, "Not implemented");
  }
};

} // namespace nvfuser
