// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <polymorphic_value.h>
#include <type.h>
#include <utils.h>

#include <string>

namespace nvfuser {

bool StructHandle::operator==(const StructHandle& other) const {
  if (struct_ptr_ == other.struct_ptr_) {
    return true;
  }
  const StructType this_type = type();
  const StructType other_type = other.type();
  if (this_type.name != other_type.name) {
    return false;
  }
  if (this_type.fields.size() != other_type.fields.size()) {
    return false;
  }
  for (size_t i : arange(this_type.fields.size())) {
    // Check that fields are in same position, have same type, and have same
    // value (recursive)
    const StructType::FieldInfo& fa = this_type.fields.at(i);
    const StructType::FieldInfo& fb = other_type.fields.at(i);
    PolymorphicValue a_val = (*this)->*(fa.name);
    PolymorphicValue b_val = other->*(fb.name);
    if (fa.name != fb.name || *fa.type != *fb.type ||
        !PolymorphicValue_functions::isSame(a_val, b_val)) {
      return false;
    }
  }
  return true;
}

namespace PolymorphicValue_functions {

size_t hash(const PolymorphicValue& v) {
  size_t hash = 0;
  if (v.is<std::monostate>()) {
    return 0;
  } else if (v.is<std::complex<double>>()) {
    std::complex<double> val = v.as<std::complex<double>>();
    std::hash<double> hasher;
    hashCombine(hash, hasher(val.real()));
    hashCombine(hash, hasher(val.imag()));
  } else if (v.is<double>()) {
    hashCombine(hash, std::hash<double>()(v.as<double>()));
  } else if (v.is<int64_t>()) {
    hashCombine(hash, std::hash<int64_t>()(v.as<int64_t>()));
  } else if (v.is<bool>()) {
    hashCombine(hash, std::hash<bool>()(v.as<bool>()));
  } else {
    NVF_THROW("Cannot hash PolymorphicValue");
  }
  return hash;
}

std::string toString(const PolymorphicValue& v) {
  std::stringstream ss;
  if (v.is<at::Tensor>()) {
    ss << debug_str(v.as<at::Tensor>());
  } else if (v.is<std::monostate>()) {
    ss << "std::monostate";
  } else if (v.is<StructHandle>()) {
    const StructHandle& hdl = v.as<StructHandle>();
    StructType type = (v->*&StructHandle::type)();
    ss << "StructHandle<" << type.name << ">{";
    bool first = true;
    for (size_t i : arange(type.fields.size())) {
      if (first) {
        first = false;
      } else {
        ss << ", ";
      }
      const std::string& fieldname = type.fields.at(i).name;
      ss << fieldname << "=";
      ss << toString(hdl->*(fieldname));
    }
    ss << "}";
  } else {
    ss << v;
  }
  return ss.str();
}

PolymorphicValue IValueToPolymorphicValue(const c10::IValue& val) {
  if (val.isTensor()) {
    return val.toTensor();
  }

  auto scalar_val = val.toScalar();
  switch (scalar_val.type()) {
    case c10::ScalarType::ComplexDouble:
      return (std::complex<double>)scalar_val.toComplexDouble();
    case c10::ScalarType::Double:
      return scalar_val.toDouble();
    case c10::ScalarType::Long:
      return scalar_val.toLong();
    case c10::ScalarType::Bool:
      return scalar_val.toBool();
    default:
      NVF_THROW("Can not convert IValue to PolymorphicValue");
  }
}

inline bool isScalar(const PolymorphicValue& x) {
  return x.is<int64_t>() || x.is<double>() || x.is<bool>() ||
      x.is<std::complex<double>>();
}

c10::IValue toIValue(const PolymorphicValue& x) {
  if (x.is<at::Tensor>()) {
    return c10::IValue(x.as<at::Tensor>());
  } else if (isScalar(x)) {
    return c10::IValue(toScalar(x));
  }
  NVF_THROW("Cannot convert provided PolymorphicValue to a c10::IValue.");
}

} // namespace PolymorphicValue_functions

} // namespace nvfuser
