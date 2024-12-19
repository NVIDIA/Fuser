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
  for (size_t i : c10::irange(this_type.fields.size())) {
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
    for (size_t i : c10::irange(type.fields.size())) {
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

} // namespace PolymorphicValue_functions

} // namespace nvfuser
