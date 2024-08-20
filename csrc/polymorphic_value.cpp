// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <polymorphic_value.h>
#include <type.h>

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

} // namespace nvfuser

