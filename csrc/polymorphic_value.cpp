// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <polymorphic_value.h>
#include <type.h>

#include <string>

namespace nvfuser {

namespace PolymorphicValue_functions {

std::string toString(const PolymorphicValue& v) {
  std::stringstream ss;
  if (v.is<at::Tensor>()) {
    const auto& t = v.as<at::Tensor>();
    ss << "Tensor(sizes=" << t.sizes() << ", "
       << "stride=" << t.strides() << ", dtype=" << t.dtype()
       << ", device=" << t.device() << ", data_ptr=" << t.data_ptr() << ")";
  } else if (v.is<std::monostate>()) {
    ss << "std::monostate";
  } else if (v.is<StructHandle>()) {
    const StructHandle& hdl = v.as<StructHandle>();
    StructType type = (v->*&StructHandle::type)();
    ss << "StructHandle<" << type.name << ">{";
    bool first = true;
    for (size_t i : c10::irange(type.fields.size())) {
      if (!first) {
        ss << ", ";
      }
      first = false;
      std::string fieldname = type.fields.at(i).name;
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
