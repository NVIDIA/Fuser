// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <multidevice/device_mesh.h>
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

bool isSame(const PolymorphicValue& a, const PolymorphicValue& b) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.is<at::Tensor>()) {
    return (a.as<at::Tensor>().is_same(b.as<at::Tensor>()));
  }
  if (a.is<double>()) {
    return isSameNanSensitive(a.as<double>(), b.as<double>());
  }
  if (a.is<std::complex<double>>()) {
    return isSameNanSensitive(
        a.as<std::complex<double>>(), b.as<std::complex<double>>());
  }
  if (a.is<python_frontend::DistributedTensor>()) {
    if (a.as<python_frontend::DistributedTensor>().mesh() ==
        b.as<python_frontend::DistributedTensor>().mesh()) {
      return a.as<python_frontend::DistributedTensor>().local() ==
          b.as<python_frontend::DistributedTensor>().local();
    }
    return false;
  }
  return a == b;
}

// Convert scalars, vector of scalars, vector of vector of scalars, etc., into
// an at::Tensor. device argument allows for the creation of CPU Scalars.
PolymorphicValue toTensor(
    const PolymorphicValue& x,
    at::DeviceType device_type,
    int8_t device_index) {
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
  if (x.is<python_frontend::DistributedTensor>()) {
    return x.as<python_frontend::DistributedTensor>().local();
  }
  NVF_THROW("PolymorphicValue toTensor not implemented for ", x.type().name());
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

c10::IValue toIValue(const PolymorphicValue& x) {
  if (x.is<at::Tensor>()) {
    return c10::IValue(x.as<at::Tensor>());
  } else if (isScalar(x)) {
    return c10::IValue(toScalar(x));
  }
  NVF_THROW("Cannot convert provided PolymorphicValue to a c10::IValue.");
}

PolymorphicValue castToDtype(PolymorphicValue value, const DataType& dtype) {
  if (!value.hasValue()) {
    return value;
  }
  // Cast the given value to the given data type. This enables interface
  // like: IrBuilder::create<Val>(0, DataType::Double) where value is
  // an integer but the desired data type is double.
  if (!hasCompatibleDataType(value, dtype)) {
    PolymorphicValue::for_all_types([&](auto _) {
      using T = typename decltype(_)::type;
      if constexpr (IsPrimitiveNativeType<T>::value) {
        if (isCompatibleDataType(NativeTypeToDataType<T>::type, dtype)) {
          value = PolymorphicValue(static_cast<T>(value));
        }
      }
      // TODO: support arrays and pointers
    });
  }
  return value;
}

} // namespace PolymorphicValue_functions

} // namespace nvfuser
