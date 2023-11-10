// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/EmptyTensor.h>
#include <polymorphic_value.h>
#include <serde/polymorphic_value_serde.h>
#include <serde/utils.h>

namespace nvfuser::serde {

namespace {

nvfuser::PolymorphicValue makeCpuScalarTensor(const ScalarCpu* scalar_cpu) {
  NVF_ERROR(scalar_cpu != nullptr);
  auto scalar = deserializePolymorphicValue(scalar_cpu->scalar_value());
  return nvfuser::PolymorphicValue_functions::toTensor(scalar, at::kCPU);
}

nvfuser::PolymorphicValue getMetaTensorArg(const TensorArg* tensor) {
  NVF_ERROR(tensor != nullptr);
  if (tensor->strides() != nullptr) {
    auto meta_tensor = at::detail::empty_strided_meta(
        parseVector(tensor->sizes()),
        parseVector(tensor->strides()),
        mapToAtenDtype(tensor->dtype()),
        c10::nullopt,
        c10::Device(c10::DeviceType::Meta, 0),
        c10::nullopt);
    return at::Tensor(meta_tensor);
  }
  return at::empty(
      parseVector(tensor->sizes()),
      mapToAtenDtype(tensor->dtype()),
      c10::nullopt,
      c10::Device(c10::DeviceType::Meta, 0),
      c10::nullopt,
      c10::nullopt);
}

} // namespace

nvfuser::PolymorphicValue deserializePolymorphicValue(const Scalar* c) {
  if (!c->has_value()) {
    return {};
  }
  switch (mapToNvfuserDtype(c->value_type())) {
    case PrimDataType::Double: {
      return nvfuser::PolymorphicValue(c->double_value());
    }
    case PrimDataType::Int: {
      return nvfuser::PolymorphicValue(c->long_value());
    }
    case PrimDataType::Bool: {
      return nvfuser::PolymorphicValue(c->bool_value());
    }
    case PrimDataType::ComplexDouble: {
      return nvfuser::PolymorphicValue(
          std::complex<double>(c->real_value(), c->imag_value()));
    }
    default:
      NVF_ERROR(false, "Unable to deserialize Scalar as PolymorphicValue.");
  }
}

void PolymorphicValueFactory::registerAllParsers() {
  auto deserializeScalar = [](const PolymorphicValue* buffer) {
    return deserializePolymorphicValue(buffer->data_as_Scalar());
  };
  registerParser(PolymorphicValueData::Scalar, deserializeScalar);

  auto deserializeScalarCpu = [](const PolymorphicValue* buffer) {
    return makeCpuScalarTensor(buffer->data_as_ScalarCpu());
  };
  registerParser(PolymorphicValueData::ScalarCpu, deserializeScalarCpu);

  // TODO Encode ptr field which corresponds to the aten tensor's data pointer.
  // It is used during scheduling for vectorization. A meta aten tensor assumes
  // that the pointer address is zero.
  auto deserializeTensorArg = [](const PolymorphicValue* buffer) {
    return getMetaTensorArg(buffer->data_as_TensorArg());
  };
  registerParser(PolymorphicValueData::TensorArg, deserializeTensorArg);
}

flatbuffers::Offset<Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor) {
  NVF_ERROR(
      tensor.is_cpu() && tensor.numel() == 1,
      "Only CPU scalar tensors are supported here.");

  switch (tensor.scalar_type()) {
    case at::ScalarType::Bool: {
      nvfuser::PolymorphicValue pv(*tensor.data_ptr<bool>());
      return serializeScalar(builder, pv, nvfuser::DataType::Bool);
    }
    case at::ScalarType::Double: {
      nvfuser::PolymorphicValue pv(*tensor.data_ptr<double>());
      return serializeScalar(builder, pv, nvfuser::DataType::Double);
    }
    case at::ScalarType::Long: {
      nvfuser::PolymorphicValue pv(*tensor.data_ptr<int64_t>());
      return serializeScalar(builder, pv, nvfuser::DataType::Int);
    }
    case at::ScalarType::ComplexDouble: {
      auto at_complex = *tensor.data_ptr<c10::complex<double>>();
      nvfuser::PolymorphicValue pv((std::complex<double>)at_complex);
      return serializeScalar(builder, pv, nvfuser::DataType::ComplexDouble);
    }
    default:
      NVF_ERROR(false, "Unsupported scalar type.");
  }
}

flatbuffers::Offset<PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    std::shared_ptr<nvfuser::PolymorphicValue> v) {
  NVF_ERROR(!v->is<std::monostate>(), "PolymorphicValue is a std::monostate.");
  NVF_ERROR(
      !v->is<StructHandle>(),
      "Serialization of arbitrary struct is not implemented.");
  NVF_ERROR(
      !v->is<nvfuser::Opaque>(),
      "Serialization of arbitrary opaque value is not implemented.");
  NVF_ERROR(
      !v->is<nvfuser::Pointer>(), "Serialization of pointer is not allowed.");
  NVF_ERROR(
      !v->is<std::vector>(), "Serialization of vector is not implemented.");

  if (v->is<at::Tensor>()) {
    const auto& tensor = v->as<at::Tensor>();

    if (tensor.is_cpu() && tensor.numel() == 1) {
      // CPU Scalar
      auto fb_scalar_data = serializeScalarCpu(builder, tensor);
      auto data = CreateScalarCpu(builder, fb_scalar_data);
      return CreatePolymorphicValue(
          builder, PolymorphicValueData::ScalarCpu, data.Union());
    } else {
      // GPU Tensor
      // Convert IntArrayRef to std::vector for flatbuffer compatibility
      std::vector<int64_t> sizes_fb;
      sizes_fb.reserve(tensor.ndimension());
      for (auto dim : c10::irange(tensor.ndimension())) {
        sizes_fb.push_back(tensor.size(dim));
      }

      // Convert IntArrayRef to std::vector for flatbuffer compatibility
      std::vector<int64_t> strides_fb;
      strides_fb.reserve(tensor.ndimension());
      for (auto dim : c10::irange(tensor.ndimension())) {
        strides_fb.push_back(tensor.stride(dim));
      }

      auto data = CreateTensorArg(
          builder,
          (size_t)tensor.data_ptr(),
          builder.CreateVector(sizes_fb),
          builder.CreateVector(strides_fb),
          castEnumToUnderlyingType(tensor.scalar_type()));
      return CreatePolymorphicValue(
          builder, PolymorphicValueData::TensorArg, data.Union());
    }
  } else {
    auto data = serializeScalar(builder, *v, getDataType(*v));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::Scalar, data.Union());
  }
}

flatbuffers::Offset<Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(castEnumToUnderlyingType(std::get<PrimDataType>(t.type)));
  if (v.is<std::monostate>()) {
    builder_.add_has_value(false);
    return builder_.Finish();
  } else if (v.is<double>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(castEnumToUnderlyingType(PrimDataType::Double));
    builder_.add_double_value(v.as<double>());
    return builder_.Finish();
  } else if (v.is<int64_t>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(castEnumToUnderlyingType(PrimDataType::Int));
    builder_.add_long_value(v.as<int64_t>());
    return builder_.Finish();
  } else if (v.is<bool>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(castEnumToUnderlyingType(PrimDataType::Bool));
    builder_.add_bool_value(v.as<bool>());
    return builder_.Finish();
  } else if (v.is<std::complex<double>>()) {
    builder_.add_has_value(true);
    auto c = v.as<std::complex<double>>();
    builder_.add_value_type(
        castEnumToUnderlyingType(PrimDataType::ComplexDouble));
    builder_.add_real_value(std::real(c));
    builder_.add_imag_value(std::imag(c));
    return builder_.Finish();
  }
  NVF_ERROR(false, "Unable to convert ", v.type().name(), " to Scalar.");
}

} // namespace nvfuser::serde
