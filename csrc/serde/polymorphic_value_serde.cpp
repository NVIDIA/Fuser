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

nvfuser::PolymorphicValue makeCpuScalarTensor(
    const serde::ScalarCpu* scalar_cpu) {
  TORCH_INTERNAL_ASSERT(scalar_cpu != nullptr);
  auto scalar = deserializePolymorphicValue(scalar_cpu->scalar_value());
  return nvfuser::PolymorphicValue_functions::toTensor(scalar, at::kCPU);
}

nvfuser::PolymorphicValue getMetaTensorArg(const serde::TensorArg* tensor) {
  TORCH_INTERNAL_ASSERT(tensor != nullptr);
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

nvfuser::PolymorphicValue deserializePolymorphicValue(const serde::Scalar* c) {
  if (!c->has_value()) {
    return {};
  } else if (c->value_type() == serde::DataType_Double) {
    return nvfuser::PolymorphicValue(c->double_value());
  } else if (c->value_type() == serde::DataType_Int) {
    return nvfuser::PolymorphicValue(c->long_value());
  } else if (c->value_type() == serde::DataType_Bool) {
    return nvfuser::PolymorphicValue(c->bool_value());
  } else if (c->value_type() == serde::DataType_ComplexDouble) {
    return nvfuser::PolymorphicValue(
        std::complex<double>(c->real_value(), c->imag_value()));
  }
  TORCH_INTERNAL_ASSERT(
      false, "Unable to deserialize serde::Scalar as PolymorphicValue.");
}

void PolymorphicValueFactory::registerAllParsers() {
  auto deserializeScalar = [](const serde::PolymorphicValue* buffer) {
    return deserializePolymorphicValue(buffer->data_as_Scalar());
  };
  registerParser(serde::PolymorphicValueData_Scalar, deserializeScalar);

  auto deserializeScalarCpu = [](const serde::PolymorphicValue* buffer) {
    return makeCpuScalarTensor(buffer->data_as_ScalarCpu());
  };
  registerParser(serde::PolymorphicValueData_ScalarCpu, deserializeScalarCpu);

  // TODO Encode ptr field which corresponds to the aten tensor's data pointer.
  // It is used during scheduling for vectorization. A meta aten tensor assumes
  // that the pointer address is zero.
  auto deserializeTensorArg = [](const serde::PolymorphicValue* buffer) {
    return getMetaTensorArg(buffer->data_as_TensorArg());
  };
  registerParser(serde::PolymorphicValueData_TensorArg, deserializeTensorArg);
}

flatbuffers::Offset<serde::Scalar> serializeScalarCpu(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_cpu() && tensor.numel() == 1,
      "Only CPU scalar tensors are supported here.");

  switch (tensor.scalar_type()) {
    case at::ScalarType::Bool:
      return serializeScalar(
          builder, *tensor.data_ptr<bool>(), nvfuser::DataType::Bool);
    case at::ScalarType::Double:
      return serializeScalar(
          builder, *tensor.data_ptr<double>(), nvfuser::DataType::Double);
    case at::ScalarType::Long:
      return serializeScalar(
          builder, *tensor.data_ptr<int64_t>(), nvfuser::DataType::Int);
    case at::ScalarType::ComplexDouble:
      return serializeScalar(
          builder,
          *tensor.data_ptr<c10::complex<double>>(),
          nvfuser::DataType::ComplexDouble);
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported scalar type.");
  }
}

flatbuffers::Offset<serde::PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    std::shared_ptr<nvfuser::PolymorphicValue> v) {
  TORCH_INTERNAL_ASSERT(
      !v->is<std::monostate>(), "PolymorphicValue is a std::monostate.");
  TORCH_INTERNAL_ASSERT(
      !v->is<nvfuser::Struct>(),
      "Serialization of arbitrary struct is not implemented.");
  TORCH_INTERNAL_ASSERT(
      !v->is<nvfuser::Pointer>(), "Serialization of pointer is not allowed.");
  TORCH_INTERNAL_ASSERT(
      !v->is<std::vector>(), "Serialization of vector is not implemented.");

  if (v->is<at::Tensor>()) {
    const auto& tensor = v->as<at::Tensor>();

    if (tensor.is_cpu() && tensor.numel() == 1) {
      // CPU Scalar
      auto fb_scalar_data = serializeScalarCpu(builder, tensor);
      auto data = serde::CreateScalarCpu(builder, fb_scalar_data);
      return CreatePolymorphicValue(
          builder, PolymorphicValueData_ScalarCpu, data.Union());
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

      auto data = serde::CreateTensorArg(
          builder,
          (size_t)tensor.data_ptr(),
          builder.CreateVector(sizes_fb),
          builder.CreateVector(strides_fb),
          mapToSerdeDtype(tensor.scalar_type()));
      return CreatePolymorphicValue(
          builder, PolymorphicValueData_TensorArg, data.Union());
    }
  } else {
    auto data = serializeScalar(builder, *v, getDataType(*v));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData_Scalar, data.Union());
  }
}

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(mapToSerdeDtype(t));
  if (v.is<std::monostate>()) {
    builder_.add_has_value(false);
    return builder_.Finish();
  } else if (v.is<double>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(serde::DataType_Double);
    builder_.add_double_value(v.as<double>());
    return builder_.Finish();
  } else if (v.is<int64_t>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(serde::DataType_Int);
    builder_.add_long_value(v.as<int64_t>());
    return builder_.Finish();
  } else if (v.is<bool>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(serde::DataType_Bool);
    builder_.add_bool_value(v.as<bool>());
    return builder_.Finish();
  } else if (v.is<std::complex<double>>()) {
    builder_.add_has_value(true);
    auto c = v.as<std::complex<double>>();
    builder_.add_value_type(serde::DataType_ComplexDouble);
    builder_.add_real_value(std::real(c));
    builder_.add_imag_value(std::imag(c));
    return builder_.Finish();
  }
  TORCH_INTERNAL_ASSERT(
      false, "Unable to convert ", v.type().name(), " to serde::Scalar.");
}

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    bool v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(mapToSerdeDtype(t));
  builder_.add_has_value(true);
  builder_.add_value_type(serde::DataType_Bool);
  builder_.add_bool_value(v);
  return builder_.Finish();
}

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    int64_t v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(mapToSerdeDtype(t));
  builder_.add_has_value(true);
  builder_.add_value_type(serde::DataType_Int);
  builder_.add_long_value(v);
  return builder_.Finish();
}

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    double v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(mapToSerdeDtype(t));
  builder_.add_has_value(true);
  builder_.add_value_type(serde::DataType_Double);
  builder_.add_double_value(v);
  return builder_.Finish();
}

flatbuffers::Offset<serde::Scalar> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    c10::complex<double> v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(mapToSerdeDtype(t));
  builder_.add_has_value(true);
  builder_.add_value_type(serde::DataType_ComplexDouble);
  builder_.add_real_value(v.real());
  builder_.add_imag_value(v.imag());
  return builder_.Finish();
}

} // namespace nvfuser::serde
