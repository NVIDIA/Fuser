#include <polymorphic_value.h>
#include <serde/utils.h>

namespace nvfuser::serde {

serde::DataType mapToSerdeDtype(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Bool:
      return serde::DataType_Bool;
    case at::ScalarType::Double:
      return serde::DataType_Double;
    case at::ScalarType::Float:
      return serde::DataType_Float;
    case at::ScalarType::Half:
      return serde::DataType_Half;
    case at::ScalarType::BFloat16:
      return serde::DataType_BFloat16;
    case at::ScalarType::Long:
      return serde::DataType_Int;
    case at::ScalarType::Int:
      return serde::DataType_Int32;
    case at::ScalarType::ComplexFloat:
      return serde::DataType_ComplexFloat;
    case at::ScalarType::ComplexDouble:
      return serde::DataType_ComplexDouble;
    default:
      return serde::DataType_None;
  }
}

at::ScalarType mapToAtenDtype(serde::DataType t) {
  switch (t) {
    case serde::DataType_Bool:
      return at::ScalarType::Bool;
    case serde::DataType_Double:
      return at::ScalarType::Double;
    case serde::DataType_Float:
      return at::ScalarType::Float;
    case serde::DataType_Half:
      return at::ScalarType::Half;
    case serde::DataType_BFloat16:
      return at::ScalarType::BFloat16;
    case serde::DataType_Int:
      return at::ScalarType::Long;
    case serde::DataType_Int32:
      return at::ScalarType::Int;
    case serde::DataType_ComplexFloat:
      return at::ScalarType::ComplexFloat;
    case serde::DataType_ComplexDouble:
      return at::ScalarType::ComplexDouble;
    case serde::DataType_None:
      return at::ScalarType::Undefined;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No nvfuser dtype found for serde data type.");
  return at::ScalarType::Undefined;
}

serde::DataType mapToSerdeDtype(nvfuser::DataType t) {
  return mapToSerdeDtype(std::get<PrimDataType>(t.type));
}

nvfuser::DataType mapToDtypeStruct(serde::DataType t) {
  return nvfuser::DataType(mapToNvfuserDtype(t));
}

serde::DataType mapToSerdeDtype(PrimDataType t) {
  switch (t) {
    case PrimDataType::Bool:
      return serde::DataType_Bool;
    case PrimDataType::Double:
      return serde::DataType_Double;
    case PrimDataType::Float:
      return serde::DataType_Float;
    case PrimDataType::Half:
      return serde::DataType_Half;
    case PrimDataType::BFloat16:
      return serde::DataType_BFloat16;
    case PrimDataType::Int:
      return serde::DataType_Int;
    case PrimDataType::Int32:
      return serde::DataType_Int32;
    case PrimDataType::ComplexFloat:
      return serde::DataType_ComplexFloat;
    case PrimDataType::ComplexDouble:
      return serde::DataType_ComplexDouble;
    case PrimDataType::Null:
      return serde::DataType_None;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No serde dtype found for nvfuser data type.");
  return serde::DataType_MAX;
}

PrimDataType mapToNvfuserDtype(serde::DataType t) {
  switch (t) {
    case serde::DataType_Bool:
      return PrimDataType::Bool;
    case serde::DataType_Double:
      return PrimDataType::Double;
    case serde::DataType_Float:
      return PrimDataType::Float;
    case serde::DataType_Half:
      return PrimDataType::Half;
    case serde::DataType_BFloat16:
      return PrimDataType::BFloat16;
    case serde::DataType_Int:
      return PrimDataType::Int;
    case serde::DataType_Int32:
      return PrimDataType::Int32;
    case serde::DataType_ComplexFloat:
      return PrimDataType::ComplexFloat;
    case serde::DataType_ComplexDouble:
      return PrimDataType::ComplexDouble;
    case serde::DataType_None:
      return PrimDataType::Null;
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No nvfuser dtype found for serde data type.");
  return PrimDataType::Null;
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

flatbuffers::Offset<serde::ArgAbstract> serializePolymorphicValue(
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
      return CreateArgAbstract(
          builder, ArgAbstractData_ScalarCpu, data.Union());
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
      return CreateArgAbstract(
          builder, ArgAbstractData_TensorArg, data.Union());
    }
  } else {
    auto data = serializeScalar(builder, *v, getDataType(*v));
    return CreateArgAbstract(builder, ArgAbstractData_Scalar, data.Union());
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

PolymorphicValue parsePolymorphicValue(const serde::Scalar* c) {
  if (!c->has_value()) {
    return {};
  } else if (c->value_type() == serde::DataType_Double) {
    return PolymorphicValue(c->double_value());
  } else if (c->value_type() == serde::DataType_Int) {
    return PolymorphicValue(c->long_value());
  } else if (c->value_type() == serde::DataType_Bool) {
    return PolymorphicValue(c->bool_value());
  } else if (c->value_type() == serde::DataType_ComplexDouble) {
    return PolymorphicValue(
        std::complex<double>(c->real_value(), c->imag_value()));
  }
  TORCH_INTERNAL_ASSERT(
      false, "Unable to deserialize serde::Scalar as PolymorphicValue.");
}

std::vector<bool> parseBoolVector(
    const flatbuffers::Vector<uint8_t>* fb_vector) {
  std::vector<bool> result(fb_vector->begin(), fb_vector->end());
  return result;
}

} // namespace nvfuser::serde
