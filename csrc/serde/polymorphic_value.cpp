// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/EmptyTensor.h>
#include <kernel_ir.h>
#include <polymorphic_value.h>
#include <serde/polymorphic_value.h>
#include <serde/utils.h>
#include <typeinfo>

namespace nvfuser::serde {

namespace {

nvfuser::PolymorphicValue makeCpuScalarTensor(const ScalarCpu* scalar_cpu) {
  NVF_ERROR(scalar_cpu != nullptr, "serde::ScalarCpu is nullptr.");
  auto scalar = makeScalar(scalar_cpu->scalar_value());
  return nvfuser::PolymorphicValue_functions::toTensor(scalar, at::kCPU);
}

nvfuser::PolymorphicValue makeMetaTensorArg(const TensorArg* tensor) {
  NVF_ERROR(tensor != nullptr, "serde::TensorArg is nullptr.");
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

nvfuser::PolymorphicValue deserializeBool(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto bool_data = buffer->data_as_Bool();
  NVF_CHECK(bool_data != nullptr, "serde::Bool is nullptr.");
  return nvfuser::PolymorphicValue(bool_data->value());
}

nvfuser::PolymorphicValue deserializeComplexDouble(
    const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto complex_data = buffer->data_as_ComplexDouble();
  NVF_CHECK(complex_data != nullptr, "serde::ComplexDouble is nullptr.");
  return nvfuser::PolymorphicValue(
      std::complex<double>(complex_data->real(), complex_data->imag()));
}

nvfuser::PolymorphicValue deserializeDouble(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto double_data = buffer->data_as_Double();
  NVF_CHECK(double_data != nullptr, "serde::Double is nullptr.");
  return nvfuser::PolymorphicValue(double_data->value());
}

nvfuser::PolymorphicValue deserializeLong(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto long_data = buffer->data_as_Long();
  NVF_CHECK(long_data != nullptr, "serde::Long is nullptr.");
  return nvfuser::PolymorphicValue(long_data->value());
}

nvfuser::PolymorphicValue makeScalar(const Scalar* c) {
  NVF_CHECK(c != nullptr, "serde::Scalar is nullptr.");
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
      NVF_ERROR(
          false, "Unable to deserialize serde::Scalar as PolymorphicValue.");
  }
}

void PolymorphicValueFactory::registerAllParsers() {
  auto deserialize_unsupported =
      [](const serde::PolymorphicValue* buffer) -> nvfuser::PolymorphicValue {
    NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
    NVF_ERROR(
        false,
        "Unsupported PolymorphicValueData\t",
        static_cast<int64_t>(toUnderlying(buffer->data_type())));
    return nvfuser::PolymorphicValue();
  };
  registerParser(PolymorphicValueData::Array, deserialize_unsupported);
  registerParser(PolymorphicValueData::AsmOptions, deserialize_unsupported);
  registerParser(PolymorphicValueData::OpaqueEnum, deserialize_unsupported);
  registerParser(
      PolymorphicValueData::ParallelTypeBitmap, deserialize_unsupported);
  registerParser(PolymorphicValueData::RNGAttributes, deserialize_unsupported);
  registerParser(PolymorphicValueData::Scope, deserialize_unsupported);

  auto deserialize_monostate =
      [](const serde::PolymorphicValue* buffer) -> nvfuser::PolymorphicValue {
    NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
    return nvfuser::PolymorphicValue();
  };
  registerParser(PolymorphicValueData::NONE, deserialize_monostate);

  registerParser(PolymorphicValueData::Bool, deserializeBool);
  registerParser(PolymorphicValueData::Double, deserializeDouble);
  registerParser(PolymorphicValueData::ComplexDouble, deserializeComplexDouble);
  registerParser(PolymorphicValueData::Long, deserializeLong);

  auto deserializeScalarCpu = [](const PolymorphicValue* buffer) {
    return makeCpuScalarTensor(buffer->data_as_ScalarCpu());
  };
  registerParser(PolymorphicValueData::ScalarCpu, deserializeScalarCpu);

  // TODO Encode ptr field which corresponds to the aten tensor's data pointer.
  // It is used during scheduling for vectorization. A meta aten tensor assumes
  // that the pointer address is zero.
  auto deserializeTensorArg = [](const PolymorphicValue* buffer) {
    return makeMetaTensorArg(buffer->data_as_TensorArg());
  };
  registerParser(PolymorphicValueData::TensorArg, deserializeTensorArg);
}

nvfuser::PolymorphicValue deserializePolymorphicValue(
    const PolymorphicValue* pv) {
  PolymorphicValueFactory pv_factory;
  return pv_factory.parse(pv->data_type(), pv);
}

flatbuffers::Offset<PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v) {
  NVF_ERROR(
      !v.is<nvfuser::Pointer>(), "Serialization of pointer is not allowed.");

  if (v.is<std::monostate>()) {
    return CreatePolymorphicValue(builder, PolymorphicValueData::NONE);
  } else if (v.is<std::vector>()) {
    auto vec = v.as<std::vector>();
    std::vector<flatbuffers::Offset<serde::PolymorphicValue>> fb_items;
    fb_items.reserve(vec.size());
    for (const auto& item : vec) {
      fb_items.push_back(serializePolymorphicValue(builder, item));
    }
    return CreatePolymorphicValue(
        builder,
        PolymorphicValueData::Array,
        CreateArrayDirect(builder, &fb_items).Union());
  } else if (v.is<nvfuser::Opaque>()) {
    return serializeOpaque(builder, v.as<nvfuser::Opaque>());
  } else if (v.is<StructHandle>()) {
    return serializeStruct(builder, v.as<nvfuser::StructHandle>());
  } else if (v.is<at::Tensor>()) {
    return serializeTensor(builder, v.as<at::Tensor>());
  } else {
    return serializeScalar(builder, v);
  }
}

// TODO Refactor
flatbuffers::Offset<PolymorphicValue> serializeStruct(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::StructHandle& v) {
  flatbuffers::Offset<void> data = 0;
  if (v.is<nvfuser::kir::AsmOptions>()) {
    nvfuser::kir::AsmOptions& options = v.as<nvfuser::kir::AsmOptions>();
    std::vector<int64_t> fb_readable_outputs(
        options.readable_outputs.begin(), options.readable_outputs.end());
    data = serde::CreateAsmOptionsDirect(
               builder, options.volatile_, options.memory, &fb_readable_outputs)
               .Union();
  } else if (v.is<nvfuser::ParallelTypeBitmap>()) {
    nvfuser::ParallelTypeBitmap& pt_bitmap =
        v.as<nvfuser::ParallelTypeBitmap>();
    data =
        serde::CreateParallelTypeBitmap(builder, pt_bitmap.toUlong()).Union();
  } else if (v.is<nvfuser::RNGOp::Attributes>()) {
    nvfuser::RNGOp::Attributes& attributes = v.as<nvfuser::RNGOp::Attributes>();
    data = serde::CreateRNGAttributes(
               builder,
               toUnderlying(attributes.rtype),
               toUnderlying(std::get<PrimDataType>(attributes.dtype.type)),
               attributes.num_parameters)
               .Union();
  } else if (v.is<nvfuser::kir::Scope>()) {
    nvfuser::kir::Scope& kir_scope = v.as<nvfuser::kir::Scope>();
    nvfuser::kir::Kernel* kernel = kir_scope.owner()->kernel();

    // TODO Refactor to use IrSerde determinstic_exprs_map
    auto exprs_to_id_map = kernel->deterministic_exprs_map();

    std::vector<int64_t> fb_exprs;
    fb_exprs.reserve(kir_scope.size());
    for (Expr* e : kir_scope.exprs()) {
      fb_exprs.push_back(exprs_to_id_map.at(e));
    }
    data = serde::CreateScopeDirect(
               builder, &fb_exprs, exprs_to_id_map.at(kir_scope.owner()))
               .Union();
  } else {
    NVF_ERROR(
        false, "Serialization of arbitrary struct handle is not implemented.");
  }
  return CreatePolymorphicValue(
      builder, PolymorphicValueData::OpaqueEnum, data);
}

// TODO Refactor
flatbuffers::Offset<PolymorphicValue> serializeOpaque(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::Opaque& v) {
  flatbuffers::Offset<OpaqueEnum> data = 0;
  if (v.any().type() == typeid(AsyncOpType)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::AsyncOpType, toUnderlying(v.as<AsyncOpType>()));
  } else if (v.any().type() == typeid(DoubleBufferLoopStage)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::DoubleBufferLoopStage,
        toUnderlying(v.as<DoubleBufferLoopStage>()));
  } else if (v.any().type() == typeid(BinaryOpType)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::BinaryOpType, toUnderlying(v.as<BinaryOpType>()));
  } else if (v.any().type() == typeid(CacheOp)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::CacheOp, toUnderlying(v.as<CacheOp>()));
  } else if (v.any().type() == typeid(LoadStoreOpType)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::LoadStoreOpType,
        toUnderlying(v.as<LoadStoreOpType>()));
  } else if (v.any().type() == typeid(MemoryType)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::MemoryType, toUnderlying(v.as<MemoryType>()));
  } else if (v.any().type() == typeid(MmaMacro)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::MmaMacro,
        (int64_t)toUnderlying(v.as<MmaMacro>()));
  } else if (v.any().type() == typeid(ScatterOpType)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::ScatterOpType,
        toUnderlying(v.as<ScatterOpType>()));
  } else if (v.any().type() == typeid(SwizzleMode)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::SwizzleMode, toUnderlying(v.as<SwizzleMode>()));
  } else if (v.any().type() == typeid(SwizzleType)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::SwizzleType, toUnderlying(v.as<SwizzleType>()));
  } else if (v.any().type() == typeid(Swizzle2DType)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::Swizzle2DType,
        toUnderlying(v.as<Swizzle2DType>()));
  } else if (v.any().type() == typeid(tma::TensorMapInterleave)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapInterleave,
        toUnderlying(v.as<tma::TensorMapInterleave>()));
  } else if (v.any().type() == typeid(tma::TensorMapL2Promotion)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapL2Promotion,
        toUnderlying(v.as<tma::TensorMapL2Promotion>()));
  } else if (v.any().type() == typeid(tma::TensorMapFloatOOBFill)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapFloatOOBFill,
        toUnderlying(v.as<tma::TensorMapFloatOOBFill>()));
  } else if (v.any().type() == typeid(TernaryOpType)) {
    data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TernaryOpType,
        toUnderlying(v.as<TernaryOpType>()));
  } else if (v.any().type() == typeid(UnaryOpType)) {
    data = CreateOpaqueEnum(
        builder, NvFuserEnum::UnaryOpType, toUnderlying(v.as<UnaryOpType>()));
  } else {
    NVF_ERROR(
        false, "Serialization of arbitrary opaque value is not implemented.");
  }
  return CreatePolymorphicValue(
      builder, PolymorphicValueData::OpaqueEnum, data.Union());
}

flatbuffers::Offset<PolymorphicValue> serializeTensor(
    flatbuffers::FlatBufferBuilder& builder,
    const at::Tensor& tensor) {
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
        nvfuser::toUnderlying(tensor.scalar_type()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::TensorArg, data.Union());
  }
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
      return serializeScalarRecord(builder, pv, nvfuser::DataType::Bool);
    }
    case at::ScalarType::Double: {
      nvfuser::PolymorphicValue pv(*tensor.data_ptr<double>());
      return serializeScalarRecord(builder, pv, nvfuser::DataType::Double);
    }
    case at::ScalarType::Long: {
      nvfuser::PolymorphicValue pv(*tensor.data_ptr<int64_t>());
      return serializeScalarRecord(builder, pv, nvfuser::DataType::Int);
    }
    case at::ScalarType::ComplexDouble: {
      auto at_complex = *tensor.data_ptr<c10::complex<double>>();
      nvfuser::PolymorphicValue pv((std::complex<double>)at_complex);
      return serializeScalarRecord(
          builder, pv, nvfuser::DataType::ComplexDouble);
    }
    default:
      NVF_ERROR(false, "Unsupported scalar type.");
  }
}

flatbuffers::Offset<PolymorphicValue> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v) {
  if (v.is<std::monostate>()) {
    return serde::CreatePolymorphicValue(
        builder, PolymorphicValueData::NONE, 0);
  } else if (v.is<double>()) {
    auto scalar = serde::CreateDouble(builder, v.as<double>());
    return serde::CreatePolymorphicValue(
        builder, PolymorphicValueData::Double, scalar.Union());
  } else if (v.is<int64_t>()) {
    auto scalar = serde::CreateLong(builder, v.as<int64_t>());
    return serde::CreatePolymorphicValue(
        builder, PolymorphicValueData::Long, scalar.Union());
  } else if (v.is<bool>()) {
    auto scalar = serde::CreateBool(builder, v.as<bool>());
    return serde::CreatePolymorphicValue(
        builder, PolymorphicValueData::Bool, scalar.Union());
  } else if (v.is<std::complex<double>>()) {
    auto c = v.as<std::complex<double>>();
    auto scalar =
        serde::CreateComplexDouble(builder, std::real(c), std::imag(c));
    return serde::CreatePolymorphicValue(
        builder, PolymorphicValueData::ComplexDouble, scalar.Union());
  }
  NVF_ERROR(false, "Unable to convert ", v.type().name(), " to Scalar.");
}

flatbuffers::Offset<Scalar> serializeScalarRecord(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::PolymorphicValue& v,
    nvfuser::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(toUnderlying(std::get<PrimDataType>(t.type)));
  if (v.is<std::monostate>()) {
    builder_.add_has_value(false);
    return builder_.Finish();
  } else if (v.is<double>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(PrimDataType::Double));
    builder_.add_double_value(v.as<double>());
    return builder_.Finish();
  } else if (v.is<int64_t>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(PrimDataType::Int));
    builder_.add_long_value(v.as<int64_t>());
    return builder_.Finish();
  } else if (v.is<bool>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(PrimDataType::Bool));
    builder_.add_bool_value(v.as<bool>());
    return builder_.Finish();
  } else if (v.is<std::complex<double>>()) {
    builder_.add_has_value(true);
    auto c = v.as<std::complex<double>>();
    builder_.add_value_type(toUnderlying(PrimDataType::ComplexDouble));
    builder_.add_real_value(std::real(c));
    builder_.add_imag_value(std::imag(c));
    return builder_.Finish();
  }
  NVF_ERROR(false, "Unable to convert ", v.type().name(), " to Scalar.");
}

} // namespace nvfuser::serde
