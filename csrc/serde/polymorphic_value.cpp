// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <ATen/EmptyTensor.h>
#include <device_lower/pass/loop_rotation.h>
#include <dynamic_transform.h>
#include <fusion.h>
#include <ir/cloner.h>
#include <kernel_ir.h>
#include <polymorphic_value.h>
#include <serde/polymorphic_value.h>
#include <serde/utils.h>
#include <optional>
#include <string>
#include <tuple>
#include <typeinfo>
#include <variant>

namespace nvf = nvfuser;

namespace nvfuser::serde {

namespace {

std::any cloneDynamicTransformInitialInfo(
    nvf::IrCloner& ir_cloner,
    std::any data) {
  return std::any_cast<nvf::DynamicTransformInitialInfo>(data).clone(ir_cloner);
}

nvf::PolymorphicValue deserializeMonostate(
    const serde::PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
  return nvf::PolymorphicValue();
}

nvf::PolymorphicValue deserializeAsmOptions(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const AsmOptions* data = buffer->data_as_AsmOptions();
  NVF_ERROR(data != nullptr, "serde::AsmOptions is nullptr.");
  std::unordered_set<int64_t> readable_outputs;
  std::copy(
      data->readable_outputs()->begin(),
      data->readable_outputs()->end(),
      std::inserter(readable_outputs, readable_outputs.begin()));
  nvf::kir::AsmOptions options{
      data->volatile_(), data->memory(), std::move(readable_outputs)};
  return nvf::PolymorphicValue(nvf::Opaque(std::move(options)));
}

nvf::PolymorphicValue deserializeDynamicTransformInitialInfo(
    const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const DynamicTransformInitialInfo* data =
      buffer->data_as_DynamicTransformInitialInfo();
  NVF_ERROR(data != nullptr, "serde::DynamicTransformInitialInfo is nullptr.");
  return nvf::PolymorphicValue(nvf::Opaque(
      nvf::DynamicTransformInitialInfo(FusionGuard::getCurFusion(), data)));
}

nvf::PolymorphicValue deserializeLoopRotation(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const LoopRotation* data = buffer->data_as_LoopRotation();
  NVF_ERROR(data != nullptr, "serde::LoopRotation is nullptr.");

  nvf::Fusion* fusion = FusionGuard::getCurFusion();

  std::vector<nvf::LoopRotationTuple> loop_rotation;
  loop_rotation.reserve(data->items()->size());
  for (auto loop_rotation_param : *data->items()) {
    std::unordered_set<nvf::Statement*> nvf_selection_stmt_set;
    nvf_selection_stmt_set.reserve(
        loop_rotation_param->selection_stmt_set()->size());
    std::transform(
        loop_rotation_param->selection_stmt_set()->begin(),
        loop_rotation_param->selection_stmt_set()->end(),
        std::inserter(nvf_selection_stmt_set, nvf_selection_stmt_set.end()),
        [&fusion](const StatementIndex* stmt_index) {
          return stmt_index->is_val()
              ? fusion->getVal<nvf::Statement>(stmt_index->index())
              : fusion->getExpr<nvf::Statement>(stmt_index->index());
        });
    loop_rotation.emplace_back(nvf::LoopRotationTuple(
        fusion->getVal<nvf::TensorView>(loop_rotation_param->tv()),
        loop_rotation_param->axis(),
        nvf_selection_stmt_set));
  }
  return nvf::PolymorphicValue(nvf::Opaque(loop_rotation));
}

// TODO Refactor
nvf::PolymorphicValue deserializeOpaqueEnum(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const OpaqueEnum* data = buffer->data_as_OpaqueEnum();
  NVF_ERROR(data != nullptr, "serde::OpaqueEnum is nullptr.");

  switch (data->data_attribute_enum()) {
    case NvFuserEnum::AsyncOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::AsyncOpType>(data->value())));
    }
    case NvFuserEnum::BinaryOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::BinaryOpType>(data->value())));
    }
    case NvFuserEnum::CacheOp: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::CacheOp>(data->value())));
    }
    case NvFuserEnum::DoubleBufferLoopStage: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::DoubleBufferLoopStage>(data->value())));
    }
    case NvFuserEnum::LoadStoreOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::LoadStoreOpType>(data->value())));
    }
    case NvFuserEnum::MemoryType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::MemoryType>(data->value())));
    }
    case NvFuserEnum::MmaLayout: {
      if (data->value() == -1) {
        return nvf::PolymorphicValue(nvf::Opaque(nvf::MmaOp::MmaLayoutOpt()));
      }
      auto enum_value = static_cast<nvf::MmaLayout>(data->value());
      return nvf::PolymorphicValue(
          nvf::Opaque(nvf::MmaOp::MmaLayoutOpt(enum_value)));
    }
    case NvFuserEnum::MmaMacro: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::MmaMacro>(data->value())));
    }
    case NvFuserEnum::ScatterOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::ScatterOpType>(data->value())));
    }
    case NvFuserEnum::SwizzleMode: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::SwizzleMode>(data->value())));
    }
    case NvFuserEnum::SwizzleType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::SwizzleType>(data->value())));
    }
    case NvFuserEnum::Swizzle2DType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::Swizzle2DType>(data->value())));
    }
    case NvFuserEnum::TensorMapInterleave: {
      return nvf::PolymorphicValue(nvf::Opaque(
          static_cast<nvf::tma::TensorMapInterleave>(data->value())));
    }
    case NvFuserEnum::TensorMapL2Promotion: {
      return nvf::PolymorphicValue(nvf::Opaque(
          static_cast<nvf::tma::TensorMapL2Promotion>(data->value())));
    }
    case NvFuserEnum::TensorMapFloatOOBFill: {
      return nvf::PolymorphicValue(nvf::Opaque(
          static_cast<nvf::tma::TensorMapFloatOOBFill>(data->value())));
    }
    case NvFuserEnum::TernaryOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::TernaryOpType>(data->value())));
    }
    case NvFuserEnum::UnaryOpType: {
      return nvf::PolymorphicValue(
          nvf::Opaque(static_cast<nvf::UnaryOpType>(data->value())));
    }
    default: {
      NVF_ERROR(
          false, "Serialization of arbitrary opaque value is not implemented.");
    }
  }
}

nvf::PolymorphicValue deserializeParallelTypeBitmap(
    const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const ParallelTypeBitmap* data = buffer->data_as_ParallelTypeBitmap();
  NVF_ERROR(data != nullptr, "serde::ParallelTypeBitmap is nullptr.");
  nvf::ParallelTypeBitmap bitmap{data->value()};
  return nvf::PolymorphicValue(nvf::Opaque(bitmap));
}

nvf::PolymorphicValue deserializeRNGAttributes(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const RNGAttributes* data = buffer->data_as_RNGAttributes();
  NVF_ERROR(data != nullptr, "serde::RNGAttributes is nullptr.");
  nvf::RNGOp::Attributes attributes{
      static_cast<RNGOpType>(data->rng_op_type_enum()),
      serde::mapToDtypeStruct(data->dtype_enum()),
      data->num_parameters()};
  return nvf::PolymorphicValue(nvf::Opaque(std::move(attributes)));
}

nvf::PolymorphicValue deserializeScalarCpu(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const ScalarCpu* scalar_cpu = buffer->data_as_ScalarCpu();
  NVF_ERROR(scalar_cpu != nullptr, "serde::ScalarCpu is nullptr.");
  auto scalar = makeScalar(scalar_cpu->scalar_value());
  return nvf::PolymorphicValue_functions::toTensor(scalar, at::kCPU);
}

nvf::PolymorphicValue deserializeString(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto data = buffer->data_as_String();
  NVF_ERROR(data != nullptr, "flatbuffer::String is nullptr.");
  return nvf::PolymorphicValue(nvf::Opaque(data->value()->str()));
}

// TODO Encode ptr field which corresponds to the aten tensor's data pointer.
// It is used during scheduling for vectorization. A meta aten tensor assumes
// that the pointer address is zero.
nvf::PolymorphicValue deserializeTensorArg(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const TensorArg* tensor = buffer->data_as_TensorArg();
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

nvf::PolymorphicValue deserializeScope(const PolymorphicValue* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  const Scope* data = buffer->data_as_Scope();
  NVF_ERROR(data != nullptr, "serde::Scope is nullptr.");
  // Expressions for the scope are added upon their creation because they do not
  // exist yet in the IrContainer.
  return nvf::PolymorphicValue(nvf::Opaque(nvf::kir::Scope()));
}

} // namespace

nvf::PolymorphicValue deserializeBool(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto bool_data = buffer->data_as_Bool();
  NVF_CHECK(bool_data != nullptr, "serde::Bool is nullptr.");
  return nvf::PolymorphicValue(bool_data->value());
}

nvf::PolymorphicValue deserializeComplexDouble(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto complex_data = buffer->data_as_ComplexDouble();
  NVF_CHECK(complex_data != nullptr, "serde::ComplexDouble is nullptr.");
  return nvf::PolymorphicValue(
      std::complex<double>(complex_data->real(), complex_data->imag()));
}

nvf::PolymorphicValue deserializeDouble(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto double_data = buffer->data_as_Double();
  NVF_CHECK(double_data != nullptr, "serde::Double is nullptr.");
  return nvf::PolymorphicValue(double_data->value());
}

nvf::PolymorphicValue deserializeLong(const PolymorphicValue* buffer) {
  NVF_CHECK(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
  auto long_data = buffer->data_as_Long();
  NVF_CHECK(long_data != nullptr, "serde::Long is nullptr.");
  return nvf::PolymorphicValue(long_data->value());
}

nvf::PolymorphicValue makeScalar(const Scalar* c) {
  NVF_CHECK(c != nullptr, "serde::Scalar is nullptr.");
  if (!c->has_value()) {
    return {};
  }
  switch (mapToNvfuserDtype(c->value_type())) {
    case nvf::PrimDataType::Bool: {
      return nvf::PolymorphicValue(c->bool_value());
    }
    case nvf::PrimDataType::ComplexDouble: {
      return nvf::PolymorphicValue(
          std::complex<double>(c->real_value(), c->imag_value()));
    }
    case nvf::PrimDataType::Double: {
      return nvf::PolymorphicValue(c->double_value());
    }
    case nvf::PrimDataType::Int: {
      return nvf::PolymorphicValue(c->long_value());
    }
    default:
      NVF_ERROR(
          false, "Unable to deserialize serde::Scalar as PolymorphicValue.");
  }
}

template <typename T>
std::vector<T> PolymorphicValueFactory::makeArray(const serde::Array* data) {
  NVF_ERROR(data != nullptr, "serde::Array is nullptr.");
  std::vector<T> array;
  array.reserve(data->items()->size());
  std::transform(
      data->items()->begin(),
      data->items()->end(),
      std::back_inserter(array),
      [this](const serde::PolymorphicValue* fb_pv) {
        return parse(fb_pv->data_type(), fb_pv).as<T>();
      });
  return array;
}

nvf::PolymorphicValue PolymorphicValueFactory::makeArray(
    const serde::Array* data) {
  NVF_ERROR(data != nullptr, "serde::Array is nullptr.");
  switch (data->type()) {
    case serde::PrimArrayType::Bool: {
      if (data->is_opaque()) {
        return nvf::PolymorphicValue(nvf::Opaque(makeArray<bool>(data)));
      }
      NVF_ERROR(
          false, "dynamic_type::Containers<std::vector> does not type bool.");
    }
    case serde::PrimArrayType::ComplexDouble: {
      if (data->is_opaque()) {
        return nvf::PolymorphicValue(
            nvf::Opaque(makeArray<std::complex<double>>(data)));
      }
      return nvf::PolymorphicValue(makeArray<std::complex<double>>(data));
    }
    case serde::PrimArrayType::Double: {
      if (data->is_opaque()) {
        return nvf::PolymorphicValue(nvf::Opaque(makeArray<double>(data)));
      }
      return nvf::PolymorphicValue(makeArray<double>(data));
    }
    case serde::PrimArrayType::Long: {
      if (data->is_opaque()) {
        return nvf::PolymorphicValue(nvf::Opaque(makeArray<int64_t>(data)));
      }
      return nvf::PolymorphicValue(makeArray<int64_t>(data));
    }
    default: {
      NVF_ERROR(false, "Unsupported Array Type.");
    }
  }
}

void PolymorphicValueFactory::registerAllParsers() {
  auto deserialize_unsupported =
      [](const serde::PolymorphicValue* buffer) -> nvf::PolymorphicValue {
    NVF_ERROR(buffer != nullptr, "serde::Value is nullptr.");
    NVF_ERROR(
        false,
        "Unsupported PolymorphicValueData\t",
        static_cast<int64_t>(toUnderlying(buffer->data_type())));
    return nvf::PolymorphicValue();
  };
  registerParser(PolymorphicValueData::Scope, deserialize_unsupported);

  auto deserialize_array = [this](const PolymorphicValue* buffer) {
    NVF_ERROR(buffer != nullptr, "serde::PolymorphicValue is nullptr.");
    return makeArray(buffer->data_as_Array());
  };
  registerParser(PolymorphicValueData::Array, deserialize_array);
  registerParser(PolymorphicValueData::AsmOptions, deserializeAsmOptions);
  registerParser(PolymorphicValueData::Bool, deserializeBool);
  registerParser(PolymorphicValueData::ComplexDouble, deserializeComplexDouble);
  registerParser(PolymorphicValueData::Double, deserializeDouble);
  registerParser(
      PolymorphicValueData::DynamicTransformInitialInfo,
      deserializeDynamicTransformInitialInfo);
  registerParser(PolymorphicValueData::LoopRotation, deserializeLoopRotation);
  registerParser(PolymorphicValueData::Long, deserializeLong);
  registerParser(PolymorphicValueData::NONE, deserializeMonostate);
  registerParser(PolymorphicValueData::OpaqueEnum, deserializeOpaqueEnum);
  registerParser(
      PolymorphicValueData::ParallelTypeBitmap, deserializeParallelTypeBitmap);
  registerParser(PolymorphicValueData::RNGAttributes, deserializeRNGAttributes);
  registerParser(PolymorphicValueData::ScalarCpu, deserializeScalarCpu);
  registerParser(PolymorphicValueData::Scope, deserializeScope);
  registerParser(PolymorphicValueData::String, deserializeString);
  registerParser(PolymorphicValueData::TensorArg, deserializeTensorArg);
}

nvf::PolymorphicValue deserializePolymorphicValue(
    nvf::IrContainer* container,
    const PolymorphicValue* pv) {
  PolymorphicValueFactory pv_factory(container);
  return pv_factory.parse(pv->data_type(), pv);
}

void deserializeManagedData(
    nvfuser::Fusion* fusion,
    const PolymorphicValue* pv) {
  std::any a = deserializePolymorphicValue(fusion, pv).as<nvf::Opaque>().any();
  if (a.type() == typeid(nvf::DynamicTransformInitialInfo)) {
    fusion->manage(a, cloneDynamicTransformInitialInfo);
  } else if (a.type() == typeid(nvf::LoopRotationParam)) {
    fusion->manage<nvf::LoopRotationParam>(
        std::any_cast<nvf::LoopRotationParam>(a));
  } else {
    NVF_ERROR(false, "Unsupported managed data type");
  }
}

void deserializeManagedNamedData(
    nvfuser::Fusion* fusion,
    const std::string& name,
    const PolymorphicValue* pv) {
  std::any a = deserializePolymorphicValue(fusion, pv).as<nvf::Opaque>().any();
  if (a.type() == typeid(nvf::DynamicTransformInitialInfo)) {
    fusion->manage(name, a, cloneDynamicTransformInitialInfo);
  } else if (a.type() == typeid(nvf::LoopRotationParam)) {
    fusion->manage<nvf::LoopRotationParam>(
        name, std::any_cast<nvf::LoopRotationParam>(a));
  } else {
    NVF_ERROR(false, "Unsupported managed named data type");
  }
}

namespace {

template <typename T, serde::PrimArrayType array_type>
flatbuffers::Offset<PolymorphicValue> serializeArray(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::Opaque& v) {
  auto vec = v.as<std::vector<T>>();
  std::vector<flatbuffers::Offset<serde::PolymorphicValue>> fb_items;
  fb_items.reserve(vec.size());
  std::transform(
      vec.begin(),
      vec.end(),
      std::back_inserter(fb_items),
      [&builder](const T& item) {
        return serializeBasicPolymorphicValue(builder, item);
      });
  return CreatePolymorphicValue(
      builder,
      PolymorphicValueData::Array,
      CreateArrayDirect(
          builder,
          /*is_opaque=*/true,
          array_type,
          &fb_items)
          .Union());
}

// TODO Refactor
flatbuffers::Offset<PolymorphicValue> serializeOpaque(
    flatbuffers::FlatBufferBuilder& builder,
    const IrSerde& container,
    const nvf::Opaque& v) {
  if (v.any().type() == typeid(nvf::kir::AsmOptions)) {
    const auto& options = v.as<nvf::kir::AsmOptions>();
    std::vector<int64_t> fb_readable_outputs(
        options.readable_outputs.begin(), options.readable_outputs.end());
    auto data =
        serde::CreateAsmOptionsDirect(
            builder, options.volatile_, options.memory, &fb_readable_outputs)
            .Union();
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::AsmOptions, data);
  } else if (v.any().type() == typeid(nvf::ParallelTypeBitmap)) {
    const auto& pt_bitmap = v.as<nvf::ParallelTypeBitmap>();
    auto data =
        serde::CreateParallelTypeBitmap(builder, pt_bitmap.toUlong()).Union();
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::ParallelTypeBitmap, data);
  } else if (v.any().type() == typeid(nvf::RNGOp::Attributes)) {
    const auto& attributes = v.as<nvf::RNGOp::Attributes>();
    auto data =
        serde::CreateRNGAttributes(
            builder,
            toUnderlying(attributes.rtype),
            toUnderlying(std::get<nvf::PrimDataType>(attributes.dtype.type)),
            attributes.num_parameters)
            .Union();
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::RNGAttributes, data);
  } else if (v.any().type() == typeid(nvf::kir::Scope)) {
    const auto& kir_scope = v.as<nvf::kir::Scope>();
    auto data = kir_scope.serialize(container, builder).Union();
    return CreatePolymorphicValue(builder, PolymorphicValueData::Scope, data);
  } else if (v.any().type() == typeid(nvf::AsyncOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::AsyncOpType,
        toUnderlying(v.as<nvf::AsyncOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::DoubleBufferLoopStage)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::DoubleBufferLoopStage,
        toUnderlying(v.as<nvf::DoubleBufferLoopStage>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::BinaryOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::BinaryOpType,
        toUnderlying(v.as<nvf::BinaryOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::CacheOp)) {
    auto data = CreateOpaqueEnum(
        builder, NvFuserEnum::CacheOp, toUnderlying(v.as<nvf::CacheOp>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::LoadStoreOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::LoadStoreOpType,
        toUnderlying(v.as<nvf::LoadStoreOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::MemoryType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::MemoryType,
        toUnderlying(v.as<nvf::MemoryType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::MmaOp::MmaLayoutOpt)) {
    auto enum_value = v.as<nvf::MmaOp::MmaLayoutOpt>();
    int64_t fb_enum_value =
        enum_value.has_value() ? toUnderlying(enum_value.value()) : -1;
    auto data =
        CreateOpaqueEnum(builder, NvFuserEnum::MmaLayout, fb_enum_value);
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::MmaMacro)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::MmaMacro,
        (int64_t)toUnderlying(v.as<nvf::MmaMacro>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::ScatterOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::ScatterOpType,
        toUnderlying(v.as<nvf::ScatterOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::SwizzleMode)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::SwizzleMode,
        toUnderlying(v.as<nvf::SwizzleMode>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::SwizzleType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::SwizzleType,
        toUnderlying(v.as<nvf::SwizzleType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::Swizzle2DType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::Swizzle2DType,
        toUnderlying(v.as<nvf::Swizzle2DType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::tma::TensorMapInterleave)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapInterleave,
        toUnderlying(v.as<nvf::tma::TensorMapInterleave>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::tma::TensorMapL2Promotion)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapL2Promotion,
        toUnderlying(v.as<nvf::tma::TensorMapL2Promotion>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::tma::TensorMapFloatOOBFill)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TensorMapFloatOOBFill,
        toUnderlying(v.as<nvf::tma::TensorMapFloatOOBFill>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::TernaryOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::TernaryOpType,
        toUnderlying(v.as<nvf::TernaryOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(nvf::UnaryOpType)) {
    auto data = CreateOpaqueEnum(
        builder,
        NvFuserEnum::UnaryOpType,
        toUnderlying(v.as<nvf::UnaryOpType>()));
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::OpaqueEnum, data.Union());
  } else if (v.any().type() == typeid(std::vector<bool>)) {
    return serializeArray<bool, PrimArrayType::Bool>(builder, v);
  } else if (v.any().type() == typeid(std::vector<std::complex<double>>)) {
    return serializeArray<double, PrimArrayType::ComplexDouble>(builder, v);
  } else if (v.any().type() == typeid(std::vector<double>)) {
    return serializeArray<double, PrimArrayType::Double>(builder, v);
  } else if (v.any().type() == typeid(std::vector<int64_t>)) {
    return serializeArray<int64_t, PrimArrayType::Long>(builder, v);
  } else if (v.any().type() == typeid(nvf::DynamicTransformInitialInfo)) {
    auto data =
        v.as<nvf::DynamicTransformInitialInfo>().serialize(builder, container);
    return CreatePolymorphicValue(
        builder,
        PolymorphicValueData::DynamicTransformInitialInfo,
        data.Union());
  } else if (v.any().type() == typeid(nvf::LoopRotationParam)) {
    auto nvf_data = v.as<nvf::LoopRotationParam>();
    std::vector<flatbuffers::Offset<LoopRotationParam>> fb_items;
    fb_items.reserve(nvf_data.size());
    std::transform(
        nvf_data.begin(),
        nvf_data.end(),
        std::back_inserter(fb_items),
        [&container, &builder](const nvf::LoopRotationTuple& item) {
          const auto& selection_stmt_set =
              std::get<std::unordered_set<nvf::Statement*>>(item);

          std::vector<flatbuffers::Offset<StatementIndex>>
              fb_selection_stmt_set;
          fb_selection_stmt_set.reserve(selection_stmt_set.size());
          for (auto stmt : selection_stmt_set) {
            bool is_val = (stmt != nullptr) ? stmt->isVal() : true;
            fb_selection_stmt_set.push_back(serde::CreateStatementIndex(
                builder, container.map(stmt), is_val));
          }

          return CreateLoopRotationParamDirect(
              builder,
              container.map(std::get<nvf::TensorView*>(item)->asVal()),
              std::get<int64_t>(item),
              &fb_selection_stmt_set);
        });
    auto data = CreateLoopRotationDirect(builder, &fb_items);
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::LoopRotation, data.Union());
  } else if (v.any().type() == typeid(std::string)) {
    const auto& nvf_string = v.as<std::string>();
    auto data = serde::CreateStringDirect(builder, nvf_string.c_str());
    return CreatePolymorphicValue(
        builder, PolymorphicValueData::String, data.Union());
  } else {
    NVF_ERROR(
        false, "Serialization of arbitrary opaque value is not implemented.");
  }
}

} // namespace

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
        nvf::toUnderlying(tensor.scalar_type()));
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
      nvf::PolymorphicValue pv(*tensor.data_ptr<bool>());
      return serializeScalarRecord(builder, pv, nvf::DataType::Bool);
    }
    case at::ScalarType::Double: {
      nvf::PolymorphicValue pv(*tensor.data_ptr<double>());
      return serializeScalarRecord(builder, pv, nvf::DataType::Double);
    }
    case at::ScalarType::Long: {
      nvf::PolymorphicValue pv(*tensor.data_ptr<int64_t>());
      return serializeScalarRecord(builder, pv, nvf::DataType::Int);
    }
    case at::ScalarType::ComplexDouble: {
      auto at_complex = *tensor.data_ptr<c10::complex<double>>();
      nvf::PolymorphicValue pv((std::complex<double>)at_complex);
      return serializeScalarRecord(builder, pv, nvf::DataType::ComplexDouble);
    }
    default:
      NVF_ERROR(false, "Unsupported scalar type.");
  }
}

flatbuffers::Offset<PolymorphicValue> serializeScalar(
    flatbuffers::FlatBufferBuilder& builder,
    const nvf::PolymorphicValue& v) {
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
    const nvf::PolymorphicValue& v,
    nvf::DataType t) {
  ScalarBuilder builder_(builder);
  builder_.add_dtype(toUnderlying(std::get<nvf::PrimDataType>(t.type)));
  if (v.is<std::monostate>()) {
    builder_.add_has_value(false);
    return builder_.Finish();
  } else if (v.is<double>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(nvf::PrimDataType::Double));
    builder_.add_double_value(v.as<double>());
    return builder_.Finish();
  } else if (v.is<int64_t>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(nvf::PrimDataType::Int));
    builder_.add_long_value(v.as<int64_t>());
    return builder_.Finish();
  } else if (v.is<bool>()) {
    builder_.add_has_value(true);
    builder_.add_value_type(toUnderlying(nvf::PrimDataType::Bool));
    builder_.add_bool_value(v.as<bool>());
    return builder_.Finish();
  } else if (v.is<std::complex<double>>()) {
    builder_.add_has_value(true);
    auto c = v.as<std::complex<double>>();
    builder_.add_value_type(toUnderlying(nvf::PrimDataType::ComplexDouble));
    builder_.add_real_value(std::real(c));
    builder_.add_imag_value(std::imag(c));
    return builder_.Finish();
  }
  NVF_ERROR(false, "Unable to convert ", v.type().name(), " to Scalar.");
}

flatbuffers::Offset<PolymorphicValue> serializePolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const IrSerde& container,
    const nvf::PolymorphicValue& v) {
  NVF_ERROR(!v.is<nvf::Pointer>(), "Serialization of pointer is not allowed.");
  NVF_ERROR(!v.is<std::vector>(), "The general vector type is not supported.");
  NVF_ERROR(
      !v.is<nvf::StructHandle>(),
      "The StructHandle PolymorphicValue type is not supported.");

  if (v.is<std::monostate>()) {
    return CreatePolymorphicValue(builder, PolymorphicValueData::NONE);
  } else if (v.is<nvf::Opaque>()) {
    return serializeOpaque(builder, container, v.as<nvf::Opaque>());
  } else {
    return serializeBasicPolymorphicValue(builder, v);
  }
}

flatbuffers::Offset<PolymorphicValue> serializeBasicPolymorphicValue(
    flatbuffers::FlatBufferBuilder& builder,
    const nvf::PolymorphicValue& v) {
  NVF_ERROR(!v.is<nvf::Pointer>(), "Serialization of pointer is not allowed.");
  NVF_ERROR(!v.is<std::vector>(), "The general vector type is not supported.");
  NVF_ERROR(
      !v.is<nvf::Opaque>(),
      "Serializing Opaque PolymorphicValue without IrSerde container is not supported.");
  NVF_ERROR(
      !v.is<nvf::StructHandle>(),
      "The StructHandle PolymorphicValue type is not supported.");

  if (v.is<std::monostate>()) {
    return CreatePolymorphicValue(builder, PolymorphicValueData::NONE);
  } else if (v.is<at::Tensor>()) {
    return serializeTensor(builder, v.as<at::Tensor>());
  } else {
    return serializeScalar(builder, v);
  }
}

} // namespace nvfuser::serde
