// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <serde/datatype.h>
#include <serde/utils.h>
#include <variant>

namespace nvf = nvfuser;

namespace nvfuser::serde {

namespace {
flatbuffers::Offset<FieldInfo> serializeFieldInfo(
    flatbuffers::FlatBufferBuilder& builder,
    const nvf::StructType::FieldInfo& field) {
  return CreateFieldInfoDirect(
      builder,
      field.name.c_str(),
      serializeDataType(builder, *field.type),
      field.used_in_kernel);
}

nvf::DataType deserializeUnsupported(const serde::DataType* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::DataType is nullptr.");
  NVF_ERROR(
      buffer->data_type() != DataTypeVariant::NONE, "Unsupported DataType.");
  return nvf::DataType();
}

nvf::DataType deserializeArrayType(const serde::DataType* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::DataType is nullptr.");
  const ArrayType* data = buffer->data_as_ArrayType();
  NVF_ERROR(data != nullptr, "serde::ArrayType is nullptr.");
  nvf::ArrayType at{
      std::make_shared<nvf::DataType>(deserializeDataType(data->type())),
      data->size()};
  return nvf::DataType(at);
}

nvf::DataType deserializePointerType(const serde::DataType* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::DataType is nullptr.");
  const PointerType* data = buffer->data_as_PointerType();
  NVF_ERROR(data != nullptr, "serde::PointerType is nullptr.");
  nvf::PointerType pt{
      std::make_shared<nvf::DataType>(deserializeDataType(data->type()))};
  return nvf::DataType(pt);
}

nvf::DataType deserializePrimDataType(const serde::DataType* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::DataType is nullptr.");
  const PrimDataType* data = buffer->data_as_PrimDataType();
  NVF_ERROR(data != nullptr, "serde::PrimDataType is nullptr.");
  return mapToDtypeStruct(data->dtype_enum());
}

nvf::DataType deserializeStructType(const serde::DataType* buffer) {
  NVF_ERROR(buffer != nullptr, "serde::DataType is nullptr.");
  const StructType* data = buffer->data_as_StructType();
  NVF_ERROR(data != nullptr, "serde::StructType is nullptr.");
  nvf::StructType st;
  st.name = data->name()->str();
  st.fields.reserve(data->fields()->size());
  std::transform(
      data->fields()->begin(),
      data->fields()->end(),
      std::back_inserter(st.fields),
      [](const serde::FieldInfo* field) {
        return nvf::StructType::FieldInfo{
            field->name()->str(),
            std::make_shared<nvf::DataType>(deserializeDataType(field->type())),
            field->used_in_kernel()};
      });
  return nvf::DataType(st);
}

} // namespace

void DataTypeFactory::registerAllParsers() {
  registerParser(DataTypeVariant::NONE, deserializeUnsupported);
  registerParser(DataTypeVariant::ArrayType, deserializeArrayType);
  registerParser(DataTypeVariant::PointerType, deserializePointerType);
  registerParser(DataTypeVariant::PrimDataType, deserializePrimDataType);
  registerParser(DataTypeVariant::StructType, deserializeStructType);
}

nvf::DataType deserializeDataType(const DataType* dtype) {
  DataTypeFactory dtype_factory;
  return dtype_factory.parse(dtype->data_type(), dtype);
}

flatbuffers::Offset<DataType> serializeDataType(
    flatbuffers::FlatBufferBuilder& builder,
    const nvf::DataType& dtype) {
  NVF_ERROR(
      !std::holds_alternative<nvf::OpaqueType>(dtype.type),
      "Serialization of OpaqueType is not allowed because it depends on runtime object.");

  if (std::holds_alternative<nvf::ArrayType>(dtype.type)) {
    auto array_dtype = std::get<nvf::ArrayType>(dtype.type);
    auto data = CreateArrayType(
        builder,
        serializeDataType(builder, *array_dtype.type),
        array_dtype.size);
    return CreateDataType(builder, DataTypeVariant::ArrayType, data.Union());
  } else if (std::holds_alternative<nvf::PointerType>(dtype.type)) {
    auto ptr_dtype = std::get<nvf::PointerType>(dtype.type);
    auto data =
        CreatePointerType(builder, serializeDataType(builder, *ptr_dtype.type));
    return CreateDataType(builder, DataTypeVariant::PointerType, data.Union());
  } else if (std::holds_alternative<nvf::PrimDataType>(dtype.type)) {
    auto prim_dtype = std::get<nvf::PrimDataType>(dtype.type);
    auto data = CreatePrimDataType(builder, toUnderlying(prim_dtype));
    return CreateDataType(builder, DataTypeVariant::PrimDataType, data.Union());
  } else if (std::holds_alternative<nvf::StructType>(dtype.type)) {
    auto struct_dtype = std::get<nvf::StructType>(dtype.type);
    std::vector<flatbuffers::Offset<FieldInfo>> fb_fieldinfo;
    fb_fieldinfo.reserve(struct_dtype.fields.size());
    for (const auto& field : struct_dtype.fields) {
      fb_fieldinfo.push_back(serializeFieldInfo(builder, field));
    }
    auto data = CreateStructTypeDirect(
        builder, struct_dtype.name.c_str(), &fb_fieldinfo);
    return CreateDataType(builder, DataTypeVariant::StructType, data.Union());
  }

  NVF_ERROR(false, "Failed to serialize unknown nvfuser::DataType.");
}

} // namespace nvfuser::serde
