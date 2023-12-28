// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#include <serde/factory.h>
#include <serde/fusion_cache_generated.h>
#include <type.h>

namespace nvfuser::serde {

//! The DataTypeFactory class is used to deserialize the flatbuffer
//! DataType table. The DataTable table represents std::variant<PrimDataType,
//! ArrayType, PointerType, StructType, OpaqueType>.
class DataTypeFactory : public Factory<DataType, nvfuser::DataType> {
 public:
  DataTypeFactory()
      : Factory((nvfuser::toUnderlying(DataTypeVariant::MAX) + 1)) {
    registerAllParsers();
  }

 private:
  void registerAllParsers();
};

nvfuser::DataType deserializeDataType(const DataType* dtype);

flatbuffers::Offset<DataType> serializeDataType(
    flatbuffers::FlatBufferBuilder& builder,
    const nvfuser::DataType& v);

} // namespace nvfuser::serde
